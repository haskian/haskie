"""Job history retention: the day partitions, the DBOS purge behind them, and the listing that
reads both.

Every test that archives real jobs indexes real documents first: the only way to be sure the copy
matches what DBOS recorded is to run the pipeline and read what it left behind. Time is passed in,
never waited for - `archive_once` takes "now", so a test looks at its own jobs from three hours in
the future.
"""

import functools
import threading
import time
from datetime import date

import anyio.to_thread
import pytest
from dbos import DBOS

from haskie import archive, db, dbos_names, jobs, pipeline, workflows
from haskie.errors import InvalidInput, JobNotFound
from haskie.library import Library
from haskie.settings import PipelineSettings, RetentionSettings, UserSettings, save_user_settings

from conftest import restart_dbos, text_pdf, wait_for  # isort: skip

pytestmark = pytest.mark.anyio

WAIT = 30.0
HOUR_MS = archive.HOUR_MS
# the shortest window the settings allow: an hour live, then an hour archived but not yet purged
SMALL = RetentionSettings(job_days=28, job_live_hours=2)
MD = b"# a\n\nhello from a\n"


def _later(hours: float = 3) -> int:
    """Now, seen from `hours` ahead: with `SMALL`, everything already finished is past both
    cutoffs, so one round archives it and purges it."""
    return int(time.time() * 1000) + int(hours * HOUR_MS)


async def _indexed_job(dbos, library: str, pages: int = 2) -> str:
    """One indexed PDF with one page per batch and one part per index write, so every stage has
    `pages` batches."""
    indexing = PipelineSettings(cpu_budget=6, batch_pages=1, index_group_parts=1)
    await dbos.apply_settings(await save_user_settings(UserSettings(pipeline=indexing)))
    lib = await Library.create(library)
    doc = await lib.save("p.pdf", text_pdf([f"alpha{i}" for i in range(pages)]))
    job_id = await dbos.start_index(library, doc.name)
    assert await wait_for(job_id) == "indexed"
    return job_id


async def _index(dbos, library: str, doc: str) -> None:
    assert await wait_for(await dbos.start_index(library, doc)) == "indexed"


def _shape(tasks: list[jobs.Task]) -> list[tuple]:
    return [
        (t.id, t.stage, t.seq, t.page_start, t.page_end, t.status, t.result, t.error) for t in tasks
    ]


async def _live_ids() -> list[str]:
    """The pipeline workflows DBOS still holds: the jobs and the stage slices under them. The
    maintenance run a document requests is a child of the job too, but it is still waiting out its
    debounce, so nothing archives or purges it."""
    found = await DBOS.list_workflows_async(
        name=[dbos_names.DOCUMENT_WORKFLOW, dbos_names.STAGE_WORKFLOW],
        load_input=False,
        load_output=False,
    )
    return [s.workflow_id for s in found]


def _events(caplog) -> list[str]:
    return [r.msg["event"] for r in caplog.records if isinstance(r.msg, dict)]


async def _tables() -> set[str]:
    async with db.connect() as conn:
        rows = await conn.execute_fetchall("select name from sqlite_master where type = 'table'")
    return {name for (name,) in rows}


# --- the round ---------------------------------------------------------------------


async def test_archive_copies_a_finished_job_and_dbos_forgets_it(dbos) -> None:
    job_id = await _indexed_job(dbos, "arch", pages=2)
    (before,) = (await jobs.list_jobs()).items
    before_tasks = await jobs.list_tasks(job_id)
    assert (before.archived, before.tasks_total, len(before_tasks)) == (False, 6, 6)

    report = await archive.archive_once(_later(), SMALL)

    assert (report.copied, report.dropped) == (1, [])
    assert await _live_ids() == [], "the DBOS rows of the job and its stage children are gone"
    (job,) = (await jobs.list_jobs()).items
    assert (job.id, job.library, job.doc, job.status) == (job_id, "arch", "p.pdf", "SUCCESS")
    assert (job.archived, job.tasks_done, job.tasks_total, job.tasks_running) == (True, 6, 6, 0)
    assert (job.created_at, job.error) == (before.created_at, None)
    assert _shape(await jobs.list_tasks(job_id)) == _shape(before_tasks), "every batch, unchanged"
    day = archive.day_suffix(int(job.updated_at * 1000))
    async with db.connect() as conn:
        assert await archive.list_partitions(conn) == [f"jobs_{day}"]


async def test_a_second_round_copies_nothing_and_leaves_the_watermark_at_the_cutoff(dbos) -> None:
    job_id = await _indexed_job(dbos, "twice", pages=1)
    now = _later()

    first = await archive.archive_once(now, SMALL)
    second = await archive.archive_once(now, SMALL)

    assert (first.copied, second.copied) == (1, 0)
    assert await archive.watermark() == now - HOUR_MS, (
        "the whole window is archived, cutoff included"
    )
    assert [j.id for j in (await jobs.list_jobs()).items] == [job_id], "copied once, not twice"
    assert len(await jobs.list_tasks(job_id)) == 3


async def test_a_running_job_stays_live(dbos, monkeypatch) -> None:
    """Only a finished job is copied: an unfinished one has no `completed_at`, which is the same
    predicate DBOS's own collection uses, so nothing can be purged out from under it."""
    lib = await Library.create("busy")
    doc = await lib.save("a.md", MD)
    # threading events, not anyio ones: the step runs on DBOS's event loop, the test on its own
    entered, release = threading.Event(), threading.Event()
    real = pipeline.convert_batch

    async def blocking(*args):
        entered.set()
        held = await anyio.to_thread.run_sync(functools.partial(release.wait, WAIT))
        assert held, "the test never released the step"
        return await real(*args)

    monkeypatch.setattr(pipeline, "convert_batch", blocking)
    job_id = await dbos.start_index("busy", doc.name)
    started = await anyio.to_thread.run_sync(functools.partial(entered.wait, WAIT))
    assert started, "the convert step never started"

    report = await archive.archive_once(_later(), SMALL)

    assert report.copied == 0
    (job,) = (await jobs.list_jobs()).items
    assert (job.id, job.archived, job.status) == (job_id, False, "PENDING")
    release.set()
    assert await wait_for(job_id) == "indexed"


async def test_the_purge_never_passes_the_watermark(dbos, monkeypatch) -> None:
    """A round that stopped halfway leaves the watermark behind the window. DBOS may then only
    collect up to the watermark: past it are jobs whose micro-batches are not copied yet."""
    now = _later()
    stopped_at = now - 10 * HOUR_MS
    async with db.connect() as conn:
        await archive._set_watermark(conn, stopped_at)
    batches = iter([archive.ARCHIVE_PAGE, 0])  # a full batch, then a round that makes no progress

    async def copy_batch(cutoff: int) -> int:
        return next(batches)

    monkeypatch.setattr(archive, "copy_batch", copy_batch)
    cutoffs: list[int | None] = []
    monkeypatch.setattr(
        archive,
        "garbage_collect",
        lambda instance, **kwargs: cutoffs.append(kwargs["cutoff_epoch_timestamp_ms"]),
    )

    report = await archive.archive_once(now, SMALL)

    assert cutoffs == [stopped_at], "the watermark, not the end of the live window"
    assert (report.purged_before_ms, report.copied) == (stopped_at, archive.ARCHIVE_PAGE)
    assert stopped_at < now - SMALL.job_live_hours * HOUR_MS, "the window would have purged more"


# --- partitions --------------------------------------------------------------------


async def test_drop_expired_partitions_drops_whole_days_and_leaves_foreign_tables_alone(
    caplog,
) -> None:
    async with db.connect() as conn:
        await archive.ensure_partition(conn, "20200101")
        await archive.ensure_partition(conn, "20991231")
        # a name that is not a partition, and one that looks like one until it is parsed
        await conn.executescript("create table jobs_bogus (x); create table jobs_2020010 (x);")

    with caplog.at_level("WARNING"):
        dropped = await archive.drop_expired_partitions(date(2025, 1, 1))

    assert dropped == ["jobs_20200101", "tasks_20200101"]
    tables = await _tables()
    assert {"jobs_20991231", "tasks_20991231", "jobs_bogus", "jobs_2020010"} <= tables
    assert not {"jobs_20200101", "tasks_20200101"} & tables
    assert _events(caplog) == ["partition_name_ignored"], "the half-matching name is reported"
    async with db.connect() as conn:
        assert await archive.list_partitions(conn) == ["jobs_20991231"]


@pytest.mark.parametrize(
    ("name", "table", "day"),
    [
        ("a jobs partition", "jobs_20250101", date(2025, 1, 1)),
        ("a tasks partition", "tasks_20250101", date(2025, 1, 1)),
        ("a leap day", "jobs_20240229", date(2024, 2, 29)),
        ("a day that does not exist", "jobs_20250229", None),
        ("a month that does not exist", "jobs_20251301", None),
        ("too few digits", "jobs_2025011", None),
        ("too many digits", "jobs_202501011", None),
        ("no digits", "jobs_bogus", None),
        ("another table entirely", "documents", None),
        ("a prefix of ours", "old_jobs_20250101", None),
        ("a suffix after the day", "jobs_20250101_old", None),
        ("SQL after the day", "jobs_20250101; drop table documents", None),
        ("a newline after the day", "jobs_20250101\n", None),
        ("the wrong case", "JOBS_20250101", None),
        ("nothing at all", "", None),
    ],
)
def test_only_a_real_day_partition_is_ever_named_in_sql(name: str, table: str, day: date) -> None:
    assert archive.partition_day(table) == day, name


async def _ensure_partition(day: str) -> None:
    async with db.connect() as conn:
        await archive.ensure_partition(conn, day)


async def _tasks_table(jobs_table: str) -> str:
    """`tasks_table` is pure; awaited here so every case of the table below has one shape."""
    return archive.tasks_table(jobs_table)


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("tasks table of a foreign name", lambda: _tasks_table("documents")),
        ("tasks table of an impossible day", lambda: _tasks_table("jobs_20259999")),
        ("partition for a day that is not one", lambda: _ensure_partition("bogus")),
        ("partition for a day the calendar has not", lambda: _ensure_partition("20250230")),
    ],
)
async def test_a_name_that_is_not_a_partition_is_refused(name: str, call) -> None:
    with pytest.raises(ValueError):
        await call()


# --- listing -----------------------------------------------------------------------


async def test_a_page_runs_from_the_live_history_into_the_archive(dbos) -> None:
    lib = await Library.create("page")
    archived = [(await lib.save(f"old{i}.md", MD)).name for i in range(3)]
    for doc in archived:
        await _index(dbos, "page", doc)
    assert (await archive.archive_once(_later(), SMALL)).copied == 3
    live = [(await lib.save(f"new{i}.md", MD)).name for i in range(2)]
    for doc in live:
        await _index(dbos, "page", doc)

    first = await jobs.list_jobs(page_size=3)
    second = await jobs.list_jobs(page_size=3, cursor=first.next_cursor)

    assert [(j.doc, j.archived) for j in first.items] == [
        (live[1], False),
        (live[0], False),
        (archived[2], True),
    ], "the live history first, then the newest archived day"
    assert first.next_cursor is not None, "the page boundary fell inside the day partition"
    assert [(j.doc, j.archived) for j in second.items] == [
        (archived[1], True),
        (archived[0], True),
    ]
    assert second.next_cursor is None, "nothing older than the last day"
    whole = [j.doc for j in (await jobs.list_jobs("page", page_size=10)).items]
    assert whole == [*reversed(live), *reversed(archived)], "one page over both sources"
    other = (await jobs.list_jobs("other", page_size=10)).items
    assert other == [], "the filter reaches the partitions too"


async def test_tasks_of_an_unknown_job_are_a_404_whichever_source_is_read(dbos) -> None:
    await _indexed_job(dbos, "gone", pages=1)
    await archive.archive_once(_later(), SMALL)

    with pytest.raises(JobNotFound):
        await jobs.list_tasks("idx:gone:p.pdf:nosuchjob")
    with pytest.raises(InvalidInput, match="invalid cursor"):
        await jobs.list_jobs(cursor=jobs.CURSOR.encode("jobs_bogus", 0))


async def test_a_cursor_into_a_dropped_day_ends_the_listing(dbos) -> None:
    """Retention can drop the day a cursor points at while the listing is being walked. Its jobs
    are gone, so the walk ends there rather than serving another day's jobs in their place."""
    job_id = await _indexed_job(dbos, "dropped", pages=1)
    await archive.archive_once(_later(), SMALL)
    (job,) = (await jobs.list_jobs()).items
    day = archive.day_suffix(int(job.updated_at * 1000))
    cursor = jobs.CURSOR.encode(f"jobs_{day}", 0)
    assert [j.id for j in (await jobs.list_jobs(cursor=cursor)).items] == [job_id]

    await archive.drop_expired_partitions(date(2099, 1, 1))

    page = await jobs.list_jobs(cursor=cursor)
    assert (page.items, page.next_cursor) == ([], None)
    with pytest.raises(JobNotFound):
        await jobs.list_tasks(job_id)


# --- schedule ----------------------------------------------------------------------


async def test_the_archive_schedule_is_written_once(dbos) -> None:
    """Schedules live in the system database, so they outlive the process: the second boot must
    find the one the first wrote instead of adding another."""

    async def registered() -> list:
        found = await DBOS.list_schedules_async()
        return [s for s in found if s["schedule_name"] == workflows.ARCHIVE_SCHEDULE]

    (schedule,) = await registered()
    assert (schedule["schedule"], schedule["queue_name"]) == (
        workflows.ARCHIVE_CRON,
        workflows.MAINTENANCE_QUEUE,
    )
    assert "archive_jobs" in schedule["workflow_name"]

    await restart_dbos()

    assert len(await registered()) == 1


def test_the_workflow_names_the_archive_spells_out_are_the_ones_dbos_records() -> None:
    assert dbos_names.DOCUMENT_WORKFLOW == workflows.index_document.__qualname__
    assert workflows.stage_slice.__name__ != dbos_names.STAGE_WORKFLOW, (
        "the durable name is pinned, not derived from the function name"
    )
