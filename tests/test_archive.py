"""Job history retention: the day partitions, the DBOS purge behind them, and the listing that
reads both.

Every test that archives real jobs runs the real pipeline first: the only way to be sure the copy
matches what DBOS recorded is to import a document, attach it to a collection and read what that
left behind. Time is passed in, never waited for - `archive_once` takes "now", so a test looks at
its own jobs from three hours in the future.
"""

import functools
import threading
import time
from datetime import date
from pathlib import Path

import anyio.to_thread
import pytest
from dbos import DBOS
from dbos._registrations import get_dbos_func_name

from haskie import archive, db, dbos_names, document, jobs, models, pipeline, workflows
from haskie.collection import Collection
from haskie.errors import InvalidInput, JobNotFound
from haskie.settings import PipelineSettings, RetentionSettings, UserSettings, save_user_settings

from conftest import attach_document, import_document, restart_dbos, text_pdf, wait_for  # isort: skip

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


async def _one_batch_per_page(dbos) -> None:
    """One page per micro-batch and one part per index write, so every stage of a PDF plans one
    batch per page. Applied before the import: the plan is made when the workflow runs."""
    indexing = PipelineSettings(cpu_budget=6, batch_pages=1, index_group_parts=1)
    await dbos.apply_settings(await save_user_settings(UserSettings(pipeline=indexing)))


async def _imported(dbos, tmp_path: Path, name: str = "p.pdf", pages: int = 2) -> str:
    """One imported PDF with `pages` pages; returns the document name."""
    await _one_batch_per_page(dbos)
    row = await import_document(dbos, name, text_pdf([f"alpha{i}" for i in range(pages)]), tmp_path)
    return row.name


async def _attached(dbos, collection: str, doc: str) -> str:
    """The document in one collection; returns the id of the collection index job."""
    await Collection.create(collection)
    await attach_document(dbos, collection, doc)
    return await _job_id(collection=collection)


async def _job_id(action: str | None = None, collection: str | None = None) -> str:
    """The one job matching an action or a collection, or an assertion about why there is none."""
    found = [
        job
        for job in (await jobs.list_jobs(page_size=50)).items
        if (action is None or job.action == action)
        and (collection is None or job.collection == collection)
    ]
    assert len(found) == 1, f"expected one {action or collection} job, got {[j.id for j in found]}"
    return found[0].id


def _shape(tasks: list[jobs.Task]) -> list[tuple]:
    return [
        (t.id, t.stage, t.seq, t.page_start, t.page_end, t.status, t.result, t.error) for t in tasks
    ]


async def _live_ids() -> list[str]:
    """The pipeline workflows DBOS still holds: the jobs and the stage slices under them. The
    maintenance run an indexed document requests is a child of the job too, but it is still
    waiting out its debounce, so nothing archives or purges it."""
    found = await DBOS.list_workflows_async(
        name=[*dbos_names.PIPELINE_WORKFLOWS, dbos_names.STAGE_WORKFLOW],
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


async def _partition_rows(columns: str) -> list[tuple]:
    """Some columns of every archived job, newest first, read straight out of the partitions."""
    out: list[tuple] = []
    async with db.connect() as conn:
        for table in await archive.list_partitions(conn):
            rows = await conn.execute_fetchall(
                f"select {columns} from {table} order by created_at desc, id desc"
            )
            # aiosqlite types every row as `sqlite3.Row`; this connection yields plain tuples
            out.extend(tuple(row) for row in rows)
    return out


# --- what an id says a job is ------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "job_id", "expected"),
    [
        ("an import", "imp:book.pdf:cafe", ("import", None, "book.pdf")),
        ("an embedding", "emb:book.pdf:cafe", ("embed", None, "book.pdf")),
        ("a collection index", "idx-col:law:book.pdf:cafe", ("index", "law", "book.pdf")),
        ("a document delete", "del-doc:book.pdf:cafe", None),
        ("a bulk index", "bulk-index:law:cafe", None),
        ("an import without its uuid", "imp:book.pdf", None),
        ("a collection index without its uuid", "idx-col:law:book.pdf", None),
        ("a name that is no id at all", "book.pdf", None),
    ],
)
def test_a_pipeline_job_id_names_its_action_collection_and_document(
    name: str, job_id: str, expected: tuple | None
) -> None:
    assert workflows.job_names(job_id) == expected, name


@pytest.mark.parametrize(
    ("name", "job_id", "collection"),
    [
        ("a bulk index", "bulk-index:law:cafe", "law"),
        ("a bulk delete", "bulk-delete:law:cafe", "law"),
        ("a maintenance run", "maint:law:cafe", "law"),
        ("a document delete, which spans every collection", "del-doc:book.pdf:cafe", None),
        ("an id with nothing in that place", "bulk-index", None),
    ],
)
def test_only_a_job_of_one_collection_carries_its_name(
    name: str, job_id: str, collection: str | None
) -> None:
    assert jobs._collection_of(job_id) == collection, name


async def test_a_document_delete_is_a_collection_job_that_belongs_to_no_collection(
    dbos, tmp_path
) -> None:
    """The three whole-collection workflows share one kind and one id shape, but a document delete
    carries a document where the other two carry a collection."""
    doc = await _imported(dbos, tmp_path, "gone.pdf", pages=1)
    job_id = await dbos.start_delete_document(doc)
    assert await wait_for(job_id) is None

    (row,) = (await jobs.list_kind("collection", page_size=10)).items
    bulk = await jobs.bulk_job(job_id)

    assert (row.id, row.kind, row.title) == (job_id, "collection", f"delete document {doc}")
    assert (bulk.kind, bulk.collection, bulk.status) == ("delete_document", None, "SUCCESS")
    assert (await jobs.list_kind("collection", collection="any", page_size=10)).items == [], (
        "a collection filter keeps the jobs of one collection, and this job has none"
    )


# --- the round ---------------------------------------------------------------------


async def test_archive_copies_every_pipeline_job_of_a_document_and_dbos_forgets_them(
    dbos, tmp_path
) -> None:
    """An import, the embedding it warms and the index into one collection are three jobs, each
    with its own tasks. Only the index one belongs to a collection."""
    doc = await _imported(dbos, tmp_path, pages=2)
    index_id = await _attached(dbos, "arch", doc)
    import_id = await _job_id(action="import")
    before = {job.id: job for job in (await jobs.list_jobs(page_size=50)).items}
    before_tasks = {job_id: await jobs.list_tasks(job_id) for job_id in before}
    assert before[import_id].tasks_total == 2, "one convert batch per page"
    assert before[index_id].tasks_total == 2, "one index batch per part"

    report = await archive.archive_once(_later(), SMALL)

    assert (report.copied, report.dropped) == (len(before), [])
    assert await _live_ids() == [], "the DBOS rows of the jobs and their stage children are gone"
    after = {job.id: job for job in (await jobs.list_jobs(page_size=50)).items}
    assert sorted(after) == sorted(before)
    imported, indexed = after[import_id], after[index_id]
    assert (imported.action, imported.collection, imported.doc) == ("import", None, doc)
    assert (indexed.action, indexed.collection, indexed.doc) == ("index", "arch", doc)
    assert [(job.archived, job.status) for job in after.values()] == [(True, "SUCCESS")] * len(
        after
    )
    for job_id, tasks in before_tasks.items():
        assert _shape(await jobs.list_tasks(job_id)) == _shape(tasks), "every batch, unchanged"
    assert (await _partition_rows("id, action, collection, doc")).count(
        (index_id, "index", "arch", doc)
    ) == 1, "the collection column is filled for the index job only"
    assert [row for row in await _partition_rows("id, collection") if row[1] is not None] == [
        (index_id, "arch")
    ]
    day = archive.day_suffix(int(indexed.updated_at * 1000))
    async with db.connect() as conn:
        assert await archive.list_partitions(conn) == [f"jobs_{day}"]


async def test_a_second_round_copies_nothing_and_leaves_the_watermark_at_the_cutoff(
    dbos, tmp_path
) -> None:
    await _imported(dbos, tmp_path, pages=1)
    before = [job.id for job in (await jobs.list_jobs(page_size=50)).items]
    now = _later()

    first = await archive.archive_once(now, SMALL)
    second = await archive.archive_once(now, SMALL)

    assert (first.copied, second.copied) == (len(before), 0)
    assert await archive.watermark() == now - HOUR_MS, (
        "the whole window is archived, cutoff included"
    )
    after = [job.id for job in (await jobs.list_jobs(page_size=50)).items]
    assert after == before, "copied once, not twice"
    assert len(await jobs.list_tasks(await _job_id(action="import"))) == 1, "one page, one batch"


async def test_a_running_job_stays_live(dbos, monkeypatch, tmp_path) -> None:
    """Only a finished job is copied: an unfinished one has no `completed_at`, which is the same
    predicate DBOS's own collection uses, so nothing can be purged out from under it."""
    await _one_batch_per_page(dbos)
    source = tmp_path / "busy.md"
    source.write_bytes(MD)
    row = await document.import_path(str(source))
    # threading events, not anyio ones: the step runs on DBOS's event loop, the test on its own
    entered, release = threading.Event(), threading.Event()
    real = pipeline.convert_batch

    async def blocking(*args):
        entered.set()
        held = await anyio.to_thread.run_sync(functools.partial(release.wait, WAIT))
        assert held, "the test never released the step"
        return await real(*args)

    monkeypatch.setattr(pipeline, "convert_batch", blocking)
    job_id = await dbos.start_import(row.name)
    started = await anyio.to_thread.run_sync(functools.partial(entered.wait, WAIT))
    assert started, "the convert step never started"

    report = await archive.archive_once(_later(), SMALL)

    assert report.copied == 0
    (job,) = (await jobs.list_jobs()).items
    assert (job.id, job.archived, job.status) == (job_id, False, "PENDING")
    assert (job.action, job.collection, job.doc) == ("import", None, row.name)
    release.set()
    assert await wait_for(job_id) == "imported"


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


async def test_a_partition_indexes_its_collection_and_leaves_it_nullable() -> None:
    """The listing filters on `collection` and orders by `created_at`, and an import has no
    collection to be filtered by: the column is nullable and the index covers both."""
    async with db.connect() as conn:
        await archive.ensure_partition(conn, "20250101")
        await conn.execute(
            "insert into jobs_20250101 (id, action, collection, doc, status, created_at, "
            "completed_at, error, tasks_done, tasks_total) values "
            "('imp:a.pdf:1', 'import', null, 'a.pdf', 'SUCCESS', 1, 2, null, 1, 1)"
        )
        rows = await conn.execute_fetchall(
            "select name from sqlite_master where type = 'index' and tbl_name = 'jobs_20250101'"
        )
        plan = await conn.execute_fetchall(
            "explain query plan select id from jobs_20250101 where collection = 'a' "
            "order by created_at desc"
        )
    # the primary key brings an automatic index of its own; this is the one we declare
    assert "jobs_20250101_collection" in [name for (name,) in rows]
    assert "jobs_20250101_collection" in " ".join(str(step[-1]) for step in plan), (
        "the filter and the order are served by the index, not by a scan"
    )


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


async def test_a_page_runs_from_the_live_history_into_the_archive(dbos, tmp_path) -> None:
    doc = await _imported(dbos, tmp_path, "old.pdf", pages=1)
    assert (await archive.archive_once(_later(), SMALL)).copied == 2, "the import and its embed"
    archived = [job.id for job in (await jobs.list_jobs(page_size=50)).items]
    await _attached(dbos, "page", doc)
    whole = [job.id for job in (await jobs.list_jobs(page_size=50)).items]
    live = [job_id for job_id in whole if job_id not in archived]
    assert len(live) >= 1 and whole[-len(archived) :] == archived, (
        "the live history first, then the newest archived day"
    )

    first = await jobs.list_jobs(page_size=len(live) + 1)
    second = await jobs.list_jobs(page_size=len(live) + 1, cursor=first.next_cursor)

    assert [job.id for job in first.items] == whole[: len(live) + 1], (
        "the page boundary fell inside the day partition"
    )
    assert [job.archived for job in first.items] == [False] * len(live) + [True]
    assert [job.id for job in second.items] == whole[len(live) + 1 :]
    assert second.next_cursor is None, "nothing older than the last day"


async def test_the_collection_filter_keeps_one_collections_jobs_in_both_sources(
    dbos, tmp_path
) -> None:
    """The same document in two collections: one index job archived, one still live. The filter
    is an id prefix in the live history and an indexed column in a partition, and an import or an
    embed matches neither."""
    doc = await _imported(dbos, tmp_path, "shared.pdf", pages=1)
    old = await _attached(dbos, "old", doc)
    assert (await archive.archive_once(_later(), SMALL)).copied >= 3
    new = await _attached(dbos, "new", doc)

    assert [job.id for job in (await jobs.list_jobs("old", page_size=50)).items] == [old], (
        "the filter reaches the partitions too"
    )
    assert [job.id for job in (await jobs.list_jobs("new", page_size=50)).items] == [new]
    assert (await jobs.list_jobs("nosuch", page_size=50)).items == []
    rows = (await jobs.list_kind("document", collection="old", page_size=50)).items
    assert [(row.id, row.title) for row in rows] == [(old, f"old / {doc}")], (
        "the same filter through the by-kind listing, with the collection in the title"
    )


async def test_tasks_of_an_unknown_job_are_a_404_whichever_source_is_read(dbos, tmp_path) -> None:
    await _imported(dbos, tmp_path, "gone.pdf", pages=1)
    await archive.archive_once(_later(), SMALL)

    with pytest.raises(JobNotFound):
        await jobs.list_tasks("imp:gone.pdf:nosuchjob")
    with pytest.raises(InvalidInput, match="invalid cursor"):
        await jobs.list_jobs(cursor=jobs.CURSOR.encode("jobs_bogus", 0))


async def test_a_cursor_into_a_dropped_day_ends_the_listing(dbos, tmp_path) -> None:
    """Retention can drop the day a cursor points at while the listing is being walked. Its jobs
    are gone, so the walk ends there rather than serving another day's jobs in their place."""
    await _imported(dbos, tmp_path, "dropped.pdf", pages=1)
    await archive.archive_once(_later(), SMALL)
    live = [job.id for job in (await jobs.list_jobs(page_size=50)).items]
    job_id = await _job_id(action="import")
    day = archive.day_suffix(int((await jobs.list_jobs()).items[0].updated_at * 1000))
    cursor = jobs.CURSOR.encode(f"jobs_{day}", 0)
    assert [job.id for job in (await jobs.list_jobs(cursor=cursor)).items] == live

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


def test_the_workflow_names_the_jobs_view_spells_out_are_the_ones_dbos_records() -> None:
    """Every name `archive` selects by and `jobs` groups by is pinned against the registration
    DBOS made: a mismatch is a silent miss in a query, not an error."""
    assert dbos_names.PIPELINE_WORKFLOWS == [
        get_dbos_func_name(workflows.import_document),
        get_dbos_func_name(workflows.ensure_embedding),
        get_dbos_func_name(workflows.index_collection_document),
    ]
    assert dbos_names.STAGE_WORKFLOW == get_dbos_func_name(workflows.stage_slice)
    assert dbos_names.STAGE_STEP == get_dbos_func_name(workflows.try_batch)
    assert workflows.stage_slice.__name__ != dbos_names.STAGE_WORKFLOW, (
        "the durable name is pinned, not derived from the function name"
    )
    assert jobs.KIND_NAMES == {
        "collection": [
            get_dbos_func_name(workflows.index_collection_workflow),
            get_dbos_func_name(workflows.delete_collection_workflow),
            get_dbos_func_name(workflows.delete_document_workflow),
        ],
        "download": [get_dbos_func_name(models.ensure_model)],
        "maintenance": [
            get_dbos_func_name(workflows.maintain_on_partition),
            get_dbos_func_name(workflows.daily_maintenance),
        ],
        "archive": [get_dbos_func_name(workflows.archive_jobs)],
    }
    assert set(jobs.KIND_BY_NAME) == set(dbos_names.PIPELINE_WORKFLOWS) | {
        name for names in jobs.KIND_NAMES.values() for name in names
    }, "every kind counts the workflows it lists, and nothing else"
