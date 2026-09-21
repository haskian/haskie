"""Job history retention: day partitions for finished jobs, and a purge of the DBOS history.

DBOS records one workflow per pipeline run over a document — an import, an embedding, an index
into one collection — plus one per stage slice below it, one step per micro-batch, and removes
none of them: the history the Jobs page reads grows with every document, forever. This module
bounds it, once an hour (`workflows.archive_jobs`):

- copy every finished pipeline workflow (`dbos_names.PIPELINE_WORKFLOWS`), with the micro-batch
  results below it, into the table of the UTC day it completed on (`jobs_YYYYMMDD` and
  `tasks_YYYYMMDD`);
- let DBOS garbage-collect its own tables up to the point that copy reached;
- drop the partitions of days that fell out of the retention window.

A day is one table, so expiring a day costs a `drop table` rather than a delete of every row in
it, and a listing pays only for the days it reads. The tables are created on demand, which is why
they cannot come from a migration: their names carry a date. Every name is checked against
`PARTITION` *and* parsed as a calendar date before it is interpolated into a statement, and the
names a listing walks come from `sqlite_master`; nothing a caller passes ever becomes an
identifier.

`retention_state.archive_watermark_ms` is the boundary between the two halves: every job that
completed at or before it is in a partition, so the purge never passes it. The watermark is
committed together with the rows it covers, so a crash mid-archive simply replays the batch the
last commit did not cover. Two runs overlapping is safe: the copy is an `insert or replace` keyed
by the job id, and the watermark only ever moves forward.

Every statement goes through `aiosqlite`, so a round never blocks the loop it was scheduled on.
DBOS's own collection has no async twin, so it runs in a worker thread.
"""

import functools
import re
import sqlite3
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from typing import cast

import aiosqlite
import anyio.to_thread
import msgspec
from dbos import DBOS
from dbos._dbos import _get_dbos_instance
from dbos._workflow_commands import garbage_collect

from haskie import db, logs
from haskie.dbos_names import PIPELINE_WORKFLOWS
from haskie.settings import RetentionSettings

_log = logs.get_logger(__name__)

ARCHIVE_PAGE = 500  # jobs copied per transaction; also a per-query id list SQLite binds well
PARTITION = re.compile(r"^(jobs|tasks)_(\d{8})\Z")  # \Z, not $: `$` also matches before a newline
DAY = "%Y%m%d"
WATERMARK = "archive_watermark_ms"
HOUR_MS = 3_600_000

# The partition columns, defined once: the inserts and the selects below share them, and `jobs`
# reads a row positionally in this order. `collection` is null for an import and an embed: both
# are collection-independent (see `jobs.Job`).
JOB_COLUMNS = (
    "id",
    "action",  # import | embed | index
    "collection",
    "doc",
    "status",
    "created_at",  # unix ms, as DBOS records it
    "completed_at",
    "error",
    "tasks_done",
    "tasks_total",
)
TASK_COLUMNS = (
    "job_id",
    "child_id",  # the stage slice the batch ran under; with `seq` it is the live task id
    "kind",
    "seq",
    "page_start",
    "page_end",
    "status",
    "result",
    "error",
)


class ArchiveReport(msgspec.Struct):
    """What one retention round did. Returned by the step, so it is the record of the run."""

    copied: int
    purged_before_ms: int
    dropped: list[str]


def _tuples[RowT: tuple](rows: Iterable[sqlite3.Row]) -> list[RowT]:
    """The rows of a read, as the tuples they are.

    `aiosqlite` types every row as `sqlite3.Row` because a row factory may make one; this
    connection keeps the default factory, which yields plain tuples."""
    return cast("list[RowT]", list(rows))


# --- partitions -------------------------------------------------------------------


def day_suffix(epoch_ms: int) -> str:
    """The UTC day a timestamp falls on, as a partition suffix."""
    return datetime.fromtimestamp(epoch_ms / 1000, UTC).strftime(DAY)


def partition_day(table: str) -> date | None:
    """The day a partition name carries, or None when the name is not one this module writes.

    Two checks, because `\\d{8}` also matches 20259999: a name is interpolated into SQL only
    after it parsed as a real date."""
    match = PARTITION.match(table)
    if match is None:
        return None
    try:
        return datetime.strptime(match[2], DAY).date()
    except ValueError:
        return None


def tasks_table(jobs_table: str) -> str:
    """The tasks partition beside a jobs partition; validated, so the result is safe to inline."""
    if partition_day(jobs_table) is None:
        raise ValueError(f"not a partition name: {jobs_table!r}")
    return f"tasks_{jobs_table.removeprefix('jobs_')}"


async def ensure_partition(conn: aiosqlite.Connection, day: str) -> None:
    """Create the two tables of one day, if they are not there yet.

    One statement per call, never `executescript`: that one commits whatever the connection is in
    the middle of, and this runs inside the copy's transaction."""
    if partition_day(f"jobs_{day}") is None:
        raise ValueError(f"not a day: {day!r}")
    await conn.execute(f"""
    create table if not exists jobs_{day} (
        id text primary key, action text not null, collection text, doc text not null,
        status text not null,
        created_at integer not null, completed_at integer not null, error text,
        tasks_done integer not null, tasks_total integer not null
    )""")
    await conn.execute(
        f"create index if not exists jobs_{day}_collection on jobs_{day} "
        "(collection, created_at desc)"
    )
    await conn.execute(f"""
    create table if not exists tasks_{day} (
        job_id text not null, child_id text not null, kind text not null, seq integer not null,
        page_start integer not null, page_end integer not null, status text not null,
        result integer, error text,
        -- `seq` is the batch's number in the stage's plan, unique across the slices it was cut
        -- into, so the key stays one row per micro-batch however the stage was split
        primary key (job_id, kind, seq)
    )""")


async def list_partitions(conn: aiosqlite.Connection) -> list[str]:
    """The jobs partitions that exist, newest day first.

    Only names this module could have written are returned, and they are the only table names a
    listing ever inlines: a cursor selects one of these, it never spells one."""
    rows = await conn.execute_fetchall(
        "select name from sqlite_master where type = 'table' and name glob 'jobs_[0-9]*' "
        "order by name desc"
    )
    out: list[str] = []
    for (name,) in rows:
        if partition_day(name) is None:
            _log.warning("partition_name_ignored", table=name)
        else:
            out.append(name)
    return out


async def drop_expired_partitions(keep_from: date) -> list[str]:
    """Drop every day older than `keep_from`, jobs and tasks together; returns the names dropped."""
    dropped: list[str] = []
    async with db.connect() as conn:
        for table in await list_partitions(conn):
            day = partition_day(table)
            if day is None or day >= keep_from:
                continue
            for name in (table, tasks_table(table)):
                await conn.execute(f"drop table if exists {name}")
                dropped.append(name)
    if dropped:
        _log.info("partitions_dropped", tables=dropped, keep_from=keep_from.isoformat())
    return dropped


# --- reads (the Jobs listing) -----------------------------------------------------


def _column_list(columns: tuple[str, ...]) -> str:
    return ", ".join(columns)


async def job_page(
    conn: aiosqlite.Connection, table: str, collection: str | None, limit: int, offset: int
) -> list[tuple]:
    """One page of archived jobs of one day, newest first, in `JOB_COLUMNS` order.

    `table` must come from `list_partitions`. The id breaks ties on the timestamp, so an offset
    into a day is stable while the page is walked. A collection keeps that collection's index
    jobs only: an import and an embed have no collection to match."""
    where = "where collection = ? " if collection else ""
    params: list[object] = ([collection] if collection else []) + [limit, offset]
    return _tuples(
        await conn.execute_fetchall(
            f"select {_column_list(JOB_COLUMNS)} from {table} {where}"
            "order by created_at desc, id desc limit ? offset ?",
            params,
        )
    )


async def has_job(conn: aiosqlite.Connection, table: str, job_id: str) -> bool:
    """Whether one day holds a job; a primary-key lookup, so walking the days is cheap."""
    cursor = await conn.execute(f"select 1 from {table} where id = ?", (job_id,))
    return await cursor.fetchone() is not None


async def task_rows(conn: aiosqlite.Connection, table: str, job_id: str) -> list[tuple]:
    """Every archived micro-batch of one job, in `TASK_COLUMNS` order."""
    return _tuples(
        await conn.execute_fetchall(
            f"select {_column_list(TASK_COLUMNS)} from {table} where job_id = ?", (job_id,)
        )
    )


async def indexed_chunks(
    conn: aiosqlite.Connection, table: str, since_ms: int
) -> list[tuple[int, str, int]]:
    """Per successful index job of one day that completed at or after `since_ms`: when it
    completed, into which collection, and how many chunks its batches wrote. `table` must come
    from `list_partitions`."""
    return _tuples(
        await conn.execute_fetchall(
            "select j.completed_at, j.collection, coalesce(sum(t.result), 0) "
            f"from {table} j join {tasks_table(table)} t "
            "on t.job_id = j.id and t.kind = 'index' and t.status = 'SUCCESS' "
            "where j.action = 'index' and j.status = 'SUCCESS' and j.completed_at >= ? "
            "group by j.id order by j.completed_at",
            (since_ms,),
        )
    )


# --- the round --------------------------------------------------------------------


async def watermark() -> int:
    """Unix ms up to which finished jobs are archived; 0 before the first round."""
    async with db.connect() as conn:
        return await _watermark(conn)


async def _watermark(conn: aiosqlite.Connection) -> int:
    cursor = await conn.execute("select value from retention_state where key = ?", (WATERMARK,))
    row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def _set_watermark(conn: aiosqlite.Connection, value: int) -> None:
    await conn.execute(
        "insert into retention_state (key, value) values (?, ?) "
        "on conflict (key) do update set value = excluded.value",
        (WATERMARK, str(value)),
    )


async def archive_once(now_ms: int, settings: RetentionSettings) -> ArchiveReport:
    """One retention round: copy, purge, expire.

    The live window is split in half: a job is copied once it is older than half of it, and the
    DBOS rows behind it are deleted once it is older than all of it. A job that runs longer than
    that first half is copied while its children are already being purged, so its micro-batches
    may be missing from the archive - hence the "must cover the longest job" on the setting."""
    window_ms = settings.job_live_hours * HOUR_MS
    copied = 0
    while True:
        batch = await copy_batch(now_ms - window_ms // 2)
        copied += batch
        if batch < ARCHIVE_PAGE:
            break
    # never past the watermark: a job's children complete before it does, so purging only what is
    # archived keeps a copied job's micro-batches copied too
    purged_before = min(now_ms - window_ms, await watermark())
    # DBOS's collection is sync SQLAlchemy over the same file, so it goes to a worker thread
    await anyio.to_thread.run_sync(
        functools.partial(
            garbage_collect,
            _get_dbos_instance(),
            cutoff_epoch_timestamp_ms=purged_before,
            rows_threshold=None,
        )
    )
    today = datetime.fromtimestamp(now_ms / 1000, UTC).date()
    # `days` days of history, today included: the oldest day kept is `days - 1` days back
    dropped = await drop_expired_partitions(today - timedelta(days=settings.job_days - 1))
    return ArchiveReport(copied=copied, purged_before_ms=purged_before, dropped=dropped)


async def copy_batch(cutoff_ms: int) -> int:
    """Copy one batch of jobs that finished at or before `cutoff_ms` into their day partitions.

    One transaction: the watermark is read, the rows are written, and the watermark moves in the
    same commit, so a crash replays exactly what the last commit did not cover. Returns how many
    jobs were copied; fewer than `ARCHIVE_PAGE` means the cutoff was reached."""
    async with db.connect() as conn:
        mark = await _watermark(conn)
        # DBOS sets `completed_at` on every terminal transition and clears it on a resume, so this
        # is both "finished" and the exact predicate its own garbage collection uses
        rows: list[tuple[str, int]] = _tuples(
            await conn.execute_fetchall(
                "select workflow_uuid, completed_at from workflow_status "
                f"where name in ({db.placeholders(len(PIPELINE_WORKFLOWS))}) "
                "and completed_at > ? and completed_at <= ? "
                "order by completed_at limit ?",
                (*PIPELINE_WORKFLOWS, mark, cutoff_ms, ARCHIVE_PAGE),
            )
        )
        completed: dict[str, int] = dict(rows)
        if completed:
            await _copy_jobs(conn, completed)
        # a short batch means the whole window is archived, so the watermark can jump to its end
        reached = cutoff_ms if len(rows) < ARCHIVE_PAGE else max(completed.values())
        await _set_watermark(conn, max(mark, reached))
        return len(rows)


async def _copy_jobs(conn: aiosqlite.Connection, completed: dict[str, int]) -> None:
    """Read the jobs and their children out of DBOS and write them into their day partitions."""
    # local: `jobs` reads the partitions written here, so importing it at module level would close
    # a cycle. It owns the read model; this module only stores it.
    from haskie import jobs

    ids = list(completed)
    children = await jobs.stage_children(ids)
    for day in {day_suffix(ms) for ms in completed.values()}:
        await ensure_partition(conn, day)  # once per day, not once per job
    for status in await DBOS.list_workflows_async(workflow_ids=ids, load_input=False):
        job, tasks = await jobs.snapshot(status, children.get(status.workflow_id, []))
        completed_at = completed[status.workflow_id]
        day = day_suffix(completed_at)
        await conn.execute(
            f"insert or replace into jobs_{day} ({_column_list(JOB_COLUMNS)}) "
            f"values ({', '.join('?' * len(JOB_COLUMNS))})",
            (
                job.id,
                job.action,
                job.collection,
                job.doc,
                job.status,
                status.created_at or 0,
                completed_at,
                job.error,
                job.tasks_done,
                job.tasks_total,
            ),
        )
        await conn.executemany(
            f"insert or replace into tasks_{day} ({_column_list(TASK_COLUMNS)}) "
            f"values ({', '.join('?' * len(TASK_COLUMNS))})",
            [
                (
                    job.id,
                    task.child_id,
                    task.stage,
                    task.seq,
                    task.page_start,
                    task.page_end,
                    task.status,
                    task.result,
                    task.error,
                )
                for task in tasks
            ],
        )
