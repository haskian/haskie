"""Read model over the job history: one `Job` per `index_document` workflow, one `Task` per
micro-batch below it, and one `BulkJob` per whole-library index or delete. Nothing here is stored;
every field comes from DBOS's workflow tables, or - once a job is older than the live window -
from the day partition `archive` copied it into.

A job has a child workflow per slice of its convert and embed stages plus one for its index stage
(see `workflows`), so its tasks are the micro-batches inside them: the batch list comes from each
child's input, the finished count from its step log (`sysdb`, one grouped query for the whole
page). Counts are summed over the children, so how a stage was sliced never shows here.

Every other kind of workflow this app runs is listed through one generic read model instead:
`list_kind` returns a page of `JobRow` for a whole-library job, a model download, a maintenance
run or an archive round, so the Jobs view has a section per kind and a model still coming down is
visible as a job instead of only as a 503 on a search. Documents are a kind there too, mapped from
the `Job` above, because they alone also have tasks, a cancel and a day partition to fall back to.

The library and the document are read out of the workflow id (`idx:{library}:{doc}:{uuid}`), never
out of the recorded input: that makes the library filter an id prefix the database can apply, and
a page of jobs costs no input payloads at all. Every other kind whose work belongs to one library
carries the name in the same place (`{prefix}:{library}:{rest}`), and a download reads its kind and
model out of its id (`dl:{kind}:{model}`), for the same reason.

Every listing is a read of SQLite - DBOS's own tables through its `*_async` API, the day
partitions through `aiosqlite` - so every one of them is awaited. The row builders below take
what those reads returned and touch nothing: they stay sync.
"""

from typing import Literal

import aiosqlite
import msgspec
from dbos import DBOS

from haskie import archive, db, models, sysdb, workflows
from haskie.dbos_names import ACTIVE_STATUS, PENDING_STATUS, STAGE_STEP, STAGE_WORKFLOW
from haskie.errors import InvalidInput, JobNotFound
from haskie.library import Library
from haskie.paging import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, OffsetCursor, Order, Page
from haskie.pipeline import Batch
from haskie.workflows import STAGE_ORDER, BatchResult, Stage

DONE_TASK_STATUS = frozenset({"SUCCESS", "ERROR"})  # a batch whose step DBOS recorded

# Jobs are always newest first, so the cursor carries no sort of its own; it names the source the
# next page continues in (the live DBOS history, or one day partition) and the offset into it,
# which is all an ordered-by-created_at listing of a workflow history can page on.
JOB_SORT = "created_at"
JOB_ORDER: Order = "desc"
# Both job listings page on an offset: DBOS's history has one fixed order and no sort of its own.
CURSOR = OffsetCursor(JOB_SORT, JOB_ORDER)
LIVE = "live"  # the source name of the DBOS history itself


class Job(msgspec.Struct):
    id: str
    library: str
    doc: str
    status: str  # DBOS: ENQUEUED | PENDING | SUCCESS | ERROR | CANCELLED | ...
    created_at: float
    updated_at: float
    error: str | None
    tasks_done: int = 0
    tasks_running: int = 0
    tasks_total: int = 0
    archived: bool = False  # read from a day partition: DBOS no longer holds the job


BulkKind = Literal["index_library", "delete_library"]

# Workflow name -> the kind the API names it by. A job id that is not one of these two is not a
# bulk job, which is how `bulk_job` tells an unknown id from a document job.
BULK_KINDS: dict[str, BulkKind] = {
    workflows.index_library_workflow.__qualname__: "index_library",
    workflows.delete_library_workflow.__qualname__: "delete_library",
}


class BulkJob(msgspec.Struct):
    """One whole-library job: what the 202 of an "index all" or a library delete points at."""

    id: str
    kind: BulkKind
    library: str
    status: str  # DBOS: ENQUEUED | PENDING | SUCCESS | ERROR | CANCELLED | ...
    progress: workflows.BulkProgress | None = None
    error: str | None = None


DOWNLOAD_PAGE = 50  # a boot needs a handful of models; the rest is history nobody scrolls


class DownloadJob(msgspec.Struct):
    """One model download: the DBOS status of its `ensure_model` workflow is the model's state."""

    id: str
    kind: str  # embedding | reranker
    model: str
    status: str  # DBOS: ENQUEUED | PENDING | SUCCESS | ERROR | CANCELLED | ...
    created_at: float
    error: str | None = None
    # SUCCESS says the files are on disk; this says the model is loaded in *this* process, which
    # is what a search needs (see `models`)
    warm: bool = False


JobKind = Literal["document", "library", "download", "maintenance", "archive"]

# The sections of the Jobs view, in the order it shows them.
KIND_ORDER: tuple[JobKind, ...] = ("document", "library", "download", "maintenance", "archive")
KIND_LABELS: dict[JobKind, str] = {
    "document": "Documents",
    "library": "Libraries",
    "download": "Model downloads",
    "maintenance": "Maintenance",
    "archive": "Archive",
}


class JobRow(msgspec.Struct):
    """One job of any kind, in the shape the Jobs view lists: what every kind has in common, plus
    the numbers only that kind has in `detail` (tasks for a document, pages for a bulk index,
    warm for a download). Kept flat and untyped on purpose - it is a read model for a table."""

    id: str
    kind: JobKind
    title: str  # human text: "library / doc", "index library X", "download reranker Y"
    status: str  # DBOS: ENQUEUED | PENDING | SUCCESS | ERROR | CANCELLED | ...
    created_at: float
    updated_at: float
    error: str | None
    archived: bool = False  # documents only: read from a day partition instead of DBOS
    detail: dict[str, int | str | bool | None] = {}


class JobKindSummary(msgspec.Struct):
    """One section of the Jobs view: what to call it, and how much of it is running right now."""

    kind: JobKind
    label: str
    active: int


class Task(msgspec.Struct):
    id: str
    stage: Stage
    seq: int
    # convert: PDF pages [start, end); embed: the one part; index: the part range written together
    page_start: int
    page_end: int
    status: str
    result: int | None  # convert: pages needing OCR; embed/index: chunks
    error: str | None


def _names(workflow_id: str) -> tuple[str, str]:
    """`idx:{library}:{doc}:{uuid}` -> (library, doc); ("?", "?") for any other id.

    `library.safe_name` keeps `:` out of both names, so the split is exact."""
    parts = workflow_id.split(":", 3)
    return (parts[1], parts[2]) if len(parts) == 4 and parts[0] == "idx" else ("?", "?")


def _download_names(workflow_id: str) -> tuple[str, str]:
    """`dl:{kind}:{model}` -> (kind, model); ("?", workflow_id) for any other id.

    The model name is the rest of the id, colons and all, so a name that carries one still reads
    back whole."""
    parts = workflow_id.split(":", 2)
    if len(parts) != 3 or parts[0] != "dl":
        return "?", workflow_id
    return parts[1], parts[2]


async def list_downloads() -> list[DownloadJob]:
    """Model downloads, newest first: one row per model the installation has ever fetched, since
    the record outlives the process that fetched it (see `models`)."""
    found = await _kind_statuses("download", None, DOWNLOAD_PAGE, 0)
    return [_download(status) for status in found]


def _download(status) -> DownloadJob:
    kind, model = _download_names(status.workflow_id)
    return DownloadJob(
        id=status.workflow_id,
        kind=kind,
        model=model,
        status=status.status,
        created_at=(status.created_at or 0) / 1000,
        error=str(status.error) if status.error else None,
        warm=models.is_warm(status.workflow_id),
    )


def _library_of(workflow_id: str) -> str:
    """`{prefix}:{library}:{rest}` -> library; "?" for any other id. Every job that belongs to one
    library carries the name in the same place (see the module docstring)."""
    parts = workflow_id.split(":", 2)
    return parts[1] if len(parts) == 3 else "?"


# --- one listing per kind of job ----------------------------------------------------------

# Stage -> the DBOS workflow names it lists. `document` is missing on purpose: it has a listing of
# its own (`list_jobs`), because it is the only kind that also reads the day partitions.
KIND_NAMES: dict[JobKind, list[str]] = {
    "library": [
        workflows.index_library_workflow.__qualname__,
        workflows.delete_library_workflow.__qualname__,
    ],
    "download": [models.ensure_model.__qualname__],
    # `maintain_library` is only the debounced handle that waits: the run itself is the child on
    # the library's partition, so that is the one worth a row.
    "maintenance": [
        workflows.maintain_on_partition.__qualname__,
        workflows.daily_maintenance.__qualname__,
    ],
    "archive": [workflows.archive_jobs.__qualname__],
}

# The same table read the other way, for counting active workflows by kind in one query.
KIND_BY_NAME: dict[str, JobKind] = {
    workflows.index_document.__qualname__: "document",
    **{name: kind for kind, names in KIND_NAMES.items() for name in names},
}

# Stage -> the id prefix that keeps one library's jobs only. Downloads and archive rounds belong to
# no library, so a library filter leaves them alone rather than emptying the section.
_LIBRARY_PREFIX: dict[JobKind, list[str]] = {
    "library": [f"{workflows.BULK_INDEX_PREFIX}:", f"{workflows.BULK_DELETE_PREFIX}:"],
    "maintenance": [f"{workflows.MAINTAIN_PREFIX}:"],
}


def _checked_kind(kind: str) -> JobKind:
    """A kind is a trust boundary: an unknown one is a mistake in the request, not an empty page."""
    for known in KIND_ORDER:
        if kind == known:
            return known
    raise InvalidInput(f"unknown job kind {kind!r}; allowed: {', '.join(KIND_ORDER)}")


def _kind_prefix(kind: JobKind, library: str | None) -> list[str] | None:
    if library is None:
        return None
    return [f"{prefix}{library}:" for prefix in _LIBRARY_PREFIX.get(kind, [])] or None


async def _kind_statuses(kind: JobKind, library: str | None, limit: int, offset: int) -> list:
    """One window of the DBOS history for one kind, newest first. Inputs stay on disk; the output
    is loaded only because DBOS carries a workflow's error alongside it."""
    return await DBOS.list_workflows_async(
        name=KIND_NAMES[kind],
        workflow_id_prefix=_kind_prefix(kind, library),
        sort_desc=True,
        limit=limit,
        offset=offset,
        load_input=False,
        load_output=True,
    )


async def list_kind(
    kind: str,
    library: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> Page[JobRow]:
    """One page of the jobs of one kind, newest first.

    Documents are the one kind with history older than the live window, so they keep their own
    listing and are mapped into the common shape here; the rest are one window of the DBOS
    history each, paged on an opaque offset cursor bound to the kind that issued it."""
    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise InvalidInput(f"page_size must be 1..{MAX_PAGE_SIZE}, got {page_size}")
    checked = _checked_kind(kind)
    if checked == "document":
        page = await list_jobs(library, page_size, cursor)
        return Page(items=[_document_row(job) for job in page.items], next_cursor=page.next_cursor)
    offset = _kind_offset(checked, cursor)
    # one row more than the page: its presence is what tells us another page exists
    found = await _kind_statuses(checked, library, page_size + 1, offset)
    return Page(
        items=[await _kind_row(checked, status) for status in found[:page_size]],
        next_cursor=_cursor(checked, offset + page_size) if len(found) > page_size else None,
    )


def _kind_offset(kind: JobKind, cursor: str | None) -> int:
    """The offset a cursor continues at; 0 without one. A cursor from another kind's listing would
    page a different history, so it is rejected rather than followed."""
    if cursor is None:
        return 0
    source, offset = CURSOR.decode(cursor)
    if source != kind:
        raise InvalidInput("invalid cursor")
    return offset


async def list_kinds() -> list[JobKindSummary]:
    """Every kind, in the order the Jobs view shows them, with how many of each are enqueued or
    running right now - one grouped query for all of them, not one per section."""
    active = await sysdb.active_counts_by_name()
    counts: dict[JobKind, int] = dict.fromkeys(KIND_ORDER, 0)
    for name, count in active.items():
        kind = KIND_BY_NAME.get(name)
        if kind is not None:
            counts[kind] += count
    return [
        JobKindSummary(kind=kind, label=KIND_LABELS[kind], active=counts[kind])
        for kind in KIND_ORDER
    ]


class QueueActivity(msgspec.Struct):
    """One queue family right now: `queued` waits for a slot, `running` holds one.

    A debounced run waiting out its period (DELAYED) is neither: it is not work anyone is waiting
    on, so it stays out of the indicator (see `sysdb.queue_activity`)."""

    queued: int
    running: int


class Activity(msgspec.Struct):
    """The indicator every view shows: coarse jobs (`job.*` queues) and the tasks they are made of
    (`task.*` queues)."""

    jobs: QueueActivity
    tasks: QueueActivity


async def activity() -> Activity:
    by_family = await sysdb.queue_activity()

    def family(name: str) -> QueueActivity:
        counts = by_family.get(name, {})
        running = counts.get(PENDING_STATUS, 0)
        return QueueActivity(queued=sum(counts.values()) - running, running=running)

    return Activity(jobs=family("job"), tasks=family("task"))


def _document_row(job: Job) -> JobRow:
    return JobRow(
        id=job.id,
        kind="document",
        title=f"{job.library} / {job.doc}",
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error=job.error,
        archived=job.archived,
        detail={
            "tasks_done": job.tasks_done,
            "tasks_running": job.tasks_running,
            "tasks_total": job.tasks_total,
        },
    )


async def _kind_row(kind: JobKind, status) -> JobRow:
    return JobRow(
        id=status.workflow_id,
        kind=kind,
        title=_title(kind, status),
        status=status.status,
        created_at=(status.created_at or 0) / 1000,
        updated_at=(status.updated_at or 0) / 1000,
        error=str(status.error) if status.error else None,
        detail=await _detail(kind, status),
    )


def _title(kind: JobKind, status) -> str:
    """Human text for one row: what this job is doing, read out of its id and its workflow name.

    `document` never arrives here: it has a row builder of its own (see `list_kind`)."""
    if kind == "library":
        verb = "index" if BULK_KINDS.get(status.name) == "index_library" else "delete"
        return f"{verb} library {_library_of(status.workflow_id)}"
    if kind == "download":
        download_kind, model = _download_names(status.workflow_id)
        return f"download {download_kind} {model}"
    if kind == "maintenance":
        if status.name == workflows.daily_maintenance.__qualname__:
            return "daily housekeeping"
        return f"maintain {_library_of(status.workflow_id)}"
    return "archive finished jobs"  # the hourly retention round


async def _detail(kind: JobKind, status) -> dict[str, int | str | bool | None]:
    """The numbers only this kind has. A bulk index publishes its progress as a DBOS event, which
    is read without waiting: a job that has not finished its first page yet simply has none.

    Async because that read is one, even with no wait: the event lives in the system database."""
    if kind == "download":
        return {"warm": models.is_warm(status.workflow_id)}
    if kind == "library":
        progress = await DBOS.get_event_async(
            status.workflow_id, workflows.PROGRESS_EVENT, timeout_seconds=0
        )
        if isinstance(progress, workflows.BulkProgress):
            return {"done": progress.done, "skipped": progress.skipped, "total": progress.total}
    return {}


def _job(status, children: list, done_by_child: dict[str, int]) -> Job:
    library, doc = _names(status.workflow_id)
    totals = [len(found[1]) if (found := workflows.stage_input(c)) else 0 for c in children]
    done = [done_by_child.get(c.workflow_id, 0) for c in children]
    return Job(
        id=status.workflow_id,
        library=library,
        doc=doc,
        status=status.status,
        created_at=(status.created_at or 0) / 1000,
        updated_at=(status.updated_at or 0) / 1000,
        error=str(status.error) if status.error else None,
        tasks_done=sum(done),
        # a running child works on exactly one batch: its steps are sequential
        tasks_running=sum(
            c.status == "PENDING" and d < t for c, d, t in zip(children, done, totals, strict=True)
        ),
        tasks_total=sum(totals),
    )


def _start(cursor: str | None) -> tuple[str, int]:
    """The source and the offset inside it a cursor continues at; the live history without one."""
    if cursor is None:
        return LIVE, 0
    source, offset = CURSOR.decode(cursor)
    if source != LIVE and archive.partition_day(source) is None:
        raise InvalidInput("invalid cursor")
    return source, offset


def _cursor(source: str, offset: int) -> str:
    return CURSOR.encode(source, offset)


async def _sources(conn: aiosqlite.Connection, source: str) -> list[str]:
    """What is still to be read, starting at the source the cursor names: the live history, then
    every day partition, newest first.

    Empty when that source is gone - retention dropped the day while the listing was being walked
    - so the walk ends there rather than serving another day's jobs in its place. This is also
    what keeps a cursor out of the SQL: only a name `sqlite_master` returned is ever read."""
    available = [LIVE, *await archive.list_partitions(conn)]
    return available[available.index(source) :] if source in available else []


async def list_jobs(
    library: str | None = None, page_size: int = DEFAULT_PAGE_SIZE, cursor: str | None = None
) -> Page[Job]:
    """One page of jobs, newest first: the live DBOS history, then one day partition at a time
    until the page is full.

    The library filter is the workflow id's prefix in the live history and an indexed column in a
    partition, so the database cuts the window after it has filtered: a busy library can no longer
    push a quiet one out of the page (A11). Inputs stay on disk; the output is loaded only because
    DBOS carries a workflow's error alongside it.

    A day partition that appears while the page is walked (the archiver runs on its own clock)
    shifts the offsets behind it, so a job can repeat or be skipped across a page boundary - the
    same trade an offset cursor over a live history always makes."""
    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise InvalidInput(f"page_size must be 1..{MAX_PAGE_SIZE}, got {page_size}")
    source, offset = _start(cursor)
    items: list[Job] = []
    async with db.connect() as conn:
        sources = await _sources(conn, source)
        for index, name in enumerate(sources):
            want = page_size - len(items)  # never 0: a full page returns below
            at = offset if index == 0 else 0
            # one row more than the page: its presence is what tells us this source has more
            found = (
                await _live_jobs(library, want + 1, at)
                if name == LIVE
                else await _archived_jobs(conn, name, library, want + 1, at)
            )
            items.extend(found[:want])
            if len(found) > want:  # the next page continues inside this source
                return Page(items=items, next_cursor=_cursor(name, at + want))
            if len(items) == page_size:  # full, and this source is exhausted: continue at the next
                older = sources[index + 1 :]
                return Page(items=items, next_cursor=_cursor(older[0], 0) if older else None)
    return Page(items=items, next_cursor=None)


async def _live_jobs(library: str | None, limit: int, offset: int) -> list[Job]:
    """One window of the DBOS history, newest first, with the batch counts of the whole window
    read in one grouped query."""
    statuses = await DBOS.list_workflows_async(
        name=workflows.index_document.__qualname__,
        workflow_id_prefix=f"idx:{library}:" if library else "idx:",
        sort_desc=True,
        limit=limit,
        offset=offset,
        load_input=False,
    )
    ids = [s.workflow_id for s in statuses]
    children: dict[str, list] = {i: [] for i in ids}
    if ids:
        found = await DBOS.list_workflows_async(
            parent_workflow_id=ids,
            name=STAGE_WORKFLOW,
            load_output=False,
        )
        for c in found:
            children.setdefault(c.parent_workflow_id or "", []).append(c)
    done = await sysdb.step_counts(
        [c.workflow_id for group in children.values() for c in group], STAGE_STEP
    )
    return [_job(s, children[s.workflow_id], done) for s in statuses]


async def _archived_jobs(
    conn: aiosqlite.Connection, table: str, library: str | None, limit: int, offset: int
) -> list[Job]:
    rows = await archive.job_page(conn, table, library, limit, offset)
    return [_archived_job(row) for row in rows]


def _archived_job(row: tuple) -> Job:
    """One row of a day partition, in `archive.JOB_COLUMNS` order. Nothing is running in an
    archived job: it was copied because it had finished."""
    job_id, library, doc, status, created_at, completed_at, error, done, total = row
    return Job(
        id=job_id,
        library=library,
        doc=doc,
        status=status,
        created_at=created_at / 1000,
        updated_at=completed_at / 1000,
        error=error,
        tasks_done=done,
        tasks_total=total,
        archived=True,
    )


def _ordered(tasks: list[Task]) -> list[Task]:
    """Stage by stage, batch by batch: the order the pipeline planned them in. Slices of one
    stage run side by side, so this is a plan order, not a finishing order."""
    return sorted(tasks, key=lambda t: (STAGE_ORDER.index(t.stage), t.seq))


async def snapshot(status, children: list) -> tuple[Job, list[Task]]:
    """The whole read model of one job, built from its own children and step logs: what `archive`
    copies into a day partition. A page of jobs counts finished batches with one grouped query
    instead, which is cheaper but cannot see the individual batches."""
    tasks = {c.workflow_id: await _stage_tasks(c) for c in children if c.input}
    done = {c: sum(t.status in DONE_TASK_STATUS for t in group) for c, group in tasks.items()}
    return _job(status, children, done), _ordered([t for group in tasks.values() for t in group])


async def list_tasks(job_id: str) -> list[Task]:
    """The micro-batches of one job, live or archived."""
    if await DBOS.get_workflow_status_async(job_id) is not None:
        out: list[Task] = []
        children = await DBOS.list_workflows_async(parent_workflow_id=job_id, name=STAGE_WORKFLOW)
        for child in children:
            if child.input:
                out.extend(await _stage_tasks(child))
        return _ordered(out)
    async with db.connect() as conn:
        # the job row decides, not the tasks: a job that failed before it planned anything has none
        for table in await archive.list_partitions(conn):
            if await archive.has_job(conn, table, job_id):
                rows = await archive.task_rows(conn, archive.tasks_table(table), job_id)
                return _ordered([_archived_task(row) for row in rows])
    raise JobNotFound(f"job not found: {job_id}")


def _archived_task(row: tuple) -> Task:
    """One row of a day partition, in `archive.TASK_COLUMNS` order."""
    _job_id, child_id, stage, seq, page_start, page_end, status, result, error = row
    return Task(
        id=f"{child_id}:{seq}",
        stage=stage,
        seq=seq,
        page_start=page_start,
        page_end=page_end,
        status=status,
        result=result,
        error=error,
    )


def _task(id: str, stage: Stage, batch: Batch, status: str, output, error) -> Task:
    return Task(
        id=id,
        stage=stage,
        seq=batch.seq,
        page_start=batch.start,
        page_end=batch.end,
        status=status,
        result=output if isinstance(output, int) else None,
        error=str(error) if error else None,
    )


async def _stage_tasks(child) -> list[Task]:
    """One Task per micro-batch of one stage child: finished batches from its step log; the next
    one is running while the child runs; the rest wait. A child holds one slice of the stage, and
    the batches keep the `seq` the plan gave them, so the slices reassemble by `seq` alone."""
    found = workflows.stage_input(child)
    if found is None:
        return []
    stage, batches = found
    steps = [
        st
        for st in await DBOS.list_workflow_steps_async(child.workflow_id)
        if st["function_name"] == workflows.try_batch.__qualname__
    ]
    out: list[Task] = []
    for i, batch in enumerate(batches):
        if i < len(steps):
            output, error = _step_outcome(steps[i])
            status = "ERROR" if error else "SUCCESS"
        else:
            output = error = None
            status = "PENDING" if child.status == "PENDING" and i == len(steps) else "ENQUEUED"
        out.append(_task(f"{child.workflow_id}:{batch.seq}", stage, batch, status, output, error))
    return out


def _step_outcome(step) -> tuple[int | None, str | None]:
    """A step that failed permanently returns its message instead of raising (see BatchResult)."""
    output, error = step["output"], step["error"]
    if isinstance(output, BatchResult):
        return output.value, output.permanent_error
    return (output if isinstance(output, int) else None), (str(error) if error else None)


async def bulk_job(job_id: str) -> BulkJob:
    """The state of one whole-library job: its DBOS status plus the progress event a bulk index
    publishes after every page. `progress` stays None for a delete, which has no pages, and for
    an index that has not finished its first page yet."""
    status = await DBOS.get_workflow_status_async(job_id)
    if status is None:
        raise JobNotFound(f"job not found: {job_id}")
    kind = BULK_KINDS.get(status.name)
    if kind is None:  # a job of another kind: not a bulk job
        raise JobNotFound(f"job not found: {job_id}")
    progress = await DBOS.get_event_async(job_id, workflows.PROGRESS_EVENT, timeout_seconds=0)
    return BulkJob(
        id=job_id,
        kind=kind,
        library=_library_of(job_id),
        status=status.status,
        progress=progress if isinstance(progress, workflows.BulkProgress) else None,
        error=str(status.error) if status.error else None,
    )


async def cancel_job(job_id: str) -> None:
    """No-op on a job that already finished: its document status is final (A12)."""
    found = await DBOS.get_workflow_status_async(job_id)
    if found is None:
        raise JobNotFound(f"job not found: {job_id}")
    if found.status not in ACTIVE_STATUS:
        return
    await DBOS.cancel_workflow_async(job_id, cancel_children=True)
    library, doc = _names(job_id)
    if library != "?":
        await Library(library).set_status(doc, "cancelled")
