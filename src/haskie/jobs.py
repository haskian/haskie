"""Read model over the job history: one `Job` per pipeline workflow, one `Task` per micro-batch
below it, and one `BulkJob` per whole-collection or whole-document job. Nothing here is stored;
every field comes from DBOS's workflow tables, which the nightly retention round bounds.

Three workflows carry a document through the pipeline (`dbos_names.PIPELINE_WORKFLOWS`), and each
is a job of its own: `import_document` converts it, `ensure_embedding` fills its embedding cache,
`index_collection_document` writes one collection's table from that cache. Every one of them cuts
its stage into `stage_slice` children with a durable step per micro-batch, so a job's tasks are
those micro-batches: the batch list comes from each child's input, the finished count from its
step log (`sysdb`, one grouped query for the whole page). Counts are summed over the children, so
how a stage was sliced never shows here.

Every other kind of workflow this app runs is listed through one generic read model instead:
`list_kind` returns a page of `JobRow` for a whole-collection job, a model download or a
maintenance run, so the Jobs view has a section per kind. Documents are a kind there too, mapped
from the `Job` above, because they alone also have tasks and a cancel.

Reads only: cancelling a job writes, so it lives in `workflows` beside the ids it writes by.

The action, the collection and the document are read out of the workflow id (`imp:{doc}:{uuid}`,
`emb:{doc}:{uuid}`, `idx-col:{collection}:{doc}:{uuid}`, through `workflows.job_names`), never out
of the recorded input: that makes the collection filter an id prefix the database can apply, and a
page of jobs costs no input payloads at all. Every other kind whose work belongs to one collection
carries the name in the same place (`{prefix}:{collection}:{rest}`), and a download reads its kind
and model out of its id (`dl:{kind}:{model}`), for the same reason.

Every listing is a read of DBOS's own tables through its `*_async` API, so every one of them is
awaited. The row builders below take what those reads returned and touch nothing: they stay sync.
"""

import asyncio
from typing import Literal, get_args

import msgspec
from dbos import DBOS

from haskie import models, session, sysdb, workflows
from haskie.dbos_names import (
    ACTIVE_STATUS,
    BULK_WORKFLOWS,
    COLLECTION_DOCUMENT_WORKFLOW,
    DAILY_MAINTENANCE_WORKFLOW,
    DOWNLOAD_WORKFLOW,
    MAINTAIN_PARTITION_WORKFLOW,
    PENDING_STATUS,
    PIPELINE_WORKFLOWS,
    STAGE_STEP,
    STAGE_WORKFLOW,
    BulkWorkflow,
    WorkflowStatus,
)
from haskie.errors import InvalidInput, NotFound
from haskie.paging import DEFAULT_PAGE_SIZE, OffsetCursor, Order, Page, check_page_size
from haskie.pipeline import Batch
from haskie.workflows import STAGE_ORDER, BatchResult, JobAction, Stage, job_names

# Jobs are always newest first, so the cursor carries no sort of its own; it names the listing the
# next page continues in and the offset into it, which is all an ordered-by-created_at listing of a
# workflow history can page on.
JOB_SORT = "created_at"
JOB_ORDER: Order = "desc"
# Both job listings page on an offset: DBOS's history has one fixed order and no sort of its own.
CURSOR = OffsetCursor(JOB_SORT, JOB_ORDER)


class Job(msgspec.Struct):
    """One run of one pipeline workflow over one document.

    `collection` is None for an import and an embed: both are collection-independent, and only the
    index of a member belongs to a collection."""

    id: str
    action: JobAction
    collection: str | None
    doc: str
    status: WorkflowStatus
    created_at: float
    updated_at: float
    error: str | None
    tasks_done: int = 0
    tasks_running: int = 0
    tasks_total: int = 0


# A whole-collection or whole-document job is reported under the name DBOS records it as, so the
# kind the API reports is the workflow name (`test_workflows` pins that against the registry).
BulkKind = BulkWorkflow
BULK_KINDS: tuple[BulkKind, ...] = BULK_WORKFLOWS
BULK_TITLES: dict[BulkKind, str] = {
    "index_collection": "index collection",
    "delete_collection": "delete collection",
    "delete_document": "delete document",
}


class BulkJob(msgspec.Struct):
    """One whole-collection or whole-document job: what the 202 of an "index all", a collection
    delete or a document delete points at.

    `collection` is None for a document delete: it spans every collection the document is in."""

    id: str
    kind: BulkKind
    collection: str | None
    status: WorkflowStatus
    progress: workflows.BulkProgress | None = None
    error: str | None = None


# Declared in the order the Jobs view shows the sections, so `KIND_ORDER` is the type itself.
JobKind = Literal["document", "collection", "download", "maintenance"]
KIND_ORDER: tuple[JobKind, ...] = get_args(JobKind)
KIND_LABELS: dict[JobKind, str] = {
    "document": "Documents",
    "collection": "Collections",
    "download": "Model downloads",
    "maintenance": "Maintenance",
}
DOCUMENT_KIND: JobKind = "document"  # the one kind with a listing of its own, and its own cursor


class StageJob(msgspec.Struct):
    """One stage of a document operation, and the workflow whose batches it is made of: the
    convert and index stages run in the operation's own workflow, the embed stage in the
    `ensure_embedding` child it spawns (see `fold_operations`)."""

    stage: Stage
    job_id: str  # pass it to `list_tasks` for this stage's batches
    status: WorkflowStatus
    tasks_done: int
    tasks_running: int
    tasks_total: int
    seconds: float | None = None  # how long the stage ran; None while it still does


class JobRow(msgspec.Struct):
    """One job of any kind, in the shape the Operations view lists: what every kind has in common,
    plus the numbers only that kind has in `detail` (tasks for a document, pages for a bulk index,
    warm for a download). Kept flat and untyped on purpose - it is a read model for a table.

    A document row is one operation (an import, or an index of one document) and lists its stages:
    the jobs it is made of, each with the batches it ran."""

    id: str
    kind: JobKind
    title: str  # human text: "collection / doc", "index collection X", "download reranker Y"
    status: WorkflowStatus
    created_at: float
    updated_at: float
    error: str | None
    origin: str | None = None  # the session whose action started it; None for the web UI
    detail: dict[str, int | str | bool | None] = {}
    stages: list[StageJob] = []  # documents only, in pipeline order


class JobKindSummary(msgspec.Struct):
    """One section of the Jobs view: what to call it, and how much of it is running right now."""

    kind: JobKind
    label: str
    active: int


class Task(msgspec.Struct):
    id: str
    child_id: str  # the `stage_slice` workflow that ran the batch; the task id is `{it}:{seq}`
    stage: Stage
    seq: int
    # convert: PDF pages [start, end); embed: the one part; index: the part range written together
    page_start: int
    page_end: int
    status: WorkflowStatus
    result: int | None  # convert: pages needing OCR; embed/index: chunks
    error: str | None


def _named(workflow_id: str) -> str:
    """The name in the second segment of an id (`{prefix}:{name}:{rest}`): the one thing the job
    is about, whichever kind of name it is. "?" for any other id."""
    parts = workflow_id.split(":", 2)
    return parts[1] if len(parts) == 3 else "?"


def _collection_of(workflow_id: str) -> str | None:
    """The collection a job belongs to; None when it belongs to none.

    A document delete is the exception: `del-doc:{doc}:{uuid}` carries a document where every
    other id carries a collection, and the delete spans every collection the document is in."""
    if workflow_id.startswith(f"{workflows.DELETE_DOCUMENT_PREFIX}:"):
        return None
    name = _named(workflow_id)
    return None if name == "?" else name


# --- one listing per kind of job ----------------------------------------------------------

# Kind -> the DBOS workflow names it lists. `document` is missing on purpose: it has a listing of
# its own (`list_jobs`), because it is the only kind whose rows carry micro-batch counts.
KIND_NAMES: dict[JobKind, list[str]] = {
    "collection": list(BULK_KINDS),
    "download": [DOWNLOAD_WORKFLOW],
    # `maintain_collection` is only the debounced handle that waits: the run itself is the child
    # on the collection's partition, so that is the one worth a row.
    "maintenance": [MAINTAIN_PARTITION_WORKFLOW, DAILY_MAINTENANCE_WORKFLOW],
}

# The same table read the other way, for counting active workflows by kind in one query.
KIND_BY_NAME: dict[str, JobKind] = {
    **dict.fromkeys(PIPELINE_WORKFLOWS, "document"),
    **{name: kind for kind, names in KIND_NAMES.items() for name in names},
}

# Kind -> the id prefix that keeps one collection's jobs only. Downloads and document deletes
# belong to no collection, so a collection filter leaves the section holding them empty rather
# than unfiltered.
_COLLECTION_PREFIX: dict[JobKind, list[str]] = {
    "collection": [f"{workflows.BULK_INDEX_PREFIX}:", f"{workflows.BULK_DELETE_PREFIX}:"],
    "maintenance": [f"{workflows.MAINTAIN_PREFIX}:"],
}


def _checked_kind(kind: str) -> JobKind:
    """A kind is a trust boundary: an unknown one is a mistake in the request, not an empty page."""
    if kind not in KIND_ORDER:
        raise InvalidInput(f"unknown job kind {kind!r}; allowed: {', '.join(KIND_ORDER)}")
    return kind


def _bulk_kind(name: str | None) -> BulkKind | None:
    """The kind a whole-collection workflow name stands for; None for any other workflow."""
    return name if name in BULK_KINDS else None


def _kind_prefix(kind: JobKind, collection: str | None) -> list[str] | None:
    if collection is None:
        return None
    return [f"{prefix}{collection}:" for prefix in _COLLECTION_PREFIX.get(kind, [])] or None


async def _kind_statuses(kind: JobKind, collection: str | None, limit: int, offset: int) -> list:
    """One window of the DBOS history for one kind, newest first. Inputs stay on disk; the output
    is loaded only because DBOS carries a workflow's error alongside it."""
    return await DBOS.list_workflows_async(
        name=KIND_NAMES[kind],
        workflow_id_prefix=_kind_prefix(kind, collection),
        sort_desc=True,
        limit=limit,
        offset=offset,
        load_input=False,
        load_output=True,
    )


async def list_kind(
    kind: str,
    collection: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> Page[JobRow]:
    """One page of the jobs of one kind, newest first.

    Documents keep a listing of their own, with the batch counts only they carry, and are mapped
    into the common shape here; the rest are one window of the DBOS history each, paged on an
    opaque offset cursor bound to the kind that issued it."""
    check_page_size(page_size)
    checked = _checked_kind(kind)
    if checked == DOCUMENT_KIND:
        page = await list_jobs(collection, page_size, cursor)
        return Page(
            items=await _with_origins(fold_operations(page.items)), next_cursor=page.next_cursor
        )
    offset = _decode_cursor(cursor, checked)
    # one row more than the page: its presence is what tells us another page exists
    found = await _kind_statuses(checked, collection, page_size + 1, offset)
    # every row costs a read of its own (see `_detail`), so the page is built in one round trip
    rows = await asyncio.gather(*(_kind_row(checked, s) for s in found[:page_size]))
    return Page(
        items=await _with_origins(list(rows)),
        next_cursor=_cursor(checked, offset + page_size) if len(found) > page_size else None,
    )


async def _with_origins(rows: list[JobRow]) -> list[JobRow]:
    """Name the session that started each row, where one did: one query for the page."""
    origins = await session.origins([row.id for row in rows])
    for row in rows:
        row.origin = origins.get(row.id)
    return rows


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
    # The embed stage an import or an index spawned is waited for by the one that spawned it (see
    # `fold_operations`): counting both would read "2 jobs" for one operation.
    by_family = await sysdb.queue_activity(skip=[workflows.EMBEDDING_QUEUE])

    def family(name: str) -> QueueActivity:
        counts = by_family.get(name, {})
        running = counts.get(PENDING_STATUS, 0)
        return QueueActivity(queued=sum(counts.values()) - running, running=running)

    return Activity(jobs=family("job"), tasks=family("task"))


def _document_title(job: Job) -> str:
    """What the job is doing, in one line: an index names the collection it writes, an import and
    an embed name what they do to the document instead."""
    if job.collection is not None:
        return f"{job.collection} / {job.doc}"
    return f"{job.action} {job.doc}"


def fold_operations(jobs: list[Job]) -> list[JobRow]:
    """The page as operations: an import or an index of one document, with the embed job it
    spawned folded in as its embed stage rather than listed as a job of its own.

    The child is found by id: `workflows._ensure_embedding` names it `emb:{doc}:{tail}` with the
    tail of its parent's id. An embed whose parent is not on this page (a page boundary fell
    between them, or the parent is gone) stays a row of its own, because hiding it would lose it.
    """
    embeds = {job.id: job for job in jobs if job.action == "embed"}
    folded = {_embed_id(job) for job in jobs if job.action != "embed"}
    out: list[JobRow] = []
    for job in jobs:
        if job.action == "embed" and job.id in folded:
            continue
        if job.action == "embed":
            out.append(_document_row(job, [_stage_job("embed", job)]))
            continue
        embed = embeds.get(_embed_id(job))
        own = _stage_job("convert" if job.action == "import" else "index", job, embed)
        embed_stage = [] if embed is None else [_stage_job("embed", embed)]
        stages = [own, *embed_stage] if job.action == "import" else [*embed_stage, own]
        out.append(_document_row(job, stages))
    return out


def _embed_id(job: Job) -> str:
    return f"{workflows.EMBED_PREFIX}:{job.doc}:{job.id.rsplit(':', 1)[-1]}"


def _stage_job(stage: Stage, job: Job, embed: Job | None = None) -> StageJob:
    """One stage, read from the workflow that runs it. The embed child sets the stages around it
    straight: an import converts before it spawns the embed, so a convert stage with an embed
    beside it is over; an index writes after the embed, so an index stage waits while the embed is
    not done and runs once it is. Without the child, the workflow's own status stands."""
    status = job.status
    if embed is not None and stage == "convert":
        status = "SUCCESS"
    if embed is not None and stage == "index" and status in ACTIVE_STATUS:
        status = "PENDING" if embed.status == "SUCCESS" else "ENQUEUED"
    return StageJob(
        stage=stage,
        job_id=job.id,
        status=status,
        tasks_done=job.tasks_done,
        tasks_running=job.tasks_running,
        tasks_total=job.tasks_total,
        seconds=_stage_seconds(stage, job, embed) if _is_over(status) else None,
    )


def _is_over(status: WorkflowStatus) -> bool:
    """Whether a stage has stopped running. DELAYED is a debounce waiting, not work in flight, but
    no stage is ever debounced, so "not active" is the whole of it here."""
    return status not in ACTIVE_STATUS


def _stage_seconds(stage: Stage, job: Job, embed: Job | None) -> float:
    """How long one finished stage ran, from the workflow timestamps alone. The owning workflow
    spans more than its own stage: an import converts and then waits for the embed it spawned, and
    an index waits for the embed before it writes. The child's timestamps split the two."""
    if embed is None:
        return max(0.0, job.updated_at - job.created_at)
    if stage == "convert":
        return max(0.0, embed.created_at - job.created_at)
    if stage == "index":
        return max(0.0, job.updated_at - embed.updated_at)
    return max(0.0, embed.updated_at - embed.created_at)


def _document_row(job: Job, stages: list[StageJob]) -> JobRow:
    return JobRow(
        id=job.id,
        kind="document",
        title=_document_title(job),
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error=job.error,
        detail={
            "tasks_done": sum(s.tasks_done for s in stages),
            "tasks_running": sum(s.tasks_running for s in stages),
            "tasks_total": sum(s.tasks_total for s in stages),
        },
        stages=stages,
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
    if kind == "collection":
        bulk = _bulk_kind(status.name)  # the listing selects exactly these three names
        # the second segment is a collection for the two bulk jobs, a document for a delete
        return f"{BULK_TITLES[bulk]} {_named(status.workflow_id)}" if bulk else "collection job"
    if kind == "download":
        download_kind, model = models.model_names(status.workflow_id)
        return f"download {download_kind} {model}"
    if status.name == DAILY_MAINTENANCE_WORKFLOW:  # the rest are maintenance runs
        return "daily housekeeping"
    return f"maintain {_named(status.workflow_id)}"


async def _detail(kind: JobKind, status) -> dict[str, int | str | bool | None]:
    """The numbers only this kind has. A bulk index publishes its progress as a DBOS event, which
    is read without waiting: a job that has not finished its first page yet simply has none.

    Async because that read is one, even with no wait: the event lives in the system database."""
    if kind == "download":
        return {"warm": models.is_warm(status.workflow_id)}
    if kind == "collection":
        progress = await DBOS.get_event_async(
            status.workflow_id, workflows.PROGRESS_EVENT, timeout_seconds=0
        )
        if isinstance(progress, workflows.BulkProgress):
            return {"done": progress.done, "skipped": progress.skipped, "total": progress.total}
    return {}


def _job(status, children: list, done_by_child: dict[str, int]) -> Job:
    # a listing selects the three pipeline workflows by name, so every id parses; an id of an
    # older shape is listed as an import of an unknown document rather than failing the page
    action, collection, doc = job_names(status.workflow_id) or ("import", None, "?")
    totals = [len(found[1]) if (found := workflows.stage_input(c)) else 0 for c in children]
    done = [done_by_child.get(c.workflow_id, 0) for c in children]
    return Job(
        id=status.workflow_id,
        action=action,
        collection=collection,
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


def _decode_cursor(cursor: str | None, source: str) -> int:
    """The offset a cursor continues at inside the listing that issued it; 0 without one.

    A cursor from another listing would page a different history, so one that names a different
    source is rejected rather than followed."""
    if cursor is None:
        return 0
    name, offset = CURSOR.decode(cursor)
    if name != source:
        raise InvalidInput("invalid cursor")
    return offset


def _cursor(source: str, offset: int) -> str:
    return CURSOR.encode(source, offset)


async def list_jobs(
    collection: str | None = None, page_size: int = DEFAULT_PAGE_SIZE, cursor: str | None = None
) -> Page[Job]:
    """One page of pipeline jobs, newest first, with the finished batches of the whole page
    counted in one grouped query.

    The collection filter is the workflow id's prefix, so the database cuts the window after it
    has filtered: a busy collection can no longer push a quiet one out of the page. It keeps
    collection index jobs only - an import and an embed belong to no collection.

    A job that starts while the page is walked shifts the offsets behind it, so a job can repeat
    or be skipped across a page boundary - the same trade an offset cursor over a live history
    always makes."""
    check_page_size(page_size)
    offset = _decode_cursor(cursor, DOCUMENT_KIND)
    # one row more than the page: its presence is what tells us another page exists
    statuses = await DBOS.list_workflows_async(
        name=PIPELINE_WORKFLOWS,
        workflow_id_prefix=(
            f"{workflows.COLLECTION_DOCUMENT_PREFIX}:{collection}:" if collection else None
        ),
        sort_desc=True,
        limit=page_size + 1,
        offset=offset,
        load_input=False,
    )
    page = statuses[:page_size]
    children = await stage_children([s.workflow_id for s in page])
    done = await sysdb.step_counts(
        [c.workflow_id for group in children.values() for c in group], STAGE_STEP
    )
    return Page(
        items=[_job(s, children[s.workflow_id], done) for s in page],
        next_cursor=(
            _cursor(DOCUMENT_KIND, offset + page_size) if len(statuses) > page_size else None
        ),
    )


async def stage_children(ids: list[str]) -> dict[str, list]:
    """The stage children of each listed pipeline workflow, keyed by parent; a parent with none
    maps to an empty list. One DBOS read for the whole page."""
    children: dict[str, list] = {i: [] for i in ids}
    if ids:
        found = await DBOS.list_workflows_async(
            parent_workflow_id=ids, name=STAGE_WORKFLOW, load_output=False
        )
        for c in found:
            children.setdefault(c.parent_workflow_id or "", []).append(c)
    return children


class ChunksAt(msgspec.Struct):
    """One finished index of one document into one collection, as a point on the Insights chart:
    when it completed, where, and how many chunks it wrote."""

    ts: float  # unix seconds
    collection: str
    chunks: int


async def chunks_since(cutoff: float) -> list[ChunksAt]:
    """Every successful index that completed on or after `cutoff`, oldest first.

    DBOS's own retention round bounds this history, so the chart reaches back as far as the
    workflow rows do and no further."""
    since_ms = int(cutoff * 1000)
    live = {
        s.workflow_id: s
        for s in await DBOS.list_workflows_async(
            name=COLLECTION_DOCUMENT_WORKFLOW, status="SUCCESS", load_input=False
        )
        if s.completed_at is not None and s.completed_at >= since_ms
    }
    # only the index stage writes chunks, and its batches record how many: read those step logs
    # alone, side by side
    indexes = [
        (parent, child)
        for parent, group in (await stage_children(list(live))).items()
        for child in group
        if (found := workflows.stage_input(child)) and found[0] == "index"
    ]
    written = await asyncio.gather(*(_stage_tasks(child) for _, child in indexes))
    chunks: dict[str, int] = dict.fromkeys(live, 0)
    for (parent, _), tasks in zip(indexes, written, strict=True):
        chunks[parent] += sum(t.result or 0 for t in tasks if t.status == "SUCCESS")
    points = [
        ChunksAt(
            (status.completed_at or 0) / 1000,
            (job_names(job_id) or ("index", None, "?"))[1] or "?",
            chunks[job_id],
        )
        for job_id, status in live.items()
    ]
    return sorted(points, key=lambda point: point.ts)


def _ordered(tasks: list[Task]) -> list[Task]:
    """Stage by stage, batch by batch: the order the pipeline planned them in. Slices of one
    stage run side by side, so this is a plan order, not a finishing order."""
    return sorted(tasks, key=lambda t: (STAGE_ORDER.index(t.stage), t.seq))


async def list_tasks(job_id: str) -> list[Task]:
    """The micro-batches of one job. Gone once retention purged the job: the batches live in the
    step log of its children, which DBOS drops with them."""
    if await DBOS.get_workflow_status_async(job_id) is None:
        raise NotFound(f"job not found: {job_id}")
    children = await DBOS.list_workflows_async(parent_workflow_id=job_id, name=STAGE_WORKFLOW)
    return _ordered([task for child in children for task in await _stage_tasks(child)])


def _task(child_id: str, stage: Stage, batch: Batch, status: WorkflowStatus, output, error) -> Task:
    return Task(
        id=f"{child_id}:{batch.seq}",
        child_id=child_id,
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
        if st["function_name"] == STAGE_STEP
    ]
    out: list[Task] = []
    for i, batch in enumerate(batches):
        if i < len(steps):
            output, error = _step_outcome(steps[i])
            status = "ERROR" if error else "SUCCESS"
        else:
            output = error = None
            status = "PENDING" if child.status == "PENDING" and i == len(steps) else "ENQUEUED"
        out.append(_task(child.workflow_id, stage, batch, status, output, error))
    return out


def _step_outcome(step) -> tuple[int | None, str | None]:
    """A step that failed permanently returns its message instead of raising (see BatchResult)."""
    output, error = step["output"], step["error"]
    if isinstance(output, BatchResult):
        return output.value, output.permanent_error
    return (output if isinstance(output, int) else None), (str(error) if error else None)


async def bulk_job(job_id: str) -> BulkJob:
    """The state of one whole-collection or whole-document job: its DBOS status plus the progress
    event a bulk index publishes after every page. `progress` stays None for a delete, which has
    no pages, and for an index that has not finished its first page yet."""
    status = await DBOS.get_workflow_status_async(job_id)
    if status is None:
        raise NotFound(f"job not found: {job_id}")
    kind = _bulk_kind(status.name)
    if kind is None:  # a job of another kind: not a bulk job
        raise NotFound(f"job not found: {job_id}")
    progress = await DBOS.get_event_async(job_id, workflows.PROGRESS_EVENT, timeout_seconds=0)
    return BulkJob(
        id=job_id,
        kind=kind,
        collection=_collection_of(job_id),
        status=status.status,
        progress=progress if isinstance(progress, workflows.BulkProgress) else None,
        error=str(status.error) if status.error else None,
    )
