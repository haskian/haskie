"""Read model over the operation history, in the three words this app counts work in.

An **operation** is the whole of what someone asked for: import a document, index one document
into a collection, index, delete or describe a whole collection, delete or describe a document,
maintain a collection, download a model. A **job** is one stage of an operation: convert, embed,
describe or index. A **task** is one micro-batch: one durable step below a job.

"Workflow" is DBOS's word for the thing that runs any of them, and it stays in the modules that
talk to DBOS (`workflows`, `dbos_names`, `sysdb`, `models`). Nothing this module returns says it.

Nothing here is stored; every field comes from DBOS's workflow tables, which the nightly retention
round bounds.

Three workflows carry a document through the pipeline (`dbos_names.PIPELINE_WORKFLOWS`), and each
runs a job of an operation: `import_document` converts it, `ensure_embedding` fills its embedding
cache, `index_collection_document` writes one collection's table from that cache. Every one of them
runs its stage as `stage_slice` children with a durable step per micro-batch, so a job's tasks are
those micro-batches. The batch list comes from each child's input, the finished count from its step
log (`sysdb`, one grouped query for the whole page).
Counts are summed over the children, so how a stage was sliced never shows here.

Every other kind of operation this app runs is listed through one generic read model instead:
`list_operations` returns a page of `Operation` for a whole-collection operation, a model download
or a maintenance run, so the Operations view has a section per kind. Documents are a kind there
too, folded out of the pipeline listing below, because they alone also have jobs, tasks and a
cancel.

Reads only: cancelling an operation writes, so it lives in `workflows` beside the ids it writes by.

The action, the collection and the document are read out of the id (`imp:{doc}:{uuid}`,
`emb:{doc}:{uuid}`, `idx-col:{collection}:{doc}:{uuid}`, through `workflows.pipeline_names`), never
out of the recorded input: that makes the collection filter an id prefix the database can apply,
and a page of operations costs no input payloads at all. Every other kind whose work belongs to one
collection carries the name in the same place (`{prefix}:{collection}:{rest}`), and a download reads
its kind and model out of its id (`dl:{kind}:{model}`), for the same reason.

Every listing is a read of DBOS's own tables through its `*_async` API, so every one of them is
awaited. The row builders below take what those reads returned and touch nothing: they stay sync.
"""

import asyncio
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum

import msgspec
from dbos import DBOS, WorkflowStatus

from haskie import sysdb
from haskie.document import document as documents
from haskie.errors import InvalidInput, NotFound
from haskie.indexing import models, workflows
from haskie.indexing.dbos_names import (
    ACTIVE_STATUS,
    BULK_WORKFLOWS,
    COLLECTION_DOCUMENT_WORKFLOW,
    COLLECTION_STEP,
    DAILY_MAINTENANCE_WORKFLOW,
    DELETE_DOCUMENT_WORKFLOW,
    DOCUMENT_OPERATION_WORKFLOWS,
    DOWNLOAD_WORKFLOW,
    EMBED_WORKFLOW,
    MAINTAIN_PARTITION_WORKFLOW,
    MEMBER_STEP,
    MEMBERS_STEP,
    STAGE_STEP,
    STAGE_WORKFLOW,
    SUMMARIZE_COLLECTION_WORKFLOW,
    BulkWorkflow,
    RunStatus,
)
from haskie.indexing.pipeline import Batch
from haskie.indexing.workflows import (
    STAGE_ORDER,
    BatchResult,
    PipelineAction,
    Stage,
    pipeline_names,
)
from haskie.paging import DEFAULT_PAGE_SIZE, OffsetCursor, Order, Page, check_page_size
from haskie.search import session

# Operations are always newest first, so the cursor carries no sort of its own; it names the
# listing the next page continues in and the offset into it, which is all an ordered-by-created_at
# listing of a run history can page on.
SORT = "created_at"
ORDER = Order.DESC
# Both listings page on an offset: DBOS's history has one fixed order and no sort of its own.
CURSOR = OffsetCursor(SORT, ORDER)


class _StageCounts(msgspec.Struct):
    """The micro-batches of one stage of one run, summed over its slices, and when its first
    slice was created."""

    done: int = 0
    running: int = 0
    total: int = 0
    started_at: float = 0.0  # unix seconds


class _StageRun(msgspec.Struct):
    """One run of one pipeline over one document, as DBOS's history holds it: the raw row
    `fold_operations` folds into an operation and its jobs. Internal on purpose: nothing outside
    this module sees it, and no route returns it.

    `collection` is None for an import and an embed: both are collection-independent, and only the
    index of a member belongs to a collection. `stages` holds the batches of each stage its slices
    ran: an embedding run has embed slices, then a describe slice."""

    id: str
    action: PipelineAction
    collection: str | None
    document_id: str  # as the workflow id carries it
    document: str  # its name (`_document_names`)
    status: RunStatus
    created_at: float
    updated_at: float
    error: str | None
    stages: dict[Stage, _StageCounts] = {}


# A whole-collection or whole-document operation is reported under the name DBOS records it as, so
# the kind the API reports is the workflow name (`test_workflows` pins that against the registry).
BulkKind = BulkWorkflow
BULK_KINDS: tuple[BulkKind, ...] = BULK_WORKFLOWS
# The bulk kinds whose id carries a document's id where the rest carry a collection's name. What a
# bulk operation works on is its title; the verb is the row's tag in the UI, so the title leaves
# it out.
DOCUMENT_BULK_KINDS = frozenset({BulkKind.DELETE_DOCUMENT, BulkKind.SUMMARIZE_DOCUMENT})


class OperationProgress(msgspec.Struct):
    """How far one whole-collection or whole-document operation got: what the 202 of an "index
    all", a collection delete, a document delete or a description points at.

    `collection` is None for a document's operation: it belongs to no one collection."""

    id: str
    kind: BulkKind
    collection: str | None
    status: RunStatus
    progress: workflows.BulkProgress | None = None
    error: str | None = None


# Declared in the order the Operations view shows the sections, so `KIND_ORDER` is the type itself.
# `DOCUMENT` is the one kind with a listing of its own, and its cursor.
class OperationKind(StrEnum):
    DOCUMENT = "document"
    COLLECTION = "collection"
    DOWNLOAD = "download"
    MAINTENANCE = "maintenance"


KIND_ORDER: tuple[OperationKind, ...] = tuple(OperationKind)
KIND_LABELS: dict[OperationKind, str] = {
    OperationKind.DOCUMENT: "Documents",
    OperationKind.COLLECTION: "Collections",
    OperationKind.DOWNLOAD: "Model downloads",
    OperationKind.MAINTENANCE: "Maintenance",
}


class Job(msgspec.Struct):
    """One stage of an operation, and the run whose batches it is made of: the convert and index
    stages run in the operation's own run, the embed and describe stages in the
    `ensure_embedding` child it spawns (see `fold_operations`). The two jobs of that child share
    its id, and its tasks say which stage each belongs to."""

    id: str  # pass it to `list_tasks` for this job's batches
    stage: Stage
    status: RunStatus
    created_at: float
    updated_at: float
    error: str | None
    tasks_done: int = 0
    tasks_running: int = 0
    tasks_total: int = 0
    seconds: float | None = None  # how long the job ran; None while it still does


class Operation(msgspec.Struct):
    """One operation of any kind, in the shape the Operations view lists: what every kind has in
    common, plus the numbers only that kind has in `detail` (tasks for a document, pages for a bulk
    index, warm for a download). Kept flat and untyped on purpose: it is a read model for a table.

    A document operation (an import, or an index of one document) lists the `jobs` it is made of,
    each with the tasks it ran; so does a collection's description, whose one job describes its
    documents one task each (`description_tasks`)."""

    id: str
    kind: OperationKind
    title: str  # what it works on: "collection / document", "collection X", "reranker Y"
    status: RunStatus
    created_at: float
    updated_at: float
    error: str | None
    origin: str | None = None  # the session whose action started it; None for the web UI
    detail: dict[str, int | str | bool | None] = {}
    jobs: list[Job] = []  # documents and a collection's description only, in pipeline order


class OperationKindSummary(msgspec.Struct):
    """One section of the Operations view: what to call it, and how much of it is running now."""

    kind: OperationKind
    label: str
    active: int


class Task(msgspec.Struct):
    id: str
    child_id: str  # the run that did it (a `stage_slice`); the task id is `{it}:{seq}`
    stage: Stage
    seq: int
    # convert: PDF pages [start, end); embed: the one part; describe: sections [start, end);
    # index: the part range written together. 0 for a task that `name` says instead
    page_start: int
    page_end: int
    status: RunStatus
    result: int | None  # convert: pages needing OCR; describe: sections; embed/index: chunks
    error: str | None
    # what a task of a collection's description covers: a document's name, or the collection
    name: str | None = None


def _second_segment(operation_id: str) -> str | None:
    """The second segment of an id (`{prefix}:{key}:{rest}`): the one thing the operation is
    about, a collection's name or a document's id. None for any other id."""
    parts = operation_id.split(":", 2)
    return parts[1] if len(parts) == 3 else None


def _collection_of(kind: BulkKind, operation_id: str) -> str | None:
    """The collection an operation belongs to; None when it belongs to none.

    A document's operation is the exception: `del-doc:{doc}:{uuid}` and `sum-doc:{doc}:{uuid}`
    carry a document where every other id carries a collection."""
    if kind in DOCUMENT_BULK_KINDS:
        return None
    return _second_segment(operation_id)


# --- one listing per kind of operation -----------------------------------------------------

# Kind -> the DBOS workflow names it lists. `document` is missing on purpose: it has a listing of
# its own (`_pipeline_page`), because it is the only kind whose rows carry micro-batch counts.
KIND_NAMES: dict[OperationKind, list[str]] = {
    OperationKind.COLLECTION: list(BULK_KINDS),
    OperationKind.DOWNLOAD: [DOWNLOAD_WORKFLOW],
    # `maintain_collection` is only the debounced handle that waits: the run itself is the child
    # on the collection's partition, so that is the one worth a row.
    OperationKind.MAINTENANCE: [MAINTAIN_PARTITION_WORKFLOW, DAILY_MAINTENANCE_WORKFLOW],
}

# The same table read the other way, for counting active runs by kind in one query.
KIND_BY_NAME: dict[str, OperationKind] = {
    # an embed run is a job of the import or index that asked for it, not an operation
    **dict.fromkeys(DOCUMENT_OPERATION_WORKFLOWS, OperationKind.DOCUMENT),
    **{name: kind for kind, names in KIND_NAMES.items() for name in names},
}

# Kind -> the id prefix that keeps one collection's operations only. Document deletes belong to no
# collection, so a collection filter drops them from the collection section. Downloads belong to no
# collection either, and have no prefix here: the filter leaves their section as it is.
_COLLECTION_PREFIX: dict[OperationKind, list[str]] = {
    OperationKind.COLLECTION: [
        f"{workflows.BULK_INDEX_PREFIX}:",
        f"{workflows.BULK_DELETE_PREFIX}:",
        f"{workflows.SUMMARIZE_COLLECTION_PREFIX}:",
    ],
    OperationKind.MAINTENANCE: [f"{workflows.MAINTAIN_PREFIX}:"],
}


def _checked_kind(kind: str) -> OperationKind:
    """A kind is a trust boundary: an unknown one is a mistake in the request, not an empty page."""
    if kind not in KIND_ORDER:
        raise InvalidInput(f"unknown operation kind {kind!r}; allowed: {', '.join(KIND_ORDER)}")
    return OperationKind(kind)


def _bulk_kind(name: str | None) -> BulkKind | None:
    """The kind a whole-collection workflow name stands for; None for any other workflow."""
    return BulkKind(name) if name in BULK_KINDS else None


def _kind_prefix(kind: OperationKind, collection: str | None) -> list[str] | None:
    if collection is None:
        return None
    return [f"{prefix}{collection}:" for prefix in _COLLECTION_PREFIX.get(kind, [])] or None


async def _kind_statuses(
    kind: OperationKind, collection: str | None, limit: int, offset: int
) -> list:
    """One window of the DBOS history for one kind, newest first. Inputs stay on disk; the output
    is loaded only because DBOS carries a run's error alongside it."""
    return await DBOS.list_workflows_async(
        name=KIND_NAMES[kind],
        workflow_id_prefix=_kind_prefix(kind, collection),
        sort_desc=True,
        limit=limit,
        offset=offset,
        load_input=False,
        load_output=True,
    )


async def list_operations(
    kind: str,
    collection: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> Page[Operation]:
    """One page of the operations of one kind, newest first.

    Documents keep a listing of their own, with the batch counts only they carry, and are folded
    into the common shape here; the rest are one window of the DBOS history each, paged on an
    opaque offset cursor bound to the kind that issued it."""
    check_page_size(page_size)
    checked = _checked_kind(kind)
    if checked == OperationKind.DOCUMENT:
        page = await _pipeline_page(collection, page_size, cursor)
        return Page(
            items=await _with_origins(fold_operations(page.items)),
            next_cursor=page.next_cursor,
        )
    offset = _decode_cursor(cursor, checked)
    # one row more than the page: its presence is what tells us another page exists
    found = await _kind_statuses(checked, collection, page_size + 1, offset)
    # every row costs a read of its own (see `_detail`), so the page is built in one round trip
    page = found[:page_size]
    # a document's operation is titled by the document's name: one lookup for the whole page
    names = await _document_names(
        doc
        for s in page
        if _bulk_kind(s.name) in DOCUMENT_BULK_KINDS
        and (doc := _second_segment(s.workflow_id)) is not None
    )
    rows = await asyncio.gather(*(_kind_row(checked, s, names) for s in page))
    return Page(
        items=await _with_origins(list(rows)),
        next_cursor=_cursor(checked, offset + page_size) if len(found) > page_size else None,
    )


async def _with_origins(rows: list[Operation]) -> list[Operation]:
    """Name the session that started each row, where one did: one query for the page."""
    origins = await session.origins([row.id for row in rows])
    for row in rows:
        row.origin = origins.get(row.id)
    return rows


async def list_kinds() -> list[OperationKindSummary]:
    """Every kind, in the order the Operations view shows them, with how many of each are enqueued
    or running right now. One grouped query counts all of them, not one per section."""
    active = await sysdb.active_counts_by_name()
    counts: dict[OperationKind, int] = dict.fromkeys(KIND_ORDER, 0)
    for name, count in active.items():
        kind = KIND_BY_NAME.get(name)
        if kind is not None:
            counts[kind] += count
    return [
        OperationKindSummary(kind=kind, label=KIND_LABELS[kind], active=counts[kind])
        for kind in KIND_ORDER
    ]


class QueueActivity(msgspec.Struct):
    """One kind of work right now: `running` is under way, `queued` waits its turn.

    A debounced run waiting out its period (DELAYED) is neither: it is not work anyone is waiting
    on, so it stays out of the indicator (see `sysdb.operation_activity`)."""

    queued: int
    running: int


class Activity(msgspec.Struct):
    """The indicator every view shows: operations (`operation.*` queues), and the micro-batches
    they are made of, running and left to run."""

    operations: QueueActivity
    tasks: QueueActivity


async def activity() -> Activity:
    # The embed job an import or an index spawned is waited for by the one that spawned it (see
    # `fold_operations`): counting both would read "2 operations" for one operation.
    # every slice on a `task.*` queue; its batches are in its input, the finished ones in its steps
    counts, slices = await asyncio.gather(
        sysdb.operation_activity(skip=[workflows.EMBEDDING_QUEUE]),
        DBOS.list_workflows_async(
            name=STAGE_WORKFLOW, status=list(ACTIVE_STATUS), load_output=False
        ),
    )
    running = counts.get(RunStatus.PENDING, 0)
    operations = QueueActivity(queued=sum(counts.values()) - running, running=running)
    done = await sysdb.step_counts([one.workflow_id for one in slices], STAGE_STEP)
    return Activity(operations=operations, tasks=batch_activity(slices, done))


def _batch_count(child: WorkflowStatus) -> int:
    """How many micro-batches a stage slice was given; 0 when its input cannot be read back."""
    found = workflows.stage_input(child)
    return len(found[1]) if found else 0


def _runs_a_batch(child: WorkflowStatus, done: int, total: int) -> bool:
    """Whether a slice is working on a batch now: it runs its batches one after another, so a
    running slice with any left works on exactly one."""
    return child.status == RunStatus.PENDING and done < total


def batch_activity(slices: list[WorkflowStatus], done: dict[str, int]) -> QueueActivity:
    """The micro-batches of the active stage slices: one running in each running slice, and every
    other batch not yet done waiting, in a running slice or in one still waiting for a slot."""
    running = left = 0
    for one in slices:
        total = _batch_count(one)
        finished = done.get(one.workflow_id, 0)
        running += _runs_a_batch(one, finished, total)
        left += max(0, total - finished)
    return QueueActivity(queued=left - running, running=running)


def _document_title(run: _StageRun) -> str:
    """What the operation works on, in one line: an index names the collection it writes and the
    document, an import or an embed the document alone. The verb is the row's tag."""
    if run.collection is not None:
        return f"{run.collection} / {run.document}"
    return run.document


# The stage a run's own job is before any of its slices exists: an import converts, an embedding
# run embeds, an index writes.
_FIRST_STAGE: dict[PipelineAction, Stage] = {
    PipelineAction.IMPORT: Stage.CONVERT,
    PipelineAction.EMBED: Stage.EMBED,
    PipelineAction.INDEX: Stage.INDEX,
}


def fold_operations(runs: list[_StageRun]) -> list[Operation]:
    """The page as operations: an import or an index of one document, with the embedding run it
    spawned folded in as that run's jobs rather than listed as an operation of its own.

    The child is found by id: `workflows._ensure_embedding` names it `emb:{doc}:{tail}` with the
    tail of its parent's id. `_pipeline_page` reads every embed together with its parent. An
    embed given without its parent stays an operation of its own, because hiding it would lose it.
    """
    embeds = {run.id: run for run in runs if run.action == PipelineAction.EMBED}
    folded = {
        workflows.embed_id(run.id, run.document_id)
        for run in runs
        if run.action != PipelineAction.EMBED
    }
    out: list[Operation] = []
    for run in runs:
        if run.action == PipelineAction.EMBED:
            if run.id not in folded:
                out.append(_document_row(run, _jobs(run)))
            continue
        embed = embeds.get(workflows.embed_id(run.id, run.document_id))
        if embed is None:
            out.append(_document_row(run, _jobs(run)))
        elif run.action == PipelineAction.IMPORT:
            # an import converts, then spawns its embedding run and waits: its jobs end there
            converted = _jobs(run, end=embed.created_at, status=RunStatus.SUCCESS)
            out.append(_document_row(run, [*converted, *_jobs(embed)]))
        else:
            # an index waits for its embedding run, then writes: its jobs start where that ends
            status = run.status
            if status in ACTIVE_STATUS:
                status = (
                    RunStatus.PENDING if embed.status == RunStatus.SUCCESS else RunStatus.ENQUEUED
                )
            own = _jobs(run, start=embed.updated_at, status=status)
            out.append(_document_row(run, [*_jobs(embed), *own]))
    return out


def _jobs(
    run: _StageRun,
    end: float | None = None,
    start: float | None = None,
    status: RunStatus | None = None,
) -> list[Job]:
    """One job per stage the run ran, in pipeline order, its first stage even before a slice of
    it exists. A run's stages run one after another, so a stage is over once the next one began,
    which ends its clock, and the last one stands as the run does. `start`, `end` and `status`
    stand for the run's own where another run sets them: an import is done converting once it
    spawned its embedding run, and an index writes only once that run has ended."""
    stages = sorted(set(run.stages) | {_FIRST_STAGE[run.action]}, key=STAGE_ORDER.index)
    # every stage after the first has a slice, so a moment it began
    starts = [
        run.created_at if start is None else start,
        *(run.stages[stage].started_at for stage in stages[1:]),
    ]
    ends = [*starts[1:], run.updated_at if end is None else end]
    last = len(stages) - 1
    return [
        _job(
            stage,
            run,
            RunStatus.SUCCESS if index < last else (status or run.status),
            starts[index],
            ends[index],
        )
        for index, stage in enumerate(stages)
    ]


def _job(stage: Stage, run: _StageRun, status: RunStatus, start: float, end: float) -> Job:
    """One job, read from the run that carries it: its status and the span it ran over, as the
    caller worked them out, and the batches of its stage."""
    counts = run.stages.get(stage, _StageCounts())
    return Job(
        id=run.id,
        stage=stage,
        status=status,
        created_at=run.created_at,
        updated_at=run.updated_at,
        # a job this listing declared successful never failed, whatever the run around it did
        error=None if status == RunStatus.SUCCESS else run.error,
        tasks_done=counts.done,
        tasks_running=counts.running,
        tasks_total=counts.total,
        # a clock that went backwards reads as zero, never negative
        seconds=max(0.0, end - start) if _is_over(status) else None,
    )


def _is_over(status: RunStatus) -> bool:
    """Whether a job has stopped running. DELAYED is a debounce waiting, not work in flight, but
    no job is ever debounced, so "not active" is the whole of it here."""
    return status not in ACTIVE_STATUS


def _document_row(run: _StageRun, jobs: list[Job]) -> Operation:
    return Operation(
        id=run.id,
        kind=OperationKind.DOCUMENT,
        title=_document_title(run),
        status=run.status,
        created_at=run.created_at,
        updated_at=run.updated_at,
        error=run.error,
        detail={
            "tasks_done": sum(j.tasks_done for j in jobs),
            "tasks_running": sum(j.tasks_running for j in jobs),
            "tasks_total": sum(j.tasks_total for j in jobs),
        },
        jobs=jobs,
    )


async def _kind_row(kind: OperationKind, status, names: dict[str, str]) -> Operation:
    row = Operation(
        id=status.workflow_id,
        kind=kind,
        title=_title(kind, status, names),
        status=status.status,
        created_at=(status.created_at or 0) / 1000,
        updated_at=(status.updated_at or 0) / 1000,
        error=str(status.error) if status.error else None,
        detail=await _detail(kind, status),
    )
    if _bulk_kind(status.name) == BulkKind.SUMMARIZE_COLLECTION:
        row.jobs = [_description_job(row)]
    return row


def _description_job(row: Operation) -> Job:
    """The one job of a collection's description: the run itself, whose tasks are its steps."""
    status = RunStatus(row.status)
    return Job(
        id=row.id,
        stage=Stage.DESCRIBE,
        status=status,
        created_at=row.created_at,
        updated_at=row.updated_at,
        error=row.error,
        tasks_done=_number(row.detail.get("done")),
        tasks_total=_number(row.detail.get("total")),
        seconds=max(0.0, row.updated_at - row.created_at) if _is_over(status) else None,
    )


def _number(value: int | str | bool | None) -> int:
    return value if isinstance(value, int) else 0


def _title(kind: OperationKind, status, names: dict[str, str]) -> str:
    """Human text for one row: what this operation works on, read out of its id and the name DBOS
    recorded it under. The verb is the row's tag, so the title leaves it out.

    `document` never arrives here: it has a row builder of its own (see `list_operations`)."""
    if kind == OperationKind.COLLECTION:
        bulk = _bulk_kind(status.name)  # the listing selects exactly these names
        if bulk is None:
            return "collection operation"
        # the second segment is a collection's name, or a document's id for a document's operation
        key = _second_segment(status.workflow_id) or "?"
        if bulk in DOCUMENT_BULK_KINDS:
            return f"document {names.get(key, key)}"
        return f"collection {key}"
    if kind == OperationKind.DOWNLOAD:
        download_kind, model = models.model_names(status.workflow_id)
        return f"{download_kind} {model}"
    if status.name == DAILY_MAINTENANCE_WORKFLOW:  # the rest are maintenance runs
        return "daily housekeeping"
    return _second_segment(status.workflow_id) or "?"


async def _detail(kind: OperationKind, status) -> dict[str, int | str | bool | None]:
    """The numbers only this kind has. A collection operation names which of them it is
    (`bulk`), since the kind alone does not. A bulk index also publishes its progress as a DBOS
    event, which is read without waiting: an operation that has not finished its first page yet
    has none.

    Async because that read is one, even with no wait: the event lives in the system database."""
    if kind == OperationKind.DOWNLOAD:
        return {"warm": models.is_warm(status.workflow_id)}
    if kind != OperationKind.COLLECTION:
        return {}
    bulk = _bulk_kind(status.name)
    if bulk == BulkKind.SUMMARIZE_COLLECTION:
        tasks = await description_tasks(status)
        done = sum(task.status == RunStatus.SUCCESS for task in tasks)
        return {"bulk": bulk, "done": done, "total": len(tasks)}
    if bulk != BulkKind.INDEX_COLLECTION:
        return {"bulk": bulk}
    progress = await DBOS.get_event_async(
        status.workflow_id, workflows.PROGRESS_EVENT, timeout_seconds=0
    )
    if isinstance(progress, workflows.BulkProgress):
        return {"bulk": bulk, "done": progress.done, "total": progress.total}
    return {"bulk": bulk}


async def _document_names(ids: Iterable[str]) -> dict[str, str]:
    """What people call each of these documents, by id. One deleted since has no row, so its
    name is the one its delete was given (`delete_document_workflow(doc, name)`). One with no
    delete on record is absent: a caller shows its id."""
    wanted = set(ids)
    names = await documents.names_of(wanted)
    gone = wanted - names.keys()
    if not gone:
        return names
    deletes = await DBOS.list_workflows_async(
        name=DELETE_DOCUMENT_WORKFLOW,
        workflow_id_prefix=[f"{workflows.DELETE_DOCUMENT_PREFIX}:{doc}:" for doc in sorted(gone)],
        load_output=False,
        sort_desc=True,  # a document deleted, imported again and deleted again: its latest name
    )
    for run in deletes:  # latest first, so the first name of each document is kept
        if run.input:
            names.setdefault(_second_segment(run.workflow_id) or "?", run.input["args"][1])
    return names


def _pipeline_of(workflow_id: str) -> tuple[PipelineAction, str | None, str]:
    """What a pipeline run does, to which collection and document. A listing selects the three
    pipeline workflows by name, so every id parses; an id of an older shape reads as an import
    of an unknown document rather than failing the page."""
    return pipeline_names(workflow_id) or (PipelineAction.IMPORT, None, "?")


def _stage_of(parent: str, child: str) -> Stage | None:
    """The stage a slice ran, read out of its id, `{parent}:{stage}[:{slice}]`, so no input is
    unpickled; None for an id of another shape."""
    found = child.removeprefix(f"{parent}:").split(":")[0]
    return Stage(found) if found in STAGE_ORDER else None


def _stage_run(
    status, children: list, done_by_child: dict[str, int], names: dict[str, str]
) -> _StageRun:
    action, collection, doc = _pipeline_of(status.workflow_id)
    stages: dict[Stage, _StageCounts] = {}
    for child in children:
        stage = _stage_of(status.workflow_id, child.workflow_id)
        if stage is None:
            continue
        created = (child.created_at or 0) / 1000
        counts = stages.setdefault(stage, _StageCounts(started_at=created))
        total = _batch_count(child)
        done = done_by_child.get(child.workflow_id, 0)
        counts.done += done
        counts.running += _runs_a_batch(child, done, total)
        counts.total += total
        counts.started_at = min(counts.started_at, created)
    return _StageRun(
        id=status.workflow_id,
        action=action,
        collection=collection,
        document_id=doc,
        document=names.get(doc, doc),
        status=status.status,
        created_at=(status.created_at or 0) / 1000,
        updated_at=(status.updated_at or 0) / 1000,
        error=str(status.error) if status.error else None,
        stages=stages,
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


async def _pipeline_page(
    collection: str | None = None, page_size: int = DEFAULT_PAGE_SIZE, cursor: str | None = None
) -> Page[_StageRun]:
    """One page of pipeline runs, newest first, with the finished batches of the whole page
    counted in one grouped query. `fold_operations` folds these into the document operations the API
    serves; nothing else reads them.

    The page is cut over imports and indexes only, and the embedding run each asked for is read by
    its id (`workflows.embed_id`) after the cut. An embedding run is created after its parent, so a
    window over every run could hold it while its parent fell to the next page, and split one
    operation in two.

    The collection filter is the id's prefix, so the database cuts the window after it has
    filtered: a busy collection can no longer push a quiet one out of the page. It keeps collection
    index runs only; an import belongs to no collection.

    A run that starts while the page is walked shifts the offsets behind it, so a row can repeat
    or be skipped across a page boundary. An offset cursor over a live history always makes that
    trade."""
    check_page_size(page_size)
    offset = _decode_cursor(cursor, OperationKind.DOCUMENT)
    # one row more than the page: its presence is what tells us another page exists
    statuses = await DBOS.list_workflows_async(
        name=DOCUMENT_OPERATION_WORKFLOWS,
        workflow_id_prefix=(
            f"{workflows.COLLECTION_DOCUMENT_PREFIX}:{collection}:" if collection else None
        ),
        sort_desc=True,
        limit=page_size + 1,
        offset=offset,
        load_input=False,
    )
    if not statuses:
        return Page(items=[], next_cursor=None)
    page = statuses[:page_size]
    docs = [_pipeline_of(s.workflow_id)[2] for s in page]
    embed_ids = [workflows.embed_id(s.workflow_id, doc) for s, doc in zip(page, docs, strict=True)]
    embeds, children, names = await asyncio.gather(
        DBOS.list_workflows_async(workflow_ids=embed_ids, load_input=False),
        stage_children([s.workflow_id for s in page] + embed_ids),
        _document_names(docs),
    )
    page += embeds
    done = await sysdb.step_counts(
        [c.workflow_id for group in children.values() for c in group], STAGE_STEP
    )
    return Page(
        items=[_stage_run(s, children[s.workflow_id], done, names) for s in page],
        next_cursor=(
            _cursor(OperationKind.DOCUMENT, offset + page_size)
            if len(statuses) > page_size
            else None
        ),
    )


async def stage_children(ids: list[str]) -> dict[str, list]:
    """The stage children of each listed pipeline run, keyed by parent; a parent with none maps to
    an empty list. One DBOS read for the whole page."""
    children: dict[str, list] = {i: [] for i in ids}
    if ids:
        found = await DBOS.list_workflows_async(
            parent_workflow_id=ids, name=STAGE_WORKFLOW, load_output=False
        )
        for c in found:
            children.setdefault(c.parent_workflow_id or "", []).append(c)
    return children


class ChunksAt(msgspec.Struct):
    """One finished indexing of one document, as a point on the Insights chart: when it completed,
    what it indexed, and how many chunks it wrote. An import embeds the document's chunks and an
    index writes them into a collection, so a document imported and then indexed is two points."""

    ts: float  # unix seconds
    document: str
    collection: str | None  # the collection an index wrote into; None for an import
    chunks: int


async def chunks_since(cutoff: float) -> list[ChunksAt]:
    """Every successful import and index that completed on or after `cutoff`, oldest first.

    An import's chunks are its embed run's; an embed run an index started is part of that index,
    which counts the chunks it writes, so it is left out. DBOS's own retention round bounds this
    history, so the chart reaches back as far as its rows do and no further."""
    live = {
        s.workflow_id: s
        for s in await DBOS.list_workflows_async(
            name=[COLLECTION_DOCUMENT_WORKFLOW, EMBED_WORKFLOW],
            status=RunStatus.SUCCESS,
            completed_after=datetime.fromtimestamp(cutoff, UTC).isoformat(),
            load_input=False,
            load_output=False,
        )
        if _counted(s)
    }
    # a finished run's slices all finished, and each returned the chunks of every batch it ran;
    # its stage is in its id, `{parent}:{stage}[:{slice}]`, so no input is unpickled
    chunks: dict[str, int] = dict.fromkeys(live, 0)
    if live:
        for child in await DBOS.list_workflows_async(
            parent_workflow_id=list(live), name=STAGE_WORKFLOW, load_input=False
        ):
            parent = child.parent_workflow_id or ""
            stage = _stage_of(parent, child.workflow_id)
            if parent in chunks and stage in (Stage.EMBED, Stage.INDEX):
                chunks[parent] += sum(child.output or [])
    points: list[ChunksAt] = []
    ran = {id: _pipeline_of(id) for id in live}
    names = await _document_names(doc for _, _, doc in ran.values())
    for workflow_id, status in live.items():
        _, collection, doc = ran[workflow_id]
        named = names.get(doc, doc)  # a document gone with no delete on record: its id
        points.append(
            ChunksAt((status.completed_at or 0) / 1000, named, collection, chunks[workflow_id])
        )
    return sorted(points, key=lambda point: point.ts)


def _counted(run: WorkflowStatus) -> bool:
    """Whether a run is a point of its own: every index, and an embed run an import started."""
    if run.name != EMBED_WORKFLOW:
        return True
    parent = pipeline_names(run.parent_workflow_id or "")
    return parent is not None and parent[0] == PipelineAction.IMPORT


def _ordered(tasks: list[Task]) -> list[Task]:
    """Stage by stage, batch by batch: the order the pipeline planned them in. Slices of one job
    run side by side, so this is a plan order, not a finishing order."""
    return sorted(tasks, key=lambda t: (STAGE_ORDER.index(t.stage), t.seq))


async def list_tasks(job_id: str) -> list[Task]:
    """The micro-batches of one job. Gone once retention purged the job: the batches live in the
    step log of its children, which DBOS drops with them."""
    found = await DBOS.get_workflow_status_async(job_id)
    if found is None:
        raise NotFound(f"job not found: {job_id}")
    if found.name == SUMMARIZE_COLLECTION_WORKFLOW:
        return await description_tasks(found)
    children = await DBOS.list_workflows_async(parent_workflow_id=job_id, name=STAGE_WORKFLOW)
    return _ordered([task for child in children for task in await _stage_tasks(child)])


def _task(child_id: str, stage: Stage, batch: Batch, status: RunStatus, output, error) -> Task:
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
    one is running while the child runs; the rest wait. A child holds one slice of the job, and
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
            status = RunStatus.ERROR if error else RunStatus.SUCCESS
        else:
            output = error = None
            status = (
                RunStatus.PENDING
                if child.status == RunStatus.PENDING and i == len(steps)
                else RunStatus.ENQUEUED
            )
        out.append(_task(child.workflow_id, stage, batch, status, output, error))
    return out


async def description_tasks(run) -> list[Task]:
    """The tasks of a collection's description: one a member it describes, in the order the run
    listed them (`workflows.summarize_collection_workflow`), then the collection itself. Finished
    ones from the run's step log, a step that only found the describer still loading left out; the
    next one is running while the run runs; the rest wait. None before the run listed its
    members."""
    steps = await DBOS.list_workflow_steps_async(run.workflow_id)
    planned = next((one["output"] for one in steps if one["function_name"] == MEMBERS_STEP), None)
    if not isinstance(planned, list):
        return []
    names = await _document_names(planned)
    collection = _second_segment(run.workflow_id)
    covers = [*(names.get(doc, doc) for doc in planned), f"collection {collection}"]
    outcomes = [
        step
        for step in steps
        if step["function_name"] in (MEMBER_STEP, COLLECTION_STEP)
        and not (isinstance(step["output"], BatchResult) and step["output"].model_loading)
    ]
    running = run.status == RunStatus.PENDING
    out: list[Task] = []
    for seq, name in enumerate(covers):
        error = None
        if seq < len(outcomes):
            _, error = _step_outcome(outcomes[seq])
            status = RunStatus.ERROR if error else RunStatus.SUCCESS
        else:
            status = RunStatus.PENDING if running and seq == len(outcomes) else RunStatus.ENQUEUED
        out.append(
            Task(
                id=f"{run.workflow_id}:{seq}",
                child_id=run.workflow_id,
                stage=Stage.DESCRIBE,
                seq=seq,
                page_start=0,
                page_end=0,
                status=status,
                result=None,
                error=error,
                name=name,
            )
        )
    return out


def _step_outcome(step) -> tuple[int | None, str | None]:
    """A step that failed permanently returns its message instead of raising (see BatchResult)."""
    output, error = step["output"], step["error"]
    if isinstance(output, BatchResult):
        return output.value, output.permanent_error
    return (output if isinstance(output, int) else None), (str(error) if error else None)


async def progress(operation_id: str) -> OperationProgress:
    """The state of one whole-collection or whole-document operation: its status plus the progress
    event a bulk index publishes after every page. `progress` stays None for a delete, which has
    no pages, and for an index that has not finished its first page yet."""
    status = await DBOS.get_workflow_status_async(operation_id)
    if status is None:
        raise NotFound(f"operation not found: {operation_id}")
    kind = _bulk_kind(status.name)
    if kind is None:  # an operation of another kind: it publishes no progress
        raise NotFound(f"operation not found: {operation_id}")
    found = await DBOS.get_event_async(operation_id, workflows.PROGRESS_EVENT, timeout_seconds=0)
    return OperationProgress(
        id=operation_id,
        kind=kind,
        collection=_collection_of(kind, operation_id),
        status=RunStatus(status.status),
        progress=found if isinstance(found, workflows.BulkProgress) else None,
        error=str(status.error) if status.error else None,
    )
