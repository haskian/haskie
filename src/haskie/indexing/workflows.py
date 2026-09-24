"""Durable execution with DBOS, on the same SQLite file as the metadata.

DBOS gives us: crash recovery (a workflow resumes at its first unfinished step or child),
step retries with backoff, queues with concurrency and per-partition limits, deduplication,
and a queryable history (workflow list, children, steps). The one thing it does not give us is a
way to stop a step that is already running, which `collection_lock` below answers for; nothing
else here keeps state of its own.

An operation is what a user asked for; a job is one stage of it (convert, embed, index); a task is
one micro-batch below a job. Operations and their stage children are DBOS workflows, and a task is
one DBOS step (`try_batch`). "Workflow" is the word this module and the DBOS-facing ones beside it
use; see `operations.py` for the read model that never does.

Layout:
- `import_document` per imported document, on `operation.indexing`: convert, then pre-warm the
  embedding cache under the user's default chunk settings (most collections use them, so
  attaching to one is then free). Ends at document status `imported`; no collection is touched.
- `ensure_embedding` per (document, `embed_cache.Params`), on `operation.embedding`, deduplicated
  by the cache id: whoever asks for a missing embedding first computes it, everyone else asking for
  the same one meanwhile waits on that run. A hit in the cache returns at once. This is the only
  place chunks and vectors are computed.
- `index_collection_document` per (collection, document), on `operation.indexing`: ensures the
  embedding the collection's chunk settings call for, then writes it from the cache into the
  collection's table. Moves the membership's status, never the document's.
- Convert and embed are cut into at most `resolve_parallelism` contiguous slices, one
  `stage_slice` child each (`dbos_names.STAGE_WORKFLOW`), with a durable step per micro-batch. So
  one large document spreads over the slots instead of trickling through a single one. The index
  stage is never sliced: it runs as one child on the collection's partition. Each stage has a
  queue of its own, capped by its share of `pipeline.cpu_budget`: `task.converting`,
  `task.embedding`, and `task.indexing` (partitioned by collection, see `INDEX_QUEUE`). A slow
  stage backs up on its own queue instead of taking every slot.
- `remove_from_collection_index` per (collection, document) leaving a collection: on that
  collection's index partition, so rows are never deleted while a step of another document writes
  them. A detach runs one; `delete_document_workflow` runs one per collection the document is in,
  then drops the document's folder and row.
- `maintain_collection` per collection, debounced, on `operation.maintenance`: compaction and index
  (re)build, handed to the collection's index partition. See `collection/maintenance.py`; a burst of
  documents coalesces into one run. The nightly housekeeping schedule sits there too.
- `index_collection_workflow` / `delete_collection_workflow` / `delete_document_workflow` on
  `operation.collection`: whole-thing work the request only starts. A collection with ten thousand
  documents costs the caller one insert instead of ten thousand, and nothing blocks an HTTP
  request for minutes.
- Model downloads live in `models.py` (`operation.downloads`), the operation/job/task read model in
  `operations.py`, and the grouped reads of DBOS's own tables it needs in `sysdb.py`.

Every workflow and every step here is `async def`. A queued async workflow is dispatched as a task
on DBOS's background event loop rather than onto its thread pool, so a workflow that only waits on
children costs a task instead of a thread, and every step awaits its IO.

Two limits on the CPU work, not one, because they answer different questions. The per-queue caps
(`stage_caps`) shape the mix: how the budget is shared out while every stage has work. The
process-wide semaphore (`cpu.cpu_slot`, taken inside `cpu.on_cpu`) is the ceiling, which is what no
queue can enforce, because no queue sees the others: their caps add up to more than the budget
whenever the floor of one slot per stage does (a budget below three), and maintenance runs on a
queue of its own beside all three stages. Every piece of CPU work holds one slot for its length, so
the number of them running at once is never above `pipeline.cpu_budget`.

Workflow ids, every one starting with a prefix that names its kind and the names it belongs to:
`imp:{doc}:{uuid}` for an import, `emb:{doc}:{uuid}` for an embedding run,
`idx-col:{collection}:{doc}:{uuid}` for a collection index, `{parent}:{stage}:{slice}` for each
convert or embed child and `{parent}:index` for the index child, `bulk-index:{collection}:{uuid}`
and `bulk-delete:{collection}:{uuid}` for the two bulk operations, `del-doc:{doc}:{uuid}` for a
document delete (and `{parent}:rm:{collection}` for each collection it leaves),
`maint:{collection}:{parent}` for a maintenance run, and `dl:{kind}:{model}` for a model download
(see `models`). `document.safe_name` keeps `:` out of every name, so a prefix is unambiguous: one
query finds a whole operation. Child ids are deterministic, so a replay after a crash re-attaches
to the child that already exists instead of starting a second one. Every workflow is registered
under an explicit name (see `dbos_names`).
"""

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from enum import StrEnum
from functools import partial
from typing import Any
from uuid import uuid4

import anyio
import anyio.to_thread
import msgspec
from dbos import (
    DBOS,
    DBOSConfig,
    Debouncer,
    SetEnqueueOptions,
    SetWorkflowID,
    SetWorkflowTimeout,
    WorkflowHandleAsync,
)

# Retention has no public entry point in DBOS 3.0: the collector takes the instance itself.
from dbos._dbos import _get_dbos_instance
from dbos._workflow_commands import garbage_collect

from haskie import APP_VERSION, audit, db, home, logs, sysdb
from haskie.audit import Actor, Outcome
from haskie.collection import maintenance
from haskie.collection.collection import Collection, MemberStatus
from haskie.cpu import configure_cpu_budget, shutdown_pool
from haskie.document import document
from haskie.document.document import Document, DocumentStatus, configure_preview_slots
from haskie.errors import Conflict, InvalidInput, NotFound, PermanentError
from haskie.indexing import embed_cache, models, pipeline
from haskie.indexing.dbos_names import (
    ACTIVE_STATUS,
    COLLECTION_DOCUMENT_WORKFLOW,
    DAILY_MAINTENANCE_WORKFLOW,
    DELETE_COLLECTION_WORKFLOW,
    DELETE_DOCUMENT_WORKFLOW,
    EMBED_WORKFLOW,
    IMPORT_WORKFLOW,
    INDEX_COLLECTION_WORKFLOW,
    MAINTAIN_PARTITION_WORKFLOW,
    MAINTAIN_WORKFLOW,
    REMOVE_FROM_INDEX_WORKFLOW,
    STAGE_WORKFLOW,
    root_cause,
)
from haskie.indexing.pipeline import Batch
from haskie.settings import (
    ChunkSettings,
    EmbeddingModel,
    PipelineSettings,
    UserSettings,
    load_user_settings,
)

_log = logs.get_logger(__name__)

# Queues. An `operation.*` queue carries operations, which are made of jobs and tasks and mostly
# wait on them; a `task.*` queue carries the work itself, and its cap is that stage's share of the
# CPU budget.
INDEXING_QUEUE = "operation.indexing"  # one import or collection-index orchestrator per document
EMBEDDING_QUEUE = "operation.embedding"  # one `ensure_embedding` per cache id; its own queue,
# because an orchestrator on `operation.indexing` waits on it, and a queue waiting on itself can
# fill up and stop
COLLECTION_QUEUE = "operation.collection"  # whole-collection index/delete and document delete
MAINTENANCE_QUEUE = "operation.maintenance"  # debounced maintenance and the nightly schedule
CONVERT_QUEUE = "task.converting"  # convert slices; cap = the stage's share of the CPU budget
EMBED_QUEUE = "task.embedding"  # embed slices; cap = the stage's share of the CPU budget
INDEX_QUEUE = "task.indexing"  # index children, maintenance and removals. LanceDB takes one
# writer per collection, so this queue is partitioned by collection and admits one workflow per
# partition (see `index_partition`)

MAINTENANCE_CONCURRENCY = 4  # each waits on a child, so this bounds tasks, not LanceDB writers
MAINTENANCE_TIMEOUT_SECONDS = 3600  # compaction of a very large collection, not a per-batch budget
COLLECTION_CONCURRENCY = 2  # a whole-collection operation only enqueues or cancels; two is plenty
DOWNLOAD_CONCURRENCY = 2  # a download is network bound; two at a time saturates any link
DOCUMENT_CONCURRENCY_CAP = 64  # an orchestrator is cheap now, but its children are not; keep a cap
ADOPT_PAGE = 500  # stale workflows resumed per query at boot
BULK_INDEX_PAGE = 500  # documents enqueued per durable page of a bulk index
CANCEL_PAGE = 200  # pipelines cancelled per sweep of a bulk delete
STAGING_TTL_SECONDS = 24 * 3600  # an upload nobody imported within a day is swept

MAINTENANCE_SCHEDULE = "daily-maintenance"  # housekeeping that costs nothing to skip for a day
MAINTENANCE_CRON = "17 3 * * *"  # nightly, and off the hour: the only clock in this app

IMPORT_PREFIX = "imp"
EMBED_PREFIX = "emb"
COLLECTION_DOCUMENT_PREFIX = "idx-col"
BULK_INDEX_PREFIX = "bulk-index"
BULK_DELETE_PREFIX = "bulk-delete"
DELETE_DOCUMENT_PREFIX = "del-doc"
MAINTAIN_PREFIX = "maint"  # `maint:{collection}:{parent}`: one collection's runs, one id prefix
PROGRESS_EVENT = "progress"  # the DBOS event a bulk index publishes after every page


# What a pipeline run does to its document: one per prefix above, and the word the Operations view
# shows for an operation that belongs to no collection.
class PipelineAction(StrEnum):
    IMPORT = "import"
    EMBED = "embed"
    INDEX = "index"


_PIPELINE_ACTIONS: dict[str, PipelineAction] = {
    IMPORT_PREFIX: PipelineAction.IMPORT,
    EMBED_PREFIX: PipelineAction.EMBED,
    COLLECTION_DOCUMENT_PREFIX: PipelineAction.INDEX,
}


def pipeline_names(workflow_id: str) -> tuple[PipelineAction, str | None, str] | None:
    """The action, the collection and the document one pipeline id carries; None when the id
    is not one of the three shapes.

    Here because this module writes those ids (see the prefixes above). `imp:{doc}:{uuid}` and
    `emb:{doc}:{uuid}` name no collection; `idx-col:{collection}:{doc}:{uuid}` names both.
    `document.safe_name` keeps `:` out of a document and a collection name alike, so the split
    is exact."""
    parts = workflow_id.split(":")
    action = _PIPELINE_ACTIONS.get(parts[0])
    if action is None:
        return None
    if action == PipelineAction.INDEX:
        return (action, parts[1], parts[2]) if len(parts) == 4 else None
    return (action, None, parts[1]) if len(parts) == 3 else None


# DBOS on SQLite has no LISTEN/NOTIFY, so queue dequeue and result waits are polls, and DBOS runs
# one polling thread per queue (`dbos._queue.queue_thread`). The interval is therefore paid
# continuously, idle or not, so the two kinds of queue get different ones.
#
# A `task.*` queue is on the critical path of a document: its interval is added at every stage
# hand-off, so it stays short. An `operation.*` queue carries work a user starts and then watches,
# where a second before it is picked up is invisible. DBOS's own default is 1 s for both.
OPERATION_POLL = 1.0
TASK_POLL = 0.25


# Declared in the order a document moves through them, which is also the order the Operations view
# lists its tasks in.
class Stage(StrEnum):
    CONVERT = "convert"
    EMBED = "embed"
    INDEX = "index"


STAGE_ORDER: tuple[Stage, ...] = tuple(Stage)
STAGE_QUEUE: dict[Stage, str] = {
    Stage.CONVERT: CONVERT_QUEUE,
    Stage.EMBED: EMBED_QUEUE,
    Stage.INDEX: INDEX_QUEUE,
}


class Context(msgspec.Struct):
    """Everything a task needs, captured once per workflow so steps stay pure.

    `document` is the row at load time: its name, suffix, parser and OCR policy are immutable,
    and they are all a pipeline step reads out of it. `chunking` is the collection's when the
    workflow serves one, the user default otherwise. `pipeline` is carried whole rather than
    field by field, so a step that needs another knob costs no new field here. `cache_id` is
    filled in by the workflow that reaches the stage needing it (embed, index)."""

    document: Document
    chunking: ChunkSettings
    embedding: EmbeddingModel | None
    pipeline: PipelineSettings
    collection: str | None = None  # the index stage's target; None for an import or an embed
    cache_id: str = ""  # the embedding being computed (embed) or read (index)


class BatchResult(msgspec.Struct):
    """Outcome of one retried step: a value, or the message of a failure that must not be retried.

    DBOS retries *every* exception raised inside a step with `retries_allowed`, so a deterministic
    failure (unsupported file, OCR policy, corrupt document) is reported as a value and raised by
    the workflow body instead."""

    value: int | None = None
    permanent_error: str | None = None


class BulkProgress(msgspec.Struct):
    """How far a bulk index got: the `progress` event the workflow publishes after every page.

    `enqueue_page` returns one page in the same shape, with `total` left at 0 and `last` naming
    the document the next page continues after (None once the collection is exhausted)."""

    done: int
    total: int = 0
    last: str | None = None


class BulkResult(msgspec.Struct):
    """Outcome of a bulk index: the documents it queued."""

    done: int


class PipelineError(RuntimeError):
    """Carries only the flat root-cause message (survives DBOS's error (de)serialization)."""


# --- the collection write lock ----------------------------------------------------

# One lock per collection, held by whoever is writing that collection's LanceDB table or taking
# it away. Not durable and not meant to be: it orders two things running in this process right
# now, and a crash leaves neither of them running.
_collection_locks: dict[str, anyio.Lock] = {}


def collection_lock(collection: str) -> anyio.Lock:
    """The lock a write to one collection's table and a removal of that collection take turns on.

    `DBOS.cancel_workflows` rewrites the status row and nothing else: a step already inside its
    LanceDB write keeps running, and the index partition frees its slot as soon as the row says
    CANCELLED. So a delete or a detach that cancelled an index workflow can reach its own removal
    while that write is still going - and the write would recreate the table folder the delete
    just took away, or put back rows the detach just removed. A queue cannot order those two; this
    lock does.

    An `anyio.Lock` rather than a thread lock because every holder runs on DBOS's background event
    loop: the index steps, the removal steps and the workflow bodies around them all do. The
    HTTP-side helpers (`detach`, `start_*`) only enqueue and never take it.

    `setdefault` with no await in between, so two callers arriving at once share one lock."""
    return _collection_locks.setdefault(collection, anyio.Lock())


# --- lifecycle --------------------------------------------------------------------


def document_concurrency(indexing: PipelineSettings) -> int:
    """How many documents may be orchestrated at once. An orchestrator does no work of its own: it
    waits on the children of its current stage, so admitting twice the CPU budget keeps the task
    queues fed while a document changes stage. Each still holds a row and a task, hence the cap."""
    return min(DOCUMENT_CONCURRENCY_CAP, 2 * indexing.cpu_budget)


def stage_caps(indexing: PipelineSettings) -> dict[Stage, int]:
    """How many tasks each stage queue admits: `cpu_budget` shared out over the stage weights.

    Largest remainder: every stage gets the whole part of its share, and the spare slots go to the
    stages that lost the most to rounding. So the caps add up to the budget exactly - except under
    the floor of one slot per stage, which a budget below three cannot pay for. That is what
    `cpu.cpu_slot` is for: the floors keep every stage alive, the semaphore keeps the total honest.
    """
    weights: dict[Stage, int] = {
        Stage.CONVERT: indexing.converting_weight,
        Stage.EMBED: indexing.embedding_weight,
        Stage.INDEX: indexing.indexing_weight,
    }
    total = sum(weights.values())
    shares = {stage: indexing.cpu_budget * weight / total for stage, weight in weights.items()}
    caps = {stage: int(share) for stage, share in shares.items()}
    spare = indexing.cpu_budget - sum(caps.values())
    by_remainder = sorted(shares, key=lambda stage: shares[stage] - caps[stage], reverse=True)
    for stage in by_remainder[:spare]:
        caps[stage] += 1
    return {stage: max(1, cap) for stage, cap in caps.items()}


# The backlog adoption in flight, if any. An event loop keeps only a weak reference to a task, so
# the module holds the strong one and `stop` cancels it.
_adoption: asyncio.Task[None] | None = None


async def start() -> None:
    """Bring the runtime up: migrations, DBOS, the queues, the schedules. Awaited by Litestar's
    startup hook, and by the tests."""
    logs.configure()
    await db.migrate_once()  # before DBOS opens the file: the one-time WAL switch needs exclusivity
    config: DBOSConfig = {
        "name": "haskie",
        # Pinned: DBOS only recovers in-flight workflows of its own version, and the default is
        # a hash of the workflow source, which changes on every edit under `--reload`.
        "application_version": APP_VERSION,
        # SQLAlchemy does not unquote the database part of a sqlite URL, so the path goes in as
        # written; `quote()` here would open a file whose name contains "%20".
        "system_database_url": f"sqlite:///{home.DB_FILE}",
        # How often the notification listener re-reads the events and messages a waiter asked
        # for. The default is a second; on a local file the same interval every other wait here
        # uses is cheap, and it is what a progress event costs before a reader sees it. It is
        # also how long `DBOS.destroy` takes to join that thread, which the tests feel most.
        "notification_listener_polling_interval_sec": TASK_POLL,
        "log_level": logs.level(),
    }
    DBOS(config=config)
    logs.adopt_dbos_logger()  # DBOS installs its own text handler while it initializes
    # In a worker thread on purpose: `launch` is sync SQLAlchemy, and it adopts the loop of the
    # thread that calls it as the one queued async workflows run on. From a thread there is none,
    # so they run on DBOS's own background loop and never share Litestar's.
    await anyio.to_thread.run_sync(DBOS.launch)
    _start_adoption()
    try:
        await apply_settings(await load_user_settings())
    except InvalidInput as exc:
        # boot must not depend on a settings row another build wrote; the UI can fix it
        _log.error("settings_invalid_at_boot", error=str(exc))
        await apply_settings(UserSettings())
    # after apply_settings: a schedule may only name a queue the system database already carries
    await _register_schedule()
    # also prune once per boot: a desktop app is rarely running at 03:17, so a run that only ever
    # happened on the schedule would never happen at all
    await _housekeeping(
        prune_audit(), "audit_prune_failed", partial(_log.info, "audit_files_pruned")
    )


async def stop() -> None:
    """Litestar calls a shutdown hook with the app when the hook takes any parameter, so this one
    takes none. In-flight steps get a grace period: a worker thread outliving DBOS blocks
    interpreter exit."""
    global _adoption
    if _adoption is not None:
        adopting, _adoption = _adoption, None
        adopting.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await adopting
    await anyio.to_thread.run_sync(partial(DBOS.destroy, workflow_completion_timeout_sec=10))
    shutdown_pool()  # after DBOS, so nothing is still submitting extraction work


async def _register_schedule() -> None:
    """Put the nightly cron definition in the system database, where it outlives the process.

    `apply_schedules_async` is an idempotent upsert by name that keeps the schedule's id, status
    and last fire time, so every boot may simply declare what this build wants."""
    await DBOS.apply_schedules_async(
        [
            {
                "schedule_name": MAINTENANCE_SCHEDULE,
                "workflow_fn": daily_maintenance,
                "schedule": MAINTENANCE_CRON,
                "queue_name": MAINTENANCE_QUEUE,
            }
        ]
    )
    _log.debug("schedule_registered", schedule=MAINTENANCE_SCHEDULE)


async def _housekeeping(
    work: Awaitable[int], failed_event: str, done: Callable[..., None], **fields: Any
) -> None:
    """Run one boot-time chore. A failure is logged and the boot continues; a count is only worth
    a line when it is not zero, and `done` is the bound event that writes it."""
    try:
        count = await work
    except Exception as exc:
        _log.error(failed_event, error=root_cause(exc))
        return
    if count:
        done(count=count, **fields)


async def adopt_orphans(batch: int = ADOPT_PAGE) -> int:
    """Re-enqueue non-terminal workflows recorded under a different application version.
    Recovery skips them; resuming replays them from their step logs under this version.

    Read and resume one page of ids at a time: after a long outage the backlog can be large, and
    neither the whole list of statuses nor a resume call per workflow belongs on the boot path. A
    row that leaves the page while we walk it was already adopted and dequeued; anything the
    shifted window skips is adopted at the next boot."""
    adopted = 0
    while True:
        stale = await sysdb.stale_active_ids(APP_VERSION, batch, adopted)
        if not stale:
            return adopted
        await DBOS.resume_workflows_async(stale)
        adopted += len(stale)


def _start_adoption() -> None:
    """Adopting a long backlog takes as long as the backlog is deep, and nothing waits for its
    result: the boot hands it to a task on the loop it runs on and returns (see `_adoption`)."""
    global _adoption
    adopting = _housekeeping(
        adopt_orphans(),
        "stale_workflow_adoption_failed",
        partial(_log.warning, "stale_workflows_resumed"),
        app_version=APP_VERSION,
    )
    _adoption = asyncio.get_running_loop().create_task(adopting, name="haskie-adopt")


class Queue(msgspec.Struct, frozen=True):
    """One registered DBOS queue: its name, how wide it is under the current settings, and
    whether it admits one workflow per partition. The `operation.`/`task.` prefix picks the polling
    interval and is the family `sysdb.queue_activity` groups by."""

    name: str
    concurrency: Callable[[PipelineSettings, dict[Stage, int]], int]
    partition_concurrency: int | None = None


_QUEUES: tuple[Queue, ...] = (
    Queue(INDEXING_QUEUE, lambda indexing, caps: document_concurrency(indexing)),
    Queue(EMBEDDING_QUEUE, lambda indexing, caps: document_concurrency(indexing)),
    Queue(COLLECTION_QUEUE, lambda indexing, caps: COLLECTION_CONCURRENCY),
    Queue(models.DOWNLOADS_QUEUE, lambda indexing, caps: DOWNLOAD_CONCURRENCY),
    Queue(MAINTENANCE_QUEUE, lambda indexing, caps: MAINTENANCE_CONCURRENCY),
    Queue(CONVERT_QUEUE, lambda indexing, caps: caps[Stage.CONVERT]),
    Queue(EMBED_QUEUE, lambda indexing, caps: caps[Stage.EMBED]),
    Queue(INDEX_QUEUE, lambda indexing, caps: caps[Stage.INDEX], partition_concurrency=1),
)


async def apply_settings(settings: UserSettings) -> None:
    """Queue limits, the CPU budget and the preview build pool follow the settings; required
    models start loading.

    `configure_preview_slots` builds primitives of the loop it is called on, and previews are
    built on that same loop: this runs either at startup or inside a settings request, both of
    which are Litestar's loop."""
    indexing = settings.pipeline
    caps = stage_caps(indexing)
    configure_cpu_budget(indexing.cpu_budget)
    configure_preview_slots(indexing.preview_workers)
    for queue in _QUEUES:
        await DBOS.register_queue_async(
            queue.name,
            global_concurrency=queue.concurrency(indexing, caps),
            partition_concurrency=queue.partition_concurrency,
            polling_interval_sec=(
                OPERATION_POLL if queue.name.startswith("operation.") else TASK_POLL
            ),
        )
    await models.ensure_models(settings)
    await schedule_pending_maintenance(indexing.maintenance_idle_seconds)


# --- steps (pure, retried) ----------------------------------------------------------

# Every step here touches SQLite or the file system, so every step may hit a transient lock.
# Read once, here: DBOS copies a step's retry settings into the decorator, so this cannot change
# after import.
RETRY_INTERVAL_SECONDS = 1.0
retried_step = DBOS.step(
    retries_allowed=True,
    max_attempts=3,
    interval_seconds=RETRY_INTERVAL_SECONDS,
    backoff_rate=2.0,
)


@retried_step
async def load_context(doc: str, collection: str | None) -> Context:
    """The document row and the settings a pipeline runs under. `collection` names the one whose
    chunk settings apply; None takes the user defaults (an import's pre-warm, an embed run that
    substitutes its own params afterwards)."""
    row = await document.get(doc)
    user = await load_user_settings()
    chunking = (
        await Collection(collection).chunk_settings() if collection else user.conversion.chunking
    )
    return Context(
        document=row,
        chunking=chunking,
        embedding=user.embedding_model,
        pipeline=user.pipeline,
        collection=collection,
    )


def resolve_parallelism(indexing: PipelineSettings, stage: Stage) -> int:
    """Slices one document may be cut into for one stage: that stage's cap when the setting is 0,
    never more - a slice occupies one slot of that stage's queue, so asking for more only queues
    them."""
    cap = stage_caps(indexing)[stage]
    if indexing.document_parallelism == 0:
        return cap
    return min(indexing.document_parallelism, cap)


@retried_step
async def set_status(doc: str, status: DocumentStatus, error: str | None = None) -> None:
    await document.set_status(doc, status, error)


@retried_step
async def set_member_status(
    collection: str, doc: str, status: MemberStatus, error: str | None = None
) -> None:
    await Collection(collection).set_member_status(doc, status, error)


async def _member_present(collection: str, doc: str) -> bool:
    try:
        await Collection(collection).member(doc)
    except NotFound:
        return False
    return True


@retried_step
async def member_present(collection: str, doc: str) -> bool:
    """Whether the membership still exists: a detach that landed between the enqueue and the run
    must not leave rows in the table with no membership to remove them by."""
    return await _member_present(collection, doc)


@contextlib.asynccontextmanager
async def index_write(collection: str, doc: str) -> AsyncIterator[bool]:
    """Hold the collection's write lock for one write of the index child, and report whether the
    membership is still there once the lock is ours.

    The lock alone only orders this write against a removal (see `collection_lock`); the re-check
    inside it is what the loser of that race acts on. A removal that went first took the
    membership with it - the whole collection row for a delete (memberships cascade), this one
    row for a detach - so a write that finds none has nothing left to write into."""
    async with collection_lock(collection):
        yield await _member_present(collection, doc)


@retried_step
async def plan(stage: Stage, ctx: Context) -> list[Batch]:
    if stage == Stage.CONVERT:
        return await pipeline.plan_convert(ctx.document, ctx.pipeline.batch_pages)
    if stage == Stage.EMBED:
        return await pipeline.plan_embed(ctx.document)
    return await pipeline.plan_index(ctx.document, ctx.cache_id, ctx.pipeline.index_group_parts)


async def _guarded(call: Awaitable[int | None]) -> BatchResult:
    """Await one pipeline call inside a retried step: retry anything transient, report a
    `PermanentError` as a value so DBOS does not retry a failure that cannot change."""
    try:
        return BatchResult(value=await call)
    except PermanentError as exc:
        return BatchResult(permanent_error=f"{type(exc).__name__}: {exc}")


def _value(result: BatchResult) -> int:
    """Called in the workflow body, outside the step: raising here fails the workflow at once."""
    if result.permanent_error is not None:
        raise PermanentError(result.permanent_error)
    return result.value or 0


async def _index_batch(batch: Batch, ctx: Context) -> int:
    """One index micro-batch, under the collection's write lock (see `index_write`)."""
    assert ctx.collection is not None, "the index stage always names a collection"
    async with index_write(ctx.collection, ctx.document.name) as present:
        if not present:
            raise PermanentError("document is no longer in the collection")
        return await pipeline.index_batch(
            Collection(ctx.collection), ctx.document, ctx.cache_id, batch, ctx.embedding
        )


@retried_step
async def try_batch(stage: Stage, batch: Batch, ctx: Context) -> BatchResult:
    """The CPU work of one micro-batch. The slot of the CPU budget is taken inside `cpu.on_cpu`,
    around the CPU work alone: the file reads, the file writes and the LanceDB commit of the same
    batch await without holding it, and neither does the bookkeeping DBOS does around the step."""
    if stage == Stage.CONVERT:
        return await _guarded(pipeline.convert_batch(ctx.document, batch))
    if stage == Stage.EMBED:
        return await _guarded(
            pipeline.embed_batch(ctx.document, batch, ctx.cache_id, ctx.chunking, ctx.embedding)
        )
    return await _guarded(_index_batch(batch, ctx))


@retried_step
async def try_finalize_convert(batches: list[Batch], ocr_total: int, ctx: Context) -> BatchResult:
    """Assemble the markdown out of every part the convert slices wrote. A step of the parent
    workflow, not of a child: it needs the OCR counts of all slices, which only the parent has."""
    return await _guarded(pipeline.finalize_convert(ctx.document, batches, ocr_total))


@retried_step
async def cache_lookup(params: embed_cache.Params) -> str | None:
    return await embed_cache.lookup(params)


@retried_step
async def forget_embeddings(doc: str) -> None:
    """Before a (re)conversion: every cached embedding was chunked from markdown that is about
    to be rewritten. A first import has none; a retried one may."""
    await embed_cache.forget(doc)


@retried_step
async def finalize_embed(params: embed_cache.Params, ctx: Context) -> str:
    """Merge the rows of every part into the cache file and publish it (see `embed_cache`)."""
    return await pipeline.finalize_embed(ctx.document, params, ctx.embedding)


@retried_step
async def prepare_index(collection: str, ctx: Context) -> None:
    """Clear the collection's table of this document before its first index batch runs, on the
    collection's partition like every other write to its table (see `pipeline.prepare_index`).

    Recreates a missing table, so it takes the write lock like the batches do. Skipped rather than
    failed once the membership is gone: the batch write is where that is reported."""
    async with index_write(collection, ctx.document.name) as present:
        if present:
            await pipeline.prepare_index(Collection(collection), ctx.document, ctx.embedding)


@retried_step
async def finalize_index(collection: str, ctx: Context) -> None:
    """Rebuild the full-text index once, on the collection's partition, under the same lock and
    the same re-check as the batches (see `prepare_index`)."""
    async with index_write(collection, ctx.document.name) as present:
        if present:
            await pipeline.finalize_index(Collection(collection), ctx.embedding)


@retried_step
async def note_indexed_step(collection: str, doc: str) -> int:
    """Count one more indexed document for the collection; returns how many await maintenance."""
    pending = await Collection(collection).note_indexed()
    _log.debug("index_pending", collection=collection, document=doc, pending=pending)
    return pending


@retried_step
async def claim_pending(collection: str) -> int:
    # nothing is reset here: a run that crashes must leave the collection pending, and documents
    # indexed while it runs must still count towards the next one (see `settle_maintenance`)
    state = await Collection(collection).maintenance_state()
    return state.pending_documents if state else 0


@retried_step
async def run_maintenance(collection: str) -> maintenance.Report:
    """`Collection(name)`, not `Collection.get(name)`: the collection may have been deleted while
    this run waited, and `maintenance.run` reports that as a skip rather than a failure.

    No slot of the CPU budget is taken around it: compaction and the vector index build are CPU,
    but they run inside LanceDB's own runtime rather than in a worker thread of ours, so there is
    nothing for `cpu.cpu_slot` to hold. `task.indexing` bounds them instead."""
    user = await load_user_settings()
    return await maintenance.run(Collection(collection), user.embedding_model, user.pipeline)


@retried_step
async def settle_maintenance(collection: str, claimed: int, report: maintenance.Report) -> None:
    await Collection(collection).settle_maintenance(claimed, report.ann_trained, report.num_rows)


async def run_batch(stage: Stage, batch: Batch, ctx: Context) -> int:
    return _value(await try_batch(stage, batch, ctx))


# --- workflows --------------------------------------------------------------------------


@DBOS.workflow(name=STAGE_WORKFLOW)
async def stage_slice(stage: Stage, batches: list[Batch], ctx: Context) -> list[int]:
    """One slice of one stage of one document: a durable step per micro-batch, in plan order.

    The batches of a slice run one after another, because the steps of a DBOS workflow do; the
    parallelism inside a document comes from `_stage` running several of these at once. The index
    stage is never sliced and runs on the collection's write partition, so this is also where its
    once-per-document hooks belong: clear the document's rows before the first batch, build the
    full-text index after the last."""
    indexed = ctx.collection if stage == Stage.INDEX else None  # the index stage always names one
    if indexed is not None:
        await prepare_index(indexed, ctx)
    results = [await run_batch(stage, batch, ctx) for batch in batches]
    if indexed is not None:
        await finalize_index(indexed, ctx)
    return results


def stage_input(child) -> tuple[Stage, list[Batch]] | None:
    """The stage and the micro-batches a `stage_slice` child was given, read back out of its
    recorded input; None when DBOS did not keep it.

    Here rather than in `operations`, so the argument positions and the signature they index into
    are edited in one place."""
    # an input DBOS can no longer unpickle comes back as its raw text, with a warning: one recorded
    # before a struct inside `Context` changed shape. Its batches are unknown, as if not kept.
    if not isinstance(child.input, dict):
        return None
    args = child.input["args"]
    return (args[0], args[1]) if args else None


def index_partition(collection: str) -> str:
    """The queue partition every write to one collection's table goes through. Index children,
    maintenance runs and removals of one collection all name it, so they never write at once."""
    return f"index:{collection}"


def _partition_key(stage: Stage, collection: str | None) -> str | None:
    """`index_partition` for an index child; None for a convert or embed slice, whose queues are
    capped globally and hold nothing a slice of the same document could corrupt."""
    return index_partition(collection) if stage == Stage.INDEX and collection else None


def _child_id(stage: Stage, slice_index: int) -> str:
    """Deterministic, so a replay after a crash re-attaches to the child that already exists. The
    index stage has exactly one child, and keeps the unsuffixed id it always had."""
    if stage == Stage.INDEX:
        return f"{DBOS.workflow_id}:{stage}"
    return f"{DBOS.workflow_id}:{stage}:{slice_index}"


def run_id(workflow_id: str) -> str:
    """The uuid tail of a workflow id: what its deterministic child ids are derived from."""
    return workflow_id.rsplit(":", 1)[-1]


def _slice_count(stage: Stage, ctx: Context) -> int:
    """Slices this stage may be cut into. The index stage is never sliced (see `INDEX_QUEUE`)."""
    return 1 if stage == Stage.INDEX else resolve_parallelism(ctx.pipeline, stage)


def _slices(batches: list[Batch], parts: int) -> list[list[Batch]]:
    """Cut the batches into at most `parts` contiguous runs of near-equal length.

    Contiguous rather than round-robin: consecutive micro-batches are consecutive part files, so
    one worker walks one region of the document. Always at least one run, even for a stage that
    planned no batches at all: the index child still has to clear the document's rows and build
    the full-text index (see `stage_slice`)."""
    count = max(1, min(len(batches), parts))
    size, extra = divmod(len(batches), count)
    out: list[list[Batch]] = []
    start = 0
    for index in range(count):
        end = start + size + (1 if index < extra else 0)
        out.append(batches[start:end])
        start = end
    return out


async def _stage(stage: Stage, ctx: Context) -> list[int]:
    """Plan the stage, run its slices as child workflows side by side, wait for all of them.

    Returns the per-batch results in plan order: the slices are contiguous, so concatenating them
    in slice order restores it. The convert finalizer runs here rather than in a child, because it
    needs the OCR counts of every slice, which only the parent sees."""
    batches = await plan(stage, ctx)
    handles: list[WorkflowHandleAsync[list[int]]] = []
    for index, batches_of_slice in enumerate(_slices(batches, _slice_count(stage, ctx))):
        with (
            SetWorkflowID(_child_id(stage, index)),
            # the budget is per batch, and the child runs them all: a long slice gets
            # proportionally longer rather than timing out for being long
            SetWorkflowTimeout(ctx.pipeline.task_timeout_seconds * max(1, len(batches_of_slice))),
            SetEnqueueOptions(queue_partition_key=_partition_key(stage, ctx.collection)),
        ):
            # the context managers wrap the await itself: `enqueue_workflow_async` reads the
            # contextvars they set before it yields to the loop
            handles.append(
                await DBOS.enqueue_workflow_async(
                    STAGE_QUEUE[stage], stage_slice, stage, batches_of_slice, ctx
                )
            )
    results = [
        result
        for handle in handles
        for result in await handle.get_result(polling_interval_sec=TASK_POLL)
    ]
    if stage == Stage.CONVERT:
        _value(await try_finalize_convert(batches, sum(results), ctx))
    return results


async def _ensure_embedding(ctx: Context) -> str:
    """The cache id of the embedding `ctx` calls for, computing it through `ensure_embedding`
    when it is missing. Deduplicated by the cache id: two callers wanting the same embedding at
    once share one run instead of computing it twice (and racing on the write). The child id is
    derived from this workflow's, so a replay re-attaches to the run it already started."""
    params = embed_cache.params(ctx.document, ctx.chunking, ctx.embedding)
    with (
        SetWorkflowID(f"{EMBED_PREFIX}:{ctx.document.name}:{run_id(DBOS.workflow_id or '')}"),
        SetEnqueueOptions(
            deduplication_id=embed_cache.key(params), duplication_policy="return-existing"
        ),
    ):
        handle = await DBOS.enqueue_workflow_async(
            EMBEDDING_QUEUE, ensure_embedding, ctx.document.name, params
        )
    return await handle.get_result(polling_interval_sec=TASK_POLL)


@contextlib.asynccontextmanager
async def _outcome(
    set_state: Callable[..., Awaitable[None]],
    done: DocumentStatus | MemberStatus,
    failed: DocumentStatus | MemberStatus,
    collection: str | None,
    doc: str,
    event: str,
) -> AsyncIterator[None]:
    """Move the status to its end state and write one audit line, whichever way the body ended.

    Both document workflows report a failure the same way: a `PermanentError` carries its own
    message, anything else is unwrapped to its root cause, and what leaves the workflow is a flat
    `PipelineError` DBOS can store and rebuild."""
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:
        message = str(exc) if isinstance(exc, PermanentError) else root_cause(exc)
        await set_state(failed, message)
        await _record(collection, doc, f"{event}.failed", started, message)
        raise PipelineError(message) from exc
    await set_state(done)
    await _record(collection, doc, f"{event}.completed", started, None)


@DBOS.workflow(name=IMPORT_WORKFLOW)
async def import_document(doc: str) -> DocumentStatus:
    """convert, then pre-warm the embedding cache under the user's default chunk settings;
    document status mirrors the stage. Collection-independent: nothing is written to any table."""
    with logs.bound(workflow_id=DBOS.workflow_id, document=doc):
        # first step, so a deduplicated submit changes nothing
        await set_status(doc, DocumentStatus.QUEUED)
        set_state = partial(set_status, doc)
        async with _outcome(
            set_state, DocumentStatus.IMPORTED, DocumentStatus.ERROR, None, doc, "import"
        ):
            ctx = await load_context(doc, None)
            await set_status(doc, DocumentStatus.CONVERTING)
            await forget_embeddings(doc)
            await _stage(Stage.CONVERT, ctx)
            await set_status(doc, DocumentStatus.EMBEDDING)
            await _ensure_embedding(ctx)
        return DocumentStatus.IMPORTED


@DBOS.workflow(name=EMBED_WORKFLOW)
async def ensure_embedding(doc: str, params: embed_cache.Params) -> str:
    """One cached embedding of one document, computed when missing; returns its cache id.

    The chunk settings come from `params`, not from any collection: the collection's settings may
    change between the enqueue and the run, and what was asked for is what the id names. The
    embedding model is the global one, so a model changed meanwhile fails the run: the parent
    asks again under the new model."""
    with logs.bound(workflow_id=DBOS.workflow_id, document=doc):
        found = await cache_lookup(params)
        if found is not None:
            return found
        ctx = await load_context(doc, None)
        current = ctx.embedding.cache_name if ctx.embedding else embed_cache.NO_MODEL
        if current != params.model:
            raise PermanentError(f"embedding model changed: wanted {params.model}, have {current}")
        ctx = msgspec.structs.replace(
            ctx,
            chunking=ChunkSettings.of(params),
            cache_id=embed_cache.key(params),
        )
        await _stage(Stage.EMBED, ctx)
        return await finalize_embed(params, ctx)


@DBOS.workflow(name=COLLECTION_DOCUMENT_WORKFLOW)
async def index_collection_document(collection: str, doc: str) -> MemberStatus:
    """Write one document into one collection's table from its embedding cache, computing the
    embedding first when the collection's chunk settings have none yet. Membership status mirrors
    the progress; the document's own status is the import's and is never touched here."""
    with logs.bound(workflow_id=DBOS.workflow_id, collection=collection, document=doc):
        await set_member_status(collection, doc, MemberStatus.INDEXING)
        set_state = partial(set_member_status, collection, doc)
        done, failed = MemberStatus.INDEXED, MemberStatus.ERROR
        async with _outcome(set_state, done, failed, collection, doc, "index"):
            if not await member_present(collection, doc):
                raise PermanentError("document is no longer in the collection")
            ctx = await load_context(doc, collection)
            if ctx.document.status != DocumentStatus.IMPORTED:
                raise PermanentError(f"document is not imported: {ctx.document.status}")
            ctx = msgspec.structs.replace(ctx, cache_id=await _ensure_embedding(ctx))
            await _stage(Stage.INDEX, ctx)
            pending = await note_indexed_step(collection, doc)
            await request_maintenance(collection, pending, ctx.pipeline)
        return MemberStatus.INDEXED


async def _record(
    collection: str | None, doc: str, event: str, started: float, error: str | None
) -> None:
    """One audit line per finished document. Not a step: a replay after a crash re-appends it,
    which an append-only trail tolerates."""
    await audit.record(
        event,
        actor=Actor.OPERATION,
        outcome=Outcome.OK if error is None else Outcome.ERROR,
        duration_ms=int((time.perf_counter() - started) * 1000),
        operation_id=DBOS.workflow_id,
        collection=collection,
        document=doc,
        error=error,
    )


@DBOS.workflow(name=MAINTAIN_PARTITION_WORKFLOW)
async def maintain_on_partition(collection: str) -> maintenance.Report:
    """The maintenance itself, on the collection's index partition, so compaction waits for the
    index stage of any document in flight, and vice versa."""
    with logs.bound(workflow_id=DBOS.workflow_id, collection=collection):
        claimed = await claim_pending(collection)
        report = await run_maintenance(collection)
        await settle_maintenance(collection, claimed, report)
        return report


@DBOS.workflow(name=MAINTAIN_WORKFLOW)
async def maintain_collection(collection: str) -> maintenance.Report:
    """Debounced entry point. It only hands the work to the collection's index partition and
    waits.

    Two hops because a debounce needs deduplication, and a partitioned queue does not support it:
    this one sits on the unpartitioned `operation.maintenance` queue, the work it enqueues on the
    partition every index write already uses.

    The child's id names the collection and this run, so the Operations view filters maintenance by
    collection with an id prefix like every other listing, and a replay re-attaches to the child
    that already exists instead of starting a second one."""
    with (
        SetWorkflowTimeout(MAINTENANCE_TIMEOUT_SECONDS),
        SetEnqueueOptions(queue_partition_key=index_partition(collection)),
        SetWorkflowID(f"{MAINTAIN_PREFIX}:{collection}:{DBOS.workflow_id}"),
    ):
        handle = await DBOS.enqueue_workflow_async(INDEX_QUEUE, maintain_on_partition, collection)
    return await handle.get_result(polling_interval_sec=TASK_POLL)


# The debounce key is the collection name, so a burst of documents coalesces into one run: each
# request pushes the delay out, and the run starts once the burst stops (or `maintenance_documents`
# documents landed, which requests it with no delay at all).
MAINTAIN = Debouncer.create_async(maintain_collection, queue=MAINTENANCE_QUEUE)


async def request_maintenance(collection: str, pending: int, indexing: PipelineSettings) -> None:
    """Ask for a maintenance run: now once `maintenance_documents` documents piled up, otherwise
    once the collection has been idle for `maintenance_idle_seconds`.

    Must be called with no `SetEnqueueOptions` partition key in context: a debounce deduplicates,
    and DBOS rejects deduplication on a partitioned enqueue."""
    idle = float(indexing.maintenance_idle_seconds)
    period = 0.0 if pending >= indexing.maintenance_documents else idle
    await MAINTAIN.debounce_async(collection, period, collection)


async def schedule_pending_maintenance(idle_seconds: int) -> None:
    """Reschedule every collection that has documents pending. A run lost to a crash or a shutdown
    leaves `pending_documents` standing, so the next boot picks the collection up again."""
    for name in await Collection.pending_names():
        await MAINTAIN.debounce_async(name, float(idle_seconds), name)


# --- removal ------------------------------------------------------------------------------


@retried_step
async def remove_index_rows(collection: str, doc: str) -> None:
    await (await Collection(collection).index()).delete_document(doc)


@retried_step
async def remove_member_row(collection: str, doc: str) -> None:
    """Last: while the row exists the membership is still listed, so a crash leaves no phantom."""
    await Collection(collection).remove_member(doc)


@DBOS.workflow(name=REMOVE_FROM_INDEX_WORKFLOW)
async def remove_from_collection_index(collection: str, doc: str) -> None:
    """Take one document out of one collection: its rows in the table, then its membership. The
    document itself, its files and its embedding cache are untouched.

    Both steps under the collection's write lock, not only the first: an index step still in
    flight (its workflow was cancelled, which stops nothing already running) would otherwise take
    the lock between them, still find the membership, and write the rows back."""
    with logs.bound(workflow_id=DBOS.workflow_id, collection=collection, document=doc):
        async with collection_lock(collection):
            await remove_index_rows(collection, doc)
            await remove_member_row(collection, doc)


@retried_step
async def cancel_document_work(doc: str) -> None:
    """Cancel every import, embedding run and collection index of the document. One call: DBOS
    writes CANCELLED for the whole list, children included, before it returns. Retried: it is a
    series of DBOS writes, and cancelling again is a no-op."""
    await DBOS.cancel_workflows_async(await _active_document_workflows(doc), cancel_children=True)


@retried_step
async def memberships(doc: str) -> list[str]:
    """The collections holding the document at this moment. Recorded, so a replay walks the same
    list; an attach after this snapshot is refused by the `deleting` status set before it."""
    return await document.collections_of(doc)


@retried_step
async def remove_document_files(doc: str) -> None:
    await document.remove_files(doc)


@retried_step
async def remove_document_row(doc: str) -> None:
    """Last: cascades to the memberships and the embeddings rows; while the row exists the
    document is still listed, so a crash leaves a document that can be deleted again."""
    await document.remove_row(doc)


@DBOS.workflow(name=DELETE_DOCUMENT_WORKFLOW)
async def delete_document_workflow(doc: str) -> None:
    """Delete a document everywhere: out of every collection's table (one child per collection,
    each on that collection's index partition), then its folder, then its row.

    `deleting` is set first, so an attach that lands after the membership snapshot below is
    refused instead of leaving rows in a table no membership points at."""
    with logs.bound(workflow_id=DBOS.workflow_id, document=doc):
        await set_status(doc, DocumentStatus.DELETING)
        await cancel_document_work(doc)
        handles: list[WorkflowHandleAsync[None]] = []
        for collection in await memberships(doc):
            with (
                SetWorkflowID(f"{DBOS.workflow_id}:rm:{collection}"),
                SetEnqueueOptions(queue_partition_key=index_partition(collection)),
            ):
                handles.append(
                    await DBOS.enqueue_workflow_async(
                        INDEX_QUEUE, remove_from_collection_index, collection, doc
                    )
                )
        for handle in handles:
            await handle.get_result(polling_interval_sec=TASK_POLL)
        await remove_document_files(doc)
        await remove_document_row(doc)


# --- whole-collection operations ---------------------------------------------------------


@retried_step
async def count_members(collection: str) -> int:
    """Only feeds the progress event, so a collection that is already gone counts as empty."""
    return (await Collection(collection).counts()).total


@retried_step
async def member_page(collection: str, after: str | None) -> list[str]:
    """One keyset page of member names, ordered by name. Empty once the collection is exhausted,
    and also when it was deleted while the bulk index ran, which ends the walk either way."""
    return await Collection(collection).member_names(after, BULK_INDEX_PAGE)


async def enqueue_page(collection: str, after: str | None, bulk_id: str) -> BulkProgress:
    """Queue the index of one page of members and report what that did.

    Not a step: DBOS refuses to start a workflow inside one. The listing above is the step, and
    its recorded output is what makes a replay walk the same names in the same order. Each child
    gets an id derived from the bulk operation, so a replay re-attaches to the workflow it already
    started; a document indexing under an older id is returned by the deduplication in
    `_enqueue_index` instead of being queued twice.

    `_enqueue_index` rather than `start_index_collection_document`: the step above just read these
    names out of this collection's membership table, so re-reading the collection and the
    membership per document would only cost two connections each.
    """
    done = 0
    last: str | None = None
    for doc in await member_page(collection, after):
        last = doc
        await _enqueue_index(
            collection, doc, f"{COLLECTION_DOCUMENT_PREFIX}:{collection}:{doc}:{bulk_id[-32:]}"
        )
        done += 1
    return BulkProgress(done=done, last=last)


@DBOS.workflow(name=INDEX_COLLECTION_WORKFLOW)
async def index_collection_workflow(collection: str) -> BulkResult:
    """(Re)index every member of one collection, one durable page of enqueues at a time. Cheap
    for a member whose embedding is cached: the embed is skipped and only the table is written."""
    with logs.bound(workflow_id=DBOS.workflow_id, collection=collection):
        bulk_id = DBOS.workflow_id or ""
        total = await count_members(collection)
        done = 0
        after: str | None = None
        while True:
            page = await enqueue_page(collection, after, bulk_id)
            done, after = done + page.done, page.last
            await DBOS.set_event_async(PROGRESS_EVENT, BulkProgress(done, total, after))
            if after is None:
                _log.info("collection_index_queued", collection=collection, done=done)
                return BulkResult(done=done)


@retried_step
async def cancel_active_batch(collection: str) -> int:
    """Cancel one sweep of the collection's active work: the bulk index that may still be queueing
    documents, plus a page of collection index workflows. Returns how many were cancelled, so
    the caller sweeps again until a sweep finds nothing."""
    ids = [
        *await _active_ids(INDEX_COLLECTION_WORKFLOW, f"{BULK_INDEX_PREFIX}:{collection}:"),
        *await _active_collection_workflows(collection, limit=CANCEL_PAGE),
    ]
    if ids:
        await DBOS.cancel_workflows_async(ids, cancel_children=True)
    return len(ids)


@retried_step
async def remove_rows(collection: str) -> None:
    """The collection row (memberships cascade) and the name in every session."""
    await Collection(collection).remove_rows()


@retried_step
async def remove_tree(collection: str) -> None:
    """Last: while the folder is there the table can still be deleted again."""
    await Collection(collection).remove_tree()


@DBOS.workflow(name=DELETE_COLLECTION_WORKFLOW)
async def delete_collection_workflow(collection: str) -> None:
    """Cancel every index workflow of the collection, then drop the rows and the folder. No
    document is touched.

    The sweep repeats until it finds nothing: a bulk index still queueing documents can add more
    while the first sweep runs. Each sweep's cancel is final for the status row and for nothing
    else - a step already inside its LanceDB write keeps running - so the removal below waits for
    the collection's write lock, which that step holds until it is done (see `collection_lock`).

    A maintenance run already debounced for this collection is left alone: it finds no row and
    reports itself skipped ("no-collection"), which costs one no-op instead of a cancellation
    race.
    """
    with logs.bound(workflow_id=DBOS.workflow_id, collection=collection):
        while await cancel_active_batch(collection) > 0:
            pass
        async with collection_lock(collection):
            await remove_rows(collection)
            await remove_tree(collection)
        _collection_locks.pop(collection, None)  # nothing may write to it again


# --- retention ----------------------------------------------------------------------------


@retried_step
async def purge_operation_history() -> int:
    """Delete every operation DBOS finished longer than `retention.operation_days` ago, with the
    stage children and step logs below it; returns the cutoff it purged before, as unix ms.

    That history is the whole Operations view, and DBOS removes none of it on its own: without this
    the system database grows with every document, forever. Retried: it is a long series of SQLite
    writes, any of which can lose the file to another writer for a moment, and a second round over
    the same cutoff deletes nothing.

    DBOS's garbage collector is sync SQLAlchemy over the same file and has no async twin, so it
    runs in a worker thread."""
    days = (await load_user_settings()).retention.operation_days
    cutoff = int((time.time() - days * 86400) * 1000)
    await anyio.to_thread.run_sync(
        partial(
            garbage_collect,
            _get_dbos_instance(),
            cutoff_epoch_timestamp_ms=cutoff,
            rows_threshold=None,
        )
    )
    return cutoff


@retried_step
async def prune_audit() -> int:
    """Delete the audit files the retention setting no longer covers. Retried: unlinking a file
    another process still holds is transient, and deleting what is already gone is a no-op."""
    return await audit.prune((await load_user_settings()).retention.audit_days)


@retried_step
async def sweep_staging() -> int:
    """Delete staged uploads nobody imported within `STAGING_TTL_SECONDS`."""
    return await document.sweep_staging(STAGING_TTL_SECONDS)


@DBOS.workflow(name=DAILY_MAINTENANCE_WORKFLOW)
async def daily_maintenance(scheduled_time: datetime, context: Any) -> None:
    """Nightly housekeeping: the operation history, the audit trail and the staging folder.
    Takes the two arguments every DBOS schedule passes."""
    purged_before_ms = await purge_operation_history()
    deleted = await prune_audit()
    swept = await sweep_staging()
    _log.info(
        "home_housekept",
        jobs_purged_before_ms=purged_before_ms,
        audit_files_pruned=deleted,
        staged_uploads_swept=swept,
    )


# --- public API -------------------------------------------------------------------------


async def _start(
    queue: str, workflow: Callable, *args: Any, workflow_id: str, dedup_id: str
) -> str:
    """Enqueue one workflow under an explicit id and return that id.

    Every operation here is deduplicated the same way: a second call made while the first is still
    queued or running returns the run already in flight rather than starting a second one."""
    with (
        SetWorkflowID(workflow_id),
        SetEnqueueOptions(deduplication_id=dedup_id, duplication_policy="return-existing"),
    ):
        handle = await DBOS.enqueue_workflow_async(queue, workflow, *args)
    return handle.workflow_id


# An import runs from a fresh document (`queued`) and from one whose import ended without its
# markdown (`error`, `cancelled`). Any other status means a pipeline - or a delete - is writing
# the same files right now, or that the markdown is already there.
IMPORTABLE: tuple[DocumentStatus, ...] = (
    DocumentStatus.QUEUED,
    DocumentStatus.ERROR,
    DocumentStatus.CANCELLED,
)


async def start_import(doc: str) -> str:
    """Queue the import of one document; a second call while it runs returns the same operation.
    The id names the document, so an operation and its stage children share one prefix.

    The only admission rule for an import, so a re-import goes through here too rather than
    repeating the check at the route."""
    row = await document.get(doc)
    if row.status not in IMPORTABLE:
        raise Conflict(
            f"document is {row.status}; only a queued, failed or cancelled import runs: {doc}"
        )
    return await _start(
        INDEXING_QUEUE,
        import_document,
        doc,
        workflow_id=f"{IMPORT_PREFIX}:{doc}:{uuid4().hex}",
        dedup_id=f"import:{doc}",
    )


async def _enqueue_index(collection: str, doc: str, workflow_id: str | None = None) -> str:
    """Queue the index of one member, with no checks of its own: for a caller that has just read
    the name out of that collection's membership table. `workflow_id` lets a bulk index derive
    the child's id from its own, so that a replay re-attaches to the run it already started."""
    return await _start(
        INDEXING_QUEUE,
        index_collection_document,
        collection,
        doc,
        workflow_id=workflow_id or f"{COLLECTION_DOCUMENT_PREFIX}:{collection}:{doc}:{uuid4().hex}",
        dedup_id=f"index:{collection}:{doc}",
    )


async def start_index_collection_document(collection: str, doc: str) -> str:
    """Queue the index of one member into its collection; a second call while it runs returns the
    same operation. Both names are checked before anything is queued."""
    await (await Collection.get(collection)).member(doc)  # NotFound before anything is queued
    return await _enqueue_index(collection, doc)


async def attach(collection: str, doc: str) -> str:
    """Add an imported document to a collection and queue its index; returns the operation id.
    Only an imported document can be attached, which `Collection.add` enforces: one still
    importing has no markdown to chunk yet, one being deleted must not gain a membership the
    delete's snapshot missed."""
    found = await Collection.get(collection)  # NotFound before anything is written
    await found.add(doc)
    # `_enqueue_index`, not `start_index_collection_document`: `add` just wrote the membership
    return await _enqueue_index(collection, doc)


async def detach(collection: str, doc: str) -> None:
    """Take a document out of one collection and wait for it: cancel its index workflow there,
    then delete its rows and membership from the collection's own partition. The document stays,
    in its folder and in every other collection."""
    await (await Collection.get(collection)).member(doc)  # NotFound before anything is cancelled
    await DBOS.cancel_workflows_async(
        await _active_collection_workflows(collection, doc), cancel_children=True
    )
    with SetEnqueueOptions(queue_partition_key=index_partition(collection)):
        handle = await DBOS.enqueue_workflow_async(
            INDEX_QUEUE, remove_from_collection_index, collection, doc
        )
    await handle.get_result(polling_interval_sec=TASK_POLL)


async def _active_ids(
    name: str | list[str], prefix: str | list[str] | None = None, limit: int | None = None
) -> list[str]:
    """Ids of the workflows of one name (or of any of several) that may still be running, narrowed
    to an id prefix when the caller has one. Ids alone: neither input nor output is loaded."""
    found = await DBOS.list_workflows_async(
        name=name,
        workflow_id_prefix=prefix,
        status=ACTIVE_STATUS,
        limit=limit,
        load_input=False,
        load_output=False,
    )
    return [s.workflow_id for s in found]


async def _active_collection_workflows(
    collection: str, doc: str | None = None, limit: int | None = None
) -> list[str]:
    """Ids of the index workflows of one collection, or of one member, that may still be running.

    The id spells out both names and ends each with `:`, so the prefix selects exactly one
    collection (`idx-col:a:` never matches `idx-col:ab:`) and the database does the filtering."""
    prefix = f"{COLLECTION_DOCUMENT_PREFIX}:{collection}:"
    return await _active_ids(
        COLLECTION_DOCUMENT_WORKFLOW, prefix if doc is None else f"{prefix}{doc}:", limit
    )


async def _active_document_workflows(doc: str) -> list[str]:
    """Every active workflow of one document, whichever collection it runs for.

    A collection index id names the collection before the document, so one prefix query per
    collection the document is in filters them in the database instead of listing every active
    index workflow and splitting the ids here."""
    ids = await _active_ids(  # the import and the embedding runs
        [IMPORT_WORKFLOW, EMBED_WORKFLOW],
        [f"{IMPORT_PREFIX}:{doc}:", f"{EMBED_PREFIX}:{doc}:"],
    )
    for collection in await document.collections_of(doc):
        ids.extend(await _active_collection_workflows(collection, doc))
    return ids


async def start_delete_document(doc: str) -> str:
    """Queue the deletion of a document from everywhere; returns the id of the operation. A second
    call while one runs is deduplicated into it."""
    await document.get(doc)  # NotFound before anything is queued
    return await _start(
        COLLECTION_QUEUE,
        delete_document_workflow,
        doc,
        workflow_id=f"{DELETE_DOCUMENT_PREFIX}:{doc}:{uuid4().hex}",
        dedup_id=f"delete-doc:{doc}",
    )


async def start_index_collection(collection: str) -> str:
    """Queue a (re)index of every member of the collection; returns the id of the operation.

    A second call while one runs is deduplicated into the operation already running, so an
    impatient "Index all" cannot queue the collection twice."""
    await Collection.get(collection)  # NotFound before anything is queued
    return await _start(
        COLLECTION_QUEUE,
        index_collection_workflow,
        collection,
        workflow_id=f"{BULK_INDEX_PREFIX}:{collection}:{uuid4().hex}",
        dedup_id=f"index-collection:{collection}",
    )


async def start_delete_collection(collection: str) -> str:
    """Queue the deletion of the collection; returns the id of the operation.

    On the `operation.collection` queue rather than the collection's index partition: there it
    would wait behind every document it is about to cancel."""
    await Collection.get(collection)  # NotFound before anything is queued
    return await _start(
        COLLECTION_QUEUE,
        delete_collection_workflow,
        collection,
        workflow_id=f"{BULK_DELETE_PREFIX}:{collection}:{uuid4().hex}",
        dedup_id=f"delete-collection:{collection}",
    )


async def cancel_operation(operation_id: str) -> None:
    """Cancel one pipeline operation and record what that left behind: an import stops the
    document, an index stops that one membership, and an embed stops neither - it writes only the
    cache.

    Here rather than in `operations`, which is a read model: this writes, and it reads the names it
    writes by out of the id grammar this module owns (see `pipeline_names`).

    No-op on an operation that already finished: its document status is final."""
    found = await DBOS.get_workflow_status_async(operation_id)
    if found is None:
        raise NotFound(f"operation not found: {operation_id}")
    if found.status not in ACTIVE_STATUS:
        return
    await DBOS.cancel_workflow_async(operation_id, cancel_children=True)
    names = pipeline_names(operation_id)
    if names is None:
        return
    action, collection, doc = names
    if action == PipelineAction.IMPORT:
        await document.set_status(doc, DocumentStatus.CANCELLED)
    elif collection is not None:
        await Collection(collection).set_member_status(doc, MemberStatus.CANCELLED)
