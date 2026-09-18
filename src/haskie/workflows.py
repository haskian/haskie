"""Durable execution with DBOS, on the same SQLite file as the metadata.

DBOS gives us: crash recovery (a workflow resumes at its first unfinished step or child),
step retries with backoff, queues with concurrency and per-partition limits, deduplication,
and a queryable history (workflow list, children, steps). Nothing here keeps state of its own.

Layout:
- `index_document` workflow per document, on `job.indexing`: convert -> embed -> index. Convert and
  embed are cut into at most `indexing.document_parallelism` contiguous slices, one `stage_slice`
  child each, with a durable step per micro-batch. So one large document spreads over the slots
  instead of trickling through a single one, while a document still costs a handful of workflows
  rather than one per micro-batch.
- Each stage has a queue of its own, capped by its share of `indexing.cpu_budget`:
  `task.converting`, `task.embedding`, and `task.indexing` (one writer per library, so it is
  partitioned by library and admits one workflow per partition). A slow stage therefore backs up
  on its own queue instead of taking every slot from the others.
- `remove_document_workflow` per removed document: runs on the same library partition as the index
  stage, so index rows are never deleted while a step of another document writes them.
- `maintain_library` per library, debounced, on `job.maintenance`: compaction and index (re)build,
  handed to the library's index partition. See `maintenance.py`; a burst of documents coalesces
  into one run. The two schedules (hourly archive, nightly housekeeping) sit there too.
- `index_library_workflow` / `delete_library_workflow` per library, on `job.library`: whole-
  library work the request only starts. A library with ten thousand documents costs the caller one
  insert instead of ten thousand, and nothing blocks an HTTP request for minutes (D2).
- Model downloads live in `models.py` (`job.downloads`), the job/task read model in `jobs.py`, and
  the grouped reads of DBOS's own tables it needs in `sysdb.py`.

Every workflow and every step here is `async def`. A queued async workflow is dispatched as a task
on DBOS's background event loop rather than onto its thread pool, so a workflow that only waits on
children costs a task instead of a thread, and every step awaits its IO. `APP_VERSION` is *not*
bumped for the move: DBOS replays a workflow from its recorded step outputs, by step name and
order, neither of which changed, so a run recorded by the sync build replays under the async one.

Two limits on the CPU work, not one, because they answer different questions. The per-queue caps
(`stage_caps`) shape the mix: how the budget is shared out while every stage has work. The
process-wide semaphore (`cpu.cpu_slot`, taken inside `cpu.on_cpu`) is the ceiling, which is what no
queue can enforce, because no queue sees the others: their caps add up to more than the budget
whenever the floor of one slot per stage does (a budget below three), and maintenance runs on a
queue of its own beside all three stages. Every piece of CPU work holds one slot for its length, so
the number of them running at once is never above `indexing.cpu_budget`.

Workflow ids: `idx:{library}:{doc}:{uuid4().hex}` for the parent, `{parent}:{stage}:{slice}` for
each convert or embed child, `{parent}:index` for the index child, `bulk-index:{library}:{uuid}`
and `bulk-delete:{library}:{uuid}` for the two bulk jobs, `maint:{library}:{run}` for a maintenance
run, and `dl:{stage}:{model}` for a model download (see `models`).
`library.safe_name` keeps `:` out of both names, so the prefix is unambiguous: one query finds a
whole job. The child ids are deterministic, so a replay after a crash re-attaches to the child that
already exists instead of starting a second one.
"""

import asyncio
import contextlib
import os
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from functools import partial
from typing import Any, Literal, get_args
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

from haskie import (
    APP_VERSION,
    archive,
    audit,
    db,
    home,
    layout,
    logs,
    maintenance,
    models,
    pipeline,
    sysdb,
)
from haskie.cpu import configure_cpu_budget, shutdown_pool
from haskie.dbos_names import ACTIVE_STATUS, STAGE_WORKFLOW, TERMINAL_STATUS
from haskie.errors import InvalidInput, NotFound, PermanentError
from haskie.library import DocStatus, Library, configure_preview_slots
from haskie.models import root_cause
from haskie.pipeline import Batch
from haskie.settings import (
    ConversionSettings,
    EmbeddingModel,
    PipelineSettings,
    UserSettings,
    load_user_settings,
)

_log = logs.get_logger(__name__)

# Queues. A `job.*` queue carries coarse jobs, which are made of tasks and mostly wait on them; a
# `task.*` queue carries the work itself, and its cap is that stage's share of the CPU budget.
INDEXING_QUEUE = "job.indexing"  # one orchestrating workflow per document, deduplicated
LIBRARY_QUEUE = "job.library"  # whole-library index and delete; unpartitioned (see start_delete)
MAINTENANCE_QUEUE = "job.maintenance"  # debounced maintenance and the two schedules
CONVERT_QUEUE = "task.converting"  # convert slices; cap = the stage's share of the CPU budget
EMBED_QUEUE = "task.embedding"  # embed slices; cap = the stage's share of the CPU budget
INDEX_QUEUE = "task.indexing"  # index children, maintenance and removals; 1 per library partition

MAINTENANCE_CONCURRENCY = 4  # each waits on a child, so this bounds tasks, not LanceDB writers
MAINTENANCE_TIMEOUT_SECONDS = 3600  # compaction of a very large library, not a per-batch budget
LIBRARY_CONCURRENCY = 2  # a whole-library job only enqueues or cancels; two at a time is plenty
DOWNLOAD_CONCURRENCY = 2  # a download is network bound; two at a time saturates any link
DOCUMENT_CONCURRENCY_CAP = 64  # an orchestrator is cheap now, but its children are not; keep a cap
ADOPT_PAGE = 500  # stale workflows resumed per query at boot
BULK_INDEX_PAGE = 500  # documents enqueued per durable page of a bulk index
CANCEL_PAGE = 200  # pipelines cancelled per sweep of a bulk delete

ARCHIVE_SCHEDULE = "archive_jobs"  # cron name; the runs are `sched-archive_jobs-{iso time}`
ARCHIVE_CRON = "10 * * * *"  # hourly, past the hour: retention is the only clock in this app

MAINTENANCE_SCHEDULE = "daily-maintenance"  # housekeeping that costs nothing to skip for a day
MAINTENANCE_CRON = "17 3 * * *"  # nightly, off the hour and off the archive round

BULK_INDEX_PREFIX = "bulk-index"
BULK_DELETE_PREFIX = "bulk-delete"
MAINTAIN_PREFIX = "maint"  # `maint:{library}:{parent}`, so one library's runs are one id prefix
PROGRESS_EVENT = "progress"  # the DBOS event a bulk index publishes after every page

# DBOS on SQLite has no LISTEN/NOTIFY, so queue dequeue and result waits are polls, and DBOS runs
# one polling thread per queue (`dbos._queue.queue_thread`) - seven of them here. The interval is
# therefore paid continuously, idle or not, so the two kinds of queue get different ones.
#
# A `task.*` queue is on the critical path of a document: its interval is added at every stage
# hand-off, so it stays short. A `job.*` queue carries work a user starts and then watches, where
# a second before it is picked up is invisible. DBOS's own default is 1 s for both.
JOB_POLL = float(os.environ.get("HASKIE_JOB_POLL_SECONDS", "1.0"))
TASK_POLL = float(os.environ.get("HASKIE_TASK_POLL_SECONDS", "0.25"))

CANCEL_WAIT_SECONDS = 30.0  # cancellation is cooperative: the running step decides when to stop


async def _wait_until(
    quiet: Callable[[], Awaitable[bool]], wait_seconds: float, **context: Any
) -> None:
    """Poll until `quiet()`, or give up after `wait_seconds` and say so.

    Cancellation only takes effect between steps, and a step that already started still writes its
    output, so everything that cancels waits rather than assumes.
    """
    deadline = time.monotonic() + wait_seconds
    while not await quiet():
        if time.monotonic() >= deadline:
            _log.warning("cancel_wait_timeout", wait_seconds=wait_seconds, **context)
            return
        await anyio.sleep(TASK_POLL)


Stage = Literal["convert", "embed", "index"]
# The order a document moves through them, which is also the order the Jobs view lists its tasks.
STAGE_ORDER: tuple[Stage, ...] = get_args(Stage)
_STAGE_STATUS: dict[Stage, DocStatus] = {
    "convert": "converting",
    "embed": "embedding",
    "index": "indexing",
}
STAGE_QUEUE: dict[Stage, str] = {
    "convert": CONVERT_QUEUE,
    "embed": EMBED_QUEUE,
    "index": INDEX_QUEUE,
}


class Context(msgspec.Struct):
    """Everything a task needs, captured once per workflow so steps stay pure."""

    settings: ConversionSettings
    embedding: EmbeddingModel | None
    batch_pages: int
    index_group_parts: int
    task_timeout_seconds: int
    maintenance_docs: int
    maintenance_idle_seconds: int
    # Slices a convert or embed stage may be cut into, each already resolved against that stage's
    # share of the CPU budget. Last, with defaults, so a context recorded before these fields
    # existed still decodes on a replay; 1 is what those runs did.
    convert_parallelism: int = 1
    embed_parallelism: int = 1


class BatchResult(msgspec.Struct):
    """Outcome of one retried step: a value, or the message of a failure that must not be retried.

    DBOS retries *every* exception raised inside a step with `retries_allowed`, so a deterministic
    failure (unsupported file, OCR policy, corrupt document) is reported as a value and raised by
    the workflow body instead (A9)."""

    value: int | None = None
    permanent_error: str | None = None


class BulkProgress(msgspec.Struct):
    """How far a bulk index got: the `progress` event the workflow publishes after every page.

    `enqueue_page` returns one page in the same shape, with `total` left at 0 and `last` naming
    the document the next page continues after (None once the library is exhausted)."""

    done: int
    skipped: int
    total: int = 0
    last: str | None = None


class BulkResult(msgspec.Struct):
    """Outcome of a bulk index: documents queued, and documents that vanished before that."""

    done: int
    skipped: int


class PipelineError(RuntimeError):
    """Carries only the flat root-cause message (survives DBOS's error (de)serialization)."""


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
        "convert": indexing.converting_weight,
        "embed": indexing.embedding_weight,
        "index": indexing.indexing_weight,
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
    await layout.migrate_layout()  # before DBOS launches: a recovered workflow sees the new layout
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
        "log_level": os.environ.get(logs.LEVEL_VAR, logs.DEFAULT_LEVEL).upper(),
    }
    DBOS(config=config)
    logs.adopt_dbos_logger()  # DBOS installs its own text handler while it initializes
    # In a worker thread on purpose: `launch` is sync SQLAlchemy, and it adopts the loop of the
    # thread that calls it as the one queued async workflows run on. From a thread there is none,
    # so they run on DBOS's own background loop and never share ours (see the plan's "two loops").
    await anyio.to_thread.run_sync(DBOS.launch)
    _start_adoption()
    try:
        await apply_settings(await load_user_settings())
    except InvalidInput as exc:
        # boot must not depend on a settings row another build wrote; the UI can fix it
        _log.error("settings_invalid_at_boot", error=str(exc))
        await apply_settings(UserSettings())
    # after apply_settings: a schedule may only name a queue the system database already carries
    if await _ensure_schedule(ARCHIVE_SCHEDULE, archive_jobs, ARCHIVE_CRON, MAINTENANCE_QUEUE):
        _log.info("schedule_registered", schedule=ARCHIVE_SCHEDULE, cron=ARCHIVE_CRON)
    if await _ensure_schedule(
        MAINTENANCE_SCHEDULE, daily_maintenance, MAINTENANCE_CRON, MAINTENANCE_QUEUE
    ):
        _log.info("schedule_registered", schedule=MAINTENANCE_SCHEDULE, cron=MAINTENANCE_CRON)
    await _prune_audit_at_boot()


async def stop() -> None:
    """Litestar calls a shutdown hook with the app when the hook takes any parameter, so this one
    takes none (A1). In-flight steps get a grace period: a worker thread outliving DBOS blocks
    interpreter exit."""
    global _adoption
    if _adoption is not None:
        adopting, _adoption = _adoption, None
        adopting.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await adopting
    await anyio.to_thread.run_sync(partial(DBOS.destroy, workflow_completion_timeout_sec=10))
    shutdown_pool()  # after DBOS, so nothing is still submitting extraction work


async def _ensure_schedule(name: str, workflow: Callable, cron: str, queue: str) -> bool:
    """Register a cron schedule, once. DBOS keeps schedules in the system database, so they
    outlive the process: every boot after the first finds this one already there.

    Returns True when this boot wrote the definition, which is the first boot and any boot whose
    build changed the cron or the queue."""
    found = await DBOS.get_schedule_async(name)
    if found is not None and (found["schedule"], found.get("queue_name")) == (cron, queue):
        return False
    # an upsert, not `create_schedule`: that one raises on a name the database already carries
    await DBOS.apply_schedules_async(
        [{"schedule_name": name, "workflow_fn": workflow, "schedule": cron, "queue_name": queue}]
    )
    return True


async def _prune_audit_at_boot() -> None:
    """Also prune once per boot: a desktop app is rarely running at 03:17, so a run that only ever
    happened on the schedule would never happen at all. Housekeeping, so a failure is logged and
    the boot continues."""
    try:
        deleted = await prune_audit()
    except Exception as exc:
        _log.error("audit_prune_failed", error=root_cause(exc))
        return
    if deleted:
        _log.info("audit_files_pruned", deleted=deleted)


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


async def _adopt() -> None:
    """`adopt_orphans` with the boot's logging, and nothing above it to catch what it raises."""
    try:
        adopted = await adopt_orphans()
    except Exception as exc:
        _log.error("stale_workflow_adoption_failed", error=root_cause(exc))
        return
    if adopted:
        _log.warning("stale_workflows_resumed", count=adopted, app_version=APP_VERSION)


def _start_adoption() -> None:
    """Adopting a long backlog takes as long as the backlog is deep, and nothing waits for its
    result: the boot hands it to a task on the loop it runs on and returns (see `_adoption`)."""
    global _adoption
    _adoption = asyncio.get_running_loop().create_task(_adopt(), name="haskie-adopt")


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
    await DBOS.register_queue_async(
        INDEXING_QUEUE,
        global_concurrency=document_concurrency(indexing),
        polling_interval_sec=JOB_POLL,
    )
    await DBOS.register_queue_async(
        LIBRARY_QUEUE, global_concurrency=LIBRARY_CONCURRENCY, polling_interval_sec=JOB_POLL
    )
    await DBOS.register_queue_async(
        models.DOWNLOADS_QUEUE,
        global_concurrency=DOWNLOAD_CONCURRENCY,
        polling_interval_sec=JOB_POLL,
    )
    await DBOS.register_queue_async(
        MAINTENANCE_QUEUE, global_concurrency=MAINTENANCE_CONCURRENCY, polling_interval_sec=JOB_POLL
    )
    await DBOS.register_queue_async(
        CONVERT_QUEUE, global_concurrency=caps["convert"], polling_interval_sec=TASK_POLL
    )
    await DBOS.register_queue_async(
        EMBED_QUEUE, global_concurrency=caps["embed"], polling_interval_sec=TASK_POLL
    )
    await DBOS.register_queue_async(
        INDEX_QUEUE,
        global_concurrency=caps["index"],
        partition_concurrency=1,  # LanceDB takes one writer per library (see `_partition_key`)
        polling_interval_sec=TASK_POLL,
    )
    await models.ensure_models(settings)
    await schedule_pending_maintenance()


# --- steps (pure, retried) ----------------------------------------------------------

# Every step here touches SQLite or the file system, so every step may hit a transient lock.
# The wait before a retry is a real one, so the test suite shortens it through the environment.
RETRY_INTERVAL_SECONDS = float(os.environ.get("HASKIE_RETRY_INTERVAL_SECONDS", "1.0"))
retried_step = DBOS.step(
    retries_allowed=True,
    max_attempts=3,
    interval_seconds=RETRY_INTERVAL_SECONDS,
    backoff_rate=2.0,
)


@retried_step
async def load_context(library: str) -> Context:
    lib = await Library.get(library)
    user = await load_user_settings()
    return Context(
        settings=await lib.effective_settings(),
        embedding=user.embedding_model,
        batch_pages=user.pipeline.batch_pages,
        index_group_parts=user.pipeline.index_group_parts,
        task_timeout_seconds=user.pipeline.task_timeout_seconds,
        maintenance_docs=user.pipeline.maintenance_docs,
        maintenance_idle_seconds=user.pipeline.maintenance_idle_seconds,
        convert_parallelism=resolve_parallelism(user.pipeline, "convert"),
        embed_parallelism=resolve_parallelism(user.pipeline, "embed"),
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
async def set_status(library: str, doc: str, status: DocStatus, error: str | None = None) -> None:
    await Library(library).set_status(doc, status, error)


@retried_step
async def plan(stage: Stage, library: str, doc: str, ctx: Context) -> list[Batch]:
    lib = Library(library)
    if stage == "convert":
        return await pipeline.plan_convert(lib, doc, ctx.batch_pages)
    if stage == "embed":
        return await pipeline.plan_embed(lib, doc)
    return await pipeline.plan_index(lib, doc, ctx.index_group_parts)


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


@retried_step
async def try_batch(
    stage: Stage, library: str, doc: str, batch: Batch, ctx: Context
) -> BatchResult:
    """The CPU work of one micro-batch. The slot of the CPU budget is taken inside `cpu.on_cpu`,
    around the CPU work alone: the file reads, the file writes and the LanceDB commit of the same
    batch await without holding it, and neither does the bookkeeping DBOS does around the step."""
    lib = Library(library)
    if stage == "convert":
        return await _guarded(pipeline.convert_batch(lib, doc, batch, ctx.settings))
    if stage == "embed":
        return await _guarded(pipeline.embed_batch(lib, doc, batch, ctx.settings, ctx.embedding))
    return await _guarded(pipeline.index_batch(lib, doc, batch, ctx.embedding))


@retried_step
async def try_finalize_convert(
    library: str, doc: str, batches: list[Batch], ocr_total: int, ctx: Context
) -> BatchResult:
    """Assemble the markdown out of every part the convert slices wrote. A step of the parent
    workflow, not of a child: it needs the OCR counts of all slices, which only the parent has."""

    async def finalize() -> None:
        await pipeline.finalize_convert(Library(library), doc, batches, ocr_total, ctx.settings)

    return await _guarded(finalize())


@retried_step
async def finalize_index(library: str, doc: str, ctx: Context) -> None:
    """Rebuild the full-text index once, on the library's partition (A2)."""
    await pipeline.finalize_index(Library(library), doc, ctx.embedding)


@retried_step
async def note_indexed_step(library: str, doc: str) -> int:
    """Count one more indexed document for the library; returns how many await maintenance."""
    pending = await Library(library).note_indexed()
    _log.debug("index_pending", library=library, doc=doc, pending=pending)
    return pending


@retried_step
async def claim_pending(library: str) -> int:
    # nothing is reset here: a run that crashes must leave the library pending, and documents
    # indexed while it runs must still count towards the next one (see `settle_maintenance`)
    state = await Library(library).maintenance_state()
    return state.pending_docs if state else 0


@retried_step
async def run_maintenance(library: str) -> maintenance.Report:
    """`Library(name)`, not `Library.get(name)`: the library may have been deleted while this run
    waited, and `maintenance.run` reports that as a skip rather than a failure.

    No slot of the CPU budget is taken around it: compaction and the vector index build are CPU,
    but they run inside LanceDB's own runtime rather than in a worker thread of ours, so there is
    nothing for `cpu.cpu_slot` to hold. `task.indexing` bounds them instead - one writer per
    library partition (see `maintenance.run`)."""
    user = await load_user_settings()
    return await maintenance.run(Library(library), user.embedding_model, user.pipeline)


@retried_step
async def settle_maintenance(library: str, claimed: int, report: maintenance.Report) -> None:
    await Library(library).settle_maintenance(claimed, report.ann_trained, report.num_rows)


@retried_step
async def cleanup_parts(library: str, doc: str) -> None:
    """Drop the micro-batch files the index stage just consumed. Last step of the stage, so a
    replayed index step still finds the `rows.json` it needs."""
    await pipeline.cleanup_parts(Library(library), doc)


async def run_batch(stage: Stage, library: str, doc: str, batch: Batch, ctx: Context) -> int:
    return _value(await try_batch(stage, library, doc, batch, ctx))


# --- workflows --------------------------------------------------------------------------


@DBOS.workflow(name=STAGE_WORKFLOW)
async def stage_slice(
    stage: Stage, library: str, doc: str, batches: list[Batch], ctx: Context
) -> list[int]:
    """One slice of one stage of one document: a durable step per micro-batch, in plan order.

    The batches of a slice run one after another, because the steps of a DBOS workflow do; the
    parallelism inside a document comes from `_stage` running several of these at once. The index
    stage is never sliced, so this is also where its finalizer and the cleanup of the part files
    belong: both must run once, after the last index batch of the document."""
    results = [await run_batch(stage, library, doc, batch, ctx) for batch in batches]
    if stage == "index":
        await finalize_index(library, doc, ctx)
        await cleanup_parts(library, doc)
    return results


def stage_input(child) -> tuple[Stage, list[Batch]] | None:
    """The stage and the micro-batches a `stage_slice` child was given, read back out of its
    recorded input; None when DBOS did not keep it.

    Here rather than in `jobs`, so the argument positions and the signature they index into are
    edited in one place."""
    args = child.input["args"] if child.input else None
    return (args[0], args[3]) if args else None


def _partition_key(stage: Stage, library: str) -> str | None:
    """Index writes are serialized per library: LanceDB takes one writer at a time, so every index
    child, maintenance run and removal of one library shares its partition, and `task.indexing`
    admits one workflow per partition.

    None for a convert or embed slice: their queues are capped globally and hold nothing a slice
    of the same document could corrupt, so a partition would only be a second limit to keep."""
    return f"index:{library}" if stage == "index" else None


def _child_id(stage: Stage, slice_index: int) -> str:
    """Deterministic, so a replay after a crash re-attaches to the child that already exists. The
    index stage has exactly one child, and keeps the unsuffixed id it always had."""
    if stage == "index":
        return f"{DBOS.workflow_id}:index"
    return f"{DBOS.workflow_id}:{stage}:{slice_index}"


def _slice_count(stage: Stage, ctx: Context) -> int:
    """Slices this stage may be cut into. The index stage is never sliced: it is one writer."""
    if stage == "convert":
        return ctx.convert_parallelism
    return ctx.embed_parallelism if stage == "embed" else 1


def _slices(batches: list[Batch], parts: int) -> list[list[Batch]]:
    """Cut the batches into at most `parts` contiguous runs of near-equal length.

    Contiguous rather than round-robin: consecutive micro-batches are consecutive part files, so
    one worker walks one region of the document. Always at least one run, even for a stage that
    planned no batches at all, because the run is also what carries the stage's finalizer."""
    count = max(1, min(len(batches), parts))
    size, extra = divmod(len(batches), count)
    out: list[list[Batch]] = []
    start = 0
    for index in range(count):
        end = start + size + (1 if index < extra else 0)
        out.append(batches[start:end])
        start = end
    return out


async def _stage(stage: Stage, library: str, doc: str, ctx: Context) -> list[int]:
    """Plan the stage, run its slices as child workflows side by side, wait for all of them.

    Returns the per-batch results in plan order: the slices are contiguous, so concatenating them
    in slice order restores it. The convert finalizer runs here rather than in a child, because it
    needs the OCR counts of every slice, which only the parent sees."""
    await set_status(library, doc, _STAGE_STATUS[stage])
    batches = await plan(stage, library, doc, ctx)
    handles: list[WorkflowHandleAsync[list[int]]] = []
    for index, batches_of_slice in enumerate(_slices(batches, _slice_count(stage, ctx))):
        with (
            SetWorkflowID(_child_id(stage, index)),
            # the budget is per batch, and the child runs them all: a long slice gets
            # proportionally longer rather than timing out for being long
            SetWorkflowTimeout(ctx.task_timeout_seconds * max(1, len(batches_of_slice))),
            SetEnqueueOptions(queue_partition_key=_partition_key(stage, library)),
        ):
            # the context managers wrap the await itself: `enqueue_workflow_async` reads the
            # contextvars they set before it yields to the loop
            handles.append(
                await DBOS.enqueue_workflow_async(
                    STAGE_QUEUE[stage], stage_slice, stage, library, doc, batches_of_slice, ctx
                )
            )
    results = [
        result
        for handle in handles
        for result in await handle.get_result(polling_interval_sec=TASK_POLL)
    ]
    if stage == "convert":
        _value(await try_finalize_convert(library, doc, batches, sum(results), ctx))
    return results


@DBOS.workflow()
async def index_document(library: str, doc: str) -> str:
    """convert -> embed -> index for one document; document status mirrors the stage."""
    started = time.perf_counter()
    with logs.bound(workflow_id=DBOS.workflow_id, library=library, doc=doc):
        # first step, so a deduplicated submit changes nothing
        await set_status(library, doc, "queued")
        ctx = await load_context(library)
        try:
            for stage in ("convert", "embed", "index"):
                await _stage(stage, library, doc, ctx)
            pending = await note_indexed_step(library, doc)
            await request_maintenance(
                library, pending, ctx.maintenance_docs, ctx.maintenance_idle_seconds
            )
        except Exception as exc:
            message = str(exc) if isinstance(exc, PermanentError) else root_cause(exc)
            await set_status(library, doc, "error", message)
            await _record(library, doc, "index.failed", started, message)
            raise PipelineError(message) from exc
        await set_status(library, doc, "indexed")
        await _record(library, doc, "index.completed", started, None)
        return "indexed"


async def _record(library: str, doc: str, event: str, started: float, error: str | None) -> None:
    """One audit line per finished document. Not a step: a replay after a crash re-appends it,
    which an append-only trail tolerates."""
    await audit.record(
        event,
        actor="workflow",
        outcome="ok" if error is None else "error",
        duration_ms=int((time.perf_counter() - started) * 1000),
        workflow_id=DBOS.workflow_id,
        library=library,
        doc=doc,
        error=error,
    )


@DBOS.workflow()
async def maintain_on_partition(library: str) -> maintenance.Report:
    """The maintenance itself, on the library's index partition: LanceDB takes one writer at a
    time, so compaction waits for the index stage of any document in flight, and vice versa."""
    with logs.bound(workflow_id=DBOS.workflow_id, library=library):
        claimed = await claim_pending(library)
        report = await run_maintenance(library)
        await settle_maintenance(library, claimed, report)
        return report


@DBOS.workflow()
async def maintain_library(library: str) -> maintenance.Report:
    """Debounced entry point. It only hands the work to the library's index partition and waits.

    Two hops because a debounce needs deduplication, and a partitioned queue does not support it:
    this one sits on the unpartitioned `job.maintenance` queue, the work it enqueues on the
    partition every index write already uses.

    The child's id names the library and this run, so the jobs view filters maintenance by library
    with an id prefix like every other listing, and a replay re-attaches to the child that already
    exists instead of starting a second one."""
    with (
        SetWorkflowTimeout(MAINTENANCE_TIMEOUT_SECONDS),
        SetEnqueueOptions(queue_partition_key=f"index:{library}"),
        SetWorkflowID(f"{MAINTAIN_PREFIX}:{library}:{DBOS.workflow_id}"),
    ):
        handle = await DBOS.enqueue_workflow_async(INDEX_QUEUE, maintain_on_partition, library)
    return await handle.get_result(polling_interval_sec=TASK_POLL)


# The debounce key is the library name, so a burst of documents coalesces into one run: each
# request pushes the delay out, and the run starts once the burst stops (or `maintenance_docs`
# documents landed, which requests it with no delay at all).
MAINTAIN = Debouncer.create_async(maintain_library, queue=MAINTENANCE_QUEUE)


async def request_maintenance(
    library: str, pending: int, after_docs: int, idle_seconds: int
) -> None:
    """Ask for a maintenance run: now once `after_docs` documents piled up, otherwise once the
    library has been idle for `idle_seconds`.

    Must be called with no `SetEnqueueOptions` partition key in context: a debounce deduplicates,
    and DBOS rejects deduplication on a partitioned enqueue."""
    period = 0.0 if pending >= after_docs else float(idle_seconds)
    await MAINTAIN.debounce_async(library, period, library)


async def schedule_pending_maintenance() -> None:
    """Reschedule every library that has documents pending. A run lost to a crash or a shutdown
    leaves `pending_docs` standing, so the next boot picks the library up again."""
    idle = float((await load_user_settings()).pipeline.maintenance_idle_seconds)
    for name in await Library.pending_names():
        await MAINTAIN.debounce_async(name, idle, name)


@retried_step
async def remove_index_rows(library: str, doc: str) -> None:
    await Library(library).remove_index_rows(doc)


@retried_step
async def remove_files(library: str, doc: str) -> None:
    await Library(library).remove_files(doc)


@retried_step
async def remove_row(library: str, doc: str) -> None:
    """Last: while the row exists the document is still listed, so a crash leaves no phantom."""
    await Library(library).remove_row(doc)


@DBOS.workflow()
async def remove_document_workflow(library: str, doc: str) -> None:
    with logs.bound(workflow_id=DBOS.workflow_id, library=library, doc=doc):
        await remove_index_rows(library, doc)
        await remove_files(library, doc)
        await remove_row(library, doc)


# --- whole-library jobs -------------------------------------------------------------------


@retried_step
async def count_documents(library: str) -> int:
    """Only feeds the progress event, so a library that is already gone counts as empty."""
    return (await Library(library).counts()).total


@retried_step
async def document_page(library: str, after: str | None) -> list[str]:
    """One keyset page of document names, ordered by name. Empty once the library is exhausted,
    and also when it was deleted while the bulk index ran, which ends the walk either way."""
    return await Library(library).document_names(after, BULK_INDEX_PAGE)


async def enqueue_page(library: str, after: str | None, bulk_id: str) -> BulkProgress:
    """Queue the pipeline for one page of documents and report what that did.

    Not a step: DBOS refuses to start a workflow inside one. The listing above is the step, and
    its recorded output is what makes a replay walk the same names in the same order. Each child
    gets an id derived from the bulk job, so a replay re-attaches to the workflow it already
    started; a document indexing under an older id is returned by the deduplication in
    `start_index` instead of being queued twice.
    """
    done = skipped = 0
    last: str | None = None
    for doc in await document_page(library, after):
        last = doc
        try:
            await start_index(library, doc, workflow_id=f"idx:{library}:{doc}:{bulk_id[-32:]}")
            done += 1
        except NotFound:  # removed between the listing and the enqueue
            skipped += 1
    return BulkProgress(done=done, skipped=skipped, last=last)


@DBOS.workflow()
async def index_library_workflow(library: str) -> BulkResult:
    """(Re)index every document of one library, one durable page of enqueues at a time."""
    with logs.bound(workflow_id=DBOS.workflow_id, library=library):
        bulk_id = DBOS.workflow_id or ""
        total = await count_documents(library)
        done = skipped = 0
        after: str | None = None
        while True:
            page = await enqueue_page(library, after, bulk_id)
            done, skipped, after = done + page.done, skipped + page.skipped, page.last
            await DBOS.set_event_async(PROGRESS_EVENT, BulkProgress(done, skipped, total, after))
            if after is None:
                _log.info("library_index_queued", library=library, done=done, skipped=skipped)
                return BulkResult(done=done, skipped=skipped)


@retried_step
async def cancel_active_batch(library: str) -> int:
    """Cancel one sweep of the library's active work: the bulk index that may still be queueing
    documents, plus a page of document pipelines. Returns how many were cancelled, so the caller
    sweeps again until a sweep finds nothing."""
    ids = [
        *await _active_bulk_index(library),
        *await _active_index_workflows(library, limit=CANCEL_PAGE),
    ]
    if ids:
        await DBOS.cancel_workflows_async(ids, cancel_children=True)
    return len(ids)


@retried_step
async def wait_quiet(library: str) -> None:
    """Wait until no pipeline of the library is active any more. A bulk delete cannot name the
    ids it is waiting for -- more keep arriving while it sweeps -- so it asks by prefix."""

    async def quiet() -> bool:
        return not await _active_index_workflows(library, limit=1)

    await _wait_until(quiet, CANCEL_WAIT_SECONDS, library=library)


@retried_step
async def remove_rows(library: str) -> None:
    """The library row (documents cascade) and the name in every session."""
    await Library(library).remove_rows()


@retried_step
async def remove_tree(library: str) -> None:
    """Last: while the folder is there the files can still be deleted again (see Library.delete)."""
    await Library(library).remove_tree()


@DBOS.workflow()
async def delete_library_workflow(library: str) -> None:
    """Cancel every pipeline of the library, wait for the last running step, then drop the rows
    and the folder (A5).

    A maintenance run already debounced for this library is left alone: it finds no row and
    reports itself skipped ("no-library"), which costs one no-op instead of a cancellation race.
    """
    with logs.bound(workflow_id=DBOS.workflow_id, library=library):
        while await cancel_active_batch(library) > 0:
            pass
        await wait_quiet(library)
        await remove_rows(library)
        await remove_tree(library)


# --- retention ----------------------------------------------------------------------------


@retried_step
async def archive_step() -> archive.ArchiveReport:
    """One retention round. Retried: it is a long series of SQLite writes, any of which can lose
    the file to another writer for a moment, and the round is idempotent."""
    retention = (await load_user_settings()).retention
    return await archive.archive_once(int(time.time() * 1000), retention)


@DBOS.workflow()
async def archive_jobs(scheduled_time: datetime, context: Any) -> None:
    """Hourly: copy finished jobs into their day partition, then let DBOS drop what it no longer
    has to keep (see `archive`). Takes the two arguments every DBOS schedule passes.

    Two runs overlapping needs no coordination: the copy replaces rows it already wrote and the
    watermark only moves forward, so a round that outlives its hour costs work, never rows."""
    report = await archive_step()
    _log.info(
        "jobs_archived",
        copied=report.copied,
        purged_before_ms=report.purged_before_ms,
        dropped=report.dropped,
    )


@retried_step
async def prune_audit() -> int:
    """Delete the audit files the retention setting no longer covers. Retried: unlinking a file
    another process still holds is transient, and deleting what is already gone is a no-op."""
    return await audit.prune((await load_user_settings()).retention.audit_days)


@DBOS.workflow()
async def daily_maintenance(scheduled_time: datetime, context: Any) -> None:
    """Nightly housekeeping of the home directory. Takes the two arguments every DBOS schedule
    passes. Only the audit trail so far; job history is archived hourly instead."""
    deleted = await prune_audit()
    _log.info("audit_files_pruned", deleted=deleted)


# --- public API -------------------------------------------------------------------------


async def start_index(library: str, doc: str, workflow_id: str | None = None) -> str:
    """Queue the pipeline for one document; a second call while it runs returns the same job.

    The id names the library and the document, so a job and its stage children share one prefix
    and read as a tree in the DBOS history. A bulk index passes its own id, derived from the bulk
    job, so that a replay re-attaches rather than queueing the document a second time."""
    lib = await Library.get(library)
    await lib.document(doc)
    with (
        SetWorkflowID(workflow_id or f"idx:{library}:{doc}:{uuid4().hex}"),
        SetEnqueueOptions(
            deduplication_id=f"{library}:{doc}", duplication_policy="return-existing"
        ),
    ):
        handle = await DBOS.enqueue_workflow_async(INDEXING_QUEUE, index_document, library, doc)
    return handle.workflow_id


async def _active_index_workflows(
    library: str, doc: str | None = None, limit: int | None = None
) -> list[str]:
    """Ids of the index workflows of one library, or of one document, that may still be running.

    The id spells out both names and ends each with `:`, so the prefix selects exactly one library
    (`idx:a:` never matches `idx:ab:`) and the database does the filtering, not this process."""
    prefix = f"idx:{library}:" if doc is None else f"idx:{library}:{doc}:"
    found = await DBOS.list_workflows_async(
        name=index_document.__qualname__,
        workflow_id_prefix=prefix,
        status=ACTIVE_STATUS,
        limit=limit,
        load_input=False,
        load_output=False,
    )
    return [s.workflow_id for s in found]


async def _active_bulk_index(library: str) -> list[str]:
    """Ids of the bulk indexes of one library that may still be queueing documents."""
    found = await DBOS.list_workflows_async(
        name=index_library_workflow.__qualname__,
        workflow_id_prefix=f"{BULK_INDEX_PREFIX}:{library}:",
        status=ACTIVE_STATUS,
        load_input=False,
        load_output=False,
    )
    return [s.workflow_id for s in found]


async def _cancel_and_wait(workflow_ids: list[str], wait_seconds: float) -> None:
    """Cancel, then wait until every one of those workflows is terminal."""
    for workflow_id in workflow_ids:
        await DBOS.cancel_workflow_async(workflow_id, cancel_children=True)
    pending = list(workflow_ids)

    async def quiet() -> bool:
        if not pending:  # nothing left to wait for; an empty id list is not "every workflow"
            return True
        found = await DBOS.list_workflows_async(
            workflow_ids=pending, load_input=False, load_output=False
        )
        # in place, so the list the timeout logs is the one still pending
        pending[:] = [s.workflow_id for s in found if s.status not in TERMINAL_STATUS]
        return not pending

    await _wait_until(quiet, wait_seconds, workflows=pending)


async def cancel_document(
    library: str, doc: str, wait_seconds: float = CANCEL_WAIT_SECONDS
) -> None:
    """Stop the pipeline of one document and wait until nothing writes its files any more."""
    await _cancel_and_wait(await _active_index_workflows(library, doc), wait_seconds)


async def remove_document(library: str, doc: str) -> None:
    """Cancel, wait, then delete index rows, files and the row from the library's index partition,
    so nothing else writes the document while it is removed (A3/A4)."""
    lib = await Library.get(library)
    await lib.document(doc)  # DocumentNotFound before anything is cancelled
    await cancel_document(library, doc)
    with SetEnqueueOptions(queue_partition_key=f"index:{library}"):
        handle = await DBOS.enqueue_workflow_async(
            INDEX_QUEUE, remove_document_workflow, library, doc
        )
    await handle.get_result(polling_interval_sec=TASK_POLL)


async def start_index_library(library: str) -> str:
    """Queue a (re)index of every document of the library; returns the id of the bulk job.

    A second call while one runs is deduplicated into the job already running, so an impatient
    "Index all" cannot queue the library twice."""
    await Library.get(library)  # LibraryNotFound before anything is queued
    with (
        SetWorkflowID(f"{BULK_INDEX_PREFIX}:{library}:{uuid4().hex}"),
        SetEnqueueOptions(
            deduplication_id=f"index-lib:{library}", duplication_policy="return-existing"
        ),
    ):
        handle = await DBOS.enqueue_workflow_async(LIBRARY_QUEUE, index_library_workflow, library)
    return handle.workflow_id


async def start_delete_library(library: str) -> str:
    """Queue the deletion of the library; returns the id of the bulk job.

    On the `job.library` queue rather than the library's index partition: there it would wait
    behind every document it is about to cancel."""
    await Library.get(library)  # LibraryNotFound before anything is queued
    with (
        SetWorkflowID(f"{BULK_DELETE_PREFIX}:{library}:{uuid4().hex}"),
        SetEnqueueOptions(
            deduplication_id=f"delete-lib:{library}", duplication_policy="return-existing"
        ),
    ):
        handle = await DBOS.enqueue_workflow_async(LIBRARY_QUEUE, delete_library_workflow, library)
    return handle.workflow_id


async def delete_library(library: str) -> None:
    """Delete the library and wait for it, for callers that must see it gone when they return."""
    handle = await DBOS.retrieve_workflow_async(await start_delete_library(library))
    await handle.get_result(polling_interval_sec=TASK_POLL)
