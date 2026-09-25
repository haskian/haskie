"""Model lifecycle: one DBOS workflow per model the settings require.

Downloaded and usable are two different things, and this module keeps them apart.

*Downloaded* is durable: `ensure_model` has one fixed id per model (`dl:{kind}:{name}`), so it is
idempotent, retried on failure, queryable, and it outlives the process — the files stay in the
disk cache, so a restart must not fetch them again (which is what a boot id in the id used to
cost: one record, one download and one row in the Downloads list per process start).

*Usable* is per process: a model lives in the caches of one process, so a boot that finds a
SUCCESS record still has cold caches. `_ready` holds the ids this process has loaded,
`ensure_models` warms the rest in a background task (a local read, no network), and
`require_ready` — which every search calls — answers from `_ready`, not from the record alone.

Loading a model is CPU work, not IO, so it goes through `cpu.on_cpu`: a worker thread, under one
slot of the CPU budget, whichever event loop asked for it.

Depends on leaf modules only (`cpu`, `embed`, `settings`, `catalogue`, `dbos_names`): `index`
and `pipeline` import this module, so it must not reach back into them or into `workflows`.
`collection` is read through a function-local import for the same reason (see
`_collection_rerankers`).
"""

import asyncio
import threading
from enum import StrEnum

import msgspec
from dbos import DBOS, SetWorkflowID
from dbos import WorkflowStatus as DbosWorkflowStatus

from haskie import cpu
from haskie.catalogue import catalogue
from haskie.errors import HaskieError, NotReady
from haskie.indexing import embed, hardware
from haskie.indexing.dbos_names import ACTIVE_STATUS, DOWNLOAD_WORKFLOW, RunStatus, root_cause
from haskie.indexing.hardware import Device
from haskie.logs import get_logger
from haskie.settings import Reranker, UserSettings, load_user_settings

_log = get_logger(__name__)

DOWNLOADS_QUEUE = "operation.downloads"

# Ids of the `ensure_model` workflows whose model is loaded in *this* process, and the ids a warm
# task is loading right now. `_warm_lock` guards the pair, so no model is warmed twice at once. A
# threading lock, because both event loops of this process read and write the pair; nothing ever
# awaits while holding it.
_ready: set[str] = set()
_warming: set[str] = set()
_warm_lock = threading.Lock()

# Strong references to the warm tasks in flight: an event loop keeps only a weak one, so a task
# nobody holds may be collected mid-load. Each discards itself when it finishes.
_warm_tasks: set[asyncio.Task[None]] = set()


class ModelKind(StrEnum):
    EMBEDDING = "embedding"
    RERANKER = "reranker"


class ModelState(StrEnum):
    PENDING = "pending"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class ModelStatus(msgspec.Struct):
    kind: ModelKind
    name: str
    state: ModelState
    error: str | None = None
    # where it runs under the hardware setting (`hardware.device`); None where it cannot
    device: Device | None = None


# A download is retried with longer waits than a local step (see
# `workflows.RETRY_INTERVAL_SECONDS`, which the same note about import time applies to).
DOWNLOAD_RETRY_INTERVAL_SECONDS = 5.0


async def warm_model(kind: ModelKind, name: str) -> None:
    """Load one model into this process's caches. Fetches it when the disk cache is cold, so a
    call made after the record says SUCCESS is a local read.

    The load itself is CPU (and, on a cold cache, a download inside fastembed), so it runs in a
    worker thread under one slot of the CPU budget rather than on the caller's loop."""
    accelerator = (await load_user_settings()).pipeline.accelerator
    warm = embed.warm_reranker if kind == ModelKind.RERANKER else embed.warm
    await cpu.on_cpu(warm, name, accelerator)


@DBOS.step(
    retries_allowed=True,
    max_attempts=5,
    interval_seconds=DOWNLOAD_RETRY_INTERVAL_SECONDS,
    backoff_rate=2.0,
)
async def load_model(kind: ModelKind, name: str) -> None:
    """A download over the network: more attempts and longer waits than a local pipeline step."""
    await warm_model(kind, name)


@DBOS.workflow(name=DOWNLOAD_WORKFLOW)
async def ensure_model(kind: ModelKind, name: str) -> ModelState:
    try:
        await load_model(kind, name)
    except Exception as exc:
        # flat message and a one-argument class: DBOS stores and rebuilds it without our traceback
        raise HaskieError(root_cause(exc)) from exc
    _mark_ready(_model_id(kind, name))  # the download ran here, so this process can search with it
    return ModelState.READY


async def _required(settings: UserSettings) -> list[tuple[ModelKind, str]]:
    """Every model this installation needs, in a stable order and without duplicates.

    A collection may override the reranker model, and a search of that collection then loads it,
    so the overrides count as required as much as the user-level pair does."""
    wanted: list[tuple[ModelKind, str]] = []
    embedding = await catalogue.embedding_model(settings)
    if embedding:
        wanted.append((ModelKind.EMBEDDING, embedding.name))
    if settings.search.reranker == Reranker.CROSS_ENCODER:
        wanted.append((ModelKind.RERANKER, settings.search.reranker_model))
    wanted.extend((ModelKind.RERANKER, name) for name in await _collection_rerankers())
    return list(dict.fromkeys(wanted))


async def _collection_rerankers() -> list[str]:
    """Reranker models the collections override. Imported here rather than at module level: the
    dependency runs `collection` -> `index` -> `models`."""
    from haskie.collection.collection import Collection

    return await Collection.reranker_overrides()


def _model_id(kind: ModelKind, name: str) -> str:
    """One durable record per model, whatever process asks for it (see the module docstring)."""
    return f"dl:{kind}:{name}"


def model_names(workflow_id: str) -> tuple[str, str]:
    """The inverse of `_model_id`: `dl:{kind}:{model}` -> (kind, model); ("?", workflow_id) for
    any other id. Here so the grammar is written and read in one place.

    The model name is the rest of the id, colons and all, so a name that carries one still reads
    back whole."""
    parts = workflow_id.split(":", 2)
    if len(parts) != 3 or parts[0] != "dl":
        return "?", workflow_id
    return parts[1], parts[2]


async def _download_records(
    wanted: list[tuple[ModelKind, str]],
) -> dict[str, DbosWorkflowStatus]:
    """The download record of every model in `wanted`, in one query, keyed by workflow id. A
    model nobody ever asked for simply has none. The output is loaded because DBOS carries a
    workflow's error alongside it, and `_model_status` reports that error."""
    if not wanted:
        return {}
    ids = [_model_id(kind, name) for kind, name in wanted]
    return {s.workflow_id: s for s in await DBOS.list_workflows_async(workflow_ids=ids)}


async def ensure_models(settings: UserSettings) -> list[ModelStatus]:
    """Idempotent: a model that is downloaded is not downloaded again, a failed one is retried,
    and one this process has not loaded yet is warmed in the background.

    One query for every record, not one per model: the same read decides what to enqueue and
    answers the statuses this returns."""
    wanted = await _required(settings)
    records = await _download_records(wanted)
    for kind, name in wanted:
        workflow_id = _model_id(kind, name)
        existing = records.get(workflow_id)
        status = existing.status if existing else None
        if status == RunStatus.SUCCESS:
            _warm_in_background(kind, name)  # on disk already; this process's caches may be cold
            continue
        if status in ACTIVE_STATUS:
            continue  # on its way: the record is the download
        if status is not None:
            # DBOS counts ERROR as complete, so resuming such a record does nothing: drop it and
            # enqueue the same id again, which is what a retry means for a download.
            _log.warning("model_load_retry", kind=kind, model=name, status=status)
            await DBOS.delete_workflow_async(workflow_id)
        # the context manager wraps the await itself: it sets a contextvar the enqueue reads
        with SetWorkflowID(workflow_id):
            handle = await DBOS.enqueue_workflow_async(DOWNLOADS_QUEUE, ensure_model, kind, name)
        records[workflow_id] = await handle.get_status()  # the record this call just wrote
    return _statuses(wanted, records, settings)


def _warm_in_background(kind: ModelKind, name: str) -> None:
    """Load a downloaded model into this process's caches, off the caller's path: `ensure_models`
    runs on the boot path and inside a settings request, while loading a model costs seconds.

    One task per model at a time: a second call while the first is still loading does nothing.
    Sync, and called only from a coroutine: the task belongs to the loop that asked for it."""
    workflow_id = _model_id(kind, name)
    with _warm_lock:
        if workflow_id in _ready or workflow_id in _warming:
            return
        _warming.add(workflow_id)
    task = asyncio.get_running_loop().create_task(_warm_task(kind, name))
    _warm_tasks.add(task)
    task.add_done_callback(_warm_tasks.discard)


async def _warm_task(kind: ModelKind, name: str) -> None:
    """`warm_model` with the bookkeeping, and nothing above it to catch what it raises.

    A failure leaves the record SUCCESS and the model cold: searches keep answering "is loading",
    and the next boot tries again."""
    workflow_id = _model_id(kind, name)
    loaded = False
    try:
        await warm_model(kind, name)
        loaded = True
        _log.info("model_warmed", kind=kind, model=name)
    except Exception as exc:
        _log.error("model_warm_failed", kind=kind, model=name, error=root_cause(exc))
    finally:
        with _warm_lock:  # both sets in one take, and no await inside it
            if loaded:
                _ready.add(workflow_id)
            _warming.discard(workflow_id)


def _mark_ready(workflow_id: str) -> None:
    """Record that this process has the model loaded (see `_ready`)."""
    with _warm_lock:
        _ready.add(workflow_id)


def is_warm(workflow_id: str) -> bool:
    """Whether the model of one download record is loaded in *this* process."""
    return workflow_id in _ready


_STATE: dict[RunStatus, ModelState] = {
    RunStatus.SUCCESS: ModelState.READY,
    RunStatus.ERROR: ModelState.ERROR,
    RunStatus.CANCELLED: ModelState.ERROR,
}


def _model_status(kind: ModelKind, name: str, workflow) -> ModelStatus:
    """The one DBOS-status -> model-state mapping. `workflow` is None when never started.

    A downloaded model this process has not loaded yet is `loading` with no error: it is warming
    up, which takes seconds rather than the minutes a download takes, but a search still cannot
    use it yet."""
    state = ModelState.PENDING
    if workflow is not None:
        state = _STATE.get(workflow.status, ModelState.LOADING)
    if state == ModelState.READY and not is_warm(_model_id(kind, name)):
        state = ModelState.LOADING
    error = str(workflow.error) if workflow is not None and workflow.error else None
    return ModelStatus(kind=kind, name=name, state=state, error=error)


async def model_statuses() -> list[ModelStatus]:
    """One status per required model. `ensure_models` builds the same list off the records it
    just read, rather than reading them again."""
    settings = await load_user_settings()
    wanted = await _required(settings)
    return _statuses(wanted, await _download_records(wanted), settings)


def _statuses(
    wanted: list[tuple[ModelKind, str]],
    records: dict[str, DbosWorkflowStatus],
    settings: UserSettings,
) -> list[ModelStatus]:
    accelerator = settings.pipeline.accelerator
    return [
        msgspec.structs.replace(
            _model_status(kind, name, records.get(_model_id(kind, name))),
            device=hardware.device(name, accelerator),
        )
        for kind, name in wanted
    ]


async def require_ready(kind: ModelKind, name: str) -> None:
    """Fail fast with a clear message instead of blocking a request on a download.

    Called on every search, so a model this process has loaded is not looked up again: nothing
    takes that answer back, because a loaded model stays loaded for the life of the process."""
    workflow_id = _model_id(kind, name)
    if is_warm(workflow_id):
        return
    found = await DBOS.list_workflows_async(workflow_ids=[workflow_id])
    status = _model_status(kind, name, found[0] if found else None)
    if status.state == ModelState.ERROR:
        raise NotReady(f"{kind} model {name} failed to load: {status.error}")
    if status.state == ModelState.PENDING:
        raise NotReady(f"{kind} model {name} is not loaded yet; check /api/status")
    if (
        found and found[0].status == RunStatus.SUCCESS
    ):  # downloaded, warming up (see `_model_status`)
        raise NotReady(f"{kind} model {name} is loading in this process; retry in a moment")
    raise NotReady(
        f"{kind} model {name} is downloading (operation {workflow_id}); "
        "check /api/operations?kind=download"
    )
