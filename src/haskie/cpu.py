"""The CPU budget: the one ceiling every piece of CPU work passes through.

Pipeline steps, preview builds, model loads and reranking are CPU work, not IO, so they run in a
worker thread (`on_cpu`) and hold one slot of the budget for the length of that work. The number
of them running at once is therefore never above `pipeline.cpu_budget`, whichever queue, request
or event loop they came from.

The per-queue caps in `workflows.stage_caps` shape the *mix* of work; this budget is the ceiling,
which no queue can enforce, because no queue sees the others.
"""

import functools
import multiprocessing
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError
from contextlib import contextmanager
from types import ModuleType
from typing import Any, cast

import anyio
import anyio.to_thread
from pebble import ProcessExpired, ProcessFuture, ProcessPool

from haskie import shutdown
from haskie.settings import PipelineSettings


class ResizableSemaphore[S]:
    """A semaphore the settings may resize while work is in flight.

    A caller acquires `current` and releases that same object, so a resize under it neither
    over-admits nor raises. Generic over the semaphore itself: the CPU budget needs a thread-safe
    one (two event loops take from it), a preview slot an anyio one.
    """

    def __init__(self, make: Callable[[int], S], size: int) -> None:
        self._make = make
        self.size = size
        self.current: S = make(size)

    def resize(self, size: int) -> None:
        if size != self.size:
            self.size, self.current = size, self._make(size)


# Sized from `cpu_budget` by `workflows.apply_settings`; the default is what a process that never
# applied settings runs on.
# a threading primitive on purpose. Two event loops take from this budget - Litestar's
# (previews, requests) and DBOS's background loop (tasks, maintenance) - and an asyncio or anyio
# primitive belongs to exactly one of them. Only a thread-safe one can be the shared ceiling.
_cpu_slots = ResizableSemaphore(threading.BoundedSemaphore, PipelineSettings().cpu_budget)

# anyio's default is 40 worker threads per event loop, which would queue previews and searches
# behind pipeline work before the budget is even reached; `_cpu_slots` is meant to be the only
# limit. anyio keeps one default limiter per loop, so widening it widens the loop we run on.
THREAD_LIMIT = 256


def configure_cpu_budget(budget: int) -> None:
    """Resize the pool of CPU slots (from `apply_settings`)."""
    _cpu_slots.resize(budget)


@contextmanager
def cpu_slot() -> Iterator[None]:
    """Hold one slot of the CPU budget for the length of one piece of CPU work.

    Sync, and taken inside the worker thread: the calling event loop never waits on it. The wait
    is unbounded on purpose: the caller's turn comes as soon as other CPU work finishes, and giving
    up would fail a document for finding the machine busy.
    """
    slots = _cpu_slots.current  # the object to release, even if the pool is resized meanwhile
    slots.acquire()
    try:
        yield
    finally:
        slots.release()


async def _in_thread[T](call: Callable[[], T]) -> T:
    """Run `call` in a worker thread of the running loop, holding one slot of the budget."""
    limiter = anyio.to_thread.current_default_thread_limiter()
    if limiter.total_tokens < THREAD_LIMIT:
        limiter.total_tokens = THREAD_LIMIT

    def run() -> T:
        with cpu_slot():
            return call()

    return await anyio.to_thread.run_sync(run)


async def on_cpu[T](fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run one piece of CPU work in a worker thread, under one slot of the budget."""
    return await _in_thread(functools.partial(fn, *args, **kwargs))


# --- work that has to leave this interpreter ---------------------------------------
#
# A thread is enough for CPU work whose extension releases the GIL. Against a control of two
# pure-Python threads sharing the GIL at 51.9% each, `convert.pdf_pages_markdown` measured 32.8%:
# it holds the GIL. Every other extension we call stayed above 90%. So only PDF extraction needs a
# process; everything else stays on a thread, where it costs no pickling and no interpreter.
#
# `0` runs it inline instead, which is what the test suite sets: a pool per xdist worker costs
# more to start than the tests would save.
CONVERT_WORKERS: int | None = None  # None sizes the pool from the CPU budget

_pool: ProcessPool | None = None
_pool_closed = False  # latched by `shutdown_pool`, so nothing builds a fresh pool behind it
_pool_lock = threading.Lock()
# What the pool is running or holding, so `shutdown_pool` can resolve it; guarded by `_pool_lock`.
_in_flight: set[ProcessFuture] = set()


def _pool_size() -> int:
    return _cpu_slots.size if CONVERT_WORKERS is None else CONVERT_WORKERS


def _convert_pool() -> ProcessPool:
    """The extraction pool, built on first use so a process that never converts never forks. It
    is sized from the CPU budget, so one setting owns how much of the machine haskie takes. The
    caller holds `_pool_lock`.

    pebble rather than `ProcessPoolExecutor`: when a worker dies - a parser that segfaults, a
    page that runs the machine out of memory - the stdlib pool breaks, and every later extraction
    fails until a restart. pebble fails that one task with `ProcessExpired` and starts a new
    worker, so the documents converting beside it never notice.
    """
    global _pool
    if _pool_closed:
        raise shutdown.ShuttingDown("the extraction pool is shut down")
    if _pool is None:
        # forkserver, not the macOS default of spawn: a spawned child re-imports `__main__`,
        # which under `uvicorn`/`litestar` is the console script -- the child would try to
        # start a second server. A forkserver child is forked from a clean, thread-free process
        # instead, so `__main__` is never re-run. Plain `fork` is no option either: it would copy
        # a process that runs threads, which is unsafe.
        context = multiprocessing.get_context("forkserver")
        # import once, not per child
        context.set_forkserver_preload(["haskie.document.convert"])
        # pebble annotates `context` as the `multiprocessing` module, but it only calls the
        # `Process`, `Pipe` and `Lock` a context object has too, as its docs say
        _pool = ProcessPool(max_workers=_pool_size(), context=cast("ModuleType", context))
    return _pool


def _run_in_pool[T](fn: Callable[..., T], args: tuple[Any, ...]) -> T:
    """Run `fn` in the pool and wait for it, known to `shutdown_pool` from the moment it is
    scheduled: both happen under the lock that shutdown takes.

    A call that shutdown ends raises `ShuttingDown`, never an error of its own: DBOS would record
    an error against the step and retry it, and three retries fit inside the exit grace, so the
    document would end up failed rather than recovered at the next boot.
    """
    with _pool_lock:
        future = _convert_pool().schedule(fn, args=args)
        _in_flight.add(future)
    try:
        return future.result()
    except (CancelledError, ProcessExpired):
        with _pool_lock:
            closed = _pool_closed
        if closed:
            raise shutdown.ShuttingDown("the extraction pool shut down under this call") from None
        raise
    finally:
        with _pool_lock:
            _in_flight.discard(future)


def open_pool() -> None:
    """Let extraction use the pool again, after a `shutdown_pool` in the same process."""
    global _pool_closed
    with _pool_lock:
        _pool_closed = False


def shutdown_pool() -> None:
    """Close the extraction pool and kill its workers. Idempotent, so a second shutdown is not an
    error. Sync and quick, but it joins processes: call it off the event loop.

    An extraction still running is lost, not waited for: it would hold the interpreter's exit,
    deaf to Ctrl-C, and one OCR batch can take minutes. Its step is durable and runs again at the
    next boot. The order matters:

    1. Latch the pool closed and take what is in flight, under the lock `_run_in_pool`
       schedules under: nothing is scheduled after, and nothing scheduled before is missed.
    2. Cancel what is in flight, so each waiting thread wakes to a cancel, which reads as
       `ShuttingDown`: a stopped pebble pool never resolves the futures it held.
    3. Stop the pool, so its manager starts no worker in place of the ones killed next.
    4. SIGKILL every worker at once. pebble's own stop sends SIGTERM and waits 3 s per worker,
       one after the other, and a worker inside a parser's C code cannot run its SIGTERM handler
       until the call returns, so four busy workers cost it 12 s.
    5. Join, which now reaps dead workers.
    """
    global _pool, _pool_closed
    with _pool_lock:
        pool, _pool, _pool_closed = _pool, None, True
        in_flight = list(_in_flight)
    for future in in_flight:
        future.cancel()
    if pool is None:
        return
    pool.stop()
    # `workers` is pebble's own record of its processes; it has no public way to kill them at
    # once. `copy` is one step under the GIL, so the manager thread cannot change it midway.
    for worker in pool._pool_manager.worker_manager.workers.copy().values():
        worker.kill()
    pool.join()


async def off_interpreter[T](fn: Callable[..., T], /, *args: Any) -> T:
    """Run one piece of CPU work in a separate interpreter, under one slot of the budget.

    Same contract as `on_cpu`, and it falls back to `on_cpu` when the pool is off, so callers do
    not branch. `fn` and `args` cross a pickle boundary: module-level function, plain arguments.
    The slot is held for the whole call, not just the local part, so threads and pool processes
    draw on one budget. The waiting thread blocks on a pipe, so it holds no GIL meanwhile.
    """
    if not _pool_size():
        return await on_cpu(fn, *args)
    return await _in_thread(functools.partial(_run_in_pool, fn, args))
