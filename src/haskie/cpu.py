"""The CPU budget: the one ceiling every piece of CPU work passes through.

Pipeline steps, preview builds, model loads and reranking are CPU work, not IO, so they run in a
worker thread (`on_cpu`) and hold one slot of the budget for the length of that work. The number
of them running at once is therefore never above `indexing.cpu_budget`, whichever queue, request
or event loop they came from.

The per-queue caps in `workflows.stage_caps` shape the *mix* of work; this budget is the ceiling,
which no queue can enforce, because no queue sees the others.
"""

import functools
import multiprocessing
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Any

import anyio
import anyio.to_thread

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
# ponytail: a threading primitive on purpose. Two event loops take from this budget - Litestar's
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
    is unbounded on purpose: the caller's turn comes as soon as another task finishes, and giving
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

_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()


def _convert_pool() -> ProcessPoolExecutor | None:
    """The extraction pool, built on first use so a process that never converts never forks. It
    is sized from the CPU budget, so one setting owns how much of the machine haskie takes."""
    global _pool
    workers = _cpu_slots.size if CONVERT_WORKERS is None else CONVERT_WORKERS
    if not workers:
        return None
    with _pool_lock:
        if _pool is None:
            # forkserver, not the macOS default of spawn: a spawned child re-imports `__main__`,
            # which under `uvicorn`/`litestar` is the console script -- the child would try to
            # start a second server. A forkserver child is forked from a clean, thread-free
            # process instead, so `__main__` is never re-run and plain `fork` stays unsafe-free.
            context = multiprocessing.get_context("forkserver")
            context.set_forkserver_preload(["haskie.convert"])  # import once, not per child
            _pool = ProcessPoolExecutor(max_workers=workers, mp_context=context)
        return _pool


def shutdown_pool() -> None:
    """Drop the extraction pool. Idempotent, so a second shutdown is not an error."""
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


async def off_interpreter[T](fn: Callable[..., T], /, *args: Any) -> T:
    """Run one piece of CPU work in a separate interpreter, under one slot of the budget.

    Same contract as `on_cpu`, and it falls back to `on_cpu` when the pool is off, so callers do
    not branch. `fn` and `args` cross a pickle boundary: module-level function, plain arguments.
    The slot is held for the whole call, not just the local part, so threads and pool processes
    draw on one budget. The waiting thread blocks on a pipe, so it holds no GIL meanwhile.
    """
    pool = _convert_pool()
    if pool is None:
        return await on_cpu(fn, *args)
    call = functools.partial(fn, *args)
    return await _in_thread(lambda: pool.submit(call).result())
