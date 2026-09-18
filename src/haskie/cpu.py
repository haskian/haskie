"""The CPU budget: the one ceiling every piece of CPU work passes through.

Pipeline steps, preview builds, model loads and reranking are CPU work, not IO, so they run in a
worker thread (`on_cpu`) and hold one slot of the budget for the length of that work. The number
of them running at once is therefore never above `indexing.cpu_budget`, whichever queue, request
or event loop they came from.

The per-queue caps in `workflows.stage_caps` shape the *mix* of work; this budget is the ceiling,
which no queue can enforce, because no queue sees the others.
"""

import asyncio
import functools
import multiprocessing
import os
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Any

import anyio
import anyio.to_thread

from haskie.logs import get_logger
from haskie.settings import PipelineSettings

_log = get_logger(__name__)

CPU_WAIT_LOG_SECONDS = 5.0  # a wait longer than this is worth a line at debug level

# Sized from `cpu_budget` by `workflows.apply_settings`; the default is what a process that never
# applied settings runs on.
# ponytail: a threading primitive on purpose. Two event loops take from this budget - Litestar's
# (previews, requests) and DBOS's background loop (tasks, maintenance) - and an asyncio or anyio
# primitive belongs to exactly one of them. Only a thread-safe one can be the shared ceiling.
_cpu_budget = PipelineSettings().cpu_budget
_cpu_slots = threading.BoundedSemaphore(_cpu_budget)

# One thread limiter per loop, for the same reason in reverse: an `anyio.CapacityLimiter` is bound
# to the loop that first used it, so the two loops cannot share one. It is deliberately far wider
# than the budget - anyio's default of 40 threads per loop would queue previews and searches behind
# pipeline work before the budget is even reached, and `_cpu_slots` is meant to be the only limit.
_limiters: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, anyio.CapacityLimiter] = (
    weakref.WeakKeyDictionary()
)
_limiters_lock = threading.Lock()


def configure_cpu_budget(budget: int) -> None:
    """Resize the pool of CPU slots (from `apply_settings`). A task in flight releases the
    semaphore it acquired, so it is unaffected by a resize under it."""
    global _cpu_budget, _cpu_slots
    if budget != _cpu_budget:
        _cpu_budget, _cpu_slots = budget, threading.BoundedSemaphore(budget)


@contextmanager
def cpu_slot(work: str) -> Iterator[None]:
    """Hold one slot of the CPU budget for the length of one piece of CPU work.

    Sync, and taken inside the worker thread: the calling event loop never waits on it. The wait
    is unbounded on purpose: the caller's turn comes as soon as another task finishes, and giving
    up would fail a document for finding the machine busy.
    """
    slots = _cpu_slots  # the object to release, even if the pool is resized meanwhile
    started = time.perf_counter()
    slots.acquire()
    waited = time.perf_counter() - started
    if waited > CPU_WAIT_LOG_SECONDS:
        _log.debug("cpu_budget_wait", work=work, seconds=round(waited, 1))
    try:
        yield
    finally:
        slots.release()


def _limiter() -> anyio.CapacityLimiter:
    """The thread limiter of the running loop, made on first use (see `_limiters`)."""
    loop = asyncio.get_running_loop()
    with _limiters_lock:
        limiter = _limiters.get(loop)
        if limiter is None:
            limiter = _limiters[loop] = anyio.CapacityLimiter(max(64, 4 * _cpu_budget))
        return limiter


async def on_cpu[T](work: str, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run one piece of CPU work in a worker thread, under one slot of the budget.

    `work` names the work in the wait log; it is the only reason the slot is taken in here rather
    than by the callers, which would have to repeat the same three lines at every site.
    """
    call = functools.partial(fn, *args, **kwargs)

    def run() -> T:
        with cpu_slot(work):
            return call()

    return await anyio.to_thread.run_sync(run, limiter=_limiter())


# --- work that has to leave this interpreter ---------------------------------------
#
# A thread is enough for CPU work whose extension releases the GIL. Measured, against a control of
# two pure-Python threads sharing the GIL at ~52% each:
#
#   pure Python (control)                 51.9%
#   convert.pdf_pages_markdown            32.8%   <- holds the GIL, worse than pure Python
#   convert.pdf_page_count (pypdf)        91.5%
#   chunk.split (semantic-text-splitter)  95.2%
#
# So only PDF extraction needs a process; everything else stays on a thread, where it costs no
# pickling and no interpreter. Moving it out measured 71.4% -> 97.4% for an unrelated thread, and
# the extraction itself got faster (0.80s -> 0.48s for 12 documents) because it finally runs in
# parallel rather than taking turns on the GIL.
#
# `0` runs it inline instead, which is what the test suite sets: a pool per xdist worker costs
# more to start than the tests would save.
CONVERT_WORKERS = (
    int(os.environ.get("HASKIE_CONVERT_WORKERS") or PipelineSettings().cpu_budget) or None
)

_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()


def _convert_pool() -> ProcessPoolExecutor | None:
    """The extraction pool, built on first use so a process that never converts never forks."""
    global _pool
    if CONVERT_WORKERS is None:
        return None
    with _pool_lock:
        if _pool is None:
            # forkserver, not the macOS default of spawn: a spawned child re-imports `__main__`,
            # which under `uvicorn`/`litestar` is the console script -- the child would try to
            # start a second server. A forkserver child is forked from a clean, thread-free
            # process instead, so `__main__` is never re-run and plain `fork` stays unsafe-free.
            context = multiprocessing.get_context("forkserver")
            context.set_forkserver_preload(["haskie.convert"])  # import once, not per child
            _pool = ProcessPoolExecutor(max_workers=CONVERT_WORKERS, mp_context=context)
        return _pool


def shutdown_pool() -> None:
    """Drop the extraction pool. Idempotent, so a second shutdown is not an error."""
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


async def off_interpreter[T](work: str, fn: Callable[..., T], /, *args: Any) -> T:
    """Run one piece of CPU work in a separate interpreter, under one slot of the budget.

    Same contract as `on_cpu`, and it falls back to `on_cpu` when the pool is off, so callers do
    not branch. `fn` and `args` cross a pickle boundary: module-level function, plain arguments.

    The slot is held for the whole call, not just the local part, so the budget stays the one
    ceiling across threads *and* pool processes - otherwise each would admit `cpu_budget` work.
    """
    pool = _convert_pool()
    if pool is None:
        return await on_cpu(work, fn, *args)
    call = functools.partial(fn, *args)

    def run() -> T:
        with cpu_slot(work):
            return pool.submit(call).result()

    # the waiting thread is blocked on a pipe, so it holds no GIL while the work runs elsewhere
    return await anyio.to_thread.run_sync(run, limiter=_limiter())
