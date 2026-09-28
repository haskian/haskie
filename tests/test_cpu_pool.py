"""PDF extraction in a separate interpreter.

The rest of the suite sets `cpu.CONVERT_WORKERS = 0` and extracts inline, because a pool per
xdist worker costs more to start than the tests save. This module is the one place the pool itself
runs, so the pickling boundary and the forkserver context stay covered.
"""

import asyncio
import hashlib
import multiprocessing
import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import anyio
import pytest
from pebble import ProcessExpired, ProcessPool

from haskie import cpu, shutdown
from haskie.document import convert

from conftest import WAIT, text_pdf, until  # isort: skip

pytestmark = [pytest.mark.anyio, pytest.mark.usefixtures("pooled")]

PAGES: list[str | None] = ["page one text", "page two text", "page three text"]


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "doc.pdf"
    path.write_bytes(text_pdf(PAGES))
    return path


WORKERS = 2
# `pbkdf2_hmac` stays in C for the whole call and checks for no signal, as a parser does; this
# many rounds run for about a minute. `time.sleep` would not do: a signal interrupts it.
NATIVE_ROUNDS = 800_000_000


@pytest.fixture
def pooled(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A fresh pool of `WORKERS`, open, and shut down after the test.

    The CPU budget is set here too. Every pool call holds a slot of it, and the budget is
    process-wide: an earlier test on the same worker may have left it at one, and then no two
    calls could ever run at once."""
    budget = cpu._cpu_slots.size
    cpu.configure_cpu_budget(WORKERS)
    monkeypatch.setattr(cpu, "CONVERT_WORKERS", WORKERS)
    monkeypatch.setattr(cpu, "_pool", None)
    monkeypatch.setattr(cpu, "_pool_closed", False)
    yield
    cpu.shutdown_pool()
    cpu.configure_cpu_budget(budget)


def _native_call() -> asyncio.Task[bytes]:
    return asyncio.create_task(
        cpu.off_interpreter(hashlib.pbkdf2_hmac, "sha256", b"key", b"salt", NATIVE_ROUNDS)
    )


async def _running(extractions: int) -> None:
    """Until the pool runs `extractions` calls: a cold forkserver imports the parsers first."""

    async def running() -> bool:
        return sum(future.running() for future in list(cpu._in_flight)) >= extractions

    await until(running, f"the pool never ran {extractions} calls at once")


async def test_extraction_in_a_pool_matches_extraction_inline(pdf: Path) -> None:
    """Same answer either way: `off_interpreter` only changes where the work runs."""
    in_pool = await cpu.off_interpreter(convert.pdf_pages_markdown, pdf)

    cpu.CONVERT_WORKERS = 0  # the fixture's monkeypatch puts it back
    inline = await cpu.off_interpreter(convert.pdf_pages_markdown, pdf)

    assert in_pool == inline
    markdown, ocr_pages, pages = in_pool
    assert (pages, ocr_pages) == (len(PAGES), [])
    assert "<!-- page 1 -->" in markdown, "the page markers the chunker reads survive the boundary"


async def test_an_error_in_the_pool_reaches_the_caller(tmp_path: Path) -> None:
    """A parser failure is a `PermanentError` whether or not it crossed a process boundary."""
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")

    with pytest.raises(convert.PermanentError, match="could not convert bad.pdf"):
        await cpu.off_interpreter(convert.pdf_pages_markdown, bad)


async def test_shutdown_kills_busy_workers_at_once() -> None:
    """A worker in C code cannot run its SIGTERM handler, and pebble's own stop waits 3 s for each
    worker in turn. Shutdown must not: it runs on the way out, and nothing hurries it.

    Every caller hears `ShuttingDown`, which DBOS records no error for, wherever its call was:
    running, queued in the pool behind the busy workers, or not yet scheduled at all. A call that
    shutdown missed would leave its thread waiting forever."""
    calls = [_native_call() for _ in range(WORKERS * 3)]
    await _running(WORKERS)
    assert cpu._pool is not None
    workers = list(cpu._pool._pool_manager.worker_manager.workers.values())

    started = time.monotonic()
    await anyio.to_thread.run_sync(cpu.shutdown_pool)
    took = time.monotonic() - started

    assert took < 1.5, f"shutdown took {took:.1f}s; pebble's stop alone takes 3 s a worker"
    for call in calls:
        with anyio.fail_after(10), pytest.raises(shutdown.ShuttingDown):
            await call
    assert all(not worker.is_alive() for worker in workers), "no worker outlives the pool"
    assert not cpu._in_flight, "nothing is left for a later shutdown to wait on"


async def test_a_closed_pool_refuses_work_until_it_opens() -> None:
    """A step still running after a hurried shutdown must not build a fresh pool behind it: its
    workers would outlive the process. A runtime started again opens it."""
    cpu.shutdown_pool()

    with pytest.raises(shutdown.ShuttingDown):
        await cpu.off_interpreter(sum, [1, 2])
    assert cpu._pool is None, "no pool was built for the refused call"

    cpu.open_pool()
    assert await cpu.off_interpreter(sum, [1, 2]) == 3


async def test_a_worker_that_dies_fails_only_its_own_extraction() -> None:
    """A parser that crashes its process fails that one call. The call beside it finishes, and
    the pool takes the next call, where a broken stdlib pool would refuse every one until a
    restart. A crash is an error, not a shutdown: the pool is open."""
    beside = asyncio.create_task(cpu.off_interpreter(time.sleep, 2))
    await _running(1)

    with pytest.raises(ProcessExpired):
        await cpu.off_interpreter(os.abort)  # SIGABRT: a crash in C, as a parser has them

    assert await beside is None, "the extraction beside the crash finished"
    assert await cpu.off_interpreter(sum, [1, 2]) == 3, "the pool takes the next call"


def _workers(pool: ProcessPool | None) -> list[multiprocessing.Process]:
    assert pool is not None
    return list(pool._pool_manager.worker_manager.workers.values())


@dataclass
class PoolResizeCase:
    convert_workers: int | None
    budget: int
    retired: bool


POOL_RESIZE_CASES = {
    "a raised budget retires the pool, and the next one runs at its size": PoolResizeCase(
        None, WORKERS + 1, retired=True
    ),
    "the same budget keeps the pool": PoolResizeCase(None, WORKERS, retired=False),
    "a set worker count keeps the pool whatever the budget": PoolResizeCase(
        WORKERS, WORKERS + 1, retired=False
    ),
}


@pytest.mark.parametrize("case", POOL_RESIZE_CASES.values(), ids=list(POOL_RESIZE_CASES))
async def test_a_resized_budget_resizes_the_extraction_pool(
    case: PoolResizeCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pool keeps the size it was built with. Left at the old size, a raised budget would park
    extractions in the pool's queue, each holding a CPU slot. The work the old pool already took
    finishes there."""
    monkeypatch.setattr(cpu, "CONVERT_WORKERS", case.convert_workers)
    taken = asyncio.create_task(cpu.off_interpreter(time.sleep, 1))
    await _running(1)
    old = cpu._pool
    old_workers = _workers(old)

    cpu.configure_cpu_budget(case.budget)

    assert (cpu._pool is not old) is case.retired
    assert await taken is None, "the work the old pool took finishes"
    if not case.retired:
        return
    calls = [_native_call() for _ in range(case.budget)]
    await _running(case.budget)
    assert _workers(cpu._pool) != old_workers, "a new pool took the new work"

    async def old_pool_gone() -> bool:
        return not cpu._retired and not any(worker.is_alive() for worker in old_workers)

    await until(old_pool_gone, "the retired pool outlived its work")
    await anyio.to_thread.run_sync(cpu.shutdown_pool)
    for call in calls:
        with anyio.fail_after(10), pytest.raises(shutdown.ShuttingDown):
            await call


async def test_shutdown_kills_a_retired_pool_still_at_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """A retired pool finishes its work only while the process runs. At shutdown its workers die
    with the rest, and its callers hear `ShuttingDown`."""
    monkeypatch.setattr(cpu, "CONVERT_WORKERS", None)
    call = _native_call()
    await _running(1)
    old = cpu._pool
    old_workers = _workers(old)
    cpu.configure_cpu_budget(WORKERS + 1)
    assert cpu._retired == {old}, "the old pool is still at work"

    started = time.monotonic()
    await anyio.to_thread.run_sync(cpu.shutdown_pool)

    assert time.monotonic() - started < 1.5, "killed, not waited for"
    with anyio.fail_after(10), pytest.raises(shutdown.ShuttingDown):
        await call
    assert not any(worker.is_alive() for worker in old_workers), "no worker outlives shutdown"
    assert not cpu._retired


# --- a budget resized under load --------------------------------------------------

# Long enough for a newcomer the budget should not admit to show up, if the budget let it in.
ADMISSION_SETTLE = 0.3


@dataclass
class ResizeCase:
    before: int
    after: int
    running_under_load: int  # what runs at once while the old holders still hold


RESIZE_CASES = {
    "a raise admits only the new slots beside the running work": ResizeCase(2, 3, 3),
    "a cut admits nothing until the running work is under it": ResizeCase(3, 1, 3),
    "the same size admits nothing more": ResizeCase(2, 2, 2),
}


@pytest.mark.parametrize("case", RESIZE_CASES.values(), ids=list(RESIZE_CASES))
async def test_a_resized_budget_counts_the_work_already_running(case: ResizeCase) -> None:
    """A resize moves the limit against the holders it finds, or a raise from 2 to 3 under
    load would run 5 at once."""
    counted = threading.Lock()
    running, peak = 0, 0
    old_may_finish = threading.Event()
    newcomers_meet = threading.Barrier(case.after)

    def work(wait: Callable[[], object]) -> None:
        nonlocal running, peak
        with counted:
            running += 1
            peak = max(peak, running)
        try:
            wait()
        finally:
            with counted:
                running -= 1

    async def ran_at_once(count: int) -> None:
        async def reached() -> bool:
            return peak >= count

        await until(reached, f"{count} callers never ran at once")

    cpu.configure_cpu_budget(case.before)
    async with anyio.create_task_group() as callers:
        for _ in range(case.before):
            callers.start_soon(cpu.on_cpu, work, lambda: old_may_finish.wait(WAIT))
        await ran_at_once(case.before)

        try:
            cpu.configure_cpu_budget(case.after)
            for _ in range(case.after):
                callers.start_soon(cpu.on_cpu, work, lambda: newcomers_meet.wait(WAIT))
            await ran_at_once(case.running_under_load)
            await anyio.sleep(ADMISSION_SETTLE)
            assert peak == case.running_under_load, "the old holders count against the new size"
        finally:
            old_may_finish.set()  # the newcomers now meet at the barrier: all `after` at once
