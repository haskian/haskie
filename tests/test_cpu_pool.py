"""PDF extraction in a separate interpreter.

The rest of the suite sets `cpu.CONVERT_WORKERS = 0` and extracts inline, because a pool per
xdist worker costs more to start than the tests save. This module is the one place the pool itself
runs, so the pickling boundary and the forkserver context stay covered.
"""

import asyncio
import hashlib
import os
import time
from collections.abc import Iterator
from pathlib import Path

import anyio
import pytest
from pebble import ProcessExpired

from haskie import cpu, shutdown
from haskie.document import convert

from conftest import text_pdf, until  # isort: skip

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
    """A fresh pool of `WORKERS`, open, and shut down after the test."""
    monkeypatch.setattr(cpu, "CONVERT_WORKERS", WORKERS)
    monkeypatch.setattr(cpu, "_pool", None)
    monkeypatch.setattr(cpu, "_pool_closed", False)
    yield
    cpu.shutdown_pool()


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
