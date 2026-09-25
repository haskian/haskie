"""How the process ends: which shutdown signals count, and how long the exit may take.

Signals are real here, raised at this process, because what is under test is what a handler
does when the kernel calls it. Every test puts the handlers back as it found them.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from haskie import shutdown


@pytest.fixture
def server_handler() -> Iterator[list[int]]:
    """Stand in for the server: a Python handler on SIGINT and SIGTERM that records each call, in
    the order the calls came."""
    seen: list[int] = []
    found = {number: signal.getsignal(number) for number in shutdown.SHUTDOWN_SIGNALS}
    for number in shutdown.SHUTDOWN_SIGNALS:
        signal.signal(number, lambda n, _frame: seen.append(n))
    yield seen
    for number, handler in found.items():
        signal.signal(number, handler)


@dataclass
class SignalCase:
    sent: list[tuple[float, signal.Signals]]  # (seconds after the previous one, signal)
    expect_server: list[signal.Signals]
    expect_listened: int
    window: float = shutdown.DUPLICATE_WINDOW


SIGNAL_CASES = {
    "the first signal reaches the server and the listeners": SignalCase(
        sent=[(0, signal.SIGINT)], expect_server=[signal.SIGINT], expect_listened=1
    ),
    "a duplicate inside the window reaches neither": SignalCase(
        sent=[(0, signal.SIGINT), (0, signal.SIGINT), (0.05, signal.SIGINT)],
        expect_server=[signal.SIGINT],
        expect_listened=1,
    ),
    "a second press after the window counts": SignalCase(
        sent=[(0, signal.SIGINT), (0.15, signal.SIGINT)],
        expect_server=[signal.SIGINT, signal.SIGINT],
        expect_listened=2,
        window=0.1,
    ),
    "SIGTERM counts on a clock of its own": SignalCase(
        sent=[(0, signal.SIGTERM), (0, signal.SIGINT)],
        expect_server=[signal.SIGTERM, signal.SIGINT],
        expect_listened=2,
    ),
}


@pytest.mark.parametrize("case", SIGNAL_CASES.values(), ids=list(SIGNAL_CASES))
def test_only_signals_that_count_reach_the_server(
    case: SignalCase, server_handler: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrapper in the terminal's process group sends one Ctrl-C on again, and uvicorn would
    read the copy as a second press: a force quit that skips the shutdown hooks."""
    monkeypatch.setattr(shutdown, "DUPLICATE_WINDOW", case.window)
    listened: list[None] = []
    shutdown.debounce_signals()

    with shutdown.listening(lambda: listened.append(None)):
        for pause, number in case.sent:
            time.sleep(pause)
            signal.raise_signal(number)

    assert server_handler == case.expect_server
    assert len(listened) == case.expect_listened


class WindowThatLetsACopyIn(float):
    """The duplicate window, which sends a copy of the signal the first time a handler compares
    against it. `elapsed < window` asks the window, so the copy lands after the check and before
    the clock update: the one place where a nested handler would count it."""

    fired = False

    def __gt__(self, elapsed: float) -> bool:
        if not WindowThatLetsACopyIn.fired:
            WindowThatLetsACopyIn.fired = True
            signal.raise_signal(signal.SIGINT)  # its handler runs at the next bytecode
        return float(self) > elapsed


def test_a_copy_that_nests_inside_the_check_is_dropped(
    server_handler: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python runs a handler between bytecodes, its own included, so `mise run`'s copy 0.1 ms
    after the terminal's can nest inside the handler of the first. It must not count as a
    second press, wherever it lands."""
    monkeypatch.setattr(WindowThatLetsACopyIn, "fired", False)
    monkeypatch.setattr(
        shutdown, "DUPLICATE_WINDOW", WindowThatLetsACopyIn(shutdown.DUPLICATE_WINDOW)
    )
    shutdown.debounce_signals()

    signal.raise_signal(signal.SIGINT)

    assert WindowThatLetsACopyIn.fired, "the copy was sent"
    assert server_handler == [signal.SIGINT], "one press, however the copy landed"


def test_signals_are_left_alone_where_they_cannot_be_wrapped() -> None:
    """Off the main thread Python refuses to install a handler, and a disposition that is not a
    Python handler (SIG_DFL, SIG_IGN) has nothing to hand a signal on to."""
    found = {number: signal.getsignal(number) for number in shutdown.SHUTDOWN_SIGNALS}
    elsewhere = threading.Thread(target=shutdown.debounce_signals)
    elsewhere.start()
    elsewhere.join()
    assert {number: signal.getsignal(number) for number in found} == found, "off the main thread"

    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        shutdown.debounce_signals()
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL, "the default stays the default"
    finally:
        for number, handler in found.items():
            signal.signal(number, handler)


# A server process in miniature: the app bounds its exit, then the main thread finishes, the way
# `uvicorn.run` returns after a shutdown. `sys.argv[1]` says whether work is still in a thread.
EXIT_SCRIPT = """
import sys, threading, time
from haskie import shutdown
shutdown.EXIT_GRACE = 0.5
shutdown.bound_exit()
shutdown.bound_exit()  # a second app in the same process adds no second bound
if sys.argv[1] == "busy":
    threading.Thread(target=time.sleep, args=(120,), name="busy-worker").start()
"""


@dataclass
class ExitCase:
    work: str
    exit_code: int
    forced: bool


EXIT_CASES = {
    "work still running is left behind after the grace": ExitCase(
        work="busy", exit_code=1, forced=True
    ),
    "a clean exit is not held for the grace": ExitCase(work="idle", exit_code=0, forced=False),
}


@pytest.mark.parametrize("case", EXIT_CASES.values(), ids=list(EXIT_CASES))
def test_the_process_exits_within_its_grace(case: ExitCase, tmp_path: Path) -> None:
    """Python joins every non-daemon thread before it exits and ignores Ctrl-C while it does, so
    CPU work a shutdown abandoned would hold the process for as long as it runs."""
    done = subprocess.run(
        [sys.executable, "-c", EXIT_SCRIPT, case.work],
        capture_output=True,
        text=True,
        timeout=60,  # the busy thread alone would take 120
        env={**os.environ, "HASKIE_HOME": str(tmp_path / "home")},
    )

    assert done.returncode == case.exit_code, done.stderr
    assert ("exit_forced" in done.stdout) is case.forced, done.stdout
    if case.forced:
        assert "busy-worker" in done.stdout, "the log names what held the exit"
        assert done.stdout.count("exit_forced") == 1, "one bound per process"
