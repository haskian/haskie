"""How the process ends: which shutdown signals count, and how long the exit may take.

The server owns SIGINT and SIGTERM. The first one starts a graceful shutdown, and a second one is
the user asking to hurry. Both rules here keep that true under the tools haskie runs behind.
"""

import concurrent.futures.thread  # noqa: F401 - registers its exit hook first; see `bound_exit`
import contextlib
import math
import os
import signal
import threading
import time
from collections.abc import Callable, Iterator
from types import FrameType

from haskie import logs

_log = logs.get_logger(__name__)

SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class ShuttingDown(BaseException):  # noqa: N818 - named for what it says, like `KeyboardInterrupt`
    """Work the shutdown took away from under its caller.

    A `BaseException`, like `KeyboardInterrupt`, so no `except Exception` mistakes it for a
    failure. DBOS records and retries only `Exception`: a step that raises this records nothing,
    its workflow stays pending, and the next boot recovers it.
    """


# One Ctrl-C reaches the server more than once. The terminal sends SIGINT to the whole foreground
# process group, and a wrapper in that group sends it on again: `mise run` at once, `uv run` 200 ms
# later. Measured, not assumed. uvicorn reads a second SIGINT as "force quit", which skips the
# shutdown hooks. So a signal this soon after the last one that counted is dropped.
DUPLICATE_WINDOW = 0.5

# Read inside signal handlers, so never mutated in place: a writer builds a new tuple under
# `_listeners_lock` and swaps it in, and a handler reads whichever whole tuple is current. A
# handler must not take the lock itself: it runs on the main thread, maybe while the main thread
# already holds it, and a `Lock` is not reentrant.
_listeners: tuple[Callable[[], object], ...] = ()
_listeners_lock = threading.Lock()


def debounce_signals() -> None:
    """Wrap the server's SIGINT and SIGTERM handlers so a duplicate never reaches them, and a
    signal that counts reaches `listening` callers first.

    A startup hook, because the server installs its handlers before the app starts, and puts its
    own originals back when it stops, which drops these wrappers with it. Handlers only install on
    the main thread; a test client runs the app elsewhere, and then this does nothing.
    """
    if threading.current_thread() is not threading.main_thread():
        return
    for number in SHUTDOWN_SIGNALS:
        handler = signal.getsignal(number)
        if callable(handler):
            signal.signal(number, _counted(handler))


def _counted(
    handler: Callable[[int, FrameType | None], object],
) -> Callable[[int, FrameType | None], None]:
    last_counted = -math.inf
    # Python runs a handler between bytecodes, including a handler's own, so a copy that lands
    # while this one runs nests inside it - between the check and the set below, it would count
    # too. Taken without blocking: a nested signal finds it held and is a copy by definition.
    checking = threading.Lock()

    def on_signal(number: int, frame: FrameType | None) -> None:
        nonlocal last_counted
        if not checking.acquire(blocking=False):
            return
        try:
            now = time.monotonic()
            if now - last_counted < DUPLICATE_WINDOW:
                return
            last_counted = now
        finally:
            checking.release()
        for listener in _listeners:
            listener()
        handler(number, frame)

    return on_signal


@contextlib.contextmanager
def listening(callback: Callable[[], object]) -> Iterator[None]:
    """Call `callback` on every shutdown signal that counts, for the length of the block. It runs
    inside a signal handler, so it must only set a flag."""
    global _listeners
    with _listeners_lock:
        _listeners = (*_listeners, callback)
    try:
        yield
    finally:
        with _listeners_lock:
            _listeners = tuple(listener for listener in _listeners if listener is not callback)


WORKFLOW_GRACE = 10  # seconds running workflows get to finish before DBOS cancels them

# How long the interpreter may take to exit once the main thread is done. Before it exits, Python
# joins every non-daemon thread, and it ignores Ctrl-C while it waits. Work a shutdown abandons -
# CPU work a request started in a worker thread, or everything when a forced shutdown skips the
# shutdown hooks - is C code that nothing can interrupt, so it holds the process for as long as it
# runs. Past this grace the process leaves it behind. Operations are durable: the next boot
# recovers them.
EXIT_GRACE = 5.0


_exit_started = threading.Event()


def _exit_when_held() -> None:
    _exit_started.wait()
    time.sleep(EXIT_GRACE)
    held_by = [t.name for t in threading.enumerate() if not t.daemon and t.is_alive()]
    _log.warning("exit_forced", grace_seconds=EXIT_GRACE, held_by=held_by)
    os._exit(1)


_bounded = False
_bounded_lock = threading.Lock()


def bound_exit() -> None:
    """Make sure the process exits within `EXIT_GRACE` of its main thread finishing. Once per
    process, however many times an app starts in it. A daemon thread, so a clean exit ends it
    without waiting and never reaches `os._exit`.

    The clock starts in a threading exit hook, not when the main thread counts as done. Python
    runs those hooks, newest first, before it marks the main thread done, and the thread-pool
    module's hook joins every thread-pool worker: a DBOS step or an MLX batch would hold the exit
    before the clock even started. So this hook is registered after that module's, which the
    import above makes sure of, and runs before it.
    """
    global _bounded
    with _bounded_lock:  # `functools.cache` would not do: two first calls may both run
        if not _bounded:
            _bounded = True
            threading._register_atexit(_exit_started.set)  # ty: ignore[unresolved-attribute]
            threading.Thread(target=_exit_when_held, name="haskie-exit-bound", daemon=True).start()
