"""Haskie home directory (~/.haskie) layout, and the file writes every module shares.

A filesystem call blocks, so every async function here runs its work in a worker thread. The sync
ones (`atomic_replace`, `atomic_write_sync`) are for code that already runs in one: `convert.py`
and `embed_cache.py` are sync by nature (pyarrow, the parsers) and would otherwise hop threads
twice for one write.

The home lock (`claim_home` and friends) is sync for a different reason: it runs before there
is an event loop at all, as the app's first startup hook.
"""

import fcntl
import os
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import anyio
import anyio.to_thread

HOME = Path(os.environ.get("HASKIE_HOME", Path.home() / ".haskie"))
COLLECTION_ROOT = HOME / "collections"  # one LanceDB index per collection
DOCUMENT_ROOT = HOME / "documents"  # one folder per imported document: original, markdown, cache
STAGING_ROOT = HOME / "staging"  # uploads not yet imported; swept by the nightly maintenance
AUDIT_DIR = HOME / "audit"
DB_FILE = HOME / "haskie.db"
MODEL_CACHE = HOME / "cache" / "models"  # compiled CoreML models (see embed.py); ORT creates it
LOCK_FILE = HOME / "haskie.lock"  # one running haskie per home (see `claim_home`)

DIR_MODE = 0o700  # documents and the audit trail are private to the user running the app


def use(root: Path) -> None:
    """Point this process at another home, before anything has read the layout.

    The paths above are resolved once, at import, so `haskie --home` (and the test suite) cannot
    just set the environment variable: the derived paths would already be built from the old root.
    Nothing here re-opens what is already open, so this belongs at startup and nowhere else.
    """
    global \
        HOME, \
        COLLECTION_ROOT, \
        DOCUMENT_ROOT, \
        STAGING_ROOT, \
        AUDIT_DIR, \
        DB_FILE, \
        MODEL_CACHE, \
        LOCK_FILE
    HOME = root
    COLLECTION_ROOT = HOME / "collections"
    DOCUMENT_ROOT = HOME / "documents"
    STAGING_ROOT = HOME / "staging"
    AUDIT_DIR = HOME / "audit"
    DB_FILE = HOME / "haskie.db"
    MODEL_CACHE = HOME / "cache" / "models"
    LOCK_FILE = HOME / "haskie.lock"


_holding: int | None = None  # the file descriptor whose flock this process holds


def claim_home() -> None:
    """Claim this home for the calling process, or refuse: one haskie per home.

    A home is one SQLite file and one durable job pipeline, and a boot is a DBOS executor that
    recovers in-flight workflows and starts polling the queues. Two of them on the same file take
    each other's tasks. The TCP port is not the guard it looks like: a server runs its whole
    startup - migrations, `DBOS.launch`, re-enqueuing orphans - before it binds.

    So this runs as the app's first startup hook rather than in the CLI: `haskie run`, `litestar
    --app haskie.app:app run` and any other ASGI server all reach the same lifespan, and only the
    lifespan is a layer every one of them passes through.

    An advisory `flock`, not a pid file, because the kernel drops it when the holder dies: a
    crashed haskie leaves nothing to clean up. POSIX only, like the `fcntl` it imports; failing at
    import beats a guard that silently leaves the home unprotected.
    """
    from haskie.errors import Conflict  # local: `errors` imports this module

    global _holding
    if _holding is not None:  # `--reload` restarts run one lifespan per child, not per process
        return
    ensure_home_sync()
    # O_RDWR rather than a mode string: append mode ignores `seek`, and the holder line is
    # rewritten in place rather than accumulated.
    handle = os.open(LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        held = _holder_line(handle)
        os.close(handle)
        raise Conflict(held) from None
    os.ftruncate(handle, 0)
    # `run` puts the address in the environment for this; an app started another way has none to
    # give, so the holder line says so rather than inventing one.
    address = os.environ.get("HASKIE_ADDRESS", "address unknown")
    os.write(handle, f"pid {os.getpid()}, {address}".encode())
    _holding = handle


def _holder_line(handle: int) -> str:
    """The refusal, in the words both the startup hook and `run` report it in: one sentence with
    one owner, so the two can never disagree about what is already running."""
    held = os.read(handle, 256).decode("utf-8", "replace").strip() or "unknown process"
    return f"haskie is already running for {HOME} ({held})"


def home_holder() -> str | None:
    """What is running for this home, said in full, or None. Takes the lock and drops it again,
    so it answers without claiming anything: a caller that wants a clean message before it starts
    a server.

    Advisory only. The claim in the app's startup hook is the authority; this can go stale between
    the answer and the claim, and then the startup hook refuses instead.
    """
    if not LOCK_FILE.is_file():
        return None
    handle = os.open(LOCK_FILE, os.O_RDONLY)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return _holder_line(handle)
    finally:
        os.close(handle)
    return None


def release_home() -> None:
    """Give the home up. Closing the descriptor is what releases the lock."""
    global _holding
    if _holding is not None:
        os.close(_holding)
        _holding = None


def ensure_home_sync() -> Path:
    """The home tree, made. Sync because the callers that need it have no event loop: the startup
    hook that claims the lock, the CLI, and the test fixtures.

    `HOME` is made first and by name, because `parents=True` does not apply `mode` to the parents
    it creates - so a home made only as a parent of its subdirectories would be world-readable.
    """
    for directory in (HOME, COLLECTION_ROOT, DOCUMENT_ROOT, STAGING_ROOT, AUDIT_DIR):
        directory.mkdir(parents=True, exist_ok=True, mode=DIR_MODE)
    return HOME


async def ensure_home() -> Path:
    for directory in (COLLECTION_ROOT, DOCUMENT_ROOT, STAGING_ROOT, AUDIT_DIR):
        await anyio.Path(directory).mkdir(parents=True, exist_ok=True, mode=DIR_MODE)
    return HOME


@contextmanager
def atomic_replace(path: Path) -> Iterator[Path]:
    """Yield a temporary path to fill, then put it at `path` in one step: a reader sees either the
    previous file or the complete new one, never a partial write.

    The temporary file shares the directory so `os.replace` stays on one filesystem, and it is
    removed when the body raises, so a failure leaves nothing to mistake for a finished file.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_sync(path: Path, data: bytes | str, encoding: str = "utf-8") -> None:
    """One whole payload through `atomic_replace`, for code that already runs in a worker thread
    (see `convert.py`). Calling it from a coroutine blocks that event loop; await `atomic_write`
    there instead."""
    payload = data.encode(encoding) if isinstance(data, str) else data
    with atomic_replace(path) as tmp:
        tmp.write_bytes(payload)


async def atomic_write(path: Path, data: bytes | str, encoding: str = "utf-8") -> None:
    """`atomic_write_sync` off the event loop."""
    await anyio.to_thread.run_sync(atomic_write_sync, path, data, encoding)


async def remove_tree(path: Path) -> None:
    """Delete a directory and everything under it, in a worker thread. Best effort, but never
    silent: a file we cannot delete is logged, not swallowed."""
    # imported here, not at module level: `errors` and `logs` both import this module
    from haskie.errors import scrub
    from haskie.logs import get_logger

    log = get_logger(__name__)

    def report(_function: Callable[..., Any], failed: str, error: BaseException) -> None:
        log.warning("remove_failed", path=scrub(str(failed)), error=f"{type(error).__name__}")

    def remove() -> None:
        if not path.exists():
            return
        shutil.rmtree(path, onexc=report)

    await anyio.to_thread.run_sync(remove)
