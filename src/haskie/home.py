"""Haskie home directory (~/.haskie): where everything lives, and the file writes every module
shares.

`~/.haskie/documents/` and `~/.haskie/collections/` each hold one folder per entry. Ten thousand
documents would make ten thousand entries in one directory, which every lookup and every listing
pays for, so both roots insert a shard directory (`shard`) between them and the entry: an entry
lives at `<root>/<shard>/<name>/`, and a root spreads over 256 directories.

A filesystem call blocks, so every async function here runs its work in a worker thread. The sync
ones (`atomic_replace`, `atomic_write_sync`) are for code that already runs in one: `convert.py`
and `embed_cache.py` are sync by nature (pyarrow, the parsers) and would otherwise hop threads
twice for one write.

The home lock (`claim_home` and friends) is sync for a different reason: it runs before there
is an event loop at all, as the app's first startup hook.
"""

import fcntl
import hashlib
import os
import re
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import anyio
import anyio.to_thread

# `logs` imports this module back, for `scrub`. Safe either way round: each side reaches for the
# other only when it is called, never while the module body runs.
from haskie import logs
from haskie.errors import Conflict

HOME = Path(os.environ.get("HASKIE_HOME", Path.home() / ".haskie"))

# The layout, relative to `HOME`. Every name here is readable as a module attribute
# (`home.DB_FILE`) and derived on access, so `use()` has one global to rebind.
_LAYOUT: dict[str, str] = {
    "COLLECTION_ROOT": "collections",  # one LanceDB index per collection
    "DOCUMENT_ROOT": "documents",  # one folder per imported document: original, markdown, cache
    "STAGING_ROOT": "staging",  # uploads not yet imported; swept by the nightly maintenance
    "AUDIT_DIR": "audit",
    "DB_FILE": "haskie.db",
    "MODEL_CACHE": "cache/models",  # compiled CoreML models (see embed.py); ORT creates it
    "LOCK_FILE": "haskie.lock",  # one running haskie per home (see `claim_home`)
}
_MADE = ("COLLECTION_ROOT", "DOCUMENT_ROOT", "STAGING_ROOT", "AUDIT_DIR")  # the rest are files

DIR_MODE = 0o700  # documents and the audit trail are private to the user running the app
PART_DIGITS = 6  # width of a micro-batch sequence number; four would cap a document at 10k parts


def __getattr__(name: str) -> Path:
    if name in _LAYOUT:
        return HOME / _LAYOUT[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def use(root: Path) -> None:
    """Point this process at another home, before anything has read the layout.

    Nothing here re-opens what is already open, so this belongs at startup and nowhere else.
    """
    global HOME
    HOME = root


def shard(name: str) -> str:
    """The prefix directory an entry lives in: the first byte of the SHA-1 of its name, hex.

    The hash is over the UTF-8 bytes of the name, so it is stable across platforms and locales,
    and it is a name, not a secret: SHA-1 is used for its spread, not for its strength.
    """
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:2]


def part_name(seq: int) -> str:
    """The stem of one micro-batch file, zero-padded so a listing sorts in sequence order."""
    return f"{seq:0{PART_DIGITS}d}"


def scrub(text: str) -> str:
    """Replace absolute paths with their symbolic root, for log lines and error bodies.
    The home directory is replaced first because it usually sits inside the user directory."""
    return text.replace(str(HOME), "$HASKIE_HOME").replace(str(Path.home()), "~")


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
    global _holding
    if _holding is not None:  # `--reload` restarts run one lifespan per child, not per process
        return
    ensure_home_sync()
    # O_RDWR rather than a mode string: append mode ignores `seek`, and the holder line is
    # rewritten in place rather than accumulated.
    handle = os.open(HOME / _LAYOUT["LOCK_FILE"], os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        held = os.read(handle, HOLDER_BYTES).decode("utf-8", "replace")
        os.close(handle)
        raise Conflict(_holder_line(held)) from None
    os.ftruncate(handle, 0)
    # `run` puts the address in the environment for this; an app started another way has none to
    # give, so the holder line says so rather than inventing one.
    address = os.environ.get("HASKIE_ADDRESS", "address unknown")
    os.write(handle, f"pid {os.getpid()}, {address}".encode())
    _holding = handle


HOLDER_BYTES = 256  # the holder line is one short sentence; anything longer is not one of ours
_HOLDER_PID = re.compile(r"pid (\d+)")  # reads back the line `claim_home` writes above


def _holder_line(held: str) -> str:
    """The refusal, in the words both the startup hook and `run` report it in: one sentence with
    one owner, so the two can never disagree about what is already running."""
    return f"haskie is already running for {HOME} ({held.strip() or 'unknown process'})"


def _held_by() -> str | None:
    """What the holder wrote into the lock, or None when nothing holds this home.

    One probe for every caller that asks about the lock without claiming it: take the lock, and
    the answer is that it was free; fail to take it, and the holder's own line says who has it.

    Advisory only. The claim in the app's startup hook is the authority; this can go stale between
    the answer and the claim, and then the startup hook refuses instead.
    """
    lock = HOME / _LAYOUT["LOCK_FILE"]
    if not lock.is_file():
        return None
    handle = os.open(lock, os.O_RDONLY)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return os.read(handle, HOLDER_BYTES).decode("utf-8", "replace")
    finally:
        os.close(handle)
    return None


def home_holder() -> str | None:
    """What is running for this home, said in full, or None: for a caller that wants a clean
    message before it starts a server."""
    held = _held_by()
    return None if held is None else _holder_line(held)


def running_pid() -> int | None:
    """The process id of the haskie holding this home, or None when nothing holds it.

    The lock is what says a haskie is running, and the pid `claim_home` writes into it is how a
    caller reaches that process: `stop` signals it. A held lock whose line is not ours (an
    interrupted write, an older format) reads as no pid rather than as a pid to signal.
    """
    held = _held_by()
    found = None if held is None else _HOLDER_PID.match(held)
    return int(found[1]) if found else None


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
    for directory in (HOME, *(HOME / _LAYOUT[name] for name in _MADE)):
        directory.mkdir(parents=True, exist_ok=True, mode=DIR_MODE)
    return HOME


async def ensure_home() -> Path:
    """`ensure_home_sync` off the event loop."""
    return await anyio.to_thread.run_sync(ensure_home_sync)


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
    log = logs.get_logger(__name__)

    def report(_function: Callable[..., Any], failed: str, error: BaseException) -> None:
        log.warning("remove_failed", path=scrub(str(failed)), error=f"{type(error).__name__}")

    def remove() -> None:
        if not path.exists():
            return
        shutil.rmtree(path, onexc=report)

    await anyio.to_thread.run_sync(remove)
