"""Haskie home directory (~/.haskie) layout, and the file writes every module shares.

Every function here is async, because a filesystem call blocks: the writes go through
`anyio.Path`, the calls that have no async form (`os.replace`, `shutil.rmtree`) through a worker
thread. `atomic_write_sync` is the one exception, for code that already runs in a worker thread.
"""

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import anyio.to_thread

HOME = Path(os.environ.get("HASKIE_HOME", Path.home() / ".haskie"))
LIBRARY_ROOT = HOME / "library"
AUDIT_DIR = HOME / "audit"
DB_FILE = HOME / "haskie.db"
MODEL_CACHE = HOME / "cache" / "models"  # compiled CoreML models (see embed.py); ORT creates it

DIR_MODE = 0o700  # documents and the audit trail are private to the user running the app


def use(root: Path) -> None:
    """Point this process at another home, before anything has read the layout.

    The paths above are resolved once, at import, so `haskie --home` (and the test suite) cannot
    just set the environment variable: the derived paths would already be built from the old root.
    Nothing here re-opens what is already open, so this belongs at startup and nowhere else.
    """
    global HOME, LIBRARY_ROOT, AUDIT_DIR, DB_FILE, MODEL_CACHE
    HOME = root
    LIBRARY_ROOT = HOME / "library"
    AUDIT_DIR = HOME / "audit"
    DB_FILE = HOME / "haskie.db"
    MODEL_CACHE = HOME / "cache" / "models"


async def ensure_home() -> Path:
    for directory in (LIBRARY_ROOT, AUDIT_DIR):
        await anyio.Path(directory).mkdir(parents=True, exist_ok=True, mode=DIR_MODE)
    return HOME


async def atomic_write(path: Path, data: bytes | str, encoding: str = "utf-8") -> None:
    """Replace `path` in one step: a reader sees either the previous file or the complete new one.
    The temporary file shares the directory so `os.replace` stays on one filesystem."""
    payload = data.encode(encoding) if isinstance(data, str) else data
    tmp = path.with_name(path.name + ".tmp")
    try:
        await anyio.Path(tmp).write_bytes(payload)
        await anyio.to_thread.run_sync(os.replace, tmp, path)
    except BaseException:
        await anyio.Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_sync(path: Path, data: bytes | str, encoding: str = "utf-8") -> None:
    """`atomic_write` for code that already runs in a worker thread (see `convert.py`). Calling it
    from a coroutine blocks that event loop; await `atomic_write` there instead."""
    payload = data.encode(encoding) if isinstance(data, str) else data
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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
