"""Haskie home directory (~/.haskie) layout, and the file writes every module shares.

A filesystem call blocks, so every async function here runs its work in a worker thread. The sync
ones (`atomic_replace`, `atomic_write_sync`) are for code that already runs in one: `convert.py`
and `embed_cache.py` are sync by nature (pyarrow, the parsers) and would otherwise hop threads
twice for one write.
"""

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

DIR_MODE = 0o700  # documents and the audit trail are private to the user running the app


def use(root: Path) -> None:
    """Point this process at another home, before anything has read the layout.

    The paths above are resolved once, at import, so `haskie --home` (and the test suite) cannot
    just set the environment variable: the derived paths would already be built from the old root.
    Nothing here re-opens what is already open, so this belongs at startup and nowhere else.
    """
    global HOME, COLLECTION_ROOT, DOCUMENT_ROOT, STAGING_ROOT, AUDIT_DIR, DB_FILE, MODEL_CACHE
    HOME = root
    COLLECTION_ROOT = HOME / "collections"
    DOCUMENT_ROOT = HOME / "documents"
    STAGING_ROOT = HOME / "staging"
    AUDIT_DIR = HOME / "audit"
    DB_FILE = HOME / "haskie.db"
    MODEL_CACHE = HOME / "cache" / "models"


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
