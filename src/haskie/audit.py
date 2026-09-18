"""Audit trail: who did what, one JSON line per action under `HASKIE_HOME/audit/`.

The file append is the durable sink, so no log level can drop a record; the `haskie.audit` logger
only mirrors it into the normal stream.
"""

import functools
import inspect
import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import anyio
import msgspec
import structlog

from haskie import APP_VERSION, errors, home
from haskie.logs import AUDIT

DEFAULT_ACTOR = "web"  # no request context: a worker thread or a direct call
LEVEL_NAME = "AUDIT"
FILE_MODE = 0o600
# The daily files `path` writes, and the only ones `prune` may delete.
FILE_NAME = re.compile(r"^audit-(\d{4}-\d{2}-\d{2})\.jsonl\Z")
# Names `attach` fills on the record itself; anything else it receives goes into `detail`.
RECORD_FIELDS = frozenset({"library", "doc", "session_id", "workflow_id"})

# A plain stdlib logger: structlog's BoundLogger only knows the five standard levels, and the
# ProcessorFormatter's ExtraAdder renders `extra` into the same JSON fields anyway.
_log = logging.getLogger("haskie.audit")


class AuditRecord(msgspec.Struct, omit_defaults=True):
    """`level` is required rather than defaulted so `omit_defaults` keeps it on every line and
    still drops the optional fields a given event does not use."""

    ts: str
    level: str
    event: str
    actor: str
    outcome: str
    duration_ms: int
    app_version: str
    request_id: str | None = None
    session_id: str | None = None
    workflow_id: str | None = None
    library: str | None = None
    doc: str | None = None
    error: str | None = None
    detail: dict[str, str | int | bool] | None = None


def path(when: datetime | None = None) -> Path:
    """One file per UTC day."""
    return home.AUDIT_DIR / f"audit-{(when or datetime.now(UTC)):%Y-%m-%d}.jsonl"


async def prune(retention_days: int, now: datetime | None = None) -> int:
    """Delete the daily audit files older than `retention_days`; returns how many were deleted.

    `retention_days` 0 keeps everything. Only files whose name is a date this module wrote are
    considered, so anything else the user put in the directory is left alone.
    """
    directory = anyio.Path(home.AUDIT_DIR)
    if retention_days <= 0 or not await directory.is_dir():
        return 0
    cutoff = (now or datetime.now(UTC)).date() - timedelta(days=retention_days)
    deleted = 0
    async for file in directory.iterdir():
        match = FILE_NAME.match(file.name)
        if match is None:
            continue
        try:
            day = date.fromisoformat(match.group(1))
        except ValueError:  # a well-formed name that is not a real date, e.g. 2026-02-31
            continue
        if day < cutoff:
            await file.unlink(missing_ok=True)
            deleted += 1
    return deleted


async def _append(entry: AuditRecord) -> None:
    await anyio.Path(home.AUDIT_DIR).mkdir(parents=True, exist_ok=True, mode=home.DIR_MODE)
    # `touch` only applies the mode when it creates the file; the append mode below opens O_APPEND,
    # which keeps each line atomic across threads and processes.
    file = anyio.Path(path())
    await file.touch(mode=FILE_MODE, exist_ok=True)
    async with await anyio.open_file(file, "ab") as handle:
        await handle.write(msgspec.json.encode(entry) + b"\n")


async def record(
    event: str, *, actor: str, outcome: str, duration_ms: int, **fields: Any
) -> AuditRecord:
    """Append one record and mirror it to the `haskie.audit` logger at level AUDIT."""
    entry = AuditRecord(
        ts=datetime.now(UTC).isoformat(),
        level=LEVEL_NAME,
        event=event,
        actor=actor,
        outcome=outcome,
        duration_ms=duration_ms,
        app_version=APP_VERSION,
        **fields,
    )
    await _append(entry)
    mirrored = {
        key: value
        for key, value in msgspec.structs.asdict(entry).items()
        # the renderer adds its own event, level and timestamp
        if value is not None and key not in ("event", "level", "ts")
    }
    _log.log(AUDIT, event, extra=mirrored)
    return entry


def _context() -> tuple[str, str | None]:
    """Actor and request id bound by the request middleware; defaults outside a request."""
    context = structlog.contextvars.get_contextvars()
    actor = context.get("actor", DEFAULT_ACTOR)
    request_id = context.get("request_id")
    return str(actor), None if request_id is None else str(request_id)


# Fields the running handler added with `attach`; None outside an `audited` call.
_attached: ContextVar[dict[str, Any] | None] = ContextVar("haskie_audit_attached", default=None)


def attach(**fields: str | int | bool) -> None:
    """Add fields to the record of the `audited` call in progress; a no-op outside one.

    A name in `RECORD_FIELDS` fills that field, every other name goes into `detail`. Only
    identifiers, names and counts belong here: never a setting value, a path or file content.
    """
    attached = _attached.get()
    if attached is not None:
        attached.update(fields)


async def _finish(
    event: str, fields: dict[str, str], started: float, exc: BaseException | None
) -> None:
    actor, request_id = _context()
    attached = _attached.get() or {}
    named = {k: v for k, v in attached.items() if k in RECORD_FIELDS}
    detail = {k: v for k, v in attached.items() if k not in RECORD_FIELDS}
    await record(
        event,
        actor=actor,
        outcome="ok" if exc is None else "error",
        duration_ms=int((time.perf_counter() - started) * 1000),
        request_id=request_id,
        error=None if exc is None else errors.scrub(f"{type(exc).__name__}: {exc}"),
        detail=detail or None,
        **{**fields, **named},
    )


def audited(
    event: str,
    *,
    library: str | None = None,
    doc: str | None = None,
    session_id: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate an async handler so every call appends one audit record, then re-raise on failure.

    `library`, `doc` and `session_id` name the decorated function's parameters whose values are
    copied into the record; `attach` adds what the handler only knows once it runs. The wrapper
    keeps the wrapped signature because Litestar builds its dependency injection from
    `inspect.signature`, so it must take no parameter of its own.
    """
    sources = {"library": library, "doc": doc, "session_id": session_id}
    fields_from = {field: name for field, name in sources.items() if name is not None}

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(func)

        def fields(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, str]:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            return {
                field: str(bound.arguments[name])
                for field, name in fields_from.items()
                if bound.arguments.get(name) is not None
            }

        @asynccontextmanager
        async def recording(args: tuple[Any, ...], kwargs: dict[str, Any]) -> AsyncIterator[None]:
            """Time the call, collect what it attaches, and append the record either way."""
            started = time.perf_counter()
            token = _attached.set({})
            try:
                yield
            except BaseException as exc:
                await _finish(event, fields(args, kwargs), started, exc)
                raise
            else:
                await _finish(event, fields(args, kwargs), started, None)
            finally:
                _attached.reset(token)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with recording(args, kwargs):
                return await func(*args, **kwargs)

        return wrapper

    return decorate
