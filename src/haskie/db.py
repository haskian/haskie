"""Metadata store: SQLite (WAL mode) at ~/.haskie/haskie.db.

Reads and writes go through `aiosqlite`, one connection per unit of work, so no unit ever blocks
the event loop it runs on. Migrations are the exception: they stay on stdlib `sqlite3` in a worker
thread, because the one-time switch to WAL needs an exclusive lock on the file.

Schema evolution: append a script to MIGRATIONS, never edit an applied one. `PRAGMA user_version`
records the version reached; the missing tail is applied once per process, before the first
connect. A home older than `SCHEMA_VERSION` is refused rather than migrated (see `migrate`).
"""

import sqlite3
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite
import anyio.to_thread
import msgspec

from haskie import home

SCHEMA_VERSION = 9
"""`pragma user_version` of the schema script below.

The number is where the migration history that built this schema stopped, so a home stamped with
it already has the script's tables and runs nothing. A later change appends a script to
MIGRATIONS and is stamped SCHEMA_VERSION + its index.
"""

MIGRATIONS: list[str] = [
    # The whole schema, in one script: every statement is `if not exists`, so a crash partway
    # through leaves `user_version` at 0 and the next boot replays it harmlessly.
    """
    create table if not exists settings (
        id integer primary key check (id = 1),
        json text not null
    );

    create table if not exists sessions (
        id text primary key
    );

    create table if not exists collections (
        name text primary key,
        settings text not null default '{}',
        description text not null default '',
        created_at real not null default 0,
        pending_docs integer not null default 0,
        last_write_at real,
        last_maintained_at real,
        vector_index_rows integer not null default 0
    );

    -- a document belongs to no collection: `collection_documents` is the many-to-many, and each
    -- membership carries the status of writing that document into that collection's table
    create table if not exists documents (
        name text primary key,
        suffix text not null,
        size integer not null,
        status text not null default 'queued',
        error text,
        preview text,
        parser text not null default 'anydoc',
        skip_ocr_pages integer not null default 1,
        created_at real not null default 0,
        updated_at real not null default 0,
        description text not null default ''
    );
    create index if not exists documents_status  on documents (status, name);
    create index if not exists documents_updated on documents (updated_at, name);
    create index if not exists documents_size    on documents (size, name);

    create table if not exists collection_documents (
        collection text not null references collections (name) on delete cascade,
        document text not null references documents (name) on delete cascade,
        status text not null default 'pending',
        error text,
        added_at real not null default 0,
        updated_at real not null default 0,
        primary key (collection, document)
    );
    create index if not exists collection_documents_document
        on collection_documents (document);
    create index if not exists collection_documents_status
        on collection_documents (collection, status, document);

    -- the durable, content-addressed embedding cache (see embed_cache.py)
    create table if not exists embeddings (
        id text primary key,
        document text not null references documents (name) on delete cascade,
        urn text not null,
        model text not null,
        chunk_size integer not null,
        chunk_overlap integer not null,
        chunker text not null,
        chunk_version integer not null,
        parser text not null,
        skip_ocr_pages integer not null,
        rows integer not null default 0,
        bytes integer not null default 0,
        created_at real not null default 0
    );
    create index if not exists embeddings_document on embeddings (document);

    create table if not exists session_collections (
        session_id text not null references sessions (id) on delete cascade,
        collection text not null references collections (name) on delete cascade,
        position integer not null,
        primary key (session_id, collection)
    );
    create index if not exists session_collections_collection
        on session_collections (collection);

    -- an upload waiting in `staging/`, before any name is taken
    create table if not exists staging (
        staging_id text primary key,
        filename text not null,
        size integer not null,
        created_at real not null default 0
    );
    """,
]

INCOMPATIBLE_HOME_MESSAGE = (
    "This version of haskie changed how documents are stored; the existing home is incompatible. "
    "Run `haskie destroy` and re-import your documents."
)

BUSY_TIMEOUT_SECONDS = 30.0  # how long a writer waits for another writer before it gives up

_migrated: set[Path] = set()
_migrate_lock = threading.Lock()  # both event loops migrate through worker threads of their own


def latest_version() -> int:
    """The version the scripts in MIGRATIONS add up to."""
    return SCHEMA_VERSION + len(MIGRATIONS) - 1


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations in order; returns the resulting schema version.

    A home stamped below `SCHEMA_VERSION` was written by a build whose storage shape no longer
    exists. There is no path from it, so it is refused with `user_version` untouched and the user
    is told to destroy it, rather than losing its rows to a silent drop."""
    from haskie.errors import HaskieError  # errors imports home, which imports nothing of ours

    (version,) = conn.execute("pragma user_version").fetchone()
    if 0 < version < SCHEMA_VERSION:
        raise HaskieError(INCOMPATIBLE_HOME_MESSAGE)
    if version == 0:
        conn.execute("pragma journal_mode = wal")  # persistent; needs an exclusive lock, so once
    applied = max(version - SCHEMA_VERSION + 1, 0)  # a fresh file has applied none
    for number, script in enumerate(MIGRATIONS[applied:], start=SCHEMA_VERSION + applied):
        conn.executescript(script)
        conn.execute(f"pragma user_version = {number}")
        conn.commit()
    return latest_version()


def _migrate_sync() -> None:
    """Runs in a worker thread, on a stdlib connection of its own: `executescript` and the WAL
    switch are one blocking burst, and the lock is a threading one because the callers are the two
    event loops of this process (Litestar's and DBOS's), not one."""
    with _migrate_lock:
        if home.DB_FILE in _migrated:
            return
        conn = sqlite3.connect(str(home.DB_FILE), timeout=BUSY_TIMEOUT_SECONDS)
        try:
            migrate(conn)
        finally:
            conn.close()
        _migrated.add(home.DB_FILE)


def invalidate_migrations() -> None:
    """Forget which database files this process has migrated.

    The set is a per-process cache of "already at the latest schema". Deleting the file behind it
    (`haskie destroy`) leaves that claim false, so the next `migrate_once` has to run again.
    """
    with _migrate_lock:
        _migrated.clear()


async def migrate_once() -> None:
    """Make the home and apply the pending migrations, once per process and database file.

    The one-time WAL switch needs an exclusive lock, so this runs before anything else (DBOS, or
    the first `connect()`) holds the file open."""
    if home.DB_FILE in _migrated:
        return
    await home.ensure_home()
    await anyio.to_thread.run_sync(_migrate_sync)


@asynccontextmanager
async def connect() -> AsyncIterator[aiosqlite.Connection]:
    """One connection per unit of work; commits on success, rolls back on error.

    A connection is never shared between the two event loops, because it never outlives the unit
    of work that opened it. `timeout` makes concurrent writers (DBOS, requests) wait instead of
    raising "database is locked"; WAL (set at migration time) lets readers proceed during a write.

    The default row factory is left alone, so every fetched row is a plain tuple; `aiosqlite`
    types it as `sqlite3.Row` anyway, which is why the reads elsewhere say `Any`.
    """
    await migrate_once()
    async with aiosqlite.connect(home.DB_FILE, timeout=BUSY_TIMEOUT_SECONDS) as conn:
        await conn.execute("pragma synchronous = normal")  # durable across crashes in WAL mode
        await conn.execute("pragma foreign_keys = on")  # per connection
        try:
            yield conn
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise


def placeholders(count: int) -> str:
    """`?, ?, ?` for a SQL `in (...)` list or a `values (...)` row."""
    return ", ".join(["?"] * count)


def row_to[T](struct: type[T], columns: tuple[str, ...], row: tuple, **json_columns: type) -> T:
    """One selected row as a struct, `columns` naming what the row holds in its order.

    `strict=False` so the integers sqlite stores for booleans arrive as the bools the struct
    declares. A column named in `json_columns` holds JSON text (sqlite has no struct type) and is
    decoded into that type first.
    """
    values: dict[str, Any] = dict(zip(columns, row, strict=True))
    for name, type_ in json_columns.items():
        values[name] = loads(values[name], type_)
    return msgspec.convert(values, struct, strict=False)


def dumps(value: msgspec.Struct | list | dict) -> str:
    return msgspec.json.encode(value).decode()


def loads[T](raw: str | None, type_: type[T]) -> T | None:
    return None if raw is None else msgspec.json.decode(raw, type=type_)
