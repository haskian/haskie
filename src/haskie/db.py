"""Metadata store: SQLite (WAL mode) at ~/.haskie/haskie.db.

Reads and writes go through `aiosqlite`, one connection per unit of work, so no unit ever blocks
the event loop it runs on. Creating the schema is the exception: it stays on stdlib `sqlite3` in a
worker thread, because the one-time switch to WAL needs an exclusive lock on the file.

Schema evolution: edit `SCHEMA` and bump `SCHEMA_VERSION`. There is no upgrade path, so a home at
any other version is refused and has to be destroyed (see `migrate`). A deliberate choice while the
storage shape is still moving: one readable schema is worth more than a history of scripts.
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
from haskie.errors import HaskieError

SCHEMA_VERSION = 16
"""`pragma user_version` of the schema below.

A home stamped with it has exactly these tables and is opened as it is. Any other stamp is a
shape this build cannot read, so the home is refused (see `migrate`). Against the last release
(15), a collection's LanceDB table holds `framed`, each chunk's heading path and text as the
models read them, and full-text search reads that column instead of `text`. A section of headings
alone no longer makes a chunk (`chunk.CHUNK_VERSION` 2), so its headings are found through the
chunks under them.

A cache file or LanceDB table written the old way must never be read by this build.

Before 1.0.0 this is the only migration there is, and it covers the stores this version does not
stamp as well. A change to what a chunk holds retires the embedding cache and every collection's
table, and rather than version each of them, the home is refused and rebuilt from the sources.
"""

# Every statement is `if not exists`, so a crash partway through leaves `user_version` at 0 and
# the next boot replays the script harmlessly.
#
# SQLite has no date type: a `timestamp` column documents what the value means, and its NUMERIC
# affinity stores the unix seconds `time.time()` returns as the float they are.
SCHEMA = """
    create table if not exists settings (
        id integer primary key check (id = 1),
        json text not null
    );

    create table if not exists sessions (
        id text primary key
    );

    create table if not exists collections (
        name text primary key,
        overrides text not null default '{}',
        description text not null default '',
        created_at timestamp not null default 0,
        pending_documents integer not null default 0,
        last_write_at timestamp,
        last_maintained_at timestamp,
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
        created_at timestamp not null default 0,
        updated_at timestamp not null default 0,
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
        added_at timestamp not null default 0,
        updated_at timestamp not null default 0,
        primary key (collection, document)
    );
    create index if not exists collection_documents_document
        on collection_documents (document);
    create index if not exists collection_documents_status
        on collection_documents (collection, status, document);

    -- the durable, content-addressed embedding cache (see indexing/embed_cache.py)
    create table if not exists embeddings (
        id text primary key,
        document text not null references documents (name) on delete cascade,
        urn text not null,
        model text not null,
        chunk_size integer not null,
        chunk_merge_below integer not null,
        chunk_frame integer not null,
        chunker text not null,
        chunk_version integer not null,
        parser text not null,
        skip_ocr_pages integer not null,
        rows integer not null default 0,
        bytes integer not null default 0,
        created_at timestamp not null default 0
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

    -- what a session did, so its history can be shown and an operation can name the session that
    -- started it; the rows go with the session
    create table if not exists session_events (
        id integer primary key,
        session_id text not null references sessions (id) on delete cascade,
        ts timestamp not null,
        action text not null,
        subject text not null,
        detail text not null default '{}',
        operation_id text,
        duration_ms integer not null default 0
    );
    create index if not exists session_events_session on session_events (session_id, ts);
    create index if not exists session_events_operation on session_events (operation_id);

    -- an upload waiting in `staging/`, before any name is taken
    create table if not exists staging (
        staging_id text primary key,
        filename text not null,
        size integer not null,
        created_at timestamp not null default 0
    );
    """


INCOMPATIBLE_HOME_MESSAGE = (
    "This version of haskie changed how documents are stored; the existing home is incompatible. "
    "Run `haskie destroy` and re-import your documents."
)

BUSY_TIMEOUT_SECONDS = 30.0  # how long a writer waits for another writer before it gives up

_migrated: set[Path] = set()
_migrate_lock = threading.Lock()  # both event loops get here through worker threads of their own


def migrate(conn: sqlite3.Connection) -> int:
    """Create the schema on a fresh file; returns the version the file is at.

    A home stamped with anything else was written by a build whose storage shape this one cannot
    read, and there is no path from it. It is refused with `user_version` untouched and the user is
    told to destroy it, rather than losing its rows to a silent drop."""
    (version,) = conn.execute("pragma user_version").fetchone()
    if version == SCHEMA_VERSION:
        return SCHEMA_VERSION
    if version != 0:
        raise HaskieError(INCOMPATIBLE_HOME_MESSAGE)
    conn.execute("pragma journal_mode = wal")  # persistent; needs an exclusive lock, so once
    conn.executescript(SCHEMA)
    conn.execute(f"pragma user_version = {SCHEMA_VERSION}")
    conn.commit()
    return SCHEMA_VERSION


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
    """Forget which database files this process has opened.

    The set is a per-process cache of "already at `SCHEMA_VERSION`". Deleting the file behind it
    (`haskie destroy`) leaves that claim false, so the next `migrate_once` has to run again.
    """
    with _migrate_lock:
        _migrated.clear()


async def migrate_once() -> None:
    """Make the home and create the schema, once per process and database file.

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
    raising "database is locked"; WAL (set when the schema is created) lets readers proceed.

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
