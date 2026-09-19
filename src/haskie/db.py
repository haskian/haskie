"""Metadata store: SQLite (WAL mode) at ~/.haskie/haskie.db.

Reads and writes go through `aiosqlite`, one connection per unit of work, so no unit ever blocks
the event loop it runs on. Migrations are the exception: they stay on stdlib `sqlite3` in a worker
thread, because the one-time switch to WAL needs an exclusive lock on the file.

Schema evolution: append a script to MIGRATIONS, never edit an applied one. `PRAGMA user_version`
records how many have run; the missing tail is applied once per process, before the first connect.
"""

import sqlite3
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
import anyio.to_thread
import msgspec

from haskie import home

MIGRATIONS: list[str] = [
    # 1: initial schema; libraries -> documents cascade on delete
    """
    create table if not exists settings (
        id integer primary key check (id = 1),
        json text not null
    );
    create table if not exists libraries (
        name text primary key,
        settings text not null default '{}'
    );
    create table if not exists documents (
        library text not null references libraries (name) on delete cascade,
        name text not null,
        size integer not null,
        status text not null default 'uploaded',
        error text,
        preview text,
        primary key (library, name)
    );
    create table if not exists sessions (
        id text primary key,
        libraries text not null
    );
    """,
    # 2: an early build kept its own job/task queues here; DBOS owns that now (same file).
    #    Kept so existing databases at user_version 2 stay numbered correctly; no-op when fresh.
    """
    drop table if exists tasks;
    drop table if exists jobs;
    """,
    # 3: listing timestamps + the indexes every paged listing sorts on. Existing rows have no
    #    history to recover, so they are stamped with the moment of the migration (unix seconds,
    #    the same clock `time.time()` writes) instead of staying at the epoch.
    """
    alter table libraries add column created_at real not null default 0;
    alter table documents add column created_at real not null default 0;
    alter table documents add column updated_at real not null default 0;
    update libraries set created_at = (julianday('now')-2440587.5)*86400.0 where created_at = 0;
    update documents set created_at = (julianday('now')-2440587.5)*86400.0,
        updated_at = (julianday('now')-2440587.5)*86400.0 where created_at = 0;
    create index if not exists documents_library_status  on documents (library, status, name);
    create index if not exists documents_library_updated on documents (library, updated_at, name);
    create index if not exists documents_library_size    on documents (library, size, name);
    """,
    # 4: per-library index maintenance bookkeeping (compaction, full-text and vector indexes).
    #    Existing libraries hold a table nobody ever compacted, so every one of them starts with
    #    one document pending: the first boot after this migration schedules a run for each.
    """
    alter table libraries add column pending_docs integer not null default 0;
    alter table libraries add column last_write_at real;
    alter table libraries add column last_maintained_at real;
    alter table libraries add column vector_index_rows integer not null default 0;
    update libraries set pending_docs = 1;
    """,
    # 5: job history retention (see archive.py). Only the watermark is schema: the day partitions
    #    `jobs_YYYYMMDD` / `tasks_YYYYMMDD` are created on demand and dropped whole, so they can
    #    never be part of a numbered migration.
    """
    create table if not exists retention_state (key text primary key, value text not null);
    insert or ignore into retention_state (key, value) values ('archive_watermark_ms', '0');
    """,
    # 6: facts about the home directory itself rather than about its rows. Its only key was
    #    `layout_version`, written by the layout migration that no longer exists, so the table has
    #    no reader left and migration 9 drops it. Kept here so the numbering stays as it shipped.
    """
    create table if not exists meta (key text primary key, value text not null);
    """,
    # 7: the libraries of a session become rows instead of a JSON column, so a deleted library
    #    leaves every session it was chosen in through the same cascade that drops its documents
    #    (no read-modify-write over every session row). `position` keeps the caller's order.
    #    A name in the JSON that no longer has a library row is dropped rather than migrated.
    """
    create table if not exists session_libraries (
        session_id text not null references sessions (id) on delete cascade,
        library text not null references libraries (name) on delete cascade,
        position integer not null,
        primary key (session_id, library)
    );
    create index if not exists session_libraries_library on session_libraries (library);
    insert or ignore into session_libraries (session_id, library, position)
        select s.id, j.value, j.key from sessions s, json_each(s.libraries) j
        where j.value in (select name from libraries);
    alter table sessions drop column libraries;
    """,
    # 8: a human label for a library and for a document, set on create/upload and editable after
    """
    alter table libraries add column description text not null default '';
    alter table documents add column description text not null default '';
    """,
    # 9: collections replace libraries; documents become collection-independent and many-to-many
    #    through collection_documents; embeddings become a durable, content-addressed cache, and
    #    an upload waiting in `staging/` gets a row instead of a sidecar file. A
    #    breaking storage-shape change: nothing is reshaped in place. A home that reaches this
    #    migration with real libraries/documents rows is refused before it runs (see `migrate`),
    #    so the script is free to drop and recreate. One transaction: a crash mid-script must not
    #    leave `user_version` at 8 over a half-dropped schema.
    """
    begin;
    drop table if exists session_libraries;
    drop table if exists documents;
    drop table if exists libraries;
    drop table if exists meta;

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

    create table if not exists staging (
        staging_id text primary key,
        filename text not null,
        size integer not null,
        created_at text not null
    );
    commit;
    """,
]

# The migration that changed the storage shape (see MIGRATIONS[8]) and the message a home holding
# data from before it gets instead of a silent drop.
INCOMPATIBLE_HOME_MIGRATION = 9
INCOMPATIBLE_HOME_MESSAGE = (
    "This version of haskie changed how documents are stored; the existing home is incompatible. "
    "Run `haskie destroy` and re-import your documents."
)

BUSY_TIMEOUT_SECONDS = 30.0  # how long a writer waits for another writer before it gives up

_migrated: set[Path] = set()
_migrate_lock = threading.Lock()  # both event loops migrate through worker threads of their own


def _holds_pre_collection_data(conn: sqlite3.Connection) -> bool:
    """Whether the pre-refactor `libraries`/`documents` tables hold any row. A table that does not
    exist (a home that never ran migration 1 with data, or a fresh file) holds nothing."""
    for table in ("libraries", "documents"):
        found = conn.execute(
            "select 1 from sqlite_master where type = 'table' and name = ?", (table,)
        ).fetchone()
        if found is not None and conn.execute(f"select 1 from {table} limit 1").fetchone():
            return True
    return False


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations in order; returns the resulting schema version.

    Checked inside the loop, right before migration 9 runs, rather than on the starting version:
    a home may start at 5, 6, 7 or 8 and reach 9 in the same run either way. A home with real
    pre-collection rows is refused there, with `user_version` left at 8, so the user can destroy
    it and start over instead of losing the rows silently."""
    from haskie.errors import HaskieError  # errors imports home, which imports nothing of ours

    (version,) = conn.execute("pragma user_version").fetchone()
    if version == 0:
        conn.execute("pragma journal_mode = wal")  # persistent; needs an exclusive lock, so once
    for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
        if number == INCOMPATIBLE_HOME_MIGRATION and _holds_pre_collection_data(conn):
            raise HaskieError(INCOMPATIBLE_HOME_MESSAGE)
        conn.executescript(script)
        conn.execute(f"pragma user_version = {number}")
        conn.commit()
    return len(MIGRATIONS)


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


def dumps(value: msgspec.Struct | list | dict) -> str:
    return msgspec.json.encode(value).decode()


def loads[T](raw: str | None, type_: type[T]) -> T | None:
    return None if raw is None else msgspec.json.decode(raw, type=type_)
