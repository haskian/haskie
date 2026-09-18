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
    # 6: facts about the home directory itself rather than about its rows. The only one so far is
    #    `layout_version` (see layout.py). The row is written by `layout.migrate_layout` once it
    #    has finished moving, not here: an old home with a fresh database must still be migrated.
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
]

BUSY_TIMEOUT_SECONDS = 30.0  # how long a writer waits for another writer before it gives up

_migrated: set[Path] = set()
_migrate_lock = threading.Lock()  # both event loops migrate through worker threads of their own


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations in order; returns the resulting schema version."""
    (version,) = conn.execute("pragma user_version").fetchone()
    if version == 0:
        conn.execute("pragma journal_mode = wal")  # persistent; needs an exclusive lock, so once
    for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
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
