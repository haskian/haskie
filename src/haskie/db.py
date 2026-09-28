"""Metadata store: SQLite (WAL mode) at ~/.haskie/haskie.db.

The tables are SQLAlchemy Core (`tables.py`), and every query is a Core statement run on an
`AsyncConnection` over `aiosqlite`, one connection per unit of work, so no unit ever blocks the
event loop it runs on. Creating the schema is the exception: it stays on stdlib `sqlite3` in a
worker thread, because the one-time switch to WAL needs an exclusive lock on the file.

Schema evolution: edit `tables.py` and bump `SCHEMA_VERSION`. There is no upgrade path, so a home
at any other version is refused and has to be destroyed (see `migrate`). A deliberate choice while
the storage shape is still moving: one readable schema is worth more than a history of scripts.
"""

import asyncio
import sqlite3
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any

import anyio.to_thread
import msgspec
from sqlalchemy import Column, Connection, Row, Table, event
from sqlalchemy.dialects import sqlite
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateIndex, CreateTable

from haskie import home, tables
from haskie.errors import HaskieError

SCHEMA_VERSION = 24
"""`pragma user_version` of the schema in `tables.py`.

A home stamped with it has these tables and columns and is opened as it is. Any other stamp is a
shape this build cannot read, so the home is refused (see `migrate`). Against the last release (23),
searches move out of `session_events` into the search log (`searches`, `search_questions`,
`search_results`), and the embedding profiles gain the cosines the Gaps page judges by
(`weak_match`, `answered_match`, `same_topic`). Against 22, a document and a staged upload keep the
MD5 of their bytes (`md5`), and an embedding cache row keeps the document's mean vector
(`embeddings.vector`). Against 21, where two parts of a document meet is cut `part` or `heading`,
not `edge` (`CutReason.PART`). Against 20, the catalogue holds each reranker's calibration
(`reranker_calibration`). Against 19, a piece without a word (`---`, a stray symbol, a page marker
alone) makes no chunk (`chunk.pack`): every document chunks differently, and a search no longer
checks for such chunks, so an index written the old way would return them.

A cache file or LanceDB table written the old way must never be read by this build.

Before 1.0.0 this is the only migration there is, and it covers the stores this version does not
stamp as well. A change to what a chunk holds retires the embedding cache and every collection's
table, and rather than version each of them, the home is refused and rebuilt from the sources.
"""


def schema_ddl() -> str:
    """The DDL of every table and index in `tables.metadata`, as one script.

    Every statement is `if not exists`, so a crash partway through leaves `user_version` at 0 and
    the next boot replays the script harmlessly."""
    dialect = sqlite.dialect()
    statements: list[Any] = []
    for table in tables.metadata.sorted_tables:
        statements.append(CreateTable(table, if_not_exists=True))
        statements += [CreateIndex(index, if_not_exists=True) for index in table.indexes]
    return "".join(
        f"{str(statement.compile(dialect=dialect)).strip()};\n" for statement in statements
    )


# The rows a fresh home starts with. A data file of the `catalogue` feature, read rather than
# imported, so this module stays a leaf.
SEED = files("haskie.catalogue") / "seed.sql"

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
    conn.executescript(schema_ddl())
    conn.executescript(SEED.read_text(encoding="utf-8"))
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
        _engines[home.DB_FILE] = asyncio.run(_first_connected_engine())
        _migrated.add(home.DB_FILE)


def invalidate_migrations() -> None:
    """Forget which database files this process has opened.

    The set is a per-process cache of "already at `SCHEMA_VERSION`". Deleting the file behind it
    (`haskie destroy`) leaves that claim false, so the next `migrate_once` has to run again.
    """
    with _migrate_lock:
        _migrated.clear()
        _engines.clear()


async def migrate_once() -> None:
    """Make the home and create the schema, once per process and database file.

    The one-time WAL switch needs an exclusive lock, so this runs before anything else (DBOS, or
    the first `connect()`) holds the file open."""
    if home.DB_FILE in _migrated:
        return
    await home.ensure_home()
    await anyio.to_thread.run_sync(_migrate_sync)


def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
    # the adapter's own `execute`: one round trip to the driver thread, where a cursor takes three
    dbapi_connection.execute("pragma synchronous = normal")  # durable across crashes in WAL mode
    dbapi_connection.execute("pragma foreign_keys = on")  # per connection


_READ_ONLY = "haskie_read_only"  # the execution option `read` sets, which `_begin` reads


def _begin(conn: Connection) -> None:
    """A unit of work that may write takes the write lock at its start, so what it read still
    holds when it writes: a unit that checks, then writes, never acts on a check another commit
    made stale.

    The driver's own transaction control is off (`isolation_level=None`): it would open none
    before a SELECT, so a check ran outside the transaction its write joined later. A plain
    (deferred) `begin` would make the check part of the transaction, but a commit landing
    between the check and the write then fails the write at once ("database is locked"): the
    busy timeout does not cover a stale snapshot, and DBOS commits to this file all the time.
    `immediate` makes that commit wait instead. Measured on one WAL file with a writer
    committing every millisecond: deferred failed every read-then-write unit, immediate none.

    The price is that units wait for each other and for DBOS's writers, so a unit that only
    reads (`read`) opens a deferred transaction instead: it never writes, so no stale snapshot
    can fail it, and in WAL it waits for no one. `query_only` makes a write through it fail
    loudly. The pragma outlives the transaction, which is harmless under `NullPool`: the
    connection is closed when the unit ends."""
    if conn.get_execution_options().get(_READ_ONLY):
        conn.exec_driver_sql("pragma query_only = 1")
        conn.exec_driver_sql("begin")
    else:
        conn.exec_driver_sql("begin immediate")


# One engine per database file, made by `_migrate_sync`: tests switch homes, and `haskie destroy`
# deletes the file.
_engines: dict[Path, AsyncEngine] = {}


async def _first_connected_engine() -> AsyncEngine:
    """A new engine for the current home, connected once already.

    `NullPool` opens a connection per unit of work and closes it after, so no connection is ever
    shared between the two event loops (Litestar's and DBOS's), and one engine serves both.
    `timeout` makes a unit wait for the write lock another unit or DBOS holds, instead of raising
    "database is locked" (see `_begin`); WAL (set when the schema is created) lets
    DBOS's readers proceed.

    SQLAlchemy guards an engine's first connection with an asyncio lock, which binds to the loop
    that waits on it. The two loops racing for that first connection fail with "bound to a
    different event loop", so it happens here, on a private loop, before either can reach it."""
    made = create_async_engine(
        f"sqlite+aiosqlite:///{home.DB_FILE}",
        poolclass=NullPool,
        connect_args={"timeout": BUSY_TIMEOUT_SECONDS, "isolation_level": None},
    )
    event.listen(made.sync_engine, "connect", _set_pragmas)
    event.listen(made.sync_engine, "begin", _begin)
    async with made.connect():
        pass
    return made


def engine() -> AsyncEngine:
    """The engine of the current home's database file; `migrate_once` has made it."""
    return _engines[home.DB_FILE]


@asynccontextmanager
async def connect() -> AsyncIterator[AsyncConnection]:
    """One connection and one transaction per unit of work that writes, holding the write lock
    from its first statement (see `_begin`); commits on success, rolls back on error. Never open
    one while holding another: the inner unit would wait for the outer one's lock."""
    await migrate_once()
    async with engine().begin() as conn:
        yield conn


@asynccontextmanager
async def read() -> AsyncIterator[AsyncConnection]:
    """One connection and one deferred transaction per unit of work that only reads (see
    `_begin`). It takes no lock, so it waits neither for other units nor for DBOS's writers.

    Every statement of the unit still reads one snapshot: in WAL, a transaction's first read fixes
    the commit it sees, and a commit landing after it stays invisible until the unit ends. A write
    through it raises ("attempt to write a readonly database"). Nothing is committed: the unit
    rolls back when it ends."""
    await migrate_once()
    async with engine().connect() as conn:
        await conn.execution_options(**{_READ_ONLY: True})
        yield conn


def record(row: Row[Any]) -> dict[str, Any]:
    """A row as a dict keyed by column name. The names SQLAlchemy returns are a `str` subclass,
    which msgspec refuses as a key, so each one is turned back into a plain `str`."""
    return {str(name): value for name, value in row._mapping.items()}


def columns_of(table: Table, struct: type[msgspec.Struct]) -> tuple[Column[Any], ...]:
    """The columns of `table` that `struct` reads, one per field, so a SELECT of them and the
    struct cannot drift apart."""
    return tuple(table.c[field.encode_name] for field in msgspec.structs.fields(struct))


def row_to[T](struct: type[T], row: Row[Any], **json_columns: type) -> T:
    """One selected row as a struct, its columns matched to the fields by name.

    `strict=False` so the integers sqlite stores for booleans arrive as the bools the struct
    declares. A column named in `json_columns` holds JSON text (sqlite has no struct type) and is
    decoded into that type first.
    """
    values = record(row)
    for name, type_ in json_columns.items():
        values[name] = loads(values[name], type_)
    return msgspec.convert(values, struct, strict=False)


def dumps(value: msgspec.Struct | list | dict) -> str:
    return msgspec.json.encode(value).decode()


def loads[T](raw: str | None, type_: type[T]) -> T | None:
    return None if raw is None else msgspec.json.decode(raw, type=type_)
