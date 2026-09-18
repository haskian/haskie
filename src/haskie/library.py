"""Libraries: metadata in the DB, files under ~/.haskie/library/<name>/.

Per-document files are sharded over 256 prefix directories keyed by the document name (`shard`):
`files/<shard>/<doc>`, `markdown/<shard>/<doc>.md`, `markdown/<shard>/<doc>.parts/`,
`preview/<shard>/<doc>/`. `layout.py` moves an older, flat home into this shape once at startup.

Document lifecycle: uploaded -> queued -> converting -> embedding -> indexing -> indexed,
ending in error or cancelled instead. The preview is built lazily on first open, in any state.

Every row read, row write and file touch is awaited: the database goes through `db.connect()`
(aiosqlite), the files through `anyio.Path` and `home`, and the one piece of CPU work here — the
preview build — through `cpu.on_cpu`. The pure parts (paths, name cleaning, row decoding, keyset
helpers) stay sync.
"""

import re
import shutil
import threading
import time
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, get_args

import aiosqlite
import anyio
import anyio.to_thread
import msgspec

from haskie import convert, cpu, db, home
from haskie.errors import (
    Conflict,
    DocumentNotFound,
    InvalidInput,
    LibraryNotFound,
    NotReady,
    UnsupportedFileType,
    scrub,
)
from haskie.index import Hit, IndexStats, LibraryIndex, forget_schema
from haskie.layout import PART_DIGITS, shard
from haskie.paging import Keyset, Page, PageRequest, resolve_sort
from haskie.settings import (
    ConversionSettings,
    EmbeddingModel,
    LibrarySettings,
    SearchSettings,
    load_user_settings,
)

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
UPLOAD_MAX_BYTES = 512 * 1024 * 1024  # also the HTTP request body cap (see app.create_app)

DocStatus = Literal[
    "uploaded", "queued", "converting", "embedding", "indexing", "indexed", "error", "cancelled"
]
DOCUMENT_STATUSES: tuple[DocStatus, ...] = get_args(DocStatus)
# in the pipeline right now: the states a poll waits on, and the reason a listing keeps refreshing
ACTIVE_STATUSES: tuple[DocStatus, ...] = ("queued", "converting", "embedding", "indexing")

# Public sort name -> SQL expression. The whitelist is the only source of column identifiers a
# listing can order by, so a request can never name a column (see paging.resolve_sort).
LIBRARY_SORTS = {"name": "name", "created_at": "created_at"}
DOCUMENT_SORTS = {"name": "name", "size": "size", "status": "status", "updated_at": "updated_at"}

# aiosqlite annotates every fetched row as `sqlite3.Row`, but `db.connect()` leaves the default row
# factory alone, so a row really is a tuple. The reads below say `Any` where the row is unpacked or
# handed to a reader that takes a tuple, rather than repeating that note at every site.


class Document(msgspec.Struct):
    name: str
    size: int
    status: DocStatus
    error: str | None = None
    preview: convert.Preview | None = None
    created_at: float = 0.0  # unix seconds; 0.0 only for rows that predate migration 3
    updated_at: float = 0.0
    description: str = ""  # what the document is, in the uploader's words


# The struct's field order is the column order, so the SELECT, the row unpack and the keyset key
# reader cannot drift apart.
DOCUMENT_COLUMNS: tuple[str, ...] = tuple(f.encode_name for f in msgspec.structs.fields(Document))
DOCUMENT_SELECT = ", ".join(DOCUMENT_COLUMNS)


class DocumentCounts(msgspec.Struct):
    """How a library's documents are spread over the lifecycle, counted in the database: the
    listing that replaced it is paged, so a caller can no longer count the rows it received."""

    total: int = 0
    indexed: int = 0
    active: int = 0
    error: int = 0
    by_status: dict[str, int] = msgspec.field(default_factory=dict)


class MaintenanceState(msgspec.Struct):
    """The maintenance columns of one library row. `maintenance` decides what to do about them;
    the row they live in belongs to `Library`."""

    pending_docs: int
    last_write_at: float | None
    last_maintained_at: float | None
    vector_index_rows: int


MAINTENANCE_COLUMNS = "pending_docs, last_write_at, last_maintained_at, vector_index_rows"


class LibrarySummary(msgspec.Struct):
    """One row of the library listing: enough for a sidebar, without reading any document."""

    name: str
    counts: DocumentCounts
    created_at: float
    description: str = ""


class IndexStatus(msgspec.Struct):
    """The library's LanceDB table as it is right now, plus how its maintenance stands.
    Read on demand (`Library.info`), never stored: the table is its own source of truth."""

    num_rows: int
    num_fragments: int
    num_small_fragments: int
    has_fts_index: bool
    has_vector_index: bool
    unindexed_rows: int
    vector_index_rows: int
    last_maintained_at: float | None
    pending_docs: int


class LibraryInfo(msgspec.Struct):
    name: str
    settings: LibrarySettings
    effective: ConversionSettings
    search: SearchSettings
    description: str
    counts: DocumentCounts  # the documents themselves are paged, at /api/libraries/{name}/documents
    index_outdated: bool = False  # index built by an older version or embedding; "Index all" fixes
    index: IndexStatus | None = None  # None until the library has a table


def _index_status(
    stats: IndexStats, last_maintained_at: float | None, pending_docs: int
) -> IndexStatus:
    """`IndexStatus` is `IndexStats` plus the two maintenance columns, so it is built from it
    rather than field by field: a new stat reaches the API by being added once."""
    return IndexStatus(
        **msgspec.structs.asdict(stats),
        last_maintained_at=last_maintained_at,
        pending_docs=pending_docs,
    )


def safe_name(name: str) -> str:
    cleaned = SAFE_NAME.sub("-", name).strip("-.")
    if not cleaned:
        raise InvalidInput(f"invalid name: {name!r}")
    return cleaned


def _document(row: tuple) -> Document:
    values = dict(zip(DOCUMENT_COLUMNS, row, strict=True))
    values["preview"] = db.loads(values["preview"], convert.Preview)
    return Document(**values)


def _counts(by_status: dict[str, int]) -> DocumentCounts:
    """Roll one `status -> count` mapping up into the shape every caller reads."""
    return DocumentCounts(
        total=sum(by_status.values()),
        indexed=by_status.get("indexed", 0),
        active=sum(by_status.get(status, 0) for status in ACTIVE_STATUSES),
        error=by_status.get("error", 0),
        by_status=by_status,
    )


async def _counts_by_library(
    conn: aiosqlite.Connection, names: list[str]
) -> dict[str, DocumentCounts]:
    """Counts for the names of one page in one grouped query, rather than one query per library.

    Takes the open connection of the listing, so the counts come from the unit of work that read
    the page rather than from a connection of their own."""
    if not names:
        return {}
    by_library: dict[str, dict[str, int]] = {name: {} for name in names}
    marks = db.placeholders(len(names))
    cursor = await conn.execute(
        f"select library, status, count(*) from documents where library in ({marks}) group by 1, 2",
        names,
    )
    for library, status, count in await cursor.fetchall():
        by_library[library][status] = count
    return {name: _counts(by_status) for name, by_status in by_library.items()}


def _keyset(sort: str, expression: str, request: PageRequest) -> Keyset:
    """`name` is the primary key of both listings, so it breaks every tie; when it is also the
    sort column it is the whole keyset rather than a column repeated twice."""
    columns = [expression] if sort == "name" else [expression, "name"]
    return Keyset(sort, columns, request.order, request)


def _key_reader(sort: str, expression: str, selected: list[str]) -> Callable[[tuple], list[Any]]:
    """Reads the keyset columns out of a row of `selected`, in the order `_keyset` built them."""
    name = selected.index("name")
    if sort == "name":
        return lambda row: [row[name]]
    value = selected.index(expression)
    return lambda row: [row[value], row[name]]


# Fixed stripes of locks, so two readers of the same document build the preview once. Striped
# rather than one lock per document: a process that opens a million documents still holds 64 locks.
# Two documents may share a stripe; the second one then waits for a build it does not need, which
# is rare and harmless.
#
# An `anyio.Lock` rather than a threading one: a preview is only ever built on Litestar's event
# loop (the handler that opens a document), so one set of loop primitives covers every builder, and
# a waiting reader yields its loop instead of blocking it. Each stripe is made on first use rather
# than at import, so no lock exists before there is a loop to await it on.
PREVIEW_LOCK_STRIPES = 64
_preview_locks: list[anyio.Lock | None] = [None] * PREVIEW_LOCK_STRIPES


def _preview_lock(library: str, doc: str) -> anyio.Lock:
    stripe = zlib.crc32(f"{library}/{doc}".encode()) % PREVIEW_LOCK_STRIPES
    lock = _preview_locks[stripe]
    if lock is None:  # no await in between, so two readers of one stripe cannot both make one
        lock = _preview_locks[stripe] = anyio.Lock()
    return lock


# A preview build parses a whole document, so a burst of opens would otherwise start one parse per
# request. The semaphore admits `indexing.preview_workers` of them; the rest wait, and a reader
# that waited this long is told to retry instead of holding its request open forever.
PREVIEW_WAIT_SECONDS = 60
_preview_workers = 2  # the default of `PipelineSettings.preview_workers`
_preview_slots = anyio.Semaphore(_preview_workers)


def configure_preview_slots(workers: int) -> None:
    """Resize the pool of preview builders. A build in flight releases the semaphore it acquired,
    so it is unaffected by a resize under it.

    Called from `workflows.apply_settings`, which runs on Litestar's event loop (a settings
    handler, or startup) — the same loop every preview waits on, as `_preview_lock` describes.
    """
    global _preview_workers, _preview_slots
    if workers != _preview_workers:
        _preview_workers, _preview_slots = workers, anyio.Semaphore(workers)


# Library settings are read on every search, listing and pipeline step, and written only through
# `set_settings` below, so the decoded struct is cached per library for this process. Anything that
# writes the `libraries.settings` column another way must call `invalidate_library_caches()`.
_settings_cache: dict[str, LibrarySettings] = {}
# A threading lock rather than an async one, because both event loops of this process (Litestar's
# and DBOS's) read library settings. It is never held across an `await` — a reader that misses runs
# its query outside the lock and then keeps whatever a writer stored meanwhile (see `settings`).
_settings_lock = threading.Lock()


def invalidate_library_caches() -> None:
    """Forget every cached library settings struct, so the next read goes to the database."""
    with _settings_lock:
        _settings_cache.clear()


def _forget_settings(name: str) -> None:
    with _settings_lock:
        _settings_cache.pop(name, None)


def _cache_settings(found: dict[str, LibrarySettings]) -> None:
    """Store rows just read, without overwriting a value a writer cached while we were reading:
    the writer's row is the newer one (see `set_settings`)."""
    with _settings_lock:
        for name, value in found.items():
            _settings_cache.setdefault(name, value)


class Library:
    def __init__(self, name: str) -> None:
        """Sync and IO-free: a `Library` is a name and the paths derived from it. `get` is the
        constructor that checks the library exists."""
        self.name = name
        self.root = home.LIBRARY_ROOT / name
        self.home = home.HOME  # everything stored in the index is relative to it
        self.files = self.root / "files"
        self.markdown = self.root / "markdown"
        self.previews = self.root / "preview"
        self.index_dir = self.root / "index"

    # --- lifecycle -------------------------------------------------------

    @staticmethod
    async def names() -> list[str]:
        """Every name, for callers inside the process (sessions, workflows). The API pages
        instead, through `page` below."""
        async with db.connect() as conn:
            cursor = await conn.execute("select name from libraries order by name")
            return [row[0] for row in await cursor.fetchall()]

    @staticmethod
    async def page(request: PageRequest) -> Page[LibrarySummary]:
        """One page of libraries with their document counts: a keyset walk over `libraries`,
        then a single grouped count for the names on the page."""
        sort, expression = resolve_sort(request.sort, LIBRARY_SORTS, "name")
        keyset = _keyset(sort, expression, request)
        boundary, params = keyset.where()
        selected = ["name", "created_at", "description"]
        async with db.connect() as conn:
            cursor = await conn.execute(
                f"select {', '.join(selected)} from libraries "
                f"{f'where {boundary} ' if boundary else ''}"
                f"{keyset.order_by()} limit {keyset.limit()}",
                params,
            )
            rows: list[Any] = list(await cursor.fetchall())
            cursor = await conn.execute("select count(*) from libraries")
            (total,) = await cursor.fetchone() or (0,)  # a count always returns its one row
            counts = await _counts_by_library(conn, [row[0] for row in rows[: request.page_size]])
        return keyset.page(
            rows,
            build=lambda row: LibrarySummary(
                name=row[0], counts=counts[row[0]], created_at=row[1], description=row[2]
            ),
            key=_key_reader(sort, expression, selected),
            total=total,
        )

    @classmethod
    async def create(cls, name: str, description: str = "") -> "Library":
        lib = cls(safe_name(name))
        async with db.connect() as conn:
            cursor = await conn.execute(
                "insert into libraries (name, created_at, description) values (?, ?, ?) "
                "on conflict (name) do nothing",
                (lib.name, time.time(), description),
            )
            created = cursor.rowcount == 1  # read on the open connection, before it is closed
        if not created:
            raise Conflict(f"library already exists: {lib.name}")
        _forget_settings(lib.name)  # a read before the insert may have cached the defaults
        for directory in (lib.files, lib.markdown, lib.previews):
            await anyio.Path(directory).mkdir(parents=True, exist_ok=True)
        return lib

    @classmethod
    async def get(cls, name: str) -> "Library":
        async with db.connect() as conn:
            cursor = await conn.execute("select 1 from libraries where name = ?", (name,))
            row = await cursor.fetchone()
        if row is None:
            raise LibraryNotFound(f"library not found: {name}")
        return cls(name)

    async def delete(self) -> None:
        """Delete the library. `delete_*` is the whole operation, `remove_*` is one step of it.

        Rows first, then the folder: while the row exists the library is still listed, so a
        crash in between leaves a library that can be deleted again rather than a phantom."""
        await self.remove_rows()
        await self.remove_tree()

    async def remove_rows(self) -> None:
        """Row delete cascades to the documents and to every session that chose the library
        (`session_libraries`); `pragma foreign_keys = on` is set on every connection."""
        async with db.connect() as conn:
            await conn.execute("delete from libraries where name = ?", (self.name,))
        _forget_settings(self.name)  # a library created again under this name starts clean

    async def remove_tree(self) -> None:
        """Files, markdown, previews and the index table of the library."""
        forget_schema(self.index_dir)
        await home.remove_tree(self.root)

    # --- settings --------------------------------------------------------

    async def settings(self) -> LibrarySettings:
        with _settings_lock:
            cached = _settings_cache.get(self.name)
        if cached is not None:
            return cached
        # The query runs outside the lock: a threading lock held across an `await` would block
        # every other reader, event loop included. Two readers that miss at the same time both
        # read the same row, which is harmless, and neither can replace a value a writer cached
        # meanwhile (see `_cache_settings`).
        async with db.connect() as conn:
            cursor = await conn.execute(
                "select settings from libraries where name = ?", (self.name,)
            )
            row = await cursor.fetchone()
        value = (db.loads(row[0], LibrarySettings) if row else None) or LibrarySettings()
        with _settings_lock:
            return _settings_cache.setdefault(self.name, value)

    async def set_settings(self, value: LibrarySettings) -> None:
        async with db.connect() as conn:
            await conn.execute(
                "update libraries set settings = ? where name = ?", (db.dumps(value), self.name)
            )
        with _settings_lock:  # after the commit, so a reader cannot cache the previous value
            _settings_cache[self.name] = value

    @staticmethod
    async def load_settings(names: list[str]) -> dict[str, LibrarySettings]:
        """The settings of several libraries in one query, keyed by name. A name with no row is
        absent from the result, which is how a caller learns the library is gone.

        A session search used to ask `Library.get` and then read the settings once per library;
        this is one SELECT for all of them, and it fills the per-library cache on the way.
        """
        wanted = list(dict.fromkeys(names))
        if not wanted:
            return {}
        marks = db.placeholders(len(wanted))
        async with db.connect() as conn:
            cursor = await conn.execute(
                f"select name, settings from libraries where name in ({marks})", wanted
            )
            rows = await cursor.fetchall()
        found = {name: (db.loads(raw, LibrarySettings) or LibrarySettings()) for name, raw in rows}
        _cache_settings(found)
        return found

    @staticmethod
    async def reranker_overrides() -> list[str]:
        """Every reranker model a library overrides, in name order and without duplicates.

        One query over the `settings` column: the model downloads have to cover the overrides too,
        and a search of that library loads whichever model it names, whether or not the library
        also overrides `reranker` itself. Fills the per-library cache on the way, like
        `load_settings` does."""
        async with db.connect() as conn:
            cursor = await conn.execute("select name, settings from libraries order by name")
            rows = await cursor.fetchall()
        found = {name: (db.loads(raw, LibrarySettings) or LibrarySettings()) for name, raw in rows}
        _cache_settings(found)
        chosen = [v.search.reranker_model for v in found.values() if v.search.reranker_model]
        return list(dict.fromkeys(chosen))

    async def effective_settings(self) -> ConversionSettings:
        return (await self.settings()).resolve(await load_user_settings())

    async def search_settings(self) -> SearchSettings:
        return (await self.settings()).resolve_search(await load_user_settings())

    async def info(self) -> LibraryInfo:
        settings, user = await self.settings(), await load_user_settings()
        index = self.index_with(user.embedding_model)  # one handle: each opens its own connection
        return LibraryInfo(
            name=self.name,
            settings=settings,
            effective=settings.resolve(user),
            search=settings.resolve_search(user),
            description=await self.description(),
            counts=await self.counts(),
            index_outdated=not await index.schema_current(),
            index=await self._index_status(index),
        )

    async def description(self) -> str:
        """Empty for a library that has none, and for one that no longer exists."""
        async with db.connect() as conn:
            cursor = await conn.execute(
                "select description from libraries where name = ?", (self.name,)
            )
            row = await cursor.fetchone()
        return row[0] if row else ""

    async def describe(self, description: str) -> None:
        """Replace the library's description. Empty clears it."""
        async with db.connect() as conn:
            await conn.execute(
                "update libraries set description = ? where name = ?", (description, self.name)
            )

    async def describe_document(self, doc: str, description: str) -> Document:
        """Replace one document's description. Empty clears it."""
        await self.document(doc)  # DocumentNotFound before anything is written
        async with db.connect() as conn:
            await conn.execute(
                "update documents set description = ? where library = ? and name = ?",
                (description, self.name, doc),
            )
        return await self.document(doc)

    async def describe_of(self, docs: set[str]) -> dict[str, str]:
        """The descriptions of several documents in one query, keyed by name.

        A document with none is absent from the result. Batched because the caller is a search
        shortlist: one query per library beats one per document."""
        if not docs:
            return {}
        names = sorted(docs)
        async with db.connect() as conn:
            cursor = await conn.execute(
                "select name, description from documents "
                f"where library = ? and name in ({db.placeholders(len(names))}) "
                "and description != ''",
                (self.name, *names),
            )
            rows = await cursor.fetchall()
        return {name: description for name, description in rows}

    # --- maintenance columns ---------------------------------------------
    # `maintenance` decides when a run is due and what it does; these are the four columns of the
    # library row it decides from, so they are read and written here.

    @staticmethod
    async def pending_names() -> list[str]:
        """Libraries with documents indexed since their last finished run: what a boot
        reschedules."""
        async with db.connect() as conn:
            cursor = await conn.execute(
                "select name from libraries where pending_docs > 0 order by name"
            )
            rows = await cursor.fetchall()
        return [name for (name,) in rows]

    async def maintenance_state(self) -> MaintenanceState | None:
        """None when the library has no row: it was never created, or it was deleted while a run
        waited. A run treats that as a skip, so the absence has to stay visible."""
        async with db.connect() as conn:
            cursor = await conn.execute(
                f"select {MAINTENANCE_COLUMNS} from libraries where name = ?", (self.name,)
            )
            row = await cursor.fetchone()
        return MaintenanceState(*row) if row is not None else None

    async def note_indexed(self) -> int:
        """One more document indexed; returns how many are pending a run. Counted in one
        statement, so two documents finishing at the same moment both count."""
        async with db.connect() as conn:
            cursor = await conn.execute(
                "update libraries set pending_docs = pending_docs + 1, last_write_at = ? "
                "where name = ? returning pending_docs",
                (time.time(), self.name),
            )
            row = await cursor.fetchone()
        return row[0] if row is not None else 0

    async def settle_maintenance(self, claimed: int, retrained: bool, num_rows: int) -> None:
        """Record a finished run: subtract what it claimed, stamp it, and remember the rows the
        vector index was trained on when it retrained. A skipped run settles too, or the library
        would stay pending for ever and be rescheduled at every boot."""
        async with db.connect() as conn:
            await conn.execute(
                "update libraries set pending_docs = max(0, pending_docs - ?), "
                "last_maintained_at = ?, "
                "vector_index_rows = case when ? then ? else vector_index_rows end "
                "where name = ?",
                (claimed, time.time(), retrained, num_rows, self.name),
            )

    async def _index_status(self, index: LibraryIndex) -> IndexStatus | None:
        """None while the library has no table."""
        stats = await index.stats()
        if stats is None:
            return None
        state = await self.maintenance_state() or MaintenanceState(0, None, None, 0)
        return _index_status(stats, state.last_maintained_at, state.pending_docs)

    async def search(self, query: str, limit: int | None = None, **overrides) -> list[Hit]:
        """`limit` and any SearchSettings field can be overridden per call."""
        settings = await self.search_settings()
        if limit:
            overrides["limit"] = limit
        if overrides:
            settings = msgspec.structs.replace(settings, **overrides)
        index = await self.index()
        return [self.resolve_hit(hit) for hit in await index.search(query, settings)]

    def resolve_hit(self, hit: Hit) -> Hit:
        """Recompute the file paths of a hit from the document name, in place.

        The index stores them home-relative so the home stays portable, but a row written before
        the sharded layout (or by a build old enough to store no path at all) still points a
        caller at the files that exist now. Changing where a document lives must never mean
        reindexing a library, so the paths are derived on read rather than migrated.
        """
        hit.source_path = self.relative(self.file_path(hit.doc))
        hit.markdown_path = self.relative(self.markdown_path(hit.doc))
        hit.source_file = str(self.file_path(hit.doc))
        hit.markdown_file = str(self.markdown_path(hit.doc))
        return hit

    async def index(self) -> LibraryIndex:
        return self.index_with((await load_user_settings()).embedding_model)

    def index_with(self, embedding: EmbeddingModel | None) -> LibraryIndex:
        """Variant without the settings read, for steps that already hold the embedding model.
        Sync, like the `LibraryIndex` constructor it calls: opening the table is what awaits."""
        return LibraryIndex(self.index_dir, self.name, self.home, embedding)

    # --- documents -------------------------------------------------------

    async def document_names(self, after: str | None = None, limit: int | None = None) -> list[str]:
        """Names alone, ordered by name: what a caller that only iterates documents needs, without
        reading the preview blob of every row.

        `after` resumes the walk past that name and `limit` caps the page, so a bulk job can walk
        a large library one page at a time instead of holding every name at once."""
        filters: list[str] = ["library = ?"]
        params: list[Any] = [self.name]
        if after is not None:
            filters.append("name > ?")
            params.append(after)
        limited = " limit ?" if limit is not None else ""
        if limit is not None:
            params.append(limit)
        async with db.connect() as conn:
            cursor = await conn.execute(
                f"select name from documents where {' and '.join(filters)} order by name{limited}",
                params,
            )
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def counts(self) -> DocumentCounts:
        async with db.connect() as conn:
            cursor = await conn.execute(
                "select status, count(*) from documents where library = ? group by 1",
                (self.name,),
            )
            rows = await cursor.fetchall()
        return _counts({status: count for status, count in rows})

    async def documents_page(
        self, request: PageRequest, status: DocStatus | None = None
    ) -> Page[Document]:
        """One page of the library's documents, optionally of one status. `total` counts the
        filtered rows, so it is what the page is a page of."""
        sort, expression = resolve_sort(request.sort, DOCUMENT_SORTS, "name")
        keyset = _keyset(sort, expression, request)
        filters, params = ["library = ?"], [self.name]
        if status is not None:
            filters.append("status = ?")
            params.append(status)
        filtered = " and ".join(filters)
        boundary, boundary_params = keyset.where()
        where = f"{filtered} and {boundary}" if boundary else filtered
        async with db.connect() as conn:
            cursor = await conn.execute(
                f"select {DOCUMENT_SELECT} from documents where {where} "
                f"{keyset.order_by()} limit {keyset.limit()}",
                [*params, *boundary_params],
            )
            rows: list[Any] = list(await cursor.fetchall())
            cursor = await conn.execute(f"select count(*) from documents where {filtered}", params)
            (total,) = await cursor.fetchone() or (0,)  # a count always returns its one row
        return keyset.page(
            rows,
            build=_document,
            key=_key_reader(sort, expression, list(DOCUMENT_COLUMNS)),
            total=total,
        )

    async def document(self, doc: str) -> Document:
        async with db.connect() as conn:
            cursor = await conn.execute(
                f"select {DOCUMENT_SELECT} from documents where library = ? and name = ?",
                (self.name, doc),
            )
            row: Any = await cursor.fetchone()
        if row is None:
            raise DocumentNotFound(f"document not found: {doc}")
        return _document(row)

    async def set_status(self, doc: str, status: DocStatus, error: str | None = None) -> None:
        """A lifecycle step is a change to the document, so it stamps `updated_at`: that is the
        column the "recently touched" listing sorts on. Building the preview is not (see
        `ensure_preview`), it only fills in what the row always described."""
        async with db.connect() as conn:
            await conn.execute(
                "update documents set status = ?, error = ?, updated_at = ? "
                "where library = ? and name = ?",
                (status, error, time.time(), self.name, doc),
            )

    def file_path(self, doc: str) -> Path:
        """Where the upload belongs, whether or not it is there. `source_path` is this plus the
        check, for the readers that need the file to exist."""
        return self.files / shard(doc) / doc

    def source_path(self, doc: str) -> Path:
        path = self.file_path(doc)
        if not path.is_file():
            raise DocumentNotFound(f"document file missing: {doc}")
        return path

    def markdown_path(self, doc: str) -> Path:
        return self.markdown / shard(doc) / f"{doc}.md"

    def preview_dir(self, doc: str) -> Path:
        return self.previews / shard(doc) / doc

    def relative(self, path: Path) -> str:
        """Path as stored in the index: relative to the haskie home, so the folder is portable."""
        return path.relative_to(self.home).as_posix()

    def parts_dir(self, doc: str) -> Path:
        return self.markdown / shard(doc) / f"{doc}.parts"

    def part_path(self, doc: str, seq: int) -> Path:
        return self.parts_dir(doc) / f"{seq:0{PART_DIGITS}d}.md"

    def rows_path(self, doc: str, seq: int) -> Path:
        """Chunks + vectors of one part (output of the embed stage)."""
        return self.parts_dir(doc) / f"{seq:0{PART_DIGITS}d}.rows.json"

    def _stored_name(self, filename: str, rename_to: str | None = None) -> str:
        """The name an upload is stored under, or the reason it is refused. Both ways into the
        library (`save`, `save_path`) accept exactly what this accepts.

        `rename_to` stores the file under a name of the caller's choosing. The suffix decides how
        the document is parsed, so a rename that drops or changes it keeps the original's: the
        uploader is naming the document, not choosing a parser.
        """
        chosen = Path(rename_to).name if rename_to else Path(filename).name
        suffix = Path(filename).suffix.lower()
        if rename_to and Path(chosen).suffix.lower() != suffix:
            chosen += suffix
        name = safe_name(chosen)
        if Path(name).suffix.lower() not in convert.SUPPORTED_SUFFIXES:
            raise UnsupportedFileType(f"unsupported file type: {name}")
        return name

    async def _record_upload(self, name: str, size: int, description: str = "") -> Document:
        """The row of a file that is already on disk: the last step of both ways in.

        A re-upload with no description keeps the one the document already had: replacing the file
        is not the same as clearing what it is."""
        now = time.time()
        async with db.connect() as conn:
            await conn.execute(
                "insert into documents (library, name, size, created_at, updated_at, description) "
                "values (?, ?, ?, ?, ?, ?) "
                "on conflict (library, name) do update set size = excluded.size, "
                "status = 'uploaded', error = null, preview = null, "
                "description = coalesce(nullif(excluded.description, ''), documents.description), "
                "updated_at = excluded.updated_at",  # created_at keeps the first upload's moment
                (self.name, name, size, now, now, description),
            )
        return await self.document(name)

    async def save(
        self, filename: str, content: bytes, rename_to: str | None = None, description: str = ""
    ) -> Document:
        """Store the upload as `uploaded`. Preview and indexing happen later, on demand.

        A pipeline running on the same document must be cancelled first (the caller does it:
        this module knows nothing about workflows).
        """
        name = self._stored_name(filename, rename_to)
        if len(content) > UPLOAD_MAX_BYTES:
            raise InvalidInput(f"file larger than {UPLOAD_MAX_BYTES} bytes: {len(content)}")
        target = self.file_path(name)
        # the shard directory, on first use
        await anyio.Path(target.parent).mkdir(parents=True, exist_ok=True)
        await home.atomic_write(target, content)
        return await self._record_upload(name, len(content), description)

    async def save_path(
        self, path: str, rename_to: str | None = None, description: str = ""
    ) -> Document:
        """Import by absolute path: a trust boundary, so the path is checked before it is read.

        The file is copied rather than read into memory: an import may be as large as the upload
        cap allows, and `shutil.copyfile` streams it (in a worker thread, so nothing blocks).
        """
        source = Path(path).expanduser()
        if not source.is_absolute():
            raise InvalidInput(f"path must be absolute: {scrub(str(source))}")
        if not await anyio.Path(source).is_file():
            raise InvalidInput(f"file not found: {scrub(str(source))}")
        name = self._stored_name(source.name, rename_to)
        size = (await anyio.Path(source).stat()).st_size
        if size > UPLOAD_MAX_BYTES:
            raise InvalidInput(f"file larger than {UPLOAD_MAX_BYTES} bytes: {size}")
        target = self.file_path(name)
        await anyio.Path(target.parent).mkdir(parents=True, exist_ok=True)
        await anyio.to_thread.run_sync(shutil.copyfile, source, target)
        return await self._record_upload(name, size, description)

    async def ensure_preview(self, doc: str) -> Document:
        """Build the side-by-side preview (first pages only for PDF) once, on first open.

        Two locks, always in this order: the per-document stripe (build this document once), then
        a slot in the process-wide pool (build at most `preview_workers` documents at a time).
        The parse itself is CPU work, so it runs in a worker thread under the CPU budget.
        """
        info = await self.document(doc)
        if info.preview is not None:
            return info
        async with _preview_lock(self.name, doc):
            info = await self.document(doc)  # another reader may have built it while we waited
            if info.preview is not None:
                return info
            settings = await self.effective_settings()
            slots = _preview_slots  # the object to release, even if the pool is resized meanwhile
            try:
                with anyio.fail_after(PREVIEW_WAIT_SECONDS):
                    await slots.acquire()
            except TimeoutError:
                raise NotReady("preview queue is full; retry") from None
            try:
                preview = await cpu.on_cpu(
                    "preview",
                    convert.build_preview,
                    self.source_path(doc),
                    self.preview_dir(doc),
                    settings.parser,
                    settings.skip_ocr_pages,
                )
                async with db.connect() as conn:
                    await conn.execute(
                        "update documents set preview = ? where library = ? and name = ?",
                        (db.dumps(preview), self.name, doc),
                    )
            finally:
                slots.release()
            return await self.document(doc)

    # --- removal ---------------------------------------------------------
    # Three steps so a workflow can run each one durably; the DB row goes last, so a crash
    # leaves a document that can be removed again instead of orphaned files.

    async def remove_index_rows(self, doc: str) -> None:
        await (await self.index()).delete_document(doc)

    async def remove_files(self, doc: str) -> None:
        await anyio.Path(self.file_path(doc)).unlink(missing_ok=True)
        await anyio.Path(self.markdown_path(doc)).unlink(missing_ok=True)
        await home.remove_tree(self.parts_dir(doc))
        await home.remove_tree(self.preview_dir(doc))

    async def remove_row(self, doc: str) -> None:
        async with db.connect() as conn:
            await conn.execute(
                "delete from documents where library = ? and name = ?", (self.name, doc)
            )

    async def remove_document(self, doc: str) -> None:
        """Index rows, then files, then the row."""
        await self.document(doc)
        await self.remove_index_rows(doc)
        await self.remove_files(doc)
        await self.remove_row(doc)
