"""Collections: a named set of documents with one LanceDB index and its own chunk and search
settings. Metadata in the DB, the index under ~/.haskie/collections/<shard>/<name>/index/.

A collection owns nothing about a document but its membership (`collection_documents`) and the
rows it wrote into its own table. The same document may sit in any number of collections; each
attach is an indexing operation of its own (embed-if-missing from the document's cache, then a
write into this collection's table), so a membership carries its own status — a document can be
`indexed` in one collection and `error` in another at the same time. Deleting a collection drops
its rows and its folder and touches no document.

Settings are read on every search, listing and pipeline step, and written only through
`set_settings`, so the decoded struct is cached per collection for this process. Anything that
writes the `collections.settings` column another way must call `invalidate_collection_caches()`.

Every row read, row write and file touch is awaited: the database goes through `db.connect()`
(aiosqlite), the files through `anyio.Path` and `home`, the table through `CollectionIndex`. The
pure parts (paths, row decoding, `resolve_hit`) stay sync.
"""

import threading
import time
from typing import Any, Literal

import aiosqlite
import anyio
import msgspec

from haskie import db, document, home
from haskie.errors import CollectionNotFound, Conflict, DocumentNotFound
from haskie.index import CollectionIndex, Hit, IndexStats, forget_schema
from haskie.layout import shard
from haskie.paging import Page, PageRequest, key_reader, keyset, resolve_sort
from haskie.settings import (
    ChunkSettings,
    CollectionSettings,
    EmbeddingModel,
    SearchSettings,
    load_user_settings,
)

MemberStatus = Literal["pending", "indexing", "indexed", "error", "cancelled"]
# being written into the collection right now: the states a poll waits on
ACTIVE_MEMBER_STATUSES: tuple[MemberStatus, ...] = ("pending", "indexing")

# Public sort name -> SQL expression. The whitelist is the only source of column identifiers a
# listing can order by, so a request can never name a column (see paging.resolve_sort).
COLLECTION_SORTS = {"name": "name", "created_at": "created_at"}
# The member listing joins `documents d` with `collection_documents cd`: the sort expressions
# name the table where a column exists in both, and `name` (documents only) breaks ties.
MEMBER_SORTS = {
    "name": "d.name",
    "size": "d.size",
    "status": "cd.status",
    "updated_at": "cd.updated_at",
}

# aiosqlite annotates every fetched row as `sqlite3.Row`, but `db.connect()` leaves the default row
# factory alone, so a row really is a tuple. The reads below say `Any` where the row is unpacked.


class DocumentCounts(msgspec.Struct):
    """How a collection's memberships are spread over the lifecycle, counted in the database: the
    listing that replaced it is paged, so a caller can no longer count the rows it received."""

    total: int = 0
    indexed: int = 0
    active: int = 0
    error: int = 0
    by_status: dict[str, int] = msgspec.field(default_factory=dict)


class MaintenanceState(msgspec.Struct):
    """The maintenance columns of one collection row. `maintenance` decides what to do about
    them; the row they live in belongs to `Collection`."""

    pending_docs: int
    last_write_at: float | None
    last_maintained_at: float | None
    vector_index_rows: int


MAINTENANCE_COLUMNS = "pending_docs, last_write_at, last_maintained_at, vector_index_rows"


class CollectionSummary(msgspec.Struct):
    """One row of the collection listing: enough for a sidebar, without reading any document."""

    name: str
    counts: DocumentCounts
    created_at: float
    description: str = ""


class IndexStatus(msgspec.Struct):
    """The collection's LanceDB table as it is right now, plus how its maintenance stands.
    Read on demand (`Collection.info`), never stored: the table is its own source of truth."""

    num_rows: int
    num_fragments: int
    num_small_fragments: int
    has_fts_index: bool
    has_vector_index: bool
    unindexed_rows: int
    vector_index_rows: int
    last_maintained_at: float | None
    pending_docs: int


class CollectionInfo(msgspec.Struct):
    name: str
    settings: CollectionSettings
    effective: ChunkSettings
    search: SearchSettings
    description: str
    counts: DocumentCounts  # the members themselves are paged, at /api/collections/{name}/documents
    index_outdated: bool = False  # built by an older version or embedding; "Index all" fixes
    index: IndexStatus | None = None  # None until the collection has a table


class Member(msgspec.Struct):
    """One document as a member of one collection: the document row, and how far this
    collection got writing it into its table."""

    document: document.Document
    status: MemberStatus
    error: str | None = None
    added_at: float = 0.0
    updated_at: float = 0.0


MEMBER_COLUMNS = "cd.status, cd.error, cd.added_at, cd.updated_at"
_MEMBER_SELECTED = [*(f"d.{c}" for c in document.DOCUMENT_COLUMNS), *MEMBER_COLUMNS.split(", ")]
_DOC_WIDTH = len(document.DOCUMENT_COLUMNS)


def _member(row: tuple) -> Member:
    status, error, added_at, updated_at = row[_DOC_WIDTH:]
    return Member(
        document=document._document(row[:_DOC_WIDTH]),
        status=status,
        error=error,
        added_at=added_at,
        updated_at=updated_at,
    )


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


def _counts(by_status: dict[str, int]) -> DocumentCounts:
    """Roll one `status -> count` mapping up into the shape every caller reads."""
    return DocumentCounts(
        total=sum(by_status.values()),
        indexed=by_status.get("indexed", 0),
        active=sum(by_status.get(status, 0) for status in ACTIVE_MEMBER_STATUSES),
        error=by_status.get("error", 0),
        by_status=by_status,
    )


async def _counts_by_collection(
    conn: aiosqlite.Connection, names: list[str]
) -> dict[str, DocumentCounts]:
    """Counts for the names of one page in one grouped query, on the listing's connection."""
    if not names:
        return {}
    by_collection: dict[str, dict[str, int]] = {name: {} for name in names}
    marks = db.placeholders(len(names))
    cursor = await conn.execute(
        "select collection, status, count(*) from collection_documents "
        f"where collection in ({marks}) group by 1, 2",
        names,
    )
    for collection, status, count in await cursor.fetchall():
        by_collection[collection][status] = count
    return {name: _counts(by_status) for name, by_status in by_collection.items()}


_settings_cache: dict[str, CollectionSettings] = {}
# A threading lock rather than an async one, because both event loops of this process (Litestar's
# and DBOS's) read collection settings. It is never held across an `await` — a reader that misses
# runs its query outside the lock and then keeps whatever a writer stored meanwhile.
_settings_lock = threading.Lock()


def invalidate_collection_caches() -> None:
    """Forget every cached settings struct, so the next read goes to the database."""
    with _settings_lock:
        _settings_cache.clear()


def _forget_settings(name: str) -> None:
    with _settings_lock:
        _settings_cache.pop(name, None)


def _cache_settings(found: dict[str, CollectionSettings]) -> None:
    """Store rows just read, without overwriting a value a writer cached while we were reading:
    the writer's row is the newer one (see `set_settings`)."""
    with _settings_lock:
        for name, value in found.items():
            _settings_cache.setdefault(name, value)


def _decode_settings(rows: list[Any]) -> dict[str, CollectionSettings]:
    """Decode `(name, settings)` rows into the struct every caller reads, filling the
    per-collection cache on the way. A row with unreadable JSON falls back to the defaults, which
    is what a collection that never set any has."""
    found = {
        name: (db.loads(raw, CollectionSettings) or CollectionSettings()) for name, raw in rows
    }
    _cache_settings(found)
    return found


def resolve_hit(hit: Hit) -> Hit:
    """Fill in the absolute file paths of a hit, in place.

    The index stores the document's paths home-relative so the home stays portable; they are
    document paths (`documents/<shard>/<doc>/original...`), not collection paths, because the file
    a hit points into belongs to the document, whichever collection matched it. Module-level for
    the same reason: it reads the home, nothing a collection owns.
    """
    hit.source_file = str(home.HOME / hit.source_path) if hit.source_path else ""
    hit.markdown_file = str(home.HOME / hit.markdown_path) if hit.markdown_path else ""
    return hit


class Collection:
    def __init__(self, name: str) -> None:
        """Sync and IO-free: a `Collection` is a name and the paths derived from it. `get` is the
        constructor that checks the collection exists."""
        self.name = name
        self.root = home.COLLECTION_ROOT / shard(name) / name
        self.index_dir = self.root / "index"

    # --- lifecycle -------------------------------------------------------

    @staticmethod
    async def names() -> list[str]:
        """Every name, for callers inside the process (sessions, workflows). The API pages
        instead, through `page` below."""
        async with db.connect() as conn:
            cursor = await conn.execute("select name from collections order by name")
            return [row[0] for row in await cursor.fetchall()]

    @staticmethod
    async def page(request: PageRequest) -> Page[CollectionSummary]:
        """One page of collections with their member counts: a keyset walk over `collections`,
        then a single grouped count for the names on the page."""
        sort, expression = resolve_sort(request.sort, COLLECTION_SORTS, "name")
        walk = keyset(sort, expression, request)
        boundary, params = walk.where()
        selected = ["name", "created_at", "description"]
        async with db.connect() as conn:
            cursor = await conn.execute(
                f"select {', '.join(selected)} from collections "
                f"{f'where {boundary} ' if boundary else ''}"
                f"{walk.order_by()} limit {walk.limit()}",
                params,
            )
            rows: list[Any] = list(await cursor.fetchall())
            cursor = await conn.execute("select count(*) from collections")
            (total,) = await cursor.fetchone() or (0,)  # a count always returns its one row
            counts = await _counts_by_collection(
                conn, [row[0] for row in rows[: request.page_size]]
            )
        return walk.page(
            rows,
            build=lambda row: CollectionSummary(
                name=row[0], counts=counts[row[0]], created_at=row[1], description=row[2]
            ),
            key=key_reader(sort, expression, selected),
            total=total,
        )

    @classmethod
    async def create(cls, name: str, description: str = "") -> "Collection":
        collection = cls(document.safe_name(name))
        async with db.connect() as conn:
            cursor = await conn.execute(
                "insert into collections (name, created_at, description) values (?, ?, ?) "
                "on conflict (name) do nothing",
                (collection.name, time.time(), description),
            )
            created = cursor.rowcount == 1  # read on the open connection, before it is closed
        if not created:
            raise Conflict(f"collection already exists: {collection.name}")
        _forget_settings(collection.name)  # a read before the insert may have cached defaults
        await anyio.Path(collection.root).mkdir(parents=True, exist_ok=True)
        return collection

    @classmethod
    async def get(cls, name: str) -> "Collection":
        async with db.connect() as conn:
            cursor = await conn.execute("select 1 from collections where name = ?", (name,))
            row = await cursor.fetchone()
        if row is None:
            raise CollectionNotFound(f"collection not found: {name}")
        return cls(name)

    async def delete(self) -> None:
        """Delete the collection. `delete_*` is the whole operation, `remove_*` is one step of it.

        Rows first, then the folder: while the row exists the collection is still listed, so a
        crash in between leaves a collection that can be deleted again rather than a phantom.
        No document is touched: the documents stay, in their folders and in every other
        collection that holds them."""
        await self.remove_rows()
        await self.remove_tree()

    async def remove_rows(self) -> None:
        """Row delete cascades to the memberships and to every session that chose the collection
        (`session_collections`); `pragma foreign_keys = on` is set on every connection."""
        async with db.connect() as conn:
            await conn.execute("delete from collections where name = ?", (self.name,))
        _forget_settings(self.name)  # a collection created again under this name starts clean

    async def remove_tree(self) -> None:
        """The index table of the collection."""
        forget_schema(self.index_dir)
        await home.remove_tree(self.root)

    # --- settings --------------------------------------------------------

    async def settings(self) -> CollectionSettings:
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
                "select settings from collections where name = ?", (self.name,)
            )
            row = await cursor.fetchone()
        value = (db.loads(row[0], CollectionSettings) if row else None) or CollectionSettings()
        with _settings_lock:
            return _settings_cache.setdefault(self.name, value)

    async def set_settings(self, value: CollectionSettings) -> None:
        async with db.connect() as conn:
            await conn.execute(
                "update collections set settings = ? where name = ?", (db.dumps(value), self.name)
            )
        with _settings_lock:  # after the commit, so a reader cannot cache the previous value
            _settings_cache[self.name] = value

    @staticmethod
    async def load_settings(names: list[str]) -> dict[str, CollectionSettings]:
        """The settings of several collections in one query, keyed by name. A name with no row
        is absent from the result, which is how a caller learns the collection is gone. Fills
        the per-collection cache on the way."""
        wanted = list(dict.fromkeys(names))
        if not wanted:
            return {}
        marks = db.placeholders(len(wanted))
        async with db.connect() as conn:
            cursor = await conn.execute(
                f"select name, settings from collections where name in ({marks})", wanted
            )
            rows: list[Any] = list(await cursor.fetchall())
        return _decode_settings(rows)

    @staticmethod
    async def reranker_overrides() -> list[str]:
        """Every reranker model a collection overrides, in name order and without duplicates.

        One query over the `settings` column: the model downloads have to cover the overrides too,
        and a search of that collection loads whichever model it names. Fills the per-collection
        cache on the way, like `load_settings` does."""
        async with db.connect() as conn:
            cursor = await conn.execute("select name, settings from collections order by name")
            rows: list[Any] = list(await cursor.fetchall())
        found = _decode_settings(rows)
        chosen = [v.search.reranker_model for v in found.values() if v.search.reranker_model]
        return list(dict.fromkeys(chosen))

    async def chunk_settings(self) -> ChunkSettings:
        """How this collection splits a document: what the embedding cache is keyed by."""
        return (await self.settings()).resolve(await load_user_settings())

    async def search_settings(self) -> SearchSettings:
        return (await self.settings()).resolve_search(await load_user_settings())

    async def info(self) -> CollectionInfo:
        """Everything the collection panel shows, off one connection: the whole `collections` row
        and the member counts. Reading them apart would let a concurrent write show up in one
        half of the answer and not the other."""
        user = await load_user_settings()
        async with db.connect() as conn:
            cursor = await conn.execute(
                f"select settings, description, {MAINTENANCE_COLUMNS} from collections "
                "where name = ?",
                (self.name,),
            )
            row: Any = await cursor.fetchone()
            if row is None:
                raise CollectionNotFound(f"collection not found: {self.name}")
            cursor = await conn.execute(
                "select status, count(*) from collection_documents where collection = ? group by 1",
                (self.name,),
            )
            counted = await cursor.fetchall()
        raw, description, *maintenance = row
        settings = _decode_settings([(self.name, raw)])[self.name]
        state = MaintenanceState(*maintenance)
        index = self.index_with(user.embedding_model)  # one handle: each opens its own connection
        stats = await index.stats()
        return CollectionInfo(
            name=self.name,
            settings=settings,
            effective=settings.resolve(user),
            search=settings.resolve_search(user),
            description=description,
            counts=_counts({status: count for status, count in counted}),
            index_outdated=not await index.schema_current(),
            # None while the collection has no table
            index=(
                _index_status(stats, state.last_maintained_at, state.pending_docs)
                if stats is not None
                else None
            ),
        )

    async def description(self) -> str:
        """Empty for a collection that has none, and for one that no longer exists."""
        async with db.connect() as conn:
            cursor = await conn.execute(
                "select description from collections where name = ?", (self.name,)
            )
            row = await cursor.fetchone()
        return row[0] if row else ""

    async def describe(self, description: str) -> None:
        """Replace the collection's description. Empty clears it."""
        async with db.connect() as conn:
            await conn.execute(
                "update collections set description = ? where name = ?", (description, self.name)
            )

    # --- maintenance columns ---------------------------------------------
    # `maintenance` decides when a run is due and what it does; these are the four columns of the
    # collection row it decides from, so they are read and written here.

    @staticmethod
    async def pending_names() -> list[str]:
        """Collections with documents indexed since their last finished run: what a boot
        reschedules."""
        async with db.connect() as conn:
            cursor = await conn.execute(
                "select name from collections where pending_docs > 0 order by name"
            )
            rows = await cursor.fetchall()
        return [name for (name,) in rows]

    async def maintenance_state(self) -> MaintenanceState | None:
        """None when the collection has no row: it was never created, or it was deleted while a
        run waited. A run treats that as a skip, so the absence has to stay visible."""
        async with db.connect() as conn:
            cursor = await conn.execute(
                f"select {MAINTENANCE_COLUMNS} from collections where name = ?", (self.name,)
            )
            row = await cursor.fetchone()
        return MaintenanceState(*row) if row is not None else None

    async def note_indexed(self) -> int:
        """One more document indexed; returns how many are pending a run. Counted in one
        statement, so two documents finishing at the same moment both count."""
        async with db.connect() as conn:
            cursor = await conn.execute(
                "update collections set pending_docs = pending_docs + 1, last_write_at = ? "
                "where name = ? returning pending_docs",
                (time.time(), self.name),
            )
            row = await cursor.fetchone()
        return row[0] if row is not None else 0

    async def settle_maintenance(self, claimed: int, retrained: bool, num_rows: int) -> None:
        """Record a finished run: subtract what it claimed, stamp it, and remember the rows the
        vector index was trained on when it retrained. A skipped run settles too, or the
        collection would stay pending for ever and be rescheduled at every boot."""
        async with db.connect() as conn:
            await conn.execute(
                "update collections set pending_docs = max(0, pending_docs - ?), "
                "last_maintained_at = ?, "
                "vector_index_rows = case when ? then ? else vector_index_rows end "
                "where name = ?",
                (claimed, time.time(), retrained, num_rows, self.name),
            )

    # --- search ----------------------------------------------------------

    async def search(self, query: str, limit: int | None = None, **overrides) -> list[Hit]:
        """`limit` and any SearchSettings field can be overridden per call."""
        settings = await self.search_settings()
        if limit:
            overrides["limit"] = limit
        if overrides:
            settings = msgspec.structs.replace(settings, **overrides)
        index = await self.index()
        return [resolve_hit(hit) for hit in await index.search(query, settings)]

    async def index(self) -> CollectionIndex:
        return self.index_with((await load_user_settings()).embedding_model)

    def index_with(self, embedding: EmbeddingModel | None) -> CollectionIndex:
        """Variant without the settings read, for steps that already hold the embedding model.
        Sync, like the `CollectionIndex` constructor it calls: opening the table is what awaits."""
        return CollectionIndex(self.index_dir, self.name, home.HOME, embedding)

    # --- members ---------------------------------------------------------

    async def add(self, doc: str) -> None:
        """Attach a document: a `pending` membership, which indexing then moves along. Attaching
        a document already attached is a no-op.

        Only an imported document may join a collection, and the check lives here rather than in
        the caller: one still importing has no markdown to chunk yet, and one being deleted must
        not gain a membership the delete's snapshot missed.
        """
        row = await document.get(doc)  # DocumentNotFound before anything is written
        if row.status != "imported":
            raise Conflict(
                f"document is {row.status}; only an imported document joins a collection: {doc}"
            )
        now = time.time()
        async with db.connect() as conn:
            await conn.execute(
                "insert into collection_documents (collection, document, added_at, updated_at) "
                "values (?, ?, ?, ?) on conflict (collection, document) do nothing",
                (self.name, doc, now, now),
            )

    async def member(self, doc: str) -> Member:
        async with db.connect() as conn:
            cursor = await conn.execute(
                f"select {', '.join(_MEMBER_SELECTED)} from documents d "
                "join collection_documents cd on cd.document = d.name "
                "where cd.collection = ? and d.name = ?",
                (self.name, doc),
            )
            row: Any = await cursor.fetchone()
        if row is None:
            raise DocumentNotFound(f"document not in collection {self.name}: {doc}")
        return _member(row)

    async def set_member_status(
        self, doc: str, status: MemberStatus, error: str | None = None
    ) -> None:
        async with db.connect() as conn:
            await conn.execute(
                "update collection_documents set status = ?, error = ?, updated_at = ? "
                "where collection = ? and document = ?",
                (status, error, time.time(), self.name, doc),
            )

    async def remove_member(self, doc: str) -> None:
        """Detach only: the document, its files and its embedding cache stay."""
        async with db.connect() as conn:
            await conn.execute(
                "delete from collection_documents where collection = ? and document = ?",
                (self.name, doc),
            )

    async def member_names(self, after: str | None = None, limit: int | None = None) -> list[str]:
        """Names alone, ordered by name: what a caller that only iterates members needs.

        `after` resumes the walk past that name and `limit` caps the page, so a bulk job can walk
        a large collection one page at a time instead of holding every name at once."""
        filters: list[str] = ["collection = ?"]
        params: list[Any] = [self.name]
        if after is not None:
            filters.append("document > ?")
            params.append(after)
        limited = " limit ?" if limit is not None else ""
        if limit is not None:
            params.append(limit)
        async with db.connect() as conn:
            cursor = await conn.execute(
                f"select document from collection_documents where {' and '.join(filters)} "
                f"order by document{limited}",
                params,
            )
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def counts(self) -> DocumentCounts:
        async with db.connect() as conn:
            cursor = await conn.execute(
                "select status, count(*) from collection_documents where collection = ? group by 1",
                (self.name,),
            )
            rows = await cursor.fetchall()
        return _counts({status: count for status, count in rows})

    async def members_page(
        self, request: PageRequest, status: MemberStatus | None = None
    ) -> Page[Member]:
        """One page of the collection's members, optionally of one membership status. `total`
        counts the filtered rows, so it is what the page is a page of."""
        sort, expression = resolve_sort(request.sort, MEMBER_SORTS, "name")
        walk = keyset(sort, expression, request)
        filters, params = ["cd.collection = ?"], [self.name]
        if status is not None:
            filters.append("cd.status = ?")
            params.append(status)
        filtered = " and ".join(filters)
        boundary, boundary_params = walk.where()
        where = f"{filtered} and {boundary}" if boundary else filtered
        async with db.connect() as conn:
            cursor = await conn.execute(
                f"select {', '.join(_MEMBER_SELECTED)} from documents d "
                "join collection_documents cd on cd.document = d.name "
                f"where {where} {walk.order_by()} limit {walk.limit()}",
                [*params, *boundary_params],
            )
            rows: list[Any] = list(await cursor.fetchall())
            cursor = await conn.execute(
                f"select count(*) from collection_documents cd where {filtered}", params
            )
            (total,) = await cursor.fetchone() or (0,)  # a count always returns its one row
        return walk.page(
            rows,
            build=_member,
            key=key_reader(sort, expression, _MEMBER_SELECTED, "d.name"),
            total=total,
        )
