"""Collections: a named set of documents with one LanceDB index and its own chunk and search
settings. Metadata in the DB, the index under ~/.haskie/collections/<shard>/<name>/index/.

A collection owns nothing about a document but its membership (`collection_documents`) and the
rows it wrote into its own table. The same document may sit in any number of collections; each
attach is an indexing operation of its own (embed-if-missing from the document's cache, then a
write into this collection's table), so a membership carries its own status — a document can be
`indexed` in one collection and `error` in another at the same time. Deleting a collection drops
its rows and its folder and touches no document.

Every row read, row write and file touch is awaited: the database goes through `db.read()` or
`db.connect()` (aiosqlite), the files through `anyio.Path` and `home`, the table through
`CollectionIndex`. The pure parts (paths, row decoding) stay sync.
"""

import time
from enum import StrEnum
from pathlib import Path
from typing import Any

import anyio
import msgspec
import numpy as np
from sqlalchemy import (
    ColumnElement,
    CompoundSelect,
    Row,
    and_,
    delete,
    func,
    not_,
    select,
    union,
    update,
)
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from haskie import claude, db, home
from haskie.catalogue import catalogue
from haskie.catalogue.catalogue import EmbeddingModel
from haskie.collection.index import CollectionIndex, IndexStats, forget_schema
from haskie.document import document
from haskie.errors import Conflict, NotFound
from haskie.paging import Page, PageRequest, count_of, keyset, resolve_sort
from haskie.settings import (
    ChunkSettings,
    CollectionOverrides,
    SearchSettings,
    load_user_settings,
)
from haskie.tables import collection_documents, collections, documents, session_collections


class MemberStatus(StrEnum):
    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    ERROR = "error"
    CANCELLED = "cancelled"
    REMOVING = "removing"  # a detach queued its removal; the membership goes once that ran


# A document on its way out of a collection, by either road: its membership is being removed, or
# the document is being deleted. Its rows stay in the table until the removal queued for them runs,
# so a search reads around it. `for_search` leaves them out, and so does `holding`.
_REMOVING = collection_documents.c.status == MemberStatus.REMOVING
_DELETING = documents.c.status == document.DocumentStatus.DELETING
LEAVING = (_REMOVING, _DELETING)
# being written into or taken out of the collection right now: the states a poll waits on
ACTIVE_MEMBER_STATUSES: tuple[MemberStatus, ...] = (
    MemberStatus.PENDING,
    MemberStatus.INDEXING,
    MemberStatus.REMOVING,
)

# The member listing selects a document row and its membership, which share `status`, `error`
# and `updated_at`: the membership's are labelled, so a row maps each name to one column.
MEMBER_STATUS = collection_documents.c.status.label("member_status")
MEMBER_UPDATED_AT = collection_documents.c.updated_at.label("member_updated_at")

# Public sort name -> column. The whitelist is the only source of columns a listing can order by,
# so a request can never name one (see paging.resolve_sort).
COLLECTION_SORTS = {"name": collections.c.name, "created_at": collections.c.created_at}
MEMBER_SORTS = {
    "name": documents.c.name,
    "size": documents.c.size,
    "status": MEMBER_STATUS,
    "updated_at": MEMBER_UPDATED_AT,
}


class DocumentCounts(msgspec.Struct):
    """How a collection's memberships are spread over the lifecycle, counted in the database: the
    member listing is paged, so a caller cannot count the rows it received."""

    total: int = 0
    indexed: int = 0
    active: int = 0
    error: int = 0
    by_status: dict[str, int] = msgspec.field(default_factory=dict)


class MaintenanceState(msgspec.Struct):
    """The maintenance columns of one collection row. `workflows` decides from them when a run is
    due, and `maintenance` what it does; the row they live in belongs to `Collection`."""

    pending_documents: int
    last_write_at: float | None
    last_maintained_at: float | None
    vector_index_rows: int


MAINTENANCE_COLUMNS = db.columns_of(collections, MaintenanceState)


class CollectionSummary(msgspec.Struct):
    """One row of the collection listing: enough for a sidebar, without reading any document."""

    name: str
    counts: DocumentCounts
    created_at: float
    description: str = ""


class CollectionInfo(msgspec.Struct):
    name: str
    overrides: CollectionOverrides
    effective: ChunkSettings
    search: SearchSettings
    description: str
    counts: DocumentCounts  # the members themselves are paged, at /api/collections/{name}/documents
    maintenance: MaintenanceState  # how the collection's own maintenance stands
    index_outdated: bool = False  # built by an older version or embedding; "Index all" fixes
    # the table as it is right now, read on demand and never stored; None until there is one
    index: IndexStats | None = None


class Member(msgspec.Struct):
    """One document as a member of one collection: the document row, and how far this
    collection got writing it into its table."""

    document: document.Document
    status: MemberStatus
    error: str | None = None
    added_at: float = 0.0
    updated_at: float = 0.0


_MEMBERS = select(
    *document.DOCUMENT_COLUMNS,
    MEMBER_STATUS,
    collection_documents.c.error.label("member_error"),
    collection_documents.c.added_at,
    MEMBER_UPDATED_AT,
).join_from(documents, collection_documents, collection_documents.c.document_id == documents.c.id)


def _member(row: Row[Any]) -> Member:
    return Member(
        document=document.from_row(row),
        status=row.member_status,
        error=row.member_error,
        added_at=row.added_at,
        updated_at=row.member_updated_at,
    )


def _counts(by_status: dict[str, int]) -> DocumentCounts:
    """Roll one `status -> count` mapping up into the shape every caller reads."""
    return DocumentCounts(
        total=sum(by_status.values()),
        indexed=by_status.get(MemberStatus.INDEXED, 0),
        active=sum(by_status.get(status, 0) for status in ACTIVE_MEMBER_STATUSES),
        error=by_status.get(MemberStatus.ERROR, 0),
        by_status=by_status,
    )


async def _counts_by_collection(
    conn: AsyncConnection, names: list[str]
) -> dict[str, DocumentCounts]:
    """Counts for the names of one page in one grouped query, on the listing's connection."""
    if not names:
        return {}
    by_collection: dict[str, dict[str, int]] = {name: {} for name in names}
    member = collection_documents.c
    rows = await conn.execute(
        select(member.collection, member.status, func.count())
        .where(member.collection.in_(names))
        .group_by(member.collection, member.status)
    )
    for collection, status, count in rows:
        by_collection[collection][status] = count
    return {name: _counts(by_status) for name, by_status in by_collection.items()}


def _leaving_query(names: list[str]) -> CompoundSelect:
    """The (collection, document id) pairs of these collections that are on their way out
    (`LEAVING`): one select per road, joined by UNION rather than one OR across the two tables.
    An OR over the join reads every membership of the collections to find a set that is almost
    always empty; each select alone is bound by an index, the membership's by
    `idx_collection_documents_status` and the document's by `idx_documents_status`. The deleting
    road is a subquery, not a join: joined on the id, the planner reads every membership first."""
    member = collection_documents.c
    pairs = select(member.collection, member.document_id).where(
        member.collection.in_(list(set(names)))
    )
    deleting = select(documents.c.id).where(_DELETING)
    return union(pairs.where(_REMOVING), pairs.where(member.document_id.in_(deleting)))


async def _leaving(conn: AsyncConnection, names: list[str]) -> dict[str, frozenset[str]]:
    """The documents on their way out of each of these collections, on the caller's connection;
    a collection nothing is leaving is absent."""
    found: dict[str, set[str]] = {}
    for collection, doc in await conn.execute(_leaving_query(names)):
        found.setdefault(collection, set()).add(doc)
    return {collection: frozenset(docs) for collection, docs in found.items()}


async def _overrides_of(conn: AsyncConnection, names: list[str]) -> dict[str, CollectionOverrides]:
    """The overrides of these collections in one query, on the caller's connection; a name with
    no row is absent."""
    wanted = list(dict.fromkeys(names))
    if not wanted:
        return {}
    rows = await conn.execute(
        select(collections.c.name, collections.c.overrides).where(collections.c.name.in_(wanted))
    )
    return {name: _overrides(raw) for name, raw in rows}


def _overrides(raw: str) -> CollectionOverrides:
    """One `overrides` column as the struct every caller reads."""
    return msgspec.json.decode(raw, type=CollectionOverrides)


class Collection:
    def __init__(self, name: str) -> None:
        """Sync and IO-free: a `Collection` is a name and the paths derived from it. `get` is the
        constructor that checks the collection exists."""
        self.name = name
        self.root = home.COLLECTION_ROOT / home.shard(name) / name
        self.index_dir = self.root / "index"

    # --- lifecycle -------------------------------------------------------

    @staticmethod
    async def names() -> list[str]:
        """Every name, for callers inside the process (sessions, workflows). The API pages
        instead, through `page` below."""
        async with db.read() as conn:
            return list(await conn.scalars(select(collections.c.name).order_by(collections.c.name)))

    @staticmethod
    async def page(request: PageRequest) -> Page[CollectionSummary]:
        """One page of collections with their member counts: a keyset walk over `collections`,
        then a single grouped count for the names on the page."""
        sort, column = resolve_sort(request.sort, COLLECTION_SORTS, "name")
        walk = keyset(sort, column, request, collections.c.name)
        listed = select(collections.c.name, collections.c.created_at, collections.c.description)
        async with db.read() as conn:
            rows = (await conn.execute(walk.apply(listed))).all()
            total = await conn.scalar(count_of(listed))
            counts = await _counts_by_collection(
                conn, [row.name for row in rows[: request.page_size]]
            )
        return walk.page(
            rows,
            build=lambda row: CollectionSummary(
                name=row.name,
                counts=counts[row.name],
                created_at=row.created_at,
                description=row.description,
            ),
            total=total,
        )

    @classmethod
    async def create(cls, name: str, description: str = "") -> "Collection":
        collection = cls(document.safe_name(name))
        async with db.connect() as conn:
            result = await conn.execute(
                insert(collections)
                .values(name=collection.name, created_at=time.time(), description=description)
                .on_conflict_do_nothing()
            )
            created = result.rowcount == 1  # read on the open connection, before it is closed
        if not created:
            raise Conflict(f"collection already exists: {collection.name}")
        await anyio.Path(collection.root).mkdir(parents=True, exist_ok=True)
        claude.refresh_in_background()
        return collection

    @classmethod
    async def get(cls, name: str) -> "Collection":
        async with db.read() as conn:
            found = await conn.scalar(select(collections.c.name).where(collections.c.name == name))
        if found is None:
            raise NotFound(f"collection not found: {name}")
        return cls(name)

    async def rename(self, name: str) -> "Collection":
        """The collection under `name`: its row, its memberships, every session that chose it,
        and its folder, moved in one transaction. Refused while a member document is being
        deleted: its removal from this collection is already queued under the old name, and would
        find nothing there to remove.

        The key is the name, and the foreign keys pointing at it do not cascade an update, so the
        row is copied under the new name, the references are moved over, and the old row goes.
        The folder moves last, inside the transaction: a move that fails rolls the rows back.
        The index table holds no collection name (`CollectionIndex` adds it on read), so it moves
        as it is. Nothing may be writing it: `workflows.rename_collection` checks that first."""
        renamed = Collection(document.safe_name(name))
        member = collection_documents.c
        async with db.connect() as conn:
            deleting = await conn.scalar(
                _MEMBERS.with_only_columns(documents.c.name)
                .where(member.collection == self.name)
                .where(documents.c.status == document.DocumentStatus.DELETING)
                .limit(1)
            )
            if deleting is not None:
                raise Conflict(
                    f"document {deleting} is being deleted; rename it once that finishes"
                )
            row = (
                await conn.execute(select(collections).where(collections.c.name == self.name))
            ).first()
            if row is None:
                raise NotFound(f"collection not found: {self.name}")
            copied = await conn.execute(
                insert(collections)
                .values({**db.record(row), "name": renamed.name})
                .on_conflict_do_nothing()
            )
            if copied.rowcount != 1:
                raise Conflict(f"collection already exists: {renamed.name}")
            for table in (collection_documents, session_collections):
                await conn.execute(
                    update(table)
                    .where(table.c.collection == self.name)
                    .values(collection=renamed.name)
                )
            await conn.execute(delete(collections).where(collections.c.name == self.name))
            forget_schema(self.index_dir)
            # A folder under the new name has no row (the insert above claimed the name), so it is
            # a leftover of a crash, unless it is this one: on a case-insensitive disk `a` and `A`
            # can share a shard.
            target = anyio.Path(renamed.root)
            if await target.exists() and not await target.samefile(self.root):
                await renamed.remove_tree()
            await anyio.Path(renamed.root.parent).mkdir(parents=True, exist_ok=True)
            await anyio.Path(self.root).rename(renamed.root)
        claude.refresh_in_background()
        return renamed

    async def remove_rows(self) -> None:
        """Row delete cascades to the memberships and to every session that chose the collection
        (`session_collections`); `pragma foreign_keys = on` is set on every connection."""
        async with db.connect() as conn:
            await conn.execute(delete(collections).where(collections.c.name == self.name))
        claude.refresh_in_background()

    async def remove_tree(self) -> None:
        """Delete the collection's folder, and with it the index table."""
        forget_schema(self.index_dir)
        await home.remove_tree(self.root)

    def _aside(self, key: str) -> Path:
        """Where a delete moves the folder before it frees the name: beside it, under a name no
        collection can take (`safe_name` strips a leading dot), one per delete (`key`)."""
        return self.root.with_name(f".{self.root.name}.deleted-{key}")

    async def move_aside(self, key: str) -> None:
        """Move the folder to `_aside(key)`, so the name is free with no folder under it: a create
        or a rename onto it, once the row goes, cannot land in a folder the delete then removes.
        Idempotent: a replay finds the folder moved already."""
        aside = anyio.Path(self._aside(key))
        forget_schema(self.index_dir)
        if await anyio.Path(self.root).exists() and not await aside.exists():
            await anyio.Path(self.root).rename(aside)

    async def remove_aside(self, key: str) -> None:
        """Delete the folder a delete moved aside, index table and all."""
        await home.remove_tree(self._aside(key))

    # --- overrides -------------------------------------------------------

    async def overrides(self) -> CollectionOverrides:
        """The defaults for a collection with no row: a caller that needs the absence to be
        visible reads `load_overrides` instead."""
        found = await Collection.load_overrides([self.name])
        return found.get(self.name, CollectionOverrides())

    async def set_overrides(self, value: CollectionOverrides) -> None:
        async with db.connect() as conn:
            await conn.execute(
                update(collections)
                .where(collections.c.name == self.name)
                .values(overrides=db.dumps(value))
            )

    @staticmethod
    async def load_overrides(names: list[str]) -> dict[str, CollectionOverrides]:
        """The overrides of several collections in one query, keyed by name. A name with no row
        is absent from the result, which is how a caller learns the collection is gone."""
        async with db.read() as conn:
            return await _overrides_of(conn, names)

    @staticmethod
    async def for_search(
        names: list[str], embedding: EmbeddingModel | None
    ) -> dict[str, tuple[CollectionIndex, CollectionOverrides]]:
        """The index of each of these collections, with its overrides, once each and in the order
        given: what a search reads. Each index leaves out the documents on their way out of its
        collection (`LEAVING`). A name with no row is absent, as in `load_overrides`.

        One unit of work, so the overrides and the documents leaving are read at one moment."""
        async with db.read() as conn:
            found = await _overrides_of(conn, names)
            leaving = await _leaving(conn, list(found))
        return {
            name: (
                Collection(name).index_with(embedding, leaving.get(name, frozenset())),
                found[name],
            )
            for name in dict.fromkeys(names)
            if name in found
        }

    @staticmethod
    async def reranker_overrides() -> list[str]:
        """Every reranker model a collection overrides, in name order and without duplicates.

        One query over the `overrides` column: the model downloads have to cover the overrides too,
        and a search of that collection loads whichever model it names."""
        async with db.read() as conn:
            rows = await conn.scalars(select(collections.c.overrides).order_by(collections.c.name))
        chosen = [_overrides(raw).search.reranker_model for raw in rows]
        return list(dict.fromkeys(model for model in chosen if model))

    async def chunk_settings(self) -> ChunkSettings:
        """How this collection splits a document: what the embedding cache is keyed by."""
        return (await self.overrides()).resolve(await load_user_settings())

    async def info(self) -> CollectionInfo:
        """Everything the collection panel shows, off one connection: the whole `collections` row
        and the member counts. Reading them apart would let a concurrent write show up in one
        half of the answer and not the other."""
        user = await load_user_settings()
        async with db.read() as conn:
            row = (
                await conn.execute(
                    select(
                        collections.c.overrides, collections.c.description, *MAINTENANCE_COLUMNS
                    ).where(collections.c.name == self.name)
                )
            ).first()
            if row is None:
                raise NotFound(f"collection not found: {self.name}")
            counts = (await _counts_by_collection(conn, [self.name]))[self.name]
        overrides = _overrides(row.overrides)
        # one handle: each opens its own connection
        index = self.index_with(await catalogue.embedding_model(user))
        return CollectionInfo(
            name=self.name,
            overrides=overrides,
            effective=overrides.resolve(user),
            search=overrides.resolve_search(user),
            description=row.description,
            counts=counts,
            index_outdated=not await index.schema_current(),
            index=await index.stats(),
            maintenance=db.row_to(MaintenanceState, row),
        )

    async def describe(self, description: str) -> None:
        """Replace the collection's description. Empty clears it."""
        async with db.connect() as conn:
            await conn.execute(
                update(collections)
                .where(collections.c.name == self.name)
                .values(description=description)
            )
        claude.refresh_in_background()

    # --- maintenance columns ---------------------------------------------
    # `workflows` decides when a run is due and `maintenance` what it does; these are the four
    # columns of the collection row they decide from, so they are read and written here.

    @staticmethod
    async def pending_names() -> list[str]:
        """Collections with documents indexed since their last finished run: what a boot
        reschedules."""
        async with db.read() as conn:
            names = await conn.scalars(
                select(collections.c.name)
                .where(collections.c.pending_documents > 0)
                .order_by(collections.c.name)
            )
            return list(names)

    async def maintenance_state(self) -> MaintenanceState | None:
        """None when the collection has no row: it was never created, or it was deleted while a
        run waited. A run treats that as a skip, so the absence has to stay visible."""
        async with db.read() as conn:
            row = (
                await conn.execute(
                    select(*MAINTENANCE_COLUMNS).where(collections.c.name == self.name)
                )
            ).first()
        return db.row_to(MaintenanceState, row) if row is not None else None

    async def note_indexed(self) -> int:
        """One more document indexed; returns how many are pending a run. Counted in one
        statement, so two documents finishing at the same moment both count."""
        async with db.connect() as conn:
            pending = await conn.scalar(
                update(collections)
                .where(collections.c.name == self.name)
                .values(
                    pending_documents=collections.c.pending_documents + 1,
                    last_write_at=time.time(),
                )
                .returning(collections.c.pending_documents)
            )
        return pending or 0

    async def settle_maintenance(self, claimed: int, retrained: bool, num_rows: int) -> None:
        """Record a finished run: subtract what it claimed, stamp it, and remember the rows the
        vector index was trained on when it retrained. A skipped run settles too, or the
        collection would stay pending for ever and be rescheduled at every boot."""
        async with db.connect() as conn:
            await conn.execute(
                update(collections)
                .where(collections.c.name == self.name)
                .values(
                    pending_documents=func.max(0, collections.c.pending_documents - claimed),
                    last_maintained_at=time.time(),
                    vector_index_rows=num_rows if retrained else collections.c.vector_index_rows,
                )
            )

    async def set_centre(self, found: tuple[np.ndarray, int] | None, model: str) -> None:
        """Record the sum of the collection's unit chunk vectors under `model` and how many it
        sums (`embed_cache.corpus_sum`); None clears it."""
        vector, rows = found if found is not None else (None, 0)
        async with db.connect() as conn:
            await conn.execute(
                update(collections)
                .where(collections.c.name == self.name)
                .values(
                    vector_sum=None if vector is None else vector.astype(np.float64).tobytes(),
                    vector_rows=rows,
                    vector_model=model if vector is not None else None,
                )
            )

    @staticmethod
    async def centre(names: list[str], model: str) -> np.ndarray | None:
        """The mean unit chunk vector over these collections under `model`, weighed by their
        chunks: what a search centres cosines on (`search.overview`). None when none of them has
        a sum under it yet, before its first maintenance or after the model changed."""
        async with db.read() as conn:
            rows = await conn.execute(
                select(collections.c.vector_sum, collections.c.vector_rows).where(
                    collections.c.name.in_(names),
                    collections.c.vector_model == model,
                    collections.c.vector_sum.is_not(None),
                )
            )
            found = [(np.frombuffer(raw, dtype=np.float64), count) for raw, count in rows]
        total = sum(count for _, count in found)
        if not total:
            return None
        return np.sum([vector for vector, _ in found], axis=0) / total

    async def index(self) -> CollectionIndex:
        return self.index_with(await catalogue.embedding_model(await load_user_settings()))

    def index_with(
        self, embedding: EmbeddingModel | None, leaving: frozenset[str] = frozenset()
    ) -> CollectionIndex:
        """Variant without the settings read, for steps that already hold the embedding model.
        A search passes the documents `leaving` the collection, which its reads then leave out.
        Sync, like the `CollectionIndex` constructor it calls: opening the table is what awaits."""
        return CollectionIndex(self.index_dir, self.name, home.HOME, embedding, leaving)

    # --- members ---------------------------------------------------------

    def _membership(self, doc: str) -> ColumnElement[bool]:
        """The filter that picks this collection's membership of `doc`."""
        member = collection_documents.c
        return and_(member.collection == self.name, member.document_id == doc)

    def _refuse_removing(self, name: str, status: str | None) -> None:
        """A membership being removed takes no new index work: the removal queued ahead of it
        would take the rows that work writes. `name` is the document's, for the message."""
        if status == MemberStatus.REMOVING:
            raise Conflict(
                f"document is being removed from collection {self.name}; "
                f"attach or index it again once it is gone: {name}"
            )

    async def _not_member(self, doc: str) -> NotFound:
        """The error for a document that is no member, naming it as people know it."""
        name = await document.name_of(doc)
        return NotFound(f"document not in collection {self.name}: {name}")

    async def add(self, doc: str) -> None:
        """Attach a document: a `pending` membership, which indexing then moves along. Attaching
        a document already attached is a no-op.

        Only an imported document may join a collection, and the check lives here rather than in
        the caller: one still importing has no markdown to chunk yet, and one being deleted must
        not gain a membership the delete's snapshot missed. A membership being removed refuses
        the attach too (`_refuse_removing`).

        The check and the insert are one unit of work: a delete that marks the document between
        them would snapshot its memberships without this one, and never remove it.
        """
        now = time.time()
        async with db.connect() as conn:
            found = (
                await conn.execute(
                    select(documents.c.status, documents.c.name).where(documents.c.id == doc)
                )
            ).first()
            if found is None:
                raise NotFound(f"document not found: {doc}")
            status, name = found
            if status != document.DocumentStatus.IMPORTED:
                raise Conflict(
                    f"document is {status}; only an imported document joins a collection: {name}"
                )
            await conn.execute(
                insert(collection_documents)
                .values(collection=self.name, document_id=doc, added_at=now, updated_at=now)
                .on_conflict_do_nothing()
            )
            membership = await conn.scalar(
                select(collection_documents.c.status).where(self._membership(doc))
            )
        self._refuse_removing(name, membership)

    async def member(self, doc: str) -> Member:
        async with db.read() as conn:
            row = (await conn.execute(_MEMBERS.where(self._membership(doc)))).first()
        if row is None:
            raise await self._not_member(doc)
        return _member(row)

    async def member_to_index(self, doc: str) -> Member:
        """The membership of `doc`, refused while it is being removed (`_refuse_removing`)."""
        found = await self.member(doc)
        self._refuse_removing(found.document.name, found.status)
        return found

    async def _move_member(
        self, doc: str, status: MemberStatus, error: str | None, *guard: ColumnElement[bool]
    ) -> bool:
        """Set the membership's status and error where every `guard` holds; whether it did."""
        async with db.connect() as conn:
            moved = await conn.scalar(
                update(collection_documents)
                .where(self._membership(doc), *guard)
                .values(status=status, error=error, updated_at=time.time())
                .returning(collection_documents.c.document_id)
            )
        return moved is not None

    async def set_member_status(
        self, doc: str, status: MemberStatus, error: str | None = None
    ) -> None:
        """Move the membership along its index. A membership being removed stays so: a cancelled
        index still finishing its write, or a cancel of it, must not show it as indexed or
        cancelled again. Only its removal ends that status (see `fail_removal`)."""
        await self._move_member(doc, status, error, not_(_REMOVING))

    async def cancel_index(self, doc: str) -> None:
        """Record a cancelled index as `cancelled`, but only while the membership is still being
        indexed. An index that ended between the cancel's read and this write keeps the status it
        ended on (DBOS keeps its SUCCESS or ERROR too), and a removal keeps `removing`."""
        await self._move_member(
            doc,
            MemberStatus.CANCELLED,
            None,
            collection_documents.c.status.in_((MemberStatus.PENDING, MemberStatus.INDEXING)),
        )

    async def start_removal(self, doc: str) -> None:
        """Mark the membership `removing`, whatever it was: a detach answers once its removal is
        queued, and this is what the member listing shows until that removal ran."""
        if not await self._move_member(doc, MemberStatus.REMOVING, None):
            raise await self._not_member(doc)

    async def fail_removal(self, doc: str, error: str) -> None:
        """A removal that failed leaves the membership in `error`, with the reason, so it does not
        read `removing` for ever; detaching again retries it."""
        await self._move_member(doc, MemberStatus.ERROR, error, _REMOVING)

    async def remove_member(self, doc: str) -> None:
        """Detach only: the document, its files and its embedding cache stay."""
        async with db.connect() as conn:
            await conn.execute(delete(collection_documents).where(self._membership(doc)))

    @staticmethod
    async def holding(docs: set[str], names: list[str]) -> dict[str, list[str]]:
        """Which of the collections `names` hold each of `docs`, in name order, but for one it is
        on its way out of (`LEAVING`), where a search no longer finds it. A document none of them
        hold is absent. One query for a whole search result: a search that folds hits to
        documents needs every membership at once."""
        if not docs or not names:
            return {}
        member = collection_documents.c
        async with db.read() as conn:
            rows = await conn.execute(
                _MEMBERS.with_only_columns(member.document_id, member.collection)
                .where(member.document_id.in_(list(docs)), member.collection.in_(list(set(names))))
                .where(*(not_(one) for one in LEAVING))
                .order_by(member.document_id, member.collection)
            )
        held: dict[str, list[str]] = {}
        for doc, collection in rows:
            held.setdefault(doc, []).append(collection)
        return held

    async def member_ids(self, after: str | None = None, limit: int | None = None) -> list[str]:
        """Document ids alone, in id order: what a caller that only iterates members needs.

        `after` resumes the walk past that id and `limit` caps the page, so a bulk index can walk
        a large collection one page at a time instead of holding every id at once."""
        member = collection_documents.c
        ids = select(member.document_id).where(member.collection == self.name)
        if after is not None:
            ids = ids.where(member.document_id > after)
        async with db.read() as conn:
            return list(await conn.scalars(ids.order_by(member.document_id).limit(limit)))

    async def counts(self) -> DocumentCounts:
        async with db.read() as conn:
            return (await _counts_by_collection(conn, [self.name]))[self.name]

    async def members_page(
        self, request: PageRequest, status: MemberStatus | None = None
    ) -> Page[Member]:
        """One page of the collection's members, optionally of one membership status. `total`
        counts the filtered rows, so it is what the page is a page of."""
        sort, column = resolve_sort(request.sort, MEMBER_SORTS, "name")
        walk = keyset(sort, column, request, documents.c.name)
        filters = [collection_documents.c.collection == self.name]
        if status is not None:
            filters.append(collection_documents.c.status == status)
        async with db.read() as conn:
            members = _MEMBERS.where(*filters)
            rows = (await conn.execute(walk.apply(members))).all()
            total = await conn.scalar(count_of(members))
        return walk.page(
            rows,
            build=_member,
            total=total,
        )
