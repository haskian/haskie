"""One LanceDB index per collection. Full-text only by default, hybrid when an embedding is set.

The table holds the chunks of every document of the collection, as rows read out of the
document's embedding cache (`embed_cache.py`): a document that sits in several collections is
chunked and embedded once per distinct chunk settings and written into each collection's table
from that cache. The table is therefore a per-collection view, never the only copy of anything
— dropping it (`reset_for_write`) and refilling it from the cache is what "Index all" does.

Every table access is awaited: LanceDB's async API (`lancedb.connect_async`, `AsyncTable`) runs on
its own tokio runtime, so nothing here blocks the event loop that called it. The pure parts —
Arrow encoding, scoring, row-to-`Hit` — stay sync.

Write and read paths differ on purpose (B1): only the index stage of a document may drop an
outdated table (`reset_for_write`), every other write no-ops on one, and a read never creates a
table at all. Deleting one document from a collection must never wipe the collection.
"""

import math
import threading
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import anyio
import lancedb
import msgspec
import pyarrow as pa
from lancedb.index import FTS, IvfPq

from haskie import cpu, models
from haskie.chunk import Chunk
from haskie.logs import get_logger
from haskie.settings import Accelerator, EmbeddingModel, SearchSettings, load_user_settings


class Row(msgspec.Struct):
    """A chunk ready for the index: metadata plus (optional) precomputed vector."""

    chunk: Chunk
    vector: list[float] | None = None


TABLE = "chunks"
_log = get_logger(__name__)

PLAIN_SCHEMA = pa.schema(
    [
        ("doc", pa.string()),
        ("source_path", pa.string()),  # relative to the haskie home (portable)
        ("markdown_path", pa.string()),
        ("part", pa.int32()),
        ("chunk_id", pa.int32()),
        ("line_start", pa.int32()),
        ("line_end", pa.int32()),
        ("char_start", pa.int32()),
        ("char_end", pa.int32()),
        ("page_start", pa.int32()),
        ("page_end", pa.int32()),
        ("parents", pa.string()),
        ("heading", pa.string()),
        ("text", pa.string()),
    ]
)
PARENT_SEP = " > "


class IndexStats(msgspec.Struct):
    """What one collection's table looks like on disk right now. Read straight from LanceDB, never
    stored: maintenance decides on the table as it is, not as it was recorded."""

    num_rows: int
    num_fragments: int
    num_small_fragments: int  # fragments small enough that compaction would merge them
    has_fts_index: bool
    has_vector_index: bool
    unindexed_rows: int  # rows the full-text index has not folded in yet (still scanned)
    vector_index_rows: int  # rows the vector index covers; 0 without one


class Hit(msgspec.Struct):
    """One matching chunk with everything needed to cite or open it."""

    collection: str  # the collection whose table matched; the document itself belongs to none
    doc: str
    home: str  # absolute haskie home; join with the relative paths below to open files
    source_path: str  # original upload, relative to home
    markdown_path: str  # full converted markdown, relative to home
    part: int  # micro-batch that produced the chunk
    chunk_id: int
    line_start: int  # 1-based, in markdown_path
    line_end: int
    char_start: int  # 0-based, in markdown_path
    char_end: int
    page_start: int | None  # 1-based PDF pages; None for non-PDF
    page_end: int | None
    parents: list[str]  # enclosing headings, outermost first
    heading: str
    header: str  # breadcrumb "parent > ... > heading", ready to cite
    location: str  # human-readable "doc p.3-4 L10-20", ready to cite
    text: str
    score: float
    # Absolute, and filled by `collection.resolve_hit` (which every search path calls) rather
    # than stored: the index keeps paths home-relative so a home stays portable. These are what a
    # tool outside the app opens or greps - `line_start`/`line_end` are lines in `markdown_file`.
    source_file: str = ""
    markdown_file: str = ""


# `schema_current` opens the table and reads its Arrow schema, and the read path asks for every
# collection listing, search and delete. The index stage is the only writer of a table and it runs
# in this process, so the answer is cached per (index directory, embedding dimensions) and
# forgotten whenever a table is created, dropped, or its collection deleted.
#
# A `threading.Lock` rather than an async one: both event loops (Litestar's and DBOS's) read this
# cache, and a lock made on one of them cannot be taken from the other. Nothing is awaited while
# it is held, so it is never contended for longer than a dict lookup.
_schema_current: dict[tuple[str, int | None], bool] = {}
_schema_lock = threading.Lock()


def forget_schema(path: Path) -> None:
    """Drop the cached `schema_current` answers for one index directory."""
    directory = str(path)
    with _schema_lock:
        for key in [k for k in _schema_current if k[0] == directory]:
            del _schema_current[key]


def _fusion(settings: SearchSettings):
    """LanceDB "reranker" that merges the vector and BM25 rankings of a hybrid query."""
    from lancedb.rerankers import LinearCombinationReranker, RRFReranker

    if settings.fusion == "linear":
        total = settings.vector_weight + settings.bm25_weight
        weight = settings.vector_weight / total if total > 0 else 0.5
        return LinearCombinationReranker(weight=weight)
    return RRFReranker(K=settings.rrf_k)


class CollectionIndex:
    def __init__(
        self, path: Path, collection: str, home: Path, embedding: EmbeddingModel | None
    ) -> None:
        """Sync and IO-free: opening the table is what `_existing` / `_for_write` do, awaited."""
        self.path = path
        self.collection = collection
        self.home = home  # stored paths are relative to it (see Document.relative)
        self.embedding = embedding
        self._conn: lancedb.AsyncConnection | None = None  # one connection per index instance
        self._cached: lancedb.AsyncTable | None = None  # one handle per index instance

    async def _connection(self) -> lancedb.AsyncConnection:
        """The connection of this instance, opened once.

        Caching it is safe across both event loops: the handle belongs to LanceDB's tokio runtime,
        not to the loop that made it. It is per instance rather than per process because a handle
        keeps reading the table version it opened, and a fresh `CollectionIndex` is how a caller
        asks for the table as it is now.
        """
        if self._conn is None:
            self._conn = await lancedb.connect_async(str(self.path))
        return self._conn

    async def _existing(self) -> lancedb.AsyncTable | None:
        """The table as it is on disk, or None. Never creates anything: `connect_async` alone
        would already create the directory."""
        if self._cached is None:
            if not await anyio.Path(self.path).exists():
                return None
            conn = await self._connection()
            if TABLE in (await conn.list_tables()).tables:
                self._cached = await conn.open_table(TABLE)
        return self._cached

    async def _for_write(self) -> lancedb.AsyncTable:
        """Write path: create the table when it is missing, never drop one."""
        table = await self._existing()
        if table is None:
            conn = await self._connection()
            self._cached = table = await conn.create_table(
                TABLE, schema=self._schema(), exist_ok=True
            )
            forget_schema(self.path)  # any cached answer is about a table that is gone
        return table

    async def reset_for_write(self) -> lancedb.AsyncTable:
        """The only place a table is dropped: the first group of a document being written. A
        table built by an older version or another embedding cannot hold new rows, and nothing in
        it is the only copy of anything: every document of the collection is rewritten from its
        embedding cache by "Index all"."""
        forget_schema(self.path)  # decide on the table as it is now, not as it was cached
        table = await self._existing()
        if table is not None and not await self.schema_current(table):
            _log.warning("index_table_outdated", collection=self.collection, path=str(self.path))
            await (await self._connection()).drop_table(TABLE)
            # the connection too, not only the table handle: an AsyncConnection that dropped a
            # table and creates it again hands out a handle still carrying the dropped table's
            # index metadata, and the first write commits that stale reference into the new
            # manifest (every later list_indices() then fails with "Not found ... _indices/...")
            self._cached = None
            self._conn = None
            forget_schema(self.path)  # the answer just cached is about a table that is gone
        return await self._for_write()

    async def schema_current(self, table: lancedb.AsyncTable | None = None) -> bool:
        """False when the table cannot hold rows written by this build: a missing column, or a
        vector of different dimensions than the current embedding model (B2).

        Cached per index directory and embedding (see `_schema_current`); a missing table is never
        cached, so one created later is inspected.
        """
        key = (str(self.path), self.embedding.dims if self.embedding else None)
        with _schema_lock:
            cached = _schema_current.get(key)
        if cached is not None:
            return cached
        table = table or await self._existing()
        if table is None:
            return True
        current = self._fits(await table.schema())
        with _schema_lock:
            _schema_current[key] = current
        return current

    def _fits(self, schema: pa.Schema) -> bool:
        if not set(PLAIN_SCHEMA.names) <= set(schema.names):
            return False
        if self.embedding is None:
            return True
        if "vector" not in schema.names:
            return False
        return schema.field("vector").type == pa.list_(pa.float32(), self.embedding.dims)

    def _schema(self) -> Any:
        if self.embedding is None:
            return PLAIN_SCHEMA
        return PLAIN_SCHEMA.append(pa.field("vector", pa.list_(pa.float32(), self.embedding.dims)))

    async def _deletable(self) -> lancedb.AsyncTable | None:
        """A delete on a missing or outdated table has nothing to remove; dropping it instead
        would wipe every document of the collection (B1)."""
        table = await self._existing()
        return table if table is not None and await self.schema_current(table) else None

    async def delete_document(self, doc: str) -> None:
        table = await self._deletable()
        if table is not None:
            await table.delete(f"doc = '{doc}'")  # doc names sanitized in document.py

    async def delete_parts(self, doc: str, start: int, end: int) -> None:
        """Drop the parts `[start, end)` of one document, leaving every other part alone."""
        table = await self._deletable()
        if table is not None:
            await table.delete(f"doc = '{doc}' and part >= {int(start)} and part < {int(end)}")

    async def add_parts(
        self,
        doc: str,
        source_path: str,
        markdown_path: str,
        parts: AsyncIterator[tuple[int, list[Row]]],
    ) -> int:
        """Write several micro-batches of one document in a single LanceDB commit; returns the
        number of rows written. `source_path` / `markdown_path` are home-relative (see
        Document.relative); rows must carry a vector when the index has an embedding.

        One commit is one fragment, so a document costs a handful of fragments instead of one per
        micro-batch. `parts` is consumed lazily and each part is turned into Arrow at once, so the
        caller can read one row group of the embedding cache at a time and only the Arrow
        buffers stay in memory. A group with no rows at all writes nothing and creates no table.
        """
        batches = [
            batch
            async for part, rows in parts
            if (batch := self._record_batch(doc, part, source_path, markdown_path, rows))
            is not None
        ]
        if not batches:
            return 0
        table = pa.Table.from_batches(batches, schema=self._schema())
        await (await self._for_write()).add(table)
        return table.num_rows

    def _record_batch(
        self, doc: str, part: int, source_path: str, markdown_path: str, rows: list[Row]
    ) -> pa.RecordBatch | None:
        """One part as Arrow, or None when the part is empty (nothing to write for it)."""
        if not rows:
            return None
        count = len(rows)
        chunks = [row.chunk for row in rows]
        columns: dict[str, Any] = {
            "doc": pa.array([doc] * count, pa.string()),
            "source_path": pa.array([source_path] * count, pa.string()),
            "markdown_path": pa.array([markdown_path] * count, pa.string()),
            "part": pa.array([part] * count, pa.int32()),
            "chunk_id": pa.array(range(count), pa.int32()),  # unique with (doc, part)
            "line_start": pa.array([c.line_start for c in chunks], pa.int32()),
            "line_end": pa.array([c.line_end for c in chunks], pa.int32()),
            "char_start": pa.array([c.char_start for c in chunks], pa.int32()),
            "char_end": pa.array([c.char_end for c in chunks], pa.int32()),
            "page_start": pa.array([c.page_start for c in chunks], pa.int32()),
            "page_end": pa.array([c.page_end for c in chunks], pa.int32()),
            "parents": pa.array([PARENT_SEP.join(c.parents) for c in chunks], pa.string()),
            "heading": pa.array([c.heading for c in chunks], pa.string()),
            "text": pa.array([c.text for c in chunks], pa.string()),
        }
        if self.embedding is not None:
            columns["vector"] = self._vectors(rows)
        return pa.RecordBatch.from_pydict(columns, schema=self._schema())

    def _vectors(self, rows: list[Row]) -> pa.FixedSizeListArray:
        """The precomputed vectors of one part as a fixed-size list column, flattened in one pass
        so no intermediate list of lists is built."""
        dims = self.embedding.dims if self.embedding else 0
        flat: list[float] = []
        for row in rows:
            if row.vector is None:
                raise ValueError("index has an embedding but the row carries no vector")
            flat.extend(row.vector)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat, pa.float32()), dims)

    # --- indexes and maintenance -----------------------------------------
    # Building an index is O(table), so the write path only ever creates a missing one: a LanceDB
    # query covers the rows an index has not folded in yet by scanning them, so a document indexed
    # after the build is found without it. `optimize` (see maintenance.py) folds them in later.

    async def has_index(self, column: str) -> bool:
        """True when this column has an index of its own on the table as it is now."""
        table = await self._existing()
        if table is None:
            return False
        return any(list(config.columns) == [column] for config in await table.list_indices())

    async def has_vector_column(self) -> bool:
        """False when the table is missing or was written without vectors (full text only)."""
        table = await self._existing()
        return table is not None and "vector" in (await table.schema()).names

    async def stats(self) -> IndexStats | None:
        """The table's size, fragmentation and indexes, or None when there is no table."""
        table = await self._existing()
        if table is None:
            return None
        # lancedb annotates stats() as a dataclass but returns plain dicts (async API included)
        raw: dict[str, Any] = await table.stats()  # ty: ignore[invalid-assignment]
        fragments = raw["fragment_stats"]
        indexes = {tuple(config.columns): config for config in await table.list_indices()}
        fts, vector = indexes.get(("text",)), indexes.get(("vector",))
        return IndexStats(
            num_rows=raw["num_rows"],
            num_fragments=fragments["num_fragments"],
            num_small_fragments=fragments["num_small_fragments"],
            has_fts_index=fts is not None,
            has_vector_index=vector is not None,
            unindexed_rows=(fts.num_unindexed_rows or 0) if fts else raw["num_rows"],
            vector_index_rows=(vector.num_indexed_rows or 0) if vector else 0,
        )

    async def finish(self) -> None:
        """Build the full-text index once per table, after a document's parts are all written.

        Rebuilding it per document costs O(rows) each time, so a collection of n documents used
        to cost O(n^2) to fill. Rows written after the build are still found (see above)."""
        table = await self._existing()
        if table is None or not await table.count_rows() or await self.has_index("text"):
            return
        # the async API has no `create_fts_index`; `FTS()` is the same index through `create_index`
        await table.create_index("text", config=FTS(), replace=True)
        _log.info("fts_index_built", collection=self.collection, rows=await table.count_rows())

    async def optimize(self, keep: timedelta) -> None:
        """Compact fragments, fold new rows into every index, and drop versions older than `keep`.

        `keep` is a grace period, not a deadline: a reader that opened the table before this call
        keeps reading the version it opened, so pruning it out from under them must not be
        possible. No-op when there is no table."""
        table = await self._existing()
        if table is not None:
            await table.optimize(cleanup_older_than=keep)

    async def build_vector_index(self, num_rows: int) -> None:
        """(Re)train the approximate vector index over `num_rows` rows.

        IVF-PQ rather than HNSW: it lives on disk, trains in seconds on a sample of the rows, and
        `optimize()` folds later rows into its partitions, so a growing collection does not need
        a rebuild. HNSW would need the whole graph in memory per collection and a full rebuild
        each time. `l2` over normalized embedding vectors ranks exactly like cosine, so switching
        a collection to an approximate index does not change what `row_score` means.

        The training itself is CPU work inside LanceDB's runtime, so it cannot be put under the
        CPU budget (`cpu.on_cpu`); maintenance runs one collection at a time instead."""
        table = await self._existing()
        if table is None or self.embedding is None:
            return
        await table.create_index(
            "vector",
            config=IvfPq(
                distance_type="l2",
                num_partitions=_partitions(num_rows),
                # one 8-bit code per 16 dimensions: the usual PQ ratio, and it divides every
                # embedding profile's dimension count
                num_sub_vectors=max(1, self.embedding.dims // 16),
                num_bits=8,
            ),
            replace=True,
        )

    # --- search ----------------------------------------------------------
    # Split into three steps so a cross-collection search embeds the query once, retrieves from
    # every index in parallel and rescores the merge once (see session.search). `search` below is
    # the single-index composition of the same steps.

    async def accelerator(self) -> Accelerator:
        """The reranker runs where the settings say, not where the embedding model happens to:
        a collection with no embedding profile still reranks, and `models.warm_model` loaded the
        cross-encoder under this same setting."""
        return (await load_user_settings()).pipeline.accelerator

    async def query_vector(self, query: str, settings: SearchSettings) -> list[float] | None:
        """The query embedding, or None when this index can only answer lexically: mode `fts`, no
        embedding model, or a table written without a vector column."""
        if settings.mode == "fts" or self.embedding is None:
            return None
        if not await self.has_vector_column():
            return None
        from haskie.embed import embed_query

        await models.require_ready("embedding", self.embedding.name)
        return await cpu.on_cpu("embed_query", embed_query, self.embedding, query)

    async def search_rows(
        self, query: str, vector: list[float] | None, settings: SearchSettings, limit: int
    ) -> list[dict]:
        """Retrieval only: at most `limit` raw LanceDB rows, neither cut to `settings.limit` nor
        rescored by a cross-encoder.

        `vector` is None for a lexical query (see `query_vector`); a table without a vector column
        falls back to full text whatever the caller passed, so one collection of a session can lack
        the embedding the others have. A hybrid query always fuses over at least
        `settings.candidates` rows, because the fusion is only as good as its candidate pool.
        """
        table = await self._existing()
        if table is None or await table.count_rows() == 0:
            return []
        if vector is None or "vector" not in (await table.schema()).names:
            return await (await table.search(query, query_type="fts")).limit(limit).to_list()
        if settings.mode == "vector":
            found = _tuned(await table.search(vector, query_type="vector"), settings)
            return await found.limit(limit).to_list()
        # the async API builds a hybrid query out of its two halves instead of `query_type=hybrid`
        hybrid = table.query().nearest_to(vector).nearest_to_text(query)
        return (
            await _tuned(hybrid, settings)
            .limit(max(settings.candidates, limit))
            .rerank(reranker=_fusion(settings))
            .to_list()
        )

    async def fts_rows(self, query: str, limit: int) -> list[dict]:
        """Lexical retrieval alone: at most `limit` BM25 rows, whatever this index could answer
        with. `[]` when it cannot answer one at all — no table, no rows, or no full-text index
        yet, which is what a collection in the middle of its first index looks like.

        A missing full-text index is a real answer here, not a scan: a cross-collection search
        asks every collection at once (see textsearch.py), and one still building its index would
        make the whole query wait for it. `search_rows` is the opposite trade for one collection.
        """
        table = await self._existing()
        if table is None or await table.count_rows() == 0 or not await self.has_index("text"):
            return []
        return await (await table.search(query, query_type="fts")).limit(limit).to_list()

    async def search(self, query: str, settings: SearchSettings) -> list[Hit]:
        table = await self._existing()
        if table is None or await table.count_rows() == 0:
            return []  # nothing indexed, so nothing to embed the question for either
        rerank = settings.reranker != "none"
        fetch = max(settings.candidates, settings.limit) if rerank else settings.limit
        vector = await self.query_vector(query, settings)
        rows = await self.search_rows(query, vector, settings, fetch)
        if rerank:
            rows = await cross_encode(query, rows, settings, await self.accelerator())
        return [self.hit(r) for r in rows[: settings.limit]]

    def hit(self, r: dict, score: float | None = None) -> Hit:
        """Tolerates rows from older index versions (missing columns -> empty values). `score`
        replaces the row's own signal: a merged ranking over several indexes scores its rows
        together, because per-index scores are not comparable (see session.search)."""
        parents = r["parents"].split(PARENT_SEP) if r.get("parents") else []
        heading = r["heading"]
        line_start, line_end = r.get("line_start", 0), r.get("line_end", 0)
        page_start, page_end = r.get("page_start"), r.get("page_end")
        pages = ""
        if page_start is not None:
            pages = f" p.{page_start}"
            if page_end != page_start:
                pages += f"-{page_end}"
        return Hit(
            collection=self.collection,
            doc=r["doc"],
            home=str(self.home),
            source_path=r.get("source_path", ""),
            markdown_path=r.get("markdown_path", ""),
            part=r.get("part", 0),
            chunk_id=r["chunk_id"],
            line_start=line_start,
            line_end=line_end,
            char_start=r.get("char_start", 0),
            char_end=r.get("char_end", 0),
            page_start=page_start,
            page_end=page_end,
            parents=parents,
            heading=heading,
            header=PARENT_SEP.join([*parents, heading] if heading else parents),
            location=f"{r['doc']}{pages} L{line_start}-{line_end}",
            text=r["text"],
            score=row_score(r) if score is None else score,
        )


SEARCH_CONCURRENCY = 8  # LanceDB reads are IO bound; more in flight than this only queues up
MIN_PARTITIONS = 16  # below this an IVF index buys nothing over a scan
MAX_PARTITIONS = 4096  # above this training costs more than the queries save


def _partitions(num_rows: int) -> int:
    """IVF partitions for a table of `num_rows`: the usual sqrt(n) rule, rounded to a power of two
    and clamped, so a collection that grows by a few rows keeps the partition count it was trained
    with instead of qualifying for a rebuild."""
    if num_rows <= 0:
        return MIN_PARTITIONS
    partitions = 2 ** round(math.log2(math.sqrt(num_rows)))
    return max(MIN_PARTITIONS, min(MAX_PARTITIONS, partitions))


def _tuned(builder: Any, settings: SearchSettings) -> Any:
    """Approximate-search knobs of a vector or hybrid query. Both are ignored by LanceDB when the
    table has no vector index, so they are safe to set on every query. Sync: building a query
    awaits nothing, only running it does."""
    return builder.nprobes(settings.nprobes).refine_factor(settings.refine_factor)


async def cross_encode(
    query: str, rows: list[dict], settings: SearchSettings, accelerator: Accelerator
) -> list[dict]:
    """Second stage for any mode: rescore candidate rows with a cross-encoder, best first.

    Module-level and told which accelerator to use, so a session rescores one merged candidate
    list instead of running a cross-encoder per collection. The cross-encoder itself is CPU work, so
    it runs in a worker thread under one slot of the CPU budget.
    """
    from haskie.embed import rerank_scores

    await models.require_ready("reranker", settings.reranker_model)
    scores = await cpu.on_cpu(
        "rerank",
        rerank_scores,
        settings.reranker_model,
        accelerator,
        query,
        [r["text"] for r in rows],
    )
    for row, score in zip(rows, scores, strict=True):
        row["_relevance_score"] = score
    return sorted(rows, key=lambda r: r["_relevance_score"], reverse=True)


def row_score(r: dict) -> float:
    """Higher is better in every mode: cross-encoder / fusion score, BM25 score, or vector
    distance mapped through 1 / (1 + d).

    The async API carries the same score columns as the sync one: `_score` for BM25, `_distance`
    for a vector query, `_relevance_score` once a reranker (fusion or cross-encoder) ran."""
    if "_relevance_score" in r:
        return float(r["_relevance_score"])
    if "_score" in r:
        return float(r["_score"])
    if "_distance" in r:
        return 1.0 / (1.0 + float(r["_distance"]))
    return 0.0


# What identifies one passage, wherever it is stored. The collection is deliberately not part of
# it: the same chunk of the same document is the same answer, whichever collection's table it came
# out of, so `session.search` and `textsearch.merge` both count it once.
RowKey = tuple[str, int, int]


def row_key(row: dict) -> RowKey:
    """(doc, part, chunk_id) of one result row. `part` defaults to 0 for a table written before
    micro-batches, the same default `hit` reads it with."""
    return (row["doc"], row.get("part", 0), row["chunk_id"])
