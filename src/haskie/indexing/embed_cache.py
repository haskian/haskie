"""The embedding cache: one parquet file per (document, chunk settings, embedding model), keyed
by a canonical URN, plus the `embeddings` row that makes it visible.

Chunking and embedding a document is the expensive part of indexing, and it depends on nothing a
collection owns except its chunk settings. So it is computed once per distinct `Params` and kept
with the document (`documents/<shard>/<doc id>/embeddings/<id>.parquet`): a collection that attaches
the document reads the rows back out of the cache into its own LanceDB table (`pipeline.index_*`)
and computes nothing when the file already exists. The parquet file has one row group per
convert part, in part order, so the index stage can stream it a group at a time.

`Params` is the whole key. Its fields are always serialized in the same order (`urn`), so the same
inputs hash to the same id. The model is keyed by `EmbeddingModel.cache_name`: its name, its
vector size and the document prefix it embeds with, everything that shapes a stored vector; the
accelerator is left out because it selects an execution provider, not a model.
`chunk.CHUNK_VERSION` is in, so a change to the splitting code retires every entry it would
have produced differently. The id is the full sha256 of the URN, not a truncated one: a collision
here serves one document's vectors as another's, so it is a correctness key, unlike `home.shard`
(spread) or `search.text.query_hash` (cursor validation).

Visibility: `lookup` answers a hit only when both the row and the file exist. `write` puts the file
in place (atomic replace) before it inserts the row (`on conflict do nothing`), so a reader never
sees a row without a file, and a retried or a losing concurrent write is a no-op, not an error.
Two callers wanting the same missing entry are serialized above this module, by the DBOS
deduplication of `workflows.ensure_embedding`; this module only makes the outcome idempotent.

Module owns the parquet schema and the row shape it is read back into (`index.Row`), the way
`collection/index.py` owns LanceDB's; the chunk columns inside both come from `chunk.record`. One
column is this module's own: `seq`, the row's 1-based position among the document's chunks, which
only the merge across parts can number (see `_merge`). File writes and reads run in a worker thread:
pyarrow is sync.

The document's outline (`outline.build`, `outline.store`) is built from one entry's file
(`build_outline`); which entry, `workflows.ensure_embedding` decides.
"""

import hashlib
import time
import weakref
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import anyio.to_thread
import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert

from haskie import cpu, db, home
from haskie.catalogue.catalogue import EmbeddingModel
from haskie.collection.collection import MemberStatus
from haskie.collection.index import Row, vector_field, vector_matrix
from haskie.document import document
from haskie.indexing import chunk
from haskie.indexing.chunk import CHUNK_VERSION, Chunk
from haskie.outline import build, keywords, store
from haskie.search import collapse
from haskie.settings import Chunker, ChunkSettings, Parser
from haskie.tables import collection_documents, documents, embeddings

NO_MODEL = "none"  # the `model` of a profile without an embedding model: chunks only, no vectors


class Params(msgspec.Struct, frozen=True):
    """Everything the cached rows of one document depend on. Field order is the URN order, and
    the `embeddings` columns `Entry` inherits."""

    document_id: str  # the document's id (`Document.id`)
    model: str  # EmbeddingModel.cache_name, or NO_MODEL
    chunk_size: int
    chunk_merge_below: int
    chunk_frame: bool
    chunker: Chunker
    chunk_version: int
    parser: Parser
    skip_ocr_pages: bool


class Entry(Params, frozen=True):
    """One `embeddings` row: the params it was computed under, plus what the write recorded about
    it. What the API and the tests read the cache as."""

    id: str
    urn: str
    rows: int
    bytes: int
    created_at: float


ENTRY_COLUMNS = db.columns_of(embeddings, Entry)


def params(
    doc: document.Document, chunking: ChunkSettings, embedding: EmbeddingModel | None
) -> Params:
    """The key of one document under one collection's chunk settings and the global model."""
    return Params(
        document_id=doc.id,
        model=embedding.cache_name if embedding else NO_MODEL,
        chunk_version=CHUNK_VERSION,
        parser=doc.parser,
        skip_ocr_pages=doc.skip_ocr_pages,
        **msgspec.structs.asdict(chunking),
    )


def urn(p: Params) -> str:
    """Canonical form: `name:value` per field in declaration order, so equal params always give
    equal text. `document_id` is safe inside it: an MD5 is hex."""
    return ";".join(
        f"{name}:{str(value).lower() if isinstance(value, bool) else value}"
        for name, value in msgspec.structs.asdict(p).items()
    )


def key(p: Params) -> str:
    return hashlib.sha256(urn(p).encode("utf-8")).hexdigest()


# --- paths ---------------------------------------------------------------------


def file_path(doc: str, id: str) -> Path:
    return document.embeddings_dir(doc) / f"{id}.parquet"


def scratch_dir(doc: str, id: str) -> Path:
    """Where the embed slices of one computation leave their `NNNNNN.rows.json`. Per cache id,
    so two computations of one document with different settings never share a file; deleted by
    `write` once the parquet file holds every part."""
    return document.embeddings_dir(doc) / f"{id}.tmp"


def rows_path(doc: str, id: str, seq: int) -> Path:
    return scratch_dir(doc, id) / f"{home.part_name(seq)}.rows.json"


# --- parquet -------------------------------------------------------------------

_PLAIN = pa.schema(
    [
        ("part", pa.int32()),
        ("seq", pa.int32()),
        ("headings", pa.list_(pa.string())),
        ("frame", pa.list_(pa.string())),
        ("pieces", pa.list_(pa.struct([("type", pa.string()), ("text", pa.string())]))),
        ("line_start", pa.int32()),
        ("line_end", pa.int32()),
        ("char_start", pa.int32()),
        ("char_end", pa.int32()),
        ("byte_start", pa.int32()),
        ("byte_end", pa.int32()),
        ("page_start", pa.int32()),
        ("page_end", pa.int32()),
        ("start_reason", pa.string()),
        ("end_reason", pa.string()),
    ]
)


def _schema(dims: int | None) -> pa.Schema:
    if dims is None:
        return _PLAIN
    return _PLAIN.append(vector_field(dims))


def _batch(part: int, rows: list[Row], dims: int | None) -> pa.RecordBatch:
    records = [chunk.record(row.chunk, row.vector, dims, part=part, seq=row.seq) for row in rows]
    return pa.RecordBatch.from_pylist(records, schema=_schema(dims))


def _rows(batch: pa.RecordBatch) -> list[Row]:
    return [
        Row(chunk=msgspec.convert(record, Chunk), vector=record.get("vector"), seq=record["seq"])
        for record in batch.to_pylist()
    ]


class Merged(msgspec.Struct):
    """What `_merge` wrote: its size, its rows, and the sum of their unit vectors."""

    bytes: int
    rows: int
    summed: np.ndarray | None  # None without an embedding model


def _merge(parts: list[Path], target: Path, dims: int | None) -> Merged:
    """Stream every `rows.json` into `target` as one row group each, through a `.tmp` and one
    replace, so a reader never sees a partial file. An empty part still gets a row group, so group
    `n` is always part `n` (an empty group is skipped on read).

    This is also where `Row.seq` is filled in: the parts are chunked in parallel and each one
    numbers its chunks from zero, so the merge is the first place that sees the whole document
    in order. And where the document vector is summed, since every row passes through here once.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    summed = None if dims is None else np.zeros(dims, dtype=np.float64)
    with home.atomic_replace(target) as tmp, pq.ParquetWriter(tmp, _schema(dims)) as writer:
        for part, path in enumerate(parts):
            rows = msgspec.json.decode(path.read_bytes(), type=list[Row])
            for seq, row in enumerate(rows, count + 1):
                row.seq = seq
            writer.write_batch(_batch(part, rows, dims))
            if summed is not None and rows:  # `_batch` has refused a row without a vector by now
                summed += collapse.unit_rows([row.vector for row in rows]).sum(axis=0)
            count += len(rows)
    return Merged(bytes=target.stat().st_size, rows=count, summed=summed)


def _document_vector(merged: Merged) -> bytes | None:
    """The mean of a document's chunk vectors, each scaled to length one first so a long chunk
    weighs no more than a short one, as the float32 bytes stored. Not normalized: its length is
    how tightly the chunks point one way, which a mean over many documents (`corpus_sum`) needs,
    and a cosine (`nearest`) ignores."""
    if merged.summed is None or not merged.rows:
        return None
    mean = merged.summed / merged.rows
    if not np.linalg.norm(mean):
        return None  # chunks that cancel out: no direction to compare
    return mean.astype(np.float32).tobytes()


# --- the cache -----------------------------------------------------------------


async def lookup(p: Params) -> str | None:
    """The cache id of a hit, else None: a row or a file on its own is an interrupted write."""
    id = key(p)
    async with db.read() as conn:
        found = await conn.scalar(select(embeddings.c.id).where(embeddings.c.id == id))
    if found is None or not await anyio.Path(file_path(p.document_id, id)).is_file():
        return None
    return id


async def write(p: Params, parts: list[Path], dims: int | None) -> str:
    """Merge the scratch rows of every part into the cache file, publish its row, then drop the
    scratch directory - last, so a retry before the row was written still finds its input."""
    id = key(p)
    target = file_path(p.document_id, id)
    merged = await anyio.to_thread.run_sync(_merge, parts, target, dims)
    entry = Entry(
        **msgspec.structs.asdict(p),
        id=id,
        urn=urn(p),
        rows=merged.rows,
        bytes=merged.bytes,
        created_at=time.time(),
    )
    async with db.connect() as conn:
        await conn.execute(
            insert(embeddings)
            .values({**msgspec.to_builtins(entry), "vector": _document_vector(merged)})
            .on_conflict_do_nothing()
        )
    await home.remove_tree(scratch_dir(p.document_id, id))
    return id


# One build of a document's outline at a time: two collections that chunk it apart embed it at
# once after a model change, and each run sees no outline. Unserialized, both would embed its
# keyword candidates, and their two writes could leave the file from one chunking and the index
# rows from the other. Every workflow step runs on DBOS's one loop, so a lock per document holds.
# Weak values: a lock lives while a run holds or awaits it, and its entry goes with the last one.
_outline_locks: weakref.WeakValueDictionary[str, anyio.Lock] = weakref.WeakValueDictionary()


async def build_outline(p: Params, embed: keywords.Embed | None) -> None:
    """Build the document's outline from the cache file of `p` and save it under `p`'s model,
    unless it has one under that model by now: `embed` embeds its keyword candidates; None ranks
    them by weight alone."""
    # setdefault, with no await in between, so two runs of one document take the same lock
    async with _outline_locks.setdefault(p.document_id, anyio.Lock()):
        if await store.current(p.document_id, p.model):
            return  # the run it waited on built it, or a retry of this one did
        path = file_path(p.document_id, key(p))
        chunks, vectors = await anyio.to_thread.run_sync(_read_all, path)
        nodes, pooled = await cpu.on_cpu(build.describe, chunks, vectors, embed)
        await store.save(p.document_id, store.Outline(model=p.model, nodes=nodes), pooled)


def _read_all(path: Path) -> tuple[list[Row], np.ndarray | None]:
    """Every chunk of a cache file in `seq` order, without its vector, and the vectors as one
    matrix: a float list per row would take four times the memory."""
    table = pq.read_table(path)
    plain = table.drop_columns(["vector"]) if "vector" in table.column_names else table
    chunks = [row for batch in plain.to_batches() for row in _rows(batch)]
    if plain is table:
        return chunks, None
    return chunks, vector_matrix(table.column("vector"))


def _num_row_groups(path: Path) -> int:
    with pq.ParquetFile(path) as file:
        return file.num_row_groups


def _group(file: pq.ParquetFile, part: int) -> list[Row]:
    """One decoded row group; empty when the part holds no chunk (see `_merge`)."""
    table = file.read_row_group(part)
    return _rows(table.to_batches()[0]) if table.num_rows else []


async def row_groups(doc: str, id: str) -> int:
    """How many parts the cache file holds: one row group each. Raises `FileNotFoundError` (from
    pyarrow) when the file is gone, which is a miss the caller recomputes from."""
    return await anyio.to_thread.run_sync(_num_row_groups, file_path(doc, id))


async def read(doc: str, id: str, start: int, end: int) -> AsyncIterator[tuple[int, list[Row]]]:
    """The rows of parts `[start, end)`, one decoded row group at a time.

    The file is opened once for the whole walk rather than once per group: a batch of parts is one
    open and one footer read, and the handle is closed when the walk ends or the caller stops.
    """
    file = await anyio.to_thread.run_sync(pq.ParquetFile, file_path(doc, id))
    try:
        for part in range(start, min(end, file.num_row_groups)):
            rows = await anyio.to_thread.run_sync(_group, file, part)
            if rows:
                yield part, rows
    finally:
        await anyio.to_thread.run_sync(file.close)


async def corpus_sum(collection: str, model: str) -> tuple[np.ndarray, int] | None:
    """The sum of the unit chunk vectors of a collection's indexed documents under `model`, and
    how many chunks it sums, each document by its newest entry; None without one. What a search
    centres its vectors on (`search.overview`), built from the document means rather than the
    collection's table: one row per document, not one per chunk."""
    async with db.read() as conn:
        rows = await conn.execute(
            select(embeddings.c.document_id, embeddings.c.vector, embeddings.c.rows)
            .join_from(
                embeddings,
                collection_documents,
                embeddings.c.document_id == collection_documents.c.document_id,
            )
            .where(
                collection_documents.c.collection == collection,
                collection_documents.c.status == MemberStatus.INDEXED,
                embeddings.c.model == model,
                embeddings.c.vector.is_not(None),
            )
            .order_by(embeddings.c.created_at.desc())
        )
        newest: dict[str, tuple[bytes, int]] = {}
        for name, vector, count in rows:
            newest.setdefault(name, (vector, count))
    if not newest:
        return None
    means = [np.frombuffer(vector, dtype=np.float32) for vector, _ in newest.values()]
    counts = np.asarray([count for _, count in newest.values()], dtype=np.float64)
    return (np.asarray(means, dtype=np.float64) * counts[:, None]).sum(axis=0), int(counts.sum())


async def forget(doc: str) -> None:
    """Drop every cached embedding of one document, rows and files. For a reconversion: the
    markdown the rows were chunked from is about to change, so none of them is reusable."""
    async with db.connect() as conn:
        await conn.execute(delete(embeddings).where(embeddings.c.document_id == doc))
    await home.remove_tree(document.embeddings_dir(doc))


async def entries(doc: str) -> list[Entry]:
    """Every cache row of one document, newest first."""
    async with db.read() as conn:
        rows = await conn.execute(
            select(*ENTRY_COLUMNS)
            .where(embeddings.c.document_id == doc)
            .order_by(embeddings.c.created_at.desc())
        )
        return [db.row_to(Entry, row) for row in rows]


class Neighbour(msgspec.Struct):
    """A document close to another one, by the cosine of their document vectors (1 is the same
    direction)."""

    document: str
    similarity: float


async def nearest(doc: str, model: str, limit: int) -> list[Neighbour]:
    """The `limit` imported documents whose vector under `model` lies closest to that of `doc`
    (an id), closest first, by name. Empty while `doc` has no vector under it: still importing,
    or no embedding model.

    Each document is compared by its newest cache entry under the model: the entries of one
    document differ only in how it was chunked, which barely moves the mean. Every vector is read
    and compared in memory; a library of thousands of books is a few megabytes of them."""
    async with db.read() as conn:
        rows = await conn.execute(
            select(embeddings.c.document_id, documents.c.name, embeddings.c.vector)
            .join_from(embeddings, documents, embeddings.c.document_id == documents.c.id)
            .where(
                embeddings.c.model == model,
                embeddings.c.vector.is_not(None),
                documents.c.status == document.DocumentStatus.IMPORTED,
            )
            .order_by(embeddings.c.created_at.desc())
        )
        vectors: dict[str, bytes] = {}
        names: dict[str, str] = {}
        for id, name, vector in rows:
            vectors.setdefault(id, vector)  # newest first, so the first one is kept
            names[id] = name
    target = vectors.pop(doc, None)
    if target is None or not vectors:
        return []
    # in a worker thread, like every other numpy and pyarrow call here: the matrix grows with
    # the library
    found = await anyio.to_thread.run_sync(_closest, vectors, target, limit)
    return [Neighbour(document=names[id], similarity=similarity) for id, similarity in found]


def _closest(vectors: dict[str, bytes], target: bytes, limit: int) -> list[tuple[str, float]]:
    """The `limit` of `vectors` (by document id) with the highest cosine to `target`, as (id,
    cosine)."""
    ids = list(vectors)
    matrix = np.frombuffer(b"".join(vectors.values()), dtype=np.float32).reshape(len(ids), -1)
    scores = collapse.unit_rows(matrix) @ collapse.unit_rows([np.frombuffer(target, np.float32)])[0]
    closest = np.argsort(-scores, kind="stable")[:limit]
    return [(ids[at], round(float(scores[at]), 4)) for at in closest]
