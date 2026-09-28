"""The embedding cache: one parquet file per (document, chunk settings, embedding model), keyed
by a canonical URN, plus the `embeddings` row that makes it visible.

Chunking and embedding a document is the expensive part of indexing, and it depends on nothing a
collection owns except its chunk settings. So it is computed once per distinct `Params` and kept
with the document (`documents/<shard>/<doc>/embeddings/<id>.parquet`): a collection that attaches
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
"""

import hashlib
import time
from collections.abc import AsyncIterator, Collection
from pathlib import Path

import anyio
import anyio.to_thread
import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert

from haskie import db, home
from haskie.catalogue.catalogue import EmbeddingModel
from haskie.collection.index import Row
from haskie.document import document
from haskie.indexing import chunk
from haskie.indexing.chunk import CHUNK_VERSION, Chunk
from haskie.settings import Chunker, ChunkSettings, Parser
from haskie.tables import documents, embeddings

NO_MODEL = "none"  # the `model` of a profile without an embedding model: chunks only, no vectors


class Params(msgspec.Struct, frozen=True):
    """Everything the cached rows of one document depend on. Field order is the URN order, and
    the `embeddings` columns `Entry` inherits."""

    document: str
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
        document=doc.name,
        model=embedding.cache_name if embedding else NO_MODEL,
        chunk_version=CHUNK_VERSION,
        parser=doc.parser,
        skip_ocr_pages=doc.skip_ocr_pages,
        **msgspec.structs.asdict(chunking),
    )


def urn(p: Params) -> str:
    """Canonical form: fixed field order, so equal params always give equal text. `document` is safe
    inside it because `document.safe_name` allows no `;` or `:`."""
    return (
        f"document:{p.document};model:{p.model};chunk_size:{p.chunk_size};"
        f"chunk_merge_below:{p.chunk_merge_below};"
        f"chunk_frame:{'true' if p.chunk_frame else 'false'};chunker:{p.chunker};"
        f"chunk_version:{p.chunk_version};parser:{p.parser};"
        f"skip_ocr_pages:{'true' if p.skip_ocr_pages else 'false'}"
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
    return _PLAIN.append(pa.field("vector", pa.list_(pa.float32(), dims)))


def _batch(part: int, rows: list[Row], dims: int | None) -> pa.RecordBatch:
    records = [chunk.record(row.chunk, row.vector, dims, part=part, seq=row.seq) for row in rows]
    return pa.RecordBatch.from_pylist(records, schema=_schema(dims))


def _rows(batch: pa.RecordBatch) -> list[Row]:
    return [
        Row(chunk=msgspec.convert(record, Chunk), vector=record.get("vector"), seq=record["seq"])
        for record in batch.to_pylist()
    ]


def _unit_sum(rows: list[Row], dims: int) -> np.ndarray:
    """The sum of the rows' vectors, each scaled to length one first, so a long chunk weighs no
    more than a short one. An empty part sums to zero."""
    vectors = [row.vector for row in rows if row.vector is not None]
    matrix = np.asarray(vectors, dtype=np.float32).reshape(-1, dims)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms > 0, norms, 1)).sum(axis=0)


def _merge(parts: list[Path], target: Path, dims: int | None) -> tuple[int, int, bytes | None]:
    """Stream every `rows.json` into `target` as one row group each, through a `.tmp` and one
    replace, so a reader never sees a partial file. Returns (rows, bytes, document vector). An
    empty part still gets a row group, so group `n` is always part `n` (an empty group is
    skipped on read).

    This is also where `Row.seq` is filled in: the parts are chunked in parallel and each one
    numbers its chunks from zero, so the merge is the first place that sees the whole document
    in order. And where the document vector is summed, since every row passes through here once.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    summed = None if dims is None else np.zeros(dims, dtype=np.float32)
    with home.atomic_replace(target) as tmp, pq.ParquetWriter(tmp, _schema(dims)) as writer:
        for part, path in enumerate(parts):
            rows = msgspec.json.decode(path.read_bytes(), type=list[Row])
            for seq, row in enumerate(rows, total + 1):
                row.seq = seq
            total += len(rows)
            writer.write_batch(_batch(part, rows, dims))
            if summed is not None:
                summed += _unit_sum(rows, len(summed))
    return total, target.stat().st_size, _document_vector(summed)


def _document_vector(summed: np.ndarray | None) -> bytes | None:
    """The mean direction of a document's chunks, normalized, as the float32 bytes stored."""
    if summed is None:
        return None
    norm = float(np.linalg.norm(summed))
    return (summed / norm).astype(np.float32).tobytes() if norm > 0 else None


# --- the cache -----------------------------------------------------------------


async def lookup(p: Params) -> str | None:
    """The cache id of a hit, else None: a row or a file on its own is an interrupted write."""
    id = key(p)
    async with db.read() as conn:
        found = await conn.scalar(select(embeddings.c.id).where(embeddings.c.id == id))
    if found is None or not await anyio.Path(file_path(p.document, id)).is_file():
        return None
    return id


async def write(p: Params, parts: list[Path], dims: int | None) -> str:
    """Merge the scratch rows of every part into the cache file, publish the row, then drop the
    scratch directory - last, so a retry before the row was written still finds its input."""
    id = key(p)
    target = file_path(p.document, id)
    rows, size, vector = await anyio.to_thread.run_sync(_merge, parts, target, dims)
    entry = Entry(
        **msgspec.structs.asdict(p),
        id=id,
        urn=urn(p),
        rows=rows,
        bytes=size,
        created_at=time.time(),
    )
    async with db.connect() as conn:
        await conn.execute(
            insert(embeddings)
            .values({**msgspec.to_builtins(entry), "vector": vector})
            .on_conflict_do_nothing()
        )
    await home.remove_tree(scratch_dir(p.document, id))
    return id


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


async def forget(doc: str) -> None:
    """Drop every cached embedding of one document, rows and files. For a reconversion: the
    markdown the rows were chunked from is about to change, so none of them is reusable."""
    async with db.connect() as conn:
        await conn.execute(delete(embeddings).where(embeddings.c.document == doc))
    await home.remove_tree(document.embeddings_dir(doc))


async def entries(doc: str) -> list[Entry]:
    """Every cache row of one document, newest first."""
    async with db.read() as conn:
        rows = await conn.execute(
            select(*ENTRY_COLUMNS)
            .where(embeddings.c.document == doc)
            .order_by(embeddings.c.created_at.desc())
        )
        return [db.row_to(Entry, row) for row in rows]


class Neighbour(msgspec.Struct):
    """A document close to another one, by the cosine of their document vectors (1 is the same
    direction)."""

    document: str
    similarity: float


async def nearest(doc: str, model: str, limit: int, but: Collection[str] = ()) -> list[Neighbour]:
    """The `limit` imported documents whose vector under `model` lies closest to `doc`'s, closest
    first, `but` left out: the copies of the same file, which would only fill the slots at 1.0.
    Empty while `doc` has no vector under it: still importing, or no embedding model.

    Each document is compared by its newest cache entry under the model: the entries of one
    document differ only in how it was chunked, which barely moves the mean. Every vector is read
    and compared in memory; a library of thousands of books is a few megabytes of them."""
    async with db.read() as conn:
        rows = await conn.execute(
            select(embeddings.c.document, embeddings.c.vector)
            .join_from(embeddings, documents, embeddings.c.document == documents.c.name)
            .where(
                embeddings.c.model == model,
                embeddings.c.vector.is_not(None),
                documents.c.status == document.DocumentStatus.IMPORTED,
            )
            .order_by(embeddings.c.created_at.desc())
        )
        vectors: dict[str, bytes] = {}
        for name, vector in rows:
            vectors.setdefault(name, vector)  # newest first, so the first one is kept
    target = vectors.pop(doc, None)
    for name in but:
        vectors.pop(name, None)
    if target is None or not vectors:
        return []
    # in a worker thread, like every other numpy and pyarrow call here: the matrix grows with
    # the library
    return await anyio.to_thread.run_sync(_closest, vectors, target, limit)


def _closest(vectors: dict[str, bytes], target: bytes, limit: int) -> list[Neighbour]:
    """The `limit` of `vectors` with the highest cosine to `target`; all are unit length."""
    names = list(vectors)
    matrix = np.frombuffer(b"".join(vectors.values()), dtype=np.float32).reshape(len(names), -1)
    scores = matrix @ np.frombuffer(target, dtype=np.float32)
    closest = np.argsort(-scores, kind="stable")[:limit]
    return [Neighbour(document=names[at], similarity=round(float(scores[at]), 4)) for at in closest]
