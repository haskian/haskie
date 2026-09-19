"""The embedding cache: one parquet file per (document, chunk settings, embedding model), keyed
by a canonical URN, plus the `embeddings` row that makes it visible.

Chunking and embedding a document is the expensive part of indexing, and it depends on nothing a
collection owns except its chunk settings. So it is computed once per distinct `Params` and kept
with the document (`documents/<shard>/<doc>/embeddings/<id>.parquet`): a collection that attaches
the document reads the rows back out of the cache into its own LanceDB table (`pipeline.index_*`)
and computes nothing when the file already exists. The parquet file has one row group per
convert part, in part order, so the index stage can stream it a group at a time.

`Params` is the whole key. Its fields are always serialized in the same order (`urn`), so the same
inputs hash to the same id; `accelerator` is left out because it selects an execution provider,
not a model, and embedding dims are left out because `settings.PROFILES` fixes them per model
name. `chunk.CHUNK_VERSION` is in, so a change to the splitting code retires every entry it would
have produced differently. The id is the full sha256 of the URN, not a truncated one: a collision
here serves one document's vectors as another's, so it is a correctness key, unlike `layout.shard`
(spread) or `textsearch.query_hash` (cursor validation).

Visibility: `lookup` answers a hit only when both the row and the file exist. `write` puts the file
in place (atomic replace) before it inserts the row (`insert or ignore`), so a reader never sees a
row without a file, and a retried or a losing concurrent write is a no-op rather than an error.
Two callers wanting the same missing entry are serialized above this module, by the DBOS
deduplication of `workflows.ensure_embedding`; this module only makes the outcome idempotent.

Module owns the parquet schema and the row shape it is read back into (`index.Row`), the way
`index.py` owns LanceDB's. File writes and reads run in a worker thread: pyarrow is sync.
"""

import hashlib
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio
import anyio.to_thread
import msgspec
import pyarrow as pa
import pyarrow.parquet as pq

from haskie import db, document, home
from haskie.chunk import CHUNK_VERSION, Chunk
from haskie.index import Row
from haskie.layout import PART_DIGITS
from haskie.settings import Chunker, ChunkSettings, EmbeddingModel

NO_MODEL = "none"  # the `model` of a profile without an embedding model: chunks only, no vectors


class Params(msgspec.Struct, frozen=True):
    """Everything the cached rows of one document depend on. Field order is the URN order."""

    doc: str
    model: str  # EmbeddingModel.name, or NO_MODEL
    chunk_size: int
    chunk_overlap: int
    chunker: Chunker
    chunk_version: int
    parser: str
    skip_ocr_pages: bool


class Entry(msgspec.Struct):
    """One `embeddings` row: what the cache holds for a document, for the API and the tests."""

    id: str
    document: str
    urn: str
    model: str
    chunk_size: int
    chunk_overlap: int
    chunker: Chunker
    chunk_version: int
    parser: str
    skip_ocr_pages: bool
    rows: int
    bytes: int
    created_at: float


# The struct's field order is the column order, so the SELECT and the row unpack cannot drift
# apart (`document.py` does the same for `documents`).
ENTRY_COLUMNS: tuple[str, ...] = tuple(f.encode_name for f in msgspec.structs.fields(Entry))
ENTRY_SELECT = ", ".join(ENTRY_COLUMNS)


def params(
    doc: document.Document, chunking: ChunkSettings, embedding: EmbeddingModel | None
) -> Params:
    """The key of one document under one collection's chunk settings and the global model."""
    return Params(
        doc=doc.name,
        model=embedding.name if embedding else NO_MODEL,
        chunk_size=chunking.chunk_size,
        chunk_overlap=chunking.chunk_overlap,
        chunker=chunking.chunker,
        chunk_version=CHUNK_VERSION,
        parser=doc.parser,
        skip_ocr_pages=doc.skip_ocr_pages,
    )


def urn(p: Params) -> str:
    """Canonical form: fixed field order, so equal params always give equal text. `doc` is safe
    inside it because `document.safe_name` allows no `;` or `:`."""
    return (
        f"document:{p.doc};model:{p.model};chunk_size:{p.chunk_size};"
        f"chunk_overlap:{p.chunk_overlap};chunker:{p.chunker};"
        f"chunk_version:{p.chunk_version};parser:{p.parser};"
        f"skip_ocr_pages:{'true' if p.skip_ocr_pages else 'false'}"
    )


def cache_id(u: str) -> str:
    return hashlib.sha256(u.encode("utf-8")).hexdigest()


def key(p: Params) -> str:
    return cache_id(urn(p))


# --- paths ---------------------------------------------------------------------


def file_path(doc: str, id: str) -> Path:
    return document.embeddings_dir(doc) / f"{id}.parquet"


def scratch_dir(doc: str, id: str) -> Path:
    """Where the embed slices of one computation leave their `NNNNNN.rows.json`. Per cache id,
    so two computations of one document with different settings never share a file; deleted by
    `write` once the parquet file holds every part."""
    return document.embeddings_dir(doc) / f"{id}.tmp"


def rows_path(doc: str, id: str, seq: int) -> Path:
    return scratch_dir(doc, id) / f"{seq:0{PART_DIGITS}d}.rows.json"


# --- parquet -------------------------------------------------------------------

_PLAIN = pa.schema(
    [
        ("part", pa.int32()),
        ("heading", pa.string()),
        ("text", pa.string()),
        ("line_start", pa.int32()),
        ("line_end", pa.int32()),
        ("char_start", pa.int32()),
        ("char_end", pa.int32()),
        ("parents", pa.list_(pa.string())),
        ("page_start", pa.int32()),
        ("page_end", pa.int32()),
    ]
)


def _schema(dims: int | None) -> pa.Schema:
    if dims is None:
        return _PLAIN
    return _PLAIN.append(pa.field("vector", pa.list_(pa.float32(), dims)))


def _batch(part: int, rows: list[Row], dims: int | None) -> pa.RecordBatch:
    chunks = [row.chunk for row in rows]
    columns: dict[str, object] = {
        "part": pa.array([part] * len(rows), pa.int32()),
        "heading": pa.array([c.heading for c in chunks], pa.string()),
        "text": pa.array([c.text for c in chunks], pa.string()),
        "line_start": pa.array([c.line_start for c in chunks], pa.int32()),
        "line_end": pa.array([c.line_end for c in chunks], pa.int32()),
        "char_start": pa.array([c.char_start for c in chunks], pa.int32()),
        "char_end": pa.array([c.char_end for c in chunks], pa.int32()),
        "parents": pa.array([c.parents for c in chunks], pa.list_(pa.string())),
        "page_start": pa.array([c.page_start for c in chunks], pa.int32()),
        "page_end": pa.array([c.page_end for c in chunks], pa.int32()),
    }
    if dims is not None:
        flat: list[float] = []
        for row in rows:
            if row.vector is None:
                raise ValueError("cache has a vector column but the row carries no vector")
            flat.extend(row.vector)
        columns["vector"] = pa.FixedSizeListArray.from_arrays(pa.array(flat, pa.float32()), dims)
    return pa.RecordBatch.from_pydict(columns, schema=_schema(dims))


def _rows(batch: pa.RecordBatch) -> list[Row]:
    columns = batch.to_pydict()
    vectors = columns.get("vector") or [None] * batch.num_rows
    return [
        Row(
            chunk=Chunk(
                heading=columns["heading"][i],
                text=columns["text"][i],
                line_start=columns["line_start"][i],
                line_end=columns["line_end"][i],
                char_start=columns["char_start"][i],
                char_end=columns["char_end"][i],
                parents=list(columns["parents"][i]),
                page_start=columns["page_start"][i],
                page_end=columns["page_end"][i],
            ),
            vector=vectors[i],
        )
        for i in range(batch.num_rows)
    ]


def _merge(parts: list[Path], target: Path, dims: int | None) -> tuple[int, int]:
    """Stream every `rows.json` into `target` as one row group each, through a `.tmp` and one
    replace, so a reader never sees a partial file. Returns (rows, bytes). An empty part still
    gets a row group, so group `n` is always part `n` (an empty group is skipped on read).

    Sync: pyarrow has no async API, and one part at a time is all that sits in memory."""
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with home.atomic_replace(target) as tmp, pq.ParquetWriter(tmp, _schema(dims)) as writer:
        for part, path in enumerate(parts):
            rows = msgspec.json.decode(path.read_bytes(), type=list[Row])
            writer.write_batch(_batch(part, rows, dims))
            total += len(rows)
    return total, target.stat().st_size


# --- the cache -----------------------------------------------------------------


async def lookup(p: Params) -> str | None:
    """The cache id when both the row and the file exist, else None: either one alone is an
    interrupted write, which is a miss to recompute, not an error."""
    id = key(p)
    async with db.connect() as conn:
        cursor = await conn.execute("select 1 from embeddings where id = ?", (id,))
        row = await cursor.fetchone()
    if row is None or not await anyio.Path(file_path(p.doc, id)).is_file():
        return None
    return id


async def write(p: Params, parts: list[Path], dims: int | None) -> str:
    """Merge the scratch rows of every part into the cache file, then publish the row.

    File first, row second (see the module docstring); `insert or ignore` keeps a retry after a
    crash, or the loser of two concurrent writers, a no-op. The scratch directory goes last, so
    a retry before the row was written still finds its input."""
    id = key(p)
    target = file_path(p.doc, id)
    rows, size = await anyio.to_thread.run_sync(_merge, parts, target, dims)
    async with db.connect() as conn:
        await conn.execute(
            "insert or ignore into embeddings (id, document, urn, model, chunk_size, "
            "chunk_overlap, chunker, chunk_version, parser, skip_ocr_pages, rows, bytes, "
            "created_at) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                id,
                p.doc,
                urn(p),
                p.model,
                p.chunk_size,
                p.chunk_overlap,
                p.chunker,
                p.chunk_version,
                p.parser,
                p.skip_ocr_pages,
                rows,
                size,
                time.time(),
            ),
        )
    await home.remove_tree(scratch_dir(p.doc, id))
    return id


def _num_row_groups(path: Path) -> int:
    with pq.ParquetFile(path) as file:
        return file.num_row_groups


def _group(file: pq.ParquetFile, part: int) -> list[Row]:
    """One decoded row group; empty when the part holds no chunk (see `_merge`)."""
    table = file.read_row_group(part)
    return _rows(table.to_batches()[0]) if table.num_rows else []


async def row_groups(doc: str, id: str) -> int:
    """How many parts the cache file holds: one row group each."""
    path = file_path(doc, id)
    if not await anyio.Path(path).is_file():
        raise FileNotFoundError(f"embedding cache missing: {doc} {id}")
    return await anyio.to_thread.run_sync(_num_row_groups, path)


async def read(doc: str, id: str, start: int, end: int) -> AsyncIterator[tuple[int, list[Row]]]:
    """The rows of parts `[start, end)`, one decoded row group at a time: a whole group of a long
    document never sits in memory as Python objects.

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
        await conn.execute("delete from embeddings where document = ?", (doc,))
    await home.remove_tree(document.embeddings_dir(doc))


def _entry(row: tuple) -> Entry:
    values = dict(zip(ENTRY_COLUMNS, row, strict=True))
    values["skip_ocr_pages"] = bool(values["skip_ocr_pages"])
    return Entry(**values)


async def entries(doc: str) -> list[Entry]:
    """Every cache row of one document, newest first."""
    async with db.connect() as conn:
        cursor = await conn.execute(
            f"select {ENTRY_SELECT} from embeddings where document = ? order by created_at desc",
            (doc,),
        )
        rows: list[Any] = list(await cursor.fetchall())
    return [_entry(row) for row in rows]
