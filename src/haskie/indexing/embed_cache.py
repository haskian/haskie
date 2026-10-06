"""The embedding cache: per (document, chunk settings, embedding model), a file of its chunks and
one of its sections, keyed by a canonical URN, plus the `embeddings` row that makes it visible.

Chunking and embedding a document is the expensive part of indexing, and it depends on nothing a
collection owns except its chunk settings. So it is computed once per distinct `Params` and kept
with the document (`documents/<shard>/<doc id>/embeddings/<id>.chunks.parquet`, `file_path`, and
`<id>.sections.parquet`, `sections_path`): a collection that attaches the document reads the rows
back out of the cache into its own LanceDB table (`pipeline.index_*`) and computes nothing when the
file already exists. The chunks' file has one row group per embed part (`pipeline.plan_embed`), in
part order, so the index stage can stream it a group at a time.

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

This module owns the parquet schema and the row shape it is read back into (`index.Row`), as
`collection/index.py` owns LanceDB's; the chunk columns inside both come from `chunk.record`. One
column is this module's own: `seq`, the row's 1-based position among the document's chunks, which
only the merge across parts can number (see `_merge`). File writes and reads run in a worker thread:
pyarrow is sync.

Each entry also keeps its document's sections (`sections_path`, `sections.build`): named at the
merge, and described by a stage of their own (`pipeline.describe_batch`), which reads both files
back (`inputs`). The sections file's schema metadata names the strategy that wrote its descriptors
(`described_by`), so an entry described by another strategy than the settings now ask for is
described again, from the cache, without embedding anything again."""

import hashlib
import time
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

from haskie import db, home
from haskie.catalogue.catalogue import EmbeddingModel
from haskie.collection.collection import MemberStatus
from haskie.collection.index import Row, vector_field
from haskie.document import document
from haskie.indexing import chunk
from haskie.indexing.chunk import CHUNK_VERSION, Chunk, Piece
from haskie.search import collapse
from haskie.sections import build
from haskie.settings import Chunker, ChunkSettings, Descriptors, Parser
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


def model_of(embedding: EmbeddingModel | None) -> str:
    """What the cache keys a model by (`Params.model`)."""
    return embedding.cache_name if embedding else NO_MODEL


def params(
    doc: document.Document, chunking: ChunkSettings, embedding: EmbeddingModel | None
) -> Params:
    """The key of one document under one collection's chunk settings and the global model."""
    return Params(
        document_id=doc.id,
        model=model_of(embedding),
        chunk_version=CHUNK_VERSION,
        parser=doc.parser,
        skip_ocr_pages=doc.skip_ocr_pages,
        **msgspec.structs.asdict(chunking),
    )


def urn(p: Params) -> str:
    """Canonical form: `name:value` per field in declaration order, so equal params always give
    equal text. `document_id` is safe inside it: an id is base58 letters and digits (`ids`)."""
    return ";".join(
        f"{name}:{str(value).lower() if isinstance(value, bool) else value}"
        for name, value in msgspec.structs.asdict(p).items()
    )


def key(p: Params) -> str:
    return hashlib.sha256(urn(p).encode("utf-8")).hexdigest()


# --- paths ---------------------------------------------------------------------


def file_path(doc: str, id: str) -> Path:
    return document.embeddings_dir(doc) / f"{id}.chunks.parquet"


def scratch_dir(doc: str, id: str) -> Path:
    """Where the embed slices of one computation leave their `NNNNNN.rows.json`. Per cache id,
    so two computations of one document with different settings never share a file; deleted by
    `write` once the parquet file holds every part."""
    return document.embeddings_dir(doc) / f"{id}.tmp"


def rows_path(doc: str, id: str, seq: int) -> Path:
    return scratch_dir(doc, id) / f"{home.part_name(seq)}.rows.json"


def descriptors_path(doc: str, id: str, seq: int) -> Path:
    """Each section's descriptors and description from one batch; gathered and deleted by
    `pipeline.finalize_describe`."""
    return scratch_dir(doc, id) / f"{home.part_name(seq)}.descriptors.json"


def descriptions_path(doc: str, id: str, seq: int) -> Path:
    """The prose descriptions one section batch wrote, before descriptors are extracted."""
    return scratch_dir(doc, id) / f"{home.part_name(seq)}.descriptions.json"


def sections_path(doc: str, id: str) -> Path:
    """The sections of one cached embedding, beside its chunks (`sections.build`)."""
    return document.embeddings_dir(doc) / f"{id}.sections.parquet"


# --- parquet -------------------------------------------------------------------

_PLAIN = pa.schema(
    [
        ("part", pa.int32()),
        ("seq", pa.int32()),
        ("id", pa.string()),
        ("section_ids", pa.list_(pa.string())),
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
    return _PLAIN if dims is None else _PLAIN.append(vector_field(dims))


def _batch(part: int, rows: list[Row], dims: int | None) -> pa.RecordBatch:
    records = [
        chunk.record(
            row.chunk,
            row.vector,
            dims,
            part=part,
            seq=row.seq,
            id=row.id,
            section_ids=row.section_ids,
        )
        for row in rows
    ]
    return pa.RecordBatch.from_pylist(records, schema=_schema(dims))


def _rows(batch: pa.RecordBatch) -> list[Row]:
    return [
        Row(
            chunk=msgspec.convert(record, Chunk),
            vector=record.get("vector"),
            seq=record["seq"],
            id=record["id"],
            section_ids=record["section_ids"],
        )
        for record in batch.to_pylist()
    ]


# a section's columns, as `build.Section` names them
_SECTIONS = pa.schema(
    [
        ("id", pa.string()),
        ("parent_id", pa.string()),  # None for the whole document
        ("headings", pa.list_(pa.string())),
        ("seq_start", pa.int32()),
        ("seq_end", pa.int32()),
        ("line_start", pa.int32()),
        ("line_end", pa.int32()),
        ("char_start", pa.int32()),
        ("char_end", pa.int32()),
        ("byte_start", pa.int32()),
        ("byte_end", pa.int32()),
        ("page_start", pa.int32()),
        ("page_end", pa.int32()),
        ("descriptors", pa.list_(pa.string())),
        ("description", pa.string()),
    ]
)


class _ChunkRow(msgspec.Struct):
    """A row without its vector: what the first pass of `_merge` reads to name the sections."""

    chunk: Chunk


class Merged(msgspec.Struct):
    """What `_merge` wrote: the sections it named, and the sum of every chunk's unit vector."""

    bytes: int
    rows: int
    sections: list[build.Section]  # without descriptors yet (`describe`)
    summed: np.ndarray | None  # None without an embedding model


def _merge(document_id: str, parts: list[Path], target: Path, dims: int | None) -> Merged:
    """Stream every `rows.json` into `target` as one row group each, through a `.tmp` and one
    replace, so a reader never sees a partial file. An empty part still gets a row group, so group
    `n` is always part `n` (an empty group is skipped on read).

    This is where the chunks are numbered (`Row.seq`) and named, and their sections too: the parts
    are chunked in parallel and each one numbers its chunks from zero, so the merge is the first
    place that sees the whole document in order. Two passes, so no more than one part's vectors
    are held at once: the first reads where every chunk runs and names the sections
    (`build.sections`), the second writes each chunk with its id and sections, and sums its unit
    vector into the document's.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    # held for naming alone, not beside the second pass's vectors
    found, chains = build.sections(
        document_id,
        [
            row.chunk
            for path in parts
            for row in msgspec.json.decode(path.read_bytes(), type=list[_ChunkRow])
        ],
    )
    summed = None if dims is None else np.zeros(dims, dtype=np.float64)
    count = 0
    with home.atomic_replace(target) as tmp, pq.ParquetWriter(tmp, _schema(dims)) as writer:
        for part, path in enumerate(parts):
            rows = msgspec.json.decode(path.read_bytes(), type=list[Row])
            for row in rows:
                chain = chains[count]
                count += 1
                row.seq = count
                row.section_ids = [found[at].id for at in chain]
                row.id = build.chunk_id(document_id, count)
            writer.write_batch(_batch(part, rows, dims))
            if summed is not None and rows:  # `_batch` has refused a row without a vector by now
                summed += collapse.unit_rows([row.vector for row in rows]).sum(axis=0)
    return Merged(bytes=target.stat().st_size, rows=count, sections=found, summed=summed)


# The key of the sections file's schema metadata that names the strategy its descriptors were
# written by (`Descriptors`); a file the merge wrote has none yet.
_DESCRIBED_BY = b"descriptors"


def _write_sections(path: Path, found: list[build.Section], by: Descriptors | None) -> None:
    schema = _SECTIONS if by is None else _SECTIONS.with_metadata({_DESCRIBED_BY: by.encode()})
    table = pa.Table.from_pylist([msgspec.to_builtins(one) for one in found], schema=schema)
    with home.atomic_replace(path) as tmp:
        pq.write_table(table, tmp)


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
    """Merge the scratch rows of every part into the cache file, name its sections into their own
    file, publish its row, then drop the scratch directory. The drop comes last, so a retry
    before the row was written still finds its input. Both files are in place before the row: a
    hit (`lookup`) has both. The sections have no descriptors yet: `describe` writes them, as a
    step of its own."""
    id = key(p)
    target = file_path(p.document_id, id)
    merged = await anyio.to_thread.run_sync(_merge, p.document_id, parts, target, dims)
    path = sections_path(p.document_id, id)
    await anyio.to_thread.run_sync(_write_sections, path, merged.sections, None)
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


def _described_by(path: Path) -> Descriptors | None:
    schema = pq.read_schema(path)
    found = (schema.metadata or {}).get(_DESCRIBED_BY)
    if found == Descriptors.LLM.encode() and "description" not in schema.names:
        return None  # the next index fills descriptions without re-embedding the chunks
    return None if found is None else Descriptors(found.decode())


async def described_by(doc: str, id: str) -> Descriptors | None:
    """The completed strategy; None before describing or for llm files needing descriptions."""
    return await anyio.to_thread.run_sync(_described_by, sections_path(doc, id))


class Described(msgspec.Struct):
    """What describing one cached embedding's sections reads (`build.describe`)."""

    sections: list[build.Section]
    prose: list[str]  # each chunk's (`build.prose`), in `seq` order
    vectors: np.ndarray | None  # each section's unit vector; None when not asked for


def _inputs(doc: str, id: str, vectors: bool) -> Described:
    """The sections and every chunk's prose, and with `vectors` each section's unit vector: the
    mean of its chunks' unit vectors, scaled to length one. Read a row group at a time, and only
    the columns needed, so no more than one part's vectors are held at once, as in `_merge`."""
    found = _read_sections(sections_path(doc, id))
    at = {one.id: position for position, one in enumerate(found)}
    prose: list[str] = []
    sums: np.ndarray | None = None
    columns = ["pieces", *(["section_ids", "vector"] if vectors else [])]
    with pq.ParquetFile(file_path(doc, id)) as file:
        for batch in file.iter_batches(columns=columns):
            pieces = msgspec.convert(batch.column("pieces").to_pylist(), list[list[Piece]])
            prose.extend(build.prose(one) for one in pieces)
            if not vectors:
                continue
            flat = batch.column("vector").flatten().to_numpy()
            units = collapse.unit_rows(flat.reshape(batch.num_rows, -1))
            if sums is None:
                sums = np.zeros((len(found), units.shape[1]), dtype=np.float64)
            for held, unit in zip(batch.column("section_ids").to_pylist(), units, strict=True):
                sums[[at[one] for one in held]] += unit
    return Described(found, prose, None if sums is None else collapse.unit_rows(sums))


async def inputs(doc: str, id: str, vectors: bool) -> Described:
    return await anyio.to_thread.run_sync(_inputs, doc, id, vectors)


async def write_descriptors(
    doc: str, id: str, described: list[build.Section], by: Descriptors | None
) -> None:
    """Replace the sections of one cached embedding with `described`, written by `by`."""
    await anyio.to_thread.run_sync(_write_sections, sections_path(doc, id), described, by)


def _read_sections(path: Path) -> list[build.Section]:
    return msgspec.convert(pq.read_table(path).to_pylist(), list[build.Section])


async def read_sections(doc: str, id: str) -> list[build.Section]:
    """The sections of one cached embedding, in document order, each with its descriptors; none
    once the entry is forgotten (a reconversion), which a search can meet midway."""
    try:
        return await anyio.to_thread.run_sync(_read_sections, sections_path(doc, id))
    except FileNotFoundError:
        return []


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
    how many chunks it sums, each document by the entry its rows were indexed from; None without
    one. What a search centres its vectors on (`search.section_map`), built from the document
    means rather than the collection's table: one row per document, not one per chunk."""
    async with db.read() as conn:
        found = (
            await conn.execute(
                select(embeddings.c.vector, embeddings.c.rows)
                .join_from(
                    embeddings,
                    collection_documents,
                    embeddings.c.id == collection_documents.c.cache_id,
                )
                .where(
                    collection_documents.c.collection == collection,
                    collection_documents.c.status == MemberStatus.INDEXED,
                    embeddings.c.model == model,
                    embeddings.c.vector.is_not(None),
                )
            )
        ).all()
    if not found:
        return None
    means = [np.frombuffer(vector, dtype=np.float32) for vector, _ in found]
    counts = np.asarray([count for _, count in found], dtype=np.float64)
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
