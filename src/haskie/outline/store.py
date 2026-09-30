"""Where a document's outline is kept: a JSON file beside its markdown, and its nodes with their
vectors in one LanceDB index across every collection.

- **The file** (`documents/<shard>/<id>/outline.json`, `Outline`) is what a reader reads:
  `document_outline` one document, `search_sections` every document a scan reached. It names the
  embedding model the outline was built under, so a model change builds it again (`current`).
  No vectors: they would make it ten times the size, and no reader of the file needs them.
- **The index** (`~/.haskie/outlines/`) holds every node of every document once, whichever
  collections hold the document, each with its vector (`build.describe`): one table per
  embedding model (`table_name`), so a model change drops nothing, and a document whose file is
  current always has its rows. A document's nodes replace its old ones in one commit
  (`merge_insert`), so a reader sees one outline or the other, never half of each. Concurrent
  writers of one table need no lock: LanceDB retries a conflicting `merge_insert` (measured, 150
  writers at once, none failed).

`save` writes the index first and the file last: the file is what `current` checks, so a write
that fails between the two is done again rather than left half done.
"""

import re
from datetime import timedelta
from pathlib import Path

import anyio
import anyio.to_thread
import lancedb
import msgspec
import numpy as np
import pyarrow as pa

from haskie import home
from haskie.collection.index import quoted, vector_field
from haskie.document import document
from haskie.outline.build import Node


class Outline(msgspec.Struct, frozen=True):
    """A document's outline as its file holds it."""

    model: str  # the `EmbeddingModel.cache_name` it was built under, or `embed_cache.NO_MODEL`
    nodes: list[Node]  # in document order, the whole document first


class _Model(msgspec.Struct):
    """The one field `current` decodes: msgspec skips the nodes without building them."""

    model: str


def path(doc: str) -> Path:
    return document.root(doc) / "outline.json"


async def current(doc: str, model: str) -> bool:
    """Whether the document has an outline built under `model`."""
    try:
        data = await anyio.Path(path(doc)).read_bytes()
    except FileNotFoundError:
        return False
    return msgspec.json.decode(data, type=_Model).model == model


async def read(docs: list[str]) -> dict[str, list[Node]]:
    """The outline nodes of each document (by id), in document order; a document without an
    outline is absent (still importing, or reconverted since a search read its rows). One
    worker-thread hop for all of them."""
    return await anyio.to_thread.run_sync(_read, docs)


def _read(docs: list[str]) -> dict[str, list[Node]]:
    found: dict[str, list[Node]] = {}
    for doc in docs:
        try:
            found[doc] = msgspec.json.decode(path(doc).read_bytes(), type=Outline).nodes
        except FileNotFoundError:
            continue
    return found


async def save(doc: str, outline: Outline, vectors: np.ndarray | None) -> None:
    """Put the document's outline in the index, then in its file. `vectors` holds one unit vector
    per node, None without an embedding model."""
    await _upsert(doc, outline, vectors)
    await home.atomic_write(path(doc), msgspec.json.encode(outline))


async def forget(doc: str) -> None:
    """Drop the document's outline, file and index rows under every model: its markdown is about
    to change, or the document is going away."""
    await anyio.Path(path(doc)).unlink(missing_ok=True)
    where = f"document_id = {quoted(doc)}"
    for table in await _tables():
        if await table.count_rows(where):  # a delete of nothing still commits a version
            await table.delete(where)


async def compact(keep: timedelta) -> int:
    """Merge the small fragments every document's write leaves into few, and drop versions older
    than `keep`, a grace for a reader still on one: how many fragments it removed. Each write
    scans every fragment of its table, so it slows as they pile up between two runs: 8 ms per
    write at one document, 92 ms at 1,500 (measured)."""
    removed = 0
    for table in await _tables():
        stats = await table.optimize(cleanup_older_than=keep)
        removed += stats.compaction.fragments_removed
    return removed


def table_name(model: str) -> str:
    """The table of one model's nodes, named after it: a table name holds letters, digits, `_`,
    `-` and `.` only, and a model's `cache_name` also `/`, `@` and `:`."""
    return "nodes-" + re.sub(r"[^\w.-]", "_", model)


# --- the index -------------------------------------------------------------------

_PLAIN = pa.schema(
    [
        ("document_id", pa.string()),  # `Document.id`
        ("position", pa.int32()),  # its place in the outline, 0 the whole document
        ("headings", pa.list_(pa.string())),  # `Node.headings`, outermost first
        ("line_start", pa.int32()),
        ("line_end", pa.int32()),
        ("char_start", pa.int32()),
        ("char_end", pa.int32()),
        ("byte_start", pa.int32()),
        ("byte_end", pa.int32()),
        ("page_start", pa.int32()),
        ("page_end", pa.int32()),
        ("keywords", pa.list_(pa.struct([("keyword", pa.string()), ("uses", pa.int32())]))),
    ]
)


def _table(doc: str, outline: Outline, vectors: np.ndarray | None, schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                **msgspec.to_builtins(node),
                "document_id": doc,
                "position": position,
                "keywords": [
                    {"keyword": word, "uses": uses} for word, uses in node.keywords.items()
                ],
                **({} if vectors is None else {"vector": vectors[position]}),
            }
            for position, node in enumerate(outline.nodes)
        ],
        schema=schema,
    )


async def _connect() -> lancedb.AsyncConnection:
    return await lancedb.connect_async(str(home.OUTLINE_ROOT))


async def _tables() -> list[lancedb.AsyncTable]:
    conn = await _connect()
    return [await conn.open_table(name) for name in (await conn.list_tables()).tables]


async def _upsert(doc: str, outline: Outline, vectors: np.ndarray | None) -> None:
    """Replace the document's rows in its model's table with the outline's nodes, in one
    commit."""
    schema = _PLAIN if vectors is None else _PLAIN.append(vector_field(vectors.shape[1]))
    conn = await _connect()
    table = await conn.create_table(table_name(outline.model), schema=schema, exist_ok=True)
    await (
        table.merge_insert(["document_id", "position"])
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .when_not_matched_by_source_delete(f"document_id = {quoted(doc)}")
        .execute(_table(doc, outline, vectors, schema))
    )
