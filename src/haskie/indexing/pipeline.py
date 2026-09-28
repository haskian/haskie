"""Convert, embed and index steps as micro-batches of `batch_pages` pages, so memory stays bounded.

Three stages, each a set of independent, idempotent steps, and each with its own owner:

- convert (once per document, at import) -> `parts/NNNNNN.md`: pages [start, end) of the PDF,
  or the whole file for anything else. `finalize_convert` assembles them into `original.<ext>.md`.
  The parts stay for the life of the document: chunking runs per part with the part's line and
  char offsets, and a PDF's page-batch boundaries decide where the parts fall, so a collection that
  chunks the document with other settings later must re-chunk from the same boundaries or its
  offsets would not match the assembled markdown.
- embed (once per distinct `embed_cache.Params`) -> `embeddings/<id>.tmp/NNNNNN.rows.json`:
  chunks of one part with their vectors (CPU/GPU heavy; runs in parallel). `finalize_embed`
  merges every part into the one parquet cache file and publishes it (`embed_cache.write`),
  which also drops the scratch files. Skipped entirely when `embed_cache.lookup` finds the file.
- index (once per collection the document is attached to) -> rows of `index_group_parts`
  consecutive parts read out of the cache file and written to that collection's LanceDB table in
  one commit (fast; one writer per collection). `prepare_index` clears the document's older rows
  once before the first of them; `finalize_index` builds the full-text index if the collection
  has none yet.

Everything that costs O(collection) rather than O(document) is deferred to
`collection/maintenance.py`. `workflows.py` orchestrates these with DBOS; nothing here touches
document or membership rows.

Every function is `async def`: the file reads and writes await, and the CPU work (pdf parsing,
chunking, embedding) is handed to a worker thread through `cpu.on_cpu`, which is also where one
slot of the CPU budget is held. So a step holds a slot for its CPU work only, never for the file
IO or the LanceDB commit around it.
"""

from pathlib import Path

import anyio
import anyio.to_thread
import msgspec

from haskie import cpu, home
from haskie.catalogue.catalogue import EmbeddingModel
from haskie.collection.collection import Collection
from haskie.collection.index import Row
from haskie.document import convert
from haskie.document.document import Document
from haskie.indexing import chunk, embed_cache, models
from haskie.indexing.segment import CutReason
from haskie.settings import ChunkSettings

JOINER = "\n\n"  # between parts in the assembled markdown


class Batch(msgspec.Struct):
    seq: int
    start: int  # convert: first page, 0-based; embed/index: part number
    end: int  # convert: exclusive page bound; embed/index: part number + 1
    line_offset: int = 0  # embed: lines / chars / bytes preceding this part in the assembled file
    char_offset: int = 0
    byte_offset: int = 0
    # embed: the headings still open where this part starts, opened in an earlier one
    opened: list[chunk.Opened] = []
    # embed: why the part's first chunk starts and its last one ends (`chunk.split`)
    start_reason: CutReason = CutReason.EDGE
    end_reason: CutReason = CutReason.EDGE


# --- convert --------------------------------------------------------------------


async def plan_convert(doc: Document, batch_pages: int) -> list[Batch]:
    """One batch per `batch_pages` pages of a PDF, one for anything else. Starts the parts and
    the assembled markdown over: they are outputs of this conversion. The cached embeddings are
    outputs of it too, rows included, so `workflows.import_document` drops them through
    `embed_cache.forget` before this runs."""
    source = doc.source_path()
    await home.remove_tree(doc.parts_dir)
    await anyio.Path(doc.markdown).unlink(missing_ok=True)
    await anyio.Path(doc.parts_dir).mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() != ".pdf":
        return [Batch(seq=0, start=0, end=1)]  # anydoc/plain convert whole files in one go
    total = await cpu.on_cpu(convert.pdf_page_count, source)
    return [
        Batch(seq=seq, start=start, end=min(start + batch_pages, total))
        for seq, start in enumerate(range(0, total, batch_pages))
    ]


async def convert_batch(doc: Document, batch: Batch) -> int:
    """Write one part file; returns how many of its pages need OCR. `parser` and
    `skip_ocr_pages` are the document's own, fixed at import."""
    source = doc.source_path()
    if source.suffix.lower() == ".pdf":
        markdown, ocr_pages, _ = await cpu.off_interpreter(
            convert.pdf_pages_markdown,
            source,
            list(range(batch.start, batch.end)),
            doc.skip_ocr_pages,
        )
    else:
        markdown = await cpu.on_cpu(convert.to_markdown, source, doc.parser)
        ocr_pages = []
    await home.atomic_write(doc.part_path(batch.seq), markdown)
    return len(ocr_pages)


async def finalize_convert(doc: Document, batches: list[Batch], ocr_total: int) -> None:
    """Apply the OCR policy over the whole document, then stream parts into one markdown file."""
    if doc.source_path().suffix.lower() == ".pdf":
        total = sum(b.end - b.start for b in batches)
        convert.check_ocr_policy(ocr_total, total, doc.skip_ocr_pages)
    parts = [
        await anyio.Path(doc.part_path(batch.seq)).read_text(encoding="utf-8")
        for batch in sorted(batches, key=lambda b: b.seq)
    ]
    # one replace, so a reader never sees a half-assembled document; the parts are already
    # in memory one at a time during convert, so holding the joined text adds no new bound
    await home.atomic_write(doc.markdown, JOINER.join(parts))


# --- embed ------------------------------------------------------------------------


async def _parts(doc: Document) -> list[Path]:
    parts_dir, markdown = doc.parts_dir, doc.markdown

    def listing() -> list[Path]:
        return sorted(p for p in parts_dir.glob("*.md") if p.stem.isdigit())

    parts = await anyio.to_thread.run_sync(listing)  # `glob` has no async form
    if not parts or not await anyio.Path(markdown).exists():
        raise FileNotFoundError(f"markdown parts missing, convert first: {doc.name}")
    return parts


async def plan_embed(doc: Document) -> list[Batch]:
    """One batch per part, carrying the part's offsets inside the assembled file, the headings
    still open where it starts (one pass), and why its first chunk starts and its last one ends:
    the document's edge, a heading the next part opens with, or a section going on (`PART`)."""
    texts = [await anyio.Path(part).read_text(encoding="utf-8") for part in await _parts(doc)]
    headed = await cpu.on_cpu(lambda: [chunk.opens_with_heading(text) for text in texts])
    # why a part's start is cut, where the one before it meets it; the first starts the document
    meets = [CutReason.EDGE] + [
        CutReason.HEADING if heading else CutReason.PART for heading in headed[1:]
    ]
    batches: list[Batch] = []
    line_offset = char_offset = byte_offset = 0
    opened: list[chunk.Opened] = []
    for i, text in enumerate(texts):
        batches.append(
            Batch(
                seq=i,
                start=i,
                end=i + 1,
                line_offset=line_offset,
                char_offset=char_offset,
                byte_offset=byte_offset,
                opened=opened,
                start_reason=meets[i],
                end_reason=meets[i + 1] if i + 1 < len(texts) else CutReason.EDGE,
            )
        )
        opened = await cpu.on_cpu(chunk.open_headings, text, opened)  # a parse: off the loop
        line_offset += text.count("\n") + JOINER.count("\n")
        char_offset += len(text) + len(JOINER)
        byte_offset += len(text.encode()) + len(JOINER.encode())
    return batches


async def embed_batch(
    doc: Document,
    batch: Batch,
    cache_id: str,
    chunking: ChunkSettings,
    embedding: EmbeddingModel | None,
) -> int:
    """Chunk one part, embed the chunks, write the part's scratch `rows.json` under the cache
    id being computed; returns the chunk count."""
    text = await anyio.Path(doc.part_path(batch.seq)).read_text(encoding="utf-8")
    if embedding is not None:
        # before the CPU work rather than between chunking and embedding, so both share one slot;
        # fail fast, not a stalled worker
        await models.require_ready(models.ModelKind.EMBEDDING, embedding.name)

    def chunk_and_embed() -> list[Row]:
        chunks = chunk.split(
            text,
            chunking,
            batch.line_offset,
            batch.char_offset,
            batch.byte_offset,
            batch.opened,
            batch.start_reason,
            batch.end_reason,
        )
        vectors: list[list[float] | None] = [None] * len(chunks)
        if embedding is not None and chunks:
            from haskie.indexing.embed import embed_texts

            read = [chunk.framed(c.frame, c.text) for c in chunks]
            vectors = list(embed_texts(embedding, read))
        return [Row(chunk=c, vector=v) for c, v in zip(chunks, vectors, strict=True)]

    rows = await cpu.on_cpu(chunk_and_embed)
    target = embed_cache.rows_path(doc.name, cache_id, batch.seq)
    await anyio.Path(target.parent).mkdir(parents=True, exist_ok=True)
    await home.atomic_write(target, msgspec.json.encode(rows))
    return len(rows)


async def finalize_embed(
    doc: Document, params: embed_cache.Params, embedding: EmbeddingModel | None
) -> str:
    """Publish one computed embedding and return its cache id.

    A part whose rows were never written raises `FileNotFoundError` out of the merge, which
    leaves no partial cache file behind (see `embed_cache._merge`)."""
    cache_id = embed_cache.key(params)
    count = len(await _parts(doc))
    paths = [embed_cache.rows_path(doc.name, cache_id, part) for part in range(count)]
    return await embed_cache.write(params, paths, embedding.dims if embedding else None)


# --- index ----------------------------------------------------------------------


async def plan_index(doc: Document, cache_id: str, group_parts: int) -> list[Batch]:
    """One batch per group of at most `group_parts` consecutive parts of the cache file. A group
    is one LanceDB commit, so the fragment count of a collection follows documents rather than
    pages."""
    count = await embed_cache.row_groups(doc.name, cache_id)
    return [
        Batch(seq=seq, start=start, end=min(start + group_parts, count))
        for seq, start in enumerate(range(0, count, group_parts))
    ]


async def prepare_index(
    collection: Collection, doc: Document, embedding: EmbeddingModel | None
) -> None:
    """Make the collection's table ready to take one document's rows again: recreate a table an
    older build or another embedding left behind, then drop the rows the document already has
    there (a previous attach, possibly under other chunk settings)."""
    index = collection.index_with(embedding)
    await index.reset_for_write()
    await index.delete_document(doc.name)


async def index_batch(
    collection: Collection,
    doc: Document,
    cache_id: str,
    batch: Batch,
    embedding: EmbeddingModel | None,
) -> int:
    """Replace the rows of parts [start, end) in the collection's index, in one commit; returns
    the row count. Idempotent: the range is deleted before it is added, so a repeat after a
    crash between the LanceDB commit and the step checkpoint rewrites exactly the same rows.

    The document's older rows are gone before the first batch runs (see `prepare_index`)."""
    index = collection.index_with(embedding)
    await index.delete_parts(doc.name, batch.start, batch.end)
    return await index.add_parts(
        doc.name,
        doc.relative(doc.source_path()),
        doc.relative(doc.markdown),
        embed_cache.read(doc.name, cache_id, batch.start, batch.end),
    )


async def finalize_index(collection: Collection, embedding: EmbeddingModel | None) -> None:
    """Build the collection's full-text index if it has none. Rows written after the build are
    found by a scan until `maintenance` folds them in, so this is O(collection) once, not per
    document."""
    await collection.index_with(embedding).finish()
