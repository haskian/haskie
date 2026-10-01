"""Convert, embed and index steps as micro-batches cut by sections, so memory stays bounded.

Three stages, each a set of independent, idempotent steps, and each with its own owner. Work is
cut where the document's sections start, packed up to `batch_pages` pages a batch, and at pages
where a section is longer, when there are pages to cut at (`parts.cuts`):

- convert (once per document, at import) -> `parts/NNNNNN.md`: pages [start, end) of the PDF,
  cut where its bookmarks start, or the whole file for anything else. `finalize_convert`
  assembles them into `original.<ext>.md`.
- embed (once per distinct `embed_cache.Params`) -> `embeddings/<id>.tmp/NNNNNN.rows.json`: chunks
  of one part of the assembled markdown with their vectors (CPU/GPU heavy; runs in parallel). A part
  is cut where a heading starts, else at a page marker inside a section longer than a batch;
  markdown without page markers is cut at headings alone (`plan_embed`). The cut depends on the
  markdown alone, so every collection that chunks the document, with any settings, chunks it from
  the same parts. `finalize_embed` merges every part into the one parquet cache file, numbering the
  chunks and naming and describing their sections, and publishes it (`embed_cache.write`), which
  also drops the scratch files. Skipped entirely when `embed_cache.lookup` finds the file.
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

import bisect
from functools import partial

import anyio
import anyio.to_thread
import msgspec

from haskie import cpu, home
from haskie.catalogue.catalogue import EmbeddingModel
from haskie.collection.collection import Collection
from haskie.collection.index import Row
from haskie.document import convert
from haskie.document.bookmarks import Bookmark
from haskie.document.document import Document
from haskie.indexing import chunk, embed_cache, models, parts, segment
from haskie.indexing.segment import CutReason, SpanKind
from haskie.sections import descriptors
from haskie.settings import ChunkSettings

JOINER = "\n\n"  # between convert parts in the assembled markdown
# About one printed page of markdown: what a page is where the text has no page markers to count
# (`plan_embed`). Judgement: one 657-page book held about 2,100 characters a page.
PAGE_CHARS = 3_000


class Batch(msgspec.Struct):
    seq: int
    start: int  # convert: first page, 0-based; embed/index: part number
    end: int  # convert: exclusive page bound; embed/index: part number + 1
    line_offset: int = 0  # embed: lines / chars / bytes preceding this part in the assembled file
    char_offset: int = 0
    byte_offset: int = 0
    byte_end: int = 0  # embed: where the part ends in the assembled file, exclusive
    # embed: the headings still open where this part starts, opened in an earlier one
    opened: list[chunk.Opened] = []
    # embed: why the part's first chunk starts and its last one ends (`chunk.split`)
    start_reason: CutReason = CutReason.EDGE
    end_reason: CutReason = CutReason.EDGE
    page: int | None = None  # embed: the page open where the part starts, by an earlier marker
    # convert: the PDF's bookmarks of these pages and the page before, which set its headings
    # (`convert.pdf_pages_markdown`), read once when the conversion is planned; None when the
    # document has none that do, so its converter's headings stand
    bookmarks: list[Bookmark] | None = None


# --- convert --------------------------------------------------------------------


async def plan_convert(doc: Document, batch_pages: int) -> list[Batch]:
    """A PDF's pages cut where its bookmarks start, at most `batch_pages` a batch unless one
    section is longer, and every `batch_pages` pages without bookmarks; one batch for anything
    else, which its converter reads whole. Starts the parts and
    the assembled markdown over: they are outputs of this conversion. The cached embeddings are
    outputs of it too, rows included, so `workflows.import_document` drops them through
    `embed_cache.forget` before this runs."""
    source = doc.source_path()
    await home.remove_tree(doc.parts_dir)
    await anyio.Path(doc.markdown).unlink(missing_ok=True)
    await anyio.Path(doc.parts_dir).mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() != ".pdf":
        return [Batch(seq=0, start=0, end=1)]  # anydoc/plain convert whole files in one go
    found = await cpu.off_interpreter(convert.pdf_bookmarks, source)
    marks = found.headings
    return [
        Batch(
            seq=seq,
            start=start,
            end=end,
            # 1-based pages start..end: the batch's own, and the one before it
            bookmarks=None if marks is None else [m for m in marks if start <= m.page <= end],
        )
        for seq, (start, end) in enumerate(
            parts.cuts(found.pages, found.starts, lambda start: start + batch_pages)
        )
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
            batch.bookmarks,
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


async def plan_embed(doc: Document, batch_pages: int) -> list[Batch]:
    """The assembled markdown cut into parts where its headings start, packed up to `batch_pages`
    pages a part, and a section longer than that at its page markers (`parts.cuts`). Pages are
    counted by the markers the converter writes, else by `PAGE_CHARS` characters each. Each part
    carries its offsets inside the file, the headings still open where it starts, and why its first
    chunk starts and its last one ends: the document's edge, a heading the next part opens with, or
    a section going on (`PART`)."""
    try:
        markdown = await anyio.Path(doc.markdown).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"markdown missing, convert first: {doc.name}") from None
    return await cpu.on_cpu(_embed_batches, markdown, batch_pages)


def _embed_batches(markdown: str, batch_pages: int) -> list[Batch]:
    # a heading right behind its page markers is cut ahead of them, so its part knows its page
    behind = {run.end(): run.end("before") for run in convert.MARKERS.finditer(markdown)}
    headings = [
        behind.get(b.start, b.start) for b in segment.blocks(markdown) if b.kind == SpanKind.HEADING
    ]
    found = list(convert.PAGE_MARKER.finditer(markdown))
    markers = [one.start() for one in found]
    reach = (
        parts.pages(markers, batch_pages)
        if markers
        else lambda start: start + batch_pages * PAGE_CHARS
    )
    ranges = parts.cuts(len(markdown), headings, reach, markers)
    texts = [markdown[start:end] for start, end in ranges]
    # why a part's start is cut, where the one before it meets it; the first starts the document
    meets = [CutReason.EDGE] + [
        CutReason.HEADING if chunk.opens_with_heading(text) else CutReason.PART
        for text in texts[1:]
    ]
    batches: list[Batch] = []
    line_offset = byte_offset = 0
    opened: list[chunk.Opened] = []
    for i, ((start, _), text) in enumerate(zip(ranges, texts, strict=True)):
        size = len(text.encode())
        before = bisect.bisect_left(markers, start) - 1  # the last marker ahead of the part
        batches.append(
            Batch(
                seq=i,
                start=i,
                end=i + 1,
                line_offset=line_offset,
                char_offset=start,
                byte_offset=byte_offset,
                byte_end=byte_offset + size,
                opened=opened,
                start_reason=meets[i],
                end_reason=meets[i + 1] if i + 1 < len(texts) else CutReason.EDGE,
                page=int(found[before].group(1)) if before >= 0 else None,
            )
        )
        opened = chunk.open_headings(text, opened)
        line_offset += text.count("\n")
        byte_offset += size
    return batches


async def _part_text(doc: Document, batch: Batch) -> str:
    """The part of the assembled markdown one embed batch chunks: its bytes, read alone."""
    async with await anyio.open_file(doc.markdown, "rb") as file:
        await file.seek(batch.byte_offset)
        return (await file.read(batch.byte_end - batch.byte_offset)).decode("utf-8")


async def embed_batch(
    doc: Document,
    batch: Batch,
    cache_id: str,
    chunking: ChunkSettings,
    embedding: EmbeddingModel | None,
) -> int:
    """Chunk one part, embed the chunks, write the part's scratch `rows.json` under the cache
    id being computed; returns the chunk count."""
    text = await _part_text(doc, batch)
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
            batch.page,
        )
        vectors: list[list[float] | None] = [None] * len(chunks)
        if embedding is not None and chunks:
            from haskie.indexing.embed import embed_texts

            read = [chunk.framed(c.frame, c.text) for c in chunks]
            vectors = list(embed_texts(embedding, read))
        return [Row(chunk=c, vector=v) for c, v in zip(chunks, vectors, strict=True)]

    rows = await cpu.on_cpu(chunk_and_embed)
    target = embed_cache.rows_path(doc.id, cache_id, batch.seq)
    await anyio.Path(target.parent).mkdir(parents=True, exist_ok=True)
    await home.atomic_write(target, msgspec.json.encode(rows))
    return len(rows)


async def finalize_embed(
    doc: Document, params: embed_cache.Params, embedding: EmbeddingModel | None, count: int
) -> None:
    """Publish one computed embedding of `count` parts, its sections described, under
    `embed_cache.key(params)`. The model embeds the descriptor candidates: it fails fast when not
    loaded, as `embed_batch` does.

    A part whose rows were never written raises `FileNotFoundError` out of the merge, which
    leaves no partial cache file behind (see `embed_cache._merge`)."""
    cache_id = embed_cache.key(params)
    paths = [embed_cache.rows_path(doc.id, cache_id, part) for part in range(count)]
    embed: descriptors.Embed | None = None
    if embedding is not None:
        await models.require_ready(models.ModelKind.EMBEDDING, embedding.name)
        embed = partial(_embed, embedding)
    await embed_cache.write(params, paths, embedding.dims if embedding else None, embed)


def _embed(embedding: EmbeddingModel, texts: list[str]) -> list[list[float]]:
    from haskie.indexing.embed import embed_texts  # heavy imports, as in `embed_batch`

    return embed_texts(embedding, texts)


# --- index ----------------------------------------------------------------------


async def plan_index(doc: Document, cache_id: str, group_parts: int) -> list[Batch]:
    """One batch per group of at most `group_parts` consecutive parts of the cache file. A group
    is one LanceDB commit, so the fragment count of a collection follows documents rather than
    pages."""
    count = await embed_cache.row_groups(doc.id, cache_id)
    return [
        Batch(seq=seq, start=start, end=min(start + group_parts, count))
        for seq, start in enumerate(range(0, count, group_parts))
    ]


async def prepare_index(
    collection: Collection, doc: Document, embedding: EmbeddingModel | None, cache_id: str
) -> None:
    """Make the collection's table ready to take one document's rows again: recreate a table an
    older build or another embedding left behind, drop the rows the document already has there
    (a previous attach, possibly under other chunk settings), and name the cache entry the new
    rows come from, so a reader of their sections finds the ones their ids name."""
    index = collection.index_with(embedding)
    await index.reset_for_write()
    await index.delete_document(doc.id)
    await collection.set_member_entry(doc.id, cache_id)


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
    await index.delete_parts(doc.id, batch.start, batch.end)
    return await index.add_parts(
        doc.id,
        doc.relative(doc.source_path()),
        doc.relative(doc.markdown),
        embed_cache.read(doc.id, cache_id, batch.start, batch.end),
    )


async def finalize_index(collection: Collection, embedding: EmbeddingModel | None) -> None:
    """Build the collection's full-text index if it has none. Rows written after the build are
    found by a scan until `maintenance` folds them in, so this is O(collection) once, not per
    document."""
    await collection.index_with(embedding).finish()
