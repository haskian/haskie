"""Convert, embed and index steps as micro-batches of `batch_pages` pages, so memory stays bounded.

Each batch is an independent, idempotent step:
- convert batch -> `<doc>.parts/NNNNNN.md` (pages [start, end) of the PDF)
- embed batch   -> `<doc>.parts/NNNNNN.rows.json`: chunks of one part with their vectors
                   (CPU/GPU heavy; runs in parallel)
  The parts directory sits beside the assembled markdown, under the document's shard
  (see `layout.shard`).
- index batch   -> rows of `index_group_parts` consecutive parts written to LanceDB in one
                   commit (fast; one writer per library)
Finalize steps assemble the full markdown file / build the full-text index if the library has
none yet, and the index stage drops the parts directory it consumed. Everything that costs
O(library) rather than O(document) is deferred to `maintenance.py`.
workflows.py orchestrates these with DBOS; nothing here touches the metadata DB.

Every function is `async def`: the file reads and writes await, and the CPU work (pdf parsing,
chunking, embedding) is handed to a worker thread through `cpu.on_cpu`, which is also where one
slot of the CPU budget is held. So a step holds a slot for its CPU work only, never for the file
IO or the LanceDB commit around it.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import anyio.to_thread
import msgspec

from haskie import chunk, convert, cpu, home, models
from haskie.index import Row
from haskie.library import Library
from haskie.settings import ConversionSettings, EmbeddingModel

JOINER = "\n\n"  # between parts in the assembled markdown


class Batch(msgspec.Struct):
    seq: int
    start: int  # convert: first page, 0-based; embed/index: part number
    end: int  # convert: exclusive page bound; embed/index: part number + 1
    line_offset: int = 0  # embed: lines / chars preceding this part in the assembled file
    char_offset: int = 0


# --- convert --------------------------------------------------------------------


async def plan_convert(lib: Library, doc: str, batch_pages: int) -> list[Batch]:
    source = lib.source_path(doc)
    await home.remove_tree(lib.parts_dir(doc))
    await anyio.Path(lib.parts_dir(doc)).mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() != ".pdf":
        return [Batch(seq=0, start=0, end=1)]  # anydoc/plain convert whole files in one go
    total = await cpu.on_cpu("convert", convert.pdf_page_count, source)
    return [
        Batch(seq=seq, start=start, end=min(start + batch_pages, total))
        for seq, start in enumerate(range(0, total, batch_pages))
    ]


async def convert_batch(lib: Library, doc: str, batch: Batch, settings: ConversionSettings) -> int:
    """Write one part file; returns how many of its pages need OCR."""
    source = lib.source_path(doc)
    if source.suffix.lower() == ".pdf":
        markdown, ocr_pages, _ = await cpu.off_interpreter(
            "convert",
            convert.pdf_pages_markdown,
            source,
            list(range(batch.start, batch.end)),
            settings.skip_ocr_pages,
        )
    else:
        markdown = await cpu.on_cpu("convert", convert.to_markdown, source, settings.parser)
        ocr_pages = []
    await home.atomic_write(lib.part_path(doc, batch.seq), markdown)
    return len(ocr_pages)


async def finalize_convert(
    lib: Library, doc: str, batches: list[Batch], ocr_total: int, settings: ConversionSettings
) -> Path:
    """Apply the OCR policy over the whole document, then stream parts into one markdown file."""
    if lib.source_path(doc).suffix.lower() == ".pdf":
        total = sum(b.end - b.start for b in batches)
        convert.check_ocr_policy(ocr_total, total, settings.skip_ocr_pages)
    target = lib.markdown_path(doc)
    await anyio.Path(target.parent).mkdir(parents=True, exist_ok=True)  # shard dir, on first use
    parts = [
        await anyio.Path(lib.part_path(doc, batch.seq)).read_text(encoding="utf-8")
        for batch in sorted(batches, key=lambda b: b.seq)
    ]
    # one replace, so a reader never sees a half-assembled document (B6); the parts are already
    # in memory one at a time during convert, so holding the joined text adds no new bound
    await home.atomic_write(target, JOINER.join(parts))
    return target


# --- embed ------------------------------------------------------------------------


async def _parts(lib: Library, doc: str) -> list[Path]:
    parts_dir, markdown = lib.parts_dir(doc), lib.markdown_path(doc)

    def listing() -> list[Path]:
        return sorted(p for p in parts_dir.glob("*.md") if p.stem.isdigit())

    parts = await anyio.to_thread.run_sync(listing)  # `glob` has no async form
    if not parts or not await anyio.Path(markdown).exists():
        raise FileNotFoundError(f"markdown parts missing, convert first: {doc}")
    return parts


async def plan_embed(lib: Library, doc: str) -> list[Batch]:
    """One batch per part, carrying the part's offsets inside the assembled file (one pass)."""
    batches: list[Batch] = []
    line_offset = char_offset = 0
    for i, part in enumerate(await _parts(lib, doc)):
        batches.append(
            Batch(seq=i, start=i, end=i + 1, line_offset=line_offset, char_offset=char_offset)
        )
        text = await anyio.Path(part).read_text(encoding="utf-8")
        line_offset += text.count("\n") + JOINER.count("\n")
        char_offset += len(text) + len(JOINER)
    return batches


async def embed_batch(
    lib: Library,
    doc: str,
    batch: Batch,
    settings: ConversionSettings,
    embedding: EmbeddingModel | None,
) -> int:
    """Chunk one part, embed the chunks, write `NNNNNN.rows.json`; returns the chunk count."""
    text = await anyio.Path(lib.part_path(doc, batch.seq)).read_text(encoding="utf-8")
    if embedding is not None:
        # before the CPU work rather than between chunking and embedding, so both share one slot
        await models.require_ready("embedding", embedding.name)  # fail fast, not a stalled worker

    def chunk_and_embed() -> list[Row]:
        # ponytail: heading ancestry is per part; headings opened in an earlier part are not parents
        chunks = chunk.split(text, settings, batch.line_offset, batch.char_offset)
        vectors: list[list[float] | None] = [None] * len(chunks)
        if embedding is not None and chunks:
            from haskie.embed import embed_texts

            vectors = list(embed_texts(embedding, [c.text for c in chunks]))
        return [Row(chunk=c, vector=v) for c, v in zip(chunks, vectors, strict=True)]

    rows = await cpu.on_cpu("embed", chunk_and_embed)
    await home.atomic_write(lib.rows_path(doc, batch.seq), msgspec.json.encode(rows))
    return len(rows)


# --- index ----------------------------------------------------------------------


async def plan_index(lib: Library, doc: str, group_parts: int) -> list[Batch]:
    """One batch per group of at most `group_parts` consecutive parts. A group is one LanceDB
    commit, so the fragment count of a library follows documents rather than pages."""
    count = len(await _parts(lib, doc))
    paths = [lib.rows_path(doc, part) for part in range(count)]

    def absent() -> list[int]:  # one thread hop for the whole check, not one per part
        return [part for part, path in enumerate(paths) if not path.exists()]

    missing = await anyio.to_thread.run_sync(absent)
    if missing:
        raise FileNotFoundError(f"rows missing for parts {missing}, embed first: {doc}")
    size = max(1, group_parts)
    return [
        Batch(seq=seq, start=start, end=min(start + size, count))
        for seq, start in enumerate(range(0, count, size))
    ]


async def _grouped_rows(
    lib: Library, doc: str, batch: Batch
) -> AsyncIterator[tuple[int, list[Row]]]:
    """The rows of parts [start, end), one decoded part at a time: a whole group of a long
    document never sits in memory as Python objects."""
    for part in range(batch.start, batch.end):
        raw = await anyio.Path(lib.rows_path(doc, part)).read_bytes()
        yield part, msgspec.json.decode(raw, type=list[Row])


async def index_batch(
    lib: Library, doc: str, batch: Batch, embedding: EmbeddingModel | None
) -> int:
    """Replace the rows of parts [start, end) in the index, in one commit; returns the row count.
    Idempotent: the range is deleted before it is added, so a repeat after a crash between the
    LanceDB commit and the step checkpoint rewrites exactly the same rows."""
    index = lib.index_with(embedding)
    if batch.seq == 0:
        # all LanceDB writes for a library happen here, serialized: recreate a table left by an
        # older build, then clear the document's old rows (stale parts of a longer, earlier
        # conversion) before its first new group
        await index.reset_for_write()
        await index.delete_document(doc)
    else:
        await index.delete_parts(doc, batch.start, batch.end)
    return await index.add_parts(
        doc,
        lib.relative(lib.source_path(doc)),
        lib.relative(lib.markdown_path(doc)),
        _grouped_rows(lib, doc, batch),
    )


async def finalize_index(lib: Library, doc: str, embedding: EmbeddingModel | None) -> None:
    """Build the library's full-text index if it has none. Rows written after the build are found
    by a scan until `maintenance` folds them in, so this is O(library) once, not per document."""
    await lib.index_with(embedding).finish()


async def cleanup_parts(lib: Library, doc: str) -> None:
    """Drop the micro-batch files of a finished document. Both the markdown parts and the rows
    files are derivable, and the rows files (chunks plus vectors) dwarf everything else a document
    stores. The assembled markdown stays: search results point into it. A re-index starts at
    `plan_convert`, which recreates the parts directory."""
    await home.remove_tree(lib.parts_dir(doc))
