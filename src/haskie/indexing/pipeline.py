"""Convert, embed and index steps as micro-batches cut by sections, so memory stays bounded.

Four stages, each a set of independent, idempotent steps, and each with its own owner. Work is
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
  chunks and naming their sections, and publishes it (`embed_cache.write`), which also drops the
  scratch files. Skipped entirely when `embed_cache.lookup` finds the file.
- describe (once per cached embedding and descriptor strategy, again on a cache hit whose
  descriptors another strategy wrote, see `embed_cache.described_by`) ->
  `embeddings/<id>.tmp/NNNNNN.descriptors.json`: the descriptors of a run of sections, sixteen a
  batch for the llm strategy, all of them in one for c-TF-IDF (`plan_describe`).
  `finalize_describe` puts them on the sections file and drops the scratch files. After it, the llm
  strategy writes what the whole document is about (`summarize`).
- vocabulary (once per burst of documents a collection indexes, under the llm strategy): its
  sections' descriptors embedded, the close pairs judged by the describer, and the variants
  clustered into preferred terms (`collection/vocabulary.py`, `sections.vocabulary`). Embedding and
  judging are steps of `VOCABULARY_EMBED` variants and `VOCABULARY_JUDGE` pairs, each skipping
  what an earlier run already did.
- index (once per collection the document is attached to) -> rows of `index_group_parts`
  consecutive parts read out of the cache file and written to that collection's LanceDB table in
  one commit (fast; one writer per collection). `prepare_index` clears the document's older rows
  once before the first of them; `finalize_index` builds the full-text index if the collection
  has none yet. The membership names its cache entry exactly while rows of it stand in the table:
  cleared before the old rows go, named again once a batch has written new ones.

Everything that costs O(collection) rather than O(document) is deferred to
`collection/maintenance.py`. `workflows.py` orchestrates these with DBOS; nothing here touches
document or membership rows.

Every function is `async def`: the file reads and writes await, and the CPU work (pdf parsing,
chunking, embedding) is handed to a worker thread through `cpu.on_cpu`, which is also where one
slot of the CPU budget is held. So a step holds a slot for its CPU work only, never for the file
IO or the LanceDB commit around it.
"""

import bisect
from collections.abc import Sequence
from functools import partial

import anyio
import anyio.to_thread
import msgspec

from haskie import cpu, home
from haskie.catalogue.catalogue import EmbeddingModel
from haskie.collection import vocabulary as collection_vocabulary
from haskie.collection.collection import Collection
from haskie.collection.index import Row
from haskie.document import convert, ocr
from haskie.document.bookmarks import Bookmark
from haskie.document.document import Document
from haskie.errors import PermanentError, Unavailable
from haskie.indexing import (
    chunk,
    embed,
    embed_cache,
    gguf_models,
    hardware,
    models,
    parts,
    segment,
)
from haskie.indexing.segment import CutReason, SpanKind
from haskie.sections import build, generated, vocabulary
from haskie.sections.descriptors import Description
from haskie.settings import (
    Accelerator,
    ChunkSettings,
    Descriptors,
    PipelineSettings,
    load_user_settings_or_none,
)

JOINER = "\n\n"  # between convert parts in the assembled markdown
# About one printed page of markdown: what a page is where the text has no page markers to count
# (`plan_embed`). Judgement: one 657-page book held about 2,100 characters a page.
PAGE_CHARS = 3_000


class Batch(msgspec.Struct):
    seq: int
    start: int  # convert: first page, 0-based; embed/index: part number; describe: first section
    end: int  # convert: exclusive page bound; embed/index: part number + 1; describe: section bound
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
    """Write one part file; returns how many of its pages are left unread: no text, and none OCR
    read. `parser` and `skip_ocr_pages` are the document's own, fixed at import; OCR follows the
    setting when the batch runs (`_ocr`).

    While the OCR model is on its way, a PDF or image batch raises `ModelLoading` before any
    work, so the workflow waits for the model without a slot."""
    source = doc.source_path()
    suffix = source.suffix.lower()
    ocr = suffix in convert.OCR_SUFFIXES and await _ocr()
    if suffix == ".pdf":
        markdown, unread, _ = await cpu.off_interpreter(
            convert.pdf_pages_markdown,
            source,
            list(range(batch.start, batch.end)),
            doc.skip_ocr_pages,
            batch.bookmarks,
            ocr,
        )
    else:
        markdown = await cpu.on_cpu(convert.to_markdown, source, doc.parser, ocr)
        unread = []
    await home.atomic_write(doc.part_path(batch.seq), markdown)
    return len(unread)


async def _ocr() -> bool:
    """Whether OCR reads a conversion's scans; raises `ModelLoading` while its model is on its
    way. Off, a home nobody has set up yet (it downloads no model, `workflows.start`), or a model
    whose download failed reads none, and those pages fall to `skip_ocr_pages`."""
    settings = await load_user_settings_or_none()
    if settings is None or not settings.pipeline.ocr:
        return False
    try:
        await models.require_ready(models.ModelKind.OCR, ocr.MODEL)
    except Unavailable:
        return False
    return True


async def finalize_convert(doc: Document, batches: list[Batch], ocr_total: int) -> int | None:
    """Apply the OCR policy over the whole document, then stream parts into one markdown file.
    Returns a PDF's page count, None for other formats."""
    total = None
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
    return total


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
    doc: Document, params: embed_cache.Params, dims: int | None, count: int
) -> None:
    """Publish one computed embedding of `count` parts, its sections named, under
    `embed_cache.key(params)`; `describe` writes their descriptors.

    A part whose rows were never written raises `FileNotFoundError` out of the merge, which
    leaves no partial cache file behind (see `embed_cache._merge`)."""
    cache_id = embed_cache.key(params)
    paths = [embed_cache.rows_path(doc.id, cache_id, part) for part in range(count)]
    await embed_cache.write(params, paths, dims)


# --- describe -------------------------------------------------------------------


# sections one batch of the llm strategy asks about: about 8 s at 0.5 s a section, so the
# Operations view moves often, and a crash repeats little
DESCRIBE_SECTIONS = 16


async def plan_describe(doc: Document, cache_id: str, by: Descriptors) -> list[Batch]:
    """The describe batches of one cached embedding, by section index. The llm strategy reads one
    section a prompt, so its sections batch up freely. c-TF-IDF weighs every section against the
    whole document, so it is one batch."""
    count = len(await embed_cache.read_sections(doc.id, cache_id))
    size = DESCRIBE_SECTIONS if by == Descriptors.LLM else max(1, count)
    return [
        Batch(seq=seq, start=start, end=min(start + size, count))
        for seq, start in enumerate(range(0, count, size))
    ]


async def describe_ready(
    embedding: EmbeddingModel | None, by: Descriptors, settings: PipelineSettings
) -> None:
    """Fail fast when the model strategy `by` describes with is not loaded, as `embed_batch`
    does: the describer for llm, the embedding model c-TF-IDF reranks its candidates with. A
    describer the hardware setting leaves nowhere to run is permanent: it would never load."""
    if by == Descriptors.LLM:
        describer = gguf_models.generator(settings).name
        if hardware.device(describer, settings.accelerator) is None:
            # the settings changed under a run asked for llm: its model never warms here
            raise PermanentError(hardware.nowhere(describer))
        await models.require_ready(models.ModelKind.DESCRIBER, describer)
    elif embedding is not None:
        await models.require_ready(models.ModelKind.EMBEDDING, embedding.name)


def describe_order(sections: Sequence[build.Section]) -> list[int]:
    """The positions of `sections` in the order the llm strategy describes them: deepest first,
    in document order within a depth, so a section's subsections are described before it reads
    their outline (`generated.Generated`). A describe batch is a run of this order."""
    return sorted(range(len(sections)), key=lambda at: (-sections[at].depth, at))


# A describe batch's scratch file: each section it described, by id. A batch an older build wrote
# is a list in document order instead, of descriptors alone before section descriptions.
type DescribedBatch = dict[str, Description] | list[Description | list[str]]


async def _batches(doc: Document, cache_id: str, count: int) -> list[DescribedBatch]:
    """The scratch files of the first `count` describe batches, in order."""
    return [
        msgspec.json.decode(
            await anyio.Path(embed_cache.descriptors_path(doc.id, cache_id, seq)).read_bytes(),
            type=DescribedBatch,
        )
        for seq in range(count)
    ]


async def _described(doc: Document, cache_id: str, batches: int) -> dict[str, Description]:
    """What the first `batches` describe batches wrote, by section id; an older build's batch
    names no ids, so it counts for nothing here."""
    found: dict[str, Description] = {}
    for batch in await _batches(doc, cache_id, batches):
        if isinstance(batch, dict):
            found.update(batch)
    return found


async def describe_batch(
    doc: Document,
    cache_id: str,
    embedding: EmbeddingModel | None,
    settings: PipelineSettings,
    batch: Batch,
) -> int:
    """Write descriptors for sections [start, end) by the settings' strategy into a scratch file
    of the batch; returns how many sections it described. The llm strategy counts them in
    `describe_order` and reads what the batches before wrote, for the outline of a section above
    them; c-TF-IDF in document order."""
    by = settings.descriptors
    await describe_ready(embedding, by, settings)
    span = slice(batch.start, batch.end)
    if by == Descriptors.LLM:
        found = await embed_cache.inputs(doc.id, cache_id, vectors=False)
        picked = [found.sections[at] for at in describe_order(found.sections)[span]]
        earlier = await _described(doc, cache_id, batch.seq)
        known = [
            (build.run(one), earlier[one.id].descriptors)
            for one in found.sections
            if one.id in earlier
        ]
        strategy = generated.Generated(_describer_reply(settings), known)
        # a worker thread without a CPU slot: the describer runs on the GPU, one prompt at a time,
        # and documents waiting their turn there must not hold the slots searches need
        described = await anyio.to_thread.run_sync(
            build.describe, picked, found.prose, None, None, strategy
        )
    else:
        embedder = None if embedding is None else partial(embed.embed_texts, embedding)
        found = await embed_cache.inputs(doc.id, cache_id, vectors=embedder is not None)
        vectors = None if found.vectors is None else found.vectors[span]
        described = await cpu.on_cpu(
            build.describe, found.sections[span], found.prose, vectors, embedder
        )
    path = embed_cache.descriptors_path(doc.id, cache_id, batch.seq)
    await anyio.Path(path.parent).mkdir(parents=True, exist_ok=True)
    about = {one.id: Description(one.descriptors, one.description) for one in described}
    await home.atomic_write(path, msgspec.json.encode(about))
    return len(described)


async def describe_sections_batch(
    doc: Document, cache_id: str, settings: PipelineSettings, batch: Batch
) -> int:
    """Write prose descriptions independently of section descriptors."""
    await describe_ready(None, Descriptors.LLM, settings)
    found = await embed_cache.inputs(doc.id, cache_id, vectors=False)
    reply = _describer_reply(settings)

    def describe() -> list[str]:
        return [
            generated.describe_section(build.run(one), found.prose, reply)
            for one in found.sections[batch.start : batch.end]
        ]

    descriptions = await anyio.to_thread.run_sync(describe)
    path = embed_cache.descriptions_path(doc.id, cache_id, batch.seq)
    await anyio.Path(path.parent).mkdir(parents=True, exist_ok=True)
    await home.atomic_write(path, msgspec.json.encode(descriptions))
    return len(descriptions)


async def finalize_section_descriptions(doc: Document, cache_id: str, count: int) -> int:
    """Publish section descriptions before extracting their descriptors."""
    found = await embed_cache.read_sections(doc.id, cache_id)
    descriptions: list[str] = []
    for seq in range(count):
        path = anyio.Path(embed_cache.descriptions_path(doc.id, cache_id, seq))
        descriptions.extend(msgspec.json.decode(await path.read_bytes(), type=list[str]))
    described = [
        msgspec.structs.replace(one, description=text)
        for one, text in zip(found, descriptions, strict=True)
    ]
    await embed_cache.write_descriptors(doc.id, cache_id, described, None)
    return len(described)


async def finalize_describe(
    doc: Document, cache_id: str, by: Descriptors, count: int, describer: str | None = None
) -> int:
    """Put the descriptions and descriptors from `count` batches on the sections of one cached
    embedding, written by `by` (and for llm by the model `describer`), then drop the scratch
    files; returns how many sections it described. The drop comes last, so a retry still finds
    its input."""
    found = await embed_cache.read_sections(doc.id, cache_id)
    by_id: dict[str, Description] = {}
    # a workflow resumed after an upgrade may hold batches an older build wrote, in document
    # order: they came before the ones this build wrote, so they are the first sections
    in_order: list[Description] = []
    legacy = False
    for batch in await _batches(doc, cache_id, count):
        if isinstance(batch, dict):
            by_id.update(batch)
            continue
        legacy |= any(isinstance(one, list) for one in batch)
        in_order.extend(Description(one) if isinstance(one, list) else one for one in batch)
    if len(by_id) + len(in_order) != len(found):
        raise ValueError(f"{len(by_id) + len(in_order)} sections described of {len(found)}")
    descriptions = [
        by_id[one.id] if one.id in by_id else in_order[at] for at, one in enumerate(found)
    ]
    described = [
        msgspec.structs.replace(
            one, descriptors=each.descriptors, description=each.description or one.description
        )
        for one, each in zip(found, descriptions, strict=True)
    ]
    # Old llm batches have no descriptions: publish them, but let the next index fill them.
    completed_by = None if legacy and by == Descriptors.LLM else by
    await embed_cache.write_descriptors(doc.id, cache_id, described, completed_by, describer)
    await home.remove_tree(embed_cache.scratch_dir(doc.id, cache_id))
    return len(described)


async def summarize(doc: Document, cache_id: str, settings: PipelineSettings) -> str:
    """What the document is about, in a few sentences the describer writes from the sections of
    one cached embedding, once the llm strategy described them (`generated.summarize`)."""
    await describe_ready(None, Descriptors.LLM, settings)
    sections = await embed_cache.read_sections(doc.id, cache_id)
    # off the CPU budget, as `describe_batch` asks the describer
    return await anyio.to_thread.run_sync(generated.summarize, sections, _describer_reply(settings))


async def summarize_collection(descriptions: list[str], settings: PipelineSettings) -> str:
    """What a collection is about, in a few sentences the describer writes from its documents'
    descriptions (`generated.summarize_collection`)."""
    await describe_ready(None, Descriptors.LLM, settings)
    return await anyio.to_thread.run_sync(
        generated.summarize_collection, descriptions, _describer_reply(settings)
    )


def _describer_reply(settings: PipelineSettings) -> generated.Reply:
    """The llm strategy's model, the one the settings pick, asked one prompt at a time."""
    return partial(embed.reply, gguf_models.generator(settings).name, settings.accelerator)


# --- vocabulary -----------------------------------------------------------------

VOCABULARY_EMBED = 1024  # variants one step embeds: about 5 s at 5 ms a variant (M4 Pro)
VOCABULARY_JUDGE = 32  # pairs one step asks about: 6 to 14 s at two 100 to 215 ms prompts a pair


def vocabulary_embedder(accelerator: Accelerator) -> EmbeddingModel:
    """The model a collection's vocabulary embeds its descriptors with, its instruction read
    ahead of every one (`sections.vocabulary.INSTRUCTION`)."""
    return EmbeddingModel(
        gguf_models.VOCABULARY_EMBEDDER,
        dims=1024,
        accelerator=accelerator,
        document_prefix=vocabulary.INSTRUCTION,
    )


async def vocabulary_ready(settings: PipelineSettings) -> None:
    """Fail fast when a model the vocabulary needs is not loaded, as `describe_ready` does: the
    embedder and the describer, which judges. One the hardware setting leaves nowhere to run is
    permanent."""
    for kind, name in (
        (models.ModelKind.VOCABULARY, gguf_models.VOCABULARY_EMBEDDER),
        (models.ModelKind.DESCRIBER, gguf_models.generator(settings).name),
    ):
        if hardware.device(name, settings.accelerator) is None:
            raise PermanentError(hardware.nowhere(name))
        await models.require_ready(kind, name)


async def embed_vocabulary(collection: str, settings: PipelineSettings) -> int:
    """Embed the next `VOCABULARY_EMBED` variants the collection's vocabulary lacks; how many."""
    await vocabulary_ready(settings)
    model = vocabulary_embedder(settings.accelerator)
    embedder = partial(embed.embed_texts, model)
    return await collection_vocabulary.embed_missing(
        collection, model.cache_name, embedder, VOCABULARY_EMBED
    )


async def plan_vocabulary(collection: str, settings: PipelineSettings) -> int:
    """Queue the pairs of the collection's vocabulary the describer has not judged; how many."""
    model = vocabulary_embedder(settings.accelerator)
    judge = gguf_models.generator(settings).name
    return await collection_vocabulary.plan(collection, model.cache_name, judge)


async def judge_vocabulary(collection: str, settings: PipelineSettings) -> int:
    """Ask the describer about the next `VOCABULARY_JUDGE` pairs; how many."""
    await vocabulary_ready(settings)
    describer = gguf_models.generator(settings).name
    judge = partial(embed.yes, describer, settings.accelerator)
    return await collection_vocabulary.judge_missing(collection, judge, describer, VOCABULARY_JUDGE)


async def build_vocabulary(collection: str, settings: PipelineSettings) -> int:
    """Cluster the collection's variants into preferred terms, by the describer's bar for one
    concept (`gguf_models.Generator.same_concept`); how many terms."""
    model = vocabulary_embedder(settings.accelerator)
    judge = gguf_models.generator(settings)
    return await collection_vocabulary.build(
        collection, model.cache_name, judge.name, judge.same_concept
    )


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
    older build or another embedding left behind, and drop the rows the document already has there
    (a previous attach, possibly under other chunk settings). The membership's cache entry is
    cleared first, so it never names rows that are gone (`Collection.searchable`); `index_batch`
    names the new one once its rows are in."""
    index = collection.index_with(embedding)
    await collection.set_member_entry(doc.id, None)
    await index.reset_for_write()
    await index.delete_document(doc.id)


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

    The document's older rows are gone before the first batch runs (see `prepare_index`). Once
    the batch's rows are in, the membership names the cache entry they come from, so a reader of
    their sections finds the ones their ids name; a repeat names the same one."""
    index = collection.index_with(embedding)
    await index.delete_parts(doc.id, batch.start, batch.end)
    written = await index.add_parts(
        doc.id,
        doc.relative(doc.source_path()),
        doc.relative(doc.markdown),
        embed_cache.read(doc.id, cache_id, batch.start, batch.end),
    )
    await collection.set_member_entry(doc.id, cache_id)
    return written


async def finalize_index(collection: Collection, embedding: EmbeddingModel | None) -> None:
    """Build the collection's full-text index if it has none. Rows written after the build are
    found by a scan until `maintenance` folds them in, so this is O(collection) once, not per
    document."""
    await collection.index_with(embedding).finish()
