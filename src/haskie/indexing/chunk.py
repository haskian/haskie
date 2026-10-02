"""Structure-Aware Chunking: markdown cut into chunks of whole sentences, with its blocks whole.

This file says what chunking does and in what order. `segment.py` holds the pure folds the steps
hand their work to, and `docs/chunking.md` draws why a chunk starts and ends where it does. Every
step is one function of the same shape. `pipeline` composes them from the settings, so a step can be
added, dropped or moved there alone:

    markdown   blocks     -> sentences -> sections -> [frames] -> pack -> locate
    text       paragraphs -> sentences -> sections -> pack -> locate

Every heading after content starts a new section (`sections`). Within a section, every paragraph
is a chunk of its own: `pack` merges short ones with their neighbours and cuts long ones into
whole sentences. A heading over text is never part of a chunk's text. It is citation metadata
(`Chunk.headings`), worked out whatever the settings say. A section of headings alone makes no
chunk: its headings go on to the next chunk's path. A text whose every word sits in a heading is
chunked as text instead (`split`): a converter can read a whole page as headings, a PDF set all in
one font, and dropping them all would leave the document unsearchable.

`frames` is the one optional step (`chunk_frame`). It frames each section with its
heading path, and the models read every chunk with that path in front (`framed`). The chunk size
counts the path, so a section's chunks pack into what the path leaves. A path longer than half a
chunk loses its outermost headings first (`_shortened`). Without the step, a chunk's frame is
empty and its text gets the whole size. The `text` chunker cuts no sections at headings and
frames nothing: every heading line stays in its text. It still files each chunk under its heading
path for citing (`_heading_paths`).

This is not a `pydantic_graph` like `search/flow.py`. A search runs on the event loop, while
chunking is CPU work in a worker thread (`pipeline.embed_batch`). There a graph would need an
event loop of its own for every part. A fold over the steps keeps the same shape without one.

`CHUNK_VERSION` is part of the embedding cache key (`embed_cache.Params`). A cached set of chunks
is reusable only while this module cuts the same text the same way. Bump it with any change that
alters the output for unchanged input and settings. Otherwise the cache keeps serving chunks the
current code would no longer produce.
"""

import re
from bisect import bisect_right
from collections.abc import Callable, Sequence
from itertools import accumulate
from typing import Any

import msgspec

from haskie.document.convert import PAGE_MARKER, without_markers
from haskie.indexing import segment
from haskie.indexing.segment import CutReason, Packed, PieceType, Span, SpanKind
from haskie.settings import Chunker, ChunkSettings

# see the module docstring; 3: e5's query and passage prefixes; 4: a part boundary is `part` or
# `heading`, not `edge`; 5: parts cut where sections start (`pipeline.plan_embed`)
CHUNK_VERSION = 5
HEADING_SEP = " > "  # between two headings of a heading path: "Part I > Chapter 2 > Retries"
WORD = re.compile(r"\w")  # what a piece needs one of to say anything
type Opened = tuple[int, str]  # a heading still open: its level, 1 to 6, and its text


class Piece(msgspec.Struct):
    """One of the pieces a chunk was packed from: a sentence, or a block kept whole."""

    type: PieceType  # the markdown it came from (`segment.PieceType`)
    text: str


class Position(msgspec.Struct):
    """Where one piece of a chunk starts in its text, in characters: what the index keeps of the
    pieces, since it keeps their text joined."""

    type: PieceType
    position: int


class Chunk(msgspec.Struct):
    # The headings the chunk sits under, outermost first and its own heading last; empty for text
    # before the first heading.
    headings: list[str]
    # The headings the models read ahead of the text (`framed`): `headings`, less its outermost
    # ones when the whole path would take over half a chunk; none without the `frames` step.
    frame: list[str]
    # The chunk's text as the pieces it was packed from: its sentences, and whole blocks where
    # the markdown has no sentences (a code block, a table), never a heading (see `headings`).
    # Joined, they are `text`, whitespace included; the last one ends where the chunk does. Page
    # markers are taken out (`convert.without_markers`): they are pages, not text, so `text` can
    # be shorter than the span the offsets below give.
    pieces: list[Piece]
    line_start: int  # 1-based, inclusive
    line_end: int  # 1-based, inclusive
    char_start: int  # 0-based offsets into the full markdown, page markers included
    char_end: int
    # The same span in bytes, which is what a file can be seeked to: a search reads the
    # bytes of a chunk rather than the whole document (see `search.retrieval`).
    byte_start: int
    byte_end: int
    page_start: int | None = None  # 1-based PDF pages, from page markers; None for non-PDF
    page_end: int | None = None
    start_reason: CutReason = CutReason.EDGE  # why the chunk starts and ends where it does
    end_reason: CutReason = CutReason.EDGE

    @property
    def text(self) -> str:
        return "".join(piece.text for piece in self.pieces)

    @property
    def header(self) -> str:
        return HEADING_SEP.join(self.headings)

    @property
    def layout(self) -> list[Position]:
        """Where each of `pieces` starts in `text`, with its type: 0 first."""
        starts = accumulate((len(piece.text) for piece in self.pieces[:-1]), initial=0)
        return [Position(p.type, at) for p, at in zip(self.pieces, starts, strict=True)]


def frame(path: Sequence[str]) -> str:
    """What the embedding model and the reranker read ahead of a chunk's text: its frame
    (`Chunk.frame`) joined, then a blank line. A paragraph saying "they rose by 15%" means little
    without the section it sits in, and no chunk's text holds its heading."""
    return f"{HEADING_SEP.join(path)}\n\n" if path else ""


def framed(path: Sequence[str], text: str) -> str:
    """A chunk exactly as the models read it: its frame, then its text."""
    return frame(path) + text


def record(
    chunk: Chunk, vector: list[float] | None, dims: int | None, **columns: Any
) -> dict[str, Any]:
    """One Arrow record for a chunk: its own fields, plus the columns the table adds around it.

    Both tables that hold chunks build their rows here (the parquet cache, `embed_cache`, and a
    collection's LanceDB table, `index`), so both see the same fields. Each table's schema
    (`embed_cache._PLAIN`, `index.PLAIN_SCHEMA`) drops a key it does not name, so a new field on
    `Chunk` must be added to both schemas to be stored. The record carries the pieces and the text
    joined from them, and each table's schema takes the one it stores: the cache keeps the pieces,
    the index the text its full-text search reads and the `layout` of the pieces in it. `dims` is
    the width of the table's vector column, or None for a table without one; pyarrow would write a
    null for a missing vector, so a row without one is refused here instead.
    """
    joined = {"text": chunk.text, "layout": msgspec.to_builtins(chunk.layout)}
    values = msgspec.to_builtins(chunk) | joined | columns
    if dims is not None:
        if vector is None:
            raise ValueError("the table has a vector column but the row carries no vector")
        values["vector"] = vector
    return values


class Chunking(msgspec.Struct, frozen=True):
    """One run of the pipeline, as every step of it sees it. The values that flow between the
    steps are their inputs and outputs; this is what they all read."""

    text: str
    settings: ChunkSettings
    # lines / chars / bytes before `text` in the whole document: it is chunked one part at a time
    line_offset: int
    char_offset: int
    byte_offset: int
    # the headings still open where `text` starts, opened in an earlier part
    opened: tuple[Opened, ...] = ()
    # why the first chunk starts and the last one ends: the document's edge, or a part boundary
    start_reason: CutReason = CutReason.EDGE
    end_reason: CutReason = CutReason.EDGE
    # the page open where `text` starts, by an earlier part's marker; None before any
    page: int | None = None


# --- the steps --------------------------------------------------------------------


def blocks(run: Chunking, _: None) -> list[Span]:
    """The markdown's leaf blocks: paragraphs and list items as prose, the rest kept whole."""
    return segment.blocks(run.text)


def paragraphs(run: Chunking, _: None) -> list[Span]:
    """Every run of non-blank lines as prose, whatever markdown it holds."""
    return segment.paragraphs(run.text)


def sentences(run: Chunking, found: list[Span]) -> list[Span]:
    """The prose cut into sentences (Unicode UAX #29), the whole blocks left as they are."""
    return segment.sentences(run.text, found)


class Section(msgspec.Struct, frozen=True):
    """One section as the steps between `sections` and `pack` see it."""

    pieces: list[Span]
    # the headings every chunk of the section is read under; empty without `frames`
    frame: list[str] = []


def sections(run: Chunking, pieces: list[Span]) -> list[Section]:
    """The pieces grouped by section: every heading after content starts a new one."""
    return [Section(found) for found in segment.sections(pieces)]


def frames(run: Chunking, found: list[Section]) -> list[Section]:
    """Each section framed with the heading path it opens, on top of the headings `run.opened`
    before the text, shortened to at most half a chunk. A section of headings alone makes no
    chunk (`pack`), but its headings still open the path of the sections after it."""
    stack = list(run.opened)
    framed_sections: list[Section] = []
    for section in found:
        for piece in section.pieces:
            if piece.kind == SpanKind.HEADING:
                _open(stack, (piece.level, piece.title))
        # after opening: a sibling chapter closes the last
        path = _shortened([title for _, title in stack], run.settings.chunk_size)
        framed_sections.append(msgspec.structs.replace(section, frame=path))
    return framed_sections


def pack(run: Chunking, found: list[Section]) -> list[Packed]:
    """Each section's pieces packed into chunks along its paragraphs (`segment.pack`). Every chunk
    of a section shares its frame, so each gets the chunk size less that frame. `segment.fit`
    first cuts any piece longer than that. Every section but the last ends at the next one's
    heading; the last ends at the edge of the document, or where the next part begins.

    A section of headings alone packs into no chunk, so its headings ride on the next chunk
    instead, where `locate` reads the heading paths from: `# Part II` over an empty `## Ch 5`
    still heads `## Ch 6` after it. A chunk without a word says nothing (`---`, a stray symbol, a
    page marker alone) and makes no chunk either (`_worded`), and a section of such chunks alone
    is one of headings alone."""
    size = run.settings.chunk_size
    short = size * run.settings.chunk_merge_below / 100  # of the whole size, whatever the frame
    ends: list[CutReason] = [CutReason.HEADING] * (len(found) - 1) + [run.end_reason]
    packed: list[Packed] = []
    carried: list[Span] = []  # the headings of sections that made no chunk
    for section, end in zip(found, ends, strict=False):
        budget = size - len(frame(section.frame))
        pieces = segment.fit(run.text, section.pieces, budget)
        cut = _worded(run.text, segment.pack(pieces, budget, short, end))
        if not cut:
            carried += [piece for piece in pieces if piece.kind == SpanKind.HEADING]
            continue
        cut[0] = msgspec.structs.replace(cut[0], headings=[*carried, *cut[0].headings])
        carried = []
        packed += [msgspec.structs.replace(chunk, frame=section.frame) for chunk in cut]
    return packed


def _worded(text: str, cut: list[Packed]) -> list[Packed]:
    """The chunks of one section that hold a word, the others dropped: a separator or a stray
    symbol packed alone says nothing, and would only be matched by its heading path. A dropped
    chunk hands its headings to the next chunk kept, and its end to the one kept before it, so a
    chunk's end still says why the next one starts where it does."""
    kept: list[Packed] = []
    headings: list[Span] = []
    for chunk in cut:
        if not WORD.search(without_markers(text[chunk.pieces[0].start : chunk.pieces[-1].end])):
            headings += chunk.headings
            if kept:
                kept[-1] = msgspec.structs.replace(kept[-1], end_reason=chunk.end_reason)
            continue
        if headings:
            chunk = msgspec.structs.replace(chunk, headings=[*headings, *chunk.headings])
            headings = []
        kept.append(chunk)
    return kept


# --- the pipelines ----------------------------------------------------------------

type Step = Callable[[Chunking, Any], Any]  # each step's output is the next one's input


def pipeline(settings: ChunkSettings) -> tuple[Step, ...]:
    """The steps one run takes, in order: the chunker's own first step, then the optional one
    the settings turn on. `frames` needs sections cut at headings, which only the markdown
    chunker makes."""
    read: Step = blocks if settings.chunker == Chunker.MARKDOWN else paragraphs
    framing: tuple[Step, ...] = ()
    if settings.chunk_frame and settings.chunker == Chunker.MARKDOWN:
        framing = (frames,)
    return (read, sentences, sections, *framing, pack, locate)


def split(
    text: str,
    settings: ChunkSettings,
    line_offset: int = 0,
    char_offset: int = 0,
    byte_offset: int = 0,
    opened: Sequence[Opened] = (),
    start_reason: CutReason = CutReason.EDGE,
    end_reason: CutReason = CutReason.EDGE,
    page: int | None = None,
) -> list[Chunk]:
    """`text` as chunks. The pipeline chunks a document one part at a time, so `line_offset`,
    `char_offset` and `byte_offset` count what comes before `text` in the whole document,
    `opened` holds the headings still open where it starts (see `open_headings`), and
    `start_reason` and `end_reason` say why its ends are cut: the document's own (`EDGE`), else
    where another part meets it, at a heading the later part opens with (`HEADING`) or partway
    through a section (`PART`). `page` is the page open where it starts, until its first page
    marker."""
    run = Chunking(
        text,
        settings,
        line_offset,
        char_offset,
        byte_offset,
        tuple(opened),
        start_reason,
        end_reason,
        page,
    )
    value: Any = None
    for step in pipeline(settings):
        value = step(run, value)
    if not value and settings.chunker == Chunker.MARKDOWN and WORD.search(without_markers(text)):
        as_text = msgspec.structs.replace(settings, chunker=Chunker.TEXT)
        return split(
            text, as_text, line_offset, char_offset, byte_offset, opened, start_reason, end_reason
        )
    return value


# --- locate -----------------------------------------------------------------------


def open_headings(text: str, opened: Sequence[Opened] = ()) -> list[Opened]:
    """The headings still open at the end of `text`, outermost first, given those open where it
    starts. What the next part of a document is chunked under: a chapter opened on page 9 still
    frames the chunks of page 11, in the next part."""
    stack = list(opened)
    for block in segment.blocks(text):
        if block.kind == SpanKind.HEADING:
            _open(stack, (block.level, block.title))
    return stack


def opens_with_heading(text: str) -> bool:
    """Whether `text` opens with a heading, before a word of its own: then the part before it
    ends at a heading (`CutReason.HEADING`), not partway through a section (`PART`)."""
    for block in segment.blocks(text):
        if block.kind == SpanKind.HEADING:
            return True
        if WORD.search(without_markers(text[block.start : block.end])):
            return False
    return False


def _shortened(path: list[str], size: int) -> list[str]:
    """The heading path a chunk of `size` is framed with: the whole path, less its outermost steps
    while it would take over half a chunk, and a last heading still too long cut to fit. The
    innermost heading says most about the text under it; the rest of a chunk is left for that
    text."""
    half = size // 2
    path = list(path)
    while len(path) > 1 and len(frame(path)) > half:
        path.pop(0)
    if path and len(frame(path)) > half:
        cut = path[0][: max(0, half - len(frame([""])))].rstrip()
        path = [cut] if cut else []
    return path


def _open(stack: list[Opened], heading: Opened) -> None:
    """A heading closes every open one at its level or deeper, then opens under the rest."""
    while stack and stack[-1][0] >= heading[0]:
        stack.pop()
    stack.append(heading)


def _heading_paths(run: Chunking, pieces: list[Span]) -> tuple[list[int], list[list[str]]]:
    """Char offset of every heading plus the heading path it opens, outermost first, on top of
    the headings `run.opened` before the text. Read off the heading pieces the markdown pipeline
    already cut (`Packed.headings`); the `text` pipeline cuts none, so its paths come from one
    parse of its own."""
    if run.settings.chunker == Chunker.TEXT:
        pieces = segment.blocks(run.text)
    offsets: list[int] = []
    paths: list[list[str]] = []
    stack = list(run.opened)
    for piece in pieces:
        if piece.kind == SpanKind.HEADING and not piece.continued:  # a cut heading is still one
            _open(stack, (piece.level, piece.title))
            offsets.append(piece.start)
            paths.append([title for _, title in stack])
    return offsets, paths


def locate(run: Chunking, packed: list[Packed]) -> list[Chunk]:
    """The chunks with their offsets, lines, pages and heading paths."""
    text = run.text
    offsets, paths = _heading_paths(run, [h for chunk in packed for h in chunk.headings])
    before_any = [title for _, title in run.opened]  # ahead of this part's first heading
    newlines = [m.start() for m in re.finditer("\n", text)]
    markers = [(m.start(), int(m.group(1))) for m in PAGE_MARKER.finditer(text)]
    marker_offsets = [m[0] for m in markers]

    def line_at(pos: int) -> int:
        return run.line_offset + bisect_right(newlines, pos - 1) + 1

    def page_at(pos: int) -> int | None:
        idx = bisect_right(marker_offsets, pos) - 1
        return markers[idx][1] if idx >= 0 else run.page

    # Byte offsets are walked, not looked up: chunks tile the text in order, so the cursor
    # encodes every character once, the gap before a chunk and then the chunk itself.
    char_at = byte_at = 0
    chunks: list[Chunk] = []
    # each chunk starts at the cut the one before it ends at
    start_reason = run.start_reason
    for chunk in packed:
        pieces = chunk.pieces
        last = pieces[-1]
        ends = [piece.end for piece in pieces[:-1]] + [last.visible_end]  # ends where its text does
        texts = [without_markers(text[p.start : e]) for p, e in zip(pieces, ends, strict=True)]
        start, end = pieces[0].start, last.visible_end
        byte_at += len(text[char_at:start].encode())
        byte_start = run.byte_offset + byte_at
        body_bytes = len(text[start:end].encode())  # the source's, page markers and all
        char_at, byte_at = end, byte_at + body_bytes
        idx = bisect_right(offsets, start) - 1  # filed under the heading its first piece is under
        headings = paths[idx] if idx >= 0 else before_any
        chunks.append(
            Chunk(
                headings=headings,
                frame=chunk.frame,
                pieces=[Piece(p.type, t) for p, t in zip(pieces, texts, strict=True)],
                line_start=line_at(start),
                line_end=line_at(end),
                char_start=run.char_offset + start,
                char_end=run.char_offset + end,
                byte_start=byte_start,
                byte_end=byte_start + body_bytes,
                page_start=page_at(start),
                page_end=page_at(end - 1),
                start_reason=start_reason,
                end_reason=chunk.end_reason,
            )
        )
        start_reason = chunk.end_reason
    return chunks
