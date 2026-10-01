"""The pure folds under Structure-Aware Chunking (`chunk.py`): markdown into blocks, blocks into
sentences, sentences into chunks.

Every offset is a char offset into the text being chunked. A `Span` is one stretch of it, and the
pieces `sentences` returns tile the text: each starts where the one before it ended, so a chunk
packed from whole pieces is one slice of the text, and its pieces joined are exactly that slice.

What the markdown means comes from the parser (pyromark, the parse the viewer renders), never
from a pattern of ours: which lines are a heading, a table or a list item, and which characters
of a paragraph are words rather than markup. The one pattern is `convert.PAGE_MARKER`, the page
marker the converter writes and every reader of it shares. A page marker is page metadata, not
content: every fold reads it as whitespace, so no piece starts or ends on one, and
`without_markers` takes it out of the text a chunk carries. Its offset is what gives a chunk its
pages (`chunk.locate`).
"""

import re
import unicodedata
from enum import StrEnum
from functools import cache
from itertools import groupby
from typing import cast

import msgspec
import pyromark
import unicode_segmentation_rs
from pyromark.event import Event, Range
from semantic_text_splitter import TextSplitter

from haskie.document import render
from haskie.document.convert import PAGE_MARKER, without_markers


# The role a span plays in the folds. prose: a paragraph or a list item's text, which `sentences`
# cuts into sentences; sentence: one of those; heading and block (code, a table, raw HTML, a
# rule): kept whole.
class SpanKind(StrEnum):
    PROSE = "prose"
    SENTENCE = "sentence"
    HEADING = "heading"
    BLOCK = "block"


# The markdown a piece came from, as a reader names it: a sentence's is the paragraph's innermost
# container (a list item or a blockquote, else plain text), every other piece its own leaf block.
# A heading is never a piece of a chunk: headings are its path (`pack`), and the type names them
# only as the chunker reads them.
class PieceType(StrEnum):
    HEADING = "heading"
    TEXT = "text"
    LIST = "list"
    QUOTE = "quote"
    TABLE = "table"
    CODE = "code"
    HTML = "html"
    RULE = "rule"
    METADATA = "metadata"


# Why a chunk starts or ends where it does, the rule that drew the line:
#   edge             the start or end of the document
#   part             where one part of a document ends and the next begins: the section may
#                    go on across it (a PDF is converted and chunked a few pages at a time)
#   heading          a heading opens the next section
#   paragraph        a blank line, and the paragraphs on either side were not merged
#   length_block     the chunk was full: cut between two blocks of one paragraph (list items, a
#                    table and the line above it)
#   length_sentence  the chunk was full: cut between two sentences of one block
#   length_oversize  one sentence, table or code block longer than a chunk: cut at a line or word
class CutReason(StrEnum):
    EDGE = "edge"
    PART = "part"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LENGTH_BLOCK = "length_block"
    LENGTH_SENTENCE = "length_sentence"
    LENGTH_OVERSIZE = "length_oversize"


class Span(msgspec.Struct, frozen=True):
    start: int
    end: int
    kind: SpanKind
    # prose only: the ranges a reader reads as words; the rest is markup, line breaks included
    words: tuple[tuple[int, int], ...] = ()
    level: int = 0  # a heading's, 1 to 6
    title: str = ""  # a heading's text, as the viewer names it (`render.headings`)
    type: PieceType = PieceType.TEXT
    block: int = 0  # which block a piece was cut from: the pieces of one paragraph share it
    # blocks only: which outermost list the block sits in, 1 and up, 0 outside every list; read
    # once, by `sentences`, to number the paragraphs
    list_id: int = 0
    paragraph: int = 0  # pieces only: which paragraph a piece is in, 0 and up (see `sentences`)
    continued: bool = False  # cut by `fit` out of the piece before it: one piece split in two
    visible_end: int = 0  # pieces only: where the text ends, the whitespace after it aside


class Packed(msgspec.Struct, frozen=True):
    """One chunk as `pack` cuts it: its pieces, and the rule of the cut after it, named by the
    step that makes the cut rather than read back off the pieces later. The first chunk of a
    section also carries the heading pieces that open the section: they are not its text, only
    what its heading path is read from."""

    pieces: list[Span]
    end_reason: CutReason
    headings: list[Span] = []
    frame: list[str] = []  # the heading path it is read under: stamped by `chunk.pack`, not here


class _Found(msgspec.Struct):
    """A block as the parse finds it, in pyromark's byte offsets, before they become chars."""

    start: int
    end: int
    kind: SpanKind
    words: list[int] = []  # flat (start, end) pairs
    level: int = 0
    list_id: int = 0
    title: str = ""
    type: PieceType = PieceType.TEXT


# The leaf blocks kept in one piece, what each becomes and the type of its pieces. A paragraph is
# one too: its sentences are cut by `sentences`, not here, and typed by where it sits (`_typed`).
LEAVES: dict[str, tuple[SpanKind, PieceType]] = {
    "Paragraph": (SpanKind.PROSE, PieceType.TEXT),
    "Heading": (SpanKind.HEADING, PieceType.HEADING),
    "CodeBlock": (SpanKind.BLOCK, PieceType.CODE),
    "Table": (SpanKind.BLOCK, PieceType.TABLE),
    "HtmlBlock": (SpanKind.BLOCK, PieceType.HTML),
    "MetadataBlock": (SpanKind.BLOCK, PieceType.METADATA),
}
CONTAINERS = {"BlockQuote", "List", "Item"}  # hold blocks; their own text is a tight list item's
WORDS = {"Text", "Code"}  # the inline events that are words; the rest is markup
# Inline HTML whose content is not part of the sentence around it: a footnote marker after a full
# stop (`data.<sup>v</sup> When`) is not the start of a sentence.
HIDDEN = re.compile(r"<(/?)(?:sup|sub)\b", re.IGNORECASE)  # the tag name whole: not `<subject>`
BLANK = re.compile(rf"(?:\s|{PAGE_MARKER.pattern})*")  # what a reader sees nothing in


def blocks(markdown: str) -> list[Span]:
    """The leaf blocks of `markdown`, in order, with the words of each prose block.

    A tight list item has no paragraph around its text, so the text directly inside an item is
    gathered into a prose block of its own, closed by the first block the item nests.
    """
    found: list[_Found] = []
    inside = 0  # depth inside a leaf block
    containers: list[str] = []
    item: _Found | None = None
    hidden = 0  # depth inside inline HTML whose content is hidden from the sentence rules
    lists = list_id = 0  # outermost lists seen so far, and the one being read (0: none)

    def close_item() -> None:
        nonlocal item
        if item is not None:
            found.append(item)
            item = None

    def read(block: _Found, name: str, value: str, span: Range) -> None:
        nonlocal hidden
        if name == "InlineHtml" and (tag := HIDDEN.match(value)):
            hidden = max(0, hidden - 1) if tag[1] else hidden + 1
        elif name in WORDS and not hidden:
            block.words += [span["start"], span["end"]]

    for event, span in pyromark.events_with_range(markdown, options=render.OPTIONS):
        tag, name, value = _names(event)
        if inside:
            inside += (tag == "Start") - (tag == "End")
            if found[-1].kind == SpanKind.PROSE:
                read(found[-1], name, value, span)
            elif found[-1].kind == SpanKind.HEADING and name in WORDS:
                found[-1].title += value  # all of it, as the viewer's table of contents reads it
        elif tag == "Start" and name in LEAVES:
            close_item()
            hidden = 0  # an unclosed `<sup>` hides no more than the rest of its own block
            kind, type_ = LEAVES[name]
            type_ = _typed(containers) if kind == SpanKind.PROSE else type_
            level = _level(event)
            found.append(_Found(span["start"], span["end"], kind, [], level, list_id, type=type_))
            inside = 1
        elif name == "Rule":
            close_item()
            found.append(
                _Found(
                    span["start"], span["end"], SpanKind.BLOCK, list_id=list_id, type=PieceType.RULE
                )
            )
        elif name in CONTAINERS:
            close_item()
            if name == "List" and tag == "Start" and "List" not in containers:
                lists += 1
                list_id = lists
            containers.append(name) if tag == "Start" else containers.pop()
            list_id = list_id if "List" in containers else 0
        elif containers and containers[-1] == "Item":
            text = item = item or _Found(
                span["start"], span["end"], SpanKind.PROSE, list_id=list_id, type=PieceType.LIST
            )
            text.end = max(text.end, span["end"])
            read(text, name, value, span)
    close_item()
    return _in_chars(markdown, found)


def paragraphs(text: str) -> list[Span]:
    """Every run of non-blank lines as prose, each line its words: the `text` chunker reads no
    markdown, so a line break is the only thing that is not a word. A page marker's line is not a
    word and does not end the run: the converter wrote it, not the author."""
    found: list[Span] = []
    lines: list[tuple[int, int]] = []

    def close() -> None:
        if lines:
            found.append(Span(lines[0][0], lines[-1][1], SpanKind.PROSE, tuple(lines)))
            lines.clear()

    at = 0
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        if not body.strip():
            close()
        elif not PAGE_MARKER.fullmatch(body.strip()):
            lines.append((at, at + len(body)))
        at += len(line)
    close()
    return found


def sentences(text: str, found: list[Span]) -> list[Span]:
    """The blocks as pieces that tile `text` from the first block on: prose cut into sentences,
    everything else whole.

    A piece starts at its first visible character and runs to where the next one starts, so it
    carries the whitespace after it, and any page marker in it. Anything else visible between two
    blocks (a list marker, a blockquote's `>`) opens the piece after it rather than trailing the
    one before.

    Every piece is numbered with its paragraph, what blank lines bound whatever the markdown
    inside it: the one boundary the author draws between two thoughts. Lines with no blank line
    between them (list items, a sentence and the table under it) are one paragraph, and a list
    is one whole, blank lines between its items or not.
    """
    starts: list[tuple[int, SpanKind, int, int]] = []  # (start, kind, heading level, block)
    at = 0
    for index, block in enumerate(found):
        start = _content_at(text, at, block.start)
        if block.kind != SpanKind.PROSE:
            starts.append((start, block.kind, block.level, index))
        else:
            cuts = _sentence_starts(text[block.start : block.end], _view(text, block))
            starts.append((start, SpanKind.SENTENCE, 0, index))
            starts.extend((block.start + cut, SpanKind.SENTENCE, 0, index) for cut in cuts)
        at = block.end
    if not starts:
        return []
    ends = [start for start, *_ in starts[1:]] + [len(text)]
    pieces: list[Span] = []
    paragraph = 0
    for (s, kind, level, index), e in zip(starts, ends, strict=True):
        if e <= s:
            continue
        if pieces and _blank_after(text, pieces[-1]):
            list_id = found[pieces[-1].block].list_id
            paragraph += not (list_id > 0 and list_id == found[index].list_id)
        visible_end = _visible_end(text, s, e)
        title = found[index].title
        pieces.append(
            Span(
                s,
                e,
                kind,
                level=level,
                title=title,
                type=found[index].type,
                block=index,
                paragraph=paragraph,
                visible_end=visible_end,
            )
        )
    return pieces


def fit(text: str, pieces: list[Span], size: int) -> list[Span]:
    """Every piece longer than `size` cut into pieces that are not: a sentence that never ends,
    a table or a code block bigger than a chunk. `TextSplitter` cuts at the largest boundary that
    fits (a line, then a word, then a character), and a cut starts past any page marker, as
    every piece does."""
    out: list[Span] = []
    for piece in pieces:
        if piece.visible_end - piece.start <= size:
            out.append(piece)
            continue
        body = text[piece.start : piece.visible_end]
        offsets = [offset for offset, _ in _splitter(size).chunk_indices(body)][1:]
        # the piece starts visible, so its first cut is its start
        cuts = [piece.start] + [_content_at(text, piece.start + at, piece.end) for at in offsets]
        ends = cuts[1:] + [piece.end]
        for i, (start, end) in enumerate(zip(cuts, ends, strict=True)):
            if end <= start:
                continue  # a cut that was only a page marker
            visible_end = _visible_end(text, start, end)
            cut = msgspec.structs.replace(
                piece, start=start, end=end, visible_end=visible_end, continued=i > 0
            )
            out.append(cut)
    return out


@cache
def _splitter(size: int) -> TextSplitter:
    return TextSplitter(size)


def sections(pieces: list[Span]) -> list[list[Span]]:
    """The pieces grouped by section: every heading that follows content starts a new one, so no
    chunk ever holds the end of one section and the start of the next.

    Headings stacked with nothing between them, each deeper than the one before (`# A`, `## B`,
    `### C`), open one section together, with the text that follows the last of them. A heading
    no deeper than the one before closes that one's section even when it is empty: two chapter
    titles in a row are two sections.
    """
    found: list[list[Span]] = []
    current: list[Span] = []
    has_content = False
    level = 0  # of the last heading in `current`
    for piece in pieces:
        opens = piece.kind == SpanKind.HEADING
        if opens and (has_content or 0 < piece.level <= level):
            found.append(current)
            current, has_content = [], False
        current.append(piece)
        has_content = has_content or piece.kind != SpanKind.HEADING
        level = piece.level if opens else level
    if current:
        found.append(current)
    return found


def pack(pieces: list[Span], size: int, short: float, end_reason: CutReason) -> list[Packed]:
    """One section's pieces packed into chunks along its paragraphs, at most `size` characters a
    chunk (trailing whitespace aside). Chunks never overlap: the heading path every chunk is
    embedded under (`chunk.framed`) is the context a neighbour's sentences would otherwise carry.

    A paragraph here is what blank lines bound, whatever the markdown inside it: a tight list, a
    line leading straight into a table, one sentence a line (`sentences` numbers them). Each is a
    chunk of its own, except a short one, under `short` characters: it goes with the
    paragraph below it when the two fit one chunk, so a lead-in is embedded with what it leads
    into, and with the short ones around it otherwise (see `_merge`). Only a paragraph longer than
    a chunk is cut: between its blocks where it can (list items, a table and the line above it),
    else between sentences (see `_fill`).

    The headings a section opens with are no part of any chunk's text: every chunk is embedded
    under the whole heading path (`chunk.framed`), so a heading in the text would be read twice.
    They ride on the section's first chunk as `Packed.headings`, for the path to be read from. A
    section of headings alone makes no chunk: a heading says where a point is, not the point, and
    in books those sections are mostly page headers, page numbers and chapter title pages the
    converter read as headings. `chunk.pack` carries its headings on to the next chunk's path.

    The section's last chunk ends for `end_reason`, the rule of the cut after the section: a
    heading, the document's edge, or where the next part begins. Every other chunk ends at a
    paragraph (between two groups `_merge` did not join) or where `_fill` found it full.
    """
    head = 0
    while head < len(pieces) and pieces[head].kind == SpanKind.HEADING:
        head += 1
    opening, body = pieces[:head], pieces[head:]
    if not body:
        return []
    paragraphs = [list(same) for _, same in groupby(body, lambda p: p.paragraph)]
    groups = _merge(paragraphs, size, short)
    chunks: list[Packed] = []
    for index, group in enumerate(groups):
        flat = [piece for paragraph in group for piece in paragraph]
        group_end = CutReason.PARAGRAPH if index < len(groups) - 1 else end_reason
        chunks.extend(
            _fill(flat, size, group_end) if _length(flat) > size else [Packed(flat, group_end)]
        )
    chunks[0] = msgspec.structs.replace(chunks[0], headings=opening)
    return chunks


type Group = list[list[Span]]  # paragraphs packed together


def _merge(paragraphs: list[list[Span]], size: int, short: float) -> list[Group]:
    """The paragraphs grouped into chunks, best effort: a run of short ones (under `short`
    characters) flows into the paragraph below it when all of them fit `size` together; a run
    that cannot (the paragraph below is too long, or the run is full) stays a chunk of its own,
    or joins the chunk above it when it fits there. Every group of more than one paragraph fits
    one chunk, and a paragraph that is not short is never merged with one that is not."""
    groups: list[Group] = []
    run: Group = []  # short paragraphs still looking for the one below them

    def fits(first: list[Span], last: list[Span]) -> bool:  # first..last as one chunk
        return last[-1].visible_end - first[0].start <= size

    def settle() -> None:
        if run and groups and fits(groups[-1][0], run[-1]):
            groups[-1] += run
        elif run:
            groups.append(run[:])
        run.clear()

    for paragraph in paragraphs:
        is_short = _length(paragraph) < short
        if run and fits(run[0], paragraph):
            run.append(paragraph)
            if not is_short:  # the paragraph the run led into closes it
                groups.append(run[:])
                run.clear()
            continue
        settle()
        if is_short:
            run.append(paragraph)
        else:
            groups.append([paragraph])
    settle()
    return groups


def _blank_after(text: str, piece: Span) -> bool:
    """Whether a blank line follows the piece: its trailing whitespace spans two line breaks, once
    any page marker in it is read as the whitespace around it."""
    return without_markers(text[piece.visible_end : piece.end]).count("\n") >= 2


def _length(pieces: list[Span]) -> int:
    """Pieces that tile one stretch of text, as a chunk's length: trailing whitespace aside."""
    return pieces[-1].visible_end - pieces[0].start


def _visible_end(text: str, start: int, end: int) -> int:
    """Where `text[start:end]` ends once the whitespace and page markers after it are set aside."""
    while True:
        end = start + len(text[start:end].rstrip())
        opens = text.rfind("<!--", start, end)
        if opens < 0 or not PAGE_MARKER.fullmatch(text, opens, end):
            return end
        end = opens


def _content_at(text: str, start: int, end: int) -> int:
    """The first character of `text[start:end]` that is neither whitespace nor in a page marker,
    or `end` when there is none."""
    found = BLANK.match(text, start, end)
    return found.end() if found else start


def _fill(pieces: list[Span], size: int, end_reason: CutReason) -> list[Packed]:
    """Whole pieces into chunks of at most `size`, the last one ending for `end_reason`. A chunk
    ends between two blocks (list items, a table and the line above it) when one lies inside what
    fits, and between sentences only when none does; inside a piece only where `fit` cut it."""

    def length(first: int, last: int) -> int:  # pieces [first, last) as a chunk's text
        return pieces[last - 1].visible_end - pieces[first].start

    chunks: list[Packed] = []
    first, count = 0, len(pieces)
    while first < count:
        last = first + 1
        while last < count and length(first, last + 1) <= size:
            last += 1
        if last < count:  # back to the last break between two blocks, if there is one
            between = last
            while between > first + 1 and pieces[between].block == pieces[between - 1].block:
                between -= 1
            if pieces[between].block != pieces[between - 1].block:
                last = between
        chunks.append(Packed(pieces[first:last], _cut(pieces, last, end_reason)))
        first = last
    return chunks


def _cut(pieces: list[Span], at: int, end_reason: CutReason) -> CutReason:
    """Why `_fill` ends a chunk before `pieces[at]`: `end_reason` past the last piece."""
    if at == len(pieces):
        return end_reason
    if pieces[at].continued:
        return CutReason.LENGTH_OVERSIZE
    if pieces[at].block != pieces[at - 1].block:
        return CutReason.LENGTH_BLOCK
    return CutReason.LENGTH_SENTENCE


def _view(text: str, block: Span) -> str:
    """The block as the sentence rules should read it, the same length as its text: its words
    where they are, every other character a space. So a line break is a space (PDF text is
    hard-wrapped, and the rules end a sentence at every newline), and `**`, `[`, `](url)` and a
    footnote marker are not read as the punctuation they look like."""
    view = [" "] * (block.end - block.start)
    for start, end in block.words:
        view[start - block.start : end - block.start] = text[start:end]
    return "".join(view)


def _sentence_starts(paragraph: str, view: str) -> list[int]:
    """Where each sentence after the first starts in `paragraph`, by UAX #29 (the Unicode sentence
    rules, no language needed) run over its `view`.

    A sentence starts at its first word, moved back over the markup that opens it (`**`, `[`) and
    over an opening bracket the rules left at the end of the sentence before (the `《` of
    `。」《`), so neither trails the previous sentence.
    """
    starts: list[int] = []
    at = 0
    for sentence in unicode_segmentation_rs.unicode_sentences(view):
        body = sentence.strip()
        if not body:
            continue
        start = view.index(body, at)
        at = start + len(body)
        floor = starts[-1] if starts else 0
        while start > floor and _opens(paragraph[start - 1], view[start - 1]):
            start -= 1
        starts.append(start)
    # the first sentence starts where its paragraph does, whatever markup was ahead of it
    return [start for start in starts[1:] if start > starts[0]]


def _opens(char: str, seen: str) -> bool:
    """Whether `char` belongs to the sentence after it: markup (blanked in the view, but not
    whitespace in the text) or an opening bracket."""
    markup = seen == " " and not char.isspace()
    return markup or unicodedata.category(char) == "Ps"


def _typed(containers: list[str]) -> PieceType:
    """A paragraph's type: the innermost list item or blockquote it sits in, else plain text."""
    for name in reversed(containers):
        if name == "Item":
            return PieceType.LIST
        if name == "BlockQuote":
            return PieceType.QUOTE
    return PieceType.TEXT


def _level(event: Event) -> int:
    """A heading's level from its Start event (`{"Start": {"Heading": {"level": "H2", ...}}}`)."""
    fields = cast(dict, event)["Start"]
    heading = fields.get("Heading") if isinstance(fields, dict) else None
    return int(heading["level"][1]) if heading else 0


def _names(event: Event) -> tuple[str, str, str]:
    """(tag, name, value) of a pyromark event: ("Start", "Heading", ""), ("Text", "Text", "the
    words"), ("InlineHtml", "InlineHtml", "<sup>"), ("Rule", "Rule", "")."""
    if isinstance(event, str):
        return event, event, ""
    ((tag, inner),) = event.items()
    if tag in ("Start", "End"):
        # a tag with fields is a one-key dict, `{"Heading": {...}}`; one without is its name
        return tag, inner if isinstance(inner, str) else next(iter(cast(dict, inner))), ""
    return tag, tag, inner if isinstance(inner, str) else ""


def _in_chars(markdown: str, found: list[_Found]) -> list[Span]:
    """The blocks in char offsets, in one incremental decode over their sorted byte offsets. An
    HTML block that is a page marker is dropped: it is page metadata, read as whitespace."""
    to_char: dict[int, int] = {}  # empty for ASCII, where a byte offset is the char offset
    if not markdown.isascii():
        offsets = sorted({o for block in found for o in (block.start, block.end, *block.words)})
        data = markdown.encode()
        byte_pos = char_pos = 0
        for offset in offsets:
            char_pos += len(data[byte_pos:offset].decode(errors="ignore"))
            byte_pos = offset
            to_char[offset] = char_pos
    spans: list[Span] = []
    for block in found:
        start, end = to_char.get(block.start, block.start), to_char.get(block.end, block.end)
        if block.kind == SpanKind.BLOCK and PAGE_MARKER.fullmatch(markdown[start:end].strip()):
            continue
        pairs = zip(block.words[::2], block.words[1::2], strict=True)
        words = tuple((to_char.get(s, s), to_char.get(e, e)) for s, e in pairs)
        title = block.title.strip()
        spans.append(
            Span(
                start, end, block.kind, words, block.level, title, block.type, list_id=block.list_id
            )
        )
    return spans
