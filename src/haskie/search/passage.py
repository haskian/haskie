"""Passages out of chunks: the pure shapes and algorithms behind the search tools.

A chunk is a retrieval unit, not a readable one. It is cut to a size the embedding model likes, it
ends where a paragraph or the size did, and a match usually lands on two or three of them at once.
What an agent wants back is one span of the document that starts and ends where a reader would stop:
a passage.

Three folds live here, all of them pure. `ranges` merges the chunks of one document that sit
next to each other (`Hit.seq`) into one range. `widen` widens one such range to the nearest newline
or sentence end of the markdown it was cut from. It is the only step that needs the text.
`top_documents` and `fold_sources` answer the other question: which documents and which
collections cover this. They fold the same hits per document instead of per range.

`widen` is given a `Window` rather than the document: the widening reaches at most `MAX_WIDEN`
characters, so a few hundred bytes around the range are enough, and `retrieval.py` reads exactly
those (the chunk rows carry the byte offsets to seek to). Line numbers come from the chunk rows
too - each one stores the line its text starts on - so nothing here counts the newlines of a
document it cannot see.

`harmonic` is the scoring rule all of them share: a range or a document scores the harmonic mean
of its best chunk and the sum of every chunk it holds. No IO: `retrieval.py` reads the markdown
and the memberships and hands them in.
"""

import re

import msgspec

from haskie.collection.index import Hit, Overlaps, Relation, location
from haskie.document.convert import without_markers

# --- scoring ---------------------------------------------------------------------


def harmonic(a: float, b: float) -> float:
    """The harmonic mean of `a` and `b`: pulled toward the smaller, so one weak value holds the
    whole down, and 0 when either is 0 or less.

    Two uses. How strongly a range or a document matches is harmonic(best chunk, sum of all its
    matched chunks): a document matched once scores its one chunk, every further chunk lifts it,
    but the mean stays under twice the best, so many weak chunks never outrank one strong one, and
    a document with a few strong chunks is not held back for having few. How strongly two results
    overlap is harmonic(contained, contains) (`collapse`): high only when each holds the other."""
    if a <= 0 or b <= 0:
        return 0.0
    return 2 * a * b / (a + b)


# --- chunk ranges ----------------------------------------------------------------


class PassageReference(msgspec.Struct):
    """Another passage that says what a passage says, folded into it rather than listed on its
    own: where else to cite the same point, not something to read again. A tree, as
    `HitReference` is."""

    collection: str
    document: str
    seq_start: int  # the chunks it covers, 1-based within the document
    seq_end: int
    header: str
    location: str
    line_start: int  # 1-based, in the document's markdown: what `/lines` reads it back by
    line_end: int
    score: float  # how well it matched the query on its own, before it was folded
    relation: Relation  # how it overlaps `to_parent`'s result
    similarity: float  # how strongly `relation` holds, as the fold decided it
    to_parent: Overlaps  # the passage or place it is listed under
    to_root: Overlaps  # the passage at the top of the tree
    also_in: list["PassageReference"] = []  # the places folded into this one, best first


class HitRange(msgspec.Struct):
    """The matched chunks of one document that sit next to each other, as one range."""

    hits: list[Hit]  # one collection and document, consecutive `seq`, ascending
    seq_start: int
    seq_end: int
    line_start: int
    line_end: int
    char_start: int  # 0-based offsets into the document's markdown
    char_end: int
    byte_start: int  # the same span in bytes, which is what the markdown file is seeked to
    byte_end: int
    page_start: int | None  # 1-based PDF pages, every member's (see `pages`); None for non-PDF
    page_end: int | None
    score: float  # harmonic(best, sum) over the members
    also_in: list[PassageReference] = []  # the near-duplicates folded in (`collapse`), a tree

    @property
    def best(self) -> Hit:
        """The hit a range is cited by: its best score, the earliest on a tie."""
        return max(self.hits, key=lambda hit: (hit.score, -hit.seq))


def pages(hits: list[Hit]) -> tuple[int | None, int | None]:
    """The pages a group of hits covers, first to last: every hit's, not only the best one's, or
    a passage over pages 1 to 5 would be cited as page 1. None for a document without pages."""
    starts = [hit.page_start for hit in hits if hit.page_start is not None]
    ends = [hit.page_end for hit in hits if hit.page_end is not None]
    return (min(starts) if starts else None, max(ends) if ends else None)


def ranges(hits: list[Hit]) -> list[HitRange]:
    """Fold hits into ranges, best range first.

    Only chunks with no gap between them merge: a gap is text the query did not match, and
    bridging it would put an unmatched paragraph inside a quoted passage. Ties sort by document
    and position, so the same hits always fold the same way.

    Grouped by (collection, document) rather than by document: two collections may chunk the same
    document with different settings, and each numbers `seq` from 1, so a run across them would
    merge ranges cut at different offsets.
    """
    # a chunk's identity should carry the settings it was cut with (its embedding cache
    # id), so grouping and deduplication can key on that instead of standing the collection in
    # for it.
    found: list[HitRange] = []
    run: list[Hit] = []
    for hit in sorted(hits, key=lambda hit: (hit.collection, hit.document, hit.seq)):
        last = run[-1] if run else None
        # the run goes on only where this hit is the next chunk of the same table's same document
        if last is not None and (hit.collection, hit.document, hit.seq) != (
            last.collection,
            last.document,
            last.seq + 1,
        ):
            found.append(_range(run))
            run = []
        run.append(hit)
    if run:
        found.append(_range(run))
    return sorted(found, key=lambda found: (-found.score, found.hits[0].document, found.seq_start))


def _range(hits: list[Hit]) -> HitRange:
    """One range out of an ascending run of hits of one document."""
    best = max(hit.score for hit in hits)
    page_start, page_end = pages(hits)
    return HitRange(
        hits=hits,
        seq_start=hits[0].seq,
        seq_end=hits[-1].seq,
        line_start=min(hit.line_start for hit in hits),
        line_end=max(hit.line_end for hit in hits),
        char_start=min(hit.char_start for hit in hits),
        char_end=max(hit.char_end for hit in hits),
        byte_start=min(hit.byte_start for hit in hits),
        byte_end=max(hit.byte_end for hit in hits),
        page_start=page_start,
        page_end=page_end,
        score=harmonic(best, sum(hit.score for hit in hits)),
    )


# --- passages --------------------------------------------------------------------


class Passage(msgspec.Struct):
    """One span of a document, widened to boundaries a reader would stop at. What a search
    answers with: `header` and `location` are what to cite it by."""

    collection: str  # the collection whose table matched; the document itself belongs to none
    document: str
    header: str  # the heading path joined, "Part I > Chapter 2", from the best chunk
    location: str  # "doc p.3-4 L10-20", rebuilt for the widened lines
    seq_start: int  # the chunks it covers, 1-based within the document
    seq_end: int
    line_start: int  # 1-based, in markdown_file
    line_end: int
    char_start: int  # 0-based, in markdown_file
    char_end: int
    page_start: int | None  # 1-based PDF pages; None for non-PDF
    page_end: int | None
    text: str
    score: float
    source_file: str  # absolute, for a tool outside the app
    markdown_file: str
    also_in: list[PassageReference] = []  # the near-duplicates folded in, a tree


class Excerpt(Passage):
    """A passage with its irrelevant parts removed. Today: the passage itself, unchanged."""


MAX_WIDEN = 300  # chars a passage may grow on each side: past this it is a page, not a quote

# A sentence ends at `.!?`, optionally through a closing quote or bracket, and is followed by
# whitespace: the "e.g." case is accepted rather than special-cased, because stopping one clause
# early reads worse than no expansion at all.
SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s")


class Window(msgspec.Struct):
    """A slice of a document's markdown wide enough to widen one range in, and where it sits.

    `char_start` is the offset of `text[0]` in the whole document, so a range's own offsets
    translate into the window and the widened ones translate back out.
    """

    text: str
    char_start: int  # 0-based, in the document this was read from

    def local(self, char: int) -> int:
        """Where a document offset falls in this window. Never negative: a markdown rewritten
        since it was indexed gives a passage that is wrong either way, but a stale offset must not
        index backwards out of the slice. Past the end needs no guard - slicing and `str.find`
        clamp there by themselves."""
        return max(0, char - self.char_start)


def widen[P: Passage](hit_range: HitRange, window: Window, cls: type[P]) -> P:
    """Widen a hit range to the nearest boundary on each side and read it out of `window`, as `cls`.

    A chunk is whole sentences, but a sentence longer than a chunk is cut on words. The text
    around a range can also still be worth reading. Widening stops at the first of three
    boundaries, in this order:

    - a newline, because a line is where the document itself stopped: a heading, a list item, a
      table row, the end of a paragraph;
    - the outermost whole sentence inside `MAX_WIDEN` characters, for a line longer than that;
    - the `MAX_WIDEN` cap, when the text offers neither.

    The line numbers are the range's own, corrected by the newlines the widening crossed: a chunk
    row records the line its text starts on, and widening moves at most `MAX_WIDEN` characters,
    so counting inside that stretch answers what scanning the whole document used to.

    `cls` is the shape the caller wants its passages in (`Passage` or one of its kinds), so an
    excerpt is built rather than converted from one.
    """
    markdown = window.text
    from_start, from_end = window.local(hit_range.char_start), window.local(hit_range.char_end)
    start = _widen_back(markdown, from_start)
    end = _widen_forward(markdown, from_end)
    raw = markdown[start:end]
    text = raw.strip()
    # the offsets have to describe `text`, not the slice it was stripped out of
    local_start = start + len(raw) - len(raw.lstrip())
    local_end = local_start + len(text)
    char_start = window.char_start + local_start
    char_end = window.char_start + local_end
    line_start = _line_shift(markdown, hit_range.line_start, from_start, local_start)
    line_end = _line_shift(
        markdown, hit_range.line_end, max(from_start, from_end - 1), max(local_start, local_end - 1)
    )
    best = hit_range.best
    page_start, page_end = hit_range.page_start, hit_range.page_end
    return cls(
        collection=best.collection,
        document=best.document,
        header=best.header,
        location=location(best.document, page_start, page_end, line_start, line_end),
        seq_start=hit_range.seq_start,
        seq_end=hit_range.seq_end,
        line_start=line_start,
        line_end=line_end,
        char_start=char_start,
        char_end=char_end,
        page_start=page_start,
        page_end=page_end,
        # the offsets and lines above describe the source; the text a reader gets has no page
        # markers, as a chunk's has none (`convert.without_markers`)
        text=without_markers(text).strip(),
        score=hit_range.score,
        source_file=best.source_file,
        markdown_file=best.markdown_file,
        also_in=hit_range.also_in,
    )


def _widen_back(markdown: str, char_start: int) -> int:
    """Where a passage starting at `char_start` begins once widened: after the nearest newline
    before it, else after the first whole sentence inside the cap, else at the cap."""
    cap = max(0, char_start - MAX_WIDEN)
    line = markdown.rfind("\n", cap, char_start)
    if line != -1:
        return line + 1
    sentence = SENTENCE_END.search(markdown, cap, char_start)
    return sentence.end() if sentence else cap


def _widen_forward(markdown: str, char_end: int) -> int:
    """Where a passage ending at `char_end` stops once widened: at the nearest newline after it,
    else after the last whole sentence inside the cap, else at the cap."""
    cap = min(len(markdown), char_end + MAX_WIDEN)
    line = markdown.find("\n", char_end, cap)
    if line != -1:
        return line
    end = cap
    for sentence in SENTENCE_END.finditer(markdown, char_end, cap):
        end = sentence.end()
    return end


def _line_shift(markdown: str, line: int, anchor: int, pos: int) -> int:
    """The line `pos` is on, given that `anchor` is on `line`: the newlines between the two.

    Either direction. Widening moves the start back and the end forward, but stripping whitespace
    moves both the other way, and either can cross a newline the chunk had counted.
    """
    if pos >= anchor:
        return line + markdown.count("\n", anchor, pos)
    return line - markdown.count("\n", pos, anchor)


# --- sources ---------------------------------------------------------------------


class HotSection(msgspec.Struct):
    """One heading of a document the query kept landing under."""

    header: str
    score: float
    chunks: int
    line_start: int
    line_end: int
    location: str


class Source(msgspec.Struct):
    """One document the query matched, and the best evidence that it did.

    The answer to "which documents should I read", not "which passages answer this": `score` is
    the harmonic mean of the document's best chunk and the sum of every scanned chunk that came
    from it (see `harmonic`), and `chunks` how many there were. The evidence fields are its best
    chunk; `sections` and `collections` say where in the document the query landed and which of
    the searched collections hold it.
    """

    collection: str  # the collection whose table held the best chunk; the document belongs to none
    document: str
    score: float
    chunks: int
    description: str
    header: str  # the best chunk's heading path joined, ready to cite
    location: str
    text: str  # the best chunk, so a caller can see why the document is on the list
    # Where the document is on disk, so a tool outside the app can open or grep it. The lines are
    # the best chunk's, in `markdown_file`: somewhere to start reading, not the whole match.
    source_file: str
    markdown_file: str
    line_start: int
    line_end: int
    collections: list[str]  # every searched collection holding it, in name order
    sections: list[HotSection]  # where in it the query landed, best first


class Sources(msgspec.Struct):
    """The answer to "which sources cover this": what to read, and what to narrow a session to."""

    documents: list[Source]
    collections: list[str]  # the fewest that together hold every document above


def _document_score(hits: list[Hit]) -> float:
    """How strongly one document matched, from its chunks in ranked order: its best chunk folded
    with the sum of all of them."""
    return harmonic(hits[0].score, sum(hit.score for hit in hits))


def top_documents(hits: list[Hit], limit: int) -> list[list[Hit]]:
    """The `limit` documents `hits` (best first) point at hardest, best first, each as the chunks
    it was matched by.

    By document name alone, not by (collection, document): a document in two collections is one
    document to read. Cut here rather than after the rows are built, because every document that
    survives costs a membership lookup and a description.
    """
    by_doc: dict[str, list[Hit]] = {}
    for hit in hits:
        by_doc.setdefault(hit.document, []).append(hit)
    ranked = sorted(by_doc.values(), key=lambda group: (-_document_score(group), group[0].document))
    return ranked[:limit]


def fold_sources(
    groups: list[list[Hit]], memberships: dict[str, list[str]], sections: int
) -> Sources:
    """Fold the kept documents (see `top_documents`) to one row each, in the order given, and
    cover them with the fewest collections."""
    documents = [_source(group, memberships, sections) for group in groups]
    return Sources(
        documents=documents,
        collections=min_cover({source.document: source.collections for source in documents}),
    )


def _source(hits: list[Hit], memberships: dict[str, list[str]], sections: int) -> Source:
    """One document's row from its matched chunks, in the order they were ranked: the first hit
    is its best one, and the passage the row shows. `description` is left empty for the caller to
    fill, because it lives in the metadata store and nothing here does IO."""
    best = hits[0]
    return Source(
        collection=best.collection,
        document=best.document,
        score=_document_score(hits),
        chunks=len(hits),
        description="",
        header=best.header,
        location=best.location,
        text=best.text,
        source_file=best.source_file,
        markdown_file=best.markdown_file,
        line_start=best.line_start,
        line_end=best.line_end,
        # a document whose memberships were not looked up is credited to the table that matched it
        collections=memberships.get(best.document, [best.collection]),
        sections=_sections(hits, sections),
    )


def _sections(hits: list[Hit], limit: int) -> list[HotSection]:
    """The `limit` headings of one document the query landed under hardest, scored the way the
    document itself is. `hits` is best first, so each group's first member is its best chunk."""
    by_header: dict[str, list[Hit]] = {}
    for hit in hits:
        by_header.setdefault(hit.header, []).append(hit)
    found: list[HotSection] = []
    for header, group in by_header.items():
        best = group[0]
        line_start = min(hit.line_start for hit in group)
        line_end = max(hit.line_end for hit in group)
        found.append(
            HotSection(
                header=header,
                score=harmonic(best.score, sum(hit.score for hit in group)),
                chunks=len(group),
                line_start=line_start,
                line_end=line_end,
                location=location(best.document, *pages(group), line_start, line_end),
            )
        )
    return sorted(found, key=lambda section: (-section.score, section.header))[:limit]


def min_cover(doc_collections: dict[str, list[str]]) -> list[str]:
    """The fewest collections that together hold every document, greedily and in pick order.

    What a session is narrowed to after a `search_sources`: naming every collection that holds
    any of the documents would widen the next search back out for nothing. Set cover is NP-hard,
    so this is the standard greedy approximation - take the collection covering the most
    uncovered documents, by name when two tie - which is within a log factor and deterministic.
    A document no collection holds is simply left uncovered.
    """
    holders: dict[str, set[str]] = {}
    for doc, names in doc_collections.items():
        for name in names:
            holders.setdefault(name, set()).add(doc)
    picked: list[str] = []
    # every pick subtracts what it covered from the collections left, so "most uncovered" stays
    # true without indexing the documents again per round
    while holders:
        name = min(holders, key=lambda name: (-len(holders[name]), name))
        covered = holders.pop(name)
        if not covered:
            break  # nothing left that any collection still holds; the rest stays uncovered
        picked.append(name)
        for docs in holders.values():
            docs -= covered
    return picked
