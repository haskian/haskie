"""Passages out of chunks: the pure shapes and algorithms behind the search tools.

A chunk is a retrieval unit, not a readable one. It is cut to a size the embedding model likes,
it overlaps its neighbours, and a match usually lands on two or three of them at once. What an
agent wants back is one span of the document that starts and ends where a reader would stop: a
passage.

Three foldings live here, all of them pure. `ranges` merges the chunks of one document that sit
next to each other (`Hit.seq`) into one span. `expand` widens one such span to the nearest
sentence, paragraph or heading boundary of the markdown it was cut from, which is the only step
that needs the file. `top_documents` and `fold_sources` answer the other question - which
documents and which collections cover this - by folding the same hits per document instead of per
span.

`harmonic` is the scoring rule all of them share: a span or a document scores the harmonic mean
of its best chunk and the sum of every chunk it holds. No IO: `retrieval.py` reads the markdown
and the memberships and hands them in.
"""

import bisect
import re

import msgspec

from haskie.collection.index import Hit, location

# --- scoring ---------------------------------------------------------------------


def harmonic(best: float, total: float) -> float:
    """How strongly a document matches: the harmonic mean of its best chunk and the sum of all
    its matched chunks. A document matched once scores its one chunk. Every further chunk lifts
    it, but the mean stays under twice the best, so many weak chunks never outrank one strong one,
    and a document with a few strong chunks is not held back for having few."""
    if best <= 0 or total <= 0:
        return 0.0
    return 2 * best * total / (best + total)


# --- chunk ranges ----------------------------------------------------------------


class ChunkRange(msgspec.Struct):
    """The matched chunks of one document that sit next to each other, as one span."""

    chunks: list[Hit]  # one collection and document, consecutive `seq`, ascending
    seq_start: int
    seq_end: int
    line_start: int
    line_end: int
    char_start: int  # 0-based offsets into the document's markdown
    char_end: int
    score: float  # harmonic(best, sum) over the members


def ranges(hits: list[Hit]) -> list[ChunkRange]:
    """Fold hits into spans, best span first.

    Only chunks with no gap between them merge: a gap is text the query did not match, and
    bridging it would put an unmatched paragraph inside a quoted passage. Ties sort by document
    and position, so the same hits always fold the same way.

    Grouped by (collection, document) rather than by document: two collections may chunk the same
    document with different settings, and each numbers `seq` from 1, so a run across them would
    merge spans cut at different offsets.
    """
    # ponytail: a chunk's identity should carry the settings it was cut with (its embedding cache
    # id), so grouping and deduplication can key on that instead of standing the collection in
    # for it.
    found: list[ChunkRange] = []
    run: list[Hit] = []
    for hit in sorted(hits, key=lambda hit: (hit.collection, hit.doc, hit.seq)):
        last = run[-1] if run else None
        # the run goes on only where this hit is the next chunk of the same table's same document
        if last is not None and (hit.collection, hit.doc, hit.seq) != (
            last.collection,
            last.doc,
            last.seq + 1,
        ):
            found.append(_range(run))
            run = []
        run.append(hit)
    if run:
        found.append(_range(run))
    return sorted(found, key=lambda found: (-found.score, found.chunks[0].doc, found.seq_start))


def _range(chunks: list[Hit]) -> ChunkRange:
    """One span out of an ascending run of chunks of one document."""
    best = max(hit.score for hit in chunks)
    return ChunkRange(
        chunks=chunks,
        seq_start=chunks[0].seq,
        seq_end=chunks[-1].seq,
        line_start=min(hit.line_start for hit in chunks),
        line_end=max(hit.line_end for hit in chunks),
        char_start=min(hit.char_start for hit in chunks),
        char_end=max(hit.char_end for hit in chunks),
        score=harmonic(best, sum(hit.score for hit in chunks)),
    )


# --- passages --------------------------------------------------------------------


class Passage(msgspec.Struct):
    """One span of a document, widened to boundaries a reader would stop at. What a search
    answers with: `header` and `location` are what to cite it by."""

    collection: str  # the collection whose table matched; the document itself belongs to none
    doc: str
    header: str  # breadcrumb "parent > ... > heading", from the best chunk
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


class Excerpt(Passage):
    """A passage with its irrelevant parts removed. Today: the passage itself, unchanged."""


MAX_EXPAND = 300  # chars per side; past this a passage stops being an excerpt

# A sentence ends at `.!?`, optionally through a closing quote or bracket, and is followed by
# whitespace: the "e.g." case is accepted rather than special-cased, because stopping one clause
# early reads worse than no expansion at all.
SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s")
PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
# The whole ATX heading line: expanding backward starts after it, forward stops before it.
HEADING_LINE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t].*\n?", re.MULTILINE)


def newline_offsets(markdown: str) -> list[int]:
    """Where every newline of a document sits, ascending. Counted once per document, so that
    `expand` finds the line a passage starts on by bisecting this rather than by counting the
    newlines before it again per passage."""
    return [match.start() for match in re.finditer("\n", markdown)]


def expand[P: Passage](span: ChunkRange, markdown: str, newlines: list[int], cls: type[P]) -> P:
    """Widen a span to the nearest boundary on each side and read it out of `markdown`, as `cls`.

    The chunk splitter cuts on size, so a span starts and ends mid-sentence as often as not.
    Widening stops at the first sentence terminator, paragraph break or heading line within
    `MAX_EXPAND` characters, and at `MAX_EXPAND` itself when the text offers none - a passage
    that ran to the next heading would be a section, not a quote.

    `newlines` is `newline_offsets(markdown)`; `cls` is the shape the caller wants its passages
    in (`Passage` or one of its kinds), so an excerpt is built rather than converted from one.
    """
    limit = len(markdown)
    start = _widen_back(markdown, min(span.char_start, limit))
    end = _widen_forward(markdown, min(span.char_end, limit))
    raw = markdown[start:end]
    text = raw.strip()
    # the offsets have to describe `text`, not the slice it was stripped out of
    char_start = start + len(raw) - len(raw.lstrip())
    char_end = char_start + len(text)
    line_start = _line_at(newlines, char_start)
    line_end = _line_at(newlines, max(char_start, char_end - 1))
    best = max(span.chunks, key=lambda hit: (hit.score, -hit.seq))
    return cls(
        collection=best.collection,
        doc=best.doc,
        header=best.header,
        location=location(best.doc, best.page_start, best.page_end, line_start, line_end),
        seq_start=span.seq_start,
        seq_end=span.seq_end,
        line_start=line_start,
        line_end=line_end,
        char_start=char_start,
        char_end=char_end,
        page_start=best.page_start,
        page_end=best.page_end,
        text=text,
        score=span.score,
        source_file=best.source_file,
        markdown_file=best.markdown_file,
    )


def _widen_back(markdown: str, char_start: int) -> int:
    """Just after the last boundary before `char_start`, or the cap when there is none.

    Offsets into the whole string rather than a slice of it: `^` only means "line start" when the
    pattern can see the character before the window.
    """
    cap = max(0, char_start - MAX_EXPAND)
    starts = [cap]
    for pattern in (SENTENCE_END, PARAGRAPH_BREAK, HEADING_LINE):
        starts += [match.end() for match in pattern.finditer(markdown, cap, char_start)]
    return max(starts)


def _widen_forward(markdown: str, char_end: int) -> int:
    """At the first boundary after `char_end`, or the cap when there is none. A sentence
    terminator belongs to the sentence it closes; a heading belongs to the section it opens."""
    cap = min(len(markdown), char_end + MAX_EXPAND)
    ends = [cap]
    for pattern in (SENTENCE_END, PARAGRAPH_BREAK):
        if (match := pattern.search(markdown, char_end, cap)) is not None:
            ends.append(match.end())
    if (heading := HEADING_LINE.search(markdown, char_end, cap)) is not None:
        ends.append(heading.start())
    return min(ends)


def _line_at(newlines: list[int], pos: int) -> int:
    """The 1-based line `pos` sits on: the newlines before it, counted by bisecting them."""
    return bisect.bisect_left(newlines, pos) + 1


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
    doc: str
    score: float
    chunks: int
    description: str
    heading: str
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
        by_doc.setdefault(hit.doc, []).append(hit)
    ranked = sorted(by_doc.values(), key=lambda group: (-_document_score(group), group[0].doc))
    return ranked[:limit]


def fold_sources(
    groups: list[list[Hit]], memberships: dict[str, list[str]], sections: int
) -> Sources:
    """Fold the kept documents (see `top_documents`) to one row each, in the order given, and
    cover them with the fewest collections."""
    documents = [_source(group, memberships, sections) for group in groups]
    return Sources(
        documents=documents,
        collections=min_cover({source.doc: source.collections for source in documents}),
    )


def _source(hits: list[Hit], memberships: dict[str, list[str]], sections: int) -> Source:
    """One document's row from its matched chunks, in the order they were ranked: the first hit
    is its best one, and the passage the row shows. `description` is left empty for the caller to
    fill, because it lives in the metadata store and nothing here does IO."""
    best = hits[0]
    return Source(
        collection=best.collection,
        doc=best.doc,
        score=_document_score(hits),
        chunks=len(hits),
        description="",
        heading=best.heading,
        location=best.location,
        text=best.text,
        source_file=best.source_file,
        markdown_file=best.markdown_file,
        line_start=best.line_start,
        line_end=best.line_end,
        # a document whose memberships were not looked up is credited to the table that matched it
        collections=memberships.get(best.doc, [best.collection]),
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
                location=location(best.doc, best.page_start, best.page_end, line_start, line_end),
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
