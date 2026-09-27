"""Passages out of chunks: the pure shapes and algorithms behind the search tools.

A chunk is a retrieval unit, not a readable one. It is cut to a size the embedding model likes, it
ends where a paragraph or the size did, and a match usually lands on two or three of them at once.
What an agent wants back is one span of the document that starts and ends where a reader would stop:
a passage.

Three folds live here, all of them pure. `ranges` merges the chunks of one section that sit
next to each other (`Hit.seq`) into one range. `quote` turns one such range and the markdown its
offsets cover into a passage. `top_documents` and `fold_sources` answer the other question:
which documents and which collections cover this. They fold the same hits per document instead
of per range.

`fold` is the scoring rule all of them share: how a range's or a document's chunk scores become one,
by the `score_fold` setting (`ScoreFold`). No IO: `retrieval.py` reads the markdown and the
memberships and hands them in.
"""

import msgspec

from haskie.collection.index import ChunkKey, Hit, Overlaps, Relation, chunk_key, location
from haskie.document.convert import without_markers
from haskie.indexing.segment import CutReason
from haskie.settings import ScoreFold

# --- scoring ---------------------------------------------------------------------


def fold(scores: list[float], how: ScoreFold) -> float:
    """The scores of several matched chunks as one, by the `score_fold` setting:

    - `sum`: every chunk adds, so a result with more matching text ranks higher. Vespa scores a
      chunked document so in its example (`sum(chunk_scores())`, "Working with chunks").
    - `max`: the best chunk alone, so a result is as good as its best part. Elasticsearch scores a
      `semantic_text` field so: "the most relevant passage will be used to compute a score".
    - `harmonic`: harmonic(best, sum), between the best and twice it. Every further chunk lifts
      the result, but many weak chunks never outrank one strong one. haskie's own rule; no source
      measures it against the other two.

    A chunk the search did not rank (grown or filled in) scores 0 and adds nothing under any of
    them. Empty is 0."""
    if not scores:
        return 0.0
    if how == ScoreFold.MAX:
        return max(scores)
    if how == ScoreFold.SUM:
        return sum(scores)
    return harmonic(max(scores), sum(scores))


def harmonic(a: float, b: float) -> float:
    """The harmonic mean of `a` and `b`: pulled toward the smaller, so one weak value holds the
    whole down, and 0 when either is 0 or less.

    Two uses. The `harmonic` rule of `fold` is harmonic(best chunk, sum of all). How strongly two
    results overlap is harmonic(contained, contains) (`collapse`): high only when each holds the
    other, whichever rule folds the scores."""
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
    """The matched chunks of one section that sit next to each other, as one range."""

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
    score: float  # its members' scores, folded by the search's rule (`fold`)
    also_in: list[PassageReference] = []  # the near-duplicates folded in (`collapse`), a tree
    aspects: list[str] = []  # the questions it answers when several were asked (`aspects`)
    # how well it matched each question whose own ranking holds its chunks (`aspects.tagged`)
    aspect_scores: dict[str, float] = {}
    # too short to stand alone, and nothing around it matched (`thin`): a passage of its own goes,
    # while it stays inside an excerpt whose section holds another passage (`section.group`)
    alone: bool = False

    @property
    def best(self) -> Hit:
        """The hit a range is cited by: its best score, the earliest on a tie."""
        return max(self.hits, key=lambda hit: (hit.score, -hit.seq))

    @property
    def location(self) -> str:
        """The range's citation, "doc p.3-4 L10-20", over every chunk it holds."""
        return location(
            self.hits[0].document, self.page_start, self.page_end, self.line_start, self.line_end
        )


def pages(hits: list[Hit]) -> tuple[int | None, int | None]:
    """The pages a group of hits covers, first to last: every hit's, not only the best one's, or
    a passage over pages 1 to 5 would be cited as page 1. None for a document without pages."""
    starts = [hit.page_start for hit in hits if hit.page_start is not None]
    ends = [hit.page_end for hit in hits if hit.page_end is not None]
    return (min(starts) if starts else None, max(ends) if ends else None)


def ends_section(hit: Hit) -> bool:
    """Whether a heading follows the chunk: the next chunk opens another section.

    A part boundary (`edge` between two batches of a PDF) is no section end: the section goes on
    in the next part.
    """
    return hit.end_reason == CutReason.HEADING


def continues(before: Hit, after: Hit) -> bool:
    """Whether `after` is the next chunk of the same table's same document, in the same section:
    what may join `before` in one passage. A passage never crosses a heading: the two would be
    read as one quote of two topics, with the heading line in the middle."""
    return (after.collection, after.document, after.seq) == (
        before.collection,
        before.document,
        before.seq + 1,
    ) and not ends_section(before)


def ranges(hits: list[Hit], how: ScoreFold) -> list[HitRange]:
    """Fold hits into ranges, best range first, each scored by `how` (`fold`).

    Only chunks with no gap between them merge, and only within one section (`continues`): a gap
    is text the query did not match, and bridging it would put an unmatched paragraph inside a
    quoted passage. Ties sort by document and position, so the same hits always fold the same
    way.

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
        if run and not continues(run[-1], hit):
            found.append(_range(run, how))
            run = []
        run.append(hit)
    if run:
        found.append(_range(run, how))
    return sorted(found, key=lambda found: (-found.score, found.hits[0].document, found.seq_start))


def part(hit: Hit, aspects: list[str] | None = None) -> HitRange:
    """One chunk a search did not rank, as a range to `rejoin` to the ranges it sits next to: its
    score 0, so a range it joins scores what its ranked chunks matched, and the questions it
    answers, if any."""
    # a score of 0 folds to 0 under every rule
    (found,) = ranges([msgspec.structs.replace(hit, score=0.0)], ScoreFold.SUM)
    return msgspec.structs.replace(found, aspects=aspects or [])


def rejoin(parts: list[HitRange], how: ScoreFold) -> list[HitRange]:
    """The ranges over every chunk `parts` hold, rebuilt by `ranges` and scored by `how`, so parts
    that overlap or continue each other in one section become one range, best first.

    A chunk several parts hold counts once, as the first of them has it. Each rebuilt range
    carries what its parts carried: their folded places, their questions in the order the parts
    are listed, and `alone` (`thin`) only when every one of them was. A part never splits, since
    its chunks continue each other, so each lands whole in one rebuilt range.
    """
    hits: dict[ChunkKey, Hit] = {}
    for one in parts:
        for hit in one.hits:
            hits.setdefault(chunk_key(hit), hit)
    starts: dict[ChunkKey, list[int]] = {}  # the parts each chunk is the first chunk of
    for at, one in enumerate(parts):
        starts.setdefault(chunk_key(one.hits[0]), []).append(at)
    rebuilt: list[HitRange] = []
    for one in ranges(list(hits.values()), how):
        held = [
            parts[at]
            for at in sorted(at for hit in one.hits for at in starts.get(chunk_key(hit), ()))
        ]
        rebuilt.append(
            msgspec.structs.replace(
                one,
                also_in=[place for kept in held for place in kept.also_in],
                aspects=list(dict.fromkeys(label for kept in held for label in kept.aspects)),
                aspect_scores=best_of([kept.aspect_scores for kept in held]),
                alone=all(kept.alone for kept in held),
            )
        )
    return rebuilt


def best_of(scores: list[dict[str, float]]) -> dict[str, float]:
    """Each question's best score over several sets, in the order they name them."""
    best: dict[str, float] = {}
    for one in scores:
        for label, score in one.items():
            best[label] = max(score, best.get(label, score))
    return best


def _range(hits: list[Hit], how: ScoreFold) -> HitRange:
    """One range out of an ascending run of hits of one document."""
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
        score=fold([hit.score for hit in hits], how),
    )


# --- passages --------------------------------------------------------------------


class Span(msgspec.Struct, kw_only=True):
    """One run of a document's matched chunks, cut where the chunker cut, and where it is: what
    to cite it by (`header`, `location`) and how it matched. An excerpt holds its passages as
    spans; a `Passage` is one with its text."""

    header: str  # the heading path it sits under, "Part I > Chapter 2", from the best chunk
    location: str  # "doc p.3-4 L10-20", over every chunk it covers
    seq_start: int  # the chunks it covers, 1-based within the document
    seq_end: int
    line_start: int  # 1-based, in markdown_file
    line_end: int
    char_start: int  # 0-based, in markdown_file
    char_end: int
    page_start: int | None  # 1-based PDF pages; None for non-PDF
    page_end: int | None
    score: float
    also_in: list[PassageReference] = []  # the near-duplicates folded in, a tree
    # the questions it answers when several were asked at once (`aspects`), else empty
    aspects: list[str] = []
    # how well it matched each question whose ranking holds its chunks, several asked, else empty
    aspect_scores: dict[str, float] = {}


class Passage(Span, kw_only=True):
    """One span of a document with its text: what the `passage` granularity answers with."""

    collection: str  # the collection whose table matched; the document itself belongs to none
    document: str
    text: str
    source_file: str  # absolute, for a tool outside the app
    markdown_file: str


class Excerpt(msgspec.Struct):
    """What one section of one document says on the question: every passage of it the search
    kept (`spans`), in document order, as one text.

    The section is the largest heading of the document whose text fits `max_section_chars`
    (`section`), so the passages of one topic come back together rather than as rivals for the
    slots. `text` is the passages joined: each opened by the headings it sits under below
    `header`, and `[…]` where the document skips text between two of them. The offsets, lines and
    pages run from the first passage to the last.
    """

    collection: str  # the collection whose table matched; the document itself belongs to none
    document: str
    header: str  # the section's heading path, "Part I > Chapter 2"; empty for a whole document
    location: str  # "doc p.3-4 L10-20", from the first passage to the last
    seq_start: int
    seq_end: int
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    text: str
    score: float  # the best passage's
    source_file: str  # absolute, for a tool outside the app
    markdown_file: str
    spans: list[Span]  # in document order
    # the questions any of its passages answers when several were asked at once, else empty
    aspects: list[str] = []
    # each question's best score over the passages, several asked, else empty
    aspect_scores: dict[str, float] = {}


class Answer(msgspec.Struct):
    """What an excerpts search answers with: the excerpts, and what they leave out."""

    excerpts: list[Excerpt]  # best first
    # the questions no excerpt answers, when several were asked at once; else empty
    uncovered: list[str]
    # the words of the questions no excerpt's text or headings hold (`probe`), in the order asked
    missing_terms: list[str]


def span(hit_range: HitRange) -> Span:
    """Where a hit range is and how it matched, as an excerpt lists it."""
    return Span(**_cited(hit_range))


def _cited(hit_range: HitRange) -> dict:
    """The fields a `Span` has, out of a hit range."""
    return {
        "header": hit_range.best.header,
        "location": hit_range.location,
        "seq_start": hit_range.seq_start,
        "seq_end": hit_range.seq_end,
        "line_start": hit_range.line_start,
        "line_end": hit_range.line_end,
        "char_start": hit_range.char_start,
        "char_end": hit_range.char_end,
        "page_start": hit_range.page_start,
        "page_end": hit_range.page_end,
        "score": hit_range.score,
        "also_in": hit_range.also_in,
        "aspects": hit_range.aspects,
        "aspect_scores": hit_range.aspect_scores,
    }


def quote(hit_range: HitRange, text: str) -> Passage:
    """A hit range as a passage, given `text`: the markdown its offsets cover.

    The range is quoted as it was cut, with nothing added around it. The chunker already cuts at
    a heading, a blank line, a block or a sentence (`docs/chunking.md`), so every range starts and
    ends where the author stopped. Snapping each edge to the nearest line or sentence end changed
    0.7% of 667 chunks of real markdown, all of them sentence cuts inside a paragraph longer than a
    chunk, and what it added was the neighbouring chunk's text, unchecked against the query.
    """
    best = hit_range.best
    return Passage(
        **_cited(hit_range),
        collection=best.collection,
        document=best.document,
        # the offsets and lines describe the source; the text a reader gets has no page markers,
        # as a chunk's has none (`convert.without_markers`)
        text=without_markers(text).strip(),
        source_file=best.source_file,
        markdown_file=best.markdown_file,
    )


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

    The answer to "which documents should I read", not "which passages answer this": `score`
    folds every scanned chunk that came from it by the search's rule (`fold`), and `chunks` says
    how many there were. The evidence fields are its best
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


def _document_score(hits: list[Hit], how: ScoreFold) -> float:
    """How strongly one document matched: its chunks' scores folded by `how` (`fold`)."""
    return fold([hit.score for hit in hits], how)


def top_documents(hits: list[Hit], limit: int, how: ScoreFold) -> list[list[Hit]]:
    """The `limit` documents `hits` (best first) point at hardest, best first, each as the chunks
    it was matched by, scored by `how` (`fold`).

    By document name alone, not by (collection, document): a document in two collections is one
    document to read. Cut here rather than after the rows are built, because every document that
    survives costs a membership lookup and a description.
    """
    by_doc: dict[str, list[Hit]] = {}
    for hit in hits:
        by_doc.setdefault(hit.document, []).append(hit)
    ranked = sorted(
        by_doc.values(), key=lambda group: (-_document_score(group, how), group[0].document)
    )
    return ranked[:limit]


def fold_sources(
    groups: list[list[Hit]], memberships: dict[str, list[str]], sections: int, how: ScoreFold
) -> Sources:
    """Fold the kept documents (see `top_documents`) to one row each, in the order given, and
    cover them with the fewest collections."""
    documents = [_source(group, memberships, sections, how) for group in groups]
    return Sources(
        documents=documents,
        collections=min_cover({source.document: source.collections for source in documents}),
    )


def _source(
    hits: list[Hit], memberships: dict[str, list[str]], sections: int, how: ScoreFold
) -> Source:
    """One document's row from its matched chunks, in the order they were ranked: the first hit
    is its best one, and the passage the row shows. `description` is left empty for the caller to
    fill, because it lives in the metadata store and nothing here does IO."""
    best = hits[0]
    return Source(
        collection=best.collection,
        document=best.document,
        score=_document_score(hits, how),
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
        sections=_sections(hits, sections, how),
    )


def _sections(hits: list[Hit], limit: int, how: ScoreFold) -> list[HotSection]:
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
                score=fold([hit.score for hit in group], how),
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
