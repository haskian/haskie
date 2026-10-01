"""Passages out of chunks: folding hits into ranges, quoting a range as it was cut, and folding
the same hits into the documents and collections that cover a query.

Every offset below is a real offset into `MARKDOWN`: the fixture builds a `Hit` from a pair of
snippets and reads its text, lines and char range out of the document, the way the index does.
"""

import random

import msgspec
import pytest
from conftest import chunk_hit, hit

from haskie.collection.index import Hit, Overlap, Overlaps, Relation, location
from haskie.indexing.chunk import Chunk, open_headings, split
from haskie.indexing.segment import CutReason
from haskie.search.passage import (
    HitRange,
    Passage,
    PassageReference,
    Sources,
    fold,
    fold_sources,
    harmonic,
    min_cover,
    part,
    quote,
    ranges,
    rejoin,
    top_documents,
)
from haskie.settings import ChunkSettings, ScoreFold

HARMONIC = ScoreFold.HARMONIC  # the rule these cases were written against


MARKDOWN = """# Retries

A background job retries a failed HTTP call. The retry has to be idempotent, or the side
effect happens twice.

## Backoff

Exponential backoff with jitter spreads the retries. A fixed delay buys a thundering herd
instead.

## Ordering

Never trust a wall clock for ordering
### Skew
Hosts drift apart by milliseconds.

## Deduplication

The consumer keys on an idempotency key. It drops any message it has already handled.
"""

DOC = "retries.md"
OTHER = "ordering.md"
COLLECTION = "backend"


def _at(snippet: str) -> int:
    """Where `snippet` starts in the fixture. It has to appear exactly once, or the span a case
    describes would not be the span it gets."""
    assert MARKDOWN.count(snippet) == 1, f"not unique in the fixture: {snippet!r}"
    return MARKDOWN.index(snippet)


def _span(begin: str, end: str) -> tuple[int, int]:
    """The char range from the start of `begin` through the end of `end`."""
    return _at(begin), _at(end) + len(end)


def _text(begin: str, end: str) -> str:
    """The fixture's own text from the start of `begin` through the end of `end`."""
    start, stop = _span(begin, end)
    return MARKDOWN[start:stop]


def _hit(
    span: tuple[int, int],
    seq: int,
    score: float,
    *,
    document: str = DOC,
    collection: str = COLLECTION,
    header: str = "Retries",
    page_start: int | None = None,
    page_end: int | None = None,
) -> Hit:
    """One indexed chunk of `MARKDOWN`, with the lines and the text its offsets give."""
    char_start, char_end = span
    line_start = MARKDOWN.count("\n", 0, char_start) + 1
    line_end = MARKDOWN.count("\n", 0, char_end - 1) + 1
    return Hit(
        collection=collection,
        document_id=document,
        document=document,
        source_path=f"documents/{document}",
        markdown_path=f"documents/{document}.md",
        part=0,
        seq=seq,
        line_start=line_start,
        line_end=line_end,
        char_start=char_start,
        char_end=char_end,
        byte_start=len(MARKDOWN[:char_start].encode()),
        byte_end=len(MARKDOWN[:char_end].encode()),
        page_start=page_start,
        page_end=page_end,
        headings=header.split(" > ") if header else [],
        frame=header.split(" > ") if header else [],
        header=header,
        location=location(document, page_start, page_end, line_start, line_end),
        text=MARKDOWN[char_start:char_end],
        score=score,
        source_file=f"/home/documents/{document}",
        markdown_file=f"/home/documents/{document}.md",
    )


# The four chunks of the fixture, as the splitter would leave them: 1 and 2 overlap, 3 starts
# past the end of 2 (the splitter dropped nothing, the query skipped a section), 4
# overlaps 3 again.
OPENING = _span("# Retries", "effect happens twice.")
BACKOFF = _span("The retry has to be idempotent", "A fixed delay buys a thundering herd\ninstead.")
SKEW = _span("### Skew", "already handled.")
DEDUP = _span("The consumer keys", "already handled.")


def _chunks(doc: str = DOC) -> list[Hit]:
    return [
        _hit(OPENING, 1, 4.0, document=doc, header="Retries"),
        _hit(BACKOFF, 2, 3.0, document=doc, header="Retries > Backoff"),
        _hit(SKEW, 3, 2.0, document=doc, header="Retries > Ordering > Skew"),
        _hit(DEDUP, 4, 1.0, document=doc, header="Retries > Deduplication"),
    ]


ONE, TWO, THREE, FOUR = _chunks()
OTHER_ONE = _hit(OPENING, 1, 4.0, document=OTHER, collection="ops", header="Retries")


# --- harmonic -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "best", "total", "expected"),
    [
        ("one chunk scores itself", 3.0, 3.0, 3.0),
        ("a second chunk as strong lifts it, short of double", 3.0, 6.0, 4.0),
        ("many weak chunks stay under twice the best", 1.0, 100.0, pytest.approx(200 / 101)),
        ("nothing matched scores nothing", 0.0, 0.0, 0.0),
    ],
)
def test_harmonic_folds_the_best_chunk_with_the_sum(
    name: str, best: float, total: float, expected: float
) -> None:
    assert harmonic(best, total) == pytest.approx(expected), name


@pytest.mark.parametrize(
    ("name", "scores", "sum_", "max_", "harmonic_"),
    [
        ("one chunk scores itself under every rule", [3.0], 3.0, 3.0, 3.0),
        ("a second chunk adds to the sum, not to the max", [3.0, 3.0], 6.0, 3.0, 4.0),
        (
            "many weak chunks: the sum outgrows the harmonic mean",
            [1.0] * 100,
            100.0,
            1.0,
            200 / 101,
        ),
        ("a chunk nobody ranked scores 0 and adds nothing", [2.0, 0.0], 2.0, 2.0, 2.0),
        ("nothing matched scores nothing", [], 0.0, 0.0, 0.0),
    ],
)
def test_a_passages_chunks_fold_by_the_rule_the_settings_choose(
    name: str, scores: list[float], sum_: float, max_: float, harmonic_: float
) -> None:
    """`sum` as Vespa's chunk example, `max` as Elasticsearch's semantic_text, `harmonic` haskie's
    own: between the best and twice it."""
    found = {how: fold(scores, how) for how in ScoreFold}

    assert found == pytest.approx(
        {ScoreFold.SUM: sum_, ScoreFold.MAX: max_, ScoreFold.HARMONIC: harmonic_}
    ), name


@pytest.mark.parametrize(
    ("how", "order"),
    [
        (ScoreFold.SUM, [OTHER, DOC]),
        (ScoreFold.MAX, [DOC, OTHER]),
        (ScoreFold.HARMONIC, [DOC, OTHER]),
    ],
)
def test_documents_rank_by_the_rule_the_settings_choose(how: ScoreFold, order: list[str]) -> None:
    """One strong chunk (5) against three fair ones (2, 2, 2): the sum ranks the document with
    more evidence first, the best chunk and the harmonic mean the strong one."""
    strong = _hit(BACKOFF, 2, 5.0)
    fair = [
        _hit(text, seq, 2.0, document=OTHER, collection="ops", header="Retries")
        for seq, text in ((1, OPENING), (2, BACKOFF), (3, SKEW))
    ]
    hits = [strong, *fair]

    found = fold_sources(top_documents(hits, 10, how), {}, 3, how).documents

    assert [one.document for one in found] == order, how
    assert [ranges(hits, how)[0].hits[0].document] == [order[0]], f"{how}: passages the same"


@pytest.mark.parametrize(
    ("name", "shares", "expected"),
    [
        ("a copy: whole both ways", (1.0, 1.0), 1.0),
        # RETRY in FULLER, by word 3-grams: 13 of RETRY's 13, 13 of FULLER's 28. The harmonic mean
        # of the two shares is their Dice coefficient: 2 * 13 / (13 + 28)
        (
            "a sentence inside a paragraph: held down by the side that is not",
            (1.0, 13 / 28),
            26 / 41,
        ),
        ("one side at zero zeroes the whole", (0.99, 0.0), 0.0),
    ],
)
def test_harmonic_of_an_overlap_is_pulled_toward_its_weakest_measure(
    name: str, shares: tuple[float, float], expected: float
) -> None:
    assert harmonic(*shares) == pytest.approx(expected), name


# --- ranges ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "hits", "expected"),
    [
        ("no hits, no ranges", [], []),
        ("one hit is a range of one", [TWO], [(COLLECTION, DOC, 2, 2)]),
        ("consecutive chunks merge", [ONE, TWO], [(COLLECTION, DOC, 1, 2)]),
        (
            "the input order does not matter: the run is sorted by seq",
            [TWO, ONE],
            [(COLLECTION, DOC, 1, 2)],
        ),
        (
            "a gap in seq splits the range, best range first",
            [ONE, THREE],
            [(COLLECTION, DOC, 1, 1), (COLLECTION, DOC, 3, 3)],
        ),
        (
            "two documents never merge, however their seq lines up",
            [ONE, _hit(BACKOFF, 2, 3.0, document=OTHER)],
            [(COLLECTION, DOC, 1, 1), (COLLECTION, OTHER, 2, 2)],
        ),
        (
            "one document in two collections never merges: each numbers its own chunks",
            [ONE, _hit(BACKOFF, 2, 3.0, collection="ops")],
            [(COLLECTION, DOC, 1, 1), ("ops", DOC, 2, 2)],
        ),
        (
            "a heading between two chunks ends the section: they never merge",
            [
                msgspec.structs.replace(ONE, end_reason=CutReason.HEADING),
                msgspec.structs.replace(TWO, start_reason=CutReason.HEADING),
            ],
            [(COLLECTION, DOC, 1, 1), (COLLECTION, DOC, 2, 2)],
        ),
        (
            "a paragraph between two chunks is one section: they merge",
            [
                msgspec.structs.replace(ONE, end_reason=CutReason.PARAGRAPH),
                msgspec.structs.replace(TWO, start_reason=CutReason.PARAGRAPH),
            ],
            [(COLLECTION, DOC, 1, 2)],
        ),
        (
            "the edge between two parts of a PDF is no section end: they merge",
            [
                msgspec.structs.replace(ONE, end_reason=CutReason.EDGE),
                msgspec.structs.replace(TWO, start_reason=CutReason.EDGE),
            ],
            [(COLLECTION, DOC, 1, 2)],
        ),
        (
            "spans that score the same sort by document, then by position",
            [_hit(SKEW, 3, 4.0), ONE, OTHER_ONE],
            [("ops", OTHER, 1, 1), (COLLECTION, DOC, 1, 1), (COLLECTION, DOC, 3, 3)],
        ),
    ],
)
def test_ranges_folds_consecutive_chunks_of_one_document(
    name: str, hits: list[Hit], expected: list[tuple[str, str, int, int]]
) -> None:
    folded = ranges(hits, how=HARMONIC)

    shape = [(r.hits[0].collection, r.hits[0].document, r.seq_start, r.seq_end) for r in folded]
    assert shape == expected, name


def test_a_range_carries_the_span_and_the_score_of_its_members() -> None:
    """The span is the union of its chunks, and the score is the document rule applied to one
    span: one strong chunk lifted by what sits next to it."""
    (folded,) = ranges([ONE, TWO], how=HARMONIC)

    assert (folded.char_start, folded.char_end) == (ONE.char_start, TWO.char_end)
    assert (folded.line_start, folded.line_end) == (ONE.line_start, TWO.line_end)
    assert folded.hits == [ONE, TWO], "the members, ascending, for a caller that wants them"
    assert folded.score == pytest.approx(2 * 4.0 * 7.0 / 11.0), "harmonic(best 4, sum 7)"


# --- quote ---------------------------------------------------------------------------


def _range(char_start: int, char_end: int, **fields) -> HitRange:
    """A range of one chunk over `[char_start, char_end)`, as `ranges` would build it."""
    return ranges([_hit((char_start, char_end), 1, 2.0, **fields)], how=HARMONIC)[0]


def _quote(hit_range: HitRange) -> Passage:
    """`hit_range` as a passage, handed the markdown its offsets cover, as `retrieval` reads it."""
    return quote(hit_range, MARKDOWN[hit_range.char_start : hit_range.char_end])


@pytest.mark.parametrize(
    ("name", "pages", "expected"),
    [
        ("chunks over several pages: the first to the last", [(1, 1), (2, 3), (4, 4)], (1, 4)),
        ("one chunk across a page break", [(2, 3)], (2, 3)),
        ("a document without pages", [(None, None), (None, None)], (None, None)),
        (
            "a chunk without a page marker keeps the pages the others know",
            [(None, None), (2, 3), (None, None)],
            (2, 3),
        ),
    ],
)
def test_a_passage_cites_every_page_its_chunks_cover(
    name: str, pages: list[tuple[int | None, int | None]], expected: tuple
) -> None:
    """Not the best chunk's pages alone: a passage over pages 1 to 4 was cited as page 1."""
    hits = [
        _hit(span, seq, 1.0 if seq == 1 else 3.0, page_start=start, page_end=end)
        for seq, (span, (start, end)) in enumerate(
            zip([OPENING, BACKOFF, SKEW], pages, strict=False), 1
        )
    ]
    (hit_range,) = ranges(hits, how=HARMONIC)

    quoted = _quote(hit_range)

    assert (hit_range.page_start, hit_range.page_end) == expected, f"{name}: set once, on the range"
    assert (quoted.page_start, quoted.page_end) == expected, name
    cited = "" if expected[0] is None else f" p.{expected[0]}-{expected[1]} "
    assert cited in quoted.location, f"{name}: the citation names the same pages"


# a place measured close to the passage it folded into, in words and in its vectors
CLOSE = Overlaps(
    words=Overlap(contained=0.9, contains=0.6, alike=0.55, score=harmonic(0.9, 0.6)),
    embedding=Overlap(contained=0.97, contains=0.93, alike=0.95, score=harmonic(0.97, 0.93)),
    chars=None,
)


def test_quoting_a_range_keeps_what_was_folded_into_it() -> None:
    """The pointers are decided on the range (`collapse`), before anything is read, and the
    passage is what the caller sees them on."""
    folded = PassageReference(
        collection="ops",
        document_id=OTHER,
        document=OTHER,
        seq_start=1,
        seq_end=1,
        header="Retries",
        location=OTHER_ONE.location,
        line_start=OTHER_ONE.line_start,
        line_end=OTHER_ONE.line_end,
        score=4.0,
        relation=Relation.EQUIVALENT,
        similarity=0.95,
        to_parent=CLOSE,
        to_root=CLOSE,
        also_in=[
            PassageReference(
                collection="notes",
                document_id="notes.md",
                document="notes.md",
                seq_start=4,
                seq_end=4,
                header="Delivery",
                location="notes.md L7-7",
                line_start=7,
                line_end=7,
                score=3.0,
                relation=Relation.CONTAINED,
                similarity=0.9,
                to_parent=CLOSE,
                to_root=CLOSE,
            )
        ],
    )
    (hit_range,) = ranges([ONE, TWO], how=HARMONIC)
    hit_range = msgspec.structs.replace(hit_range, also_in=[folded])

    quoted = _quote(hit_range)

    assert quoted.also_in == [folded]


@pytest.mark.parametrize(
    ("name", "span", "lines"),
    [
        (
            "a range that starts and ends mid-line gains nothing around it",
            _span("has to be idempotent", "or the side"),
            (3, 3),
        ),
        ("a heading line", _span("### Skew", "### Skew"), (14, 14)),
        ("the start of the file", (0, len("# Retries")), (1, 1)),
        (
            "a whole paragraph, as the chunker cuts most of them",
            _span("The consumer keys", "already handled."),
            (19, 19),
        ),
    ],
)
def test_a_passage_is_its_range_and_nothing_around_it(
    name: str, span: tuple[int, int], lines: tuple[int, int]
) -> None:
    """The chunker already cuts where the author did, so the range is quoted as it was cut: the
    text, the offsets and the lines are the range's own."""
    hit_range = _range(*span)

    passage = _quote(hit_range)

    assert passage.text == MARKDOWN[span[0] : span[1]], name
    assert (passage.char_start, passage.char_end) == span, f"{name}: the range's offsets"
    assert (passage.line_start, passage.line_end) == lines, f"{name}: the range's lines"
    assert (passage.line_start, passage.line_end) == (hit_range.line_start, hit_range.line_end)


def test_a_passage_across_a_page_break_carries_no_page_marker() -> None:
    """A page marker is the converter's, not the document's: a passage spanning one reads without
    it, as a chunk does, while its offsets still cut the source the file holds."""
    markdown = "# Retries\n\nThe retry waits.\n\n<!-- page 2 -->\n\nThen it runs again.\n"
    span = (markdown.index("The retry"), markdown.index("again.") + len("again."))
    passage = quote(_range(*span), markdown[span[0] : span[1]])
    assert "<!--" not in passage.text
    assert passage.text == "The retry waits.\n\nThen it runs again."
    assert "<!-- page 2 -->" in markdown[passage.char_start : passage.char_end], "the source's"


def test_a_passage_carries_the_citation_of_the_best_chunk_over_its_lines() -> None:
    """A passage is cited the way a chunk is: `header` from the chunk that ranked it, `location`
    over the lines all its chunks cover and the pages every one of its chunks is on."""
    hits = [
        _hit(_span("# Retries", "HTTP call."), 1, 1.0, page_start=1, page_end=1),
        _hit(
            _span("The retry", "effect happens twice."),
            2,
            5.0,
            header="Retries > Backoff",
            page_start=2,
            page_end=3,
        ),
    ]

    passage = _quote(ranges(hits, how=HARMONIC)[0])

    assert passage.header == "Retries > Backoff", "the best-scoring chunk names the passage"
    assert (passage.page_start, passage.page_end) == (1, 3), "both chunks' pages, not the best's"
    assert passage.location == f"{DOC} p.1-3 L1-4", "rebuilt over the passage's own lines"
    assert (passage.seq_start, passage.seq_end) == (1, 2)
    assert passage.collection == COLLECTION
    assert passage.markdown_file == f"/home/documents/{DOC}.md"
    assert passage.score == pytest.approx(2 * 5.0 * 6.0 / 11.0)
    assert passage.text.startswith("# Retries") and passage.text.endswith("twice.")


# --- fold_sources ---------------------------------------------------------------------


def _sources(hits: list[Hit], memberships=None, limit: int = 10, sections: int = 3) -> Sources:
    return fold_sources(
        top_documents(hits, limit, how=HARMONIC), memberships or {}, sections, how=HARMONIC
    )


def test_one_document_folds_to_one_row_of_evidence() -> None:
    """The first hit of a document is its best one, so the row shows that chunk and scores the
    whole group; the description is left for the caller, which is the only part that needs IO."""
    found = _sources([ONE, TWO, THREE])

    (source,) = found.documents
    assert (source.document, source.chunks) == (DOC, 3)
    assert source.score == pytest.approx(2 * 4.0 * 9.0 / 13.0), "harmonic(best 4, sum 9)"
    assert (source.text, source.header, source.location) == (ONE.text, ONE.header, ONE.location)
    assert (source.line_start, source.line_end) == (ONE.line_start, ONE.line_end)
    assert (source.source_file, source.markdown_file) == (ONE.source_file, ONE.markdown_file)
    assert source.description == "", "filled in by the caller, from the metadata store"
    assert source.collections == [COLLECTION], "nothing looked up, so the table that matched it"
    assert found.collections == [COLLECTION]


def test_a_document_in_two_collections_names_both_and_is_covered_by_one() -> None:
    found = _sources([ONE, OTHER_ONE], memberships={DOC: ["archive", COLLECTION], OTHER: ["ops"]})

    assert [(s.document, s.collections) for s in found.documents] == [
        (OTHER, ["ops"]),
        (DOC, ["archive", COLLECTION]),
    ], "equal scores sort by document name; the memberships are carried as given"
    assert found.collections == ["archive", "ops"], "one per document, ties picked by name"


@pytest.mark.parametrize(
    ("name", "hits", "limit", "expected"),
    [
        ("no hits, no sources", [], 10, []),
        (
            "one strong chunk outranks three weak ones",
            [_hit(OPENING, 1, 5.0)] + [_hit(OPENING, i, 2.0, document=OTHER) for i in (1, 2, 3)],
            10,
            [(DOC, 5.0), (OTHER, 2 * 2.0 * 6.0 / 8.0)],
        ),
        (
            "the limit cuts the tail of the ranking",
            [_hit(OPENING, 1, 5.0), _hit(OPENING, 1, 4.0, document=OTHER)],
            1,
            [(DOC, 5.0)],
        ),
        (
            "documents that score the same sort by name",
            [_hit(OPENING, 1, 3.0), _hit(OPENING, 1, 3.0, document=OTHER)],
            10,
            [(OTHER, 3.0), (DOC, 3.0)],
        ),
    ],
)
def test_fold_sources_ranks_documents_by_the_harmonic_of_best_and_sum(
    name: str, hits: list[Hit], limit: int, expected: list[tuple[str, float]]
) -> None:
    found = _sources(hits, limit=limit)

    assert [(s.document, pytest.approx(s.score)) for s in found.documents] == expected, name


def test_sections_are_the_headings_the_query_kept_landing_under() -> None:
    """A document is worth reading in one place more than another. The sections score the way the
    document does, so two weak chunks under one heading can outrank one middling chunk alone."""
    hits = [
        _hit(BACKOFF, 2, 3.0, header="Retries > Backoff"),
        _hit(SKEW, 3, 2.5, header="Retries > Ordering > Skew"),
        _hit(DEDUP, 4, 2.0, header="Retries > Backoff"),
        _hit(OPENING, 1, 1.0, header="Retries"),
    ]

    (source,) = _sources(hits, sections=2).documents

    assert [(s.header, s.chunks) for s in source.sections] == [
        ("Retries > Backoff", 2),
        ("Retries > Ordering > Skew", 1),
    ], "top 2 by score; the lone weak heading is cut"
    hot = source.sections[0]
    assert hot.score == pytest.approx(2 * 3.0 * 5.0 / 8.0), "harmonic(best 3, sum 5)"
    assert (hot.line_start, hot.line_end) == (3, 19), "min and max over the heading's chunks"
    assert hot.location == f"{DOC} L3-19", "cited over the whole heading, not one chunk"


def test_a_section_cites_every_page_its_chunks_cover() -> None:
    """The best chunk sits on page 2, a weaker one under the same heading on page 4: the section
    spans both, so citing the best chunk's page alone would send the reader to half of it."""
    hits = [
        _hit(BACKOFF, 2, 3.0, header="Retries > Backoff", page_start=2, page_end=2),
        _hit(DEDUP, 4, 2.0, header="Retries > Backoff", page_start=4, page_end=4),
    ]

    (source,) = _sources(hits, sections=1).documents

    (hot,) = source.sections
    assert hot.location == location(DOC, 2, 4, hot.line_start, hot.line_end)


def test_sections_that_score_the_same_sort_by_header() -> None:
    hits = [_hit(OPENING, 1, 2.0, header=h) for h in ("Retries > Zoning", "Retries > Backoff")]

    (source,) = _sources(hits, sections=5).documents

    assert [s.header for s in source.sections] == ["Retries > Backoff", "Retries > Zoning"]


# --- min_cover ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "doc_collections", "expected"),
    [
        ("nothing to cover", {}, []),
        ("one collection holds every document", {"a.md": ["ops"], "b.md": ["ops"]}, ["ops"]),
        (
            "the collection holding the most uncovered documents is picked first",
            {"a.md": ["ops"], "b.md": ["ops"], "c.md": ["backend"]},
            ["ops", "backend"],
        ),
        (
            "two collections are needed and tie, so they are picked by name",
            {"a.md": ["ops"], "b.md": ["backend"]},
            ["backend", "ops"],
        ),
        ("a document in two collections needs only one", {"a.md": ["ops", "backend"]}, ["backend"]),
        (
            "a document no collection holds is left uncovered rather than looped over",
            {"a.md": ["ops"], "b.md": []},
            ["ops"],
        ),
    ],
)
def test_min_cover_is_the_fewest_collections_that_hold_every_document(
    name: str, doc_collections: dict[str, list[str]], expected: list[str]
) -> None:
    assert min_cover(doc_collections) == expected, name


# --- a passage of every cut the chunker makes -------------------------------------------------

RULES = "\n".join(f"- Rule {i} tells the consumer how to retry one failed call." for i in range(12))
LONG = " ".join(f"Sentence {i} explains how the consumer handles a duplicate." for i in range(12))
CODE = "\n".join(f"retry_{i} = backoff(attempt={i}, jitter=True)" for i in range(20))
EVERY_CUT = f"""# Retries

A background job retries a failed HTTP call. The retry has to be idempotent, or the side effect
happens twice, and the consumer has no way to tell the second call from the first.

Exponential backoff with jitter spreads the retries out over time. A fixed delay buys a thundering
herd instead, because every client that failed at once also retries at once.

## Rules

{RULES}

## Long

{LONG}

## Code

```
{CODE}
```
"""


def test_every_chunk_is_quoted_as_it_was_cut() -> None:
    """Whatever the reason a chunk starts or ends where it does, its passage is its own text:
    nothing from the chunk before or after it joins. Snapping to line or sentence ends used to
    add text past a sentence cut, from a neighbour the query never matched. The fixture is
    chunked in two parts, as a PDF is, so a part boundary is one of the cuts."""
    settings = ChunkSettings(chunk_size=300)
    at = EVERY_CUT.index("## Long")
    first, rest = EVERY_CUT[:at], EVERY_CUT[at:]
    chunks = [
        *split(first, settings, end_reason=CutReason.PART),
        *split(
            rest,
            settings,
            line_offset=first.count("\n"),
            char_offset=len(first),
            byte_offset=len(first.encode()),
            opened=open_headings(first),
            start_reason=CutReason.PART,
        ),
    ]
    reasons = {chunk.start_reason for chunk in chunks} | {chunk.end_reason for chunk in chunks}

    passages = [
        quote(_one_chunk(chunk), EVERY_CUT[chunk.char_start : chunk.char_end]) for chunk in chunks
    ]

    assert reasons == set(CutReason), "the fixture makes every cut the chunker knows"
    for chunk, passage in zip(chunks, passages, strict=True):
        cut = f"{chunk.start_reason} -> {chunk.end_reason}"
        assert passage.text == chunk.text.strip(), cut
        assert (passage.char_start, passage.char_end) == (chunk.char_start, chunk.char_end), cut
        assert (passage.line_start, passage.line_end) == (chunk.line_start, chunk.line_end), cut


def _one_chunk(chunk: Chunk) -> HitRange:
    """A chunk of `EVERY_CUT` as the range a search would build from its hit."""
    (hit_range,) = ranges([chunk_hit(chunk, 1, document=DOC, collection=COLLECTION)], how=HARMONIC)
    return hit_range


# --- rejoin ---------------------------------------------------------------------------


def _place(document: str) -> PassageReference:
    """A place folded under a range, as `collapse` lists it."""
    return PassageReference(
        collection=COLLECTION,
        document_id=document,
        document=document,
        seq_start=1,
        seq_end=1,
        header="Retries",
        location=f"{document} L1-1",
        line_start=1,
        line_end=1,
        score=1.0,
        relation=Relation.DUPLICATE,
        similarity=1.0,
        to_parent=CLOSE,
        to_root=CLOSE,
    )


def _part(
    *hits: Hit, aspects: list[str] | None = None, alone: bool = False, place: str = ""
) -> HitRange:
    (found,) = ranges(list(hits), how=HARMONIC)
    return msgspec.structs.replace(
        found, aspects=aspects or [], alone=alone, also_in=[_place(place)] if place else []
    )


@pytest.mark.parametrize(
    ("name", "parts", "expected"),
    [
        (
            "parts that continue each other become one, carrying what each carried",
            [_part(ONE, aspects=["a"], place="x.md"), _part(TWO, aspects=["b", "a"], place="y.md")],
            [((1, 2), ["a", "b"], ["x.md", "y.md"], False)],
        ),
        (
            "parts apart stay apart, each with its own",
            [_part(ONE, aspects=["a"]), _part(THREE, aspects=["b"])],
            [((1, 1), ["a"], [], False), ((3, 3), ["b"], [], False)],
        ),
        (
            "alone only when every part it holds was",
            [_part(ONE, alone=True), _part(TWO, alone=True), _part(FOUR, alone=True)],
            [((1, 2), [], [], True), ((4, 4), [], [], True)],
        ),
        (
            "one part that stands makes the whole stand",
            [_part(ONE, alone=True), _part(TWO)],
            [((1, 2), [], [], False)],
        ),
        (
            "overlapping parts hold a shared chunk once",
            [_part(ONE, TWO, aspects=["a"]), _part(TWO, THREE, aspects=["b"])],
            [((1, 3), ["a", "b"], [], False)],
        ),
        ("nothing to rejoin", [], []),
    ],
)
def test_rejoin_rebuilds_ranges_over_every_part_and_keeps_what_they_carried(
    name: str,
    parts: list[HitRange],
    expected: list[tuple[tuple[int, int], list[str], list[str], bool]],
) -> None:
    rebuilt = rejoin(parts, how=HARMONIC)

    shape = [
        (
            (one.seq_start, one.seq_end),
            one.aspects,
            [place.document for place in one.also_in],
            one.alone,
        )
        for one in sorted(rebuilt, key=lambda one: one.seq_start)
    ]
    assert shape == expected, name


def test_a_shared_chunk_is_held_as_the_first_part_listed_has_it() -> None:
    ranked, unranked = TWO, msgspec.structs.replace(TWO, score=0.0)

    (rebuilt,) = rejoin([_part(ONE, ranked), _part(unranked, THREE)], how=HARMONIC)

    assert [hit.score for hit in rebuilt.hits] == [4.0, 3.0, 2.0]


def test_a_part_is_an_unranked_chunk_with_the_questions_it_answers() -> None:
    found = part(TWO, ["a"])

    assert [(hit.seq, hit.score) for hit in found.hits] == [(2, 0.0)]
    assert (found.aspects, found.alone) == (["a"], False)
    assert part(TWO).aspects == []


# --- rejoin, over many random parts ---------------------------------------------------------


def _random_parts(rng: random.Random) -> list[HitRange]:
    """Runs of consecutive chunks of one section of twelve, some overlapping, some apart, with
    random questions and random `alone` marks."""
    chunks = {
        seq: hit(f"Chunk {seq} says one thing.", 1.0, seq=seq, char_start=seq * 40)
        for seq in range(1, 13)
    }
    parts = []
    for _ in range(rng.randint(1, 6)):
        start = rng.randint(1, 12)
        end = rng.randint(start, min(12, start + 2))
        (found,) = ranges([chunks[seq] for seq in range(start, end + 1)], how=HARMONIC)
        parts.append(
            msgspec.structs.replace(
                found,
                aspects=rng.sample(["a", "b", "c"], rng.randint(0, 2)),
                alone=rng.random() < 0.5,
            )
        )
    return parts


@pytest.mark.parametrize("seed", range(200))
def test_rejoin_keeps_every_chunk_once_and_what_each_part_carried(seed: int) -> None:
    rng = random.Random(seed)
    parts = _random_parts(rng)

    rebuilt = rejoin(parts, how=HARMONIC)

    held = [hit.seq for one in rebuilt for hit in one.hits]
    assert sorted(held) == sorted({hit.seq for one in parts for hit in one.hits}), "each chunk once"
    spans = sorted((one.seq_start, one.seq_end) for one in rebuilt)
    for start, end in spans:
        assert [hit.seq for one in rebuilt if one.seq_start == start for hit in one.hits] == list(
            range(start, end + 1)
        ), "a range is a run of consecutive chunks"
    assert all(after[0] > before[1] + 1 for before, after in zip(spans, spans[1:], strict=False)), (
        "ranges that touch are one range"
    )
    for one in rebuilt:
        inside = [part for part in parts if one.seq_start <= part.seq_start <= one.seq_end]
        assert one.alone == all(part.alone for part in inside)
        assert one.aspects == list(
            dict.fromkeys(label for part in inside for label in part.aspects)
        )
