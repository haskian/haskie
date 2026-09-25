"""Passages out of chunks: folding hits into ranges, widening a range to boundaries a reader would
stop at, and folding the same hits into the documents and collections that cover a query.

Every offset below is a real offset into `MARKDOWN`: the fixture builds a `Hit` from a pair of
snippets and reads its text, lines and char range out of the document, the way the index does.
"""

import msgspec
import pytest

from haskie.collection.index import Hit, Overlap, Overlaps, Relation, location
from haskie.search.passage import (
    MAX_WIDEN,
    Excerpt,
    HitRange,
    Passage,
    PassageReference,
    Sources,
    Window,
    fold_sources,
    harmonic,
    min_cover,
    ranges,
    top_documents,
    widen,
)

# A long line with no sentence terminator and no newline in it: what the widening has nothing to
# stop at, so the cap is all that bounds it.
RUN = ", ".join(f"service-{i:03d}" for i in range(80))

# A long line that does offer sentences: past the cap in both directions, so the widening has to
# fall back from the newline rule to the outermost whole sentences it can reach.
SENTENCE_TAIL = "explains how a consumer handles a duplicate message."
SENTENCES = " ".join(f"Sentence {i} {SENTENCE_TAIL}" for i in range(12))

MARKDOWN = f"""# Retries

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

## Backpressure

{RUN}

## Long line

{SENTENCES}
"""

# What `widen` is handed: `retrieval` reads a window around the range, and the whole fixture is
# one such window that happens to start at the beginning of the document.
WHOLE = Window(text=MARKDOWN, char_start=0)

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
    """One indexed chunk of `MARKDOWN`, with the lines and the text its offsets really give."""
    char_start, char_end = span
    line_start = MARKDOWN.count("\n", 0, char_start) + 1
    line_end = MARKDOWN.count("\n", 0, char_end - 1) + 1
    return Hit(
        collection=collection,
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
# past the end of 2 (the splitter dropped nothing, the query simply skipped a section), 4
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
            "spans that score the same sort by document, then by position",
            [_hit(SKEW, 3, 4.0), ONE, OTHER_ONE],
            [("ops", OTHER, 1, 1), (COLLECTION, DOC, 1, 1), (COLLECTION, DOC, 3, 3)],
        ),
    ],
)
def test_ranges_folds_consecutive_chunks_of_one_document(
    name: str, hits: list[Hit], expected: list[tuple[str, str, int, int]]
) -> None:
    folded = ranges(hits)

    shape = [(r.hits[0].collection, r.hits[0].document, r.seq_start, r.seq_end) for r in folded]
    assert shape == expected, name


def test_a_range_carries_the_span_and_the_score_of_its_members() -> None:
    """The span is the union of its chunks, and the score is the document rule applied to one
    span: one strong chunk lifted by what sits next to it."""
    (folded,) = ranges([ONE, TWO])

    assert (folded.char_start, folded.char_end) == (ONE.char_start, TWO.char_end)
    assert (folded.line_start, folded.line_end) == (ONE.line_start, TWO.line_end)
    assert folded.hits == [ONE, TWO], "the members, ascending, for a caller that wants them"
    assert folded.score == pytest.approx(2 * 4.0 * 7.0 / 11.0), "harmonic(best 4, sum 7)"


# --- widen ---------------------------------------------------------------------------


def _range(char_start: int, char_end: int, **fields) -> HitRange:
    """A range of one chunk over `[char_start, char_end)`, as `ranges` would build it."""
    return ranges([_hit((char_start, char_end), 1, 2.0, **fields)])[0]


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
    (hit_range,) = ranges(hits)

    widened = widen(hit_range, WHOLE, Passage)

    assert (hit_range.page_start, hit_range.page_end) == expected, f"{name}: set once, on the range"
    assert (widened.page_start, widened.page_end) == expected, name
    cited = "" if expected[0] is None else f" p.{expected[0]}-{expected[1]} "
    assert cited in widened.location, f"{name}: the citation names the same pages"


# a place measured close to the passage it folded into, in words and in its vectors
CLOSE = Overlaps(
    words=Overlap(contained=0.9, contains=0.6, alike=0.55, score=harmonic(0.9, 0.6)),
    embedding=Overlap(contained=0.97, contains=0.93, alike=0.95, score=harmonic(0.97, 0.93)),
    chars=None,
)


def test_widening_a_range_keeps_what_was_folded_into_it() -> None:
    """The pointers are decided on the range (`collapse`), before anything is read, and the
    passage is what the caller sees them on."""
    folded = PassageReference(
        collection="ops",
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
    (hit_range,) = ranges([ONE, TWO])
    hit_range = msgspec.structs.replace(hit_range, also_in=[folded])

    widened = widen(hit_range, WHOLE, Passage)

    assert widened.also_in == [folded]


@pytest.mark.parametrize(
    ("name", "span", "expected", "lines"),
    [
        (
            "a newline bounds the passage: the line the span sits on",
            _span("has to be idempotent", "or the side"),
            _text("A background job", "or the side"),
            (3, 3),
        ),
        (
            "a heading is a line of its own, so it is never widened into",
            _span("drift apart", "by milliseconds"),
            "Hosts drift apart by milliseconds.",
            (15, 15),
        ),
        (
            "the line before a heading stops at itself",
            _span("a wall clock", "clock for ordering"),
            "Never trust a wall clock for ordering",
            (13, 13),
        ),
        (
            "the start of the file is a boundary of its own",
            (0, len("# Retries")),
            "# Retries",
            (1, 1),
        ),
        (
            "a line longer than the cap falls back to the outermost whole sentences",
            (_at("Sentence 5 explains"), _at("Sentence 5 explains") + 30),
            _text("Sentence 1 explains", f"Sentence 9 {SENTENCE_TAIL}"),
            (27, 27),
        ),
        (
            "neither a newline nor a sentence end in range: the cap is all there is",
            (_at(RUN) + 400, _at(RUN) + 450),
            MARKDOWN[_at(RUN) + 400 - MAX_WIDEN : _at(RUN) + 450 + MAX_WIDEN].strip(),
            (23, 23),
        ),
        (
            "the end of the file stops the widening",
            (len(MARKDOWN) - 40, len(MARKDOWN)),
            _text("Sentence 7 explains", f"Sentence 11 {SENTENCE_TAIL}"),
            (27, 27),
        ),
    ],
)
def test_expand_widens_a_range_to_the_nearest_boundary(
    name: str, span: tuple[int, int], expected: str, lines: tuple[int, int]
) -> None:
    passage = widen(_range(*span), WHOLE, Passage)

    assert passage.text == expected, name
    assert (passage.line_start, passage.line_end) == lines, f"{name}: lines recounted"
    assert MARKDOWN[passage.char_start : passage.char_end] == expected, f"{name}: offsets agree"
    assert not passage.text[:1].isspace() and not passage.text[-1:].isspace(), name


def test_a_passage_across_a_page_break_carries_no_page_marker() -> None:
    """A page marker is the converter's, not the document's: a passage spanning one reads without
    it, as a chunk does, while its offsets still cut the source the file holds."""
    markdown = "# Retries\n\nThe retry waits.\n\n<!-- page 2 -->\n\nThen it runs again.\n"
    span = (markdown.index("The retry"), markdown.index("again.") + len("again."))
    passage = widen(_range(*span), Window(text=markdown, char_start=0), Passage)
    assert "<!--" not in passage.text
    assert passage.text == "The retry waits.\n\nThen it runs again."
    assert "<!-- page 2 -->" in markdown[passage.char_start : passage.char_end], "the source's"


@pytest.mark.parametrize(
    ("name", "span", "before"),
    [
        ("a window opening mid-line", _span("The consumer keys", "idempotency key."), 200),
        (
            "a window opening exactly at the widened start",
            _span("Hosts drift", "milliseconds."),
            35,
        ),
        ("a window with nothing to spare after it", _span("### Skew", "milliseconds."), 500),
    ],
)
def test_expand_reports_document_offsets_from_a_window(
    name: str, span: tuple[int, int], before: int
) -> None:
    """A search reads a few hundred bytes around the range, not the document, so `widen` works in
    window coordinates and has to hand back offsets and lines of the document itself."""
    start = max(0, span[0] - before)
    window = Window(text=MARKDOWN[start : span[1] + before], char_start=start)
    folded = _range(*span)

    passage = widen(folded, window, Passage)

    whole = widen(folded, WHOLE, Passage)
    assert (passage.char_start, passage.char_end) == (whole.char_start, whole.char_end), name
    assert (passage.line_start, passage.line_end) == (whole.line_start, whole.line_end), name
    assert passage.text == whole.text, name
    assert MARKDOWN[passage.char_start : passage.char_end] == passage.text, name


def test_expand_carries_the_citation_of_the_best_chunk_over_the_widened_lines() -> None:
    """A passage is cited the way a chunk is: `header` from the chunk that ranked it, `location`
    over the lines it ended up covering and the pages every one of its chunks is on."""
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

    passage = widen(ranges(hits)[0], WHOLE, Passage)

    assert passage.header == "Retries > Backoff", "the best-scoring chunk names the passage"
    assert (passage.page_start, passage.page_end) == (1, 3), "both chunks' pages, not the best's"
    assert passage.location == f"{DOC} p.1-3 L1-4", "rebuilt over the passage's own lines"
    assert (passage.seq_start, passage.seq_end) == (1, 2)
    assert passage.collection == COLLECTION
    assert passage.markdown_file == f"/home/documents/{DOC}.md"
    assert passage.score == pytest.approx(2 * 5.0 * 6.0 / 11.0)
    assert passage.text.startswith("# Retries") and passage.text.endswith("twice.")


def test_an_excerpt_is_a_passage() -> None:
    """The trimming step is not written yet, so the type exists, the shape is the passage's, and
    `widen` builds whichever of the two the caller asked for."""
    span = _range(*OPENING)
    passage = widen(span, WHOLE, Passage)

    excerpt = widen(span, WHOLE, Excerpt)

    assert isinstance(excerpt, Excerpt) and isinstance(excerpt, Passage)
    assert excerpt.text == passage.text and excerpt.location == passage.location


# --- fold_sources ---------------------------------------------------------------------


def _sources(hits: list[Hit], memberships=None, limit: int = 10, sections: int = 3) -> Sources:
    return fold_sources(top_documents(hits, limit), memberships or {}, sections)


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
