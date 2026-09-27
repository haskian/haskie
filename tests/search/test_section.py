"""Passages grouped by the section they sit in: which section a chunk is grouped in, how the
sections take the slots, and how one section is written out as one excerpt.

Every chunk below is a real chunk of `MARKDOWN`, cut by the chunker at 150 characters with no
merging of short paragraphs, so each paragraph is its own chunk:

    seq 1  Guide                      the introduction, under the title alone
    seq 2  Guide > Storage > Tables
    seq 3  Guide > Storage > Indexes
    seq 4  Guide > Search > Ranking   three paragraphs
    seq 5  Guide > Search > Ranking
    seq 6  Guide > Search > Ranking
    seq 7  Guide > Search > Folding
"""

import msgspec
import pytest
from conftest import chunk_hit

from haskie.collection.index import Hit
from haskie.indexing.chunk import split
from haskie.search import section
from haskie.search.passage import HitRange, ranges
from haskie.search.section import Entry, Section
from haskie.settings import ChunkSettings, ScoreFold

HARMONIC = ScoreFold.HARMONIC  # the rule these cases were written against


MARKDOWN = """# Guide

The guide explains how a search reads a collection, from storage to the answer it returns.

## Storage

### Tables

One table holds every chunk of a document, with its vector and the offsets it was cut at.

### Indexes

A full-text index reads the framed text, so the words of a heading find every chunk under it.

## Search

### Ranking

The ranking fuses the vector and the full-text lists by rank, since their scores do not compare.

A reranker may rescore the fused list, reading the question and each chunk together as one pair.

The scan goes deeper than the answer, because several chunks fold into one passage and repeats go.

### Folding

A near-duplicate folds under the passage it repeats, and its slot goes to the next one down.
"""
DOC = "guide.md"
COLLECTION = "notes"
CHUNKS = split(MARKDOWN, ChunkSettings(chunk_size=150, chunk_merge_below=0))
OUTLINE = [
    Entry(seq, tuple(chunk.headings), chunk.char_start, chunk.char_end)
    for seq, chunk in enumerate(CHUNKS, start=1)
]
OUTLINES = {(COLLECTION, DOC): OUTLINE}
GUIDE, STORAGE, SEARCH = ("Guide",), ("Guide", "Storage"), ("Guide", "Search")


def _hit(seq: int, score: float = 1.0) -> Hit:
    return chunk_hit(CHUNKS[seq - 1], seq, score, document=DOC, collection=COLLECTION)


def _range(
    seq: int, score: float = 1.0, aspects: list[str] | None = None, doc: str = DOC
) -> HitRange:
    (found,) = ranges([msgspec.structs.replace(_hit(seq, score), document=doc)], how=HARMONIC)
    return msgspec.structs.replace(found, aspects=aspects or [])


def _text(seq: int) -> str:
    return CHUNKS[seq - 1].text


def test_the_fixture_is_the_outline_the_docstring_names() -> None:
    assert [entry.headings for entry in OUTLINE] == [
        GUIDE,
        (*STORAGE, "Tables"),
        (*STORAGE, "Indexes"),
        (*SEARCH, "Ranking"),
        (*SEARCH, "Ranking"),
        (*SEARCH, "Ranking"),
        (*SEARCH, "Folding"),
    ]


# --- outlines -------------------------------------------------------------------------


def test_outlines_are_read_per_document_and_ordered_by_seq() -> None:
    rows = [
        ("notes", {"document": DOC, "seq": 2, "headings": ["A"], "char_start": 5, "char_end": 9}),
        ("notes", {"document": DOC, "seq": 1, "headings": None, "char_start": 0, "char_end": 4}),
        ("other", {"document": DOC, "seq": 1, "headings": ["B"], "char_start": 0, "char_end": 3}),
    ]

    found = section.outlines(rows)

    assert found == {
        ("notes", DOC): [Entry(1, (), 0, 4), Entry(2, ("A",), 5, 9)],
        ("other", DOC): [Entry(1, ("B",), 0, 3)],
    }


# --- section_of -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "outline", "seq", "max_chars", "expected"),
    [
        (
            "a title over the whole document says nothing: the level below it is taken",
            OUTLINE,
            2,
            1000,
            Section(STORAGE, 2, 3),
        ),
        (
            "a section over the size splits one heading down",
            OUTLINE,
            2,
            150,
            Section((*STORAGE, "Tables"), 2, 2),
        ),
        (
            "the first level that fits, several levels down",
            OUTLINE,
            5,
            300,
            Section((*SEARCH, "Ranking"), 4, 6),
        ),
        (
            "the deepest heading is taken however large its section is",
            OUTLINE,
            5,
            50,
            Section((*SEARCH, "Ranking"), 4, 6),
        ),
        (
            "a chunk under the title alone groups in the whole document",
            OUTLINE,
            1,
            100,
            Section(GUIDE, 1, 7),
        ),
        (
            "a document without headings is one section",
            [Entry(1, (), 0, 90), Entry(2, (), 92, 180)],
            2,
            100,
            Section((), 1, 2),
        ),
    ],
)
def test_a_chunk_groups_in_the_largest_section_that_fits(
    name: str, outline: list[Entry], seq: int, max_chars: int, expected: Section
) -> None:
    assert section.section_of(outline, seq, max_chars) == expected, name


def test_a_chunk_its_outline_does_not_hold_is_a_broken_search() -> None:
    """The outline is read through the table the hits came from, so it holds every chunk a hit
    names: a missing one is not guessed at."""
    with pytest.raises(LookupError, match="chunk 99"):
        section.section_of(OUTLINE, 99, 1000)


# --- documents ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "found", "limit", "expected"),
    [
        (
            "the first documents in rank order, as many as the slots",
            [_range(4), _range(2), _range(6, doc="b.md"), _range(1, doc="c.md")],
            2,
            {(COLLECTION, DOC), (COLLECTION, "b.md")},
        ),
        (
            "a document of a short range is read too, up to the last document the slots need",
            [msgspec.structs.replace(_range(4, doc="b.md"), alone=True), _range(2)],
            1,
            {(COLLECTION, "b.md"), (COLLECTION, DOC)},
        ),
        (
            "none past it",
            [_range(2), msgspec.structs.replace(_range(4, doc="b.md"), alone=True)],
            1,
            {(COLLECTION, DOC)},
        ),
        ("no ranges, no documents", [], 3, set()),
    ],
)
def test_only_the_documents_a_section_can_open_in_are_read(
    name: str, found: list[HitRange], limit: int, expected: set[tuple[str, str]]
) -> None:
    assert section.documents(found, limit) == expected, name


def test_the_documents_read_hold_the_sections_a_short_range_opens() -> None:
    """A section opens on its first range even when it is short, and its standing one ranks later:
    Search opens second, on a short passage, though its standing one comes after another
    document's. Reading only where standing ranges are would lose it for that document."""
    found = [
        _range(2),
        msgspec.structs.replace(_range(4, doc="b.md"), alone=True),
        _range(2, doc="c.md"),
        _range(6, doc="b.md"),
    ]
    read = section.documents(found, 2)
    outlines = {place: OUTLINE for place in read}  # every document cut alike

    groups = section.group(found, outlines, 1000, 2)

    assert [(one.document, one.section.path) for one in groups] == [
        (DOC, STORAGE),
        ("b.md", SEARCH),
    ]


# --- group ----------------------------------------------------------------------------

# best first: a Search passage, then a Storage one, then the rest of both sections
BEST_FIRST = [_range(4, 5.0), _range(2, 4.0), _range(6, 3.0), _range(7, 2.0), _range(3, 1.0)]


@pytest.mark.parametrize(
    ("name", "found", "outlines", "limit", "expected"),
    [
        (
            "each section takes the place of its best passage and holds every one of its own",
            BEST_FIRST,
            OUTLINES,
            2,
            [(SEARCH, [4, 6, 7]), (STORAGE, [2, 3])],
        ),
        (
            "the slots count sections: past the limit none opens, the open ones still fill",
            BEST_FIRST,
            OUTLINES,
            1,
            [(SEARCH, [4, 6, 7])],
        ),
        (
            "a passage of a document whose outline was not read belongs to no kept section",
            [_range(4), _range(6, doc="b.md")],
            OUTLINES,
            5,
            [(SEARCH, [4])],
        ),
        (
            "a section of passages too short to stand alone is none, and frees its slot",
            [_range(4), msgspec.structs.replace(_range(2), alone=True), _range(7)],
            OUTLINES,
            2,
            [(SEARCH, [4, 7])],
        ),
        (
            "a short passage stays in a section another passage stands in, wherever it ranked",
            [msgspec.structs.replace(_range(6), alone=True), _range(2), _range(4)],
            OUTLINES,
            2,
            [(SEARCH, [6, 4]), (STORAGE, [2])],
        ),
        ("no passages, no sections", [], OUTLINES, 5, []),
    ],
)
def test_passages_group_by_section_and_the_sections_take_the_slots(
    name: str,
    found: list[HitRange],
    outlines: dict,
    limit: int,
    expected: list[tuple[tuple[str, ...], list[int]]],
) -> None:
    groups = section.group(found, outlines, 1000, limit)

    shape = [(one.section.path, [kept.seq_start for kept in one.ranges]) for one in groups]
    assert shape == expected, name


# --- within -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "budget", "expected"),
    [
        ("everything fits", 10_000, 3),
        ("the sections past the budget go, the last first", 300, 2),
        ("a later small section does not jump the queue", 200, 1),
        ("the first stays even over the budget", 10, 1),
    ],
)
def test_the_sections_are_cut_to_the_budget_in_order(name: str, budget: int, expected: int) -> None:
    storage = section.Group(COLLECTION, DOC, Section(STORAGE, 2, 3), [_range(2), _range(3)])
    ranking = section.Group(COLLECTION, DOC, Section(SEARCH, 4, 6), [_range(5)])
    folding = section.Group(COLLECTION, DOC, Section(SEARCH, 7, 7), [_range(7)])
    groups = [ranking, storage, folding]

    assert [one.chars for one in groups] == [96, 89 + 93, 92], "the fixture's passages"
    assert len(section.within(groups, budget)) == expected, name


# --- excerpt --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "seqs", "path", "expected"),
    [
        (
            "sub-headings open the passages under them, and [...] marks a skip",
            [6, 4, 7],
            SEARCH,
            ["### Ranking", _text(4), section.ELISION, _text(6), "### Folding", _text(7)],
        ),
        (
            "two passages a heading apart need no [...]: the heading is all that is between",
            [2, 3],
            STORAGE,
            ["### Tables", _text(2), "### Indexes", _text(3)],
        ),
        (
            "one passage of a section it is the whole of: no heading repeats the header",
            [5],
            (*SEARCH, "Ranking"),
            [_text(5)],
        ),
        (
            "the whole document: every heading of the passage opens it",
            [2],
            (),
            ["# Guide", "## Storage", "### Tables", _text(2)],
        ),
    ],
)
def test_a_section_is_written_out_as_one_excerpt(
    name: str, seqs: list[int], path: tuple[str, ...], expected: list[str]
) -> None:
    found = [_range(seq) for seq in seqs]
    group = section.Group(COLLECTION, DOC, Section(path, 1, 7), found)

    excerpt = section.excerpt(group, [_text(seq) for seq in seqs])

    assert excerpt.text == "\n\n".join(expected), name
    assert [span.seq_start for span in excerpt.spans] == sorted(seqs), f"{name}: document order"


def test_an_excerpt_cites_the_section_and_every_passage_in_it() -> None:
    """The excerpt runs from its first passage to its last; each span keeps its own citation,
    score, folded places and questions."""
    first = _range(4, 1.0, aspects=["b"])
    last = _range(7, 3.0, aspects=["a", "b"])
    group = section.Group(COLLECTION, DOC, Section(SEARCH, 4, 7), [last, first])

    excerpt = section.excerpt(group, [_text(7), _text(4)])

    assert excerpt.header == "Guide > Search"
    assert (excerpt.seq_start, excerpt.seq_end) == (4, 7)
    assert (excerpt.line_start, excerpt.line_end) == (first.line_start, last.line_end)
    assert excerpt.location == f"{DOC} L{first.line_start}-{last.line_end}"
    assert excerpt.score == 3.0, "the best passage's"
    assert excerpt.aspects == ["b", "a"], "every question a passage answers, in document order"
    assert [(span.header, span.score, span.aspects) for span in excerpt.spans] == [
        ("Guide > Search > Ranking", 1.0, ["b"]),
        ("Guide > Search > Folding", 3.0, ["a", "b"]),
    ]
    assert excerpt.markdown_file == f"/home/documents/{DOC}.md"


def test_an_excerpt_carries_no_page_marker() -> None:
    marked = "Before the break.\n\n<!-- page 2 -->\n\nAfter the break."
    group = section.Group(COLLECTION, DOC, Section((*SEARCH, "Ranking"), 4, 6), [_range(5)])

    excerpt = section.excerpt(group, [marked])

    assert excerpt.text == "Before the break.\n\nAfter the break."
