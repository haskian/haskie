"""A map of sections: which sections the scan reached, which the map picks to cover it, what is
related to each pick, and what each is about.

`SAGAS` is a real document cut by the chunker at 200 characters with no merging:

    seq 1  Sagas > Choreography
    seq 2  Sagas > Choreography
    seq 3  Sagas > Orchestration
    seq 4  Sagas > Orchestration
"""

import msgspec
import numpy as np
import pytest
from conftest import chunk_hit

from haskie.collection.index import Hit
from haskie.indexing.chunk import split
from haskie.outline.build import Node
from haskie.search import overview
from haskie.search.overview import Pooled
from haskie.search.section import Entry
from haskie.settings import ChunkSettings, ScoreFold

SAGAS = """# Sagas

## Choreography

In a choreographed saga every service listens for events and emits compensating events.

Choreography keeps services loosely coupled: each compensating event undoes one step.

## Orchestration

An orchestrator tells each service which step to run and which compensation to call.

The orchestrator holds the saga state, so a failed step triggers its compensation.
"""
DOC = "sagas.md"
CHUNKS = split(SAGAS, ChunkSettings(chunk_size=200, chunk_merge_below=0))
CHOREOGRAPHY, ORCHESTRATION = ("Sagas", "Choreography"), ("Sagas", "Orchestration")


def _outline() -> list[Entry]:
    return [
        Entry(
            seq,
            tuple(chunk.headings),
            chunk.char_start,
            chunk.char_end,
            chunk.line_start,
            chunk.line_end,
            None,
            None,
        )
        for seq, chunk in enumerate(CHUNKS, start=1)
    ]


def _hit(seq: int, score: float, collection: str = "notes") -> Hit:
    return chunk_hit(CHUNKS[seq - 1], seq, score, document=DOC, collection=collection)


def test_the_fixture_is_cut_as_the_module_says() -> None:
    assert [tuple(chunk.headings) for chunk in CHUNKS] == [CHOREOGRAPHY] * 2 + [ORCHESTRATION] * 2


# --- pool -------------------------------------------------------------------------


def test_pool_groups_hits_by_section_in_the_order_their_best_hit_ranks() -> None:
    hits = [_hit(3, 0.5), _hit(1, 0.4), _hit(4, 0.3)]

    pooled = overview.pool(hits, {("notes", DOC): _outline()}, 12_000, ScoreFold.SUM)

    assert [one.path for one in pooled] == [ORCHESTRATION, CHOREOGRAPHY]
    orchestration = pooled[0]
    assert orchestration.at == [0, 2], "positions in the scan, best first"
    assert orchestration.score == pytest.approx(0.8), "folded by the search's rule"
    assert [entry.seq for entry in orchestration.entries] == [3, 4], "every chunk of it"


def test_pool_counts_one_span_once_across_collections() -> None:
    """One document in two collections: its section is one section, credited to the collection
    whose hit ranked first."""
    hits = [_hit(1, 0.5, "alpha"), _hit(2, 0.4, "beta")]
    outlines = {("alpha", DOC): _outline(), ("beta", DOC): _outline()}

    (only,) = overview.pool(hits, outlines, 12_000, ScoreFold.SUM)

    assert (only.collection, only.path, only.at) == ("alpha", CHOREOGRAPHY, [0, 1])


# --- nearness ---------------------------------------------------------------------


def _pooled(document: str, at: list[int], score: float) -> Pooled:
    return Pooled("notes", document, document, (document,), [], at, score)


def test_nearness_is_the_nearest_chunk_clipped_at_zero() -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    pooled = [_pooled("a", [0], 1.0), _pooled("b", [1, 2], 1.0)]

    near = overview.nearness(vectors, None, pooled)

    assert near.shape == (3, 2)
    assert near[0, 0] == pytest.approx(1.0), "a section covers its own chunk wholly"
    assert near[0, 1] == pytest.approx(0.0), "orthogonal, and the opposite chunk clips to 0"
    assert near[2, 1] == pytest.approx(1.0)


def test_nearness_centres_on_the_corpus_mean() -> None:
    """Two chunks sharing the corpus's common direction look alike raw; centred, only what sets
    them apart is left."""
    vectors = [[1.0, 0.2, 0.0], [1.0, 0.0, 0.2]]
    pooled = [_pooled("a", [0], 1.0), _pooled("b", [1], 1.0)]
    centre = np.asarray([0.95, 0.1, 0.1])

    raw = overview.nearness(vectors, None, pooled)[0, 1]
    centred = overview.nearness(vectors, centre, pooled)[0, 1]

    assert raw > 0.9 and centred == 0.0, "they part ways once the common part is gone"


# --- cover ------------------------------------------------------------------------

# Two topics as two axes: a section is one chunk of one of them, or a near copy of one.
X, Y = [1.0, 0.0], [0.0, 1.0]


def _cover(
    vectors: list[list[float]], pooled: list[Pooled], weights: list[float], k: int
) -> overview.Picked:
    near = overview.nearness(vectors, None, pooled)
    return overview.cover(np.asarray(weights), near, pooled, k)


def test_cover_starts_with_the_most_relevant_section() -> None:
    pooled = [_pooled("a", [0], 0.2), _pooled("b", [1], 0.9)]

    assert _cover([X, Y], pooled, [0.2, 0.9], 1).picks == [1]


def test_cover_takes_the_second_topic_over_a_near_copy_and_lists_the_copy_under_the_first() -> None:
    vectors = [X, [0.99, 0.05], Y]
    pooled = [_pooled("a", [0], 0.9), _pooled("b", [1], 0.8), _pooled("c", [2], 0.3)]

    picked = _cover(vectors, pooled, [0.9, 0.8, 0.3], 2)

    assert picked.picks == [0, 2], "replication's one weak chunk, not a second saga"
    assert [at for at, _ in picked.related[0]] == [1], "the copy is related to what it repeats"
    assert picked.related[0][0][1] > 0.99
    assert picked.related[2] == []
    assert picked.coverage[-1] == pytest.approx(1.0, abs=0.01), "every chunk is covered"
    assert picked.coverage == sorted(picked.coverage), "coverage only grows"


def test_cover_stops_when_nothing_left_covers_anything_new() -> None:
    pooled = [_pooled("a", [0], 0.9), _pooled("b", [1], 0.8)]

    picked = _cover([X, X], pooled, [0.9, 0.8], 5)

    assert picked.picks == [0], "an exact copy adds nothing: it is related, not picked"
    assert [at for at, _ in picked.related[0]] == [1]


@pytest.mark.parametrize(
    ("name", "other", "expected", "lifted"),
    [
        ("another document's section on the topic takes the slot", 0.6, [0, 1, 3], 0),
        ("one barely on the topic does not: the cap gives way", 0.05, [0, 1, 2], 1),
    ],
)
def test_cover_holds_a_document_to_its_cap_for_a_section_that_covers_enough(
    name: str, other: float, expected: list[int], lifted: int
) -> None:
    """Three sections of one long book, each its own topic, and one of another book, the third
    pick due. The other book's section takes it only when it covers at least `CAP_SHARE` of what
    the long book's third would."""
    vectors = [X, Y, [0.7, -0.7], [-0.7, 0.7]]
    weights = [0.9, 0.8, 0.7, other]
    pooled = [
        _pooled("book", [0], 0.9),
        _pooled("book", [1], 0.8),
        _pooled("book", [2], 0.7),
        _pooled("other", [3], other),
    ]

    picked = _cover(vectors, pooled, weights, 3)

    assert (picked.picks, picked.lifted) == (expected, lifted), name


def test_cover_edges() -> None:
    assert overview.cover(np.zeros(0), np.zeros((0, 0)), [], 5) == overview.Picked(
        picks=[], related={}, coverage=[], lifted=0
    ), "an empty scan picks nothing"
    one = _cover([X], [_pooled("a", [0], 0.0)], [0.0], 3)
    assert one.picks == [0], "no relevance at all: every chunk weighs the same"


# --- by rank ----------------------------------------------------------------------

LONG = "the saga coordinates compensating steps across services when one step fails"


def test_by_rank_goes_by_relevance_holds_the_cap_and_folds_a_repeat() -> None:
    texts = [LONG, LONG, "replication ships the leader log to every follower in order", "quorum"]
    pooled = [
        _pooled("book", [0], 0.9),
        _pooled("copy", [1], 0.8),
        _pooled("book", [2], 0.7),
        _pooled("book", [3], 0.6),
    ]

    picked = overview.by_rank(texts, pooled, 3)

    assert picked.picks == [0, 2, 3], "the copy folds; the cap lifts once nothing else is left"
    assert picked.related[0] == [(1, 1.0)], "the same words: related to what it repeats"
    assert picked.lifted == 1
    assert picked.coverage == [], "no vectors: nothing to measure coverage by"


def test_by_rank_folds_a_section_the_cap_held_back_into_a_later_pick_it_repeats() -> None:
    """The long book's third section waits for the cap; the other book's section is picked, and
    the waiting one turns out to repeat it: related, not picked."""
    replication = "replication ships the leader log to every follower in order it was written"
    texts = [LONG, "a quorum read overlaps a quorum write so it sees the latest one", replication]
    texts.append(replication)
    pooled = [
        _pooled("book", [0], 0.9),
        _pooled("book", [1], 0.8),
        _pooled("book", [2], 0.7),
        _pooled("other", [3], 0.1),
    ]

    picked = overview.by_rank(texts, pooled, 4)

    assert picked.picks == [0, 1, 3]
    assert picked.related[3] == [(2, 1.0)] and picked.lifted == 0


@pytest.mark.parametrize(
    ("name", "texts"),
    [
        ("a candidate too short to judge", [LONG, "quorum"]),
        ("a pick too short to judge", ["quorum", LONG]),
    ],
)
def test_by_rank_never_folds_a_text_too_short_to_judge(name: str, texts: list[str]) -> None:
    pooled = [_pooled("a", [0], 0.9), _pooled("b", [1], 0.8)]

    assert overview.by_rank(texts, pooled, 2).picks == [0, 1], name


# --- distinct and mapped ----------------------------------------------------------


def test_distinct_keeps_the_keywords_that_set_a_pick_apart_on_this_map() -> None:
    found = [
        {"saga": 4, "data": 4, "Compensating Events": 2},
        {"data": 4, "replication": 4},
        {"data": 4, "quorum": 4},
        {},
    ]

    words = overview.distinct(found, [0, 3])

    assert set(words[0][:2]) == {"saga", "Compensating Events"}, "the words only it uses"
    assert "data" in words[0][2:], "shared by every section: last"
    assert words[3] == [], "a section with no outline has none"


def test_mapped_cites_each_pick_with_its_outline_keywords() -> None:
    hits = [_hit(3, 0.5), _hit(1, 0.4)]
    pooled = overview.pool(hits, {("notes", DOC): _outline()}, 12_000, ScoreFold.SUM)
    orchestration = pooled[0].entries
    node = Node(
        headings=list(ORCHESTRATION),
        char_start=orchestration[0].char_start,
        char_end=orchestration[-1].char_end,
        byte_start=0,
        byte_end=0,
        line_start=orchestration[0].line_start,
        line_end=orchestration[-1].line_end,
        page_start=None,
        page_end=None,
        keywords={"orchestrator": 2, "compensation": 2},
    )
    nodes = overview.stored(pooled, {DOC: [node], "other.pdf": [node]})
    picked = overview.Picked(picks=[0, 1], related={0: [(1, 0.4)], 1: []}, coverage=[], lifted=0)

    first, second = overview.mapped(hits, pooled, picked, nodes, {0: ["orchestrator"], 1: []})

    assert (first.header, first.depth, first.seq_start, first.seq_end) == (
        "Sagas > Orchestration",
        2,
        3,
        4,
    )
    assert first.location == f"{DOC} L{orchestration[0].line_start}-{orchestration[-1].line_end}"
    assert first.chars == orchestration[-1].char_end - orchestration[0].char_start
    assert (first.chunks, first.score) == (1, 0.5)
    assert first.keywords == ["orchestrator", "compensation"] and first.distinct == ["orchestrator"]
    assert [one.header for one in first.related] == ["Sagas > Choreography"]
    assert first.related[0].similarity == 0.4
    assert second.keywords == [] and second.distinct == [], "no outline node: nothing to say"


@pytest.mark.parametrize(
    ("name", "nodes", "expected"),
    [
        ("no outline for the document", {}, None),
        ("the node under the same path", {DOC: ["ours"]}, "ours"),
        (
            "a node cut a few characters apart still overlaps",
            {DOC: ["shifted"]},
            "shifted",
        ),
        ("another path, the same span", {DOC: ["elsewhere"]}, None),
        ("the same path, another document", {"other.pdf": ["ours"]}, None),
        ("the same path, no shared character", {DOC: ["later"]}, None),
        ("the path twice: the one overlapping most", {DOC: ["later", "shifted"]}, "shifted"),
    ],
)
def test_stored_finds_each_sections_node(
    name: str, nodes: dict[str, list[str]], expected: str | None
) -> None:
    """The outline is cut from one chunking and the section from its collection's: the node of
    the same path whose span overlaps the section's most."""
    (orchestration,) = [
        one
        for one in overview.pool(
            [_hit(3, 0.5)], {("notes", DOC): _outline()}, 12_000, ScoreFold.SUM
        )
    ]
    start, end = orchestration.entries[0].char_start, orchestration.entries[-1].char_end
    base = Node(
        headings=list(ORCHESTRATION),
        char_start=start,
        char_end=end,
        byte_start=0,
        byte_end=0,
        line_start=1,
        line_end=1,
        page_start=None,
        page_end=None,
        keywords={},
    )
    made = {
        "ours": base,
        "shifted": msgspec.structs.replace(base, char_start=start + 3, char_end=end + 3),
        "elsewhere": msgspec.structs.replace(base, headings=["Sagas"]),
        "later": msgspec.structs.replace(base, char_start=end, char_end=end + 100),
    }

    (found,) = overview.stored(
        [orchestration], {doc: [made[one] for one in named] for doc, named in nodes.items()}
    )

    assert found == (made[expected] if expected else None), name


def test_by_rank_stops_at_k() -> None:
    pooled = [_pooled("a", [0], 0.9), _pooled("b", [1], 0.8)]

    picked = overview.by_rank([LONG, "quorum"], pooled, 1)

    assert (picked.picks, picked.lifted) == ([0], 0), "the second waits for a slot never freed"
