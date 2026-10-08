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
from haskie.search import section_map
from haskie.search.passage import top_documents
from haskie.search.section import Placement
from haskie.search.section_map import Candidate
from haskie.sections.descriptors import Description
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


def _placements() -> list[Placement]:
    return [
        Placement(
            seq,
            tuple(chunk.headings),
            chunk.char_start,
            chunk.char_end,
            chunk.line_start,
            chunk.line_end,
            None,
            None,
            ("doc", "sagas", chunk.headings[-1].lower()),
        )
        for seq, chunk in enumerate(CHUNKS, start=1)
    ]


def _hit(seq: int, score: float, collection: str = "notes") -> Hit:
    return chunk_hit(CHUNKS[seq - 1], seq, score, document=DOC, collection=collection)


def test_the_fixture_is_cut_as_the_module_says() -> None:
    assert [tuple(chunk.headings) for chunk in CHUNKS] == [CHOREOGRAPHY] * 2 + [ORCHESTRATION] * 2


# --- candidates -------------------------------------------------------------------------


def test_candidates_group_hits_by_section_in_the_order_their_best_hit_ranks() -> None:
    hits = [_hit(3, 0.5), _hit(1, 0.4), _hit(4, 0.3)]

    candidates = section_map.candidates(
        hits, {("notes", DOC): _placements()}, 12_000, ScoreFold.SUM
    )

    assert [one.path for one in candidates] == [ORCHESTRATION, CHOREOGRAPHY]
    orchestration = candidates[0]
    assert orchestration.at == [0, 2], "positions in the scan, best first"
    assert orchestration.score == pytest.approx(0.8), "folded by the search's rule"
    assert [one.seq for one in orchestration.placements] == [3, 4], "every chunk of it"
    assert orchestration.id == "orchestration", "the id its chunks name at its depth"


def test_candidates_count_one_span_once_across_collections() -> None:
    """One document in two collections: its section is one section, credited to the collection
    whose hit ranked first."""
    hits = [_hit(1, 0.5, "alpha"), _hit(2, 0.4, "beta")]
    placed = {("alpha", DOC): _placements(), ("beta", DOC): _placements()}

    (only,) = section_map.candidates(hits, placed, 12_000, ScoreFold.SUM)

    assert (only.collection, only.path, only.at) == ("alpha", CHOREOGRAPHY, [0, 1])


# --- nearness ---------------------------------------------------------------------


def _candidate(document: str, at: list[int], score: float) -> Candidate:
    return Candidate("notes", document, document, (document,), f"id-{document}", [], at, score)


def test_nearness_is_the_nearest_chunk_clipped_at_zero() -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    candidates = [_candidate("a", [0], 1.0), _candidate("b", [1, 2], 1.0)]

    near = section_map.nearness(vectors, None, candidates)

    assert near.shape == (3, 2)
    assert near[0, 0] == pytest.approx(1.0), "a section covers its own chunk wholly"
    assert near[0, 1] == pytest.approx(0.0), "orthogonal, and the opposite chunk clips to 0"
    assert near[2, 1] == pytest.approx(1.0)


def test_nearness_centres_on_the_corpus_mean() -> None:
    """Two chunks sharing the corpus's common direction look alike raw; centred, only what sets
    them apart is left."""
    vectors = [[1.0, 0.2, 0.0], [1.0, 0.0, 0.2]]
    candidates = [_candidate("a", [0], 1.0), _candidate("b", [1], 1.0)]
    centre = np.asarray([0.95, 0.1, 0.1])

    raw = section_map.nearness(vectors, None, candidates)[0, 1]
    centred = section_map.nearness(vectors, centre, candidates)[0, 1]

    assert raw > 0.9 and centred == 0.0, "they part ways once the common part is gone"


# --- cover ------------------------------------------------------------------------

# Two topics as two axes: a section is one chunk of one of them, or a near copy of one.
X, Y = [1.0, 0.0], [0.0, 1.0]


def _cover(
    vectors: list[list[float]], candidates: list[Candidate], weights: list[float], k: int
) -> section_map.Picked:
    near = section_map.nearness(vectors, None, candidates)
    return section_map.cover(np.asarray(weights), near, candidates, k)


def test_cover_starts_with_the_most_relevant_section() -> None:
    candidates = [_candidate("a", [0], 0.2), _candidate("b", [1], 0.9)]

    assert _cover([X, Y], candidates, [0.2, 0.9], 1).picks == [1]


def test_cover_takes_the_second_topic_over_a_near_copy_and_lists_the_copy_under_the_first() -> None:
    vectors = [X, [0.99, 0.05], Y]
    candidates = [_candidate("a", [0], 0.9), _candidate("b", [1], 0.8), _candidate("c", [2], 0.3)]

    picked = _cover(vectors, candidates, [0.9, 0.8, 0.3], 2)

    assert picked.picks == [0, 2], "replication's one weak chunk, not a second saga"
    assert [at for at, _ in picked.related[0]] == [1], "the copy is related to what it repeats"
    assert picked.related[0][0][1] > 0.99
    assert picked.related[2] == []
    assert picked.coverage[-1] == pytest.approx(1.0, abs=0.01), "every chunk is covered"
    assert picked.coverage == sorted(picked.coverage), "coverage only grows"


def test_cover_stops_when_nothing_left_covers_anything_new() -> None:
    candidates = [_candidate("a", [0], 0.9), _candidate("b", [1], 0.8)]

    picked = _cover([X, X], candidates, [0.9, 0.8], 5)

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
    candidates = [
        _candidate("book", [0], 0.9),
        _candidate("book", [1], 0.8),
        _candidate("book", [2], 0.7),
        _candidate("other", [3], other),
    ]

    picked = _cover(vectors, candidates, weights, 3)

    assert (picked.picks, picked.lifted) == (expected, lifted), name


def test_cover_edges() -> None:
    assert section_map.cover(np.zeros(0), np.zeros((0, 0)), [], 5) == section_map.Picked(
        picks=[], related={}, coverage=[], lifted=0
    ), "an empty scan picks nothing"
    one = _cover([X], [_candidate("a", [0], 0.0)], [0.0], 3)
    assert one.picks == [0], "no relevance at all: every chunk weighs the same"


# --- by rank ----------------------------------------------------------------------

LONG = "the saga coordinates compensating steps across services when one step fails"


def test_by_rank_goes_by_relevance_holds_the_cap_and_folds_a_repeat() -> None:
    texts = [LONG, LONG, "replication ships the leader log to every follower in order", "quorum"]
    candidates = [
        _candidate("book", [0], 0.9),
        _candidate("copy", [1], 0.8),
        _candidate("book", [2], 0.7),
        _candidate("book", [3], 0.6),
    ]

    picked = section_map.by_rank(texts, candidates, 3)

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
    candidates = [
        _candidate("book", [0], 0.9),
        _candidate("book", [1], 0.8),
        _candidate("book", [2], 0.7),
        _candidate("other", [3], 0.1),
    ]

    picked = section_map.by_rank(texts, candidates, 4)

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
    candidates = [_candidate("a", [0], 0.9), _candidate("b", [1], 0.8)]

    assert section_map.by_rank(texts, candidates, 2).picks == [0, 1], name


# --- mapped -------------------------------------------------------------------------


def test_mapped_cites_each_pick_with_its_descriptors() -> None:
    hits = [_hit(3, 0.5), _hit(1, 0.4)]
    candidates = section_map.candidates(
        hits, {("notes", DOC): _placements()}, 12_000, ScoreFold.SUM
    )
    orchestration = candidates[0].placements
    described = {
        ("notes", "orchestration"): Description(
            ["orchestrator", "compensation"], "Explains compensation."
        ),
        ("beta", "choreography"): Description(
            ["events"], "Covers events."
        ),  # the same id in another collection's chunking
    }
    picked = section_map.Picked(picks=[0, 1], related={0: [(1, 0.4)], 1: []}, coverage=[], lifted=0)

    first, second = section_map.mapped(candidates, picked, described)

    assert (first.id, first.header, first.depth, first.seq_start, first.seq_end) == (
        "orchestration",
        "Sagas > Orchestration",
        2,
        3,
        4,
    )
    assert first.location == f"{DOC} L{orchestration[0].line_start}-{orchestration[-1].line_end}"
    assert first.chars == orchestration[-1].char_end - orchestration[0].char_start
    assert (first.chunks, first.score) == (1, 0.5)
    assert first.descriptors == ["orchestrator", "compensation"]
    assert first.description == "Explains compensation."
    assert second.description == ""
    assert [one.header for one in first.related] == ["Sagas > Choreography"]
    assert first.related[0].similarity == 0.4
    assert second.descriptors == [], "described in another collection only: nothing to say"


def test_by_rank_stops_at_k() -> None:
    candidates = [_candidate("a", [0], 0.9), _candidate("b", [1], 0.8)]

    picked = section_map.by_rank([LONG, "quorum"], candidates, 1)

    assert (picked.picks, picked.lifted) == ([0], 0), "the second waits for a slot never freed"


# --- documents ------------------------------------------------------------------------


def _documents(
    hits: list[Hit],
    held: dict[str, list[str]] | None = None,
    about: dict[str, str] | None = None,
    sections: list[section_map.MappedSection] | None = None,
) -> list[section_map.MappedDocument]:
    groups = top_documents(hits, ScoreFold.SUM)
    return section_map.documents(groups, sections or [], held or {}, about or {}, ScoreFold.SUM)


def _picked(document_id: str) -> section_map.MappedSection:
    """A map section of `document_id`, for counting a document's picks."""
    return section_map.MappedSection(
        collection="notes",
        document_id=document_id,
        document=document_id,
        header="Sagas",
        id="s",
        location=f"{document_id} L1-9",
        line_start=1,
        line_end=9,
        score=1.0,
        depth=1,
        seq_start=1,
        seq_end=2,
        chars=100,
        chunks=1,
        descriptors=[],
    )


def test_a_document_is_one_row_over_every_chunk_it_matched() -> None:
    """Its best chunk names it and its files; its score folds every matched chunk of it."""
    hits = [_hit(3, 0.5), _hit(1, 0.4), _hit(2, 0.25)]

    (one,) = _documents(hits, about={DOC: "Sagas in microservices"}, sections=[_picked(DOC)])

    assert (one.document_id, one.document, one.chunks) == (DOC, DOC, 3)
    assert one.score == pytest.approx(1.15), "the sum of its chunks' scores"
    assert one.description == "Sagas in microservices"
    assert one.sections == 1, "the map picked one section of it"
    assert (one.markdown_file, one.source_file) == (hits[0].markdown_file, hits[0].source_file)
    assert one.collections == ["notes"], "nothing looked up, so the collection that matched it"


def _other(seq: int, score: float, collection: str = "notes") -> Hit:
    """A chunk of another document."""
    return chunk_hit(CHUNKS[seq - 1], seq, score, document="raft.md", collection=collection)


@pytest.mark.parametrize(
    ("name", "hits", "held", "expected"),
    [
        ("no hits, no documents", [], {}, []),
        (
            "more matched chunks outrank one stronger chunk, by the sum",
            [_hit(1, 0.9), *(_other(seq, 0.4) for seq in (1, 2, 3))],
            {},
            [("raft.md", ["notes"]), (DOC, ["notes"])],
        ),
        (
            "a document in two collections is one row naming both",
            [_hit(1, 0.9), _hit(1, 0.9, collection="archive")],
            {DOC: ["archive", "notes"]},
            [(DOC, ["archive", "notes"])],
        ),
        (
            "a document whose memberships were not read keeps the collection that matched it",
            [_other(1, 0.9, collection="ops")],
            {DOC: ["notes"]},
            [("raft.md", ["ops"])],
        ),
    ],
)
def test_documents_rank_by_every_chunk_the_search_read(
    name: str,
    hits: list[Hit],
    held: dict[str, list[str]],
    expected: list[tuple[str, list[str]]],
) -> None:
    found = _documents(hits, held=held)

    assert [(one.document, one.collections) for one in found] == expected, name


@pytest.mark.parametrize(
    ("name", "ranked", "mapped", "expected"),
    [
        ("nothing reached", 0, set(), []),
        ("fewer than the shortlist: all of them", 3, set(), ["d0", "d1", "d2"]),
        ("past the shortlist: cut", 12, set(), [f"d{n}" for n in range(10)]),
        (
            "a mapped document past the shortlist keeps its row, in its place",
            12,
            {"d11", "d2"},
            [*(f"d{n}" for n in range(10)), "d11"],
        ),
    ],
)
def test_a_map_lists_its_shortlist_and_every_document_a_section_is_in(
    name: str, ranked: int, mapped: set[str], expected: list[str]
) -> None:
    groups = [[_hit(1, 1.0 / (n + 1))] for n in range(ranked)]
    for n, group in enumerate(groups):
        group[0] = msgspec.structs.replace(group[0], document_id=f"d{n}", document=f"d{n}")

    found = section_map.listed(groups, mapped)

    assert [group[0].document_id for group in found] == expected, name
