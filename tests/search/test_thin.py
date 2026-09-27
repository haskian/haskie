"""Passages too short to stand alone: which chunks a thin range may grow into, which it takes,
and when it goes instead.

Every hit below is a real chunk of `MARKDOWN`, cut by the chunker with a 300-character chunk, at
the offsets, lines and cut reasons it really has:

    seq 1  edge -> paragraph             "Three rules follow:" (a lead-in)
    seq 2  paragraph -> length_oversize  the table, first rows
    seq 3  length_oversize -> paragraph  the table, last row
    seq 4  paragraph -> heading          "The rules hold for every consumer." (a section's tail)
    seq 5  heading -> heading            "## Summary", a whole short section
    seq 6  heading -> edge               "## Notes", a whole section
"""

import msgspec
import pytest
from conftest import chunk_hit

from haskie.collection.index import ChunkKey
from haskie.indexing.chunk import split
from haskie.search import thin
from haskie.search.fill import Candidate
from haskie.search.passage import HitRange, ranges
from haskie.settings import ChunkSettings

MARKDOWN = """# Retries

## Rules

Three rules follow:

| rule | why it holds |
|---|---|
| retry with backoff and jitter | it spreads the load so a failed service can recover |
| make every call idempotent | the side effect happens once however often it runs |
| cap the attempts | a call that never succeeds stops costing anything at all |

The rules hold for every consumer.

## Summary

Retries are safe only when calls are idempotent.

## Notes

A retry storm starts when every client retries at the same moment after one shared failure, so the \
load arrives in waves.
"""
DOC = "retries.md"
COLLECTION = "backend"
CHUNKS = split(MARKDOWN, ChunkSettings(chunk_size=300))


HITS = {
    seq: chunk_hit(chunk, seq, document=DOC, collection=COLLECTION)
    for seq, chunk in enumerate(CHUNKS, start=1)
}
# a separator line of another document: a chunk without a word
RULE = msgspec.structs.replace(HITS[5], document="other.md", seq=9, text="---", char_end=3)


def _range(*seqs: int) -> HitRange:
    (found,) = ranges([HITS[seq] for seq in seqs])
    return found


def _key(seq: int) -> ChunkKey:
    return (COLLECTION, DOC, seq)


def test_the_fixture_is_the_chunks_the_docstring_names() -> None:
    reasons = [(str(chunk.start_reason), str(chunk.end_reason)) for chunk in CHUNKS]

    assert reasons == [
        ("edge", "paragraph"),
        ("paragraph", "length_oversize"),
        ("length_oversize", "paragraph"),
        ("paragraph", "heading"),
        ("heading", "heading"),
        ("heading", "edge"),
    ]


# --- around ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "found", "min_chars", "grow", "expected"),
    [
        ("a lead-in: the chunks after it, none before the first", [_range(1)], 300, 2, {2, 3}),
        (
            "a tail: only the side no heading closes",
            [_range(4)],
            300,
            1,
            {3},
        ),
        ("a whole section has nothing to grow into", [_range(5)], 300, 2, set()),
        ("a range long enough is not thin", [_range(2)], 100, 2, set()),
        ("growing turned off", [_range(1)], 300, 0, set()),
        ("the check turned off", [_range(1)], 0, 2, set()),
    ],
)
def test_around_asks_for_the_chunks_a_thin_range_could_take(
    name: str, found: list[HitRange], min_chars: int, grow: int, expected: set[int]
) -> None:
    wanted = thin.around(found, min_chars, grow)

    assert wanted == {_key(seq) for seq in expected}, name


def test_a_range_without_a_word_asks_for_nothing() -> None:
    assert thin.around(ranges([RULE]), 300, 2) == set()


# --- fill -----------------------------------------------------------------------------

MATCH, WEAK = 0.8, -0.4  # values around the scanned hits (`fill.value`)


def _neighbours(values: dict[int, float]) -> dict[ChunkKey, Candidate]:
    return {_key(seq): Candidate(HITS[seq], worth) for seq, worth in values.items()}


@pytest.mark.parametrize(
    ("name", "found", "values", "min_chars", "reach", "expected", "counts"),
    [
        (
            "a lead-in grows into its table",
            [_range(1)],
            {2: MATCH, 3: MATCH},
            300,
            2,
            [(1, 3, False)],
            (1, 0, 0),
        ),
        (
            "growing stops at `reach` chunks",
            [_range(1)],
            {2: MATCH, 3: MATCH},
            300,
            1,
            [(1, 2, False)],
            (1, 0, 0),
        ),
        (
            "a section's tail grows back, never past the heading after it",
            [_range(4)],
            {3: MATCH, 2: MATCH, 5: MATCH},
            300,
            2,
            [(2, 4, False)],
            (1, 0, 0),
        ),
        (
            "both sides grow",
            [_range(3)],
            {2: MATCH, 4: MATCH},
            300,
            1,
            [(2, 4, False)],
            (1, 0, 0),
        ),
        (
            "a weak neighbour joins when a stronger one past it pays for it",
            [_range(1)],
            {2: -0.2, 3: MATCH},
            300,
            2,
            [(1, 3, False)],
            (1, 0, 0),
        ),
        (
            "a weak neighbour is no match: the thin range stays, alone",
            [_range(6), _range(1)],
            {2: WEAK},
            100,
            2,
            [(1, 1, True), (6, 6, False)],
            (0, 1, 0),
        ),
        (
            "a neighbour the read did not return is no match either",
            [_range(6), _range(1)],
            {},
            100,
            2,
            [(1, 1, True), (6, 6, False)],
            (0, 1, 0),
        ),
        (
            "the best range stays, thin, when nothing matches: a short exact answer",
            [_range(1)],
            {2: WEAK},
            300,
            2,
            [(1, 1, False)],
            (0, 0, 0),
        ),
        (
            "a whole section is kept as the author wrote it",
            [_range(2), _range(5)],
            {},
            200,
            2,
            [(2, 2, False), (5, 5, False)],
            (0, 0, 0),
        ),
        (
            "two thin ranges that take the same chunk become one",
            [_range(1), _range(3)],
            {2: MATCH, 4: WEAK},
            300,
            1,
            [(1, 3, False)],
            (2, 0, 0),
        ),
        (
            "a range long enough is left alone",
            [_range(2)],
            {1: MATCH, 3: MATCH},
            100,
            2,
            [(2, 2, False)],
            (0, 0, 0),
        ),
        ("the check turned off", [_range(1)], {2: MATCH}, 0, 2, [(1, 1, False)], (0, 0, 0)),
    ],
)
def test_a_thin_range_grows_by_matching_neighbours_or_goes(
    name: str,
    found: list[HitRange],
    values: dict[int, float],
    min_chars: int,
    reach: int,
    expected: list[tuple[int, int, bool]],
    counts: tuple[int, int, int],
) -> None:
    filled = thin.fill(found, _neighbours(values), min_chars, reach)

    assert [(one.seq_start, one.seq_end, one.alone) for one in filled.ranges] == expected, name
    counted = (filled.grown, sum(one.alone for one in filled.ranges), filled.dropped)
    assert counted == counts, f"{name}: grown, alone, dropped"


def test_a_neighbour_that_joins_leaves_the_range_score_as_it_matched() -> None:
    """The score says how strongly a range matched the query; the chunk it grew by was never
    ranked, so it lifts nothing."""
    found = _range(1)

    (grown,) = thin.fill([found], _neighbours({2: MATCH}), 300, 1).ranges

    assert grown.score == found.score
    assert [(hit.seq, hit.score) for hit in grown.hits] == [(1, 1.0), (2, 0.0)]


def test_a_range_without_a_word_goes_even_when_it_ranked_first() -> None:
    filled = thin.fill([*ranges([RULE]), _range(6)], {}, 300, 2)

    assert [(one.hits[0].document, one.seq_start) for one in filled.ranges] == [(DOC, 6)]
    assert filled.dropped == 1


# --- words, when there is neither a reranker nor a vector ------------------------------


@pytest.mark.parametrize(
    ("name", "question", "text", "expected"),
    [
        ("every topic word found", "Why are retries idempotent?", MARKDOWN, 1.0),
        ("half of them", "Which retries are idempotent jitter storms?", "idempotent jitter", 0.5),
        ("none", "How should retries stay idempotent?", "Summary", 0.0),
        ("a question of stopwords alone says nothing", "What should they do?", MARKDOWN, 0.0),
    ],
)
def test_overlap_is_the_share_of_the_question_words_a_text_holds(
    name: str, question: str, text: str, expected: float
) -> None:
    assert thin.overlap(thin.terms(question), set(thin.terms(text))) == pytest.approx(expected), (
        name
    )


def test_terms_drop_question_words_and_short_ones() -> None:
    assert thin.terms("How should an Order be retried, and the order kept?") == [
        "order",
        "retried",
        "kept",
    ], "once each, in the order they come"
