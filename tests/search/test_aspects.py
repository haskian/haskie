"""Several questions in one search: the questions a caller may send, how the parts take turns at
the slots, and which questions each kept passage answers once the near-duplicates have folded.

Every range below is one or more real chunks of a backend book: a sentence at its own offsets in
its own document, numbered the way the index numbers them.
"""

import msgspec
import pytest
from conftest import hit, words_scan

from haskie.errors import InvalidInput
from haskie.indexing.segment import CutReason
from haskie.search import aspects, collapse
from haskie.search.passage import HitRange, PassageReference, ranges

# What each chunk of `patterns.md` and the other documents says, by its `seq`.
SENTENCES = {
    1: "A background job retries a failed HTTP call, so the call has to be idempotent.",
    2: "The consumer keys on an idempotency key and drops a message it has already handled.",
    3: "Exponential backoff with jitter spreads the retries and avoids a thundering herd.",
    4: "Never trust a wall clock for ordering: hosts drift apart by milliseconds.",
    5: "A transactional outbox writes the event in the same transaction as the state change.",
}
RETRY = SENTENCES[1]
# RETRY with a sentence either side: a fuller passage that holds the whole of RETRY
FULLER = f"Retries are where most duplicate side effects come from. {RETRY} Key it on a request id."


def _span(
    document: str,
    *seqs: int,
    collection: str = "backend",
    text: str | None = None,
    score: float = 1.0,
    end: CutReason = CutReason.EDGE,
) -> HitRange:
    """One range over consecutive chunks of one document, 100 characters a chunk, as
    `passage.ranges` merges them. `text` stands in for a one-chunk range's sentence, and `end` is
    the cut after its last chunk."""
    hits = [
        hit(
            text or SENTENCES[seq],
            score,
            document=document,
            collection=collection,
            seq=seq,
            char_start=(seq - 1) * 100,
        )
        for seq in seqs
    ]
    hits[-1] = msgspec.structs.replace(hits[-1], end_reason=end)
    (found,) = ranges(hits)
    return found


def _shape(picks: list[aspects.Pick]) -> list[tuple[str, str, int, int, list[int]]]:
    """What a case asserts: each pick's place, the chunks it covers, and the parts it answers."""
    return [
        (
            pick.span.hits[0].collection,
            pick.span.hits[0].document,
            pick.span.seq_start,
            pick.span.seq_end,
            sorted(pick.questions | pick.ranked_high),
        )
        for pick in picks
    ]


# --- the questions asked --------------------------------------------------------------

AGGREGATE = "Should one aggregate hold a direct object reference to another, or only its identity?"
EVENTS = "How should a change in one aggregate update another without one shared transaction?"
ORDERING = "How do hosts agree on the order of events when their clocks drift apart?"
CONTEXT = "Designing an Order service with DDD; Order and Customer are separate aggregates."


FIVE = [f"{EVENTS} {n}" for n in range(5)]


@pytest.mark.parametrize(
    ("name", "asked", "context", "questions", "queries", "shared"),
    [
        ("one question, searched as it is", [AGGREGATE], None, [AGGREGATE], [AGGREGATE], None),
        (
            "blank parts dropped, repeats kept once, order kept",
            [f"  {AGGREGATE} ", "", EVENTS, AGGREGATE],
            None,
            [AGGREGATE, EVENTS],
            [AGGREGATE, EVENTS],
            None,
        ),
        ("a blank context is none", [AGGREGATE], "   ", [AGGREGATE], [AGGREGATE], None),
        (
            "the context goes in front of every part",
            [AGGREGATE, EVENTS],
            f" {CONTEXT} ",
            [AGGREGATE, EVENTS],
            [f"{CONTEXT}\n\n{AGGREGATE}", f"{CONTEXT}\n\n{EVENTS}"],
            CONTEXT,
        ),
        ("five parts is the most", FIVE, None, FIVE, FIVE, None),
    ],
)
def test_the_questions_a_caller_sends_are_stripped_and_deduplicated(
    name: str,
    asked: list[str],
    context: str | None,
    questions: list[str],
    queries: list[str],
    shared: str | None,
) -> None:
    checked = aspects.questions(asked, context)

    assert checked.questions == questions, name
    assert checked.framed == queries, name
    assert checked.context == shared, name


@pytest.mark.parametrize(
    ("name", "asked", "context", "message"),
    [
        ("no question", [], None, "q must hold 1..5 questions, got 0"),
        ("only blanks", ["  ", ""], None, "got 0"),
        ("six parts", [*FIVE, f"{EVENTS} 5"], None, "got 6"),
        ("a question past 500 characters", ["why " * 126], None, "at most 500 characters"),
        ("a context past 200 characters", [AGGREGATE], "x" * 201, "got 201"),
    ],
)
def test_questions_a_search_cannot_run_are_refused(
    name: str, asked: list[str], context: str | None, message: str
) -> None:
    with pytest.raises(InvalidInput, match=message):
        aspects.questions(asked, context)


@pytest.mark.parametrize(
    ("parts", "limit", "expected"),
    [(3, 10, 4), (2, 10, 5), (5, 5, 1), (1, 25, 25)],
)
def test_each_part_is_owed_its_share_of_the_limit(parts: int, limit: int, expected: int) -> None:
    assert aspects.depth(parts, limit) == expected


def test_a_limit_below_the_number_of_parts_is_refused() -> None:
    with pytest.raises(InvalidInput, match=r"at least the number of questions \(3\), got 2"):
        aspects.depth(3, 2)


# --- the turns ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "ranked", "depth", "cap", "expected"),
    [
        ("no part found anything", [[], []], 2, 10, []),
        (
            "the parts take turns, best first",
            [
                [_span("a.md", 1), _span("a.md", 3)],
                [_span("b.md", 1), _span("b.md", 3)],
            ],
            2,
            10,
            [
                ("backend", "a.md", 1, 1, [0]),
                ("backend", "b.md", 1, 1, [1]),
                ("backend", "a.md", 3, 3, [0]),
                ("backend", "b.md", 3, 3, [1]),
            ],
        ),
        (
            "a part that runs out leaves its turns to the others",
            [
                [_span("a.md", 1)],
                [_span("b.md", 1), _span("b.md", 3), _span("b.md", 5)],
            ],
            2,
            10,
            [
                ("backend", "a.md", 1, 1, [0]),
                ("backend", "b.md", 1, 1, [1]),
                ("backend", "b.md", 3, 3, [1]),
                ("backend", "b.md", 5, 5, [1]),
            ],
        ),
        (
            "a part another's pick answers skips its turn, and its copy joins that pick",
            [
                [_span("shared.md", 1), _span("a.md", 1)],
                [_span("b.md", 1), _span("shared.md", 1)],
            ],
            2,
            10,
            [
                ("backend", "shared.md", 1, 1, [0, 1]),
                ("backend", "a.md", 1, 1, [0]),
                ("backend", "b.md", 1, 1, [1]),
            ],
        ),
        (
            "ranked below its share, an overlap earns no credit and no tag",
            [
                [_span("shared.md", 1)],
                [_span("b.md", 1), _span("b.md", 3), _span("shared.md", 1)],
            ],
            2,
            3,
            [
                ("backend", "shared.md", 1, 1, [0]),
                ("backend", "b.md", 1, 1, [1]),
                ("backend", "b.md", 3, 3, [1]),
            ],
        ),
        (
            "a range next to a pick joins it: one passage, tagged with both",
            [
                [_span("patterns.md", 1, 2)],
                [_span("patterns.md", 3)],
            ],
            1,
            10,
            [("backend", "patterns.md", 1, 3, [0, 1])],
        ),
        (
            "a range past a heading after a pick is a section of its own: it takes a slot",
            [
                [_span("patterns.md", 1, 2, end=CutReason.HEADING)],
                [_span("patterns.md", 3)],
            ],
            1,
            10,
            [("backend", "patterns.md", 1, 2, [0]), ("backend", "patterns.md", 3, 3, [1])],
        ),
        (
            "a range between two picks merges them into the first",
            [[_span("patterns.md", 1)], [_span("patterns.md", 3)], [_span("patterns.md", 2)]],
            1,
            10,
            [("backend", "patterns.md", 1, 3, [0, 1, 2])],
        ),
        (
            "one document in two collections never joins across them",
            [
                [_span("patterns.md", 1)],
                [_span("patterns.md", 2, collection="archive")],
            ],
            1,
            10,
            [
                ("backend", "patterns.md", 1, 1, [0]),
                ("archive", "patterns.md", 2, 2, [1]),
            ],
        ),
        (
            "the cap stops the turns, mid-round too",
            [
                [_span("a.md", 1), _span("a.md", 3)],
                [_span("b.md", 1), _span("b.md", 3)],
            ],
            2,
            3,
            [
                ("backend", "a.md", 1, 1, [0]),
                ("backend", "b.md", 1, 1, [1]),
                ("backend", "a.md", 3, 3, [0]),
            ],
        ),
    ],
)
def test_the_parts_take_turns_at_the_slots(
    name: str,
    ranked: list[list[HitRange]],
    depth: int,
    cap: int,
    expected: list[tuple[str, str, int, int, list[int]]],
) -> None:
    assert _shape(aspects.interleave(ranked, depth, cap)) == expected, name


def test_a_joined_pick_is_one_passage_over_every_chunk_it_holds() -> None:
    """Two parts landing on neighbouring chunks get one range, with its offsets and lines, that
    `read` reads once rather than two touching passages, and it counts both ranges it took."""
    (pick,) = aspects.interleave([[_span("patterns.md", 2, 3)], [_span("patterns.md", 4)]], 1, 10)

    assert pick.taken == 2

    assert [hit.seq for hit in pick.span.hits] == [2, 3, 4]
    assert (pick.span.char_start, pick.span.char_end) == (100, 300 + len(SENTENCES[4]))
    assert (pick.span.line_start, pick.span.line_end) == (2, 4)


# --- the tags, through the fold -------------------------------------------------------


def _tree(references: list[PassageReference]) -> list:
    """Each place's document, with what sits under it."""
    return [(place.document, _tree(place.also_in)) for place in references]


@pytest.mark.parametrize(
    ("name", "ranked", "kept", "tags"),
    [
        (
            "nothing folded: each passage answers the part that picked it",
            [
                [_span("a.md", 1, text=RETRY, score=0.9)],
                [_span("b.md", 1, text=SENTENCES[3], score=0.8)],
            ],
            [("a.md", []), ("b.md", [])],
            [[AGGREGATE], [EVENTS]],
        ),
        (
            "a copy folds into the first pick and brings its part's tag along",
            [
                [_span("a.md", 1, text=RETRY, score=0.9)],
                [_span("copy.md", 1, text=RETRY, score=0.8)],
            ],
            [("a.md", [("copy.md", [])])],
            [[AGGREGATE, EVENTS]],
        ),
        (
            "a fuller passage takes the slot of the one it holds, with both tags",
            [
                [_span("a.md", 1, text=RETRY, score=0.9)],
                [_span("fuller.md", 1, text=FULLER, score=0.8)],
            ],
            [("fuller.md", [("a.md", [])])],
            [[AGGREGATE, EVENTS]],
        ),
        (
            "a place two levels down still tags: a copy under the pick, then a fuller swap above",
            [
                [_span("a.md", 1, text=RETRY, score=0.9)],
                [_span("copy.md", 1, text=RETRY, score=0.8)],
                [_span("fuller.md", 1, text=FULLER, score=0.7)],
            ],
            [("fuller.md", [("a.md", [("copy.md", [])])])],
            [[AGGREGATE, EVENTS, ORDERING]],
        ),
    ],
)
def test_a_kept_passage_answers_its_own_part_and_every_folded_one(
    name: str, ranked: list[list[HitRange]], kept: list[tuple], tags: list[list[str]]
) -> None:
    picks = aspects.interleave(ranked, 1, 10)
    scanned = [one for found in ranked for span in found for one in span.hits]
    folded = collapse.ranges([pick.span for pick in picks], scanned, words_scan(scanned), 2)

    found = aspects.tagged(folded, picks, [AGGREGATE, EVENTS, ORDERING][: len(ranked)])

    assert [(one.hits[0].document, _tree(one.also_in)) for one in found] == kept, name
    assert [one.aspects for one in found] == tags, name
