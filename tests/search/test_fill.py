"""The text around and between the passages of a section: what each chunk near them is worth,
which stretches a section could take, which of them the budget pays for, and how they join.

Every chunk below is a real chunk of `MARKDOWN`, cut at 150 characters with no merging of short
paragraphs: eight paragraphs of 93 characters under "Guide > Retries" (seq 1-8), then "Guide >
Other" (seq 9).
"""

import msgspec
import pytest
from conftest import chunk_hit

from haskie.indexing.chunk import split
from haskie.search import fill
from haskie.search.fill import Candidate, Fill
from haskie.search.passage import HitRange, ranges
from haskie.search.section import Group, Section
from haskie.settings import ChunkSettings

MARKDOWN = (
    "# Guide\n\n## Retries\n\n"
    + "\n\n".join(
        f"Paragraph {i} of the retries section says one thing about backoff, jitter and "
        "idempotency keys."
        for i in range(1, 9)
    )
    + "\n\n## Other\n\nThe other section talks about something else entirely, far from retries.\n"
)
DOC = "retries.md"
COLLECTION = "notes"
CHUNKS = split(MARKDOWN, ChunkSettings(chunk_size=150, chunk_merge_below=0))
HITS = {
    seq: chunk_hit(chunk, seq, document=DOC, collection=COLLECTION)
    for seq, chunk in enumerate(CHUNKS, start=1)
}
RETRIES = Section(("Guide", "Retries"), 1, 8)
SIZE = 93  # every paragraph's length


def _range(*seqs: int, aspects: list[str] | None = None) -> HitRange:
    (found,) = ranges([HITS[seq] for seq in seqs])
    return msgspec.structs.replace(found, aspects=aspects or [])


def _group(*found: HitRange, section: Section = RETRIES) -> Group:
    return Group(COLLECTION, DOC, section, list(found))


def _candidates(values: dict[int, float], aspect: str | None = None) -> dict[int, Candidate]:
    return {seq: Candidate(HITS[seq], worth, aspect) for seq, worth in values.items()}


def test_the_fixture_is_the_chunks_the_docstring_names() -> None:
    assert [chunk.char_end - chunk.char_start for chunk in CHUNKS[:8]] == [SIZE] * 8
    assert [tuple(chunk.headings) for chunk in CHUNKS] == [("Guide", "Retries")] * 8 + [
        ("Guide", "Other")
    ]


# --- value ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "score", "floor", "top", "expected"),
    [
        ("as good as the median kept chunk", 0.6, 0.6, 0.8, 0.0),
        ("as good as the best", 0.8, 0.6, 0.8, 1.0),
        ("halfway", 0.7, 0.6, 0.8, 0.5),
        ("better than the best is still 1", 0.95, 0.6, 0.8, 1.0),
        ("far below is at most -1", 0.0, 0.6, 0.8, -1.0),
        ("kept chunks all alike: as good is worth 1", 0.6, 0.6, 0.6, 1.0),
        ("kept chunks all alike: weaker is worth -1", 0.5, 0.6, 0.6, -1.0),
    ],
)
def test_a_chunk_is_worth_its_score_around_the_kept_chunks(
    name: str, score: float, floor: float, top: float, expected: float
) -> None:
    assert fill.value(score, floor, top) == pytest.approx(expected), name


# --- within ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "budget", "expected"),
    [
        ("everything fits", 1000, 3),
        ("the sections past the budget go, the last first", 2 * SIZE, 2),
        ("a later small section does not jump the queue", 3 * SIZE - 1, 2),
        ("the first stays even over the budget", 10, 1),
    ],
)
def test_the_sections_are_cut_to_the_budget_in_order(name: str, budget: int, expected: int) -> None:
    groups = [_group(_range(1)), _group(_range(3)), _group(_range(5))]

    assert len(fill.within(groups, budget)) == expected, name


# --- near -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "found", "section", "expected"),
    [
        ("up to REACH either side", [_range(4)], RETRIES, {1, 2, 3, 5, 6, 7, 8}),
        ("never out of the section", [_range(1)], Section(("Guide", "Retries"), 1, 3), {2, 3}),
        ("not the passages' own chunks", [_range(2), _range(5, 6)], RETRIES, {1, 3, 4, 7, 8}),
    ],
)
def test_the_chunks_a_fill_could_take_are_near_the_passages(
    name: str, found: list[HitRange], section: Section, expected: set[int]
) -> None:
    assert fill.near(_group(*found, section=section)) == expected, name


# --- fills ----------------------------------------------------------------------------


def _shape(found: list[Fill]) -> list[list[int]]:
    return [[chunk.hit.seq for chunk in one.chunks] for one in found]


@pytest.mark.parametrize(
    ("name", "found", "values", "expected"),
    [
        (
            "a gap that sums above 0 is bridged, a weak chunk paid for by a strong one",
            [_range(2), _range(5)],
            {3: 0.6, 4: -0.2},
            [[3, 4]],
        ),
        ("a gap that sums to 0 or less stays", [_range(2), _range(5)], {3: 0.2, 4: -0.2}, []),
        ("a gap with an unread chunk stays", [_range(2), _range(5)], {3: 0.9}, []),
        (
            "a passage grows outward by the run that sums highest",
            [_range(4)],
            {5: 0.5, 6: -0.1, 7: 0.3, 8: -0.9},
            [[5, 6, 7]],
        ),
        ("both ends of the section's passages grow", [_range(4)], {3: 0.4, 5: 0.2}, [[3], [5]]),
        ("a run that never sums above 0 is none", [_range(4)], {5: -0.1, 6: 0.05}, []),
        ("growth stops at a chunk not read", [_range(4)], {5: 0.5, 7: 0.9}, [[5]]),
    ],
)
def test_a_section_could_bridge_its_gaps_and_grow_its_ends(
    name: str, found: list[HitRange], values: dict[int, float], expected: list[list[int]]
) -> None:
    assert _shape(fill.fills(0, _group(*found), _candidates(values))) == expected, name


# --- choose ---------------------------------------------------------------------------


def test_the_budget_pays_for_the_fills_worth_most_per_character() -> None:
    rich = Fill(0, list(_candidates({3: 0.9}).values()))  # 0.9 over one chunk
    long = Fill(1, list(_candidates({5: 0.6, 6: 0.6}).values()))  # 1.2 over two
    weak = Fill(2, list(_candidates({8: 0.1}).values()))

    chosen = fill.choose([weak, long, rich], 2 * SIZE)

    assert chosen == [rich, weak], "the long one no longer fits once the rich one is paid for"
    assert fill.choose([rich], SIZE - 1) == [], "nothing that does not fit"


# --- apply ----------------------------------------------------------------------------


def test_a_bridged_gap_makes_two_passages_one_that_keeps_what_both_held() -> None:
    first = _range(2, aspects=["a"])
    second = _range(5, aspects=["b"])
    taken = list(_candidates({3: 0.6, 4: 0.1}, aspect="c").values())

    (joined,) = fill.apply(_group(first, second), taken).ranges

    assert (joined.seq_start, joined.seq_end) == (2, 5)
    assert joined.aspects == ["a", "b", "c"], "the passages' questions, then the fill's"
    assert [hit.score for hit in joined.hits] == [1.0, 0.0, 0.0, 1.0], "a fill is not ranked"


def test_a_group_that_takes_nothing_is_unchanged() -> None:
    one = _group(_range(2))

    assert fill.apply(one, []) is one
