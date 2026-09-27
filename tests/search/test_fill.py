"""The text around and between the passages of a section: what each chunk near them is worth,
which stretches a section could take, which of them the budget pays for, and how they join.

Every chunk below is a real chunk of `MARKDOWN`, cut at 150 characters with no merging of short
paragraphs: eight paragraphs of 93 characters under "Guide > Retries" (seq 1-8), then "Guide >
Other" (seq 9).
"""

import random

import msgspec
import pytest
from conftest import chunk_hit

from haskie.collection.index import ChunkKey
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


def _candidates(values: dict[int, float], aspect: str | None = None) -> dict[ChunkKey, Candidate]:
    return {
        (COLLECTION, DOC, seq): Candidate(HITS[seq], worth, aspect) for seq, worth in values.items()
    }


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


# --- near -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "found", "section", "expected"),
    [
        ("up to `reach` either side", [_range(4)], RETRIES, {1, 2, 3, 5, 6, 7, 8}),
        ("never out of the section", [_range(1)], Section(("Guide", "Retries"), 1, 3), {2, 3}),
        ("not the passages' own chunks", [_range(2), _range(5, 6)], RETRIES, {1, 3, 4, 7, 8}),
    ],
)
def test_the_chunks_a_fill_could_take_are_near_the_passages(
    name: str, found: list[HitRange], section: Section, expected: set[int]
) -> None:
    assert fill.near(_group(*found, section=section), 4) == expected, name


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
    assert _shape(fill.fills(0, _group(*found), _candidates(values), 4)) == expected, name


# --- run ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "values", "expected"),
    [
        ("the prefix that sums highest", [0.5, -0.1, 0.3, -0.9], [5, 6, 7]),
        ("a weak chunk first is paid for by a strong one after", [-0.2, 0.6], [5, 6]),
        ("nothing sums above 0: none", [-0.1, 0.05], []),
        ("nothing read: none", [], []),
    ],
)
def test_a_run_outward_takes_the_prefix_worth_most(
    name: str, values: list[float], expected: list[int]
) -> None:
    outward = [Candidate(HITS[seq], worth) for seq, worth in zip(range(5, 9), values, strict=False)]

    assert [chunk.hit.seq for chunk in fill.run(outward)] == expected, name


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


# --- the fill, over many random sections ----------------------------------------------------


@pytest.mark.parametrize("seed", range(200))
def test_a_fill_takes_only_unkept_chunks_of_its_section_within_the_room(seed: int) -> None:
    """Whatever passages a section holds and whatever its neighbours are worth: a fill takes only
    chunks the section holds and its passages do not, within `reach` of a passage, never more
    than the room; and the group it makes holds every chunk once."""
    rng = random.Random(seed)
    kept = sorted(rng.sample(range(1, 9), rng.randint(1, 4)))
    found = [_range(seq) for seq in kept]
    reach = rng.randint(0, 3)
    one = _group(*found)
    near = fill.near(one, reach)
    values = {seq: rng.uniform(-1, 1) for seq in near}
    room = rng.randint(0, 6) * SIZE

    chosen = fill.choose(fill.fills(0, one, _candidates(values), reach), room)
    taken = [chunk for piece in chosen for chunk in piece.chunks]
    joined = fill.apply(one, taken)

    seqs = [chunk.hit.seq for chunk in taken]
    assert len(seqs) == len(set(seqs)), "a chunk is taken once"
    assert set(seqs) <= near, "only chunks near a passage, in the section, not a passage's own"
    assert all(min(abs(seq - k) for k in kept) <= reach for seq in seqs), "within reach"
    assert sum(piece.chars for piece in chosen) <= room, "never more than the room"
    assert all(piece.value > 0 for piece in chosen), "only what is worth taking"
    held = sorted(hit.seq for hit_range in joined.ranges for hit in hit_range.hits)
    assert held == sorted([*kept, *seqs]), "the group holds every chunk once"
