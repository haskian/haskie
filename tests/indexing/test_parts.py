"""Where a document's work is cut: at the starts of its sections, packed up to a batch, and at a
page inside a section longer than a batch."""

import sys

import pytest

from haskie.indexing import parts


def _within(budget: int):
    return lambda start: start + budget


@pytest.mark.parametrize(
    ("name", "total", "starts", "budget", "fallback", "expected"),
    [
        ("nothing to cut", 0, [], 4, None, []),
        ("everything fits one batch", 10, [3, 6], 20, None, [(0, 10)]),
        (
            "sections packed up to the budget, cut at the last start inside it",
            12,
            [2, 5, 7, 11],
            6,
            None,
            [(0, 5), (5, 11), (11, 12)],
        ),
        ("no sections: every budget, anywhere", 10, [], 4, None, [(0, 4), (4, 8), (8, 10)]),
        (
            "a section longer than the budget: at the last fallback inside it",
            20,
            [15],
            6,
            [3, 5, 9, 13],
            [(0, 5), (5, 9), (9, 15), (15, 20)],
        ),
        (
            "a section longer than the budget with no fallback inside: whole to the next place",
            20,
            [15],
            6,
            [11],
            [(0, 11), (11, 15), (15, 20)],
        ),
        (
            "no fallback anywhere: the section runs whole to the next start",
            20,
            [15],
            6,
            [],
            [(0, 15), (15, 20)],
        ),
        (
            "positions at the edges or twice change nothing",
            10,
            [0, 5, 5, 10, 12],
            6,
            None,
            [(0, 5), (5, 10)],
        ),
    ],
)
def test_cuts(
    name: str,
    total: int,
    starts: list[int],
    budget: int,
    fallback: list[int] | None,
    expected: list[tuple[int, int]],
) -> None:
    found = parts.cuts(total, starts, _within(budget), fallback)

    assert found == expected, name
    assert [start for start, _ in found[1:]] == [end for _, end in found[:-1]], "they tile"


@pytest.mark.parametrize(
    ("name", "start", "expected"),
    [
        ("before the first page: what comes before it counts as one", 0, 20),
        ("at a page's start: that page and the next", 20, 60),
        ("inside a page: to the start of the page two after", 25, 60),
        ("on the last pages: the end", 60, sys.maxsize),
    ],
)
def test_pages_reach_count_pages_by_their_markers(name: str, start: int, expected: int) -> None:
    reach = parts.pages([5, 20, 40, 60], 2)

    assert reach(start) == expected, name
