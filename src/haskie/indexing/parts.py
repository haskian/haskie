"""Where a document's work is cut into batches: where its sections start, else every so many
pages.

A batch that ends where a section starts keeps that section whole, so no step has to join it
back together across two batches. Batches are packed greedily: one grows up to how far a batch
may reach and is cut at the last section start inside it. A section longer than a batch cannot
fit whole. It is cut at the last fallback position inside the reach, a page, and runs whole to
the next place it can be cut only when it holds none.

The convert stage cuts pages at the pages a PDF's bookmarks start on; the embed stage cuts the
converted markdown at its headings, with its page markers as the fallback. No IO here.
"""

import bisect
import sys
from collections.abc import Callable, Sequence


def cuts(
    total: int,
    starts: Sequence[int],
    reach: Callable[[int], int],
    fallback: Sequence[int] | None = None,
) -> list[tuple[int, int]]:
    """`[start, end)` ranges tiling `[0, total)`, each ending no further than `reach(start)`
    unless one section alone is longer. `starts` are where sections start; `fallback` where a
    section too long may be cut, None for anywhere. Positions outside `(0, total)` are ignored.
    No range for `total` 0."""
    bounds = _inside(starts, total)
    backup = None if fallback is None else _inside(fallback, total)
    found: list[tuple[int, int]] = []
    start = 0
    while start < total:
        limit = max(reach(start), start + 1)
        if limit >= total:
            found.append((start, total))
            break
        cut = _last(bounds, start, limit)
        if cut is None:
            cut = limit if backup is None else _last(backup, start, limit)
        if cut is None:  # nowhere to cut inside the reach: whole to the next place there is
            cut = min(_after(bounds, limit, total), _after(backup or [], limit, total))
        found.append((start, cut))
        start = cut
    return found


def pages(markers: Sequence[int], count: int) -> Callable[[int], int]:
    """How far a batch of `count` pages reaches from a position, given where each page starts
    (`markers`, sorted): to the start of the page `count` pages after the one it starts in, else
    the end."""

    def reach(start: int) -> int:
        at = bisect.bisect_right(markers, start) - 1 + count  # the page it starts in, then on
        return markers[at] if at < len(markers) else sys.maxsize  # past the last page: the end

    return reach


def _inside(positions: Sequence[int], total: int) -> list[int]:
    return sorted({at for at in positions if 0 < at < total})


def _last(positions: list[int], low: int, high: int) -> int | None:
    """The last of `positions` in `(low, high]`, else None."""
    at = bisect.bisect_right(positions, high) - 1
    return positions[at] if at >= 0 and positions[at] > low else None


def _after(positions: list[int], low: int, total: int) -> int:
    """The first of `positions` past `low`, else `total`."""
    at = bisect.bisect_right(positions, low)
    return positions[at] if at < len(positions) else total
