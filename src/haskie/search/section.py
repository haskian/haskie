"""Passages grouped by the section they sit in, and each group written out as one excerpt.

A search keeps passages one by one, so three passages under one heading come back as three
results competing for the slots, and a reader stitches them together again. Here they become one
excerpt: the section they share, with each passage in document order.

Which section: the largest one that still reads as a quote. A passage's heading path is tried
from the top, and a level is taken when its section fits `max_section_chars`. A level that
holds the whole document says nothing (a title heading over everything), and a level too large
is split one heading down, so a book groups by chapter or section and a short note by its title.
A passage under the deepest heading it has takes that heading's section, however large.

A section is found from the document's outline (`Entry`: every chunk's `seq`, heading path and
char span), which the chunker makes exact: a chunk never spans two sections. No IO here.
"""

import bisect
import os
from collections.abc import Sequence

import msgspec

from haskie.collection.index import Hit, location
from haskie.document.convert import without_markers
from haskie.indexing.chunk import HEADING_SEP
from haskie.search.passage import Excerpt, HitRange, pages, span

ELISION = "[…]"  # between two passages the document has text between


class Entry(msgspec.Struct, frozen=True):
    """One chunk of a document's outline: where it sits and under which headings."""

    seq: int
    headings: tuple[str, ...]
    char_start: int
    char_end: int


type Outline = list[Entry]  # one document's chunks, by `seq`
type Place = tuple[str, str]  # (collection, document)


class Section(msgspec.Struct, frozen=True):
    """A run of a document's chunks under one heading path."""

    path: tuple[str, ...]
    seq_start: int
    seq_end: int


def outlines(rows: Sequence[tuple[str, dict]]) -> dict[Place, Outline]:
    """Each document's outline out of the rows `CollectionIndex.outline_rows` read, one pair of
    (collection, row) each, in any order."""
    found: dict[Place, Outline] = {}
    for collection, row in rows:
        entry = Entry(
            seq=row["seq"],
            headings=tuple(row["headings"] or ()),
            char_start=row["char_start"],
            char_end=row["char_end"],
        )
        found.setdefault((collection, row["document"]), []).append(entry)
    for outline in found.values():
        outline.sort(key=lambda entry: entry.seq)
    return found


def section_of(outline: Outline, seq: int, max_chars: int) -> Section:
    """The section chunk `seq` is grouped in (see the module).

    The outline is read through the same open table as the hits (`retrieval.sections`), so it
    holds every chunk a hit names; one it does not is a broken search, not a case to guess at.
    """
    at = bisect.bisect_left(outline, seq, key=lambda entry: entry.seq)
    if at == len(outline) or outline[at].seq != seq:
        raise LookupError(f"chunk {seq} is not in the outline its hit was read with")
    path = outline[at].headings
    for level in range(1, len(path)):
        first, last = _run(outline, at, path[:level])
        whole = first == 0 and last == len(outline) - 1
        if not whole and outline[last].char_end - outline[first].char_start <= max_chars:
            return Section(path[:level], outline[first].seq, outline[last].seq)
    first, last = _run(outline, at, path)
    return Section(path, outline[first].seq, outline[last].seq)


def _run(outline: Outline, at: int, prefix: tuple[str, ...]) -> tuple[int, int]:
    """The outline positions of the longest run around `at` whose headings open with `prefix`."""
    level = len(prefix)
    first = last = at
    while first > 0 and outline[first - 1].headings[:level] == prefix:
        first -= 1
    while last + 1 < len(outline) and outline[last + 1].headings[:level] == prefix:
        last += 1
    return first, last


class Group(msgspec.Struct):
    """The kept passages of one section, in the order they were kept."""

    collection: str
    document: str
    section: Section
    ranges: list[HitRange]

    @property
    def hits(self) -> list[Hit]:
        """Every chunk its passages hold."""
        return [hit for hit_range in self.ranges for hit in hit_range.hits]

    @property
    def chars(self) -> int:
        """How long its passages are together."""
        return sum(hit_range.char_end - hit_range.char_start for hit_range in self.ranges)


def within(groups: list[Group], budget: int) -> list[Group]:
    """The groups whose passages fit `budget` characters together, in order. The first always
    stays: an answer of one section over the budget beats no answer."""
    kept: list[Group] = []
    used = 0
    for one in groups:
        if kept and used + one.chars > budget:
            break
        kept.append(one)
        used += one.chars
    return kept


def group(
    hit_ranges: list[HitRange], found: dict[Place, Outline], max_chars: int, limit: int
) -> list[Group]:
    """The first `limit` sections the ranges (best first) fall in, each with every range of it.

    A section is placed where its best range was, and a range further down joins the section it
    belongs to rather than taking a slot: the slots count sections. A range whose chunk the
    `found` holds the outline of every document a range that opens a section is in (see
    `documents`); a range of any other document belongs to no section that is kept. A section
    whose ranges are all too short to stand alone (`HitRange.alone`) is no excerpt, and frees its
    slot.
    """
    placed: list[tuple[tuple[str, str, Section], HitRange]] = []
    for hit_range in hit_ranges:
        first = hit_range.hits[0]
        outline = found.get((first.collection, first.document))
        if outline is not None:
            where = section_of(outline, first.seq, max_chars)
            placed.append(((first.collection, first.document, where), hit_range))
    standing = {key for key, hit_range in placed if not hit_range.alone}
    groups: dict[tuple[str, str, Section], Group] = {}
    for key, hit_range in placed:
        if key in groups:
            groups[key].ranges.append(hit_range)
        elif key in standing and len(groups) < limit:
            groups[key] = Group(*key, ranges=[hit_range])
    return list(groups.values())


def documents(hit_ranges: list[HitRange], limit: int) -> set[Place]:
    """The documents whose outlines `group` needs: the first `limit` ones, in rank order, that a
    range standing alone is in. Every kept section opens on such a range, so the k-th section is
    in one of the first k of them, and a range of any other document joins none of them."""
    found: dict[Place, None] = {}
    for hit_range in hit_ranges:
        if len(found) == limit:
            break
        if not hit_range.alone:
            found[(hit_range.hits[0].collection, hit_range.hits[0].document)] = None
    return set(found)


def excerpt(found: Group, texts: list[str]) -> Excerpt:
    """One group as an excerpt, given the markdown each of its ranges covers, in the group's
    order.

    The passages go in document order. Each opens with the headings it sits under that the one
    before it did not (below the section's own), as markdown headings of their depth, and `[…]`
    stands where the document has text between two of them that the search did not keep.
    """
    ordered = sorted(zip(found.ranges, texts, strict=True), key=lambda pair: pair[0].seq_start)
    parts: list[str] = []
    before: tuple[str, ...] = found.section.path
    last_seq: int | None = None
    for hit_range, text in ordered:
        path = tuple(hit_range.hits[0].headings)
        if last_seq is not None and hit_range.seq_start > last_seq + 1:
            parts.append(ELISION)
        opened = max(len(os.path.commonprefix([before, path])), len(found.section.path))
        parts.extend(f"{'#' * (depth + 1)} {path[depth]}" for depth in range(opened, len(path)))
        parts.append(without_markers(text).strip())
        before, last_seq = path, hit_range.seq_end
    spans = [span(hit_range) for hit_range, _ in ordered]
    first, last = ordered[0][0], ordered[-1][0]
    page_start, page_end = pages([hit for hit_range, _ in ordered for hit in hit_range.hits])
    best = first.best
    return Excerpt(
        collection=found.collection,
        document=found.document,
        header=HEADING_SEP.join(found.section.path),
        location=location(found.document, page_start, page_end, first.line_start, last.line_end),
        seq_start=first.seq_start,
        seq_end=last.seq_end,
        line_start=first.line_start,
        line_end=last.line_end,
        char_start=first.char_start,
        char_end=last.char_end,
        page_start=page_start,
        page_end=page_end,
        text="\n\n".join(parts),
        score=max(one.score for one in spans),
        source_file=best.source_file,
        markdown_file=best.markdown_file,
        spans=spans,
        aspects=list(dict.fromkeys(label for one in spans for label in one.aspects)),
    )
