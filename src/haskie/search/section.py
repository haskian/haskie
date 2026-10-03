"""Passages grouped by the section they sit in, and each group written out as one excerpt.

A search keeps passages one by one, so three passages under one heading come back as three
results competing for the slots, and a reader stitches them together again. Here they become one
excerpt: the section they share, with each passage in document order.

Which section: the largest one that still reads as a quote. A passage's heading path is tried
from the top, and a level is taken when its section fits `max_section_chars`. A level that
holds the whole document says nothing (a title heading over everything), and a level too large
is split one heading down, so a book groups by chapter or section and a short note by its title.
A passage under the deepest heading it has takes that heading's section, however large.

A section is found from the document's chunk placements (`Placement`: every chunk's `seq`,
heading path and char span), which the chunker makes exact: a chunk never spans two sections. No
IO here.
"""

import bisect
import os
from collections.abc import Sequence

import msgspec

from haskie.collection.index import Hit, location
from haskie.document.convert import without_markers
from haskie.indexing.chunk import HEADING_SEP
from haskie.search.passage import Excerpt, HitRange, best_of, pages, span

ELISION = "[…]"  # between two passages the document has text between


class Placement(msgspec.Struct, frozen=True):
    """Where one chunk of a document sits, and under which headings."""

    seq: int
    headings: tuple[str, ...]
    char_start: int
    char_end: int
    line_start: int  # what a section cites (`search.section_map`); the grouping reads none of it
    line_end: int
    page_start: int | None
    page_end: int | None
    # the ids of the sections that hold it, the whole document first (`sections.build`): the
    # section of depth d is `section_ids[d]`
    section_ids: tuple[str, ...] = ()


type Placements = list[Placement]  # one document's chunks, by `seq`
type Place = tuple[str, str]  # (collection, document id)


class Section(msgspec.Struct, frozen=True):
    """A run of a document's chunks under one heading path."""

    path: tuple[str, ...]
    seq_start: int
    seq_end: int
    id: str = ""  # its id (`sections.build`); empty for chunks indexed without one


def placements(rows: Sequence[tuple[str, dict]]) -> dict[Place, Placements]:
    """Each document's placements out of the rows `CollectionIndex.placement_rows` read, one pair
    of (collection, row) each, in any order."""
    found: dict[Place, Placements] = {}
    for collection, row in rows:
        one = Placement(
            seq=row["seq"],
            headings=tuple(row["headings"] or ()),
            char_start=row["char_start"],
            char_end=row["char_end"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            section_ids=tuple(row.get("section_ids") or ()),
        )
        found.setdefault((collection, row["document_id"]), []).append(one)
    for chunks in found.values():
        chunks.sort(key=lambda one: one.seq)
    return found


def section_of(
    chunks: Placements, seq: int, max_chars: int, kept: frozenset[str] = frozenset()
) -> Section:
    """The section chunk `seq` is grouped in (see the module). `kept` names the sections a search
    keeps to (`Scope.section_ids`): `chunks` then holds theirs alone, so one of them can span all
    of it and still be no whole document.

    The placements are read through the same open table as the hits (`retrieval.sections`), so
    they hold every chunk a hit names; one they do not is a broken search, not a case to guess at.
    """
    at = bisect.bisect_left(chunks, seq, key=lambda one: one.seq)
    if at == len(chunks) or chunks[at].seq != seq:
        raise LookupError(f"chunk {seq} is not in the placements its hit was read with")
    path, ids = chunks[at].headings, chunks[at].section_ids
    for level in range(1, len(path)):
        first, last = _run(chunks, at, path[:level])
        scoped = level < len(ids) and ids[level] in kept
        whole = first == 0 and last == len(chunks) - 1 and not scoped
        if not whole and chunks[last].char_end - chunks[first].char_start <= max_chars:
            return _section(chunks, at, path[:level], first, last)
    first, last = _run(chunks, at, path)
    return _section(chunks, at, path, first, last)


def _section(chunks: Placements, at: int, path: tuple[str, ...], first: int, last: int) -> Section:
    ids = chunks[at].section_ids
    id = ids[len(path)] if len(path) < len(ids) else ""
    return Section(path, chunks[first].seq, chunks[last].seq, id)


def _run(chunks: Placements, at: int, prefix: tuple[str, ...]) -> tuple[int, int]:
    """Where in `chunks` the longest run around `at` whose headings open with `prefix` runs."""
    level = len(prefix)
    first = last = at
    while first > 0 and chunks[first - 1].headings[:level] == prefix:
        first -= 1
    while last + 1 < len(chunks) and chunks[last + 1].headings[:level] == prefix:
        last += 1
    return first, last


class Group(msgspec.Struct):
    """The kept passages of one section, in the order they were kept."""

    collection: str
    document_id: str
    section: Section
    ranges: list[HitRange]

    @property
    def hits(self) -> list[Hit]:
        """Every chunk its passages hold."""
        return [hit for hit_range in self.ranges for hit in hit_range.hits]

    @property
    def standing(self) -> bool:
        """Whether it is an excerpt: a section whose passages are all too short to stand alone
        (`HitRange.alone`) is none."""
        return not all(hit_range.alone for hit_range in self.ranges)

    @property
    def cost(self) -> int:
        """What it costs an agent as an excerpt (`excerpt_cost`)."""
        return excerpt_cost(self.ranges)


# Beside its text, what an excerpt and each of its passages (a span) cost in the tool's answer
# (`api.agent.Excerpt`, `Span`): the field names, ids, citation, score and path every one carries.
# Rounded up from 19 real excerpts of 24 spans: an excerpt's fields took at most 567 characters,
# a span's 295. Text is counted raw, so the JSON escapes of its newlines and quotes (under 1%) and
# the headings an excerpt opens its passages with ride on these too.
EXCERPT_CHARS = 600
SPAN_CHARS = 300
ASPECT_CHARS = 4  # a question's quotes and separator, on the excerpt
SPAN_ASPECTS_CHARS = len('"aspects":[],')  # a span's list of its questions' positions (`Span`)
SPAN_ASPECT_CHARS = 2  # one position in it: a digit, at most 5 questions, and a separator


def excerpt_cost(ranges: list[HitRange]) -> int:
    """What an excerpt of these passages costs an agent, in characters of the tool's JSON answer:
    what it carries beside its text (`EXCERPT_CHARS`), each question it answers, which it names in
    full, and each passage (`passage_cost`). None at all for no passages."""
    if not ranges:
        return 0
    labels = {label for hit_range in ranges for label in hit_range.aspects}
    return (
        EXCERPT_CHARS
        + sum(len(label) + ASPECT_CHARS for label in labels)
        + sum(passage_cost(hit_range) for hit_range in ranges)
    )


def passage_cost(hit_range: HitRange) -> int:
    """What one passage costs in the tool's answer: its fields (`SPAN_CHARS`), its text, the
    positions of the questions it answers, and the places it folded in (`also_in`), counted as
    their full references encode, a bound on the agent's own, which keep fewer fields
    (`api.agent.Place`)."""
    named = (
        SPAN_ASPECTS_CHARS + SPAN_ASPECT_CHARS * len(hit_range.aspects) if hit_range.aspects else 0
    )
    folded = len(msgspec.json.encode(hit_range.also_in)) if hit_range.also_in else 0
    return SPAN_CHARS + hit_range.char_end - hit_range.char_start + named + folded


def passages(groups: list[Group]) -> int:
    """How many passages the groups hold together."""
    return sum(len(one.ranges) for one in groups)


def within(groups: list[Group], budget: int, first: HitRange | None = None) -> list[Group]:
    """The passages of `groups` that fit `budget` characters of answer (`Group.cost`), each whole:
    the ranking judged its chunks together.

    Room goes by passage, not by section, so one long section no longer takes a short answer to
    another question out with it. First `first`, the passage found for words nothing else holds
    (`retrieval.probe_gaps`), the only evidence for them; then each section's best passage that
    stands alone, in the groups' order, which already takes the questions in turns; then the
    further passages of the sections kept, the next best across all of them first
    (`HitRange.rank`), so a section's weak passages never go in before another's strong one. One
    that does not fit is skipped and the next is tried. The first section's best passage always
    stays: an answer over the budget beats no answer.

    Repeats take no room: the ranges come folded (`collapse`), a passage another one says again
    sitting in its `also_in`, never in the pool.
    """
    kept: list[set[int]] = [set() for _ in groups]
    used = 0

    def chosen(at: int, indices: set[int]) -> list[HitRange]:
        return [each for index, each in enumerate(groups[at].ranges) if index in indices]

    def take(at: int, index: int, always: bool = False) -> None:
        nonlocal used
        added = excerpt_cost(chosen(at, kept[at] | {index})) - excerpt_cost(chosen(at, kept[at]))
        if always or used + added <= budget:
            kept[at].add(index)
            used += added

    for at, one in enumerate(groups):
        for index, hit_range in enumerate(one.ranges):
            if hit_range is first:
                take(at, index)
    for at, one in enumerate(groups):
        best = next(index for index, hit_range in enumerate(one.ranges) if not hit_range.alone)
        if best not in kept[at]:
            take(at, best, always=at == 0)
    further = sorted(
        ((at, index) for at, one in enumerate(groups) for index in range(len(one.ranges))),
        key=lambda place: groups[place[0]].ranges[place[1]].rank,
    )
    for at, index in further:
        if kept[at] and index not in kept[at]:
            take(at, index)
    return [
        msgspec.structs.replace(one, ranges=chosen(at, kept[at]))
        for at, one in enumerate(groups)
        if kept[at]
    ]


def group(
    hit_ranges: list[HitRange],
    found: dict[Place, Placements],
    max_chars: int,
    limit: int,
    kept: frozenset[str] = frozenset(),
) -> list[Group]:
    """The first `limit` sections the ranges (best first) fall in, each with every range of it.

    A section is placed where its best range was, and a range further down joins the section it
    belongs to rather than taking a slot: the slots count sections. `found` holds the placements
    of every document a range that opens a section is in (see `documents`); a range of any other
    document belongs to no section that is kept. A section whose ranges are all too short to stand
    alone (`HitRange.alone`) is no excerpt, and frees its slot.
    """
    placed: list[tuple[tuple[str, str, Section], HitRange]] = []
    for hit_range in hit_ranges:
        first = hit_range.hits[0]
        chunks = found.get((first.collection, first.document_id))
        if chunks is not None:
            where = section_of(chunks, first.seq, max_chars, kept)
            placed.append(((first.collection, first.document_id, where), hit_range))
    groups: dict[tuple[str, str, Section], Group] = {}
    for rank, (key, hit_range) in enumerate(placed):
        ranked = msgspec.structs.replace(hit_range, rank=rank)
        groups.setdefault(key, Group(*key, ranges=[])).ranges.append(ranked)
    return [one for one in groups.values() if one.standing][:limit]


def documents(hit_ranges: list[HitRange], limit: int) -> set[Place]:
    """The documents whose placements `group` needs: those of every range, in rank order, up to the
    one where the `limit`-th document a range standing alone is in turns up. Sections of at least
    `limit` documents have a standing range by then, so the first `limit` sections open before
    it. A section opens on its first range, which may be alone and come before its standing one,
    so the documents of alone ranges are read too."""
    found: set[Place] = set()
    standing: set[Place] = set()
    for hit_range in hit_ranges:
        if len(standing) == limit:
            break
        place = (hit_range.hits[0].collection, hit_range.hits[0].document_id)
        found.add(place)
        if not hit_range.alone:
            standing.add(place)
    return found


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
        document_id=best.document_id,
        document=best.document,
        header=HEADING_SEP.join(found.section.path),
        section_id=found.section.id,
        location=location(best.document, page_start, page_end, first.line_start, last.line_end),
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
        aspect_scores=best_of([one.aspect_scores for one in spans]),
    )
