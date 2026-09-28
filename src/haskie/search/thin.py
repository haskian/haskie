"""Passages too short to stand alone: grown by the neighbouring chunks that match the question,
else dropped.

A chunk is cut where the author cut, so a short one is usually a section's lead-in ("Three
rules:", with the list in the next chunk), or a section's last line. Alone
it tells a reader little and costs a slot. So a thin range grows by the chunks of its section
next to it that match the question, by the rule every passage of an excerpt grows by
(`fill.run`): on each side, the run of chunks whose values sum highest, when that is above 0. A
range with no such run is marked `alone`, unless it is the best range of the search: a short
exact answer ("Tim Cook, CEO") is still an answer. What `alone` costs is decided where the answer
is cut: a passage that is alone goes, and so does an excerpt whose passages all are, while a
short passage inside a section with another passage stays in that section's excerpt. A thin
range with a section end on both sides is a whole section, a short note say, not a piece of one:
it is kept as the author wrote it.

"Matches" is measured, not assumed: `retrieval.fill_thin` scores the neighbours the way it scores
the scanned hits (the reranker, the query vector, or the question's words), and values each
around them (`fill.value`): 0 as good as the median scanned hit, 1 as good as the best. The
floor is this search's own, so no calibrated threshold is needed. With `fill_values = absolute`
the reranker's score is valued as dsRAG values it instead (`fill.absolute`). Either way
`grow_bias` moves each value (`fill.biased`), so a stricter bias leaves more thin ranges alone.
No IO here.
"""

import msgspec

from haskie.collection.index import ChunkKey, Hit, chunk_key
from haskie.indexing.segment import CutReason
from haskie.search.collapse import MIN_WORDS, WORD
from haskie.search.fill import Candidate, grow
from haskie.search.passage import HitRange, ends_section, part, rejoin
from haskie.search.section import Group
from haskie.settings import ScoreFold

# The words of a question that say nothing about its topic. A short list on purpose: the question
# words and the glue of English questions, so the words left are the ones a passage should share.
STOPWORDS = frozenset(
    "the and for are was were has have had how what when where which who why whom whose does did "
    "should would could can may might must shall will not but with from into onto that this these "
    "those than then there their them they its it's about over under between each any all some "
    "one two more most other such only own same very just also".split()
)

# Where a whole section begins and ends: at a heading, or at the document's edge. A part boundary
# (`CutReason.PART`) is neither: the section goes on in the next part.
_BOUNDS = frozenset({CutReason.HEADING, CutReason.EDGE})


def _fragment(hit_range: HitRange, min_chars: int) -> bool:
    """Thin (shorter than `min_chars`, or under `MIN_WORDS` words) and part of a longer section:
    what grows or goes. A range with a section bound on both sides is a whole
    section, a short note say, with nothing of it to grow into. `min_chars` 0 turns this off."""
    if min_chars <= 0:
        return False
    first, last = hit_range.hits[0], hit_range.hits[-1]
    if first.start_reason in _BOUNDS and last.end_reason in _BOUNDS:
        return False
    words = sum(len(WORD.findall(hit.text)) for hit in hit_range.hits)
    return last.char_end - first.char_start < min_chars or words < MIN_WORDS


def around(hit_ranges: list[HitRange], min_chars: int, grow: int) -> set[ChunkKey]:
    """The chunks a thin range could grow into: up to `grow` on each side that no heading closes
    (`passage.ends_section`). Whether a chunk further out is still in the section is only known
    once the nearer one is read, so all of them are asked for."""
    wanted: set[ChunkKey] = set()
    for hit_range in hit_ranges:
        if grow <= 0 or not _fragment(hit_range, min_chars):
            continue
        first, last = hit_range.hits[0], hit_range.hits[-1]
        for step in range(1, grow + 1):
            if first.start_reason != CutReason.HEADING and first.seq - step >= 1:
                wanted.add((first.collection, first.document, first.seq - step))
            if not ends_section(last):
                wanted.add((last.collection, last.document, last.seq + step))
    return wanted


class Filled(msgspec.Struct):
    """The ranges after the thin ones grew, and what that took."""

    ranges: list[HitRange]  # best first, as `passage.ranges` orders them
    added: list[Hit]  # the neighbours that joined, each once
    grown: int  # thin ranges that took at least one neighbour


def fill(
    hit_ranges: list[HitRange],
    neighbours: dict[ChunkKey, Candidate],
    min_chars: int,
    reach: int,
    how: ScoreFold,
    grows: bool = True,
) -> Filled:
    """Grow each thin range by the neighbours worth taking, as every passage of an excerpt grows
    (`fill.run`): on each side, the run of up to `reach` chunks whose values sum highest, when
    that is above 0, never past a heading (`passage.continues`). A thin range that took none is
    marked `alone`, unless it is the first of `hit_ranges`, the best the search found. A thin
    range that is a whole section is kept as it is.

    `grows` False only judges: a thin range with a run worth taking stands, `owed`, but takes
    nothing, for a later step to grow it once (the fill of an excerpts search, `fill.fills`), so
    no passage grows twice.

    The ranges are rebuilt at the end (`passage.rejoin`), so two ranges a neighbour now joins
    become one, and a neighbour scores 0: a range's score says how strongly it matched, not how
    much it grew.
    """
    kept: list[HitRange] = []
    added: dict[ChunkKey, Hit] = {}
    grown = 0
    for position, hit_range in enumerate(hit_ranges):
        if _fragment(hit_range, min_chars):
            took = [
                chunk.hit
                for side in grow(hit_range.hits[0], hit_range.hits[-1], neighbours, reach, reach)
                for chunk in side
            ]
            if took:
                grown += 1
                if grows:
                    added.update((chunk_key(hit), hit) for hit in took)
                elif position > 0:
                    hit_range = msgspec.structs.replace(hit_range, owed=True)
            elif position > 0:
                hit_range = msgspec.structs.replace(hit_range, alone=True)
        kept.append(hit_range)
    held = {chunk_key(hit) for one in kept for hit in one.hits}
    joined = [hit for key, hit in added.items() if key not in held]
    rebuilt = rejoin([*kept, *(part(hit) for hit in joined)], how)
    return Filled(ranges=rebuilt, added=joined, grown=grown)


def settle(groups: list[Group], offered: list[set[ChunkKey]]) -> list[Group]:
    """The groups after the fill: each range still `owed` that the fill found no run worth taking
    next to (`offered`, each group's own) marked `alone`, and each group then no excerpt
    (`Group.standing`) dropped. Its slot is not handed on: the sections were counted before the
    fill, and so were the answer's budget and the words the probe looks for.

    An excerpts search only judges a thin range (`fill` with `grows` off): past the best range, it
    stands, owed, when a run worth taking is next to it, valued against the scanned hits. The fill
    values the same chunks against the kept ones, whose median is higher, and may find none worth
    taking. The range is then as short as one the passages answer drops, and goes the same way.
    One the fill found a run for keeps standing, even when the answer's budget could not pay for
    it; one it grew is rebuilt, and owes nothing (`passage.rejoin`)."""
    settled: list[Group] = []
    for one, near in zip(groups, offered, strict=True):
        if any(hit_range.owed for hit_range in one.ranges):
            ranges = [_settled(hit_range, near) for hit_range in one.ranges]
            one = msgspec.structs.replace(one, ranges=ranges)
        if one.standing:
            settled.append(one)
    return settled


def _settled(hit_range: HitRange, offered: set[ChunkKey]) -> HitRange:
    if not hit_range.owed:
        return hit_range
    first, last = hit_range.hits[0], hit_range.hits[-1]
    beside = {
        (first.collection, first.document, first.seq - 1),
        (last.collection, last.document, last.seq + 1),
    }
    return msgspec.structs.replace(hit_range, owed=False, alone=not beside & offered)


def terms(text: str) -> list[str]:
    """The words of `text` a passage on its topic would share, once each in the order they come:
    lowercase, three letters or more, no stopwords."""
    found = WORD.findall(text.lower())
    return [word for word in dict.fromkeys(found) if len(word) > 2 and word not in STOPWORDS]


def overlap(question: list[str], held: set[str]) -> float:
    """The share of the question's words (`terms`) that a text's words (`held`) hold: the match
    score when there is neither a reranker nor a vector to ask. A chunk's heading path is left
    out, since every chunk of a section shares it."""
    if not question:
        return 0.0
    return sum(1 for word in question if word in held) / len(question)
