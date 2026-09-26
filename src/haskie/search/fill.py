"""The text around and between the passages of a section, where it answers too.

A section's passages are the chunks the ranking kept. The text between two of them, or just past
them, often finishes the answer: the list under "three rules:", the paragraph that explains the
term the passage used. It did not rank, so only the passages came back, with `[…]` between them.

So every chunk near a kept passage is scored against the question, as a value around the kept
chunks' own: 0 for a chunk that matches as well as the median kept chunk, 1 for one as good as the
best, below 0 for one that matches less. Relevant Segment Extraction (dsRAG, rse.py) picks spans
by the same arithmetic: a stretch of chunks joins when its values sum above 0, so a weak chunk
comes in only when stronger ones around it pay for it.

- A gap between two passages is filled when its values sum above 0: the two become one.
- A passage grows outward by the run of chunks next to it whose values sum highest, when that is
  above 0, up to `REACH` chunks and never out of its section.

What is added is bounded by the answer's budget (`max_answer_chars`), spent on the fills worth
most per character first. Sections that do not fit the budget with their passages alone go first,
the last kept first. No IO here: `retrieval.fill` reads and scores the chunks.
"""

import msgspec

from haskie.collection.index import Hit
from haskie.search.passage import HitRange, ranges
from haskie.search.section import Group

REACH = 4  # chunks a passage may grow by on each side, and half the longest gap filled


class Candidate(msgspec.Struct, frozen=True):
    """A chunk near a kept passage, as fill weighs it."""

    hit: Hit
    value: float  # around the kept chunks: 0 the median one, 1 the best, below 0 weaker
    aspect: str | None = None  # the question it answers best, when several were asked


class Fill(msgspec.Struct, frozen=True):
    """A stretch of chunks one group could take: a gap bridged, or a passage grown."""

    group: int  # its position in the groups
    chunks: list[Candidate]

    @property
    def value(self) -> float:
        return sum(one.value for one in self.chunks)

    @property
    def chars(self) -> int:
        return sum(one.hit.char_end - one.hit.char_start for one in self.chunks)


def value(score: float, floor: float, top: float) -> float:
    """`score` around the kept chunks' own scores: 0 at their median `floor`, 1 at their best
    `top`, clipped to [-1, 1]. When every kept chunk scores the same, a chunk at least as good is
    worth 1 and a weaker one -1."""
    if top <= floor:
        return 1.0 if score >= floor else -1.0
    return max(-1.0, min(1.0, (score - floor) / (top - floor)))


def chars(hit_ranges: list[HitRange]) -> int:
    return sum(one.char_end - one.char_start for one in hit_ranges)


def within(groups: list[Group], budget: int) -> list[Group]:
    """The groups whose passages fit `budget` characters together, in order. The first always
    stays: an answer of one section over the budget beats no answer."""
    kept: list[Group] = []
    used = 0
    for one in groups:
        size = chars(one.ranges)
        if kept and used + size > budget:
            break
        kept.append(one)
        used += size
    return kept


def near(one: Group) -> set[int]:
    """The chunks of a group's section a fill could take: `REACH` either side of each passage,
    and not the passages' own."""
    held = {hit.seq for hit_range in one.ranges for hit in hit_range.hits}
    wanted = {
        seq
        for hit_range in one.ranges
        for seq in range(hit_range.seq_start - REACH, hit_range.seq_end + REACH + 1)
    }
    return {seq for seq in wanted if one.section.seq_start <= seq <= one.section.seq_end} - held


def fills(at: int, one: Group, candidates: dict[int, Candidate]) -> list[Fill]:
    """What group `at` could take out of `candidates` (by `seq`): each gap between two of its
    passages whose chunks are all read and sum above 0, and the best run outward from its first
    and last passage when it sums above 0."""
    spans = sorted(one.ranges, key=lambda hit_range: hit_range.seq_start)
    found: list[Fill] = []
    for before, after in zip(spans, spans[1:], strict=False):
        gap = [candidates.get(seq) for seq in range(before.seq_end + 1, after.seq_start)]
        if gap and all(chunk is not None for chunk in gap):
            chunks = [chunk for chunk in gap if chunk is not None]
            if sum(chunk.value for chunk in chunks) > 0:
                found.append(Fill(at, chunks))
    for run in (
        _run(candidates, range(spans[0].seq_start - 1, spans[0].seq_start - REACH - 1, -1)),
        _run(candidates, range(spans[-1].seq_end + 1, spans[-1].seq_end + REACH + 1)),
    ):
        if run:
            found.append(Fill(at, run))
    return found


def _run(candidates: dict[int, Candidate], outward: range) -> list[Candidate]:
    """The chunks outward from a passage, nearest first, up to the point where their values sum
    highest; none when no prefix sums above 0."""
    taken: list[Candidate] = []
    best: list[Candidate] = []
    total = best_total = 0.0
    for seq in outward:
        chunk = candidates.get(seq)
        if chunk is None:
            break
        taken.append(chunk)
        total += chunk.value
        if total > best_total:
            best, best_total = list(taken), total
    return best


def choose(found: list[Fill], room: int) -> list[Fill]:
    """The fills worth most per character, while they fit `room` characters."""
    chosen: list[Fill] = []
    for one in sorted(found, key=lambda fill: -fill.value / max(fill.chars, 1)):
        if one.chars <= room:
            chosen.append(one)
            room -= one.chars
    return chosen


def apply(one: Group, taken: list[Candidate]) -> Group:
    """The group with `taken` chunks joined to its passages: the ranges rebuilt over every chunk,
    so a bridged gap makes two passages one. Each rebuilt range keeps the folded places and the
    questions of the passages it holds, and adds the questions its new chunks answer best."""
    if not taken:
        return one
    joined = [msgspec.structs.replace(chunk.hit, score=0.0) for chunk in taken]
    tags = {chunk.hit.seq: chunk.aspect for chunk in taken if chunk.aspect is not None}
    rebuilt: list[HitRange] = []
    for hit_range in ranges([hit for kept in one.ranges for hit in kept.hits] + joined):
        seqs = {hit.seq for hit in hit_range.hits}
        held = [kept for kept in one.ranges if kept.seq_start in seqs]
        aspects = [label for kept in held for label in kept.aspects]
        aspects += [label for seq, label in sorted(tags.items()) if seq in seqs]
        rebuilt.append(
            msgspec.structs.replace(
                hit_range,
                also_in=[place for kept in held for place in kept.also_in],
                aspects=list(dict.fromkeys(aspects)),
            )
        )
    return msgspec.structs.replace(one, ranges=rebuilt)
