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
  above 0 (`run`), up to `max_passage_grow` chunks and never out of its section.

`run` and `value` are the one growth rule of a search: a passage too short to stand alone grows
by them too (`thin`), against the scanned hits rather than the kept ones.

What is added is bounded by the room the answer's budget (`max_answer_chars`) leaves after its
sections (`section.within`), spent on the fills worth most per character first. No IO here:
`retrieval.fill` reads and scores the chunks.
"""

import msgspec

from haskie.collection.index import ChunkKey, Hit
from haskie.search.passage import continues, part, rejoin
from haskie.search.section import Group


class Candidate(msgspec.Struct, frozen=True):
    """A chunk next to a passage, as a growth weighs it."""

    hit: Hit
    value: float  # around the ranked chunks: 0 the median one, 1 the best, below 0 weaker
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
    """`score` around the ranked chunks' own scores: 0 at their median `floor`, 1 at their best
    `top`, clipped to [-1, 1]. When every ranked chunk scores the same, a chunk at least as good
    is worth 1 and a weaker one -1."""
    if top <= floor:
        return 1.0 if score >= floor else -1.0
    return max(-1.0, min(1.0, (score - floor) / (top - floor)))


def run(outward: list[Candidate]) -> list[Candidate]:
    """The chunks outward from a passage, nearest first, up to the point where their values sum
    highest; none when no prefix sums above 0. A weak chunk comes in only when stronger ones past
    it pay for it."""
    best: list[Candidate] = []
    total = best_total = 0.0
    for at, chunk in enumerate(outward):
        total += chunk.value
        if total > best_total:
            best, best_total = outward[: at + 1], total
    return best


def near(one: Group, reach: int) -> set[int]:
    """The chunks of a group's section a fill could take: `reach` either side of each passage,
    and not the passages' own."""
    held = {hit.seq for hit in one.hits}
    wanted = {
        seq
        for hit_range in one.ranges
        for seq in range(hit_range.seq_start - reach, hit_range.seq_end + reach + 1)
    }
    return {seq for seq in wanted if one.section.seq_start <= seq <= one.section.seq_end} - held


def fills(at: int, one: Group, candidates: dict[ChunkKey, Candidate], reach: int) -> list[Fill]:
    """What group `at` could take out of `candidates`: each gap between two of its passages whose
    chunks are all read and sum above 0, and the best `run` of up to `reach` chunks outward from
    its first and last passage."""
    spans = sorted(one.ranges, key=lambda hit_range: hit_range.seq_start)
    found: list[Fill] = []
    for before, after in zip(spans, spans[1:], strict=False):
        keys = [
            (one.collection, one.document, seq)
            for seq in range(before.seq_end + 1, after.seq_start)
        ]
        if keys and all(key in candidates for key in keys):
            chunks = [candidates[key] for key in keys]
            if sum(chunk.value for chunk in chunks) > 0:
                found.append(Fill(at, chunks))
    found += [
        Fill(at, side)
        for side in grow(spans[0].hits[0], spans[-1].hits[-1], candidates, reach)
        if side
    ]
    return found


def grow(
    first: Hit, last: Hit, candidates: dict[ChunkKey, Candidate], reach: int
) -> list[list[Candidate]]:
    """What a passage from `first` to `last` grows by out of `candidates`: the best `run` of up to
    `reach` chunks before it and after it, never past a heading (`passage.continues`)."""
    return [run(_outward(first, candidates, reach, -1)), run(_outward(last, candidates, reach, 1))]


def _outward(
    edge: Hit, candidates: dict[ChunkKey, Candidate], reach: int, step: int
) -> list[Candidate]:
    """The read chunks past `edge` one way (`step` -1 or 1), nearest first, up to `reach` of
    them, while each continues the section: what a `run` is taken from."""
    found: list[Candidate] = []
    current = edge
    for _ in range(reach):
        chunk = candidates.get((current.collection, current.document, current.seq + step))
        if chunk is None or not continues(
            *((chunk.hit, current) if step < 0 else (current, chunk.hit))
        ):
            break
        found.append(chunk)
        current = chunk.hit
    return found


def choose(found: list[Fill], room: int) -> list[Fill]:
    """The fills worth most per character, while they fit `room` characters."""
    chosen: list[Fill] = []
    for one in sorted(found, key=lambda fill: -fill.value / max(fill.chars, 1)):
        if one.chars <= room:
            chosen.append(one)
            room -= one.chars
    return chosen


def apply(one: Group, taken: list[Candidate]) -> Group:
    """The group with `taken` chunks joined to its passages (`passage.rejoin`), so a bridged gap
    makes two passages one. Each joined chunk brings the question it answers best."""
    if not taken:
        return one
    added = [part(chunk.hit, [chunk.aspect] if chunk.aspect else None) for chunk in taken]
    return msgspec.structs.replace(one, ranges=rejoin([*one.ranges, *added]))
