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
- Every passage grows outward by the run of chunks next to it whose values sum highest, when that
  is above 0 (`run`), up to `max_passage_grow` chunks and never out of its section. Into a gap it
  grows at most halfway, so the passages on either side never reach for the same chunk; the
  first half goes to the passage before it.

`run` and `value` are the one growth rule of a search: a passage too short to stand alone grows
by them too (`thin`), against the scanned hits rather than the kept ones.

`grow_bias` moves every value before it is summed (`biased`), whichever way it was valued: above
0 a passage grows more eagerly, below 0 more strictly, and at -1 nothing grows.

With a reranker on, `fill_values = absolute` (an experiment) values a chunk as dsRAG does instead
(`absolute`): its reranker score spread by the reranker's calibrated beta curve, minus a fixed
penalty. It needs no kept chunks to compare with, but it trusts the calibration: an uncalibrated
reranker's curve is the identity.

What is added is bounded by the room the answer's budget (`max_answer_chars`) leaves after its
sections (`section.within`), spent on the fills worth most per character first. No IO here:
`retrieval.fill` reads and scores the chunks.
"""

import math

import msgspec

from haskie.collection.index import ChunkKey, Hit, chunk_key
from haskie.search.passage import continues, part, rejoin
from haskie.search.section import Group
from haskie.settings import ScoreFold


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


IRRELEVANT_PENALTY = 0.18  # dsRAG's "balanced" preset: segments of about 4 to 10 chunks


def absolute(score: float, beta_a: float, beta_b: float) -> float:
    """A reranker score in 0 to 1 as dsRAG's Relevant Segment Extraction values a chunk: spread by
    the reranker's beta curve (`beta_cdf`), so its scores are about even over 0 to 1, minus
    `IRRELEVANT_PENALTY`. A chunk the reranker read counts with its own score, where dsRAG decays
    a chunk by its rank: every chunk here was scored, none was only ranked."""
    return beta_cdf(score, beta_a, beta_b) - IRRELEVANT_PENALTY


def beta_cdf(x: float, a: float, b: float) -> float:
    """The regularized incomplete beta function I_x(a, b): the beta distribution's cumulative
    function, by its continued fraction (Numerical Recipes, `betacf`). The identity at a = b = 1."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    # the fraction converges fast on the side of the mean it is evaluated on
    if x < (a + 1) / (a + b + 2):
        return front * _fraction(x, a, b) / a
    return 1.0 - front * _fraction(1.0 - x, b, a) / b


def _fraction(x: float, a: float, b: float) -> float:
    """The continued fraction of the incomplete beta function, by Lentz's method: two terms a
    round, the even one and the odd one, until a round changes it by under 1e-12."""
    tiny = 1e-300
    c, d = 1.0, _nonzero(1.0 - (a + b) * x / (a + 1.0))
    h = d
    for m in range(1, 300):
        even = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        odd = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        for term in (even, odd):
            d = _nonzero(1.0 + term * d)
            c = 1.0 + term / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-12:
            break
    return h


def _nonzero(x: float) -> float:
    """1 / x, with x kept off 0 as Lentz's method needs."""
    return 1.0 / (x if abs(x) > 1e-300 else 1e-300)


def biased(candidates: dict[ChunkKey, Candidate], bias: float) -> dict[ChunkKey, Candidate]:
    """Each candidate's value moved by `bias` (`grow_bias`), however it was valued: above 0 a
    weaker chunk pays its way, below 0 only a stronger one does."""
    if not bias:
        return candidates
    return {
        key: msgspec.structs.replace(one, value=one.value + bias) for key, one in candidates.items()
    }


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
    each passage, into a gap only as far as its half (`fill` module docstring)."""
    spans = sorted(one.ranges, key=lambda hit_range: hit_range.seq_start)
    found: list[Fill] = []
    for before, after in zip(spans, spans[1:], strict=False):
        keys = [
            (one.collection, one.document_id, seq)
            for seq in range(before.seq_end + 1, after.seq_start)
        ]
        if keys and all(key in candidates for key in keys):
            chunks = [candidates[key] for key in keys]
            if sum(chunk.value for chunk in chunks) > 0:
                found.append(Fill(at, chunks))
    # the chunks between each two passages; the section's ends are open as far as `reach`
    between = zip(spans, spans[1:], strict=False)
    gaps = [2 * reach, *(b.seq_start - a.seq_end - 1 for a, b in between), 2 * reach]
    for index, span in enumerate(spans):
        # a gap of g chunks: the passage before it may take ceil(g / 2), the one after floor(g / 2)
        back, ahead = gaps[index] // 2, (gaps[index + 1] + 1) // 2
        sides = grow(span.hits[0], span.hits[-1], candidates, min(back, reach), min(ahead, reach))
        found += [Fill(at, side) for side in sides if side]
    return found


def grow(
    first: Hit, last: Hit, candidates: dict[ChunkKey, Candidate], back: int, ahead: int
) -> list[list[Candidate]]:
    """What a passage from `first` to `last` grows by out of `candidates`: the best `run` of up to
    `back` chunks before it and `ahead` after it, never past a heading (`passage.continues`)."""
    return [run(_outward(first, candidates, back, -1)), run(_outward(last, candidates, ahead, 1))]


def _outward(
    edge: Hit, candidates: dict[ChunkKey, Candidate], reach: int, step: int
) -> list[Candidate]:
    """The read chunks past `edge` one way (`step` -1 or 1), nearest first, up to `reach` of
    them, while each continues the section: what a `run` is taken from."""
    found: list[Candidate] = []
    current = edge
    for _ in range(reach):
        chunk = candidates.get((current.collection, current.document_id, current.seq + step))
        if chunk is None or not continues(
            *((chunk.hit, current) if step < 0 else (current, chunk.hit))
        ):
            break
        found.append(chunk)
        current = chunk.hit
    return found


def choose(found: list[Fill], room: int) -> list[Fill]:
    """The fills worth most per character, while they fit `room` characters. A bridged gap and the
    runs into it share chunks: a fill brings only the chunks none chosen before holds, and only
    when those are still worth taking."""
    chosen: list[Fill] = []
    taken: set[ChunkKey] = set()
    for one in sorted(found, key=lambda fill: -fill.value / max(fill.chars, 1)):
        new = Fill(one.group, [chunk for chunk in one.chunks if chunk_key(chunk.hit) not in taken])
        if new.chunks and new.value > 0 and new.chars <= room:
            chosen.append(new)
            taken.update(chunk_key(chunk.hit) for chunk in new.chunks)
            room -= new.chars
    return chosen


def apply(one: Group, taken: list[Candidate], how: ScoreFold) -> Group:
    """The group with `taken` chunks joined to its passages (`passage.rejoin`), so a bridged gap
    makes two passages one. Each joined chunk brings the question it answers best."""
    if not taken:
        return one
    added = [part(chunk.hit, [chunk.aspect] if chunk.aspect else None) for chunk in taken]
    return msgspec.structs.replace(one, ranges=rejoin([*one.ranges, *added], how))
