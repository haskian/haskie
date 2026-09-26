"""Several questions in one search: each gets its turn at the slots, and each result says which
questions it answers.

When the parts of a question are answered in different places, one search of the whole question
tends to fill every slot with one part. The reranker scores one passage at a time, so it never
sees that another part went unanswered. So the caller names the parts, each part runs its own
search, and the lists take turns at the slots (round-robin), as TREC RAG pipelines give "the top
5 per query across 10 queries" (Samuel et al., ICTIR 2026, https://arxiv.org/abs/2603.08819).
Round-robin needs only ranks, so it works with and without a reranker or embeddings, unlike PM-2
or xQuAD, which need every result scored against every part on one calibrated scale.

Two rules keep the turns fair:
- A part already answered takes no turn. Its credit is how many picks overlap its own top
  `depth`, and in round r it takes a turn only while its credit is at most r. A pick made for
  another part that its own list ranks high counts for it too.
- A range next to or over a pick of the same document joins that pick rather than taking a slot:
  two parts that land on one passage get one passage, tagged with both.

`depth` is the share of the answer one part is owed, `ceil(limit / parts)`. A result is tagged
with every part that picked it, joined it, or ranks it in its own top `depth`. No IO, and no
scores: which part a result answers is read off ranks alone, since no calibrated relevance floor
exists yet.
"""

import math
from collections.abc import Iterator

import msgspec

from haskie.errors import InvalidInput
from haskie.search.passage import HitRange, PassageReference, ranges

MAX_QUESTIONS = 5  # Perplexity and OpenSearch cap several queries at 5; xQuAD degrades past few
MAX_QUESTION = 500  # judgement: a question, not a pasted document
MAX_CONTEXT = 200  # a long context outweighs a short question in the embedding and the reranker


class Questions(msgspec.Struct, frozen=True):
    """What the caller asked, checked: one question or several, and the background they share."""

    questions: list[str]
    context: str | None = None

    @property
    def queries(self) -> list[str]:
        """What each question is searched with: the shared context, then the question."""
        if self.context is None:
            return list(self.questions)
        return [f"{self.context}\n\n{question}" for question in self.questions]


def questions(asked: list[str], context: str | None = None) -> Questions:
    """The questions a caller sent, stripped and deduplicated in order, or `InvalidInput`.

    A blank question is dropped and a blank context is none. A list that deduplicates to one
    question is one question: the search it runs is the single one.
    """
    kept = list(dict.fromkeys(stripped for question in asked if (stripped := question.strip())))
    if not 1 <= len(kept) <= MAX_QUESTIONS:
        raise InvalidInput(f"q must hold 1..{MAX_QUESTIONS} questions, got {len(kept)}")
    for question in kept:
        if len(question) > MAX_QUESTION:
            raise InvalidInput(f"a question is at most {MAX_QUESTION} characters: {question[:40]}")
    shared = (context or "").strip() or None
    if shared is not None and len(shared) > MAX_CONTEXT:
        raise InvalidInput(f"context is at most {MAX_CONTEXT} characters, got {len(shared)}")
    return Questions(questions=kept, context=shared)


def depth(parts: int, limit: int) -> int:
    """How many of its own best results each part is owed: its share of `limit`. A limit below the
    number of parts would leave a part no slot at all, so it is refused."""
    if limit < parts:
        raise InvalidInput(f"limit must be at least the number of questions ({parts}), got {limit}")
    return math.ceil(limit / parts)


class Pick(msgspec.Struct):
    """One range the turns chose. `questions` are the parts that picked or joined it and
    `ranked_high` those whose own top `depth` it overlaps, by their position in the question
    list; together they are the parts it answers. `taken` counts the parts' ranges it holds:
    more than one where a range joined it rather than taking a slot."""

    span: HitRange
    questions: set[int]
    ranked_high: set[int]
    taken: int = 1


def interleave(ranked: list[list[HitRange]], depth: int, cap: int) -> list[Pick]:
    """The ranges of every part in the order the turns pick them, at most `cap`.

    `ranked` is one list per part, best first. The output is deeper than any answer: the fold
    after it (`collapse.ranges`) keeps the first `limit` that repeat nothing and folds the rest,
    so a freed slot goes to the next pick down.
    """
    tops = [found[:depth] for found in ranked]
    picks: list[Pick] = []
    cursors = [0] * len(ranked)
    turn = 0
    while len(picks) < cap and any(
        at < len(found) for at, found in zip(cursors, ranked, strict=True)
    ):
        for part, found in enumerate(ranked):
            if len(picks) >= cap:
                break
            credit = sum(1 for pick in picks if part in pick.ranked_high)
            if cursors[part] >= len(found) or credit > turn:
                continue
            _take(picks, found[cursors[part]], part, tops)
            cursors[part] += 1
        turn += 1
    return picks


def _ranked_high(span: HitRange, tops: list[list[HitRange]]) -> set[int]:
    """The parts whose own top ranges `span` overlaps."""
    return {part for part, top in enumerate(tops) if any(_near(span, one, 0) for one in top)}


def _near(one: HitRange, other: HitRange, gap: int) -> bool:
    """Whether two ranges of one document overlap (gap 0) or touch (gap 1)."""
    first, second = one.hits[0], other.hits[0]
    if (first.collection, first.document) != (second.collection, second.document):
        return False
    return one.seq_start <= other.seq_end + gap and other.seq_start <= one.seq_end + gap


def _take(picks: list[Pick], candidate: HitRange, part: int, tops: list[list[HitRange]]) -> None:
    """`candidate` as a new pick, or joined into the picks of its document it touches or
    overlaps: those merge into the earliest of them, which keeps its place in the order."""
    touching = [pick for pick in picks if _near(pick.span, candidate, 1)]
    if not touching:
        picks.append(Pick(candidate, {part}, _ranked_high(candidate, tops)))
        return
    first, *rest = touching
    first.span = _merged([first.span, *(pick.span for pick in rest), candidate])
    first.questions |= {part}.union(*(pick.questions for pick in rest))
    first.taken += 1 + sum(pick.taken for pick in rest)
    first.ranked_high = _ranked_high(first.span, tops)
    for pick in rest:
        picks.remove(pick)


def _merged(spans: list[HitRange]) -> HitRange:
    """One range over the chunks of ranges that touch or overlap in one document. A chunk two of
    them hold counts once, with the hit of the range listed first."""
    hits = {hit.seq: hit for span in reversed(spans) for hit in span.hits}
    (merged,) = ranges(sorted(hits.values(), key=lambda hit: hit.seq))
    return merged


def tagged(kept: list[HitRange], picks: list[Pick], labels: list[str]) -> list[HitRange]:
    """The kept ranges, each with the questions it answers (`HitRange.aspects`): its own pick's,
    and those of every place folded into it (`also_in`), in the order they were asked.

    The fold keeps one of the picks as each result and lists others under it, so every place is
    looked up by the chunks it covers - picks never overlap, so those are unique.
    """
    by_place = {
        _place(pick.span.hits[0].collection, pick.span.hits[0].document, pick.span): (
            pick.questions | pick.ranked_high
        )
        for pick in picks
    }

    def answered(one: HitRange) -> list[str]:
        first = one.hits[0]
        parts = set(by_place[_place(first.collection, first.document, one)])
        for place in _walk(one.also_in):
            parts |= by_place.get(_place(place.collection, place.document, place), set())
        return [labels[part] for part in sorted(parts)]

    return [msgspec.structs.replace(one, aspects=answered(one)) for one in kept]


def _place(
    collection: str, document: str, span: HitRange | PassageReference
) -> tuple[str, str, int, int]:
    return (collection, document, span.seq_start, span.seq_end)


def _walk(references: list[PassageReference]) -> Iterator[PassageReference]:
    """Every place in a tree of references, parents before their children."""
    for reference in references:
        yield reference
        yield from _walk(reference.also_in)
