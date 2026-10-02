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
- A range over a pick, or next to one in the same section, joins that pick rather than taking a
  slot: two parts that land on one passage get one passage, tagged with both.

`depth` is the share of the answer one part is owed, `ceil(limit / parts)`. A result is tagged
with every part that picked it, joined it, or ranks it in its own top `depth`: the turns read
ranks alone. With a reranker on, its score has a scale and a floor (`min_rerank_score`), so the
tags come from the reranker's scores instead (`tagged`). No IO.
"""

import math
from collections.abc import Iterator

import msgspec

from haskie.collection.index import ChunkKey, Hit, chunk_key
from haskie.errors import InvalidInput
from haskie.search import probe
from haskie.search.passage import HitRange, PassageReference, continues, fold, rejoin
from haskie.settings import ScoreFold

MAX_QUESTIONS = 5  # Perplexity and OpenSearch cap several queries at 5; xQuAD degrades past few
MAX_QUESTION = 500  # judgement: a question, not a pasted document
MAX_CONTEXT = 200  # a long context outweighs a short question in the embedding


class Questions(msgspec.Struct, frozen=True):
    """What the caller asked, checked: one question or several, and the background they share."""

    questions: list[str]
    context: str | None = None

    @property
    def framed(self) -> list[str]:
        """What the query embedding reads each question as: the shared context, then the question,
        as it reads a chunk under its heading path, so the context steers what a question means.
        Full-text search, the word scores and the reranker read the question alone: the context's
        words would make any question match any passage that shares them, one the sources say
        nothing about included. The reranker reads this form only when `rerank_with_context` is
        on (`flow.Search.rerank_query`)."""
        if self.context is None:
            return list(self.questions)
        return [f"{self.context}\n\n{question}" for question in self.questions]

    def asked(self, vectors: list[list[float] | None] | None = None) -> list[probe.Question]:
        """Each question as the steps after the ranking read it, given the embedding of its framed
        form (None for a lexical search, or none planned). Tagged with itself only when several
        were asked."""
        several = len(self.questions) > 1
        vectors = vectors or [None] * len(self.questions)
        return [
            probe.Question(vector=vector, asked=asked, label=asked if several else None)
            for asked, vector in zip(self.questions, vectors, strict=True)
        ]


def question(asked: str) -> str:
    """The one question of a single-question search, stripped, or `InvalidInput` when blank: a
    blank search finds nothing, and would be logged as an unnamed gap."""
    stripped = asked.strip()
    if not stripped:
        raise InvalidInput("q is empty")
    return stripped


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


def interleave(ranked: list[list[HitRange]], depth: int, cap: int, how: ScoreFold) -> list[Pick]:
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
            _take(picks, found[cursors[part]], part, tops, how)
            cursors[part] += 1
        turn += 1
    return picks


def _ranked_high(span: HitRange, tops: list[list[HitRange]]) -> set[int]:
    """The parts whose own top ranges `span` overlaps."""
    return {part for part, top in enumerate(tops) if any(_overlaps(span, one) for one in top)}


def _overlaps(one: HitRange, other: HitRange) -> bool:
    """Whether two ranges of one document share a chunk."""
    first, second = one.hits[0], other.hits[0]
    if (first.collection, first.document_id) != (second.collection, second.document_id):
        return False
    return one.seq_start <= other.seq_end and other.seq_start <= one.seq_end


def _joins(pick: HitRange, candidate: HitRange) -> bool:
    """Whether `candidate` is part of the same passage as `pick`: over it, or right next to it
    in the same section (`passage.continues`)."""
    return (
        _overlaps(pick, candidate)
        or continues(pick.hits[-1], candidate.hits[0])
        or continues(candidate.hits[-1], pick.hits[0])
    )


def _take(
    picks: list[Pick],
    candidate: HitRange,
    part: int,
    tops: list[list[HitRange]],
    how: ScoreFold,
) -> None:
    """`candidate` as a new pick, or joined into the picks it overlaps or continues in the same
    section: those merge into the earliest of them, which keeps its place in the order."""
    touching = [pick for pick in picks if _joins(pick.span, candidate)]
    if not touching:
        picks.append(Pick(candidate, {part}, _ranked_high(candidate, tops)))
        return
    first, *rest = touching
    # overlapping or continuing each other in one section, they rebuild into one range
    (first.span,) = rejoin([first.span, *(pick.span for pick in rest), candidate], how)
    first.questions |= {part}.union(*(pick.questions for pick in rest))
    first.taken += 1 + sum(pick.taken for pick in rest)
    first.ranked_high = _ranked_high(first.span, tops)
    for pick in rest:
        picks.remove(pick)


def tagged(
    kept: list[HitRange],
    picks: list[Pick],
    labels: list[str],
    scans: list[dict[ChunkKey, float]],
    by_score: bool,
    how: ScoreFold,
) -> list[HitRange]:
    """The kept ranges, each with how well it matched every question whose own ranking (`scans`,
    one per question, chunk to score) holds its chunks or its folded places' (`aspect_scores`:
    the best such chunk, `question_scores`), and with the questions it answers (`aspects`), in
    the order they were asked.

    A question it answers is, with a reranker on (`by_score`), one whose ranking holds it: the
    reranker's score has a scale, and chunks under its floor were dropped from every ranking, so
    being there means the reranker judged it an answer. The questions' scores then share that
    scale, so each chunk scores its best question and the range folds those by `how`, as any
    range does. Without one, a score says nothing across questions, so the answered questions are
    those that picked it or rank it in their own top (`Pick`), and the range keeps the scores it
    was picked with.

    The fold keeps one of the picks as each result and lists others under it, so every place is
    looked up by the chunks it covers. Picks never overlap, so those are unique.
    """
    by_place = {
        _place(pick.span.hits[0].collection, pick.span.hits[0].document_id, pick.span): (
            pick.questions | pick.ranked_high
        )
        for pick in picks
    }

    def answered(one: HitRange) -> list[str]:
        first = one.hits[0]
        parts = set(by_place[_place(first.collection, first.document_id, one)])
        for place in _walk(one.also_in):
            parts |= by_place.get(_place(place.collection, place.document_id, place), set())
        return [labels[part] for part in sorted(parts)]

    by_label = dict(zip(labels, scans, strict=True))

    def keys(one: HitRange) -> list[ChunkKey]:
        return [chunk_key(hit) for hit in one.hits] + [
            (place.collection, place.document_id, seq)
            for place in _walk(one.also_in)
            for seq in range(place.seq_start, place.seq_end + 1)
        ]

    def best(hit: Hit) -> Hit:
        """The chunk at its best question's score."""
        key = chunk_key(hit)
        return msgspec.structs.replace(
            hit, score=max([hit.score, *(scan[key] for scan in scans if key in scan)])
        )

    def lifted(one: HitRange) -> HitRange:
        """Each chunk at its best question's score, and the range folded over them."""
        hits = [best(hit) for hit in one.hits]
        return msgspec.structs.replace(one, hits=hits, score=fold([h.score for h in hits], how))

    tagged: list[HitRange] = []
    for one in kept:
        scores = question_scores(keys(one), by_label)
        if by_score:
            one = lifted(one)
            answers = [label for label in labels if label in scores]
        else:
            answers = answered(one)
        tagged.append(msgspec.structs.replace(one, aspects=answers, aspect_scores=scores))
    return tagged


def question_scores(
    keys: list[ChunkKey], scans: dict[str, dict[ChunkKey, float]]
) -> dict[str, float]:
    """How well chunks `keys` matched each question whose own ranking (`scans`, question to chunk
    to score) holds any of them: its best such chunk, in the order `scans` lists. The best chunk,
    not a fold over them, so the score stays on the reranker's scale, the one its floor
    (`min_rerank_score`) is set on and the one each chunk was judged on."""
    found: dict[str, float] = {}
    for label, scan in scans.items():
        values = [scan[key] for key in keys if key in scan]
        if values:
            found[label] = max(values)
    return found


def _place(
    collection: str, document_id: str, span: HitRange | PassageReference
) -> tuple[str, str, int, int]:
    return (collection, document_id, span.seq_start, span.seq_end)


def _walk(references: list[PassageReference]) -> Iterator[PassageReference]:
    """Every place in a tree of references, parents before their children."""
    for reference in references:
        yield reference
        yield from _walk(reference.also_in)
