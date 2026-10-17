"""How one ranked list of search results scores against a record's gold passages.

A result counts for a gold passage when it is from the same document and holds most of the quote:
at least `COVERAGE` of the quote's four-word runs, compared letters and digits only
(`sources.compact`). Not the whole quote, because haskie cuts passages at its own chunk
boundaries and a quote can straddle two; not the page, because a page holds several passages.

Recall@k is the share of a record's gold passages that some result in the top k matches; MRR the
reciprocal rank of the first result matching any of them; nDCG@10 gives a result gain 1 when it
matches a gold passage no higher result matched. An unanswerable record is scored apart: it has
no passage to recall, and what counts is whether the search returned nothing (`abstained`).

`judged` scores the same ranking against graded judgments instead (`qrels.py`): every passage
returned graded on its own, so an answer the book gives outside the gold quotes counts too.
Success@k is whether a passage stating the answer (grade 2) is in the top k; MRR the reciprocal
rank of the first; nDCG@10 gains 2^grade - 1 against the best order of every passage judged for
the question.
"""

from __future__ import annotations

import math
import re

import msgspec

from evals.bookqa import qrels, sources
from evals.bookqa.schema import Passage, Record

K = (1, 5, 10)
COVERAGE = 0.5
RUN = 4  # words per run a quote is cut into for matching


class Found(msgspec.Struct):
    """One ranked search result, as much of it as scoring and review need."""

    document: str
    text: str
    score: float
    header: str = ""
    page_start: int | None = None
    page_end: int | None = None


class Scores(msgspec.Struct):
    recall: dict[int, float]  # k -> recall@k
    mrr: float
    ndcg: float  # @10
    document_hit: bool  # a relevant document anywhere in the top 10
    first_match: int | None  # 1-based rank of the first matching result


class Outcome(msgspec.Struct):
    """One question searched in one mode: what came back, what it cost, and how it scored."""

    id: str
    source: str
    query_type: str
    answerable: bool
    mode: str
    seconds: float
    bytes: int  # the response body as sent
    results: list[Found]
    scores: Scores | None  # None for an unanswerable question
    abstained: bool


def _readable(text: str) -> str:
    """A result's text without the markup haskie keeps in it: tags and link targets."""
    return re.sub(r"\]\([^)]*\)", "]", re.sub(r"<[^>]+>", "", text))


def _runs(quote: str) -> list[str]:
    words = quote.split()
    if len(words) <= RUN:
        return [sources.compact(quote)]
    return [sources.compact(" ".join(words[i : i + RUN])) for i in range(len(words) - RUN + 1)]


def matches(found: Found, passage: Passage) -> bool:
    if found.document != passage.document:
        return False
    runs = [run for run in _runs(passage.quote) if run]
    if not runs:
        return False
    here = sources.compact(_readable(found.text))
    return sum(run in here for run in runs) / len(runs) >= COVERAGE


def score(record: Record, results: list[Found]) -> Scores:
    """An answerable record's scores for the ranked `results`, best first."""
    gold = record.relevant_passages
    matched_at: dict[int, int] = {}  # gold passage index -> first rank that matched it
    first: int | None = None
    gains = []
    for rank, found in enumerate(results[: max(K)], start=1):
        hits = [i for i, passage in enumerate(gold) if matches(found, passage)]
        first = first or (rank if hits else None)
        fresh = [i for i in hits if i not in matched_at]
        for i in fresh:
            matched_at[i] = rank
        gains.append(1.0 if fresh else 0.0)
    recall = {k: sum(r <= k for r in matched_at.values()) / len(gold) for k in K}
    dcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), max(K)) + 1))
    return Scores(
        recall=recall,
        mrr=1 / first if first else 0.0,
        ndcg=dcg / ideal,
        document_hit=any(f.document in record.relevant_documents for f in results[: max(K)]),
        first_match=first,
    )


def abstained(results: list[Found]) -> bool:
    """The search returned nothing: for an unanswerable question, the right answer."""
    return not results


class Judged(msgspec.Struct):
    success: dict[int, float]  # k -> whether a passage stating the answer is in the top k
    mrr: float
    ndcg: float  # @10, graded
    judged: float  # share of the top 10 with a judgment


def judged(outcome: Outcome, grades: qrels.Grades) -> Judged:
    """`outcome`'s ranking scored against the graded judgments of its question."""
    top = outcome.results[: max(K)]
    got = [grades.get((outcome.id, qrels.key(f.document, f.text))) for f in top]
    first = next((rank for rank, g in enumerate(got, start=1) if g == qrels.ANSWERS), None)
    pool = sorted((g for (rid, _), g in grades.items() if rid == outcome.id), reverse=True)
    dcg = sum((2 ** (g or 0) - 1) / math.log2(rank + 1) for rank, g in enumerate(got, start=1))
    ideal = sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(pool[: max(K)], 1))
    return Judged(
        success={k: float(first is not None and first <= k) for k in K},
        mrr=1 / first if first else 0.0,
        ndcg=dcg / ideal if ideal else 0.0,
        judged=sum(g is not None for g in got) / len(top) if top else 1.0,
    )
