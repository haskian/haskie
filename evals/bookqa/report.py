"""The report of one run: answerable questions by search mode, then by mode and source and by mode
and query type; unanswerable ones in a table of their own, since Recall@k means nothing for them.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from itertools import groupby

from evals.bookqa.metrics import K, Outcome

ANSWERABLE = ["n", *(f"R@{k}" for k in K), "MRR", "nDCG@10", "doc@10", "empty", "kB", "ms p50"]
UNANSWERABLE = ["n", "abstained", "top score", "kB", "ms p50"]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _answerable_row(group: list[Outcome]) -> list[str]:
    scored = [o.scores for o in group if o.scores is not None]
    return [
        str(len(group)),
        *(f"{_mean([s.recall[k] for s in scored]):.2f}" for k in K),
        f"{_mean([s.mrr for s in scored]):.2f}",
        f"{_mean([s.ndcg for s in scored]):.2f}",
        f"{_mean([float(s.document_hit) for s in scored]):.2f}",
        str(sum(o.abstained for o in group)),
        *_cost(group),
    ]


def _unanswerable_row(group: list[Outcome]) -> list[str]:
    tops = [o.results[0].score for o in group if o.results]
    return [
        str(len(group)),
        f"{sum(o.abstained for o in group)}/{len(group)}",
        f"{_mean(tops):.3f}" if tops else "-",
        *_cost(group),
    ]


def _cost(group: list[Outcome]) -> list[str]:
    return [
        f"{_mean([o.bytes / 1000 for o in group]):.1f}",
        f"{statistics.median(o.seconds * 1000 for o in group):.0f}",
    ]


def _table(
    title: str,
    fields: tuple[str, ...],
    outcomes: list[Outcome],
    columns: list[str],
    row: Callable[[list[Outcome]], list[str]],
) -> str:
    """One markdown table, a row per value of `fields` (attributes of `Outcome`)."""

    def key(o: Outcome) -> tuple[str, ...]:
        return tuple(str(getattr(o, field)) for field in fields)

    lines = [f"## {title}", "", f"| {' | '.join([*fields, *columns])} |"]
    lines.append(f"|{'---|' * (len(fields) + len(columns))}")
    for values, group in groupby(sorted(outcomes, key=key), key=key):
        lines.append(f"| {' | '.join([*values, *row(list(group))])} |")
    return "\n".join(lines)


BY = {
    "by mode": ("mode",),
    "by mode and source": ("mode", "source"),
    "by mode and query type": ("mode", "query_type"),
}


def render(outcomes: list[Outcome]) -> str:
    answerable = [o for o in outcomes if o.answerable]
    unanswerable = [o for o in outcomes if not o.answerable]
    parts = [
        "# Book query retrieval",
        "Passage-level recall of the gold quotes in haskie's top 10 (`metrics.py`). `doc@10`: a "
        "relevant document anywhere in the top 10. `empty`: no result at all. `abstained`: an "
        "unanswerable question that got no result - the right answer.",
    ]
    if answerable:
        parts += [
            _table(f"Answerable, {name}", fields, answerable, ANSWERABLE, _answerable_row)
            for name, fields in BY.items()
        ]
    if unanswerable:
        parts += [
            _table(f"Unanswerable, {name}", fields, unanswerable, UNANSWERABLE, _unanswerable_row)
            for name, fields in BY.items()
        ]
    return "\n\n".join(parts) + "\n"
