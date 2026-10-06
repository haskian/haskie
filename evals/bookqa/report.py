"""The report of one run: answerable questions by search mode, then by mode and source and by mode
and query type; unanswerable ones in a table of their own, since Recall@k means nothing for them.
With graded judgments (`qrels.py`), the same groups scored against them too.

Run as a module to render an existing run's report again, with the judgments as they are now:
`python -m evals.bookqa.report evals/bookqa/reports/<run>/outcomes.jsonl`.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from functools import partial
from itertools import groupby
from pathlib import Path

import msgspec

from evals.bookqa import metrics, qrels, schema
from evals.bookqa.metrics import K, Outcome

ANSWERABLE = ["n", *(f"R@{k}" for k in K), "MRR", "nDCG@10", "doc@10", "empty", "kB", "ms p50"]
UNANSWERABLE = ["n", "abstained", "top score", "kB", "ms p50"]
JUDGED = ["n", *(f"S@{k}" for k in K), "MRR", "nDCG@10", "judged@10"]


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


def _judged_row(grades: qrels.Grades, group: list[Outcome]) -> list[str]:
    scored = [metrics.judged(o, grades) for o in group]
    return [
        str(len(group)),
        *(f"{_mean([s.success[k] for s in scored]):.2f}" for k in K),
        f"{_mean([s.mrr for s in scored]):.2f}",
        f"{_mean([s.ndcg for s in scored]):.2f}",
        f"{_mean([s.judged for s in scored]):.2f}",
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


def render(outcomes: list[Outcome], grades: qrels.Grades | None = None) -> str:
    answerable = [o for o in outcomes if o.answerable]
    unanswerable = [o for o in outcomes if not o.answerable]
    parts = [
        "# Book query retrieval",
        "Passage-level recall of the gold quotes in haskie's top 10 (`metrics.py`). `doc@10`: a "
        "relevant document anywhere in the top 10. `empty`: no result at all. `abstained`: an "
        "unanswerable question that got no result - the right answer.",
    ]
    if answerable and grades:
        parts.append(
            "Judged: every returned passage graded on its own (`qrels.py`). `S@k`: a passage "
            "stating the answer in the top k. `judged@10`: the share of the top 10 graded - under "
            "1.00, run `eval:bookqa:judge` on this run's outcomes."
        )
        parts += [
            _table(f"Judged, {name}", fields, answerable, JUDGED, partial(_judged_row, grades))
            for name, fields in BY.items()
        ]
    if unanswerable and grades:
        ids = {o.id for o in unanswerable}
        found = sorted({rid for (rid, _), g in grades.items() if rid in ids and g == qrels.ANSWERS})
        if found:
            listed = "\n".join(f"- {rid}" for rid in found)
            parts.append(
                "## Unanswerable, but a passage was judged to answer them\n\n"
                f"Review these records: the book may answer them after all.\n\n{listed}"
            )
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outcomes", type=Path, help="a run's outcomes.jsonl")
    parser.add_argument("--judgments", type=Path, default=qrels.JUDGMENTS)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset.jsonl",
        help="score only the questions still in it: a record removed since the run drops out",
    )
    args = parser.parse_args(argv)
    decoder = msgspec.json.Decoder(Outcome)
    lines = args.outcomes.read_text(encoding="utf-8").splitlines()
    current = {record.id for record in schema.load(args.dataset)[0]}
    outcomes = [o for o in (decoder.decode(line) for line in lines if line) if o.id in current]
    text = render(outcomes, qrels.grades(qrels.load(args.judgments)))
    target = args.outcomes.with_name("report.md")
    target.write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
