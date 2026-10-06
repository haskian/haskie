"""One row per task and arm, aggregated over `--samples` independent runs of that cell.

Iteration 1 ran each cell once - a table of means over one sample would have just been the
sample dressed up, so it reported a single pass/fail instead. This is iteration 2: enough
repeats per cell to report a rate instead of a coin flip. `results` must already be grouped by
cell - consecutive entries sharing a (task, arm) pair - which is how `run.py` produces them
(arm, then task, then sample, as nested loops), so no separate sort is needed here."""

from __future__ import annotations

from itertools import groupby
from pathlib import Path

HEAD = (
    f"{'task':<28}{'arm':<5}{'n':>3}{'correct':>8}{'tests':>10}{'disc':>9}"
    f"{'search1st':>10}{'illegit':>8}{'sub':>6}{'evid':>9}{'secs':>7}"
    f"{'turns':>7}{'ktok':>7}{'usd':>7}"
)


def _cell(passed: int, total: int) -> str:
    return f"{passed}/{total}" if total else "n/a"


def table(results: list) -> str:
    lines = [HEAD, "-" * len(HEAD)]
    for (task, arm), group in groupby(results, key=lambda r: (r.task, r.arm)):
        cell = list(group)
        n = len(cell)

        search_first = [
            r.behaviour.search_first for r in cell if r.behaviour.search_first is not None
        ]
        sub_rates = [
            r.behaviour.substitution_rate for r in cell if r.behaviour.substitution_rate is not None
        ]
        sub = f"{sum(sub_rates) / len(sub_rates):.2f}" if sub_rates else "n/a"
        tests = _cell(sum(r.tests_passed for r in cell), sum(r.tests_total for r in cell))
        disc = _cell(
            sum(r.discriminating_passed for r in cell), sum(r.discriminating_total for r in cell)
        )
        evid = _cell(sum(len(r.retrieved) for r in cell), sum(r.evidence_total for r in cell))

        lines.append(
            f"{task:<28}{arm:<5}{n:>3}"
            f"{_cell(sum(1 for r in cell if r.correct), n):>8}"
            f"{tests:>10}"
            f"{disc:>9}"
            f"{_cell(sum(search_first), len(search_first)):>10}"
            f"{sum(r.behaviour.illegitimate_doc_store_lookups for r in cell):>8}"
            f"{sub:>6}"
            f"{evid:>9}"
            f"{sum(r.seconds for r in cell) / n:>7.1f}"
            f"{sum(r.cost.turns for r in cell) / n:>7.1f}"
            f"{sum(r.cost.tokens for r in cell) / n / 1000:>7.0f}"
            f"{sum(r.cost.usd for r in cell) / n:>7.3f}"
        )
    return "\n".join(lines)


def write_report(results: list, runs: Path) -> None:
    text = table(results)
    (runs / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
