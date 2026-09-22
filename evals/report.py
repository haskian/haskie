"""One row per task and arm. Deliberately no averaging: iteration 1 runs each cell once, and a
table of means over one sample would just be the sample dressed up."""

from __future__ import annotations

from pathlib import Path

HEAD = (
    f"{'task':<18}{'arm':<5}{'correct':>8}{'tests':>8}{'disc':>8}"
    f"{'search1st':>10}{'illegit':>8}{'sub':>6}{'evid':>7}{'secs':>7}"
)


def _cell(passed: int, total: int) -> str:
    return f"{passed}/{total}"


def _bool(value: bool | None) -> str:
    return "n/a" if value is None else ("yes" if value else "no")


def table(results: list) -> str:
    lines = [HEAD, "-" * len(HEAD)]
    for r in results:
        b = r.behaviour
        sub = "n/a" if b.substitution_rate is None else f"{b.substitution_rate:.2f}"
        lines.append(
            f"{r.task:<18}{r.arm:<5}"
            f"{'yes' if r.correct else 'no':>8}"
            f"{_cell(r.tests_passed, r.tests_total):>8}"
            f"{_cell(r.discriminating_passed, r.discriminating_total):>8}"
            f"{_bool(b.search_first):>10}"
            f"{b.illegitimate_doc_store_lookups:>8}"
            f"{sub:>6}"
            f"{_cell(len(r.retrieved), r.evidence_total):>7}"
            f"{r.seconds:>7.1f}"
        )
    return "\n".join(lines)


def write_report(results: list, runs: Path) -> None:
    text = table(results)
    (runs / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
