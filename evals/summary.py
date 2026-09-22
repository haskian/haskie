"""Reading a finished grid: one row per cell, then the means per task and arm.

The two halves are kept side by side on purpose. `disc` is whether the run got the detail right,
`look` and `cost` are what it spent finding out. Against a public corpus the first tends to tie
and the second is where the library shows up, so a table that reported only pass rates would say
haskie changed nothing.
"""

import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import msgspec

from evals import metrics, tasks, trace
from evals.runner import EVENTS, workspace
from evals.score import score


class Row(msgspec.Struct):
    task: str
    arm: str
    lookup_required: bool
    lookup_expected: bool
    repeat: int
    finished: bool
    passed: int
    total: int
    discriminating_passed: int
    discriminating_total: int
    lookups: int
    haskie: int
    substitution: float | None
    leaks: int
    spills: int
    filesystem_first: bool
    haskie_then_filesystem: int
    evidence_seen: int
    evidence_total: int
    turns: int
    cost_usd: float
    seconds: float


def read_cell(directory: Path, task: tasks.Task) -> Row | None:
    events = directory / EVENTS
    if not events.is_file():
        return None
    recorded = trace.read(events)
    summary = metrics.summarize(recorded)
    scored = score(task, workspace(directory), recorded)
    # A grid cell is `task/arm/repeat`, a calibration run is `task/arm`: the last segment says
    # which, and reporting on a calibration is worth more than insisting on the grid's shape.
    numbered = directory.name.isdigit()
    return Row(
        task=task.name,
        arm=directory.parent.name if numbered else directory.name,
        lookup_required=task.lookup_required,
        lookup_expected=task.lookup_expected,
        repeat=int(directory.name) if numbered else 0,
        finished=recorded.ok,
        passed=scored.passed,
        total=scored.total,
        discriminating_passed=scored.discriminating_passed,
        discriminating_total=scored.discriminating_total,
        lookups=summary.lookups,
        haskie=summary.haskie,
        substitution=summary.substitution_rate,
        leaks=summary.leaks,
        spills=summary.spills,
        filesystem_first=summary.filesystem_first,
        haskie_then_filesystem=summary.haskie_then_filesystem,
        evidence_seen=len(scored.evidence_seen),
        evidence_total=scored.evidence_total,
        turns=summary.num_turns,
        cost_usd=summary.cost_usd,
        seconds=summary.duration_ms / 1000,
    )


def collect(root: Path) -> list[Row]:
    rows = []
    for task in tasks.every():
        here = root / task.name
        cells = sorted(p.parent for p in here.glob("**/events.jsonl")) if here.is_dir() else []
        for directory in cells:
            row = read_cell(directory, task)
            if row is not None and row.finished:
                rows.append(row)
    return rows


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _share(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def _spread(values: list[float]) -> float:
    """Population standard deviation, 0 for a single run. Agent runs vary enough that a mean over
    repeats means little without it, and a table of means alone reads as far more settled than the
    numbers are."""
    return statistics.pstdev(values) if len(values) > 1 else 0.0


# Each column is its heading, its width, and what it reads off one run. `disc` and `look`/`cost`
# sit side by side because against a public corpus the first tends to tie between arms and the
# second is where a library shows up at all.
COLUMNS: tuple[tuple[str, int, Callable[[Row], float]], ...] = (
    ("pass", 7, lambda r: _share(r.passed, r.total)),
    ("disc", 7, lambda r: _share(r.discriminating_passed, r.discriminating_total)),
    ("evid", 7, lambda r: _share(r.evidence_seen, r.evidence_total)),
    ("fs1st", 7, lambda r: float(r.filesystem_first)),
    ("h2fs", 6, lambda r: r.haskie_then_filesystem),
    ("look", 6, lambda r: r.lookups),
    ("hask", 6, lambda r: r.haskie),
    ("leak", 6, lambda r: r.leaks),
    ("spill", 6, lambda r: r.spills),
    ("turns", 7, lambda r: r.turns),
    ("cost", 8, lambda r: r.cost_usd),
    ("secs", 6, lambda r: r.seconds),
)


def table(rows: list[Row]) -> str:
    cells: dict[tuple[str, str], list[Row]] = {}
    for row in rows:
        cells.setdefault((row.task, row.arm), []).append(row)
    head = f"{'task':<16}{'arm':<20}{'n':>3}{'sub':>6}"
    head += "".join(f"{name:>{width}}" for name, width, _ in COLUMNS)
    head += f"{'turns~':>8}{'cost~':>8}"
    lines = [head, "-" * len(head)]
    for (task, arm), group in sorted(cells.items()):
        rates = [r.substitution for r in group if r.substitution is not None]
        line = f"{task:<16}{arm:<20}{len(group):>3}{_mean(rates):>6.2f}"
        for _, width, read in COLUMNS:
            line += f"{_mean([read(r) for r in group]):>{width}.2f}"
        line += f"{_spread([r.turns for r in group]):>8.2f}"
        line += f"{_spread([r.cost_usd for r in group]):>8.3f}"
        lines.append(line)
    return "\n".join([*lines, "", *_trigger_lines(rows), "", *_first_lookup_lines(rows)])


def _trigger_lines(rows: list[Row]) -> list[str]:
    """Did the arm reach for the library when it should have, and leave it alone when not.

    Only for the arms that have it: an arm without the library cannot search, so every task would
    count as a miss and the number would describe the arm rather than the behaviour.
    """
    by_arm: dict[str, list[tuple[bool, bool]]] = {}
    for row in rows:
        if row.arm != "a-no-library":
            by_arm.setdefault(row.arm, []).append((row.lookup_expected, bool(row.haskie)))
    lines = ["searched when it should have:"]
    for arm, runs in sorted(by_arm.items()):
        scored = metrics.trigger(runs)
        precision = "n/a" if scored.precision is None else f"{scored.precision:.2f}"
        recall = "n/a" if scored.recall is None else f"{scored.recall:.2f}"
        lines.append(
            f"  {arm:<20}precision {precision}  recall {recall}"
            f"  ({scored.searched_when_not}/{scored.not_needed} needless)"
        )
    return lines



def _first_lookup_lines(rows: list[Row]) -> list[str]:
    by_arm: dict[str, dict[str, int]] = {}
    for row in rows:
        # This is a descriptive trajectory count; unlike trigger precision/recall it is not an
        # assertion that any particular first tool was universally correct.
        arm = by_arm.setdefault(row.arm, {})
        # `first_lookup` is not stored in Row, so derive the coarse filesystem-first signal here.
        key = "filesystem-first" if row.filesystem_first else ("haskie-first" if row.haskie else "other/no-lookup")
        arm[key] = arm.get(key, 0) + 1
    lines = ["first knowledge lookup:"]
    for arm, counts in sorted(by_arm.items()):
        parts = ", ".join(f"{kind} {count}" for kind, count in sorted(counts.items()))
        lines.append(f"  {arm:<20}{parts}")
    return lines

def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evals/runs")
    all_cells = sum(1 for task in tasks.every() for _ in ((root / task.name).glob("**/events.jsonl") if (root / task.name).is_dir() else []))
    rows = collect(root)
    if not rows:
        print(f"no finished runs under {root}")
        return 1
    print(table(rows))
    excluded = all_cells - len(rows)
    if excluded:
        print(f"\nexcluded unfinished runs: {excluded}")
    (root / "rows.json").write_bytes(msgspec.json.encode(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
