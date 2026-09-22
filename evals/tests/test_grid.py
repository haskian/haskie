"""The parts of the grid that hold without a server: the corpus manifest, the run layout, and
the table a finished grid prints."""

from pathlib import Path

import msgspec

from evals import corpus, grid, summary
from evals.runner import Arm
from evals.tasks import Task

BASE = summary.Row(
    task="t",
    arm="a",
    lookup_required=True,
    lookup_expected=True,
    repeat=0,
    finished=True,
    passed=6,
    total=7,
    discriminating_passed=1,
    discriminating_total=4,
    lookups=5,
    haskie=4,
    substitution=0.2,
    leaks=0,
    spills=0,
    filesystem_first=False,
    haskie_then_filesystem=0,
    evidence_seen=1,
    evidence_total=2,
    turns=9,
    cost_usd=0.3,
    seconds=60.0,
)


def _row(task: str, arm: str, repeat: int, **over: object) -> summary.Row:
    return msgspec.structs.replace(BASE, task=task, arm=arm, repeat=repeat, **over)


def test_every_source_is_named_once_and_fetched_over_https() -> None:
    names = [source.name for source in corpus.SOURCES]

    assert len(set(names)) == len(names)
    assert all(source.url.startswith("https://") for source in corpus.SOURCES)
    assert all(name.endswith(".pdf") for name in names)


def test_a_cell_is_one_directory_per_task_arm_and_repeat() -> None:
    task = Task(name="t", module="m", collections=["c"], evidence=["d"])

    first = grid.cell(Path("runs"), task, Arm(name="b-mcp-only"), 0)
    second = grid.cell(Path("runs"), task, Arm(name="b-mcp-only"), 1)

    assert first == Path("runs/t/b-mcp-only/0")
    assert first != second, "repeats never share a directory, or they share an auto-memory"


def test_the_table_folds_repeats_into_one_row_per_arm() -> None:
    rows = [_row("t", "a-no-library", i) for i in range(3)] + [_row("t", "b-mcp-only", 0)]

    printed = summary.table(rows).split("\n\n")[0].splitlines()

    assert len(printed) == 4, "a heading, a rule, and one row per arm"
    assert printed[2].split()[:3] == ["t", "a-no-library", "3"]
    assert printed[3].split()[:3] == ["t", "b-mcp-only", "1"]


def test_a_run_that_gathered_nothing_does_not_drag_the_substitution_rate_down() -> None:
    """`None` means the run never looked anything up, which is not the same as looking in the
    right place, and averaging it in as zero would read as the good outcome."""
    rows = [_row("t", "a", 0, substitution=None), _row("t", "a", 1, substitution=1.0)]

    printed = summary.table(rows).splitlines()[2]

    assert " 1.00" in printed


def test_a_needless_search_counts_against_the_arm() -> None:
    """A task the model already knows is not a failure of the corpus, and a run that searches it
    anyway paid for nothing. Without this the two look the same in the pass rate."""
    warranted = _row("needs", "b-mcp-only", 0, lookup_required=True, lookup_expected=True, haskie=3)
    needless = _row("knows", "b-mcp-only", 0, lookup_required=False, lookup_expected=False, haskie=2)

    printed = summary.table([warranted, needless])

    assert "precision 0.50" in printed
    assert "recall 1.00" in printed
    assert "(1/1 needless)" in printed


def test_an_arm_without_the_library_is_left_out_of_the_trigger_count() -> None:
    """It cannot search, so counting its every task as a miss would describe the arm."""
    printed = summary.table([_row("t", "a-no-library", 0, haskie=0)])

    assert "a-no-library" not in printed.split("searched when it should have:")[1]
