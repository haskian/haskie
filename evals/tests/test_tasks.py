"""Every task is checked against its own reference before any agent is asked to do it.

A failing assertion has to mean the run got it wrong. If the reference cannot pass, the assertion
is wrong, and a grid run against it would measure the eval rather than the agent.
"""

import shutil
from pathlib import Path

import pytest

from evals import score, tasks


@pytest.mark.parametrize("task", tasks.every(), ids=lambda t: t.name)
def test_the_reference_solution_scores_full_marks(task: tasks.Task, tmp_path: Path) -> None:
    shutil.copy(task.solution, tmp_path / f"{task.module}.py")

    outcomes = score.run_tests(task, tmp_path)

    assert outcomes, "the task's tests did not even collect"
    failed = {node: v["outcome"] for node, v in outcomes.items() if v["outcome"] != "passed"}
    assert not failed


def test_a_task_is_scored_from_a_relative_workspace(tmp_path, monkeypatch) -> None:
    """Scoring runs pytest inside the workspace, so a relative path given here resolves against
    the wrong directory and the report lands where nothing reads it. That scored every run zero
    and read as three tasks failing to separate."""
    task = tasks.every()[0]
    monkeypatch.chdir(tmp_path)
    work = Path("runs/t/arm/work")
    work.mkdir(parents=True)
    shutil.copy(task.solution, work / f"{task.module}.py")

    outcomes = score.run_tests(task, work)

    assert outcomes


@pytest.mark.parametrize("task", tasks.every(), ids=lambda t: t.name)
def test_a_task_that_needs_the_library_can_separate_the_arms(task: tasks.Task) -> None:
    """Without a discriminating assertion a lookup task cannot tell a run that read the library
    from one that remembered well, whatever its pass rate says.

    Nothing is asserted about an in-model task here. It may well have evidence - `mlfq_priority`
    is in the corpus and still does not need it, which is what makes it worth running - and
    whether the model really knows it is a fact about the model, so calibration decides it by
    seeing the baseline solve it with no lookups at all.
    """
    if not task.lookup_required:
        return
    marked = [
        line
        for line in task.tests.read_text(encoding="utf-8").splitlines()
        if "discriminating" in line
    ]
    assert marked, f"{task.name} has no discriminating assertion"
    assert task.evidence
    assert task.collections


@pytest.mark.parametrize("task", tasks.every(), ids=lambda t: t.name)
def test_every_task_asks_for_something(task: tasks.Task) -> None:
    assert task.prompt
    assert task.module
