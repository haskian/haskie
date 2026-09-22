"""Checking a task before it is used to judge anything.

A task earns its place only if the baseline fails its discriminating assertions and a run with the
library passes them. A task both arms pass is measuring the model's memory; a task neither passes
is measuring the assertion. Either way it tells us nothing about haskie, so it is dropped rather
than averaged in.
"""

import sys
from pathlib import Path

import msgspec

from evals import metrics, tasks, trace
from evals.runner import EVENTS, Arm, execute, workspace
from evals.score import Score, score
from evals.tasks import Task


class Calibration(msgspec.Struct):
    task: str
    arm: str
    finished: bool
    returncode: int
    trace_ok: bool
    score: Score
    lookups: int  # how much the run went looking, whether or not it should have


def calibrate(
    task: Task, arm: Arm, directory: Path, *, model: str = "sonnet", max_turns: int = 40
) -> Calibration:
    run = execute(arm, task.name, task.prompt, directory, model=model, max_turns=max_turns)
    recorded = trace.read(Path(run.directory) / EVENTS)
    return Calibration(
        task=task.name,
        arm=arm.name,
        finished=recorded.ok,
        returncode=run.returncode,
        trace_ok=recorded.ok,
        score=score(task, workspace(directory), recorded),
        lookups=len(metrics.lookups(recorded)),
    )


def verdict(task: Task, baseline: Calibration, library: Calibration) -> str:
    """Whether a task is worth running, or why the question cannot be asked of it.

    Two bars, because tasks come in two kinds. One that needs the library has to be solvable with
    it; whether the baseline also manages decides if it separates the arms on knowledge or only
    on cost, and both are worth running against a public corpus. One that does not need the
    library - where the model already knows the answer and searching is a tax - has to be solvable
    by the baseline without looking anything up, or it is not really a task the model knows and it
    cannot measure a needless search.
    """
    if not (baseline.finished and library.finished):
        return "DID NOT RUN - read stderr.txt in the run directory"
    if not task.lookup_required:
        if baseline.score.total == 0 or baseline.score.passed < baseline.score.total:
            return "NOT KNOWN WITHOUT THE LIBRARY - it does need a lookup after all"
        if baseline.lookups:
            return f"BASELINE STILL LOOKED ({baseline.lookups}) - cannot show a search was needless"
        return "in-model (the baseline solves it looking nothing up)"
    total = library.score.discriminating_total
    if total == 0:
        return "NO DISCRIMINATING ASSERTION - the task cannot separate anything"
    if library.score.discriminating_passed < total:
        return f"UNSOLVED WITH THE LIBRARY ({library.score.discriminating_passed}/{total})"
    if baseline.score.discriminating_passed >= total:
        # Kept rather than dropped. It does need a lookup, so it belongs in the denominator of
        # the trigger metric, and both arms solving it is what a public corpus looks like - the
        # comparison that remains is what each route cost.
        return "cost-only (both arms solve it; compared on lookups, turns and cost)"
    without = baseline.score.discriminating_passed
    return f"separates ({without}/{total} without the library, {total}/{total} with)"


def admits(task: Task, baseline: Calibration, library: Calibration) -> bool:
    return verdict(task, baseline, library).startswith(("separates", "in-model", "cost-only"))


def main() -> int:
    """Calibrate every task on the two arms that decide whether it is worth running: no library,
    and the library with nothing but the tools. Exits non-zero if any task fails to separate
    them, because that task cannot tell a run that read the library from one that remembered."""
    from evals.arms import arms  # local: it reaches the server, and only `main` needs to

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evals/calibration")
    baseline, library = arms()[:2]
    verdicts = {}
    for task in tasks.every():
        results = {}
        for arm in (baseline, library):
            done = calibrate(task, arm, root / task.name / arm.name)
            results[arm.name] = done
            print(msgspec.json.format(msgspec.json.encode(done).decode(), indent=1), flush=True)
        verdicts[task.name] = verdict(task, results[baseline.name], results[library.name])
    for name, said in verdicts.items():
        print(f"{name}: {said}")
    kept = [v.startswith(("separates", "in-model", "cost-only")) for v in verdicts.values()]
    return 0 if kept and all(kept) else 1


if __name__ == "__main__":
    raise SystemExit(main())
