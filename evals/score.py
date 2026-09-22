"""Scoring one run: what the tests say, and whether the run ever saw the evidence.

The two are kept apart on purpose. Code that passes without the evidence having been retrieved is
the model answering from memory, and counting it as a win for the library would be the whole
eval's mistake.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import msgspec

from evals.report import DISCRIMINATING, KEY
from evals.tasks import Task
from evals.trace import Trace

REPO = Path(__file__).resolve().parent.parent


class Score(msgspec.Struct):
    task: str
    passed: int
    total: int
    discriminating_passed: int
    discriminating_total: int
    evidence_seen: list[str]
    evidence_total: int
    outcomes: dict[str, str]


def run_tests(task: Task, work: Path) -> dict[str, dict]:
    """Run the task's tests against whatever the agent left in `work`.

    From `work` rather than the repo, so the agent's module is the one imported and the repo's own
    pytest options do not apply to a task's tests. Every path handed over is absolute for the same
    reason: pytest runs in `work`, so a relative one resolves against the wrong place and the
    report lands where nothing reads it.
    """
    work = work.resolve()
    report = work / "report.json"
    environment = {
        **os.environ,
        KEY: str(report),
        "PYTHONPATH": os.pathsep.join([str(work), str(REPO)]),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(task.tests.resolve()),
            "-q",
            "--import-mode=importlib",
            "-p",
            "evals.report",
            "-p",
            "no:cacheprovider",
        ],
        cwd=work,
        env=environment,
        capture_output=True,
        check=False,
    )
    if not report.is_file():
        # Nothing ran. Scoring that as zero in silence reads as a task the agent failed, so say
        # what pytest said instead.
        print(completed.stdout.decode(errors="replace")[-2000:], file=sys.stderr)
        return {}
    return json.loads(report.read_text(encoding="utf-8"))


def evidence_seen(trace: Trace, task: Task) -> list[str]:
    """Which of the task's documents a search actually handed back. The document name is in every
    path a hit carries, so `returned` is enough without holding the payloads."""
    paths = " ".join(path for call in trace.calls for path in call.returned)
    return [name for name in task.evidence if name in paths]


def score(task: Task, work: Path, trace: Trace) -> Score:
    outcomes = run_tests(task, work)
    seen = evidence_seen(trace, task)
    discriminating = {k: v for k, v in outcomes.items() if v[DISCRIMINATING]}
    return Score(
        task=task.name,
        passed=sum(1 for v in outcomes.values() if v["outcome"] == "passed"),
        total=len(outcomes),
        discriminating_passed=sum(1 for v in discriminating.values() if v["outcome"] == "passed"),
        discriminating_total=len(discriminating),
        evidence_seen=seen,
        evidence_total=len(task.evidence),
        outcomes={k: v["outcome"] for k, v in outcomes.items()},
    )
