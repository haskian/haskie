"""A task: what the agent is asked, which collections answer it, and which documents say so.

`evidence` is the point of the format. The tests say whether the code came out right; `evidence`
says whether the run ever saw the passage that makes it right, which is what separates a task the
library carried from one the model already knew.
"""

from pathlib import Path

import msgspec

ROOT = Path(__file__).parent / "tasks"
PROMPT = "task.md"
TESTS = "test_task.py"
META = "task.json"


SOLUTION = "solution.py"


class Task(msgspec.Struct):
    name: str
    module: str  # what the agent is asked to write, and what the tests import
    collections: list[str]
    evidence: list[str]  # document names whose passages state the answer
    # `lookup_required` means retrieval is required to pass the source-specific discriminating
    # assertions. `lookup_expected` is the behavioural expectation used by the trigger metric.
    # They differ for source-specific facts the model may already know: lookup is expected even if
    # baseline Claude can answer without it.
    lookup_required: bool = True
    lookup_expected: bool = True
    prompt: str = ""

    @property
    def solution(self) -> Path:
        """The reference the task is checked against. Never shown to an agent: it is how we know
        a failing assertion is the run's fault and not the assertion's."""
        return ROOT / self.name / SOLUTION

    @property
    def tests(self) -> Path:
        return ROOT / self.name / TESTS


def load(name: str) -> Task:
    directory = ROOT / name
    task = msgspec.json.decode((directory / META).read_bytes(), type=Task)
    return msgspec.structs.replace(
        task, prompt=(directory / PROMPT).read_text(encoding="utf-8").strip()
    )


def every() -> list[Task]:
    return [load(d.name) for d in sorted(ROOT.iterdir()) if (d / META).is_file()]
