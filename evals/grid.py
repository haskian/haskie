"""Every arm on every task, repeated.

Resumable: a cell that already has a transcript is left alone, so a grid interrupted halfway
carries on rather than paying for its first half twice.
"""

import sys
from pathlib import Path

from evals import tasks
from evals.arms import arms
from evals.runner import EVENTS, Arm, execute
from evals.trace import read
from evals.tasks import Task

REPEATS = 5  # agent trajectories vary enough that three will not separate anything


def cell(root: Path, task: Task, arm: Arm, repeat: int) -> Path:
    return root / task.name / arm.name / str(repeat)


def run(
    root: Path,
    *,
    repeats: int = REPEATS,
    model: str = "sonnet",
    max_turns: int = 40,
) -> list[Path]:
    done = []
    for task in tasks.every():
        for arm in arms():
            for repeat in range(repeats):
                directory = cell(root, task, arm, repeat)
                done.append(directory)
                if (directory / EVENTS).exists():
                    try:
                        if read(directory / EVENTS).ok:
                            print(f"have {directory}", flush=True)
                            continue
                    except (OSError, ValueError):
                        pass
                    print(f"rerunning incomplete {directory}", flush=True)
                print(f"running {task.name} {arm.name} {repeat}", flush=True)
                execute(
                    arm,
                    task.name,
                    task.prompt,
                    directory,
                    model=model,
                    max_turns=max_turns,
                )
    return done


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evals/runs")
    given = sys.argv[2] if len(sys.argv) > 2 else str(REPEATS)
    if not given.isdigit():
        print(f"usage: grid [runs-dir] [repeats] [model]; repeats was {given!r}")
        return 2
    repeats = int(given)
    model = sys.argv[3] if len(sys.argv) > 3 else "sonnet"
    run(root, repeats=repeats, model=model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
