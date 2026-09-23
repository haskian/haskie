"""Run one task under one arm against the isolated eval instance, and grade the result.

Arm A has no haskie access at all. Arm B has haskie's MCP tools and nothing else - no skill, no
project instructions. The two exist to answer one question before any other: with nothing but
the raw tools, does the agent reach for search before it reaches for a file. Arm C has the same
tools as B plus one written instruction telling it to search before writing code - the "skill or
CLAUDE.md instruction" step, now that A/B is trusted. It isolates one variable at a time: B vs A
shows what the bare tool does on its own; C vs B shows what explicit coaching adds on top of that,
each holding everything else about the task fixed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from evals import metrics
from evals.report import write_report

ROOT = Path(__file__).resolve().parent
TASK_ROOT = ROOT / "tasks"
RUNS = ROOT / "runs"

CODING_TOOLS = ("Read", "Write", "Edit", "Grep", "Glob", "Bash", "TodoWrite")
HASKIE_TOOLS = (
    "mcp__haskie__list_collections",
    "mcp__haskie__get_collection",
    "mcp__haskie__list_collection_documents",
    "mcp__haskie__set_session_collections",
    "mcp__haskie__search",
    "mcp__haskie__search_collection",
    "mcp__haskie__search_text",
    "mcp__haskie__document_passages",
)
ARMS = ("a", "b", "c")
TASKS = (
    "mlfq_priority",
    "reusable_barrier",
    "revision_ranges",
    "bounded_buffer",
    "git_objects",
    "h2o",
    "git_history_split",
    "github_api_breaking_changes",
    "raft_election",
    "collapsed_forwarding",
)

# pytest's summary line lists whichever outcomes occurred, in its own fixed order - "failed"
# before "passed" when both are present - not always "passed" first. Matching each `N <word>`
# independently, rather than assuming an order, is what the first version of this got wrong: a
# run with failures reported "1 passed" and silently dropped the failures it never looked for.
OUTCOME = re.compile(r"(\d+) (passed|failed|errors?)\b")


@dataclass(frozen=True)
class Task:
    name: str
    module: str
    evidence: list[str]

    @property
    def prompt(self) -> str:
        return (TASK_ROOT / self.name / "task.md").read_text(encoding="utf-8")

    @property
    def tests(self) -> Path:
        return TASK_ROOT / self.name / "test_task.py"


def load_task(name: str) -> Task:
    meta = json.loads((TASK_ROOT / name / "meta.json").read_text())
    return Task(name, meta["module"], meta["evidence"])


def claude_binary() -> str:
    return (
        os.environ.get("EVAL_CLAUDE_BIN")
        or shutil.which("claude")
        or str(Path.home() / ".local/bin/claude")
    )


# Spawned from inside a live Claude Code session, the child otherwise inherits this session's own
# CLAUDE_CODE_* variables (its socket, its session id) and tries to join that session instead of
# starting its own, which surfaces as an opaque authentication failure.
NESTED_SESSION_VARS = (
    "CLAUDECODE",
    "CLAUDE_CODE_SSE_PORT",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_EXECPATH",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
)


def subprocess_environment() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in NESTED_SESSION_VARS}


def prompt_for(task: Task, arm: str, collection: str) -> str:
    header = (
        f"You are completing the {task.name} evaluation task. Work only in the current "
        "directory; do not read anything outside it.\n\n"
    )
    if arm == "a":
        return f"{header}{task.prompt}"
    if arm == "c":
        coaching = (
            f"\n\nYou have a Haskie MCP collection named {collection} available, containing the "
            "source this task is grounded in. Search it before writing any code - confirm the "
            "source's exact wording and design first, rather than relying on general "
            "familiarity with the topic."
        )
        return f"{header}{task.prompt}{coaching}"
    haskie_note = (
        f"\n\nYou have a Haskie MCP collection named {collection} available. It may or may not "
        "have material relevant to this task."
    )
    return f"{header}{task.prompt}{haskie_note}"


def allowed_tools(arm: str) -> list[str]:
    return [*CODING_TOOLS, *(HASKIE_TOOLS if arm in ("b", "c") else ())]


# A parent Claude Code session and its spawned `claude -p` child share the same on-disk OAuth
# credentials; if the child's launch lands mid-refresh, it fails its very first turn with this
# exact message and no tool calls, indistinguishable from a real task failure except for the
# text - and a retry a moment later succeeds cleanly.
AUTH_RETRY_MARKER = "OAuth session expired and could not be refreshed"
AUTH_RETRY_ATTEMPTS = 3


def run_agent(task: Task, arm: str, directory: Path, model: str, api: str, collection: str) -> int:
    work = directory / "work"
    work.mkdir(parents=True, exist_ok=True)
    mcp = directory / "mcp.json"
    servers = (
        {"haskie": {"type": "http", "url": f"{api.rstrip('/')}/mcp"}} if arm in ("b", "c") else {}
    )
    mcp.write_text(json.dumps({"mcpServers": servers}, indent=2) + "\n")

    for attempt in range(1, AUTH_RETRY_ATTEMPTS + 1):
        argv = [
            claude_binary(),
            "-p",
            prompt_for(task, arm, collection),
            "--output-format",
            "stream-json",
            "--verbose",
            "--setting-sources",
            "project",
            "--strict-mcp-config",
            "--mcp-config",
            str(mcp.resolve()),
            "--allowedTools",
            *allowed_tools(arm),
            "--model",
            model,
            "--max-turns",
            os.environ.get("EVAL_MAX_TURNS", "40"),
            "--session-id",
            str(uuid.uuid4()),
        ]
        (directory / "argv.json").write_text(json.dumps(argv, indent=2) + "\n")
        with (
            (directory / "transcript.jsonl").open("wb") as transcript,
            (directory / "stderr.txt").open("wb") as stderr,
        ):
            completed = subprocess.run(
                argv,
                cwd=work,
                stdout=transcript,
                stderr=stderr,
                stdin=subprocess.DEVNULL,
                env=subprocess_environment(),
                check=False,
            )
        if completed.returncode == 0:
            return completed.returncode
        transcript_text = (directory / "transcript.jsonl").read_text(
            encoding="utf-8", errors="replace"
        )
        if AUTH_RETRY_MARKER not in transcript_text:
            return completed.returncode
        if attempt < AUTH_RETRY_ATTEMPTS:
            backoff = 3 * attempt
            print(
                f"  {task.name}/{arm}: transient auth failure, retrying "
                f"({attempt}/{AUTH_RETRY_ATTEMPTS}) in {backoff}s...",
                file=sys.stderr,
            )
            time.sleep(backoff)
    return completed.returncode


def run_pytest(test_file: Path, work: Path, marker: str | None = None) -> tuple[int, int]:
    """`(passed, total)` from pytest's own summary line, for the whole file or one `-m` subset."""
    argv = [sys.executable, "-m", "pytest", "-q", str(test_file.resolve())]
    if marker:
        argv += ["-m", marker]
    completed = subprocess.run(
        argv,
        cwd=work,
        env={**os.environ, "PYTHONPATH": str(work)},
        capture_output=True,
        text=True,
        check=False,
    )
    if "no tests ran" in completed.stdout:
        return 0, 0
    # Only the true summary line, the last non-empty line pytest prints - never the whole output.
    # A collection error also prints "Interrupted: N error during collection" earlier, which
    # matches the same shape and would otherwise be double-counted alongside the real total.
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    summary = lines[-1] if lines else ""
    counts: dict[str, int] = {}
    for number, word in OUTCOME.findall(summary):
        key = "error" if word.startswith("error") else word
        counts[key] = counts.get(key, 0) + int(number)
    if not counts:
        return 0, 0  # collection failed outright: the agent's file didn't even import
    passed = counts.get("passed", 0)
    total = passed + counts.get("failed", 0) + counts.get("error", 0)
    return passed, total


@dataclass(frozen=True)
class Result:
    task: str
    arm: str
    model: str
    directory: str
    agent_returncode: int
    tests_passed: int
    tests_total: int
    discriminating_passed: int
    discriminating_total: int
    retrieved: list[str]
    evidence_total: int
    behaviour: metrics.Behaviour
    seconds: float

    @property
    def correct(self) -> bool:
        return self.tests_total > 0 and self.tests_passed == self.tests_total


def grade(
    task: Task,
    arm: str,
    directory: Path,
    returncode: int,
    doc_root: Path,
    seconds: float,
    model: str,
) -> Result:
    work = directory / "work"
    passed, total = run_pytest(task.tests, work)
    disc_passed, disc_total = run_pytest(task.tests, work, marker="discriminating")
    transcript = (directory / "transcript.jsonl").read_text(encoding="utf-8", errors="replace")
    found = metrics.retrieved(transcript, doc_root, task.evidence) if arm in ("b", "c") else []
    behaviour = metrics.behaviour(transcript, doc_root)
    result = Result(
        task.name,
        arm,
        model,
        str(directory),
        returncode,
        passed,
        total,
        disc_passed,
        disc_total,
        found,
        len(task.evidence),
        behaviour,
        seconds,
    )
    (directory / "result.json").write_text(
        json.dumps(asdict(result) | {"correct": result.correct}, indent=2, default=str) + "\n"
    )
    return result


def warm_auth() -> None:
    """`claude auth status` is a cheap, no-turn call that still forces a pending OAuth refresh to
    happen. Doing that once here, before any eval subprocess starts, means the run's later
    subprocesses see an already-fresh token instead of each independently racing to refresh the
    same on-disk credentials the moment it goes stale - the actual cause behind AUTH_RETRY_MARKER,
    which retrying alone doesn't fully cover when the contention window outlasts the retries."""
    subprocess.run(
        [claude_binary(), "auth", "status"],
        env=subprocess_environment(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def run_one(
    task: Task, arm: str, sample: int, model: str, api: str, home: Path, collection: str
) -> Result:
    directory = RUNS / arm / task.name / str(sample)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    started = time.monotonic()
    returncode = run_agent(task, arm, directory, model, api, collection)
    seconds = time.monotonic() - started
    doc_root = home / "documents"
    return grade(task, arm, directory, returncode, doc_root, seconds, model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", choices=[*TASKS, "all"], default="all")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--model", default=os.environ.get("EVAL_MODEL", "sonnet"))
    parser.add_argument("--api", default=os.environ.get("HASKIE_EVAL_URL", "http://127.0.0.1:8123"))
    parser.add_argument(
        "--home", type=Path, default=Path(os.environ.get("HASKIE_EVAL_HOME", ROOT / ".haskie-eval"))
    )
    parser.add_argument(
        "--collection", default=os.environ.get("HASKIE_EVAL_COLLECTION", "eval-programming-books")
    )
    parser.add_argument(
        "--samples", type=int, default=int(os.environ.get("EVAL_SAMPLES", "3"))
    )
    args = parser.parse_args()

    tasks = [load_task(t) for t in (TASKS if args.task == "all" else [args.task])]
    RUNS.mkdir(parents=True, exist_ok=True)
    warm_auth()
    results = [
        run_one(task, arm, sample, args.model, args.api, args.home, args.collection)
        for arm in args.arms
        for task in tasks
        for sample in range(args.samples)
    ]
    write_report(results, RUNS)
    return 0 if all(r.correct for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
