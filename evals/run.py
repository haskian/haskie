"""Run one task under one arm against the isolated eval instance, and grade the result.

Arm A has no haskie access at all. Arm B has haskie's MCP tools and nothing else - no skill, no
project instructions. The two exist to answer one question before any other: with nothing but
the raw tools, does the agent reach for search before it reaches for a file. Arm C has the same
tools as B plus one written instruction telling it to search before writing code - the "skill or
CLAUDE.md instruction" step, now that A/B is trusted. It isolates one variable at a time: B vs A
shows what the bare tool does on its own; C vs B shows what explicit coaching adds on top of that,
each holding everything else about the task fixed.

Arm D is the baseline haskie actually competes with: the same documents as plain files in the
working directory, no haskie. A vs B only shows that a corpus is needed; D vs B shows whether
haskie is the right way to reach one.

Arm E is arm B against a second haskie instance that has an embedding profile, so its searches
are hybrid (vectors + BM25) where B's are full-text only. B vs E isolates what semantic search
adds - the variable the synthetic tasks' paraphrase level is built to exercise.

Arm F is arm B searching sibling collections chunked at 300 characters instead of the default
1200 (`setup.SMALL_CHUNKS`). An excerpt is a chunk widened to sentence boundaries, so chunk size
sets how much every search result puts into the agent's context - B vs F isolates that.

Arms G and H drop the one unrealistic thing every haskie arm above does: tell the agent a
collection exists. G has B's tools and a prompt with no mention of haskie, as a user's own request
would be - it measures whether the agent reaches for the library unprompted. H is G plus a
`UserPromptSubmit` hook (`steer.py`) that searches haskie with the prompt and puts the best few
passages in front of the agent: steering on every prompt, with no tool call to think of making.
G vs B is what the hint in B's prompt was worth; H vs G is what steering is.

Arm I is G as a user who ran `haskie install claude` has it: the rule and the skill haskie writes
into Claude Code (`haskie.claude.render_rule`, `render_skill`), rendered for the collection the task
searches and put in the agent's project `.claude/`. The MCP tool descriptions say what each tool
does; the always-loaded rule says when to reach for haskie at all. I vs G is what that is worth.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from evals import metrics, scope, setup, synth
from evals.report import write_report
from evals.setup import CORPUS_DIR

ROOT = Path(__file__).resolve().parent
TASK_ROOT = ROOT / "tasks"
RUNS = ROOT / "runs"

CODING_TOOLS = ("Read", "Write", "Edit", "Grep", "Glob", "Bash", "TodoWrite")
# Every read-only tool the server exposes (`grep -rn mcp_tool= src/haskie`); the write tools
# (add/remove document) stay out so an arm can't change the corpus it's being measured against.
# These must track the server: a tool missing here is silently unavailable to the agent.
HASKIE_TOOLS = (
    "mcp__haskie__list_collections",
    "mcp__haskie__get_collection",
    "mcp__haskie__list_collection_documents",
    "mcp__haskie__list_documents",
    "mcp__haskie__describe_document",
    "mcp__haskie__get_document",
    "mcp__haskie__set_session_collections",
    "mcp__haskie__search_sections",
    "mcp__haskie__search_excerpts",
    "mcp__haskie__report_gap",
)
# Deliberately left out: `list_searches` and `list_gaps` show other sessions' searches and near
# misses - in an eval instance, earlier runs of the same task, citing its answer. `review_gaps` and
# `replay_gaps` curate collections; an agent answering a task has no use for them.
ARMS = ("a", "b", "c", "d", "e", "f", "g", "h", "i")
HASKIE_ARMS = ("b", "c", "e", "f", "g", "h", "i")
DEFAULT_ARMS = ("a", "b", "c", "d", "e", "f")  # G, H and I run when named: `--arms g h i`
UNMENTIONED_ARMS = ("g", "h", "i")  # haskie connected, but the prompt says nothing of it
BOOK_TASKS = (
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
SYNTH_TASKS = tuple(task for seed in synth.SEEDS for task in synth.task_names(seed))
TASKS = (*BOOK_TASKS, *SYNTH_TASKS)
GROUPS = {"all": TASKS, "books": BOOK_TASKS, "synth": SYNTH_TASKS}

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
    corpus: Path  # what arm D gets as files
    collection: str | None  # what the haskie arms search; None means the run's --collection

    @property
    def prompt(self) -> str:
        return (TASK_ROOT / self.name / "task.md").read_text(encoding="utf-8")

    @property
    def tests(self) -> Path:
        return TASK_ROOT / self.name / "test_task.py"


def load_task(name: str) -> Task:
    meta = json.loads((TASK_ROOT / name / "meta.json").read_text())
    corpus = ROOT / meta["corpus"] if "corpus" in meta else CORPUS_DIR
    return Task(name, meta["module"], meta["evidence"], corpus, meta.get("collection"))


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
    """This environment without this session's own variables, and with Claude Code's auto memory
    off: run from inside this repository, an agent otherwise loads the memory of whoever works
    on it - notes about them, and how they like haskie used - into every arm alike."""
    env = {k: v for k, v in os.environ.items() if k not in NESTED_SESSION_VARS}
    return env | {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}


# Where an agent actually works: outside the home directory, so no `CLAUDE.md` of a directory above
# it (the user's own `~/.claude/CLAUDE.md` among them) reaches the agent. The run's own `work/`
# links here while the agent runs, and becomes the directory itself afterwards (`keep_work`).
WORK_ROOT = Path(
    os.environ.get("EVAL_WORK_ROOT") or Path(tempfile.gettempdir()) / "haskie-eval-work"
)


def scratch_work(directory: Path) -> Path:
    """A fresh working directory for the run at `directory`, under `WORK_ROOT`, linked from
    `directory / "work"`. The agent's cwd resolves to it, so Claude Code reads its instructions
    from there up, not from this repository up."""
    real = WORK_ROOT / directory.resolve().relative_to(RUNS.resolve())
    if real.exists():
        shutil.rmtree(real)
    real.mkdir(parents=True)
    (directory / "work").symlink_to(real, target_is_directory=True)
    return real


def keep_work(directory: Path) -> None:
    """Move the agent's work back under the run, in place of the link: a temp directory may be
    gone by the time someone reads the run again."""
    link = directory / "work"
    if link.is_symlink():
        real = link.resolve()
        link.unlink()
        shutil.move(real, link)


def prompt_for(task: Task, arm: str, collection: str) -> str:
    header = (
        f"You are completing the {task.name} evaluation task. Work only in the current "
        "directory; do not read anything outside it.\n\n"
    )
    if arm == "a" or arm in UNMENTIONED_ARMS:
        return f"{header}{task.prompt}"
    if arm == "d":
        files_note = (
            "\n\nThe `corpus/` directory here holds a collection of documents. It may or may not "
            "have material relevant to this task."
        )
        return f"{header}{task.prompt}{files_note}"
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


def denied_reads() -> list[str]:
    """Permission rules for every place an agent could read an answer from rather than find it:
    the graders and task metadata, the generated corpora, other runs, the eval's own notes and
    results, and the Claude Code transcripts of whoever works on this repository. "Work only in
    the current directory" is an instruction; a stuck agent greps the disk (one read a task's
    `meta.json` and passed). haskie's own document stores stay readable: search results disclose
    those paths, and opening one is haskie working as designed, scored by `metrics.behaviour`.
    """
    repo = ROOT.parent
    config = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    project = "-" + str(repo).strip("/").replace("/", "-")  # how Claude Code names its folder
    folders = [ROOT / "tasks", ROOT / "synth", ROOT / "corpus", ROOT / "bookqa", ROOT / "runs"]
    folders += sorted(ROOT.glob("runs.*"))
    files = [ROOT / "synth.py", ROOT / "results.md", repo / "NOTES.md"]
    rules = [f"Read(/{folder}/**)" for folder in folders]
    rules += [f"Read(/{file})" for file in files]
    rules.append(f"Read(/{config}/projects/{project}*/**)")
    scratch = Path("/private/tmp") / f"claude-{os.getuid()}"  # Claude Code's session scratchpads
    rules.append(f"Read(/{scratch}/**)")
    return rules


def agent_settings(arm: str, api: str, collection: str) -> dict:
    """The settings `--settings` hands every arm: the reads it is denied, with Bash sandboxed so
    the denials hold for a shell too (`cat`, `grep -r ~`), and for arm H its steering hook. A hook
    loaded this way fires in print mode, beside `--setting-sources project`."""
    settings: dict = {"permissions": {"deny": denied_reads()}, "sandbox": {"enabled": True}}
    if arm == "h":
        command = (
            f"PYTHONPATH={shlex.quote(str(ROOT.parent))} {shlex.quote(sys.executable)} "
            f"-m evals.steer --api {shlex.quote(api)} --collection {shlex.quote(collection)}"
        )
        hook = {"type": "command", "command": command}
        settings["hooks"] = {"UserPromptSubmit": [{"hooks": [hook]}]}
    return settings


def install_haskie(work: Path, api: str, collection: str) -> list[Path]:
    """Arm I's `.claude/`: haskie's rule and skill as `haskie install claude --scope project`
    writes them, rendered by haskie's own code for `collection` as the instance describes it."""
    from haskie import claude
    from haskie.collection.collection import CollectionSummary, DocumentCounts

    info = setup.call("GET", f"/api/collections/{collection}", api)
    summary = CollectionSummary(collection, DocumentCounts(), 0.0, info.get("description", ""))
    written = []
    for relative, text in (
        (f"rules/{claude.SKILL_NAME}.md", claude.render_rule([summary])),
        (f"skills/{claude.SKILL_NAME}/SKILL.md", claude.render_skill([summary])),
    ):
        target = work / ".claude" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        written.append(target)
    return written


def allowed_tools(arm: str) -> list[str]:
    return [*CODING_TOOLS, *(HASKIE_TOOLS if arm in HASKIE_ARMS else ())]


# A parent Claude Code session and its spawned `claude -p` child share the same on-disk OAuth
# credentials; if the child's launch lands mid-refresh, it fails its very first turn with this
# exact message and no tool calls, indistinguishable from a real task failure except for the
# text - and a retry a moment later succeeds cleanly.
AUTH_RETRY_MARKER = "OAuth session expired and could not be refreshed"
AUTH_RETRY_ATTEMPTS = 3


def run_agent(task: Task, arm: str, directory: Path, model: str, api: str, collection: str) -> int:
    """The agent's run. An unprompted arm reaches haskie through `scope.Proxy`, which shows it
    the task's collection alone: told no collection, it would otherwise search every task's."""
    if arm not in UNMENTIONED_ARMS:
        endpoint = f"{api.rstrip('/')}/mcp"
        return _run_agent(task, arm, directory, model, api, collection, endpoint)
    with scope.Proxy(api, collection) as proxy:
        return _run_agent(task, arm, directory, model, api, collection, proxy.url)


def _run_agent(
    task: Task, arm: str, directory: Path, model: str, api: str, collection: str, endpoint: str
) -> int:
    work = directory / "work"
    work.mkdir(parents=True, exist_ok=True)
    mcp = directory / "mcp.json"
    servers = {"haskie": {"type": "http", "url": endpoint}} if arm in HASKIE_ARMS else {}
    mcp.write_text(json.dumps({"mcpServers": servers}, indent=2) + "\n")
    if arm == "i":
        install_haskie(work, api, collection)
    settings = directory / "settings.json"  # outside `work/`, among the reads it is denied
    settings.write_text(json.dumps(agent_settings(arm, api, collection), indent=2) + "\n")

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
            "--settings",
            str(settings.resolve()),
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
    cost: metrics.Cost

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
    if arm in HASKIE_ARMS:
        found = metrics.retrieved(transcript, task.evidence)
    elif arm == "d":
        found = metrics.opened(transcript, task.evidence)
    else:
        found = []
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
        metrics.cost(transcript),
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
    directory = RUNS / model / arm / task.name / str(sample)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    scratch_work(directory)
    # A copy, not a hardlink: an agent editing a corpus file in place would otherwise rewrite the
    # shared original under every later run. Removed afterwards so 30 runs don't hold 30 copies.
    corpus = directory / "work" / "corpus"
    if arm == "d":
        shutil.copytree(task.corpus, corpus)
    started = time.monotonic()
    name = task.collection or collection
    if arm == "f":
        name = setup.small_chunks(name)
    returncode = run_agent(task, arm, directory, model, api, name)
    seconds = time.monotonic() - started
    if arm == "d":
        shutil.rmtree(corpus)
    keep_work(directory)
    doc_root = home / "documents"
    return grade(task, arm, directory, returncode, doc_root, seconds, model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="*", help=f"task names or groups: {', '.join(GROUPS)}")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(DEFAULT_ARMS))
    parser.add_argument("--model", default=os.environ.get("EVAL_MODEL", "sonnet"))
    parser.add_argument("--api", default=os.environ.get("HASKIE_EVAL_URL", "http://127.0.0.1:8123"))
    parser.add_argument(
        "--home", type=Path, default=Path(os.environ.get("HASKIE_EVAL_HOME", ROOT / ".haskie-eval"))
    )
    parser.add_argument(
        "--embed-api", default=os.environ.get("HASKIE_EVAL_EMBED_URL", "http://127.0.0.1:8124")
    )
    parser.add_argument(
        "--embed-home",
        type=Path,
        default=Path(os.environ.get("HASKIE_EVAL_EMBED_HOME", ROOT / ".haskie-eval-embed")),
    )
    parser.add_argument(
        "--collection", default=os.environ.get("HASKIE_EVAL_COLLECTION", "eval-programming-books")
    )
    parser.add_argument("--samples", type=int, default=int(os.environ.get("EVAL_SAMPLES", "3")))
    parser.add_argument(
        "--first-sample",
        type=int,
        default=0,
        help="index of the first sample to run: add samples to a cell without redoing earlier ones",
    )
    args = parser.parse_args()

    unknown = [name for name in args.tasks if name not in TASKS and name not in GROUPS]
    if unknown:
        parser.error(f"unknown task or group: {', '.join(unknown)}")
    names = [t for name in args.tasks or ["all"] for t in GROUPS.get(name, (name,))]
    tasks = [load_task(t) for t in dict.fromkeys(names)]
    instance = {arm: (args.api, args.home) for arm in ARMS}
    instance["e"] = (args.embed_api, args.embed_home)
    RUNS.mkdir(parents=True, exist_ok=True)
    warm_auth()
    results = [
        run_one(task, arm, sample, args.model, *instance[arm], args.collection)
        for arm in args.arms
        for task in tasks
        for sample in range(args.first_sample, args.first_sample + args.samples)
    ]
    write_report(results, RUNS)
    return 0 if all(r.correct for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
