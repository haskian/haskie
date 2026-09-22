"""Running one task under one arm, in a directory of its own.

An arm is everything the agent is given besides the prompt: whether the haskie server is reachable
at all, and what guidance comes with it. Each run gets a fresh directory because Claude Code keys
its auto-memory on the working directory, so repeats sharing one would let the second read what
the first wrote.

Permissions are an explicit allowlist rather than `--permission-mode bypassPermissions`: the arms
have to differ only in guidance, and an allowlist also keeps a run from writing to the library it
is supposed to be reading.
"""

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import msgspec

# Read-only haskie tools. Importing nothing from the server keeps the list a deliberate choice:
# an arm that could `add_document` would be editing the corpus it is measured against.
HASKIE_TOOLS = (
    "mcp__haskie__list_collections",
    "mcp__haskie__get_collection",
    "mcp__haskie__list_collection_documents",
    "mcp__haskie__list_documents",
    "mcp__haskie__get_document",
    "mcp__haskie__set_session_collections",
    "mcp__haskie__search",
    "mcp__haskie__search_text",
    "mcp__haskie__search_documents",
    "mcp__haskie__search_collection",
    # The library's own answer to opening a shortlisted document: leaving it out would measure a
    # haskie missing the alternative to the grep this eval exists to watch for.
    "mcp__haskie__document_passages",
)
# What a coding task needs, in every arm. `Grep` and `Read` are here on purpose: taking them away
# would hide the behaviour the eval exists to measure. So is web search - the baseline is meant to
# be Claude as it normally works, and one that cannot look anything up is not a baseline but a
# handicap, which would credit haskie for the blindfold.
CODING_TOOLS = (
    "Read",
    "Write",
    "Edit",
    "Grep",
    "Glob",
    "Bash",
    "TodoWrite",
    "ToolSearch",
    "WebSearch",
    "WebFetch",
)

EVENTS = "events.jsonl"
MCP_FILE = "mcp.json"
# The agent's working directory, kept apart from the run's own record. A baseline run read
# `mcp.json` and its own `events.jsonl` out of the directory it was working in and reported on the
# arm it was in, so the record now sits one level up where the sandbox cannot reach it.
WORK = "work"
DEFAULT_URL = "http://127.0.0.1:8000/mcp"


def binary() -> str:
    """The Claude Code executable. `claude` is a shell alias on some machines, so the real path
    wins over the name."""
    return (
        os.environ.get("EVAL_CLAUDE_BIN")
        or shutil.which("claude")
        or str(Path.home() / ".local/bin/claude")
    )


class Arm(msgspec.Struct):
    """One configuration under test. `skill` and `memory` are file contents, not paths, so an arm
    is a value that can be written down beside its results."""

    name: str
    mcp: bool = True
    skill: str | None = None  # SKILL.md, discovered as a project skill
    memory: str | None = None
    memory_scope: str = "project"  # "project" or "user"
    tools: tuple[str, ...] = CODING_TOOLS


class Run(msgspec.Struct):
    arm: str
    task: str
    directory: str
    session_id: str
    argv: list[str]
    returncode: int = 0


def allowed(arm: Arm) -> list[str]:
    return [*arm.tools, *(HASKIE_TOOLS if arm.mcp else ())]


def workspace(directory: Path) -> Path:
    return directory / WORK


def prepare(arm: Arm, directory: Path, url: str = DEFAULT_URL) -> Path:
    """Write the arm's files out and return the directory the agent will work in."""
    work = workspace(directory)
    work.mkdir(parents=True, exist_ok=True)
    # `--strict-mcp-config` with an empty file is what makes the no-haskie arm honest: the server
    # may well be running, and nothing else should be able to reach it either.
    servers = {"haskie": {"type": "http", "url": url}} if arm.mcp else {}
    (directory / MCP_FILE).write_bytes(msgspec.json.encode({"mcpServers": servers}))
    if arm.skill is not None:
        skill = work / ".claude" / "skills" / "haskie" / "SKILL.md"
        skill.parent.mkdir(parents=True, exist_ok=True)
        skill.write_text(arm.skill, encoding="utf-8")
    if arm.memory is not None:
        if arm.memory_scope == "project":
            (work / "CLAUDE.md").write_text(arm.memory, encoding="utf-8")
        elif arm.memory_scope == "user":
            config_dir = os.environ.get("EVAL_CLAUDE_CONFIG_DIR") or os.environ.get("CLAUDE_CONFIG_DIR")
            if not config_dir:
                raise RuntimeError(
                    "arm memory_scope=user requires EVAL_CLAUDE_CONFIG_DIR (a dedicated Claude config "
                    "directory used only for evals)"
                )
            user_claude = Path(config_dir)
            user_claude.mkdir(parents=True, exist_ok=True)
            (user_claude / "CLAUDE.md").write_text(arm.memory, encoding="utf-8")
        else:
            raise ValueError(f"unknown memory scope: {arm.memory_scope!r}")
    return work


def argv(
    arm: Arm,
    prompt: str,
    directory: Path,
    *,
    session_id: str,
    model: str = "sonnet",
    max_turns: int = 40,
) -> list[str]:
    """The command line for one run. Pure, so an arm can be reviewed without running it.

    Every path here is absolute: the run starts in the workspace, one level below `directory`, so
    a relative `--mcp-config` resolves against the wrong place and the run dies before its first
    turn rather than quietly losing its tools.
    """
    return [
        binary(),
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        # Without this the user's own CLAUDE.md and settings join the run and the arms stop
        # differing only in what the eval set.
        "--setting-sources",
        "user,project" if arm.memory_scope == "user" else "project",
        "--strict-mcp-config",
        "--mcp-config",
        str((directory / MCP_FILE).resolve()),
        "--allowedTools",
        *allowed(arm),
        "--model",
        model,
        "--max-turns",
        str(max_turns),
        "--session-id",
        session_id,
    ]


def execute(
    arm: Arm,
    task: str,
    prompt: str,
    directory: Path,
    *,
    model: str = "sonnet",
    max_turns: int = 40,
    url: str = DEFAULT_URL,
    timeout: float = 900.0,
) -> Run:
    """Run the task and leave `events.jsonl` and `run.json` in `directory`."""
    directory = directory.resolve()
    work = prepare(arm, directory, url)
    session_id = str(uuid.uuid4())
    line = argv(arm, prompt, directory, session_id=session_id, model=model, max_turns=max_turns)
    with (directory / EVENTS).open("wb") as events:
        completed = subprocess.run(
            line,
            cwd=work,
            env={**os.environ, **({"CLAUDE_CONFIG_DIR": os.environ["EVAL_CLAUDE_CONFIG_DIR"]} if os.environ.get("EVAL_CLAUDE_CONFIG_DIR") else {})},
            stdout=events,
            stderr=(directory / "stderr.txt").open("wb"),
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    run = Run(
        arm=arm.name,
        task=task,
        directory=str(directory),
        session_id=session_id,
        argv=line,
        returncode=completed.returncode,
    )
    (directory / "run.json").write_bytes(msgspec.json.encode(run))
    return run
