"""Installing haskie into Claude Code: the MCP entry, the SessionStart hook, the skill and the rule.

Everything Claude Code's own configuration looks like lives here - where its files are, the argv
its CLI takes, the shape of a hook in its settings - so `cli` stays the way in and never a second
way of doing the work. Failures are `HaskieError`, not Typer's: this module knows nothing about a
terminal, and a second client (or a route) must be able to call it.

The MCP tool descriptions are the handler docstrings, so they say what each tool does. What they
cannot say is when to reach for haskie at all, which search to start with, or that these documents
are the user's own and outrank a web result. That is what a skill is for, and it is why the trigger
line is generated from the collections a home actually holds rather than shipped as a fixed string.

A skill is only weighed when Claude is choosing a tool for a task. A plain knowledge question, a
plan, or a moment of doubt does not read as a task, so the skill never fires and the answer comes
from memory. The rule under `rules/` closes that gap: Claude Code loads every file there into the
system prompt of every session, which is how Context7 gets consulted "even when you think you know
the answer". The rule says *when*; the skill says *how*.
"""

import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

from haskie import home
from haskie.errors import Conflict, InvalidInput

if TYPE_CHECKING:
    from haskie.collection.collection import CollectionSummary


class Scope(StrEnum):  # where Claude Code keeps a setting: this user, or this project
    USER = "user"
    PROJECT = "project"


SKILL_NAME = "haskie"
USER_CLAUDE = Path.home() / ".claude"
# The skill and the rule are markdown, laid out under `claude_code/` exactly as they land under
# `.claude/`, with `{topics}` and `{announcement}` for what only install time knows.
TEMPLATES = files("haskie") / "claude_code"
HOOK_MARKER = " ensure --home "  # what identifies a hook of ours, whatever path invoked it
HOOK_TIMEOUT_SECONDS = 90
DEFAULT_HOST = "127.0.0.1"  # loopback: one user's documents, and nothing authenticates a caller
# An environment variable, so a development install (`haskie-dev`, the mise tasks) can serve
# beside the installed haskie on 8451 without a `--port` on every command.
DEFAULT_PORT = int(os.environ.get("HASKIE_PORT", "8451"))
# Spelled out rather than imported from `app`: importing the Litestar app would cost every
# `haskie` invocation the whole web stack. `test_the_default_url_matches_where_mcp_is_mounted`
# is what keeps this in step with `app.MCP_PATH`.
MCP_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/mcp"

# Long enough to be recognisable, short enough that a collection with an essay for a description
# does not crowd every other collection out of the trigger line.
DESCRIPTION_BUDGET = 120

_ANNOUNCEMENT = "This conversation's haskie session id is {session_id}."


def session_announcement(session_id: str) -> str:
    """What the SessionStart hook prints so the conversation's id reaches the tools. The skill
    quotes the same sentence back at the agent, so both come from here."""
    return (
        f"{_ANNOUNCEMENT.format(session_id=session_id)} "
        "Pass it as `session_id` on every haskie tool call that takes one."
    )


def template(relative: str) -> str:
    """One of the markdown files under `claude_code/`, by the path it will have under `.claude/`."""
    return (TEMPLATES / relative).read_text(encoding="utf-8")


def _claude_dir(scope: Scope) -> Path:
    """Claude Code's configuration directory for `scope`: the user's, or the working directory's."""
    return USER_CLAUDE if scope == Scope.USER else Path.cwd() / ".claude"


def skill_path(scope: Scope) -> Path:
    """Where the skill file goes. `project` keeps it with a repository, `user` with the user."""
    return _claude_dir(scope) / "skills" / SKILL_NAME / "SKILL.md"


def rule_path(scope: Scope) -> Path:
    """Where the rule goes. Claude Code loads every `rules/*.md` into each session's context."""
    return _claude_dir(scope) / "rules" / f"{SKILL_NAME}.md"


def settings_path(scope: Scope) -> Path:
    """Where the SessionStart hook goes, beside the skill it keeps working."""
    return _claude_dir(scope) / "settings.json"


def _topics(collections: "list[CollectionSummary]") -> str:
    """The clause that decides when Claude loads the skill, built from the real collections.

    A collection with no description contributes its name alone: a name like "roasting" is already
    a topic, and inventing a description for it would be worse than saying nothing.
    """
    if not collections:
        return ""
    described = []
    for collection in collections:
        # `shorten` collapses the whitespace and truncates on a word boundary; the `rstrip` is
        # because trailing punctuation would collide with the separator: "…defects.; adr".
        summary = textwrap.shorten(
            collection.description, DESCRIPTION_BUDGET, placeholder="…"
        ).rstrip(" .;")
        described.append(f"{collection.name}: {summary}" if summary else collection.name)
    return " — currently " + "; ".join(described)


def render_skill(collections: "list[CollectionSummary]") -> str:
    """The SKILL.md for this home. Deterministic, so re-running rewrites rather than accumulates."""
    return template(f"skills/{SKILL_NAME}/SKILL.md").format(
        topics=_topics(collections), announcement=_ANNOUNCEMENT.format(session_id="…")
    )


def render_rule(collections: "list[CollectionSummary]") -> str:
    """The always-loaded rule for this home, naming the same collections as the skill's trigger."""
    return template(f"rules/{SKILL_NAME}.md").format(topics=_topics(collections))


def _write(destination: Path, text: str) -> Path:
    """Write into Claude Code's directory, making it first.

    Atomic, because Claude Code reads these files while we write them and half of one is worse
    than none: a broken skill, or a settings file that takes the rest of its contents with it.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    home.atomic_write_sync(destination, text)
    return destination


def write_skill(scope: Scope, collections: "list[CollectionSummary]") -> Path:
    """Put the skill where Claude Code looks for it, and say where that was."""
    return _write(skill_path(scope), render_skill(collections))


def write_rule(scope: Scope, collections: "list[CollectionSummary]") -> Path:
    """Put the rule where Claude Code loads it every session, and say where that was."""
    return _write(rule_path(scope), render_rule(collections))


def own_command() -> list[str]:
    """How to invoke this haskie from somewhere else: absolute, because a hook and an MCP client
    both run with a PATH of their own. The script beside this interpreter, not the first `haskie`
    on PATH, which may be another install (`haskie-dev` beside the system tool). Falls back to
    this interpreter, which `__main__` makes work."""
    script = Path(sys.executable).with_name("haskie")
    return [str(script)] if script.is_file() else [sys.executable, "-m", "haskie"]


def register_mcp(url: str, scope: Scope) -> str | None:
    """Add the HTTP entry to Claude Code, replacing any entry of ours already there.

    HTTP rather than stdio: litestar-mcp serves MCP `2026-07-28`, which replaced `initialize`
    with `server/discover`, and a stdio client that opens with `initialize` never connects.

    Returns the command to run by hand when the `claude` CLI is not installed, so a missing CLI
    costs the user one copy-paste rather than the whole install.
    """
    arguments = ["mcp", "add", "-s", scope, "--transport", "http", SKILL_NAME, url]
    claude_cli = shutil.which("claude")
    if claude_cli is None:
        return "claude " + " ".join(arguments)
    # Remove first, so re-running updates the entry instead of failing on the name. No entry is
    # the normal case, so that failure is the expected one.
    subprocess.run(
        [claude_cli, "mcp", "remove", "-s", scope, SKILL_NAME], capture_output=True, check=False
    )
    done = subprocess.run([claude_cli, *arguments], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise Conflict((done.stderr or done.stdout).strip() or "`claude mcp add` failed")
    return None


def hook_command(home_dir: Path, url: str) -> str:
    """The SessionStart command, as one shell string: that is the shape Claude Code runs."""
    invocation = " ".join(shlex.quote(part) for part in own_command())
    return f"{invocation} ensure --home {shlex.quote(str(home_dir))} --url {url} --no-wait"


def install_hook(scope: Scope, home_dir: Path, url: str) -> bool:
    """Teach Claude Code to bring haskie up at the start of a session.

    The MCP entry is HTTP, so a session that starts while nothing is serving gets no haskie tools
    at all, and nothing says why. A SessionStart hook running `haskie ensure` fixes that: it costs
    one loopback request when the server is already up, which is the usual case.

    Returns whether this call added the hook. Reads and rewrites the file as a whole, so an
    existing settings file keeps everything else in it.
    """
    import msgspec  # only this writes a settings file; `cli` imports this module on every run

    settings_file = settings_path(scope)
    command = hook_command(home_dir, url)
    settings: dict[str, Any] = {}
    if settings_file.is_file():
        try:
            settings = msgspec.json.decode(settings_file.read_bytes(), type=dict[str, Any])
        except msgspec.DecodeError as exc:
            raise InvalidInput(f"{settings_file} is not valid JSON: {exc}") from None
    matchers = settings.setdefault("hooks", {}).setdefault("SessionStart", [])
    # Matched on the shape of the command, not on the path `haskie` happens to have today: an
    # upgrade that moves the executable must still replace the hook rather than stack a copy.
    ours = [
        hook
        for matcher in matchers
        for hook in matcher.get("hooks", [])
        if HOOK_MARKER in str(hook.get("command", ""))
    ]
    for hook in ours:
        hook["command"] = command
    if not ours:
        matchers.append(
            {"hooks": [{"type": "command", "command": command, "timeout": HOOK_TIMEOUT_SECONDS}]}
        )
    _write(settings_file, msgspec.json.format(msgspec.json.encode(settings)).decode() + "\n")
    return not ours


async def read_collections() -> "list[CollectionSummary]":
    """Straight from the database, not over HTTP: installing must work with the server stopped.

    `collection` is imported here rather than at module level: it reaches LanceDB, and the CLI
    imports this module on every invocation for `MCP_URL`.
    """
    from haskie.collection.collection import Collection
    from haskie.paging import MAX_PAGE_SIZE, PageRequest

    page = await Collection.page(PageRequest(page_size=MAX_PAGE_SIZE))
    return page.items
