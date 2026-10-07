"""Installing haskie into Claude Code: the MCP entry, the SessionStart hook, the skill and the rule.

Codex reuses the instruction templates, hook format, atomic writes and installation registry.
Its TOML configuration lives in `codex.py`; its hook emits the rule as session context.

Everything about Claude Code's own configuration lives here: where its files are, the argv its
CLI takes, the shape of a hook in its settings. So `cli` stays the way in and never a second way
of doing the work. Failures are `HaskieError`, not Typer's: this module knows nothing about a
terminal, and a second client (or a route) must be able to call it.

The MCP tool descriptions are the handler docstrings, so they say what each tool does. What they
cannot say is when to reach for haskie at all, which search to start with, or that these documents
are the user's own and outrank a web result. That is what a skill is for, and it is why the trigger
line is generated from the collections a home holds rather than shipped as a fixed string.

A skill is only weighed when Claude is choosing a tool for a task. A plain knowledge question, a
plan, or a moment of doubt does not read as a task, so the skill never fires and the answer comes
from memory. The rule under `rules/` closes that gap: Claude Code loads every file there into the
system prompt of every session, which is how Context7 gets consulted "even when you think you know
the answer". The rule says *when*; the skill says *how*.

Both name the collections, so both go stale when one changes. `install claude` records the
directory it wrote into (the `installations` table), and every collection change rewrites the
skill and rule there in the background (`refresh_in_background`).
"""

import asyncio
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import textwrap
import threading
from enum import StrEnum
from importlib.resources import files
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from haskie import home
from haskie.errors import Conflict, InvalidInput
from haskie.logs import get_logger

if TYPE_CHECKING:
    from haskie.collection.collection import CollectionSummary


class Scope(StrEnum):  # where an agent keeps a setting: this user, or this project
    USER = "user"
    PROJECT = "project"


SKILL_NAME = "haskie"
# Claude Code moves every `~/.claude` path under `CLAUDE_CONFIG_DIR` when it is set, so the skill,
# rule and hook must follow it or land where it never reads them.
USER_CLAUDE = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude").expanduser()
# The skill and the rule are markdown, laid out under `claude_code/` exactly as they land under
# `.claude/`, with `{topics}` and `{announcement}` for what only install time knows.
TEMPLATES = files("haskie") / "claude_code"
# What identifies a hook of ours, whatever path invoked it. The second is the shape before `ensure`
# became `run`: still ours, so re-installing replaces it rather than leaving a hook that fails at
# every session start beside the new one.
HOOK_MARKERS = (" run --home ", " ensure --home ")
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

_log = get_logger(__name__)

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


def claude_dir(scope: Scope) -> Path:
    """Claude Code's configuration directory for `scope`: the user's, or the working directory's.
    Absolute, because an installation is recorded by it and refreshed from any working directory."""
    return (USER_CLAUDE if scope == Scope.USER else Path.cwd() / ".claude").absolute()


def skill_path(directory: Path) -> Path:
    """Where the skill file goes in a Claude Code configuration directory."""
    return directory / "skills" / SKILL_NAME / "SKILL.md"


def rule_path(directory: Path) -> Path:
    """Where the rule goes. Claude Code loads every `rules/*.md` into each session's context."""
    return directory / "rules" / f"{SKILL_NAME}.md"


def settings_path(directory: Path) -> Path:
    """Where the SessionStart hook goes, beside the skill it keeps working."""
    return directory / "settings.json"


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
        ).rstrip(" .,;:")
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
    # The file the rename replaces is the one a link points at, not the link: a settings file
    # linked in from a dotfiles repository must stay linked, and a private one must stay private.
    target = destination.resolve()
    # Claude Code watches these files, and most refreshes change nothing in them
    if target.is_file() and target.read_text(encoding="utf-8") == text:
        return destination
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode: int | None = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        mode = None
    with home.atomic_replace(target) as tmp, open(tmp, "w", encoding="utf-8") as stream:
        # On the open descriptor and before the text: never readable wider than the file it
        # replaces, and a read-only file still gets written.
        if mode is not None:
            os.fchmod(stream.fileno(), mode)
        stream.write(text)
    return destination


def write_instructions(directory: Path, collections: "list[CollectionSummary]") -> list[Path]:
    """Put the skill and the rule where Claude Code reads them, and say where that was."""
    return [
        _write(skill_path(directory), render_skill(collections)),
        _write(rule_path(directory), render_rule(collections)),
    ]


def remove_instructions(directory: Path) -> list[Path]:
    """Delete the skill and the rule, and say which were there. The skill's folder goes too once
    it is empty; a file the user put in it keeps it."""
    removed = []
    for path in (skill_path(directory), rule_path(directory)):
        if path.is_file():
            path.unlink()
            removed.append(path)
    folder = skill_path(directory).parent
    if folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()
    return removed


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
        return shlex.join(["claude", *arguments])
    unregister_mcp(scope)  # so re-running updates the entry instead of failing on the name
    done = subprocess.run([claude_cli, *arguments], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise Conflict((done.stderr or done.stdout).strip() or "`claude mcp add` failed")
    return None


def unregister_mcp(scope: Scope) -> str | None:
    """Remove haskie's entry from Claude Code. Returns the command to run by hand when the
    `claude` CLI is not installed, as `register_mcp` does.

    `claude mcp remove` fails when there is no entry, and uninstalling what is not installed is no
    error, so its outcome is not checked: the entry is gone either way."""
    arguments = ["mcp", "remove", "-s", scope, SKILL_NAME]
    claude_cli = shutil.which("claude")
    if claude_cli is None:
        return shlex.join(["claude", *arguments])
    subprocess.run([claude_cli, *arguments], capture_output=True, check=False)
    return None


def address(url: str) -> tuple[str, int]:
    """The host and port a server for `url` binds, with haskie's defaults for what it leaves out."""
    parts = urlsplit(url)
    return parts.hostname or DEFAULT_HOST, parts.port or DEFAULT_PORT


def run_command(home_dir: Path, url: str, mode: Literal["--hook", "--foreground"]) -> list[str]:
    """This haskie's `run` for `home_dir` at the address of `url`, with `mode`: `--hook` for the
    SessionStart hook, `--foreground` for the server the CLI starts."""
    host, port = address(url)
    return [
        *own_command(),
        "run",
        "--home",
        str(home_dir),
        "--host",
        host,
        "--port",
        str(port),
        mode,
    ]


def hook_command(home_dir: Path, url: str) -> str:
    """The SessionStart command, as one shell string: that is the shape Claude Code runs."""
    return shlex.join(run_command(home_dir, url, "--hook"))


def install_hook(
    directory: Path,
    home_dir: Path,
    url: str,
    *,
    filename: str = "settings.json",
    rules: Path | None = None,
) -> bool:
    """Make the agent start haskie when a session starts.

    The MCP entry is HTTP, so a session that starts while nothing is serving gets no haskie tools
    at all, and nothing says why. A SessionStart hook running `haskie run` fixes that: it costs
    one loopback request when the server is already up, which is the usual case.

    Returns whether this call added the hook. Reads and rewrites the file as a whole, so an
    existing settings file keeps everything else in it. Codex uses `hooks.json` and needs
    the rule emitted by the hook because it does not load prose from `rules/` itself.
    """
    settings_file = directory / filename
    command = hook_command(home_dir, url)
    if rules is not None:
        command += " " + shlex.join(["--hook-rules", str(rules)])
    settings = _read_settings(settings_file)
    matchers = _session_start(settings, settings_file)
    ours = [hook for matcher in matchers for hook in matcher.get("hooks", []) if _is_ours(hook)]
    for hook in ours:
        hook["command"] = command
    if not ours:
        matchers.append(
            {"hooks": [{"type": "command", "command": command, "timeout": HOOK_TIMEOUT_SECONDS}]}
        )
    _write(settings_file, json.dumps(settings, indent=2, ensure_ascii=False) + "\n")
    return not ours


def uninstall_hook(directory: Path, *, filename: str = "settings.json") -> bool:
    """Remove every haskie SessionStart hook from the settings file, whichever home it starts,
    and leave every other hook and setting as it was. A matcher that held only haskie's goes with
    it. Returns whether there was one."""
    settings_file = directory / filename
    settings = _read_settings(settings_file)
    matchers = _session_start(settings, settings_file)
    removed = False
    for matcher in list(matchers):
        hooks = matcher.get("hooks", [])
        theirs = [hook for hook in hooks if not _is_ours(hook)]
        if len(theirs) == len(hooks):
            continue
        removed = True
        if theirs:
            matcher["hooks"] = theirs
        else:
            matchers.remove(matcher)
    if not removed:
        return False
    # the keys `install_hook` made, when nothing else is left in them
    if not matchers:
        del settings["hooks"]["SessionStart"]
    if not settings["hooks"]:
        del settings["hooks"]
    _write(settings_file, json.dumps(settings, indent=2, ensure_ascii=False) + "\n")
    return True


def _read_settings(settings_file: Path) -> Any:
    """A Claude Code settings file, or an empty one when it does not exist yet."""
    if not settings_file.is_file():
        return {}
    try:
        return json.loads(settings_file.read_bytes())
    except ValueError as exc:
        raise InvalidInput(f"{settings_file} is not valid JSON: {exc}") from None


def _is_ours(hook: dict[str, Any]) -> bool:
    """Whether a hook is haskie's. Matched on the shape of the command, not on the path `haskie`
    happens to have today: an upgrade that moves the executable must still find it."""
    return any(marker in str(hook.get("command", "")) for marker in HOOK_MARKERS)


def _session_start(settings: Any, settings_file: Path) -> list[dict[str, Any]]:
    """The SessionStart matchers of a settings file, made when missing.

    Refuses a file shaped otherwise, such as `{"hooks": null}`, rather than guess where the hook
    goes in it: rewriting a file we could not follow would throw the user's settings away.
    """
    hooks = settings.setdefault("hooks", {}) if isinstance(settings, dict) else None
    matchers = hooks.setdefault("SessionStart", []) if isinstance(hooks, dict) else None
    if not isinstance(matchers, list) or not all(
        isinstance(matcher, dict)
        and isinstance(matcher.get("hooks", []), list)
        and all(isinstance(hook, dict) for hook in matcher.get("hooks", []))
        for matcher in matchers
    ):
        raise InvalidInput(f"{settings_file} does not hold `hooks.SessionStart` as a list of hooks")
    return matchers


async def read_collections() -> "list[CollectionSummary]":
    """The collections the trigger names: those a search can find anything in
    (`Collection.searchable`). One with none answers every search with nothing, so naming it would
    send an agent there for nothing.

    Straight from the database, not over HTTP: installing must work with the server stopped.

    `collection` is imported here rather than at module level: it reaches LanceDB, and the CLI
    imports this module on every invocation for `MCP_URL`.
    """
    from haskie.collection.collection import Collection
    from haskie.paging import MAX_PAGE_SIZE, PageRequest

    page = await Collection.page(PageRequest(page_size=MAX_PAGE_SIZE))
    searchable = set(await Collection.searchable([one.name for one in page.items]))
    return [collection for collection in page.items if collection.name in searchable]


# --- keeping installations in step with the collections ---------------------------------------

AGENT = "claude"  # this agent's value in the `installations` table

# One refresh at a time, and one more after it when a change landed while it ran: a refresh that
# read the collections before the change must not be the last to write. `_refresh_lock` guards
# the pair, because the pipeline changes collections from an event loop of its own.
_refresh_wanted = False
_refreshing = False
_refresh_lock = threading.Lock()
_refresh_tasks: set[asyncio.Task[None]] = set()  # the loop keeps only a weak reference


async def record_installation(directory: Path, agent: str = AGENT) -> None:
    """Remember that the skill and rule live in `directory`, so a collection change reaches them."""
    from sqlalchemy.dialects.sqlite import insert

    from haskie import db
    from haskie.tables import installations

    async with db.connect() as conn:
        await conn.execute(
            insert(installations)
            .values(agent=agent, directory=str(directory))
            .on_conflict_do_nothing()
        )


async def forget_installation(directory: Path, agent: str = AGENT) -> bool:
    """Stop rewriting the skill and rule in `directory`; returns whether it was recorded."""
    from sqlalchemy import delete

    from haskie import db
    from haskie.tables import installations

    async with db.connect() as conn:
        result = await conn.execute(
            delete(installations)
            .where(installations.c.agent == agent)
            .where(installations.c.directory == str(directory))
        )
        return result.rowcount == 1


def _hooks_this_home(directory: Path, *, filename: str = "settings.json") -> bool:
    """Whether the SessionStart hook in `directory` starts this home. The last install into a
    directory takes its MCP entry and its hook, so it takes the skill and rule too: a home that
    kept rewriting them would name collections the registered server does not serve."""
    settings_file = directory / filename
    matchers = _session_start(_read_settings(settings_file), settings_file)
    return any(
        _is_ours(hook) and ("--home", str(home.HOME)) in pairwise(shlex.split(hook["command"]))
        for matcher in matchers
        for hook in matcher.get("hooks", [])
    )


def _refresh_directory(
    directory: Path, collections: "list[CollectionSummary]", filename: str = "settings.json"
) -> None:
    """Rewrite one installation while it is still this home's: not when the skill and rule are
    gone (removed by hand, or with the whole project), which writing would bring back, and not
    when another home's install took the directory over."""
    installed = skill_path(directory).is_file() or rule_path(directory).is_file()
    if installed and _hooks_this_home(directory, filename=filename):
        write_instructions(directory, collections)


async def refresh_installations() -> None:
    """Rewrite the skill and rule of every recorded installation from the collections as they are
    now. One that is not this home's any more is skipped, not forgotten: only `uninstall` forgets,
    so no order of install steps can lose a record. One that fails is logged, and the next change
    tries it again."""
    import anyio.to_thread
    from sqlalchemy import select

    from haskie import db
    from haskie.tables import installations

    async with db.read() as conn:
        recorded = list(
            await conn.execute(
                select(installations.c.agent, installations.c.directory).where(
                    installations.c.agent.in_([AGENT, "codex"])
                )
            )
        )
    if not recorded:
        return
    collections = await read_collections()
    for agent, directory in recorded:
        try:
            filename = "hooks.json" if agent == "codex" else "settings.json"
            await anyio.to_thread.run_sync(
                _refresh_directory, Path(directory), collections, filename
            )
        except (OSError, InvalidInput) as exc:  # unwritable, or a settings file we cannot follow
            _log.warning(
                "installation_refresh_failed",
                directory=home.scrub(directory),
                error=type(exc).__name__,
            )


def refresh_in_background() -> None:
    """Ask for `refresh_installations`, off the caller's path: a collection request should not wait
    on writing into Claude Code's files. Sync, and called only from a coroutine: the task belongs to
    the loop that asked for it.

    At most once: a crash between the change and the refresh loses it, and the refresh at the next
    startup repairs it."""
    global _refresh_wanted, _refreshing
    with _refresh_lock:
        _refresh_wanted = True
        if _refreshing:
            return
        _refreshing = True
    task = asyncio.get_running_loop().create_task(_refresh_while_wanted())
    _refresh_tasks.add(task)
    task.add_done_callback(_refresh_tasks.discard)


def _next_refresh() -> bool:
    """Whether another round is wanted, taking the request; the task ends when none is."""
    global _refresh_wanted, _refreshing
    with _refresh_lock:
        wanted, _refresh_wanted = _refresh_wanted, False
        _refreshing = wanted
        return wanted


async def _refresh_while_wanted() -> None:
    """`refresh_installations` until no change is waiting, with nothing above it to catch what it
    raises. Cancelled with its loop, it leaves the next request free to start another."""
    global _refreshing
    try:
        while _next_refresh():
            try:
                await refresh_installations()
            except Exception:
                _log.exception("installations_refresh_failed")
    except BaseException:
        with _refresh_lock:
            _refreshing = False
        raise
