"""Installing haskie into Claude Code: the MCP entry, the SessionStart hook and the skill.

Everything Claude Code's own configuration looks like lives here - where its files are, the argv
its CLI takes, the shape of a hook in its settings - so `cli` stays the way in and never a second
way of doing the work. Failures are `HaskieError`, not Typer's: this module knows nothing about a
terminal, and a second client (or a route) must be able to call it.

The MCP tool descriptions are the handler docstrings, so they say what each tool does. What they
cannot say is when to reach for haskie at all, which search to start with, or that these documents
are the user's own and outrank a web result. That is what a skill is for, and it is why the trigger
line is generated from the collections a home actually holds rather than shipped as a fixed string.
"""

import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from haskie import home
from haskie.errors import Conflict, InvalidInput

if TYPE_CHECKING:
    from haskie.collection.collection import CollectionSummary

Scope = Literal["user", "project"]  # where Claude Code keeps a setting: this user, or this project

SKILL_NAME = "haskie"
USER_CLAUDE = Path.home() / ".claude"
HOOK_MARKER = " ensure --home "  # what identifies a hook of ours, whatever path invoked it
HOOK_TIMEOUT_SECONDS = 90
DEFAULT_HOST = "127.0.0.1"  # loopback: one user's documents, and nothing authenticates a caller
DEFAULT_PORT = 8000
# Spelled out rather than imported from `app`: importing the Litestar app would cost every
# `haskie` invocation the whole web stack. `test_the_default_url_matches_where_mcp_is_mounted`
# is what keeps this in step with `app.MCP_PATH`.
MCP_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/mcp"

# Long enough to be recognisable, short enough that a collection with an essay for a description
# does not crowd every other collection out of the trigger line.
DESCRIPTION_BUDGET = 120

_TRIGGER = (
    "Search the user's own curated document collections instead of answering from the web or from "
    "memory. Use whenever a question touches a topic they have collected sources on{topics}; when "
    'they say "my documents", "my collection", "what do my sources say"; or when they want an '
    "answer cited to something they own."
)

_ANNOUNCEMENT = "This conversation's haskie session id is {session_id}."


def session_announcement(session_id: str) -> str:
    """What the SessionStart hook prints so the conversation's id reaches the tools. The skill
    quotes the same sentence back at the agent, so both come from here."""
    return (
        f"{_ANNOUNCEMENT.format(session_id=session_id)} "
        "Pass it as `session_id` on every haskie tool call that takes one."
    )


_BODY = f"""\
# haskie — the user's own sources

The collections behind these tools are documents the user chose and trusts. For anything they
cover, they outrank a web search: prefer them, and say which document the answer came from.

Run `list_collections` whenever you are unsure what exists. Each collection's `description` says
what it is for, and that list is authoritative — the trigger above is a snapshot from install
time.

## Which search

Two tools, for the two questions.

- **"What do the sources say about X?"** — `search_excerpts`. Hybrid retrieval over the scope,
  returned as excerpts rather than raw chunks: hits that landed on neighbouring chunks are one
  piece of text, widened both ways to whole sentences, best first. This is what you answer from.
- **"Which documents cover X, and which collections hold them?"** — `search_sources`. One row per
  document — its best passage, how much of it matched, the hot sections inside it — plus the
  smallest set of collections that covers every document returned.

Both take an optional list of collections. Given one, they search those; given none, the
session's collections; with no session selection either, everything the user owns.

Run `search_sources` first to see what covers the topic, hand the collections it names to
`set_session_collections`, then stay on `search_excerpts` for the rest of the conversation.

## The session id

haskie's SessionStart hook prints this conversation's id into your context, as
"{_ANNOUNCEMENT.format(session_id="…")}" — use that id, unchanged, on every haskie tool call that
takes one. The argument is optional in the schema, but a call without it belongs to no session, so
the user's Sessions page never shows what this conversation searched, imported or started. If that
line is not in your context, use one short stable string for the whole conversation instead.

`set_session_collections` takes that id too: pass the collections that match the topic — the ones
`search_sources` named — then call `search_excerpts` with the same id. It is the difference between
searching the user's shelf on this subject and searching everything they own.

## Reading the results

Every excerpt carries the text plus `header` (the enclosing headings, as a breadcrumb) and
`location` (`doc p.3-4 L10-20`). Both are written to be quoted — cite the document by name, not
"your collection says". An excerpt already begins and ends on a sentence boundary, so quote it as
it comes. A source row carries the same two fields for its best passage, and each of its sections
names the headings its matches sit under.

## Rules

- No hits is an answer. Say the collections do not cover it, then fall back to the web — never
  pass a web result off as one of their sources.
- A 503 means a model is still downloading. Wait and try again, or tell the user what is holding
  the search up.
- A document exists on its own and belongs to any number of collections. `add_document` imports it
  once; `add_document_to_collection` attaches it and queues the index, and
  `remove_document_from_collection` detaches it without deleting it. It is not searchable in a
  collection until that index finishes; poll `list_collection_documents`.
- Creating and deleting collections is not exposed over MCP. Point the user at the web UI.
"""


def _claude_dir(scope: Scope) -> Path:
    """Claude Code's configuration directory for `scope`: the user's, or the working directory's."""
    return USER_CLAUDE if scope == "user" else Path.cwd() / ".claude"


def skill_path(scope: Scope) -> Path:
    """Where the skill file goes. `project` keeps it with a repository, `user` with the user."""
    return _claude_dir(scope) / "skills" / SKILL_NAME / "SKILL.md"


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
    trigger = _TRIGGER.format(topics=_topics(collections))
    return f"---\nname: {SKILL_NAME}\ndescription: >-\n  {trigger}\n---\n\n{_BODY}"


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


def own_command() -> list[str]:
    """How to invoke haskie from somewhere else: absolute, because a hook and an MCP client both
    run with a PATH of their own. Falls back to this interpreter, which `__main__` makes work."""
    found = shutil.which("haskie")
    return [found] if found else [sys.executable, "-m", "haskie"]


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
