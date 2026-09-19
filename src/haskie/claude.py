"""Installing haskie into Claude Code: the MCP entry and the skill that decides when to use it.

The MCP tool descriptions are the handler docstrings, so they say what each tool does. What they
cannot say is when to reach for haskie at all, which search to start with, or that these documents
are the user's own and outrank a web result. That is what a skill is for, and it is why the trigger
line is generated from the collections a home actually holds rather than shipped as a fixed string.
"""

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from haskie.collection import CollectionSummary

Scope = Literal["user", "project"]  # where Claude Code keeps a setting: this user, or this project

SKILL_NAME = "haskie"
USER_CLAUDE = Path.home() / ".claude"
PROJECT_CLAUDE = Path(".claude")
DEFAULT_HOST = "127.0.0.1"  # loopback: one user's documents, and nothing authenticates a caller
DEFAULT_PORT = 8000
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

_BODY = """\
# haskie — the user's own sources

The collections behind these tools are documents the user chose and trusts. For anything they
cover, they outrank a web search: prefer them, and say which document the answer came from.

Run `list_collections` whenever you are unsure what exists. Each collection's `description` says
what it is for, and that list is authoritative — the trigger above is a snapshot from install
time.

## Which search

- **Cold start, one question, no setup** — `search_text`. BM25 over every collection at once, with
  no session and no embedding model.
- **"Which documents cover X?"** — `search_documents`. One row per document, to pick a shortlist
  before reading passages.
- **A conversation scoped to a topic** — `set_session_collections` once, then `search` for the
  rest of it.
- **One known collection, tuned options** — `search_collection`, with `mode`, `fusion`, `reranker`
  and `candidates`.

`set_session_collections` takes a session id you choose: use one stable id for the whole
conversation (the conversation's own id is a good one), pass the collections that match the topic,
then call `search` with that same id. It is the difference between searching the user's shelf on
this subject and searching everything they own.

Start with `search_text` when in doubt. It needs nothing set up and it answers from a cold start.

## Reading the results

Every hit carries the text plus `header` (the enclosing headings, as a breadcrumb) and `location`
(`doc p.3-4 L10-20`). Both are written to be quoted — cite the document by name, not "your
collection says".

## Rules

- No hits is an answer. Say the collections do not cover it, then fall back to the web — never
  pass a web result off as one of their sources.
- A 503 means a model is still downloading. `search_text` needs no model, so use it meanwhile.
- A document exists on its own and belongs to any number of collections. `add_document` imports it
  once; `add_document_to_collection` attaches it and queues the index, and
  `remove_document_from_collection` detaches it without deleting it. It is not searchable in a
  collection until that index finishes; poll `list_collection_documents`.
- Creating and deleting collections is not exposed over MCP. Point the user at the web UI.
"""


def _claude_dir(scope: Scope) -> Path:
    """Claude Code's configuration directory for `scope`: the user's, or the working directory's."""
    return USER_CLAUDE if scope == "user" else Path.cwd() / PROJECT_CLAUDE


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


def write_skill(scope: Scope, collections: "list[CollectionSummary]") -> Path:
    """Put the skill where Claude Code looks for it, and say where that was.

    Atomic, because Claude Code reads this file and a half-written one is a broken skill.
    """
    from haskie import home  # local: `home` is what `cli` already imported to call this

    destination = skill_path(scope)
    destination.parent.mkdir(parents=True, exist_ok=True)
    home.atomic_write_sync(destination, render_skill(collections))
    return destination


async def read_collections() -> "list[CollectionSummary]":
    """Straight from the database, not over HTTP: installing must work with the server stopped.

    Imported here rather than at module level: `collection` reaches LanceDB, and this module is
    what the CLI reads `MCP_URL` from on every invocation.
    """
    from haskie.collection import Collection
    from haskie.paging import MAX_PAGE_SIZE, page_request

    page = await Collection.page(page_request(page_size=MAX_PAGE_SIZE))
    return page.items
