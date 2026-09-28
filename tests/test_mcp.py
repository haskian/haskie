"""The MCP surface, spoken as an agent speaks it: JSON-RPC over HTTP at `/mcp`, every tool called
with real arguments against a real index, and every error an agent can cause answered as a tool
error rather than a crash.

`test_api` covers which handlers are tools; this covers what calling them does. The requests carry
the metadata and headers the protocol version the server speaks requires, as a real client sends
them.
"""

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
import structlog
from conftest import NO_MODELS, attach_via_api, stage_and_import, wait_for, wait_import
from litestar.testing import AsyncTestClient

from haskie.app import MCP_PATH
from haskie.collection.collection import Collection
from haskie.document import document

pytestmark = pytest.mark.anyio

VERSION = "2026-07-28"  # the protocol version `litestar_mcp` speaks
SESSION = "agent-1"
TOOLS = {
    "search_excerpts",
    "search_sources",
    "set_session_collections",
    "list_collections",
    "get_collection",
    "list_collection_documents",
    "list_documents",
    "get_document",
    "add_document",
    "add_document_to_collection",
    "remove_document_from_collection",
    "describe_document",
    "list_searches",
    "list_gaps",
    "replay_gaps",
    "review_gaps",
    "report_gap",
}
# Two notes on retries and one on ordering, with no word in common between the two topics, so a
# full-text search finds each question in its own note.
RETRIES = "# Retries\n\nA background job retries a failed call, so the call must be idempotent.\n"
ORDERING = (
    "# Ordering\n\nHosts drift apart by milliseconds; never trust wall clocks for sequence.\n"
)
BY_RETRY = "Why must a retried background call be idempotent?"
BY_CLOCK = "Why never trust wall clocks for sequence when hosts drift?"


async def _rpc(
    client: AsyncTestClient, method: str, params: dict | None = None, status: int = 200
) -> dict:
    """One JSON-RPC request, with the metadata and headers the protocol requires."""
    params = {
        **(params or {}),
        "_meta": {
            "io.modelcontextprotocol/protocolVersion": VERSION,
            "io.modelcontextprotocol/clientCapabilities": {},
            "io.modelcontextprotocol/clientInfo": {"name": "test-agent", "version": "0"},
        },
    }
    headers = {
        "accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": VERSION,
        "Mcp-Method": method,
    }
    if "name" in params:
        headers["Mcp-Name"] = params["name"]
    response = await client.post(
        MCP_PATH,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        headers=headers,
    )
    assert response.status_code == status, response.text
    return response.json()


async def _call(client: AsyncTestClient, name: str, arguments: dict) -> tuple[bool, Any]:
    """What one tool call answered: whether it is a tool error, and its payload decoded."""
    reply = await _rpc(client, "tools/call", {"name": name, "arguments": arguments})
    assert "result" in reply, reply
    (content,) = reply["result"]["content"]
    return bool(reply["result"].get("isError")), json.loads(content["text"])


@pytest.fixture
async def library(client: AsyncTestClient) -> AsyncTestClient:
    """An initialized full-text app with one collection holding two indexed notes, the state an
    agent finds on a first call."""
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "notes"})
    for name, body in (("retries.md", RETRIES), ("ordering.md", ORDERING)):
        await stage_and_import(client, name, body.encode())
        await attach_via_api(client, "notes", name)
    return client


async def test_the_tools_an_agent_is_offered(client: AsyncTestClient) -> None:
    """Exactly the documented tools, each described and typed, with `q` a list of questions."""
    reply = await _rpc(client, "tools/list")

    tools = {one["name"]: one for one in reply["result"]["tools"]}
    assert set(tools) == TOOLS
    for name, one in tools.items():
        assert len(one["description"]) > 40, f"{name}: an agent chooses a tool by its description"
        assert one["inputSchema"]["type"] == "object", name
        for argument, schema in one["inputSchema"]["properties"].items():
            # litestar-mcp types an enum as a bare object, so its values ride in the description
            if "<enum" in json.dumps(schema):
                assert schema.get("description", "").startswith("One of: "), (name, argument)
    search = tools["search_excerpts"]["inputSchema"]
    assert search["properties"]["q"] == {"type": "array", "items": {"type": "string"}}
    assert search["required"] == ["q"]
    assert "context" in search["properties"]


@pytest.mark.parametrize(
    ("name", "tool", "arguments", "expected"),
    [
        (
            "the collections",
            "list_collections",
            {},
            lambda found: [one["name"] for one in found["items"]] == ["notes"],
        ),
        (
            "one collection, counted",
            "get_collection",
            {"collection": "notes"},
            lambda found: found["counts"]["indexed"] == 2,
        ),
        (
            "its indexed members",
            "list_collection_documents",
            {"collection": "notes", "status": "indexed"},
            lambda found: (
                {one["document"]["name"] for one in found["items"]} == {"retries.md", "ordering.md"}
            ),
        ),
        (
            "the documents, one page of one",
            "list_documents",
            {"page_size": 1, "sort": "name"},
            lambda found: (
                [one["name"] for one in found["items"]] == ["ordering.md"]
                and found["next_cursor"] is not None
                and found["total"] == 2
            ),
        ),
        (
            "one document",
            "get_document",
            {"document": "retries.md"},
            lambda found: found["status"] == "imported",
        ),
        (
            "a description",
            "describe_document",
            {"document": "retries.md", "description": "on retries", "session_id": SESSION},
            lambda found: found["description"] == "on retries",
        ),
        (
            "a session's collections",
            "set_session_collections",
            {"session_id": SESSION, "collections": ["notes"]},
            lambda found: found == ["notes"],
        ),
        (
            "one question",
            "search_excerpts",
            {"q": [BY_RETRY], "session_id": SESSION},
            lambda found: (
                [one["document"] for one in found["excerpts"]] == ["retries.md"]
                and found["excerpts"][0]["aspects"] == []
                and found["uncovered"] == []
            ),
        ),
        (
            "two questions, each tagged on its own note",
            "search_excerpts",
            {"q": [BY_RETRY, BY_CLOCK], "session_id": SESSION, "limit": 4},
            lambda found: (
                {one["document"]: one["aspects"] for one in found["excerpts"]}
                == {"retries.md": [BY_RETRY], "ordering.md": [BY_CLOCK]}
                and found["uncovered"] == []
            ),
        ),
        (
            "the sources, one section each",
            "search_sources",
            {"q": BY_RETRY, "sections": 1},
            lambda found: (
                [one["document"] for one in found["documents"]] == ["retries.md"]
                and found["collections"] == ["notes"]
                and len(found["documents"][0]["sections"]) == 1
            ),
        ),
    ],
)
async def test_every_read_tool_answers(
    library: AsyncTestClient, name: str, tool: str, arguments: dict, expected
) -> None:
    error, found = await _call(library, tool, arguments)

    assert not error, f"{name}: {found}"
    assert expected(found), f"{name}: {found}"


@pytest.mark.parametrize(
    ("name", "tool", "arguments", "message"),
    [
        (
            "an unknown collection",
            "get_collection",
            {"collection": "ghost"},
            "collection not found",
        ),
        ("an unknown document", "get_document", {"document": "ghost.md"}, "document not found"),
        (
            "a session over an unknown collection",
            "set_session_collections",
            {"session_id": SESSION, "collections": ["ghost"]},
            "collection not found: ghost",
        ),
        (
            "a question sent as a string, not a list",
            "search_excerpts",
            {"q": BY_RETRY},
            "Expected `array`, got `str`",
        ),
        ("no question", "search_excerpts", {}, "q"),
        (
            "fewer slots than questions",
            "search_excerpts",
            {"q": [BY_RETRY, BY_CLOCK], "limit": 1},
            "limit must be at least the number of questions (2), got 1",
        ),
        (
            "six questions",
            "search_excerpts",
            {"q": [f"{BY_RETRY} {n}" for n in range(6)]},
            "q must hold 1..5 questions, got 6",
        ),
        (
            "a context past 200 characters",
            "search_excerpts",
            {"q": [BY_RETRY], "context": "x" * 201},
            "context is at most 200 characters",
        ),
        (
            "an unknown collection to search",
            "search_sources",
            {"q": "x", "collections": "ghost"},
            "collection not found: ghost",
        ),
        (
            "a document that is not a member",
            "remove_document_from_collection",
            {"collection": "notes", "document": "ghost.md"},
            "ghost.md",
        ),
        (
            "attaching an unknown document",
            "add_document_to_collection",
            {"collection": "notes", "document": "ghost.md"},
            "document not found",
        ),
        ("a file that does not exist", "add_document", {"path": "/nowhere/at/all.md"}, "all.md"),
        (
            "a log reaching back past a year",
            "list_searches",
            {"days": 367},
            "days=367: Expected `int` <= 366",
        ),
        ("more searches than a page holds", "list_searches", {"limit": 201}, "limit"),
        ("a review nobody can decide", "review_gaps", {"ids": [1], "review": "maybe"}, "review"),
        ("too many gaps to replay at once", "replay_gaps", {"ids": list(range(51))}, "at most 50"),
    ],
)
async def test_every_mistake_an_agent_makes_is_a_tool_error(
    library: AsyncTestClient, name: str, tool: str, arguments: dict, message: str
) -> None:
    """A tool error the agent can read and act on, never a protocol failure or a crash."""
    error, found = await _call(library, tool, arguments)

    assert error, f"{name}: {found}"
    assert message in json.dumps(found), f"{name}: {found}"


async def test_an_agent_imports_attaches_finds_and_detaches_a_document(
    library: AsyncTestClient, tmp_path: Path
) -> None:
    """The write tools end to end, as an agent chains them, and the session keeps each step."""
    note = tmp_path / "keys.md"
    note.write_text("# Keys\n\nAn idempotency key lets a consumer drop a message it already saw.\n")

    error, added = await _call(
        library, "add_document", {"path": str(note), "description": "keys", "session_id": SESSION}
    )
    assert not error and added["status"] == "queued", added
    error, again = await _call(library, "add_document", {"path": str(note)})
    assert error and "keys.md" in json.dumps(again), "the name is taken now"
    assert (await wait_import(library, "keys.md"))["status"] == "imported"

    error, attached = await _call(
        library,
        "add_document_to_collection",
        {"collection": "notes", "document": "keys.md", "session_id": SESSION},
    )
    assert not error and attached["operation_id"], attached
    assert await wait_for(attached["operation_id"]) == "indexed"
    error, found = await _call(
        library, "search_excerpts", {"q": ["What lets a consumer drop a message it already saw?"]}
    )
    assert not error and found["excerpts"][0]["document"] == "keys.md", found

    error, removed = await _call(
        library,
        "remove_document_from_collection",
        {"collection": "notes", "document": "keys.md", "session_id": SESSION},
    )
    assert not error, removed
    error, after = await _call(
        library, "search_excerpts", {"q": ["What lets a consumer drop a message it already saw?"]}
    )
    assert not error and "keys.md" not in {one["document"] for one in after["excerpts"]}, after

    history = (await library.get(f"/api/sessions/{SESSION}/history")).json()
    assert [event["action"] for event in history] == ["detach", "attach", "import"]


async def test_an_agent_finds_a_gap_closes_it_and_resolves_it(
    library: AsyncTestClient, tmp_path: Path
) -> None:
    """The log and the gaps as an agent reads them: a question the notes cannot answer shows in
    the log and as a gap; once a note answers it, a replay says so, and the agent resolves it."""
    unanswered = "How long should sourdough starter ferment?"
    await _call(library, "search_excerpts", {"q": [BY_RETRY, unanswered], "session_id": SESSION})

    error, logged = await _call(library, "list_searches", {"session_id": SESSION})
    assert not error, logged
    (search,) = logged
    assert [(one["question"], one["uncovered"]) for one in search["questions"]] == [
        (BY_RETRY, False),
        (unanswered, True),
    ]
    assert [one["document"] for one in search["results"]] == ["retries.md"]
    assert "vector" not in search["questions"][0], "a query vector never reaches a caller"

    error, topics = await _call(library, "list_gaps", {})
    assert not error, topics
    (topic,) = topics
    (gap,) = topic["questions"]
    assert (gap["question"], gap["signal"], gap["session_id"]) == (unanswered, "uncovered", SESSION)

    note = tmp_path / "bread.md"
    note.write_text("# Bread\n\nLet a sourdough starter ferment for twelve hours before baking.\n")
    await stage_and_import(library, "bread.md", note.read_bytes())
    await attach_via_api(library, "notes", "bread.md")
    error, replayed = await _call(library, "replay_gaps", {"ids": [gap["id"]]})
    assert not error, replayed
    assert [(one["signal"], one["results"][0]["document"]) for one in replayed] == [
        (None, "bread.md")
    ]

    error, reported = await _call(
        library,
        "report_gap",
        {"session_id": SESSION, "question": BY_RETRY, "verdict": "partial", "missing": "backoff"},
    )
    assert not error and reported["verdict"] == "partial", reported
    error, both = await _call(library, "list_gaps", {})
    assert {one["questions"][0]["signal"] for one in both} == {"reported", "uncovered"}

    error, reviewed = await _call(
        library, "review_gaps", {"ids": [gap["id"], reported["id"]], "review": "resolved"}
    )
    assert (error, reviewed) == (False, 2)
    assert (await _call(library, "list_gaps", {}))[1] == []
    error, resolved = await _call(library, "list_gaps", {"review": "resolved"})
    assert sorted(one["question"] for one in resolved) == sorted([unanswered, BY_RETRY])


async def test_a_web_only_route_is_not_a_tool(library: AsyncTestClient) -> None:
    """The searches the web UI drives stay REST-only: calling one is a protocol error naming the
    tool, not a search served."""
    for name in ("explore", "search_text", "no_such_tool"):
        reply = await _rpc(
            library, "tools/call", {"name": name, "arguments": {"q": "x"}}, status=400
        )
        assert reply["error"]["message"] == f"Tool not found: {name}", reply


@pytest.mark.parametrize(
    ("name", "tool", "arguments", "expected"),
    [
        (
            "a collection tool binds the collection",
            "get_collection",
            {"collection": "notes"},
            {"collection": "notes"},
        ),
        (
            "a document tool binds the document alone",
            "get_document",
            {"document": "retries.md"},
            {"document": "retries.md"},
        ),
        ("a tool with neither binds neither", "list_collections", {}, {}),
    ],
)
async def test_a_tool_call_logs_the_names_it_is_routed_by(
    library: AsyncTestClient,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    tool: str,
    arguments: dict,
    expected: dict,
) -> None:
    """The app's request hook ran for the outer `/mcp` request only, which has no path
    parameters; the log lines a tool writes still carry the names its route is keyed by."""
    seen: list[dict[str, Any]] = []

    def seeing[**P, R](real: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        """`real`, noting the log context of the handler that called it."""

        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            seen.append(dict(structlog.contextvars.get_contextvars()))
            return await real(*args, **kwargs)

        return wrapped

    # what each of the three handlers reads first
    monkeypatch.setattr(Collection, "get", staticmethod(seeing(Collection.get)))
    monkeypatch.setattr(Collection, "page", staticmethod(seeing(Collection.page)))
    monkeypatch.setattr(document, "get", seeing(document.get))

    error, found = await _call(library, tool, arguments)

    assert not error, found
    (context,) = seen
    assert {k: context[k] for k in ("collection", "document") if k in context} == expected, name
