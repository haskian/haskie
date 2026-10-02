"""`scope.py`: the MCP proxy that shows the unprompted arms one collection. A fake upstream that
records what it receives stands in for haskie, so these assert what was forwarded."""

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from evals import run, scope

COLLECTION = "eval-synth-s1-absent-n500"


def _call(name: str, arguments: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments, "_meta": {"m": 1}},
    }


class Upstream(BaseHTTPRequestHandler):
    received: list[tuple[dict, dict]] = []  # (headers, body)
    status = 200

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        Upstream.received.append((dict(self.headers.items()), body))
        page = {
            "items": [{"name": "eval-synth-s1-n500"}, {"name": COLLECTION}],
            "total": 2,
            "next_cursor": "abc",
        }
        content = [{"type": "text", "text": json.dumps(page)}]
        reply = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {"content": content}})
        reply = reply.encode()
        self.send_response(Upstream.status)
        self.send_header("content-type", "application/json")
        self.send_header("mcp-protocol-version", "2026-07-28")
        self.send_header("content-length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def proxy() -> Iterator[scope.Proxy]:
    Upstream.received, Upstream.status = [], 200
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with scope.Proxy(f"http://127.0.0.1:{server.server_address[1]}", COLLECTION) as running:
        yield running
    server.shutdown()


def _post(url: str, rpc: dict) -> tuple[int, dict, dict]:
    name = rpc["params"]["name"]
    request = urllib.request.Request(
        url,
        data=json.dumps(rpc).encode(),
        method="POST",
        headers={"content-type": "application/json", "mcp-method": "tools/call", "mcp-name": name},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, dict(response.headers.items()), json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers.items()), json.loads(error.read())


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("search_excerpts", {"q": ["x"]}, COLLECTION),
        ("search_sections", {"q": "x", "collections": "eval-synth-s1-n500"}, COLLECTION),
        ("set_session_collections", {"collections": ["a", "b"]}, [COLLECTION]),
    ],
)
def test_a_search_or_selection_is_held_to_the_tasks_collection(
    proxy: scope.Proxy, name: str, arguments: dict, expected: object
) -> None:
    _post(proxy.url, _call(name, arguments))

    ((headers, body),) = Upstream.received
    assert body["params"]["arguments"]["collections"] == expected
    assert body["params"]["_meta"] == {"m": 1}, "the rest of the request passes as it is"
    assert (headers["Mcp-Method"], headers["Mcp-Name"]) == ("tools/call", name)


def test_list_collections_lists_the_tasks_collection_alone(proxy: scope.Proxy) -> None:
    status, headers, reply = _post(proxy.url, _call("list_collections", {}))

    page = json.loads(reply["result"]["content"][0]["text"])
    assert status == 200 and headers["mcp-protocol-version"] == "2026-07-28"
    assert (page["items"], page["total"], page["next_cursor"]) == ([{"name": COLLECTION}], 1, None)


def test_another_tool_passes_untouched(proxy: scope.Proxy) -> None:
    _, _, reply = _post(proxy.url, _call("get_document", {"document": "doc-1.md"}))

    ((_, body),) = Upstream.received
    assert body["params"]["arguments"] == {"document": "doc-1.md"}
    assert json.loads(reply["result"]["content"][0]["text"])["total"] == 2, (
        "reply as haskie sent it"
    )


def test_an_error_status_is_forwarded(proxy: scope.Proxy) -> None:
    Upstream.status = 400

    status, _, _ = _post(proxy.url, _call("search_sections", {"q": "x"}))

    assert status == 400


def test_only_the_unprompted_arms_go_through_the_proxy() -> None:
    assert run.UNMENTIONED_ARMS == ("g", "h", "i")
