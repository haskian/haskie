"""haskie as the unprompted arms (G, H, I) see it: one collection, as if the instance held nothing
else.

Told no collection, an agent searches all of them, and the eval instance holds every task's
corpus side by side - so the "absent" service is present in a sibling collection, and a stale
runbook sits next to its change notice's. A user's library holds their own documents, not the
eval's other tasks. This proxy stands between the agent and haskie's MCP endpoint and forwards
every request unchanged but these: a search is scoped to the task's collection, a session's
selection is that collection, and `list_collections` lists only it. Other tools pass as they
are: they return names and metadata, never a document's text.

Only the arguments change, never a request's method or tool name, so the `Mcp-Method` and
`Mcp-Name` headers haskie checks against the body still match.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

SEARCHES = {"search_excerpts", "search_sources"}
HOP = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}


def scoped_request(rpc: Any, collection: str) -> Any:
    """A JSON-RPC request with every search and session selection held to `collection`."""
    if not isinstance(rpc, dict) or rpc.get("method") != "tools/call":
        return rpc
    params = rpc.setdefault("params", {})
    arguments = params.get("arguments") or {}
    if params.get("name") in SEARCHES:
        arguments["collections"] = collection
    elif params.get("name") == "set_session_collections":
        arguments["collections"] = [collection]
    params["arguments"] = arguments
    return rpc


def scoped_response(rpc: Any, reply: Any, collection: str) -> Any:
    """A `list_collections` reply listing `collection` alone; any other reply as it is."""
    if not (
        isinstance(rpc, dict)
        and rpc.get("method") == "tools/call"
        and rpc.get("params", {}).get("name") == "list_collections"
        and isinstance(reply, dict)
    ):
        return reply
    for block in reply.get("result", {}).get("content", []):
        if block.get("type") != "text":
            continue
        try:
            page = json.loads(block["text"])
        except (KeyError, json.JSONDecodeError):
            continue
        page["items"] = [item for item in page.get("items", []) if item.get("name") == collection]
        page["total"], page["next_cursor"] = len(page["items"]), None
        block["text"] = json.dumps(page)
    return reply


class Proxy:
    """A local MCP endpoint forwarding to `upstream`'s, scoped to `collection`. Use as a context
    manager; `url` is what the agent's MCP config points at."""

    def __init__(self, upstream: str, collection: str) -> None:
        self.upstream = upstream.rstrip("/")
        self.collection = collection
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/mcp"

    def __enter__(self) -> Proxy:
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self._forward()

            def do_GET(self) -> None:
                self._forward()

            def do_DELETE(self) -> None:
                self._forward()

            def _forward(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length) if length else None
                rpc = None
                if body:
                    try:
                        rpc = scoped_request(json.loads(body), proxy.collection)
                        body = json.dumps(rpc).encode()
                    except json.JSONDecodeError:
                        pass
                headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
                request = urllib.request.Request(
                    f"{proxy.upstream}{self.path}", data=body, headers=headers, method=self.command
                )
                try:
                    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
                        status, reply_headers, reply = (
                            response.status,
                            response.headers,
                            response.read(),
                        )
                except urllib.error.HTTPError as error:
                    status, reply_headers, reply = error.code, error.headers, error.read()
                if rpc is not None and "json" in (reply_headers.get("content-type") or ""):
                    try:
                        scoped = scoped_response(rpc, json.loads(reply), proxy.collection)
                        reply = json.dumps(scoped).encode()
                    except json.JSONDecodeError:
                        pass
                self.send_response(status)
                for key, value in reply_headers.items():
                    if key.lower() not in HOP:
                        self.send_header(key, value)
                self.send_header("content-length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, format: str, *args: object) -> None:
                pass

        return Handler
