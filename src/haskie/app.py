"""Litestar app: the REST API for the web UI, the same handlers exposed as MCP tools at /mcp.

The routes live in `haskie.api`, one module per feature. What stays here is what is true of
every request: the request context, the error mapping, and how the application is assembled.
"""

import os
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit
from uuid import uuid4

import msgspec
from litestar import Litestar, MediaType, Request, Response
from litestar.datastructures import Headers, MutableScopeHeaders
from litestar.exceptions import HTTPException, ValidationException
from litestar.exceptions.responses import create_exception_response
from litestar.handlers import HTTPRouteHandler
from litestar.openapi import OpenAPIConfig
from litestar.openapi.spec import OpenAPIMediaType, OpenAPIResponse, OpenAPIType, Operation, Schema
from litestar.static_files import create_static_files_router
from litestar.types import (
    ASGIApp,
    ControllerRouterHandler,
    ExceptionHandlersMap,
    Message,
    Receive,
    Scope,
    Send,
)
from litestar_mcp import LitestarMCP, MCPConfig

from haskie import APP_VERSION, home, logs, shutdown
from haskie.api import ROUTE_HANDLERS
from haskie.audit import Actor
from haskie.document.document import UPLOAD_MAX_BYTES
from haskie.errors import Forbidden, HaskieError
from haskie.indexing import workflows
from haskie.search import flow

# The built UI, wherever it is: inside the package when haskie was installed, or `web/dist` in a
# checkout. Absent in both places means API and MCP only.
_PACKAGED_WEB = Path(__file__).resolve().parent / "web"
_CHECKOUT_WEB = Path(__file__).resolve().parents[2] / "web" / "dist"
WEB_DIST = _PACKAGED_WEB if _PACKAGED_WEB.is_dir() else _CHECKOUT_WEB
REQUEST_ID_KEY = "request_id"
REQUEST_ID_HEADER = "X-Request-Id"
TRACE_KEY = "search_trace"
SCORING_HEADER = "X-Score-Lineage"
MCP_PATH = "/mcp"

_log = logs.get_logger(__name__)


# --- request context --------------------------------------------------------


async def bind_request_context(request: Request) -> None:
    """One id per request, on every log line and audit record it produces, and in the response
    header so a user can quote it. Runs before body validation, so a 422 is traceable too.

    Async on purpose: Litestar runs a sync hook in a worker thread, whose context vars are
    dropped when it returns, while an async one binds in the task the handler inherits from.
    """
    request_id = uuid4().hex
    request.scope["state"][REQUEST_ID_KEY] = request_id
    request.scope["state"][TRACE_KEY] = flow.start_trace()
    logs.clear()
    path = request.scope["path"]
    # `collection` and `document` are what every collection and document route is keyed by, so the
    # request context carries them for free instead of each handler binding them again. A document
    # route has no collection at all: the document belongs to none.
    routed = request.path_params
    scoped: dict[str, str] = {
        key: routed[key] for key in ("collection", "document") if routed.get(key)
    }
    logs.bind(
        request_id=request_id,
        actor=Actor.MCP if path.startswith(MCP_PATH) else Actor.WEB,
        method=request.method,
        path=path,
        **scoped,
    )


async def add_request_id(message: Message, scope: Scope) -> None:
    """Runs for every response, including the ones Litestar itself produces."""
    if message["type"] == "http.response.start":
        request_id = scope["state"].get(REQUEST_ID_KEY)
        if request_id is not None:
            MutableScopeHeaders(message)[REQUEST_ID_HEADER] = request_id
        # a search's step timings and how it scored (`search.flow`), which the web UI shows
        trace = scope["state"].get(TRACE_KEY)
        if trace is not None and trace.steps:
            MutableScopeHeaders(message)["Server-Timing"] = flow.server_timing(trace.steps)
        if trace is not None and trace.scoring:
            # JSON, percent-encoded: a header is Latin-1, and the formulas are not
            lineage = msgspec.json.encode(trace.scoring).decode()
            MutableScopeHeaders(message)[SCORING_HEADER] = quote(lineage)


# --- who may call -----------------------------------------------------------

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})  # a bind to every interface: exposed on purpose
READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
ALLOWED_ORIGINS_ENV = "HASKIE_ALLOWED_ORIGINS"


def allowed_origins() -> frozenset[str]:
    """Browser origins trusted beside haskie's own, for an agent UI that runs in a browser:
    `HASKIE_ALLOWED_ORIGINS`, comma-separated, e.g. `https://agent.example`."""
    listed = os.environ.get(ALLOWED_ORIGINS_ENV, "").split(",")
    return frozenset(origin.strip().rstrip("/").lower() for origin in listed if origin.strip())


def served_hosts() -> frozenset[str] | None:
    """The host names a request may address: loopback and the address `run` bound, or None for a
    wildcard bind, where any name can reach the server and none can be told apart."""
    bound = urlsplit(os.environ.get("HASKIE_ADDRESS", "")).hostname
    if bound in WILDCARD_HOSTS:
        return None
    return LOOPBACK_HOSTS | ({bound} if bound else set())


def _refusal(scope: Scope, hosts: frozenset[str] | None, origins: frozenset[str]) -> str | None:
    """Why a request is refused, or None. Agents, MCP clients and scripts send no `Origin`, so
    they pass. A browser does, and a browser ignores the loopback bind: without these checks any
    page the user opens could post to the API, and a DNS-rebinding page could read it too."""
    headers = Headers.from_scope(scope)
    host = headers.get("host", "")
    try:
        hostname = urlsplit(f"//{host}").hostname
    except ValueError:
        hostname = None
    if hosts is not None and hostname not in hosts:
        return f"host {host!r} is not one haskie serves"
    origin = headers.get("origin")
    if origin is None or scope["method"] in READ_METHODS:
        return None
    if origin.lower() in origins | {f"{scope['scheme']}://{host}".lower()}:
        return None
    return f"requests from {origin} are not accepted; list it in {ALLOWED_ORIGINS_ENV} to trust it"


def guard_callers(app: ASGIApp) -> ASGIApp:
    """Middleware refusing a request from a host or browser origin haskie does not serve."""
    hosts, origins = served_hosts(), allowed_origins()

    async def guarded(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and (reason := _refusal(scope, hosts, origins)):
            raise Forbidden(reason)
        await app(scope, receive, send)

    return guarded


# --- errors -----------------------------------------------------------------


def _client_error(
    exc: Exception,
    status_code: int,
    headers: dict[str, str] | None = None,
    message: str | None = None,
) -> Response:
    detail = home.scrub(message if message is not None else str(exc))
    _log.warning(
        "request_rejected", status_code=status_code, error=type(exc).__name__, detail=detail
    )
    return Response({"detail": detail}, status_code=status_code, headers=headers)


def haskie_error(_: Request, exc: HaskieError) -> Response:
    """Every expected failure, at the status code and headers its class declares."""
    return _client_error(exc, exc.status_code, exc.headers)


def validation_error(_: Request, exc: ValidationException) -> Response:
    """msgspec turns the `InvalidInput` a settings struct raises while decoding into Litestar's
    own 400; a rejected body is invalid input, so it keeps our 422. `extra` carries the message
    of the field that failed, which the bare exception text drops."""
    reasons = [str(item.get("message", item)) for item in exc.extra or [] if isinstance(item, dict)]
    return _client_error(
        exc, 422, message=": ".join([exc.detail, *reasons]) if reasons else exc.detail
    )


def internal_error(request: Request, exc: Exception) -> Response:
    """Registered for `Exception`, which Litestar's MRO lookup also matches for its own
    HTTPExceptions (405, 413, unknown route): those keep their status and default body."""
    if isinstance(exc, HTTPException):
        return create_exception_response(request, exc)
    _log.exception("unhandled_error", error=type(exc).__name__)
    # local single-user app: the UI shows the real message instead of a bare 500
    return Response({"detail": home.scrub(f"{type(exc).__name__}: {exc}")}, status_code=500)


EXCEPTION_HANDLERS: ExceptionHandlersMap = {
    HaskieError: haskie_error,
    ValidationException: validation_error,
    Exception: internal_error,
}

# What `validation_error` and an `InvalidInput` answer, for the OpenAPI document. Inline rather than
# a component: Litestar replaces the configured component schemas with the ones it generates.
REJECTED = OpenAPIResponse(
    description="The request is invalid: a parameter or body that does not decode, or a value "
    "the handler refuses.",
    content={
        MediaType.JSON: OpenAPIMediaType(
            schema=Schema(
                type=OpenAPIType.OBJECT,
                required=["detail"],
                properties={"detail": Schema(type=OpenAPIType.STRING)},
            )
        )
    },
)


@dataclass
class RejectingOperation(Operation):
    """An operation as haskie answers it. Litestar documents every route that validates its input
    with its own 400 `{status_code, detail, extra}` body, and offers no app-wide way to change
    that; `validation_error` answers 422 `{detail}`, so that is what the document says instead."""

    def __post_init__(self) -> None:
        # the handlers declare no `raises`, so a 400 here is only ever Litestar's validation one
        if self.responses is not None and self.responses.pop("400", None) is not None:
            self.responses["422"] = REJECTED


def documented(handler: HTTPRouteHandler) -> HTTPRouteHandler:
    """`handler` with the operation class that documents its rejections truthfully. A copy,
    because the handler objects are module globals and Litestar copies what it registers anyway."""
    rejecting = copy(handler)
    rejecting.operation_class = RejectingOperation
    return rejecting


# --- app --------------------------------------------------------------------


async def stop_runtime() -> None:
    """Stop the pipeline, then give the home up. A hurried stop keeps the home: DBOS is still
    running workflows until the process exits, and a second haskie claiming the home meanwhile
    would run them too. The kernel drops the lock at the exit, which `shutdown.EXIT_GRACE`
    bounds."""
    if await workflows.stop():
        home.release_home()


def create_app() -> Litestar:
    """Factory so logging is configured before Litestar builds anything; `app` below keeps
    `litestar --app haskie.app:app` working."""
    logs.configure()
    route_handlers: list[ControllerRouterHandler] = [
        documented(handler) for handler in ROUTE_HANDLERS
    ]
    if WEB_DIST.is_dir():
        route_handlers.append(
            create_static_files_router("/", directories=[WEB_DIST], html_mode=True)
        )
    else:
        # The UI is a build product, not source: a wheel built without `mise run build`, or a
        # checkout that never ran it, serves the API and MCP and nothing at `/`. Say so, or the
        # only symptom is a 404 on the root with no explanation.
        _log.warning("web_ui_missing", expected=str(WEB_DIST), serving="api and mcp only")
    return Litestar(
        route_handlers=route_handlers,
        plugins=[LitestarMCP(MCPConfig(allowed_origins=sorted(allowed_origins())))],
        middleware=[guard_callers],
        openapi_config=OpenAPIConfig(title="haskie", version=APP_VERSION),
        logging_config=None,  # `logs.configure` above owns it (see logs.py)
        exception_handlers=EXCEPTION_HANDLERS,
        before_request=bind_request_context,
        before_send=[add_request_id],
        request_max_body_size=UPLOAD_MAX_BYTES,
        # `claim_home` first: everything after it migrates the database or launches DBOS, and a
        # second haskie on the same home must refuse before any of that, not after.
        on_startup=[
            home.claim_home,
            shutdown.bound_exit,
            shutdown.debounce_signals,
            home.ensure_home,
            workflows.start,
        ],
        on_shutdown=[stop_runtime],
    )


app = create_app()
