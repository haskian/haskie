"""Litestar app: the REST API for the web UI, the same handlers exposed as MCP tools at /mcp.

The routes live in `haskie.api`, one module per feature. What stays here is what is true of
every request: the request context, the error mapping, and how the application is assembled.
"""

from pathlib import Path
from uuid import uuid4

from litestar import Litestar, Request, Response
from litestar.datastructures import MutableScopeHeaders
from litestar.exceptions import HTTPException, ValidationException
from litestar.exceptions.responses import create_exception_response
from litestar.openapi import OpenAPIConfig
from litestar.static_files import create_static_files_router
from litestar.types import ControllerRouterHandler, ExceptionHandlersMap, Message, Scope
from litestar_mcp import LitestarMCP

from haskie import APP_VERSION, home, logs
from haskie.api import ROUTE_HANDLERS
from haskie.audit import Actor
from haskie.document.document import UPLOAD_MAX_BYTES
from haskie.errors import HaskieError
from haskie.indexing import workflows

# The built UI, wherever it is: inside the package when haskie was installed, or `web/dist` in a
# checkout. Absent in both places means API and MCP only.
_PACKAGED_WEB = Path(__file__).resolve().parent / "web"
_CHECKOUT_WEB = Path(__file__).resolve().parents[2] / "web" / "dist"
WEB_DIST = _PACKAGED_WEB if _PACKAGED_WEB.is_dir() else _CHECKOUT_WEB
REQUEST_ID_KEY = "request_id"
REQUEST_ID_HEADER = "X-Request-Id"
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


# --- app --------------------------------------------------------------------


def create_app() -> Litestar:
    """Factory so logging is configured before Litestar builds anything; `app` below keeps
    `litestar --app haskie.app:app` working."""
    logs.configure()
    route_handlers: list[ControllerRouterHandler] = list(ROUTE_HANDLERS)
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
        plugins=[LitestarMCP()],
        openapi_config=OpenAPIConfig(title="haskie", version=APP_VERSION),
        logging_config=None,  # `logs.configure` above owns it (see logs.py)
        exception_handlers=EXCEPTION_HANDLERS,
        before_request=bind_request_context,
        before_send=[add_request_id],
        request_max_body_size=UPLOAD_MAX_BYTES,
        # `claim_home` first: everything after it migrates the database or launches DBOS, and a
        # second haskie on the same home must refuse before any of that, not after.
        on_startup=[home.claim_home, home.ensure_home, workflows.start],
        on_shutdown=[workflows.stop, home.release_home],
    )


app = create_app()
