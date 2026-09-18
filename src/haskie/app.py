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

from haskie import APP_VERSION, errors, home, logs, workflows
from haskie.api import ROUTE_HANDLERS
from haskie.errors import HaskieError, NotReady
from haskie.library import UPLOAD_MAX_BYTES

# The built UI, wherever it is: inside the wheel when haskie was installed (`uv tool install`),
# or `web/dist` when it is run from a checkout. Absent in both cases means API and MCP only.
_PACKAGED_WEB = Path(__file__).resolve().parent / "web"
_CHECKOUT_WEB = Path(__file__).resolve().parents[2] / "web" / "dist"
WEB_DIST = _PACKAGED_WEB if _PACKAGED_WEB.is_dir() else _CHECKOUT_WEB
REQUEST_ID_KEY = "request_id"
REQUEST_ID_HEADER = "X-Request-Id"
MCP_PATH = "/mcp"
RETRY_AFTER_SECONDS = "2"

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
    # `name` and `doc` are what every library and document route is keyed by, so the request
    # context carries them for free instead of each handler binding them again.
    routed = request.path_params
    logs.bind(
        request_id=request_id,
        actor="mcp" if request.scope["path"].startswith(MCP_PATH) else "web",
        method=request.method,
        path=request.scope["path"],
        **{
            field: routed[param]
            for field, param in (("library", "name"), ("doc", "doc"))
            if routed.get(param)
        },
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
    detail = errors.scrub(message if message is not None else str(exc))
    _log.warning(
        "request_rejected", status_code=status_code, error=type(exc).__name__, detail=detail
    )
    return Response({"detail": detail}, status_code=status_code, headers=headers)


def haskie_error(_: Request, exc: HaskieError) -> Response:
    """Every expected failure, at the status code its class declares."""
    retry_after = {"Retry-After": RETRY_AFTER_SECONDS} if isinstance(exc, NotReady) else None
    return _client_error(exc, exc.status_code, retry_after)


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
    return Response({"detail": errors.scrub(f"{type(exc).__name__}: {exc}")}, status_code=500)


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
        logging_config=logs.logging_config,
        exception_handlers=EXCEPTION_HANDLERS,
        before_request=bind_request_context,
        before_send=[add_request_id],
        request_max_body_size=UPLOAD_MAX_BYTES,
        on_startup=[home.ensure_home, workflows.start],
        on_shutdown=[workflows.stop],
    )


app = create_app()
