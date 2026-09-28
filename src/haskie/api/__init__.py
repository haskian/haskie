"""The HTTP/MCP surface, one module per feature.

`app.py` owns the application: the request context, the error mapping and `create_app`. Each
module here owns one feature's routes and the request/response shapes only that feature uses;
what two of them share lives in `common.py`.
"""

from litestar.handlers import HTTPRouteHandler
from litestar.types import ControllerRouterHandler

from haskie.api import collections, documents, gaps, operations, search, settings

# The module order is the order the routes appear in the OpenAPI document, so it follows the UI:
# set up, then documents, then the collections holding them, then the work, then searching, then
# what searching found missing.
# Inside a module it is definition order, which is what `vars()` yields.
ROUTE_HANDLERS: list[ControllerRouterHandler] = [
    handler
    for module in (settings, documents, collections, operations, search, gaps)
    for handler in vars(module).values()
    if isinstance(handler, HTTPRouteHandler)
]
