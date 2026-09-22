"""Request and response shapes shared by more than one feature module."""

from typing import Annotated

import msgspec
from litestar.di import Provide
from litestar.params import Parameter

from haskie.paging import page_request

# The four paging query arguments, declared once. Litestar reads a provider's own parameters from
# the query string, so a handler that asks for `page: PageRequest` takes `cursor`, `page_size`,
# `sort` and `order` on the wire, in the OpenAPI document and in the MCP tool schema. Per route
# rather than on the app: every handler advertises every dependency in scope to MCP.
PAGED = {"page": Provide(page_request, sync_to_thread=False)}


class BulkStarted(msgspec.Struct):
    """An operation was accepted and runs in the background; follow it at
    /api/operations/{operation_id}/progress."""

    operation_id: str


class Describe(msgspec.Struct):
    """What a collection or a document is said to hold; empty clears it."""

    description: str


# A search returns at least one result or none at all; the bound rides along into the schema.
Limit = Annotated[int | None, Parameter(ge=1)]
