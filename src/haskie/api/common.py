"""Request and response shapes shared by more than one feature module."""

import time
from typing import Annotated

import msgspec
from litestar.di import Provide
from litestar.params import Parameter

from haskie import ids
from haskie.collection.index import Scope
from haskie.errors import InvalidInput
from haskie.paging import page_request
from haskie.search import session
from haskie.settings import MAX_SCAN

# The four paging query arguments, declared once. Litestar reads a provider's own parameters from
# the query string, so a handler that asks for `page: PageRequest` takes `cursor`, `page_size`,
# `sort` and `order` on the wire, in the OpenAPI document and in the MCP tool schema. Per route
# rather than on the app: every handler advertises every dependency in scope to MCP.
PAGED = {"page": Provide(page_request, sync_to_thread=False)}


class BulkStarted(msgspec.Struct):
    """An operation was accepted and runs in the background. A whole-collection index or delete,
    a document delete and a description asked for of either report their progress at
    /api/operations/{operation_id}/progress; every operation is listed at /api/operations."""

    operation_id: str


class Rename(msgspec.Struct):
    name: str


class Describe(msgspec.Struct):
    """What a collection or a document is said to hold; empty clears it."""

    description: str


# A search returns at least one result or none at all, and no more than it scans: an explicit
# limit sets the scan depth, so above `MAX_SCAN` it would make every collection return, and the
# reranker score, that many chunks. The bounds also appear in the schema.
Limit = Annotated[int | None, Parameter(ge=1, le=MAX_SCAN)]

# The documents and sections a search keeps to, by id (`ids`: `Document.id`, the section ids of
# `sections.build`). Bounded, so one call cannot make a filter of any size.
MAX_IDS = 100
Ids = Annotated[list[str] | None, Parameter(max_items=MAX_IDS)]


def scope_of(document_ids: list[str] | None, section_ids: list[str] | None = None) -> Scope:
    """The scope a search keeps to, refused before it runs when an id is not one: a base58 MD5, as
    every document and section id is (`ids.ID`)."""
    for one in [*(document_ids or ()), *(section_ids or ())]:
        if not ids.ID.fullmatch(one):
            raise InvalidInput(f"not a document or section id: {one!r}")
    return Scope(frozenset(document_ids or ()), frozenset(section_ids or ()))


MAX_DAYS = 366  # a window of history: Insights charts and Gaps read at most a year back
Days = Annotated[int, Parameter(ge=1, le=MAX_DAYS)]

# A conversation's id, refused before the handler runs: refused inside it, after the change the
# id would record, the change stands while the caller reads a 422, and its retry meets a
# conflict. `SessionId | None` would drop the bounds, so the optional form is its own alias.
_SESSION_ID = Parameter(min_length=1, max_length=session.MAX_SESSION_ID)
SessionId = Annotated[str | None, _SESSION_ID]
RequiredSessionId = Annotated[str, _SESSION_ID]


def days_ago(days: int) -> float:
    """Unix seconds `days` days ago: where a window of history begins."""
    return time.time() - days * 86400
