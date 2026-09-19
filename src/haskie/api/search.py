"""Session selection and the two searches that are not scoped to one collection."""

import msgspec
from litestar import get, put

from haskie import audit, session, textsearch
from haskie.api.common import Limit
from haskie.index import Hit
from haskie.paging import DEFAULT_PAGE_SIZE, Page


class SessionCollections(msgspec.Struct):
    collections: list[str]


@get("/api/sessions")
async def list_sessions() -> dict[str, list[str]]:
    return await session.load()


@put("/api/sessions/{session_id:str}", mcp_tool="set_session_collections")
@audit.audited("session.collections.set")
async def put_session(session_id: str, data: SessionCollections) -> list[str]:
    """Choose which collections a session searches. Call before `search` with that session id."""
    chosen = await session.set_collections(session_id, data.collections)
    audit.attach(collections=len(chosen))
    return chosen


@get("/api/search", mcp_tool="search")
async def search_session(session_id: str, q: str, limit: Limit = None) -> list[Hit]:
    """Search the collections selected for `session_id`; `limit` defaults to the user setting.

    A passage held by several of those collections is returned once, not once per collection.
    """
    return await session.search(session_id, q, limit)


@get("/api/search/text", mcp_tool="search_text")
async def search_text(
    q: str,
    collections: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> Page[Hit]:
    """Full-text (BM25) search across all collections, or the comma-separated `collections`.

    No embedding model and no session needed. A passage held by several collections is returned
    once. Pass the `next_cursor` of a response back as `cursor` for the next page; it is null on
    the last page. Every page recomputes the ranking, so a document indexed between two pages can
    move a result across a page boundary.
    """
    return await textsearch.search(q, textsearch.split_collections(collections), page_size, cursor)


@get("/api/search/documents", mcp_tool="search_documents")
async def search_documents(
    q: str, collections: str | None = None, limit: int = 10
) -> list[textsearch.DocumentMatch]:
    """Which documents to read for a query, rather than which passages answer it.

    The same full-text scan as `search_text`, folded to one row per document: `score` is the
    document's best chunk and `chunks` is how many of the scanned chunks came from it, so a
    document that matches throughout outranks one that matches once as well. Use it to narrow to
    a shortlist, then `search_text` or `search` for the passages themselves.
    """
    return await textsearch.search_documents(q, textsearch.split_collections(collections), limit)
