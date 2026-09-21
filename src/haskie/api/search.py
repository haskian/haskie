"""Session selection and the two searches that are not scoped to one collection."""

import time

import msgspec
from litestar import get, put

from haskie import audit, jobs, session, textsearch
from haskie.api.common import Limit
from haskie.errors import InvalidInput
from haskie.index import Hit
from haskie.paging import DEFAULT_PAGE_SIZE, Page


class SessionCollections(msgspec.Struct):
    collections: list[str]


@get("/api/sessions")
async def list_sessions() -> list[session.SessionSummary]:
    """Every session: its collections and when it last did anything."""
    return await session.summaries()


@put("/api/sessions/{session_id:str}", mcp_tool="set_session_collections")
@audit.audited("session.collections.set")
async def put_session(session_id: str, data: SessionCollections) -> list[str]:
    """Choose which collections a session searches. Call before `search` with that session id."""
    chosen = await session.set_collections(session_id, data.collections)
    audit.attach(collections=len(chosen))
    await session.record(
        session_id, "collections", ", ".join(chosen), detail={"collections": chosen}
    )
    return chosen


@get("/api/search", mcp_tool="search")
async def search_session(session_id: str, q: str, limit: Limit = None) -> list[Hit]:
    """Search the collections selected for `session_id`; `limit` defaults to the user setting.

    A passage held by several of those collections is returned once, not once per collection.
    """
    started = time.perf_counter()
    found = await session.search(session_id, q, limit)
    await session.record_search(session_id, "session", q, found, started)
    return found


MAX_TREND_DAYS = 366


def _trend_cutoff(days: int) -> float:
    """Unix seconds `days` days ago: where an Insights chart begins."""
    if not 1 <= days <= MAX_TREND_DAYS:
        raise InvalidInput(f"days must be 1..{MAX_TREND_DAYS}, got {days}")
    return time.time() - days * 86400


@get("/api/insights/searches")
async def search_trend(days: int = 7) -> list[session.SearchAt]:
    """Every search of the last `days` days, oldest first, for the Insights chart."""
    return await session.searches_since(_trend_cutoff(days))


@get("/api/insights/chunks")
async def chunk_trend(days: int = 7) -> list[jobs.ChunksAt]:
    """Every finished index of the last `days` days, oldest first, for the Insights chart."""
    return await jobs.chunks_since(_trend_cutoff(days))


@get("/api/sessions/{session_id:str}/history")
async def session_history(session_id: str) -> list[session.SessionEvent]:
    """What a session did, newest first; the last 100 events at most."""
    return await session.history(session_id)


@get("/api/search/text", mcp_tool="search_text")
async def search_text(
    q: str,
    collections: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
    session_id: str | None = None,
) -> Page[Hit]:
    """Full-text (BM25) search across all collections, or the comma-separated `collections`.

    No embedding model and no session needed. A passage held by several collections is returned
    once. Pass the `next_cursor` of a response back as `cursor` for the next page; it is null on
    the last page. Every page recomputes the ranking, so a document indexed between two pages can
    move a result across a page boundary.

    Args:
        session_id: The conversation's id; the search then shows in that session's history.
    """
    started = time.perf_counter()
    page = await textsearch.search(q, textsearch.split_collections(collections), page_size, cursor)
    if cursor is None:  # one event per search, not one per page of it
        await session.record_search(session_id, "text", q, page.items, started)
    return page


@get("/api/search/documents", mcp_tool="search_documents")
async def search_documents(
    q: str, collections: str | None = None, limit: int = 10, session_id: str | None = None
) -> list[textsearch.DocumentMatch]:
    """Which documents to read for a query, rather than which passages answer it.

    The same full-text scan as `search_text`, folded to one row per document: `score` blends the
    document's best chunk with the sum of every chunk that matched (a harmonic mean, so many weak
    chunks never outrank one strong one) and `chunks` says how many there were. Use it to narrow
    to a shortlist, then `search_text` or `search` for the passages themselves.

    Args:
        session_id: The conversation's id; the search then shows in that session's history.
    """
    started = time.perf_counter()
    found = await textsearch.search_documents(q, textsearch.split_collections(collections), limit)
    await session.record_search(session_id, "documents", q, found, started)
    return found


@get("/api/search/documents/{doc:str}", mcp_tool="document_passages")
async def document_passages(
    doc: str, q: str, collections: str | None = None, limit: int = 10, session_id: str | None = None
) -> list[Hit]:
    """The passages of one document that `search_documents` counted for it, best first.

    The same scan as `search_documents` with the same `limit`, kept to `doc`. Use it to read why a
    shortlisted document is there before opening the whole thing.

    Args:
        session_id: The conversation's id; the search then shows in that session's history.
    """
    started = time.perf_counter()
    found = await textsearch.document_passages(
        q, doc, textsearch.split_collections(collections), limit
    )
    await session.record_search(session_id, "passages", q, found, started)
    return found
