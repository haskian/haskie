"""Session selection and the searches that are not scoped to one collection."""

import time
from enum import StrEnum

import msgspec
from litestar import get, put

from haskie import audit
from haskie.api.common import Limit
from haskie.collection.index import Hit
from haskie.errors import InvalidInput
from haskie.indexing import operations
from haskie.paging import DEFAULT_PAGE_SIZE, Page
from haskie.search import flow, retrieval, session, text
from haskie.search.passage import Excerpt, Passage, Sources


# What one search returns: the chunks the index holds, the passages they merge into, or the
# excerpt of a passage an agent quotes.
class Granularity(StrEnum):
    CHUNK = "chunk"
    PASSAGE = "passage"
    EXCERPT = "excerpt"


class SessionCollections(msgspec.Struct):
    collections: list[str]


@get("/api/sessions")
async def list_sessions() -> list[session.SessionSummary]:
    """Every session: its collections and when it last did anything."""
    return await session.summaries()


@put("/api/sessions/{session_id:str}", mcp_tool="set_session_collections")
@audit.audited("session.collections.set")
async def put_session(session_id: str, data: SessionCollections) -> list[str]:
    """Choose which collections a session searches. Call before `search_excerpts` with that
    session id."""
    chosen = await session.set_collections(session_id, data.collections)
    audit.attach(collections=len(chosen))
    await session.record(
        session_id,
        session.Action.COLLECTIONS,
        ", ".join(chosen),
        detail=session.EventDetail(collections=chosen),
    )
    return chosen


@get("/api/search/explore")
async def explore(
    q: str,
    granularity: Granularity = Granularity.CHUNK,
    session_id: str | None = None,
    collections: str | None = None,
    limit: Limit = None,
) -> list[Hit] | list[Passage] | list[Excerpt]:
    """Search at the granularity the caller wants: the exploration endpoint the UI drives.

    Where it looks: the comma-separated `collections` if given, else the collections selected for
    `session_id`, else every collection. `limit` defaults to the user setting.

    What comes back per granularity: `chunk`, the index rows themselves, as they were stored;
    `passage`, the consecutive chunks of one document merged and widened to the line or the whole
    sentences around them; `excerpt`, a passage with the parts that do not answer the query left
    out (today the passage itself).
    """
    started = time.perf_counter()
    names = await retrieval.scope(session_id, collections)
    if granularity == Granularity.PASSAGE:
        found: list[Hit] | list[Passage] | list[Excerpt] = await flow.passages(names, q, limit)
    elif granularity == Granularity.EXCERPT:
        found = await flow.excerpts(names, q, limit)
    else:
        found = await flow.chunks(names, q, limit)
    await session.record_search(session_id, "explore", q, found, started)
    return found


@get("/api/search/excerpts", mcp_tool="search_excerpts")
async def search_excerpts(
    q: str, session_id: str | None = None, collections: str | None = None, limit: Limit = None
) -> list[Excerpt]:
    """What the sources say about a question, as passages ready to quote, best first.

    Each excerpt is what one document says in one place: the chunks that matched, merged where
    they sit next to each other, and widened to the line or the whole sentences around them. So
    it begins and ends where the author stopped. Cite it by its `header` (the heading path inside
    the document) and its `location` (document, pages, lines). `markdown_file` is the whole
    document on disk when the excerpt is not enough.

    Where it looks: the comma-separated `collections` if given, else the collections selected for
    `session_id`, else every collection. Run `search_sources` first when the question is which
    documents or collections cover a topic, then `set_session_collections` with the cover it
    returns. No results is an answer: the sources do not cover this, and saying so beats guessing.

    Args:
        session_id: The conversation's id; the search then shows in that session's history.
    """
    started = time.perf_counter()
    found = await flow.excerpts(await retrieval.scope(session_id, collections), q, limit)
    await session.record_search(session_id, "excerpts", q, found, started)
    return found


@get("/api/search/sources", mcp_tool="search_sources")
async def search_sources(
    q: str,
    session_id: str | None = None,
    collections: str | None = None,
    limit: Limit = None,
    sections: int | None = None,
) -> Sources:
    """Which documents cover a topic, and which collections to select to read them.

    One row per document rather than per passage: `score` folds its best matching chunk with all
    of them (so many weak mentions never outrank one strong one), `chunks` counts them, `sections`
    names the hottest headings inside it with their `location`, and `collections` says which of
    the searched collections hold it. `documents` is that list, best first; `collections` at the
    top level is the smallest set of collections covering every document in it — pass it to
    `set_session_collections`, then ask `search_excerpts` for the passages themselves.

    Where it looks: the comma-separated `collections` if given, else the collections selected for
    `session_id`, else every collection. No results is an answer: nothing here covers the topic.

    Args:
        session_id: The conversation's id; the search then shows in that session's history.
    """
    started = time.perf_counter()
    names = await retrieval.scope(session_id, collections)
    found = await flow.sources(names, q, limit, sections)
    await session.record_search(session_id, "sources", q, found.documents, started)
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
async def chunk_trend(days: int = 7) -> list[operations.ChunksAt]:
    """Every finished index of the last `days` days, oldest first, for the Insights chart."""
    return await operations.chunks_since(_trend_cutoff(days))


@get("/api/sessions/{session_id:str}/history")
async def session_history(session_id: str) -> list[session.SessionEvent]:
    """What a session did, newest first; the last 100 events at most."""
    return await session.history(session_id)


@get("/api/search/text")
async def search_text(
    q: str,
    collections: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
    session_id: str | None = None,
) -> Page[Hit]:
    """Full-text (BM25) search across all collections, or the comma-separated `collections`.

    No embedding model and no session needed. A chunk held by several collections is returned
    once. Pass the `next_cursor` of a response back as `cursor` for the next page; it is null on
    the last page. Every page recomputes the ranking, so a document indexed between two pages can
    move a result across a page boundary.

    Args:
        session_id: The conversation's id; the search then shows in that session's history.
    """
    started = time.perf_counter()
    page = await text.search(q, text.split_collections(collections), page_size, cursor)
    if cursor is None:  # one event per search, not one per page of it
        await session.record_search(session_id, "text", q, page.items, started)
    return page
