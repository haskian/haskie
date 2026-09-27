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
from haskie.search import aspects, flow, retrieval, session, text
from haskie.search.passage import Answer, Passage, Sources


# What an exploration returns: the chunks the index holds, or the passages they merge into. The
# excerpts an agent reads have their own route (`search_excerpts`), since they answer with more
# than a list: what they leave out.
class Granularity(StrEnum):
    CHUNK = "chunk"
    PASSAGE = "passage"


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
) -> list[Hit] | list[Passage]:
    """Search at the granularity the caller wants: the exploration endpoint the UI drives.

    Where it looks: the comma-separated `collections` if given, else the collections selected for
    `session_id`, else every collection. `limit` defaults to that collection's setting when one
    collection is searched, else to the user's.

    What comes back per granularity: `chunk`, the matching index rows; `passage`, the consecutive
    chunks of one section merged into one span, cut where the chunker cut. The excerpts an agent
    reads are `search_excerpts`. At every granularity a near-duplicate is folded into the better
    result it repeats: it is listed in that result's `also_in` rather than on its own, and its
    slot goes to the next result down.
    """
    started = time.perf_counter()
    names = await retrieval.scope(session_id, collections)
    if granularity == Granularity.PASSAGE:
        found: list[Hit] | list[Passage] = await flow.passages(names, q, limit)
    else:
        found = await flow.chunks(names, q, limit)
    await session.record_search(session_id, "explore", q, found, started)
    return found


@get("/api/search/excerpts", mcp_tool="search_excerpts")
async def search_excerpts(
    q: list[str],
    context: str | None = None,
    session_id: str | None = None,
    collections: str | None = None,
    limit: Limit = None,
) -> Answer:
    """What the sources say about a question, one section of a document per excerpt, best first,
    and what they leave out.

    The answer is `excerpts`, `uncovered` and `missing_terms`. `uncovered` lists the questions no
    excerpt answers, when several were asked. `missing_terms` lists the words of the questions
    (stopwords aside) that no excerpt's text or headings hold, a form of the word counting ("keeps"
    holds "keep"). Before answering, the search looks for those words once more by full text, and
    the best passage it finds joins the answer: in the section it belongs to, or as one excerpt past
    `limit` and the budget. What is still missing after that is what the sources do not say in those
    words; a synonym in the text does not count.

    Each excerpt is one section of one document: the largest heading whose text is at most a few
    pages, with every passage of it the search matched, in document order, and the text around
    and between them that matches the question as well as they do. `text` joins them: each
    passage opens with the headings it sits under below `header`, and `[…]` marks text skipped
    between two of them because it did not match. Chunks are cut at headings, blank lines,
    blocks and sentences, so each passage begins and ends where the author stopped. `limit`
    counts excerpts, and the sections are cut to `max_answer_chars` characters.
    Cite the excerpt by its `header` (the section's heading path) and `location` (document, pages,
    lines), or one passage by its span's `header` and `location`. `spans` lists the passages, each
    with its lines, its score, the questions it answers and the places that repeat it.
    `markdown_file` is the whole document on disk when the excerpt is not enough.

    Several questions at once: when parts of a question may be answered in different places,
    pass each part as its own `q` (2 to 5), and the background they share once as `context`.
    Each part is searched on its own and the parts take turns at the `limit` slots, so one part
    cannot crowd out the others. Each span's `aspects` lists the questions it answers and
    `aspect_scores` how well it matched each, and an excerpt's the questions any of its spans
    does. With a reranker on, a tag is its judgement: chunks it scores under the floor are
    dropped. Without one, a tag is rank, not a judgement: a vector or hybrid search finds a
    nearest passage for any question, so read the text before citing it as the answer to a
    part. A question in `uncovered` found nothing. A question no excerpt
    lists found nothing at all. Write each part as a full question, not a keyword. Keep in one
    `q` the conditions one passage must meet together. Resolve an ambiguous question before
    searching; when you cannot ask, pass one part per reading.

    A passage that says what other places say lists every one of them in its span's `also_in`
    rather than returning each on its own. `also_in` is a tree: each place sits under what it
    repeats, the passage or a place above it, and has its own `also_in`. Its `relation` to that
    parent says how. `duplicate` is an exact character match: the same text, whitespace aside,
    in any document. `contained` sits inside its parent, which says more. `equivalent` is a
    semantic equivalent: other wording, the same meaning, so a nearly identical vector (a
    hybrid or full-text search also counts nearly the same words; a vector search does not).
    `to_parent` and `to_root` measure it against its parent and against the passage: how much
    of it is in the other (`contained`), how much of the other is in it (`contains`), how alike
    the two are (`alike`), by `words` and by `embedding`, and by `chars` within one document.
    A place may be
    elsewhere in the same document: check its `document` before citing it as a second source.

    Where it looks: the comma-separated `collections` if given, else the collections selected for
    `session_id`, else every collection. Run `search_sources` first when the question is which
    documents or collections cover a topic, then `set_session_collections` with the cover it
    returns. No excerpts is an answer: the sources do not cover this, and saying so beats
    guessing.

    Args:
        q: The question, or 2 to 5 parts of one question, each at most 500 characters.
        context: Background every part shares, at most 200 characters. The query embedding reads
            it in front of each part to find candidates; full-text search and the reranker read
            the part alone, so the context never outranks what a part asks.
        session_id: The conversation's id; the search then shows in that session's history.
    """
    started = time.perf_counter()
    asked = aspects.questions(q, context)
    found = await flow.answers(await retrieval.scope(session_id, collections), asked, limit)
    several = asked.questions if len(asked.questions) > 1 else None
    await session.record_search(
        session_id,
        "excerpts",
        " | ".join(asked.questions),
        found.excerpts,
        started,
        questions=several,
        context=asked.context,
    )
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
