"""Sessions, every search over the collections, the search log, and the Insights trends."""

import asyncio
from enum import StrEnum
from typing import Annotated

import msgspec
from litestar import get, put
from litestar.params import Parameter

from haskie import audit
from haskie.api import agent
from haskie.api.common import (
    Days,
    Ids,
    Limit,
    RequiredSessionId,
    SessionId,
    days_ago,
    scope_of,
)
from haskie.collection.index import Hit
from haskie.indexing import operations
from haskie.paging import DEFAULT_PAGE_SIZE, Page
from haskie.search import aspects, flow, log, retrieval, session, text
from haskie.search.passage import Answer, Passage
from haskie.search.section_map import SectionMap


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
    return await session.summaries(await log.last_searched())


@put("/api/sessions/{session_id:str}", mcp_tool="set_session_collections")
@audit.audited("session.collections.set")
async def put_session(session_id: RequiredSessionId, data: SessionCollections) -> list[str]:
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
    session_id: SessionId = None,
    collections: str | None = None,
    limit: Limit = None,
    document_ids: Ids = None,
    section_ids: Ids = None,
) -> list[Hit] | list[Passage]:
    """Search at the granularity the caller wants: the exploration endpoint the UI drives, and
    what searches chunks and passages.

    Where it looks: the comma-separated `collections` if given, else the collections selected for
    `session_id`, else every collection. `limit` defaults to that collection's setting when one
    collection is searched, else to the user's.

    Where it looks, narrower: `document_ids` keeps to these documents and `section_ids` to these
    sections and every section under them, by the ids `search_sections` and each result carry; both
    together keep to what both allow.

    What comes back per granularity: `chunk`, the matching index rows; `passage`, the consecutive
    chunks of one section merged into one span, cut where the chunker cut. The excerpts an agent
    reads are `search_excerpts`. At every granularity a near-duplicate is folded into the better
    result it repeats: it is listed in that result's `also_in` rather than on its own, and its
    slot goes to the next result down.
    """
    q = aspects.question(q)  # stripped as `report_gap` matches it
    within = scope_of(document_ids, section_ids)
    async with log.capturing(log.Tool.EXPLORE, [q], session_id) as capture:
        names = await retrieval.scope(session_id, collections)
        if granularity == Granularity.PASSAGE:
            found: list[Hit] | list[Passage] = await flow.passages(names, q, limit, within)
        else:
            found = await flow.chunks(names, q, limit, scope=within)
        capture.answer(found)
    return found


@get("/api/search/excerpts")
async def search_excerpts(
    q: list[str],
    context: str | None = None,
    session_id: SessionId = None,
    collections: str | None = None,
    limit: Limit = None,
    document_ids: Ids = None,
    section_ids: Ids = None,
) -> Answer:
    """What the sources say about a question, one section of a document per excerpt, and what
    they leave out. The MCP tool `search_excerpts` answers the same with fewer fields."""
    return await _search_excerpts(
        q, context, session_id, collections, limit, document_ids, section_ids
    )


@get("/api/agent/search/excerpts", mcp_tool="search_excerpts", include_in_schema=False)
async def agent_search_excerpts(
    q: list[str],
    context: str | None = None,
    session_id: SessionId = None,
    collections: str | None = None,
    limit: Limit = None,
    document_ids: Ids = None,
    section_ids: Ids = None,
) -> agent.Answer:
    """What the sources say about a question, one section of a document per excerpt, and what
    they leave out: best first for one question, in the order the parts took turns for several.

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
    counts excerpts, 10 by default, and the sections are cut to `max_answer_chars` characters.
    Cite the excerpt by its `header` (the section's heading path) and `location` (document, pages,
    lines), or one passage by its span's `header` and `location`. `spans` lists the passages, each
    with its lines, its score, the questions it answers and the places that repeat it.
    `markdown_file` is the whole document on disk when the excerpt is not enough.

    Several questions at once: when parts of a question may be answered in different places, pass
    each part as its own `q` (2 to 5), and the background they share once as `context`. Each part is
    searched on its own and the parts take turns at the `limit` slots, so one part cannot crowd out
    the others. Each span's `aspects` lists the questions it answers and `aspect_scores` how well it
    matched each (its best chunk for it), and an excerpt's the questions any of its spans does. With
    a reranker on, a tag is its judgement: chunks it scores under the floor are dropped. Without
    one, a tag is rank, not a judgement: a vector or hybrid search finds a nearest passage for any
    question, so read the text before citing it as the answer to a part. A question in `uncovered`
    found nothing. A question no excerpt lists found nothing at all. Write each part as a full
    question, not a keyword. Keep in one `q` the conditions one passage must meet together. Resolve
    an ambiguous question before searching; when you cannot ask, pass one part per reading.

    A passage that says what other places say lists every one of them in its span's `also_in`
    rather than returning each on its own. Each one passed a repeat test, so it adds nothing to
    read, unlike `related` in `search_sections`. `also_in` is a tree: each place sits under what it
    repeats, the passage or a place above it, and has its own `also_in`. Its `relation` to that
    parent says how. `duplicate` is an exact character match: the same text, whitespace aside,
    in any document. `contained` sits inside its parent, which says more. `equivalent` is a
    semantic equivalent: other wording, the same meaning, so a nearly identical vector (a
    hybrid or full-text search also counts nearly the same words; a vector search does not).
    `to_parent` and `to_root` measure it against its parent and against the passage: how much
    of it is in the other (`contained`), how much of the other is in it (`contains`), how alike
    the two are (`alike`), by `words` and by `embedding`, and by `chars` within one document.
    A place may be elsewhere in the same document: check its `document` before citing it as a
    second source.

    Where it looks: the comma-separated `collections` if given, else the collections selected for
    `session_id`, else every collection; `searched` names them. Run `search_sections` first when
    the question is which documents or collections cover a topic, then `set_session_collections`
    with the cover it returns. No excerpts is an answer: the sources do not cover this. Say so
    rather than guess.

    Narrower still: `document_ids` keeps to these documents and `section_ids` to these sections and
    every section under them, by the ids `search_sections` and each excerpt carry (`document_id`,
    `section_id`); both together keep to what both allow. The text an excerpt adds around a passage
    stays inside them too. No excerpts then says only that these documents or sections do not cover
    it: search again without them before saying the sources do not.

    Args:
        q: The question, or 2 to 5 parts of one question, each at most 500 characters.
        context: Background every part shares, at most 200 characters. The query embedding reads
            it in front of each part to find candidates; full-text search and the reranker read
            the part alone, so the context never outranks what a part asks (unless the
            `rerank_with_context` setting is on).
        session_id: The conversation's id; the search then shows in that session's history.
    """
    return agent.view(
        await _search_excerpts(
            q, context, session_id, collections, limit, document_ids, section_ids
        ),
        agent.Answer,
    )


async def _search_excerpts(
    q: list[str],
    context: str | None = None,
    session_id: SessionId = None,
    collections: str | None = None,
    limit: Limit = None,
    document_ids: Ids = None,
    section_ids: Ids = None,
) -> Answer:
    # a malformed question is a refused request, not a search: it is checked before the capture
    asked = aspects.questions(q, context)
    within = scope_of(document_ids, section_ids)
    async with log.capturing(
        log.Tool.EXCERPTS, asked.questions, session_id, context=asked.context
    ) as capture:
        names = await retrieval.scope(session_id, collections)
        found = await flow.answers(names, asked, limit, within)
        capture.answer(found.excerpts, found.uncovered, found.missing_terms)
    return found


@get("/api/search/sections")
async def search_sections(
    q: str,
    session_id: SessionId = None,
    collections: str | None = None,
    limit: Limit = None,
    document_ids: Ids = None,
) -> SectionMap:
    """A map of a topic: which sections of which documents touch it, and what each is about,
    without their text. The MCP tool `search_sections` answers the same with fewer fields."""
    return await _search_sections(q, session_id, collections, limit, document_ids)


@get("/api/agent/search/sections", mcp_tool="search_sections", include_in_schema=False)
async def agent_search_sections(
    q: str,
    session_id: SessionId = None,
    collections: str | None = None,
    limit: Limit = None,
    document_ids: Ids = None,
) -> agent.SectionMap:
    """A map of a topic: which sections of which documents touch it, near topics included, and
    what each is about, without their text. Fast: no reranker, no text read.

    The search scans deep and groups what it finds into the sections an excerpt would quote. It
    then picks `limit` of them (15 by default, at most 40) to cover everything the scan found
    rather than to repeat its best part: the most relevant section first, then each time the one
    that covers the most of what the picks so far leave out, at most two of one document while
    another has a section on the topic left. `sections` is that list, in pick order.

    Each section the map did not pick is listed under the pick closest to it, in `related`: at
    most five per pick, closest first, each with its `similarity` to the pick. `related` means
    nearby, not repeated. A near copy of the pick lands there, but so does a section the pick only
    sits close to, and that one may be the best section on the topic. Read the `related` headers
    before choosing what to read. Only `also_in` in `search_excerpts` lists repeats.

    `descriptors` say what each section is about: one to five words or phrases it uses more than the
    other sections of its depth in its document. They skip what its `header` says unless it has no
    other words, and they are fixed when the document was indexed. `chars` is how long it is,
    `chunks` how many of its chunks matched. Cite it by `header` and `location`. Its `document_id`
    names its row in `documents`.

    `documents` lists, best first, the ten documents the search reached hardest, and any other
    document a listed section is in. The ranking reads every chunk the search read, not only the
    ones the map covers, so a document the map picked no section from can still lead. Per document:
    its `score` (its matched chunks' scores folded by the `score_fold` setting, by default their
    sum), `chunks` (how many matched: a sum grows with how much a document says, so read the two
    together), `sections` (how many of the map's sections are in it), its `description`, the
    `collections` holding it and its `markdown_file` on disk. Search a document the map picked
    nothing from with this tool again, its id in `document_ids`.

    Use it before `search_excerpts` to see what the sources hold on a topic and nearby, then ask
    `search_excerpts` about the sections worth reading: pass their `id`s as its `section_ids` to
    read those alone. `collections` at the top level is the smallest set of collections holding
    every section and document listed, for `set_session_collections`.

    Where it looks: the comma-separated `collections` if given, else the collections selected for
    `session_id`, else every collection, and only the documents `document_ids` names, when
    given; `searched` names the collections. No sections is an answer: nothing here covers the
    topic (within `document_ids`, when given).

    Args:
        session_id: The conversation's id; the search then shows in that session's history.
    """
    return agent.view(
        await _search_sections(q, session_id, collections, limit, document_ids), agent.SectionMap
    )


async def _search_sections(
    q: str,
    session_id: SessionId = None,
    collections: str | None = None,
    limit: Limit = None,
    document_ids: Ids = None,
) -> SectionMap:
    q = aspects.question(q)  # stripped as `report_gap` matches it
    within = scope_of(document_ids)
    async with log.capturing(log.Tool.SECTIONS, [q], session_id) as capture:
        names = await retrieval.scope(session_id, collections)
        found = await flow.sections(names, q, limit, within)
        capture.answer(found.sections)
    return found


MAX_SEARCHES = 200  # a page of the log an agent reads through, not an export


@get("/api/searches", mcp_tool="list_searches")
async def list_searches(
    session_id: str | None = None,
    days: Days = 7,
    limit: Annotated[int, Parameter(ge=1, le=MAX_SEARCHES)] = 50,
) -> list[log.LoggedSearch]:
    """What was searched, newest first: every search of the last `days` days, or one session's.

    Each search lists its `questions`, each with how close its best match came
    (`best_similarity`, the cosine of the query to the nearest chunk; `best_rerank`, the
    reranker's score of the best chunk, 0 to 1) and whether no excerpt answered it (`uncovered`,
    when several were asked at once). `results` are what it returned, cited by `header` and
    `location`. A search that failed has an `error` and no results. `list_gaps` groups the
    questions that found no answer.

    Args:
        session_id: Only the searches of this conversation.
        days: How far back, in days.
        limit: At most this many searches, the newest.
    """
    return await log.listed(days_ago(days), session_id, limit)


@get("/api/insights/searches")
async def search_trend(days: Days = 7) -> list[log.SearchAt]:
    """Every search of the last `days` days, oldest first, for the Insights chart; `session_id`
    is null for a search made without one."""
    return await log.searches_since(days_ago(days))


@get("/api/insights/chunks")
async def chunk_trend(days: Days = 7) -> list[operations.ChunksAt]:
    """Every finished index of the last `days` days, oldest first, for the Insights chart."""
    return await operations.chunks_since(days_ago(days))


@get("/api/sessions/{session_id:str}/history")
async def session_history(session_id: str) -> list[session.SessionEvent]:
    """What a session did, newest first: its searches and its other actions, the last 100 at
    most."""
    actions, searched = await asyncio.gather(
        session.history(session_id), log.history(session_id, session.MAX_HISTORY)
    )
    events = actions + searched
    return sorted(events, key=lambda one: one.ts, reverse=True)[: session.MAX_HISTORY]


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
    # one search per query, not one per page of it
    q = aspects.question(q)  # stripped as `report_gap` matches it
    async with log.capturing(log.Tool.TEXT, [q], session_id, record=cursor is None) as capture:
        page = await text.search(q, text.split_collections(collections), page_size, cursor)
        capture.answer(page.items)
    return page
