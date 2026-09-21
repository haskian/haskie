"""Agent sessions: which collections a session searches, and the search across them.

One document may sit in several collections, so the same passage can come back from more than one
of them. A session search is about passages, not about memberships, so a chunk is merged once (see
`search`).
"""

import asyncio
import sqlite3
import time
from collections.abc import Sequence
from typing import Any, Literal, Protocol

import anyio
import msgspec

from haskie import cpu, db, models
from haskie.collection import Collection, resolve_hit
from haskie.embed import embed_query, rerank_scores
from haskie.errors import CollectionNotFound, InvalidInput
from haskie.index import SEARCH_CONCURRENCY, CollectionIndex, Hit, RowKey, row_key
from haskie.logs import get_logger
from haskie.settings import SearchSettings, load_user_settings

MAX_SESSION_ID = 128
MAX_COLLECTIONS = 100  # a session selects collections by hand; a longer list is a client mistake
MAX_HISTORY = 100  # ponytail: the newest events only; page it when someone scrolls past 100
# What a session can be seen doing. `collections` is the selection itself being set.
type Action = Literal["search", "import", "attach", "detach", "describe", "collections"]

_log = get_logger(__name__)


async def load() -> dict[str, list[str]]:
    """Every session with its collections, in the order the session chose them. Two queries rather
    than a join: a session that selected nothing still has to be listed."""
    sessions: dict[str, list[str]] = {}
    async with db.connect() as conn:
        cursor = await conn.execute("select id from sessions order by id")
        for (session_id,) in await cursor.fetchall():
            sessions[session_id] = []
        cursor = await conn.execute(
            "select session_id, collection from session_collections order by session_id, position"
        )
        rows = await cursor.fetchall()
    for session_id, collection in rows:
        sessions.setdefault(session_id, []).append(collection)
    return sessions


async def set_collections(session: str, collections: list[str]) -> list[str]:
    """Replace the selection of one session, in one transaction: the session row, then its
    collection rows with the caller's order as `position`."""
    _checked(session)
    chosen = list(dict.fromkeys(collections))  # deduplicate, keep the caller's order
    if len(chosen) > MAX_COLLECTIONS:
        raise InvalidInput(f"at most {MAX_COLLECTIONS} collections per session, got {len(chosen)}")
    for name in chosen:
        await Collection.get(
            name
        )  # fail fast on an unknown collection, with its name in the message
    async with db.connect() as conn:
        await conn.execute(
            "insert into sessions (id) values (?) on conflict (id) do nothing", (session,)
        )
        await conn.execute("delete from session_collections where session_id = ?", (session,))
        try:
            await conn.executemany(
                "insert into session_collections (session_id, collection, position) "
                "values (?, ?, ?)",
                [(session, name, position) for position, name in enumerate(chosen)],
            )
        except sqlite3.IntegrityError as exc:
            # the foreign key, not a duplicate: a collection deleted since the check above
            raise CollectionNotFound(
                "a chosen collection was deleted meanwhile; try again"
            ) from exc
    return chosen


async def collections_for(session: str) -> list[str]:
    async with db.connect() as conn:
        cursor = await conn.execute(
            "select collection from session_collections where session_id = ? order by position",
            (session,),
        )
        rows = await cursor.fetchall()
    return [collection for (collection,) in rows]


class SessionSummary(msgspec.Struct):
    """One session as the listing shows it: its selection, and when it was last seen doing
    anything (None for one that did nothing yet)."""

    id: str
    collections: list[str]
    last_at: float | None  # unix seconds of its newest event


async def summaries() -> list[SessionSummary]:
    """Every session with its collections and its latest event, in id order; the page sorts."""
    loaded = await load()
    async with db.connect() as conn:
        cursor = await conn.execute(
            "select session_id, max(ts) from session_events group by session_id"
        )
        rows: list[Any] = list(await cursor.fetchall())
    last = dict(rows)
    return [SessionSummary(id, collections, last.get(id)) for id, collections in loaded.items()]


class SessionEvent(msgspec.Struct):
    """One thing a session did: what, to what, and what came of it in one line.

    `detail` is whatever that action has to say: `hits`, `docs` and `scope` for a search,
    `collections` for a selection. `workflow_id` names the operation the action started, if any."""

    ts: float  # unix seconds
    action: Action
    subject: str  # the query, the document, "document -> collection", the chosen collections
    detail: dict[str, Any]
    workflow_id: str | None
    duration_ms: int


def _checked(session: str) -> str:
    if not session or len(session) > MAX_SESSION_ID:
        raise InvalidInput(f"session id must be 1..{MAX_SESSION_ID} characters")
    return session


async def record(
    session: str | None,
    action: Action,
    subject: str,
    *,
    detail: dict[str, Any] | None = None,
    workflow_id: str | None = None,
    duration_ms: int = 0,
) -> None:
    """Append one event to the session's history; a no-op without a session.

    The first action under an id is what creates the session: an agent names its conversation
    and imports before it selects anything, and that import belongs to the conversation too."""
    if session is None:
        return
    _checked(session)
    async with db.connect() as conn:
        await conn.execute(
            "insert into sessions (id) values (?) on conflict (id) do nothing", (session,)
        )
        await conn.execute(
            "insert into session_events "
            "(session_id, ts, action, subject, detail, workflow_id, duration_ms) "
            "values (?, ?, ?, ?, ?, ?, ?)",
            (
                session,
                time.time(),
                action,
                subject,
                msgspec.json.encode(detail or {}),
                workflow_id,
                duration_ms,
            ),
        )


class Found(Protocol):
    """What every search result has in common, as far as its history event is concerned."""

    doc: str


async def record_search(
    session: str | None, scope: str, query: str, found: Sequence[Found], started: float
) -> None:
    """A search as one event: the query, how many hits, which documents, and where it looked
    (`session`, `text`, `documents` or a collection's name). `started` is a `perf_counter`."""
    docs = list(dict.fromkeys(hit.doc for hit in found))
    await record(
        session,
        "search",
        query,
        detail={"scope": scope, "hits": len(found), "docs": docs},
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


class SearchAt(msgspec.Struct):
    """One search, as a point on a trend: when, and which session ran it."""

    ts: float
    session_id: str


async def searches_since(cutoff: float) -> list[SearchAt]:
    """Every search on or after `cutoff`, oldest first. Raw points, not buckets: the reader
    buckets them by its own day boundaries, which the server does not know."""
    async with db.connect() as conn:
        cursor = await conn.execute(
            "select ts, session_id from session_events "
            "where action = 'search' and ts >= ? order by ts",
            (cutoff,),
        )
        rows = await cursor.fetchall()
    return [SearchAt(ts, session_id) for ts, session_id in rows]


async def history(session: str, limit: int = MAX_HISTORY) -> list[SessionEvent]:
    """What the session did, newest first."""
    async with db.connect() as conn:
        cursor = await conn.execute(
            "select ts, action, subject, detail, workflow_id, duration_ms from session_events "
            "where session_id = ? order by ts desc, id desc limit ?",
            (session, limit),
        )
        rows = await cursor.fetchall()
    return [
        SessionEvent(ts, action, subject, msgspec.json.decode(detail), workflow_id, duration_ms)
        for ts, action, subject, detail, workflow_id, duration_ms in rows
    ]


async def origins(workflow_ids: list[str]) -> dict[str, str]:
    """Which session started each of these operations; an id nobody's session started is absent.
    One query for a whole page of jobs."""
    if not workflow_ids:
        return {}
    marks = db.placeholders(len(workflow_ids))
    async with db.connect() as conn:
        cursor = await conn.execute(
            f"select workflow_id, session_id from session_events where workflow_id in ({marks})",
            workflow_ids,
        )
        rows = await cursor.fetchall()
    return {workflow_id: session_id for workflow_id, session_id in rows}


def rrf_merge[T](ranked: list[list[T]], k: int) -> list[tuple[T, float]]:
    """Reciprocal rank fusion: every item scores the sum of `1 / (k + rank)` over the rankings it
    appears in, best first. Ties keep the order of first appearance.

    Pure, and the only merge that needs no calibration between the inputs: two LanceDB indexes
    score rows on their own scale, so their ranks are comparable where their scores are not.
    """
    scores: dict[T, float] = {}
    for ranking in ranked:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


async def search(session: str, query: str, limit: int | None = None) -> list[Hit]:
    """Search every collection of the session and merge the results into one ranking.

    The query is embedded once and each model is checked once for the whole fan-out, the
    collections are then read concurrently, and the per-collection rankings are fused by rank (see
    `rrf_merge`): the merged `Hit.score` is an RRF score, or the cross-encoder's when a reranker is
    on. A single collection keeps its own scores, because there is nothing to compare them with.

    A passage counts once. The same document may be a member of several of the chosen collections,
    and each of their tables then holds the same chunk; a caller searching them wants one hit per
    passage, not one per collection that happens to hold it. So a chunk enters the fusion from the
    first collection in the session's order that returned it, and the later collections' copies are
    dropped before the ranks are counted — otherwise a document in two collections would be fused
    with itself and outrank an equally good one that sits in a single collection.

    A collection deleted since the session chose it is skipped, so one stale name does not break
    every search. A collection that fails to answer is not: a silent hole in the results would be
    read as "no match".
    """
    user = await load_user_settings()
    base = user.search
    limit = limit or base.limit
    names = await collections_for(session)
    found = await Collection.load_settings(names)
    for name in names:
        if name not in found:
            _log.warning("session_collection_missing", session_id=session, collection=name)
    plans = [
        (Collection(name), found[name].resolve_search(user)) for name in names if name in found
    ]
    if not plans:
        return []
    if len(plans) == 1:
        return await plans[0][0].search(query, limit)

    embedding = user.embedding_model
    vector: list[float] | None = None
    if embedding is not None and any(settings.mode != "fts" for _, settings in plans):
        await models.require_ready("embedding", embedding.name)
        vector = await cpu.on_cpu("embed_query", embed_query, embedding, query)
    if base.reranker != "none":
        await models.require_ready("reranker", base.reranker_model)  # fail before the fan-out
    candidates = max(base.candidates, limit)
    # One semaphore per call, never at import time: an anyio primitive belongs to the loop that
    # first used it, and both the Litestar loop and the DBOS loop run searches (see the plan).
    slots = anyio.Semaphore(SEARCH_CONCURRENCY)

    async def retrieve(
        plan: tuple[Collection, SearchSettings],
    ) -> tuple[CollectionIndex, list[dict]]:
        collection, settings = plan
        index = collection.index_with(embedding)
        wanted = None if settings.mode == "fts" else vector
        try:
            async with slots:
                return index, await index.search_rows(query, wanted, settings, candidates)
        except Exception:
            _log.exception(
                "session_collection_search_failed", collection=collection.name, session_id=session
            )
            raise

    # no `return_exceptions`: the first collection that cannot answer fails the whole search
    retrieved = await asyncio.gather(*(retrieve(plan) for plan in plans))

    # `retrieved` is in the order of `plans`, which is the session's own order, so the first
    # collection that holds a passage is the one it is credited to.
    rows: dict[RowKey, tuple[CollectionIndex, dict]] = {}
    rankings: list[list[RowKey]] = []
    for index, found_rows in retrieved:
        keys: list[RowKey] = []
        for row in found_rows:
            key = row_key(row)
            if key in rows:
                continue  # already in the fusion from an earlier collection
            rows[key] = (index, row)
            keys.append(key)
        rankings.append(keys)
    merged = rrf_merge(rankings, base.rrf_k)[:candidates]
    if base.reranker != "none":
        scores = await cpu.on_cpu(
            "rerank",
            rerank_scores,
            base.reranker_model,
            user.pipeline.accelerator,
            query,
            [rows[key][1]["text"] for key, _ in merged],
        )
        merged = sorted(zip([key for key, _ in merged], scores, strict=True), key=_by_score)
    # the index knows the row, `resolve_hit` knows where the document's files are today
    return [resolve_hit(rows[key][0].hit(rows[key][1], score)) for key, score in merged[:limit]]


def _by_score(scored: tuple[RowKey, float]) -> float:
    """Sort key for the merged ranking: best first, so the score is negated rather than the list
    reversed (reversing would also flip the stable tie order)."""
    return -scored[1]
