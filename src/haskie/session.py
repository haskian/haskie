"""Agent sessions: which libraries a session searches, and the cross-library search itself."""

import asyncio
import sqlite3

import anyio

from haskie import cpu, db, models
from haskie.embed import embed_query, rerank_scores
from haskie.errors import InvalidInput, LibraryNotFound
from haskie.index import SEARCH_CONCURRENCY, Hit, LibraryIndex
from haskie.library import Library
from haskie.logs import get_logger
from haskie.settings import SearchSettings, load_user_settings

MAX_SESSION_ID = 128
MAX_LIBRARIES = 100  # a session selects libraries by hand; a longer list is a client mistake

# What identifies one chunk across the whole session: (library, doc, part, chunk_id).
RowKey = tuple[str, str, int, int]

_log = get_logger(__name__)


async def load() -> dict[str, list[str]]:
    """Every session with its libraries, in the order the session chose them. Two queries rather
    than a join: a session that selected nothing still has to be listed."""
    sessions: dict[str, list[str]] = {}
    async with db.connect() as conn:
        cursor = await conn.execute("select id from sessions order by id")
        for (session_id,) in await cursor.fetchall():
            sessions[session_id] = []
        cursor = await conn.execute(
            "select session_id, library from session_libraries order by session_id, position"
        )
        rows = await cursor.fetchall()
    for session_id, library in rows:
        sessions.setdefault(session_id, []).append(library)
    return sessions


async def set_libraries(session: str, libraries: list[str]) -> list[str]:
    """Replace the selection of one session, in one transaction: the session row, then its
    library rows with the caller's order as `position`."""
    if not session or len(session) > MAX_SESSION_ID:
        raise InvalidInput(f"session id must be 1..{MAX_SESSION_ID} characters")
    chosen = list(dict.fromkeys(libraries))  # deduplicate, keep the caller's order
    if len(chosen) > MAX_LIBRARIES:
        raise InvalidInput(f"at most {MAX_LIBRARIES} libraries per session, got {len(chosen)}")
    for name in chosen:
        await Library.get(name)  # fail fast on an unknown library, with its name in the message
    async with db.connect() as conn:
        await conn.execute(
            "insert into sessions (id) values (?) on conflict (id) do nothing", (session,)
        )
        await conn.execute("delete from session_libraries where session_id = ?", (session,))
        try:
            await conn.executemany(
                "insert into session_libraries (session_id, library, position) values (?, ?, ?)",
                [(session, name, position) for position, name in enumerate(chosen)],
            )
        except sqlite3.IntegrityError as exc:
            # the foreign key, not a duplicate: a library deleted since the check above
            raise LibraryNotFound("a chosen library was deleted meanwhile; try again") from exc
    return chosen


async def libraries_for(session: str) -> list[str]:
    async with db.connect() as conn:
        cursor = await conn.execute(
            "select library from session_libraries where session_id = ? order by position",
            (session,),
        )
        rows = await cursor.fetchall()
    return [library for (library,) in rows]


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


def _row_key(library: str, row: dict) -> RowKey:
    return (library, row["doc"], row.get("part", 0), row["chunk_id"])


async def search(session: str, query: str, limit: int | None = None) -> list[Hit]:
    """Search every library of the session and merge the results into one ranking.

    The query is embedded once and each model is checked once for the whole fan-out, the libraries
    are then read concurrently, and the per-library rankings are fused by rank (see `rrf_merge`):
    the merged `Hit.score` is an RRF score, or the cross-encoder's when a reranker is on. A single
    library keeps its own scores, because there is nothing to compare them with.

    A library deleted since the session chose it is skipped, so one stale name does not break
    every search. A library that fails to answer is not: a silent hole in the results would be
    read as "no match".
    """
    user = await load_user_settings()
    base = user.search
    limit = limit or base.limit
    names = await libraries_for(session)
    found = await Library.load_settings(names)
    for name in names:
        if name not in found:
            _log.warning("session_library_missing", session_id=session, library=name)
    plans = [(Library(name), found[name].resolve_search(user)) for name in names if name in found]
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

    async def retrieve(plan: tuple[Library, SearchSettings]) -> tuple[LibraryIndex, list[dict]]:
        library, settings = plan
        index = library.index_with(embedding)
        wanted = None if settings.mode == "fts" else vector
        try:
            async with slots:
                return index, await index.search_rows(query, wanted, settings, candidates)
        except Exception:
            _log.exception(
                "session_library_search_failed", library=library.name, session_id=session
            )
            raise

    # no `return_exceptions`: the first library that cannot answer fails the whole search
    retrieved = await asyncio.gather(*(retrieve(plan) for plan in plans))

    libraries = {library.name: library for library, _ in plans}

    rows: dict[RowKey, tuple[LibraryIndex, dict]] = {}
    rankings: list[list[RowKey]] = []
    for index, found_rows in retrieved:
        keys = [_row_key(index.library, row) for row in found_rows]
        rows.update({key: (index, row) for key, row in zip(keys, found_rows, strict=True)})
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
    # the index knows the row, the library knows where its files are today (Library.resolve_hit)
    return [
        libraries[rows[key][0].library].resolve_hit(rows[key][0].hit(rows[key][1], score))
        for key, score in merged[:limit]
    ]


def _by_score(scored: tuple[RowKey, float]) -> float:
    """Sort key for the merged ranking: best first, so the score is negated rather than the list
    reversed (reversing would also flip the stable tie order)."""
    return -scored[1]
