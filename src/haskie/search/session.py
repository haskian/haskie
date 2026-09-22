"""Agent sessions: which collections a session searches, and what it was seen doing.

The search itself is `retrieval.chunks` over the collections a session selected, which is where
it belongs: one document may sit in several collections, so the same passage can come back from
more than one of them, and a search is about passages rather than about memberships.
"""

import sqlite3
import time
from collections.abc import Sequence
from typing import Any, Literal, Protocol

import msgspec

from haskie import db
from haskie.collection.collection import Collection
from haskie.errors import InvalidInput, NotFound

MAX_SESSION_ID = 128
MAX_COLLECTIONS = 100  # a session selects collections by hand; a longer list is a client mistake
MAX_HISTORY = 100  # ponytail: the newest events only; page it when someone scrolls past 100
# What a session can be seen doing. `collections` is the selection itself being set.
type Action = Literal["search", "import", "attach", "detach", "describe", "collections"]


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
        sessions[session_id].append(collection)
    return sessions


async def set_collections(session: str, collections: list[str]) -> list[str]:
    """Replace the selection of one session, in one transaction: the session row, then its
    collection rows with the caller's order as `position`."""
    _checked(session)
    chosen = list(dict.fromkeys(collections))  # deduplicate, keep the caller's order
    if len(chosen) > MAX_COLLECTIONS:
        raise InvalidInput(f"at most {MAX_COLLECTIONS} collections per session, got {len(chosen)}")
    # fail fast on an unknown collection, in one query and with its name in the message
    known = await Collection.load_settings(chosen)
    missing = [name for name in chosen if name not in known]
    if missing:
        raise NotFound(f"collection not found: {missing[0]}")
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
            raise NotFound("a chosen collection was deleted meanwhile; try again") from exc
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


class EventDetail(msgspec.Struct, omit_defaults=True):
    """Whatever the action has to say beyond its subject. One struct rather than a free dict, so
    the generated client knows the fields: every one is optional, and an action fills the few that
    apply to it."""

    # search: "explore", "excerpts", "sources", "text", or a collection name
    scope: str | None = None
    hits: int | None = None  # search: how many passages came back
    docs: list[str] | None = None  # search: the distinct documents among the hits, best first
    collection: str | None = None  # attach, detach
    collections: list[str] | None = None  # collections: the selection that was set


class SessionEvent(msgspec.Struct):
    """One thing a session did: what, to what, and what came of it in one line.

    `detail` is whatever that action has to say: `hits`, `docs` and `scope` for a search,
    `collections` for a selection. `operation_id` names the operation the action started, if any."""

    ts: float  # unix seconds
    action: Action
    subject: str  # the query, the document, "document -> collection", the chosen collections
    detail: EventDetail
    operation_id: str | None
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
    detail: EventDetail | None = None,
    operation_id: str | None = None,
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
            "(session_id, ts, action, subject, detail, operation_id, duration_ms) "
            "values (?, ?, ?, ?, ?, ?, ?)",
            (
                session,
                time.time(),
                action,
                subject,
                msgspec.json.encode(detail or EventDetail()),
                operation_id,
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
    (the scopes `EventDetail` lists, or a collection's name). `started` is a `perf_counter`."""
    docs = list(dict.fromkeys(hit.doc for hit in found))
    await record(
        session,
        "search",
        query,
        detail=EventDetail(scope=scope, hits=len(found), docs=docs),
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
            "select ts, action, subject, detail, operation_id, duration_ms from session_events "
            "where session_id = ? order by ts desc, id desc limit ?",
            (session, limit),
        )
        rows = await cursor.fetchall()
    return [
        SessionEvent(
            ts,
            action,
            subject,
            msgspec.json.decode(detail, type=EventDetail),
            operation_id,
            duration_ms,
        )
        for ts, action, subject, detail, operation_id, duration_ms in rows
    ]


async def origins(operation_ids: list[str]) -> dict[str, str]:
    """Which session started each of these operations; an id nobody's session started is absent.
    One query for a whole page of operations."""
    if not operation_ids:
        return {}
    marks = db.placeholders(len(operation_ids))
    async with db.connect() as conn:
        cursor = await conn.execute(
            f"select operation_id, session_id from session_events where operation_id in ({marks})",
            operation_ids,
        )
        rows = await cursor.fetchall()
    return {operation_id: session_id for operation_id, session_id in rows}
