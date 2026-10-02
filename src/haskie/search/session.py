"""Agent sessions: which collections a session searches, and what it was seen doing.

A session's selection is only the default scope of a search (`retrieval.scope`). The search
itself runs over the collections, where it belongs: one document may sit in several of them, and
the search counts its chunks once whichever collections hold them (`retrieval.fan_out`).
"""

import time
from enum import StrEnum

import msgspec
from sqlalchemy import delete, func, literal, select
from sqlalchemy.dialects.sqlite import Insert, insert
from sqlalchemy.exc import IntegrityError

from haskie import db
from haskie.collection.collection import Collection
from haskie.errors import InvalidInput, NotFound
from haskie.tables import (
    session_collections,
    session_events,
    sessions,
)

MAX_SESSION_ID = 128
MAX_COLLECTIONS = 100  # a session selects collections by hand; a longer list is a client mistake
MAX_HISTORY = 100  # the newest events only; page it when someone scrolls past 100


# What a session can be seen doing. `collections` is the selection itself being set.
class Action(StrEnum):
    SEARCH = "search"
    IMPORT = "import"
    ATTACH = "attach"
    DETACH = "detach"
    DESCRIBE = "describe"
    COLLECTIONS = "collections"


async def load() -> dict[str, list[str]]:
    """Every session with its collections, in the order the session chose them. Two queries rather
    than a join: a session that selected nothing still has to be listed."""
    chosen = session_collections.c
    async with db.read() as conn:
        ids = await conn.scalars(select(sessions.c.id).order_by(sessions.c.id))
        loaded: dict[str, list[str]] = {session_id: [] for session_id in ids}
        rows = await conn.execute(
            select(chosen.session_id, chosen.collection).order_by(
                chosen.session_id, chosen.position
            )
        )
    for session_id, collection in rows:
        loaded[session_id].append(collection)
    return loaded


async def set_collections(session: str, collections: list[str]) -> list[str]:
    """Replace the selection of one session, in one transaction: the session row, then its
    collection rows with the caller's order as `position`."""
    checked(session)
    chosen = list(dict.fromkeys(collections))  # deduplicate, keep the caller's order
    if len(chosen) > MAX_COLLECTIONS:
        raise InvalidInput(f"at most {MAX_COLLECTIONS} collections per session, got {len(chosen)}")
    # fail fast on an unknown collection, in one query and with its name in the message
    known = await Collection.load_overrides(chosen)
    missing = [name for name in chosen if name not in known]
    if missing:
        raise NotFound(f"collection not found: {missing[0]}")
    async with db.connect() as conn:
        await conn.execute(create_session(session))
        await conn.execute(
            delete(session_collections).where(session_collections.c.session_id == session)
        )
        try:
            if chosen:
                await conn.execute(
                    insert(session_collections),
                    [
                        {"session_id": session, "collection": name, "position": position}
                        for position, name in enumerate(chosen)
                    ],
                )
        except IntegrityError as exc:
            # the foreign key, not a duplicate: a collection deleted since the check above
            raise NotFound("a chosen collection was deleted meanwhile; try again") from exc
    return chosen


async def collections_for(session: str) -> list[str]:
    async with db.read() as conn:
        chosen = await conn.scalars(
            select(session_collections.c.collection)
            .where(session_collections.c.session_id == session)
            .order_by(session_collections.c.position)
        )
        return list(chosen)


class SessionSummary(msgspec.Struct):
    """One session as the listing shows it: its selection, and when it was last seen doing
    anything (None for one that did nothing yet)."""

    id: str
    collections: list[str]
    last_at: float | None  # unix seconds of its newest event


async def summaries(last_searched: dict[str, float]) -> list[SessionSummary]:
    """Every session with its collections and its latest event or search, in id order; the page
    sorts. `last_searched` is when each session last searched (`log.last_searched`)."""
    loaded = await load()
    async with db.read() as conn:
        rows = await conn.execute(
            select(session_events.c.session_id, func.max(session_events.c.ts)).group_by(
                session_events.c.session_id
            )
        )
        last = dict(last_searched)
        for session_id, ts in rows:
            last[session_id] = max(ts, last.get(session_id, ts))
    return [SessionSummary(id, collections, last.get(id)) for id, collections in loaded.items()]


class EventDetail(msgspec.Struct, omit_defaults=True):
    """Whatever the action has to say beyond its subject. One struct rather than a free dict, so
    the generated client knows the fields: every one is optional, and an action fills the few that
    apply to it."""

    # search: the tool that ran it, "explore", "excerpts", "sections" or "text"
    scope: str | None = None
    hits: int | None = None  # search: how many results came back
    documents: list[str] | None = None  # search: the distinct documents among the hits, best first
    error: str | None = None  # search: why it failed
    collection: str | None = None  # attach, detach
    collections: list[str] | None = None  # collections: the selection that was set
    questions: list[str] | None = None  # search: the parts, when several were asked at once
    context: str | None = None  # search: the context an excerpts search sent


class SessionEvent(msgspec.Struct):
    """One thing a session did: what, to what, and what came of it in one line.

    `detail` is whatever that action has to say: `hits`, `documents`, `scope` and `error` for a
    search, `collections` for a selection. `operation_id` names the operation the action started,
    if any. A search is read from the search log (`searches`), everything else from
    `session_events`."""

    ts: float  # unix seconds
    action: Action
    subject: str  # the query, the document, "document -> collection", the chosen collections
    detail: EventDetail
    operation_id: str | None
    duration_ms: int


def create_session(session: str) -> Insert:
    return insert(sessions).values(id=session).on_conflict_do_nothing()


def checked(session: str) -> str:
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
) -> None:
    """Append one event to the session's history; a no-op without a session. A search is not an
    event: the search log records it (`log.capturing`).

    The first action under an id is what creates the session: an agent names its conversation
    and imports before it selects anything, and that import belongs to the conversation too."""
    if session is None:
        return
    checked(session)
    async with db.connect() as conn:
        await conn.execute(create_session(session))
        await conn.execute(
            insert(session_events).values(
                session_id=session,
                ts=time.time(),
                action=action,
                subject=subject,
                detail=db.dumps(detail or EventDetail()),
                operation_id=operation_id,
            )
        )


async def history(session: str, limit: int = MAX_HISTORY) -> list[SessionEvent]:
    """What the session did besides searching, newest first; its searches are in the search log
    (`log.history`)."""
    event = session_events.c
    async with db.read() as conn:
        rows = await conn.execute(
            select(
                event.ts,
                event.action,
                event.subject,
                event.detail,
                event.operation_id,
                literal(0).label("duration_ms"),  # only a search takes time worth showing
            )
            .where(event.session_id == session)
            .order_by(event.ts.desc(), event.id.desc())
            .limit(limit)
        )
        return [db.row_to(SessionEvent, row, detail=EventDetail) for row in rows]


async def origins(operation_ids: list[str]) -> dict[str, str]:
    """Which session started each of these operations; an id nobody's session started is absent.
    One query for a whole page of operations."""
    if not operation_ids:
        return {}
    async with db.read() as conn:
        rows = await conn.execute(
            select(session_events.c.operation_id, session_events.c.session_id).where(
                session_events.c.operation_id.in_(operation_ids)
            )
        )
        return dict(rows.tuples().all())
