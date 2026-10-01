"""The search log: every search as it ran, each question it asked, and what it returned.

A handler opens a `Capture` around the search (`capturing`), and the search fills it in as it
goes, through a context variable, so no step has to pass it along:

- the scope, the mode and the limit, where the search settles them (`observe_scope`: the flow's
  plan, the full-text listing);
- per question, what its ranking measured (`observe_ranking`, after the flow's `rerank` step,
  which every ranked search runs once per question): the query vector, the best cosine between it
  and any row read, and the reranker's best score before its floor dropped any;
- the answer itself, every result and every place folded into one, and the questions no excerpt
  answers (`Capture.answer`).

The capture is written when the search ends, failed or not: a failure is recorded with its error
and raised on. Without a session too: a search is the curator's evidence whoever ran it.

The raw signals are stored and nothing is judged here. Which questions count as gaps is decided
on read (`search/gaps.py`), so a bar that is measured again re-judges every search already stored.
"""

import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from enum import StrEnum
from typing import TYPE_CHECKING

import anyio
import msgspec
import numpy as np
from sqlalchemy import delete, func, insert, select

from haskie import audit, db, home
from haskie.collection.index import Hit, HitReference, Relation
from haskie.search import collapse, session
from haskie.search.passage import Excerpt, Passage, PassageReference
from haskie.search.section_map import MappedSection
from haskie.settings import Reranker, SearchMode
from haskie.tables import search_questions, search_results, searches

if TYPE_CHECKING:  # `retrieval` imports this module through `text`: at run time, a cycle
    from haskie.search.retrieval import Plan, Pool


class Tool(StrEnum):
    """Which endpoint ran a search."""

    EXCERPTS = "excerpts"
    SECTIONS = "sections"
    EXPLORE = "explore"
    TEXT = "text"


class LoggedResult(msgspec.Struct):
    """One place a search returned: a result, or a place folded into one (`also_in`).

    `position` numbers the places of one search in preorder, so a result comes before the places
    folded into it; `parent` is the position of the place it is folded under, None for a result.
    The citation (`header`, `location`) is stored with it, so the log reads without the document.
    """

    position: int
    parent: int | None
    relation: Relation | None  # how it overlaps its parent; None for a result
    collection: str
    document: str
    seq_start: int  # the chunks it covers
    seq_end: int
    line_start: int
    line_end: int
    header: str
    location: str
    score: float


Place = Hit | HitReference | Passage | PassageReference | Excerpt | MappedSection


def _folded(place: Place) -> list[PassageReference] | list[HitReference]:
    """The places folded into `place`. An excerpt's are its passages', in document order: the
    passages are the excerpt itself, what repeats them is somewhere else."""
    if isinstance(place, Excerpt):
        return [folded for span in place.spans for folded in span.also_in]
    return [] if isinstance(place, MappedSection) else place.also_in


def flatten(found: Sequence[Place]) -> list[LoggedResult]:
    """Every result and every place folded into one, in preorder."""
    flat: list[LoggedResult] = []

    def visit(place: Place, parent: int | None) -> None:
        position = len(flat)
        flat.append(_result(place, position, parent))
        for child in _folded(place):
            visit(child, position)

    for place in found:
        visit(place, None)
    return flat


def _result(place: Place, position: int, parent: int | None) -> LoggedResult:
    match place:
        case Hit() | HitReference():
            seq = (place.seq, place.seq)
        case Passage() | PassageReference() | Excerpt() | MappedSection():
            seq = (place.seq_start, place.seq_end)
    return LoggedResult(
        position=position,
        parent=parent,
        relation=place.relation if isinstance(place, HitReference | PassageReference) else None,
        collection=place.collection,
        document=place.document,
        seq_start=seq[0],
        seq_end=seq[1],
        line_start=place.line_start,
        line_end=place.line_end,
        header=place.header,
        location=place.location,
        score=place.score,
    )


PROFILE = 20  # scores kept per ranking: the head of the list, where an answer would be


def similarities(query: Sequence[float], rows: Sequence[Sequence[float]]) -> list[float]:
    """The `PROFILE` best cosines of a query to the vectors of the rows its ranking read, best
    first. Pure, so the search log (`observe_ranking`) and an offline evaluation measure alike."""
    if not rows:
        return []
    cosines = collapse.unit_rows(rows) @ collapse.unit_rows([query])[0]
    return sorted(cosines.tolist(), reverse=True)[:PROFILE]


class LoggedQuestion(msgspec.Struct):
    """One question of a search, and what its own ranking measured. `id` is the log's, None until
    the search is written; `review` is the curator's decision on it as a gap (`search.gaps`).

    `best_similarity` and `best_rerank` are the heads of the two score lists, filled from them."""

    question: str
    id: int | None = None
    similarities: list[float] = []  # the best cosines of the query to any row read, best first
    rerank_scores: list[float] = []  # the reranker's best scores before its floor, best first
    best_similarity: float | None = None
    best_rerank: float | None = None
    uncovered: bool = False  # several were asked, and no excerpt answers this one
    review: str | None = None
    # the agent's verdict on what the search gave it (`gaps.report`): insufficient or partial
    agent_verdict: str | None = None
    agent_note: str | None = None  # what the agent said the excerpts lacked

    def __post_init__(self) -> None:
        if self.similarities:
            self.best_similarity = self.similarities[0]
        if self.rerank_scores:
            self.best_rerank = self.rerank_scores[0]


class Asked(LoggedQuestion):
    """A question while its search runs: with its query embedding, which the log stores to group
    gaps by topic (`vectors`) and never sends to a caller."""

    vector: list[float] | None = None


class Searched(msgspec.Struct, kw_only=True):
    """What a search settled and what it got, while it runs (`Capture`) or read back (`Logged`):
    what the gap detectors judge it by. One `searches` row holds each field."""

    tool: Tool
    session_id: str | None
    context: str | None = None  # the background an excerpts search's questions shared
    collections: list[str] = []  # the collections it searched
    mode: SearchMode | None = None  # how it ranked: hybrid, vector or fts
    embedding: str | None = None  # the profile key the queries were embedded under
    reranker: str | None = None  # the cross-encoder model, None when the search did not rerank
    # the floor the settings set in place of the reranker's own, None when they set none
    min_rerank_score: float | None = None
    result_limit: int | None = None
    result_count: int = 0  # the results the caller got, without the places folded into them
    scoped: bool = False  # kept to some documents or sections: a miss says nothing of the rest
    missing_terms: list[str] = []  # the words of its questions no excerpt held; excerpts only
    error: str | None = None  # why it failed; a failed search returned nothing


class Capture(Searched, kw_only=True):
    """One search while it runs: what it was asked, what it settled, what it measured."""

    asked: list[Asked]
    results: list[LoggedResult] = []

    def answer(
        self,
        found: Sequence[Place],
        uncovered: Sequence[str] = (),
        missing_terms: Sequence[str] = (),
    ) -> None:
        """What the caller got back, the questions no excerpt answers, and the words of the
        questions no excerpt holds."""
        self.results = flatten(found)
        self.result_count = sum(result.parent is None for result in self.results)
        self.missing_terms = list(missing_terms)
        for one in self.asked:
            one.uncovered = one.question in uncovered


_capture: ContextVar[Capture | None] = ContextVar("haskie_search_capture", default=None)


@asynccontextmanager
async def capturing(
    tool: Tool,
    questions: Sequence[str],
    session_id: str | None,
    *,
    context: str | None = None,
    record: bool = True,
) -> AsyncIterator[Capture]:
    """Capture the search run inside the block, and write it to the log when the block ends.

    A failed search is written with its error, then the error goes on; so is a cancelled one,
    which must not read as a search that found nothing. `record=False` measures
    without writing: a replay (`gaps.replay`) is a check, not a search anyone asked for. The
    session id is checked first, because a search recorded under it creates the session.
    """
    if session_id is not None:
        session.checked(session_id)
    capture = Capture(
        tool=tool,
        session_id=session_id,
        asked=[Asked(question) for question in questions],
        context=context,
    )
    token = _capture.set(capture)
    started = time.perf_counter()
    try:
        yield capture
    except BaseException as exc:
        capture.error = home.scrub(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _capture.reset(token)
        if record:
            # a cancelled search is still written: the cancellation must not cut the write short
            with anyio.CancelScope(shield=True):
                await _write(capture, int((time.perf_counter() - started) * 1000))


def observe_scope(
    where: "Plan | None", collections: list[str], mode: SearchMode, limit: int
) -> None:
    """Where the search looks and how, once it is settled, and with which models when it ran a
    plan (`retrieval.plan`); a full-text listing runs none. A no-op outside a capture."""
    capture = _capture.get()
    if capture is None:
        return
    capture.collections = collections
    capture.mode = mode
    capture.result_limit = limit
    if where is None:
        return
    capture.scoped = where.scope.narrows
    if where.vector is not None and where.embedding is not None:
        capture.embedding = where.embedding.profile or None
    if where.settings.reranker != Reranker.NONE:
        capture.reranker = where.settings.reranker_model
        capture.min_rerank_score = where.settings.min_rerank_score


def observe_ranking(question: str, where: "Plan", pool: "Pool") -> None:
    """What one question's ranking measured, once it is reranked: its query vector and its score
    profile, the best cosines to the rows read and the reranker's best scores before its floor.

    The cosine is measured on the vectors rather than read off a score column: a hybrid query's
    fusion keeps only a rank score, which says nothing about how close the best row came.
    """
    capture = _capture.get()
    if capture is None:
        return
    asked = next((one for one in capture.asked if one.question == question), None)
    if asked is None:
        return
    asked.rerank_scores = pool.rerank_scores[:PROFILE]
    if where.vector is not None:
        asked.vector = where.vector
        stored = [row["vector"] for _, row in pool.rows.values() if row.get("vector") is not None]
        asked.similarities = similarities(where.vector, stored)
    asked.__post_init__()  # the heads of the lists just set


def _vector_bytes(vector: list[float] | None) -> bytes | None:
    return None if vector is None else np.asarray(vector, np.float32).tobytes()


def _floats(values: list[float]) -> bytes | None:
    return _vector_bytes(values) if values else None


def _unfloats(raw: bytes | None) -> list[float]:
    return [] if raw is None else np.frombuffer(raw, np.float32).tolist()


async def _write(capture: Capture, duration_ms: int) -> None:
    """One `searches` row, its `search_questions` and its `search_results`, in one transaction."""
    actor, _ = audit.request_context()
    facts = {k: v for k, v in msgspec.structs.asdict(capture).items() if k in searches.c}
    async with db.connect() as conn:
        if capture.session_id is not None:
            await conn.execute(session.create_session(capture.session_id))
        written = await conn.execute(
            insert(searches).values(
                {
                    **facts,
                    "ts": time.time(),
                    "actor": actor,
                    "collections": db.dumps(capture.collections),
                    "missing_terms": db.dumps(capture.missing_terms),
                    "duration_ms": duration_ms,
                }
            )
        )
        (search_id,) = written.inserted_primary_key or (None,)
        await conn.execute(
            insert(search_questions),
            [
                {
                    "search_id": search_id,
                    "position": position,
                    "question": one.question,
                    "query_vector": _vector_bytes(one.vector),
                    "similarities": _floats(one.similarities),
                    "rerank_scores": _floats(one.rerank_scores),
                    "uncovered": one.uncovered,
                }
                for position, one in enumerate(capture.asked)
            ],
        )
        if capture.results:
            await conn.execute(
                insert(search_results),
                [
                    {"search_id": search_id, **msgspec.structs.asdict(result)}
                    for result in capture.results
                ],
            )


# --- reading the log ----------------------------------------------------------------


class Logged(Searched, kw_only=True):
    """One search as the log holds it, with each question it asked."""

    id: int
    ts: float  # unix seconds
    actor: str  # who ran it: "mcp" for an agent, "web" for the UI or a script
    duration_ms: int
    questions: list[LoggedQuestion] = []


class LoggedSearch(Logged, kw_only=True):
    """One search as a caller reads it: its questions, and the results it returned."""

    results: list[LoggedResult] = []  # best first, without the places folded into them


# the columns a `LoggedQuestion` is read from, labelled apart from the search's own; its best
# scores are derived, not stored
_QUESTION = tuple(
    search_questions.c[name].label(f"question_{name}")
    for name in LoggedQuestion.__struct_fields__
    if name in search_questions.c
)


async def load(
    *,
    since: float | None = None,
    session_id: str | None = None,
    question_ids: Sequence[int] | None = None,
    limit: int | None = None,
) -> list[Logged]:
    """The searches matching every filter given, newest first, with every question each asked,
    in the order asked, and without their results or query vectors (`vectors`). `question_ids`
    keeps the searches that asked one of them."""
    statement = select(searches).order_by(searches.c.ts.desc(), searches.c.id.desc())
    if since is not None:
        statement = statement.where(searches.c.ts >= since)
    if session_id is not None:
        statement = statement.where(searches.c.session_id == session_id)
    if question_ids is not None:
        asked = select(search_questions.c.search_id).where(search_questions.c.id.in_(question_ids))
        statement = statement.where(searches.c.id.in_(asked))
    if limit is not None:
        statement = statement.limit(limit)
    # one statement, so the searches and their questions are read from one snapshot
    chosen = statement.subquery()
    joined = (
        select(chosen, *_QUESTION)
        .outerjoin(search_questions, search_questions.c.search_id == chosen.c.id)
        .order_by(chosen.c.ts.desc(), chosen.c.id.desc(), search_questions.c.position)
    )
    found: dict[int, Logged] = {}
    async with db.read() as conn:
        for row in await conn.execute(joined):
            logged = found.get(row.id)
            if logged is None:
                logged = found[row.id] = db.row_to(
                    Logged, row, collections=list[str], missing_terms=list[str]
                )
            record = {
                column.name.removeprefix("question_"): row._mapping[column.name]
                for column in _QUESTION
            }
            if record["question"] is None:  # a search that asked nothing
                continue
            record["similarities"] = _unfloats(record["similarities"])
            record["rerank_scores"] = _unfloats(record["rerank_scores"])
            logged.questions.append(msgspec.convert(record, LoggedQuestion, strict=False))
    return list(found.values())


async def vectors(question_ids: Sequence[int]) -> dict[int, np.ndarray]:
    """The query vector of each of these questions that has one, read straight from its bytes."""
    async with db.read() as conn:
        rows = await conn.execute(
            select(search_questions.c.id, search_questions.c.query_vector).where(
                search_questions.c.id.in_(question_ids),
                search_questions.c.query_vector.is_not(None),
            )
        )
        return {id: np.frombuffer(raw, np.float32) for id, raw in rows}


async def top_results(
    search_ids: Sequence[int], per_search: int | None = None
) -> dict[int, list[LoggedResult]]:
    """The first `per_search` results of each search, or all of them, best first, without the
    places folded into them."""
    found: dict[int, list[LoggedResult]] = {search_id: [] for search_id in search_ids}
    if not search_ids:
        return found
    ranked = select(
        search_results,
        func.row_number()
        .over(partition_by=search_results.c.search_id, order_by=search_results.c.position)
        .label("rank"),
    ).where(search_results.c.search_id.in_(search_ids), search_results.c.parent.is_(None))
    kept = ranked.subquery()
    statement = select(kept.c.search_id, *(kept.c[name] for name in LoggedResult.__struct_fields__))
    if per_search is not None:
        statement = statement.where(kept.c.rank <= per_search)
    async with db.read() as conn:
        for row in await conn.execute(statement.order_by(kept.c.search_id, kept.c.position)):
            found[row.search_id].append(db.row_to(LoggedResult, row))
    return found


async def listed(since: float, session_id: str | None, limit: int) -> list[LoggedSearch]:
    """The `limit` newest searches since `since`, of one session if named, as a caller reads
    them: their questions, without query vectors, and their results."""
    found = await load(since=since, session_id=session_id, limit=limit)
    results = await top_results([search.id for search in found])
    return [
        LoggedSearch(**msgspec.structs.asdict(search), results=results[search.id])
        for search in found
    ]


async def history(session_id: str, limit: int) -> list[session.SessionEvent]:
    """A session's newest searches as its history shows them: the questions joined as the
    subject, and the distinct documents of the results, best first."""
    found = await load(session_id=session_id, limit=limit)
    results = await top_results([search.id for search in found])
    return [
        session.SessionEvent(
            ts=search.ts,
            action=session.Action.SEARCH,
            subject=" | ".join(one.question for one in search.questions),
            detail=session.EventDetail(
                scope=search.tool,
                hits=search.result_count,
                documents=list(dict.fromkeys(result.document for result in results[search.id])),
                error=search.error,
                questions=[one.question for one in search.questions]
                if len(search.questions) > 1
                else None,
                context=search.context,
            ),
            operation_id=None,
            duration_ms=search.duration_ms,
        )
        for search in found
    ]


async def last_searched() -> dict[str, float]:
    """When each session last searched, in unix seconds; a session that never did is absent."""
    async with db.read() as conn:
        rows = await conn.execute(
            select(searches.c.session_id, func.max(searches.c.ts))
            .where(searches.c.session_id.is_not(None))
            .group_by(searches.c.session_id)
        )
        return dict(rows.tuples().all())


async def recent_questions(limit: int) -> list[str]:
    """The `limit` distinct questions this home asked most recently, newest first: each question
    of a search of several counts as one."""
    asked_at = func.max(searches.c.ts)
    async with db.read() as conn:
        rows = await conn.scalars(
            select(search_questions.c.question)
            .join(searches, searches.c.id == search_questions.c.search_id)
            .where(func.trim(search_questions.c.question) != "")
            .group_by(search_questions.c.question)
            .order_by(asked_at.desc())
            .limit(limit)
        )
        return list(rows)


class SearchAt(msgspec.Struct):
    """One search, as a point on a trend: when, and which session ran it (None: no session)."""

    ts: float
    session_id: str | None


async def searches_since(cutoff: float) -> list[SearchAt]:
    """Every search on or after `cutoff`, oldest first. Raw points, not buckets: the reader
    buckets them by its own day boundaries, which the server does not know."""
    async with db.read() as conn:
        rows = await conn.execute(
            select(searches.c.ts, searches.c.session_id)
            .where(searches.c.ts >= cutoff)
            .order_by(searches.c.ts, searches.c.id)
        )
        return [SearchAt(ts, session_id) for ts, session_id in rows]


async def prune(days: int, now: float | None = None) -> int:
    """Delete the searches older than `days` days, with their questions and results; 0 keeps
    everything. Returns how many searches went."""
    if days == 0:
        return 0
    cutoff = (time.time() if now is None else now) - days * 86400
    async with db.connect() as conn:
        deleted = await conn.execute(delete(searches).where(searches.c.ts < cutoff))
        return deleted.rowcount
