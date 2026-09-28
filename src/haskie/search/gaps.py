"""Gaps: the questions the shelf does not answer, grouped by topic.

The search log (`log.py`) stores what every search measured, per question. This module judges
those measurements on read, so a bar measured again re-judges every stored search.

A question is a gap when one of the detectors fires (`DETECTORS`), first one wins:

- `reported`: the agent that asked it said the excerpts do not answer it (`report`). It reads
  the excerpts, so its verdict outranks every score.
- `empty`: its search returned nothing.
- `uncovered`: several questions were asked at once, and no excerpt answers this one.
- `weak`: something came back, but the question's best match is under the bar. A reranked search
  is judged by the floor it dropped chunks under: the settings' `min_rerank_score` when set, else
  the reranker's calibrated floor (`catalogue.calibration`), since it reads query and passage
  together. Otherwise the embedding profile's cosine bar (`weak_match`) decides. With no bar
  known there is no verdict.
- `borderline`: no reranker judged it, and its best cosine sits between `weak_match` and
  `answered_match`: maybe answered. The page shows these apart; an agent asks for them.

A failed search is an error, not a gap, and is left out.

Gap questions are grouped into topics by leader clustering, newest first, like `collapse` folds
results: each is compared with the topics kept so far and joins the closest one it matches. Two
questions match when their query vectors, embedded under one profile that has a `same_topic`
bar, clear it; without vectors, when they are the same words.

A new signal is one more `Signal` and one more detector.
"""

import asyncio
import time
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Protocol

import msgspec
import numpy as np
from sqlalchemy import select, update

from haskie import cpu, db
from haskie.catalogue import catalogue
from haskie.collection.collection import Collection
from haskie.errors import Conflict, InvalidInput, NotFound
from haskie.search import aspects, collapse, flow, log, retrieval, session
from haskie.search.collapse import WORD
from haskie.search.log import LoggedQuestion, LoggedResult
from haskie.tables import search_questions, searches

NEAR_MISSES = 3  # results shown per gap question: what came closest, not a page to read
MAX_REPLAY = 50  # questions one replay asks again; each embeds and searches again


class Signal(StrEnum):
    """Why a question counts as a gap."""

    REPORTED = "reported"
    EMPTY = "empty"
    UNCOVERED = "uncovered"
    WEAK = "weak"
    BORDERLINE = "borderline"  # between the bars: maybe answered; shown apart, opt-in for agents


class Verdict(StrEnum):
    """What an agent says the excerpts of a search gave it (`report`)."""

    INSUFFICIENT = "insufficient"  # a careful reader could not answer from them alone
    PARTIAL = "partial"  # they answer part of the question, not all of it


class Review(StrEnum):
    """What the curator decided about a gap. `open` is stored as no decision."""

    OPEN = "open"
    DISMISSED = "dismissed"  # out of scope: the shelf is not meant to answer it
    RESOLVED = "resolved"  # a document was added that answers it


class Bars(msgspec.Struct, frozen=True):
    """What questions are judged by, keyed as a search names its models."""

    weak_match: dict[str, float]  # per embedding profile
    answered_match: dict[str, float]  # per embedding profile: the top of the borderline band
    same_topic: dict[str, float]  # per embedding profile
    floor: dict[str, float]  # per reranker model: its calibrated floor


async def bars(rerankers: set[str]) -> Bars:
    """The profiles' bars, and the floor of each of `rerankers`."""
    profiles = await catalogue.embedders()
    names = sorted(rerankers)
    floors = await asyncio.gather(*(catalogue.calibration(name) for name in names))
    return Bars(
        weak_match={key: m.weak_match for key, m in profiles.items() if m.weak_match is not None},
        answered_match={
            key: m.answered_match for key, m in profiles.items() if m.answered_match is not None
        },
        same_topic={key: m.same_topic for key, m in profiles.items() if m.same_topic is not None},
        floor={name: one.floor for name, one in zip(names, floors, strict=True)},
    )


# --- judging one question ---------------------------------------------------------


class Searched(Protocol):
    """What a detector reads of the search a question was asked in: a logged search, or a
    capture that was never written (a replay)."""

    @property
    def result_count(self) -> int: ...
    @property
    def embedding(self) -> str | None: ...
    @property
    def reranker(self) -> str | None: ...
    @property
    def min_rerank_score(self) -> float | None: ...
    @property
    def error(self) -> str | None: ...


def _reported(_: Searched, asked: LoggedQuestion, __: Bars) -> Signal | None:
    return Signal.REPORTED if asked.agent_verdict is not None else None


def _empty(search: Searched, _: LoggedQuestion, __: Bars) -> Signal | None:
    return Signal.EMPTY if search.result_count == 0 else None


def _uncovered(_: Searched, asked: LoggedQuestion, __: Bars) -> Signal | None:
    return Signal.UNCOVERED if asked.uncovered else None


def _weak(search: Searched, asked: LoggedQuestion, bars: Bars) -> Signal | None:
    if asked.best_rerank is not None and search.reranker in bars.floor:
        # the floor the search itself dropped chunks under
        floor = retrieval.rerank_floor(search.min_rerank_score, bars.floor[search.reranker])
        return Signal.WEAK if asked.best_rerank < floor else None
    profile = search.embedding
    low = bars.weak_match.get(profile) if profile else None
    if asked.best_similarity is None or low is None:
        return None
    if asked.best_similarity < low:
        return Signal.WEAK
    high = bars.answered_match.get(profile) if profile else None
    # the reranker's floor gives no band: measured, no bar kept its false gaps under 15%
    return Signal.BORDERLINE if high is not None and asked.best_similarity < high else None


Detector = Callable[[Searched, LoggedQuestion, Bars], Signal | None]
DETECTORS: tuple[Detector, ...] = (_reported, _empty, _uncovered, _weak)


def signal(search: Searched, asked: LoggedQuestion, bars: Bars) -> Signal | None:
    """Why `asked`, one question of `search`, is a gap, or None when it is not one."""
    if search.error is not None:
        return None
    return next((found for detect in DETECTORS if (found := detect(search, asked, bars))), None)


# --- grouping gaps into topics ----------------------------------------------------


class GapQuestion(msgspec.Struct):
    """One question that found no answer, the search it was asked in, and what came closest."""

    id: int  # the question's, in the search log
    search_id: int
    ts: float
    session_id: str | None
    actor: str
    tool: log.Tool
    question: str
    context: str | None
    collections: list[str]
    signal: Signal
    result_count: int
    best_similarity: float | None
    best_rerank: float | None
    near_misses: list[LoggedResult]  # its search's best results, at most `NEAR_MISSES`
    agent_verdict: Verdict | None = None  # what the agent said, when it reported the gap
    agent_note: str | None = None


class GapTopic(msgspec.Struct):
    """Questions that asked about one thing and found no answer, newest first."""

    question: str  # the newest one's: what the topic is called
    questions: list[GapQuestion]
    sessions: int  # distinct sessions among them; questions asked without one are not counted
    first_at: float
    last_at: float
    collections: list[str]  # every collection they searched, in name order


class Gap(msgspec.Struct):
    """One gap question with the search it came from, and its query vector, before it is
    listed."""

    search: log.Logged
    asked: LoggedQuestion
    signal: Signal
    vector: np.ndarray | None = None


def _words(question: str) -> str:
    """A question as its words, lowercased: what two questions without vectors are matched by."""
    return " ".join(WORD.findall(question.lower()))


def _units(gaps: Sequence[Gap], bars: Bars) -> list[np.ndarray | None]:
    """Each question's query vector at unit length, normalised once per profile, or None where
    its profile has no `same_topic` bar: those are matched by their words alone."""
    by_profile: dict[str, list[int]] = {}
    for at, gap in enumerate(gaps):
        profile = gap.search.embedding
        if gap.vector is not None and profile in bars.same_topic:
            by_profile.setdefault(profile, []).append(at)
    units: list[np.ndarray | None] = [None] * len(gaps)
    for members in by_profile.values():
        rows = collapse.unit_rows([gaps[at].vector for at in members])
        for at, row in zip(members, rows, strict=True):
            units[at] = row
    return units


def topics(
    gaps: Sequence[Gap], near_misses: dict[int, list[LoggedResult]], bars: Bars
) -> list[GapTopic]:
    """Group gap questions into topics, the most asked first, then the most recent.

    `gaps` comes newest first, so each topic's leader, the question the others are compared
    with, is its newest. Comparing with leaders only keeps a chain of near matches from merging
    two topics that do not match each other (as in `collapse`). A question joins the closest
    leader it matches: by cosine when both were embedded under one profile with a bar, else by
    the same words (a match of 1.0). The cosines against every leader of a profile are one matrix
    product, since a page of gaps compares each question with every topic so far.

    CPU work that grows with gaps times topics: `load` runs it in a worker thread.
    """
    words = [_words(gap.asked.question) for gap in gaps]
    units = _units(gaps, bars)
    groups: list[list[int]] = []
    by_words: dict[str, list[int]] = {}  # the groups whose leader has these words
    # per profile, the unit vectors of the leaders that have one, and the group each leads
    rows = {
        gaps[at].search.embedding: np.empty((len(gaps), unit.shape[0]))
        for at, unit in enumerate(units)
        if unit is not None
    }
    led: dict[str | None, list[int]] = {profile: [] for profile in rows}
    for at, gap in enumerate(gaps):
        unit, profile = units[at], gap.search.embedding
        candidates: list[tuple[float, int]] = []
        # a unit vector means its profile has a bar (`_units`)
        if unit is not None and profile is not None and led[profile]:
            cosines = rows[profile][: len(led[profile])] @ unit
            best = int(np.argmax(cosines))
            if cosines[best] > bars.same_topic[profile]:
                candidates.append((float(cosines[best]), led[profile][best]))
        for group in by_words.get(words[at], []):
            leader = groups[group][0]
            if unit is None or units[leader] is None or gaps[leader].search.embedding != profile:
                candidates.append((1.0, group))
        if candidates:
            # the closest, and the earliest topic of equally close ones
            groups[max(candidates, key=lambda one: (one[0], -one[1]))[1]].append(at)
            continue
        group = len(groups)
        groups.append([at])
        by_words.setdefault(words[at], []).append(group)
        if unit is not None:
            rows[profile][len(led[profile])] = unit
            led[profile].append(group)
    built = [_topic([gaps[at] for at in members], near_misses) for members in groups]
    return sorted(built, key=lambda topic: (-len(topic.questions), -topic.last_at))


def _listed(gap: Gap, near_misses: dict[int, list[LoggedResult]]) -> GapQuestion:
    search, asked = gap.search, gap.asked
    assert asked.id is not None  # read out of the log
    return GapQuestion(
        id=asked.id,
        search_id=search.id,
        ts=search.ts,
        session_id=search.session_id,
        actor=search.actor,
        tool=search.tool,
        question=asked.question,
        context=search.context,
        collections=search.collections,
        signal=gap.signal,
        result_count=search.result_count,
        best_similarity=asked.best_similarity,
        best_rerank=asked.best_rerank,
        near_misses=near_misses.get(search.id, []),
        agent_verdict=Verdict(asked.agent_verdict) if asked.agent_verdict else None,
        agent_note=asked.agent_note,
    )


def _topic(group: list[Gap], near_misses: dict[int, list[LoggedResult]]) -> GapTopic:
    listed = [_listed(gap, near_misses) for gap in group]
    return GapTopic(
        question=listed[0].question,
        questions=listed,
        sessions=len({one.session_id for one in listed if one.session_id is not None}),
        first_at=min(one.ts for one in listed),
        last_at=max(one.ts for one in listed),
        collections=sorted({name for one in listed for name in one.collections}),
    )


# --- what the Gaps page calls -----------------------------------------------------


def _stored(review: Review) -> str | None:
    """A review as the log stores it: `open` is no decision at all."""
    return None if review == Review.OPEN else review.value


# what `list_gaps` returns unless asked for more: the gaps, not the maybes
CONFIRMED = frozenset(Signal) - {Signal.BORDERLINE}


async def load(
    since: float, review: Review, signals: frozenset[Signal] = CONFIRMED
) -> list[GapTopic]:
    """The gap topics among the questions asked since `since` that carry `review`, of the
    `signals` asked for."""
    stored = _stored(review)
    searched = await log.load(since=since)
    judged = await bars({one.reranker for one in searched if one.reranker})
    gaps = [
        Gap(search, asked, found)
        for search in searched
        for asked in search.questions
        if asked.review == stored and (found := signal(search, asked, judged)) in signals
    ]
    # the vectors of the gaps only: most questions are answered, and a vector is kilobytes
    vectors = await log.vectors([gap.asked.id for gap in gaps if gap.asked.id is not None])
    for gap in gaps:
        gap.vector = vectors.get(gap.asked.id) if gap.asked.id is not None else None
    near = await log.top_results(sorted({gap.search.id for gap in gaps}), NEAR_MISSES)
    return await cpu.on_cpu(topics, gaps, near, judged)


async def review(ids: list[int], decision: Review) -> int:
    """Record the curator's decision on these questions; `open` takes it back. Returns how many
    questions it reached."""
    async with db.connect() as conn:
        done = await conn.execute(
            update(search_questions)
            .where(search_questions.c.id.in_(ids))
            .values(review=_stored(decision))
        )
        return done.rowcount


REPORT_WINDOW = 3600  # seconds after a search in which its agent may still judge it
MAX_NOTE = 300  # what the excerpts lacked, in a sentence or two


class Reported(msgspec.Struct):
    """The logged question an agent's verdict was recorded on."""

    id: int  # the question's, in the search log, as `list_gaps` lists it
    question: str
    verdict: Verdict


async def report(
    session_id: str, question: str, verdict: Verdict, note: str | None = None
) -> Reported:
    """Record an agent's verdict on the question it asked last in this session with these words:
    the excerpts did not let it answer (`insufficient`), or answered only part (`partial`).

    Only a question of this session's last hour, and only one whose search ran: a verdict judges
    what a search returned, so without the search there is nothing to judge. Reporting again
    replaces the verdict.
    """
    session.checked(session_id)
    asked = question.strip()
    note = (note or "").strip() or None
    if not asked:
        raise InvalidInput("question is empty")
    if note is not None and len(note) > MAX_NOTE:
        raise InvalidInput(f"missing is at most {MAX_NOTE} characters, got {len(note)}")
    since = time.time() - REPORT_WINDOW
    async with db.connect() as conn:
        found = (
            await conn.execute(
                select(search_questions.c.id, searches.c.error)
                .join(searches, searches.c.id == search_questions.c.search_id)
                .where(
                    searches.c.session_id == session_id,
                    searches.c.ts >= since,
                    search_questions.c.question == asked,
                )
                .order_by(searches.c.ts.desc(), searches.c.id.desc())
                .limit(1)
            )
        ).first()
        if found is None:
            raise NotFound(
                f"no search in session {session_id} asked this in the last hour: {asked[:60]}"
            )
        if found.error is not None:
            raise Conflict(f"that search failed, so it returned nothing to judge: {found.error}")
        await conn.execute(
            update(search_questions)
            .where(search_questions.c.id == found.id)
            .values(agent_verdict=verdict.value, agent_note=note)
        )
    return Reported(id=found.id, question=asked, verdict=verdict)


class ReplayedGap(msgspec.Struct):
    """One gap question asked again, now: whether the shelf answers it yet."""

    id: int
    question: str
    signal: Signal | None  # None: it is answered now
    result_count: int
    best_similarity: float | None
    best_rerank: float | None
    results: list[LoggedResult]  # its best results now, at most `NEAR_MISSES`


async def _ask_again(names: list[str], asked: LoggedQuestion, context: str | None) -> log.Capture:
    """One question asked on its own as `search_excerpts` over `names`, measured, not logged."""
    question = aspects.Questions(questions=[asked.question], context=context)
    async with log.capturing(
        log.Tool.EXCERPTS, question.questions, None, context=context, record=False
    ) as capture:
        found = await flow.answers(names, question, NEAR_MISSES)
        capture.answer(found.excerpts, found.uncovered, found.missing_terms)
    return capture


async def replay(ids: list[int]) -> list[ReplayedGap]:
    """Ask these questions again, each on its own as `search_excerpts` over every collection with
    the context it had, and judge them again.

    Every collection rather than the scope each one had: the question is whether the shelf
    answers it now, and the document that closes a gap is often in a collection the search never
    looked in. The questions run at once, as searches from several agents would. Nothing is
    written to the log: a replay is a check, not a search anyone made.
    """
    if len(ids) > MAX_REPLAY:
        raise InvalidInput(f"at most {MAX_REPLAY} questions per replay, got {len(ids)}")
    wanted = set(ids)
    names = await Collection.names()
    asked = [
        (question, search.context)
        for search in await log.load(question_ids=ids)
        for question in search.questions
        if question.id in wanted
    ]
    captures = await asyncio.gather(*(_ask_again(names, one, context) for one, context in asked))
    judged = await bars({capture.reranker for capture in captures if capture.reranker})
    replayed: list[ReplayedGap] = []
    for (question, _), capture in zip(asked, captures, strict=True):
        assert question.id is not None  # read out of the log
        (again,) = capture.asked
        replayed.append(
            ReplayedGap(
                id=question.id,
                question=question.question,
                signal=signal(capture, again, judged),
                result_count=capture.result_count,
                best_similarity=again.best_similarity,
                best_rerank=again.best_rerank,
                results=[result for result in capture.results if result.parent is None],
            )
        )
    return replayed
