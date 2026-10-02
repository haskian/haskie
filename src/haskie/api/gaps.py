"""Gaps: the questions the collections do not answer, for the curator and for an agent."""

from typing import Annotated

import msgspec
from litestar import get, post, put

from haskie import audit
from haskie.api.common import Days, RequiredSessionId, days_ago
from haskie.paging import one_of
from haskie.search import gaps


class GapReview(msgspec.Struct):
    """The curator's decision on some gap questions; `open` takes a decision back."""

    ids: list[int]
    review: gaps.Review


class GapReplay(msgspec.Struct):
    ids: list[int]


class GapReport(msgspec.Struct):
    """An agent's verdict on what a search it just ran gave it."""

    question: str  # as it was asked: one `q` of `search_excerpts`, or `search_sections`' `q`
    verdict: gaps.Verdict
    missing: str | None = None  # what the excerpts lacked


@get("/api/gaps", mcp_tool="list_gaps")
async def list_gaps(
    review: Annotated[gaps.Review, one_of(gaps.Review)] = gaps.Review.OPEN,
    days: Days = 30,
    signals: Annotated[list[gaps.Signal] | None, one_of(gaps.Signal)] = None,
) -> list[gaps.GapTopic]:
    """The questions the collections could not answer, grouped by topic, the most asked first:
    what to add to the collections next.

    Each question says why (`signal`): `reported`, you or another agent said the excerpts did not
    answer it; `empty`, its search returned nothing; `uncovered`, asked with others, no excerpt
    answers it; `weak`, its best match is under the bar its models set; `borderline`, maybe
    answered, its best match between the bars, left out unless `signals` asks for it.
    `near_misses` cites what came closest. A topic is one thing asked, in one wording or several,
    in any session. To close a gap, add a document that answers it (`add_document`,
    `add_document_to_collection`), check with `replay_gaps`, then `review_gaps` with `resolved`.

    Args:
        review: open (the default), or the ones already dismissed or resolved.
        days: How far back, in days.
        signals: Which reasons to list; every one but `borderline` by default.
    """
    wanted = frozenset(signals) if signals else gaps.CONFIRMED
    return await gaps.load(days_ago(days), review, wanted)


@put("/api/gaps/review", mcp_tool="review_gaps")
@audit.audited("gaps.review")
async def review_gaps(data: GapReview) -> int:
    """Record a decision on gap questions, by their `id` from `list_gaps`: `dismissed` (the
    collections are not meant to answer it), `resolved` (a document now answers it) or `open`
    (take the decision back). Returns how many questions it reached."""
    reviewed = await gaps.review(data.ids, data.review)
    audit.attach(review=data.review, questions=reviewed)
    return reviewed


@post("/api/gaps/replay", status_code=200, mcp_tool="replay_gaps")
async def replay_gaps(data: GapReplay) -> list[gaps.ReplayedGap]:
    """Ask gap questions again, by their `id` from `list_gaps` (at most 50), each on its own over
    every collection, and judge each again: `signal` null means the collections answer it now,
    and `results` cites where. Not recorded as searches."""
    return await gaps.replay(data.ids)


@post("/api/gaps/report", status_code=200, mcp_tool="report_gap")
@audit.audited("gaps.report")
async def report_gap(session_id: RequiredSessionId, data: GapReport) -> gaps.Reported:
    """Say that a search you just ran did not answer your question: the gap then shows to the
    user as one the collections should close.

    Call it when the excerpts do not let a careful reader answer the question from them alone
    (`insufficient`), or answer only part of it (`partial`). Not when they answer it in other
    words, and not for a question you did not search. `question` is one of the questions you
    passed to `search_excerpts` (or `search_sections`) with this `session_id`, word for word, in
    the last hour. `missing` says in a sentence what the excerpts lacked, at most 300 characters.
    One call per miss; reporting again replaces the verdict.

    Args:
        session_id: The conversation's id, the one the search ran with.
    """
    reported = await gaps.report(session_id, data.question, data.verdict, data.missing)
    audit.attach(session_id=session_id, verdict=reported.verdict)
    return reported
