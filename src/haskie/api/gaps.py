"""Gaps: the questions the collections do not answer, for the curator and for an agent."""

import msgspec
from litestar import get, post, put

from haskie import audit
from haskie.api.common import days_ago
from haskie.search import gaps


class GapReview(msgspec.Struct):
    """The curator's decision on some gap questions; `open` takes a decision back."""

    ids: list[int]
    review: gaps.Review


class GapReplay(msgspec.Struct):
    ids: list[int]


@get("/api/gaps", mcp_tool="list_gaps")
async def list_gaps(review: gaps.Review = gaps.Review.OPEN, days: int = 30) -> list[gaps.GapTopic]:
    """The questions the collections could not answer, grouped by topic, the most asked first:
    what to add to the collections next.

    Each question says why (`signal`): `empty`, its search returned nothing; `uncovered`, asked
    with others, no excerpt answers it; `weak`, its best match is under the bar its models set.
    `near_misses` cites what came closest. A topic is one thing asked, in one wording or several,
    in any session. To close a gap, add a document that answers it (`add_document`,
    `add_document_to_collection`), check with `replay_gaps`, then `review_gaps` with `resolved`.

    Args:
        review: open (the default), or the ones already dismissed or resolved.
        days: How far back, 1 to 366.
    """
    return await gaps.load(days_ago(days), review)


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
