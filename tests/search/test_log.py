"""The search log: what a search records about itself, through the real flow over real indexes.

The embedding and the cross-encoder are replaced by fixed numbers, so what the log measured can
be computed here and compared exactly: per question, the best cosine between its query and any
row read, and the reranker's best score.
"""

import asyncio
import math
import time

import anyio
import numpy as np
import pytest
from conftest import one_part
from sqlalchemy import func, select

from haskie import db, home
from haskie.catalogue import catalogue
from haskie.collection.collection import Collection
from haskie.collection.index import Row
from haskie.errors import InvalidInput
from haskie.indexing import chunk, embed, models
from haskie.search import aspects, flow, log, retrieval
from haskie.settings import (
    DEFAULT_RERANKER,
    ChunkSettings,
    Reranker,
    SearchMode,
    SearchSettings,
    UserSettings,
    save_user_settings,
)
from haskie.tables import search_questions, search_results

pytestmark = pytest.mark.anyio

BODIES = {
    "a": "# Retries\n\nA retried call has to be idempotent, or it happens twice.\n",
    "b": "# Sourdough\n\nFeed the starter twice a day and keep it warm.\n",
}
DIMS = 384


def _unit(values: list[float]) -> list[float]:
    array = np.asarray(values + [0.0] * (DIMS - len(values)))
    return (array / np.linalg.norm(array)).tolist()


# the chunks' vectors, and one query near each of them
VECTORS = {"a": _unit([1.0, 0.5]), "b": _unit([0.0, 0.2, 1.0])}
QUERIES = {"idempotent retries": _unit([1.0]), "sourdough starter": _unit([0.0, 0.0, 1.0, 0.3])}


async def _two_collections() -> None:
    """Collections `a` and `b`, one chunk each with a vector of its own, and the markdown the
    chunks were cut from, which an excerpt is read out of."""
    compact = await catalogue.embedding_model(UserSettings(embedding="compact"))
    for name, body in BODIES.items():
        collection = await Collection.create(name)
        (chunk_,) = chunk.split(body, ChunkSettings())
        markdown = home.HOME / "documents" / f"{name}.md.md"
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(body)
        index = collection.index_with(compact)
        row = Row(chunk=chunk_, vector=VECTORS[name], seq=1)
        await index.add_parts(
            f"{name}.md", f"documents/{name}.md", f"documents/{name}.md.md", one_part(0, [row])
        )
        await index.finish()


@pytest.fixture
def fixed_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each query embeds to its fixed vector, every model is ready, and the cross-encoder scores
    the texts it is given 2.5, 1.5, 0.5... in order, as logits."""

    async def require_ready(kind: str, model: str) -> None:
        return None

    def rerank(model: str, accelerator: str, text: str, texts: list[str]) -> list[float]:
        return [2.5 - i for i in range(len(texts))]

    def embed_query(model: object, text: str) -> list[float]:
        # an excerpts search embeds its context in front of the question (`aspects.framed`)
        return QUERIES[text.rsplit("\n\n", 1)[-1]]

    monkeypatch.setattr(retrieval, "embed_query", embed_query)
    monkeypatch.setattr(models, "require_ready", require_ready)
    monkeypatch.setattr(embed, "rerank_scores", rerank)


def _best_cosine(question: str) -> float:
    return max(float(np.dot(QUERIES[question], vector)) for vector in VECTORS.values())


async def test_a_search_records_its_scope_answer_and_signals(seeded_home, fixed_models) -> None:
    await save_user_settings(
        UserSettings(
            embedding="compact",
            search=SearchSettings(mode=SearchMode.VECTOR, reranker=Reranker.CROSS_ENCODER),
        )
    )
    await _two_collections()

    async with log.capturing(log.Tool.EXPLORE, ["idempotent retries"], "claude-code a3f9") as c:
        found = await flow.chunks(["a", "b"], "idempotent retries", 5)
        c.answer(found)

    (logged,) = await log.load(session_id="claude-code a3f9")
    assert (logged.tool, logged.actor, logged.context) == (log.Tool.EXPLORE, "web", None)
    assert (logged.collections, logged.mode, logged.result_limit) == (
        ["a", "b"],
        SearchMode.VECTOR,
        5,
    )
    assert (logged.embedding, logged.reranker) == ("compact", DEFAULT_RERANKER)
    assert (logged.result_count, logged.error) == (len(found), None)
    assert logged.duration_ms >= 0 and logged.ts <= time.time()
    (asked,) = logged.questions
    assert (asked.question, asked.uncovered, asked.review) == ("idempotent retries", False, None)
    assert asked.best_similarity == pytest.approx(_best_cosine("idempotent retries"))
    assert asked.best_rerank == pytest.approx(1 / (1 + math.exp(-2.5))), "the sigmoid of 2.5"
    cosines = sorted(float(np.dot(QUERIES["idempotent retries"], v)) for v in VECTORS.values())
    assert asked.similarities == pytest.approx(cosines[::-1], abs=1e-6), "both rows, best first"
    assert asked.rerank_scores == pytest.approx([1 / (1 + math.exp(-x)) for x in (2.5, 1.5)])
    assert asked.id is not None
    stored = (await log.vectors([asked.id]))[asked.id]
    assert stored == pytest.approx(QUERIES["idempotent retries"], abs=1e-6), "float32, read apart"
    assert len(found) == 2, "both chunks, so a cap of one has something to cut"
    assert [r.position for r in (await log.top_results([logged.id], 1))[logged.id]] == [0]
    top = (await log.top_results([logged.id], 5))[logged.id]
    assert [(r.document, r.seq_start, r.location) for r in top] == [
        (one.document, one.seq, one.location) for one in found
    ]


async def test_an_excerpts_search_records_each_question_on_its_own(
    seeded_home, fixed_models
) -> None:
    """Several questions: each runs its own ranking, and each is measured by it; the questions
    no excerpt answers are marked. The reranker's floor drops every chunk here, which is what
    leaves both uncovered, yet each keeps the best score it had before the floor."""
    await save_user_settings(
        UserSettings(
            embedding="compact",
            search=SearchSettings(
                mode=SearchMode.VECTOR, reranker=Reranker.CROSS_ENCODER, min_rerank_score=0.99
            ),
        )
    )
    await _two_collections()
    asked = aspects.questions(["idempotent retries", "sourdough starter"], "home cooking")

    async with log.capturing(
        log.Tool.EXCERPTS, asked.questions, "s1", context=asked.context
    ) as capture:
        found = await flow.answers(["a", "b"], asked, 4)
        capture.answer(found.excerpts, found.uncovered, found.missing_terms)

    (logged,) = await log.load()
    assert logged.context == "home cooking"
    assert logged.missing_terms == found.missing_terms, "the words no excerpt held, kept"
    assert found.uncovered == asked.questions, "the floor dropped every chunk"
    assert [(one.question, one.uncovered) for one in logged.questions] == [
        ("idempotent retries", True),
        ("sourdough starter", True),
    ]
    for one in logged.questions:
        assert one.best_similarity == pytest.approx(_best_cosine(one.question)), one.question
        assert one.best_rerank == pytest.approx(1 / (1 + math.exp(-2.5))), one.question
    assert len({one.id for one in logged.questions}) == 2, "each question has its own id"


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        (InvalidInput("limit must be 1..100, got 0"), "InvalidInput: limit must be 1..100, got 0"),
        # a cut-off search found nothing only because it never finished: not an `empty` gap
        (asyncio.CancelledError(), "CancelledError: "),
    ],
    ids=["failed", "cancelled"],
)
async def test_a_search_that_did_not_finish_is_recorded_with_its_error_then_raised(
    seeded_home, failure: BaseException, error: str
) -> None:
    with pytest.raises(type(failure)):
        async with log.capturing(log.Tool.SOURCES, ["kafka"], None):
            raise failure

    (logged,) = await log.load()
    assert logged.error == error
    assert (logged.session_id, logged.result_count) == (None, 0), "recorded without a session too"
    assert [one.question for one in logged.questions] == ["kafka"]


async def test_a_search_cancelled_by_its_scope_is_still_written(seeded_home) -> None:
    """A scope's cancellation reaches every await inside it, the log's write included."""
    with anyio.CancelScope() as scope:
        async with log.capturing(log.Tool.SOURCES, ["kafka"], None):
            scope.cancel()
            await anyio.sleep(1)

    (logged,) = await log.load()
    assert logged.error is not None and logged.error.startswith("CancelledError")


async def test_a_replay_measures_without_writing(seeded_home, fixed_models) -> None:
    await save_user_settings(UserSettings(embedding="compact"))
    await _two_collections()
    async with log.capturing(
        log.Tool.EXPLORE, ["idempotent retries"], None, record=False
    ) as capture:
        capture.answer(await flow.chunks(["a", "b"], "idempotent retries", 5))
    assert capture.asked[0].best_similarity == pytest.approx(_best_cosine("idempotent retries"))
    assert await log.load() == []


async def test_a_question_the_capture_was_not_asked_is_left_out(seeded_home, fixed_models) -> None:
    """A search the capture did not name (a probe's, say) measures nothing into it."""
    await save_user_settings(UserSettings(embedding="compact"))
    await _two_collections()
    async with log.capturing(log.Tool.EXPLORE, ["kafka"], None, record=False) as capture:
        await flow.chunks(["a", "b"], "idempotent retries", 5)
    (kafka,) = capture.asked
    assert (kafka.vector, kafka.best_similarity, kafka.best_rerank) == (None, None, None)


async def test_recent_questions_are_distinct_newest_first(seeded_home) -> None:
    """Each question of a search of several counts on its own, in the order asked; a question
    asked again counts once, where it was asked last."""
    for questions in (["kafka retries"], ["sourdough", "kafka retries"], ["zebra", " "]):
        async with log.capturing(log.Tool.EXCERPTS, questions, None):
            pass

    assert await log.recent_questions(10) == ["zebra", "sourdough", "kafka retries"]
    assert await log.recent_questions(2) == ["zebra", "sourdough"]


async def test_an_invalid_session_id_fails_before_anything_is_written(seeded_home) -> None:
    with pytest.raises(InvalidInput, match="session id"):
        async with log.capturing(log.Tool.EXCERPTS, ["kafka"], "s" * 129):
            pass
    assert await log.load() == []


async def test_prune_deletes_old_searches_with_their_questions_and_results(
    seeded_home, fixed_models
) -> None:
    await save_user_settings(UserSettings(embedding="compact"))
    await _two_collections()
    for question in QUERIES:
        async with log.capturing(log.Tool.EXPLORE, [question], None) as capture:
            capture.answer(await flow.chunks(["a", "b"], question, 5))

    assert await log.prune(0) == 0, "0 keeps everything"
    assert await log.prune(1) == 0, "nothing is a day old yet"
    assert await log.prune(1, now=time.time() + 2 * 86400) == 2
    assert await log.load() == []
    async with db.connect() as conn:
        for table in (search_questions, search_results):
            assert await conn.scalar(select(func.count()).select_from(table)) == 0, table.name


async def test_a_borderline_question_is_listed_only_when_asked_for(
    seeded_home, fixed_models
) -> None:
    """Its best cosine sits between the profile's bars: a maybe, not a gap. The Gaps page asks for
    it; an agent's default list leaves it out."""
    from sqlalchemy import update

    from haskie.search import gaps
    from haskie.tables import embedding_profiles

    async with db.connect() as conn:
        await conn.execute(
            update(embedding_profiles)
            .where(embedding_profiles.c.profile == "compact")
            .values(weak_match=0.5, answered_match=0.95)
        )
    await save_user_settings(UserSettings(embedding="compact"))
    await _two_collections()
    async with log.capturing(log.Tool.EXPLORE, ["idempotent retries"], "s1") as capture:
        capture.answer(await flow.chunks(["a", "b"], "idempotent retries", 5))

    since = time.time() - 60
    assert await gaps.load(since, gaps.Review.OPEN) == [], "a maybe is no confirmed gap"
    (topic,) = await gaps.load(since, gaps.Review.OPEN, frozenset(gaps.Signal))
    (question,) = topic.questions
    assert question.signal == gaps.Signal.BORDERLINE
    assert 0.5 <= (question.best_similarity or 0) < 0.95
