"""The Gaps judgement and grouping, and how the search log flattens an answer: pure functions,
so every branch is a row of a table."""

import msgspec
import numpy as np
import pytest

from haskie import ids
from haskie.collection.index import Hit, Overlap, Overlaps, Relation
from haskie.search import gaps, log
from haskie.search.gaps import Bars, Gap, Signal
from haskie.search.log import Asked, LoggedQuestion
from haskie.search.passage import Excerpt, HotSection, Passage, PassageReference, Source, Span

MINILM = "Xenova/ms-marco-MiniLM-L-6-v2"
BARS = Bars(
    weak_match={"compact": 0.70, "arctic-m": 0.40},
    answered_match={"compact": 0.775},
    same_topic={"compact": 0.70},
    floor={MINILM: 0.05},
)

# one excerpts search under the default profile and reranker, as `log.load` returns it
SEARCH = log.Logged(
    id=1,
    ts=1_790_000_000.0,
    session_id="claude-code a3f9",
    actor="mcp",
    tool=log.Tool.EXCERPTS,
    context="a Python service on Kafka",
    collections=["distributed-systems", "kafka"],
    mode=None,
    embedding="compact",
    reranker=MINILM,
    min_rerank_score=None,
    result_limit=25,
    result_count=25,
    duration_ms=48,
    error=None,
    missing_terms=[],
)
ASKED = Asked(
    question="How should a background job retry a failed Kafka message without duplicates?",
    id=7,
    best_similarity=0.81,
    best_rerank=0.93,
)
NO_RERANK = {"reranker": None}


@pytest.mark.parametrize(
    ("name", "search", "asked", "expected"),
    [
        ("an answered question is no gap", {}, {}, None),
        (
            "the agent's verdict outranks every score",
            {},
            {"agent_verdict": "partial", "best_rerank": 0.99},
            Signal.REPORTED,
        ),
        (
            "a failed search is no gap, whatever the agent said",
            {"error": "NotReady: loading"},
            {"agent_verdict": "insufficient"},
            None,
        ),
        ("its search returned nothing", {"result_count": 0}, {}, Signal.EMPTY),
        (
            "a failed search is an error",
            {"result_count": 0, "error": "NotReady: loading"},
            {},
            None,
        ),
        ("no excerpt answers it", {}, {"uncovered": True}, Signal.UNCOVERED),
        (
            "a scoped search that missed is no gap: the rest of the shelf may answer",
            {"scoped": True, "result_count": 0},
            {"uncovered": True, "best_rerank": 0.01, "best_similarity": 0.1},
            None,
        ),
        (
            "the agent's verdict stands on a scoped search",
            {"scoped": True, "result_count": 0},
            {"agent_verdict": "insufficient"},
            Signal.REPORTED,
        ),
        ("the reranker's best under its floor", {}, {"best_rerank": 0.04}, Signal.WEAK),
        ("exactly at the floor is an answer", {}, {"best_rerank": 0.05}, None),
        (
            "the settings' floor stands in for the reranker's, as it did in the search",
            {"min_rerank_score": 0.01},
            {"best_rerank": 0.03},
            None,
        ),
        (
            "under the settings' floor",
            {"min_rerank_score": 0.5},
            {"best_rerank": 0.3},
            Signal.WEAK,
        ),
        (
            "the reranker outranks the cosine",
            {},
            {"best_rerank": 0.9, "best_similarity": 0.6},
            None,
        ),
        (
            "a reranker the bars do not know leaves it to the cosine",
            {"reranker": "BAAI/bge-reranker-base"},
            {"best_rerank": 0.0, "best_similarity": 0.6},
            Signal.WEAK,
        ),
        ("the cosine under its bar", NO_RERANK, {"best_similarity": 0.692}, Signal.WEAK),
        ("the cosine between the bars", NO_RERANK, {"best_similarity": 0.723}, Signal.BORDERLINE),
        (
            "exactly at the low bar is borderline",
            NO_RERANK,
            {"best_similarity": 0.70},
            Signal.BORDERLINE,
        ),
        ("at the high bar is an answer", NO_RERANK, {"best_similarity": 0.775}, None),
        (
            "a profile with no high bar has no band",
            {**NO_RERANK, "embedding": "arctic-m"},
            {"best_similarity": 0.5},
            None,
        ),
        (
            "a reranked search has no band",
            {},
            {"best_rerank": 0.06, "best_similarity": 0.71},
            None,
        ),
        (
            "a profile without a bar gives no verdict",
            {**NO_RERANK, "embedding": "gte-base"},
            {"best_similarity": 0.1},
            None,
        ),
        (
            "a full-text search has nothing to judge",
            {**NO_RERANK, "embedding": None},
            {"best_similarity": None, "best_rerank": None},
            None,
        ),
    ],
)
def test_signal_judges_each_question(
    name: str, search: dict, asked: dict, expected: Signal | None
) -> None:
    judged = gaps.signal(
        msgspec.structs.replace(SEARCH, **search), msgspec.structs.replace(ASKED, **asked), BARS
    )
    assert judged == expected, name


def _gap(id: int, question: str, ts: float, vector: list[float] | None, **changes: object) -> Gap:
    search = msgspec.structs.replace(SEARCH, id=id, ts=ts, result_count=0, **changes)
    unit = None if vector is None else np.asarray(vector, np.float32)
    return Gap(search, LoggedQuestion(question, id=id * 10), Signal.EMPTY, unit)


# unit vectors: cos(A, B) = 0.8 and cos(B, C) = 0.8, both over the bar; cos(A, C) = 0.28
A, B, C = [1.0, 0.0], [0.8, 0.6], [0.28, 0.96]


def test_topics_group_by_vector_then_by_words() -> None:
    """Newest first in, the most asked first out. Under a profile with a bar the vectors decide;
    otherwise only the same words match. Only a topic's leader is compared: C is close to B but B
    joined A's topic, and C is far from A, so C starts a topic of its own."""
    found = [
        _gap(6, "Kafka retries", 600.0, A, session_id="s1"),
        _gap(5, "idempotent Kafka consumers", 500.0, B, session_id="s2", collections=["notes"]),
        _gap(4, "sourdough starter", 400.0, C, session_id=None),
        _gap(3, "Sourdough, starter?", 300.0, None, embedding="gte-base"),  # words: 4's
        _gap(2, "Kafka retries", 200.0, None, embedding=None),  # words: 6's
        _gap(1, "vitamin D", 100.0, [0.0, 1.0], embedding="gte-base"),  # no bar: words only
        _gap(8, "rioja grapes", 50.0, None),
    ]
    near = {6: [_result(0, None)], 5: []}

    listed = gaps.topics(found, near, BARS)

    assert [[one.search_id for one in topic.questions] for topic in listed] == [
        [6, 5, 2],
        [4, 3],
        [1],
        [8],
    ]
    kafka, sourdough, *_ = listed
    assert kafka.question == "Kafka retries", "named by its newest question"
    assert [one.id for one in kafka.questions] == [60, 50, 20], "each question's own id"
    assert kafka.sessions == 3 and sourdough.sessions == 1, "distinct, without the sessionless"
    assert (kafka.first_at, kafka.last_at) == (200.0, 600.0)
    assert kafka.collections == ["distributed-systems", "kafka", "notes"]
    assert kafka.questions[0].near_misses == near[6] and kafka.questions[2].near_misses == []
    assert (kafka.questions[0].signal, kafka.questions[0].context) == (
        Signal.EMPTY,
        "a Python service on Kafka",
    )


def test_topics_of_nothing_is_nothing() -> None:
    assert gaps.topics([], {}, BARS) == []


# --- flattening an answer -------------------------------------------------------------


def _overlaps(score: float) -> Overlaps:
    measured = Overlap(contained=score, contains=score, alike=score, score=score)
    return Overlaps(words=measured, embedding=measured, chars=None)


def _reference(
    document: str, seq: tuple[int, int], relation: Relation, also_in: list[PassageReference]
) -> PassageReference:
    return PassageReference(
        collection="books",
        document_id=document,
        document=document,
        seq_start=seq[0],
        seq_end=seq[1],
        header="Part II > Replication",
        location=f"{document} p.{seq[0]} L{seq[0] * 10}-{seq[1] * 10 + 9}",
        line_start=seq[0] * 10,
        line_end=seq[1] * 10 + 9,
        score=0.02,
        relation=relation,
        similarity=0.95,
        to_parent=_overlaps(0.95),
        to_root=_overlaps(0.9),
        also_in=also_in,
    )


def _passage(document: str, seq: tuple[int, int], also_in: list[PassageReference]) -> Passage:
    return Passage(
        collection="books",
        document_id=document,
        document=document,
        header="Part II > Replication > Leaders and Followers",
        section_id="leaders",
        location=f"{document} p.151-152 L4210-4231",
        seq_start=seq[0],
        seq_end=seq[1],
        line_start=4210,
        line_end=4231,
        char_start=198_400,
        char_end=199_950,
        page_start=151,
        page_end=152,
        text="Every write goes to the leader, which sends it to its followers.",
        score=0.031,
        source_file="/home/documents/ddia.pdf",
        markdown_file="/home/documents/ddia.pdf.md",
        also_in=also_in,
    )


def _result(position: int, parent: int | None) -> log.LoggedResult:
    return log.LoggedResult(
        position=position,
        parent=parent,
        relation=None,
        collection="books",
        document="ddia.pdf",
        seq_start=3,
        seq_end=4,
        line_start=4210,
        line_end=4231,
        header="Part II > Replication",
        location="ddia.pdf p.151-152 L4210-4231",
        score=0.031,
    )


def test_an_answer_flattens_in_preorder_under_its_parents() -> None:
    """Every result, then every place folded into it, depth first: a place's parent is the
    position of the place it is listed under."""
    nested = _reference("kleppmann-notes.md", (1, 2), Relation.EQUIVALENT, [])
    answer = [
        _passage(
            "ddia.pdf",
            (3, 4),
            [
                _reference("ddia-2nd.pdf", (7, 7), Relation.DUPLICATE, [nested]),
                _reference("replication.md", (9, 9), Relation.CONTAINED, []),
            ],
        ),
        _passage("raft.pdf", (12, 12), []),
    ]
    capture = log.Capture(
        tool=log.Tool.EXPLORE, session_id=None, asked=[Asked("leader replication")]
    )

    capture.answer(answer)

    assert capture.result_count == 2, "the results, not the places folded into them"
    assert [
        (r.position, r.parent, r.document, r.seq_start, r.seq_end, r.relation)
        for r in capture.results
    ] == [
        (0, None, "ddia.pdf", 3, 4, None),
        (1, 0, "ddia-2nd.pdf", 7, 7, Relation.DUPLICATE),
        (2, 1, "kleppmann-notes.md", 1, 2, Relation.EQUIVALENT),
        (3, 0, "replication.md", 9, 9, Relation.CONTAINED),
        (4, None, "raft.pdf", 12, 12, None),
    ]
    assert capture.results[0].location == "ddia.pdf p.151-152 L4210-4231", "the citation is kept"


def test_a_chunk_and_a_document_row_flatten_with_their_own_spans() -> None:
    """A chunk covers one `seq`; a document row (`search_sources`) covers none."""
    hit = Hit(
        collection="books",
        document_id="ddia.pdf",
        document="ddia.pdf",
        source_path="documents/ddia.pdf",
        markdown_path="documents/ddia.pdf.md",
        part=3,
        seq=41,
        line_start=4210,
        line_end=4222,
        char_start=198_400,
        char_end=199_100,
        byte_start=201_000,
        byte_end=201_720,
        page_start=151,
        page_end=151,
        headings=["Part II", "Replication"],
        frame=["Part II", "Replication"],
        header="Part II > Replication",
        location="ddia.pdf p.151 L4210-4222",
        text="Every write goes to the leader.",
        score=0.4,
    )
    source = Source(
        collection="books",
        document_id="raft.pdf",
        document="raft.pdf",
        score=0.2,
        chunks=4,
        description="The Raft paper.",
        header="Leader election",
        location="raft.pdf p.5 L120-140",
        text="A server remains in follower state as long as it receives valid RPCs.",
        source_file="/home/documents/raft.pdf",
        markdown_file="/home/documents/raft.pdf.md",
        line_start=120,
        line_end=140,
        collections=["books"],
        sections=[HotSection("Leader election", 0.2, 4, 120, 160, "raft.pdf p.5-6 L120-160")],
    )

    (chunk,) = log.flatten([hit])
    (row,) = log.flatten([source])

    assert (chunk.seq_start, chunk.seq_end, chunk.score) == (41, 41, 0.4)
    assert (row.seq_start, row.seq_end, row.document) == (None, None, "raft.pdf")


def _span(seq: tuple[int, int], also_in: list[PassageReference]) -> Span:
    passage = _passage("ddia.pdf", seq, also_in)
    fields = {field.name for field in msgspec.structs.fields(Span)}
    return Span(**{name: getattr(passage, name) for name in fields})


def test_an_excerpt_flattens_its_passages_repeats_under_itself() -> None:
    """An excerpt's passages are the excerpt: what repeats any of them is listed under the
    excerpt, in document order. The questions no excerpt answers are marked on the capture."""
    one = _reference("ddia-2nd.pdf", (7, 7), Relation.DUPLICATE, [])
    other = _reference("replication.md", (9, 9), Relation.CONTAINED, [])
    spans = [_span((3, 4), [one]), _span((6, 6), []), _span((8, 8), [other])]
    excerpt = Excerpt(
        collection="books",
        document_id="ddia.pdf",
        document="ddia.pdf",
        header="Part II > Replication",
        section_id=ids.md5(b"doc/s/1"),
        location="ddia.pdf p.151-153 L4210-4290",
        seq_start=3,
        seq_end=8,
        line_start=4210,
        line_end=4290,
        char_start=198_400,
        char_end=202_000,
        page_start=151,
        page_end=153,
        text="Every write goes to the leader. […] Followers apply the log in order.",
        score=0.9,
        source_file="/home/documents/ddia.pdf",
        markdown_file="/home/documents/ddia.pdf.md",
        spans=spans,
    )
    capture = log.Capture(
        tool=log.Tool.EXCERPTS,
        session_id=None,
        asked=[Asked("How do leaders replicate?"), Asked("How do followers catch up?")],
    )

    capture.answer([excerpt], uncovered=["How do followers catch up?"])

    assert [
        (r.position, r.parent, r.document, r.seq_start, r.seq_end) for r in capture.results
    ] == [
        (0, None, "ddia.pdf", 3, 8),
        (1, 0, "ddia-2nd.pdf", 7, 7),
        (2, 0, "replication.md", 9, 9),
    ]
    assert [one.uncovered for one in capture.asked] == [False, True]


# --- the score profile ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "rows", "similarities"),
    [
        ("no rows read", [], []),
        ("best first", [[0.0, 1.0], [1.0, 0.0], [1.0, 0.1]], [1.0, 0.995, 0.0]),
        ("a zero vector scores 0, never divides by 0", [[0.0, 0.0], [2.0, 0.0]], [1.0, 0.0]),
        ("the head of a long ranking", [[1.0, float(n)] for n in range(50)], None),
    ],
)
def test_the_similarities_of_a_ranking(
    name: str, rows: list[list[float]], similarities: list[float] | None
) -> None:
    found = log.similarities([1.0, 0.0], rows)
    if similarities is None:
        assert len(found) == log.PROFILE and found == sorted(found, reverse=True), name
    else:
        assert found == pytest.approx(similarities, abs=1e-3), name


def test_best_scores_are_the_heads_of_the_lists() -> None:
    asked = log.LoggedQuestion("kafka", similarities=[0.8, 0.5], rerank_scores=[0.9])
    assert (asked.best_similarity, asked.best_rerank) == (0.8, 0.9)
    assert log.LoggedQuestion("kafka").best_similarity is None
