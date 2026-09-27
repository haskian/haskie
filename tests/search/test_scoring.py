"""A search's score lineage: what each step that sets or changes a score says about how, from the
search, what the step read and what it answered, and nothing from a step that left them alone."""

from pathlib import Path
from typing import Any

import pytest
from conftest import hit

from haskie.catalogue.catalogue import EmbeddingModel
from haskie.collection.index import CollectionIndex
from haskie.search import flow
from haskie.search.flow import Search
from haskie.search.passage import ranges
from haskie.search.retrieval import Plan, Pool
from haskie.search.scoring import RULES
from haskie.search.section import Group, Section
from haskie.settings import Fusion, Reranker, SearchMode, SearchSettings

MODEL = EmbeddingModel("test/model", 2)
ON = [1.0, 0.0]
BM25 = (
    "BM25 score from LanceDB's full-text index: unbounded, higher is better, larger for rarer "
    "query words. Comparable within one search only."
)
VECTOR = (
    "1 / (1 + d), d the squared L2 distance between the query and chunk embeddings: "
    "1 / (3 − 2·cosine) for unit vectors, 1 at cosine 1 and 0.33 at cosine 0."
)
PASSAGE = (
    "A passage scores the harmonic mean of its best chunk and the sum of its matched chunks: "
    "2·best·sum / (best + sum), between the best and twice it."
)


def _search(
    *settings: SearchSettings,
    vector: list[float] | None = ON,
    embedding: EmbeddingModel | None = MODEL,
) -> Search:
    """A search over one collection per settings, `c0`, `c1`, …, the first settings the plan's."""
    chosen = settings or (SearchSettings(),)
    indexes = [
        (CollectionIndex(Path(f"/tmp/c{n}"), f"c{n}", Path("/tmp"), embedding), one)
        for n, one in enumerate(chosen)
    ]
    where = Plan(settings=chosen[0], indexes=indexes, vector=vector, embedding=embedding)
    return Search(query="q", framed="q", plan=where, limit=5, scan=20, candidates=50, questions=[])


def _pool(*answered: str, empty: tuple[str, ...] = (), ranked: int = 0) -> Pool:
    """What the retrieval read: a ranking per collection, `empty` ones with no row."""
    rankings: dict[str, list[Any]] = {name: [("a.md", 1)] for name in answered}
    rankings |= {name: [] for name in empty}
    scored = [(("a.md", n), 1.0) for n in range(ranked)]
    return Pool(rows={}, rankings=rankings, ranked=scored)


def _groups(*seqs: int) -> list[Group]:
    found = ranges([hit(f"chunk {seq}", 1.0, document="a.md", seq=seq) for seq in seqs])
    return [Group("c0", "a.md", Section(("A",), 1, 9), found)]


@pytest.mark.parametrize(
    ("name", "state", "expected"),
    [
        (
            "full text when the mode asks for it",
            _search(SearchSettings(mode=SearchMode.FTS), vector=None),
            BM25,
        ),
        (
            "full text in any mode without an embedding model",
            _search(SearchSettings(), vector=None, embedding=None),
            f"{BM25} No embedding model, so every mode is BM25.",
        ),
        (
            "vector: the distance, turned into a score",
            _search(SearchSettings(mode=SearchMode.VECTOR)),
            VECTOR,
        ),
        (
            "hybrid, linear: the weights as shares of their sum",
            _search(SearchSettings(fusion=Fusion.LINEAR, vector_weight=3.0, bm25_weight=1.0)),
            "Hybrid: 0.75 × vector + 0.25 × BM25, each min-max scaled to 0–1 over the candidates, "
            "a half that missed the chunk counting 0. The best candidate of any search scores "
            "near 1.",
        ),
        (
            "hybrid, linear, both weights 0: an even split",
            _search(SearchSettings(fusion=Fusion.LINEAR, vector_weight=0.0, bm25_weight=0.0)),
            "Hybrid: 0.50 × vector + 0.50 × BM25, each min-max scaled to 0–1 over the candidates, "
            "a half that missed the chunk counting 0. The best candidate of any search scores "
            "near 1.",
        ),
        (
            "hybrid, reciprocal rank fusion",
            _search(SearchSettings(fusion=Fusion.RRF, rrf_k=10)),
            "Hybrid: reciprocal rank fusion of the vector and BM25 ranks, the sum of 1 / (10 + "
            "rank) over the two, at most 0.1818. Rank only.",
        ),
        (
            "collections of one rule say it once",
            _search(SearchSettings(mode=SearchMode.VECTOR), SearchSettings(mode=SearchMode.VECTOR)),
            VECTOR,
        ),
        (
            "collections of other rules each name theirs",
            _search(
                SearchSettings(mode=SearchMode.VECTOR),
                SearchSettings(mode=SearchMode.FTS),
                SearchSettings(mode=SearchMode.VECTOR),
            ),
            f"c0, c2: {VECTOR} c1: {BM25}",
        ),
    ],
)
def test_retrieval_says_how_each_collection_scored(name: str, state: Search, expected: str) -> None:
    assert RULES["retrieve"](state, None, _pool("c0")) == expected, name


@pytest.mark.parametrize(
    ("name", "step", "state", "read", "answered", "expected"),
    [
        (
            "one collection keeps its scores",
            "merge",
            _search(),
            _pool("c0"),
            None,
            "One collection: its scores are kept.",
        ),
        (
            "several fuse by rank",
            "merge",
            _search(SearchSettings(rrf_k=60), SearchSettings()),
            _pool("c0", "c1"),
            None,
            "Reciprocal rank fusion over the 2 collections' rankings replaces their scores: the "
            "sum of 1 / (60 + rank) over the rankings a chunk is in, 0.0164 for first place in "
            "one. Rank only.",
        ),
        ("no reranker changes nothing", "rerank", _search(), None, _pool("c0", ranked=3), None),
        (
            "a reranker replaces every score, whatever the mode",
            "rerank",
            _search(SearchSettings(reranker=Reranker.CROSS_ENCODER, mode=SearchMode.VECTOR)),
            None,
            _pool("c0", ranked=3),
            "The cross-encoder Xenova/ms-marco-MiniLM-L-6-v2 rescores up to 50 candidates, and its "
            "raw logit replaces every score before it: unbounded, higher is better. It reads the "
            "query and the chunk together, so the mode only decides which candidates it reads, "
            "and a chunk scores the same in every mode that finds it.",
        ),
        ("passages that are only judged", "judge_thin", _search(), None, None, PASSAGE),
        (
            "passages that grow",
            "fill_thin",
            _search(),
            None,
            None,
            f"{PASSAGE} A chunk a short passage grew into scores 0, so it adds nothing.",
        ),
        ("one question's fold keeps its order", "fold", _search(), [None], None, None),
        (
            "several questions take turns",
            "fold",
            _search(),
            [None, None],
            None,
            "Each question's passages keep the scores of their own search, and the questions take "
            "turns at the slots, so the list is not in score order.",
        ),
        ("an excerpt", "group", _search(), None, None, "An excerpt scores its best passage."),
        ("a probe that added nothing", "probe_gaps", _search(), _groups(1), _groups(1), None),
        (
            "a probe's passage, from one collection",
            "probe_gaps",
            _search(),
            _groups(1),
            _groups(1, 5),
            "The passage the probe added scores by a full-text search of the missing words alone: "
            "BM25, on another scale than the passages ranked for the question.",
        ),
        (
            "a probe's passage, from several",
            "probe_gaps",
            _search(SearchSettings(), SearchSettings()),
            _groups(1),
            _groups(1, 5),
            "The passage the probe added scores by a full-text search of the missing words alone: "
            "BM25 fused by rank across collections, on another scale than the passages ranked for "
            "the question.",
        ),
        ("a fill that added nothing", "fill", _search(), _groups(1), _groups(1), None),
        (
            "a fill that added text",
            "fill",
            _search(),
            _groups(1),
            _groups(1, 2),
            "Text filled in around and between passages scores 0; each passage keeps its score.",
        ),
        (
            "a document",
            "shortlist",
            _search(),
            None,
            None,
            "A document scores the harmonic mean of its best chunk and the sum of all its matched "
            "chunks: 2·best·sum / (best + sum), between the best and twice it. Each section of it "
            "scores the same over its own chunks.",
        ),
    ],
)
def test_a_step_says_how_it_changed_the_scores(
    name: str, step: str, state: Search, read: Any, answered: Any, expected: str | None
) -> None:
    assert RULES[step](state, read, answered) == expected, name


@pytest.mark.parametrize(
    "step", ["hits", "collapse_hits", "collapse_ranges", "read", "budget", "quote"]
)
def test_a_step_that_leaves_scores_alone_has_no_rule(step: str) -> None:
    assert step not in RULES


def test_several_questions_record_each_rule_once_whatever_each_read() -> None:
    """Each question of an `answers` runs the ranking on its own pool: one found 12 candidates,
    another 30. The rules come from the settings, so the lineage says each once."""
    state = _search(SearchSettings(reranker=Reranker.CROSS_ENCODER))
    trace = flow.start_trace()

    for ranked in (12, 30):
        pool = _pool("c0", ranked=ranked)
        for step in ("retrieve", "merge", "rerank"):
            flow._lineage(step, state, pool, pool)

    assert [one.step for one in trace.scoring] == ["retrieve", "merge", "rerank"]
