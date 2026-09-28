"""A search's score lineage: what each step that sets or changes a score says about how, from the
search, what the step read and what it answered, and nothing from a step that left them alone."""

from pathlib import Path
from typing import Any

import msgspec
import pytest
from conftest import hit

from haskie.catalogue.catalogue import UNCALIBRATED, EmbeddingModel, RerankerCalibration
from haskie.collection.index import CollectionIndex
from haskie.search import flow
from haskie.search.flow import Search
from haskie.search.passage import ranges
from haskie.search.retrieval import Plan, Pool
from haskie.search.scoring import RULES
from haskie.search.section import Group, Section
from haskie.settings import Fusion, Reranker, ScoreFold, SearchMode, SearchSettings

HARMONIC = ScoreFold.HARMONIC  # the rule these cases were written against

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
RERANKS = (
    "The cross-encoder Xenova/ms-marco-MiniLM-L-6-v2 rescores up to 50 candidates, and the "
    "sigmoid of its logit, 1 / (1 + e^−logit), replaces every score before it: 0 to 1, 0.5 at "
    "logit 0, bounded but not calibrated. It reads the query and the chunk together, so the mode "
    "only decides which candidates it reads, and a chunk scores the same in every mode that finds "
    "it."
)
RERANK = f"{RERANKS} Chunks it scores under 0.05 (uncalibrated) are dropped."
PASSAGE = "A passage scores the sum of its matched chunks' scores."


def _search(
    *settings: SearchSettings,
    vector: list[float] | None = ON,
    embedding: EmbeddingModel | None = MODEL,
    calibration: RerankerCalibration | None = UNCALIBRATED,
) -> Search:
    """A search over one collection per settings, `c0`, `c1`, …, the first settings the plan's,
    its reranker read by `calibration` (the seed's uncalibrated floor, 0.05)."""
    chosen = settings or (SearchSettings(),)
    indexes = [
        (CollectionIndex(Path(f"/tmp/c{n}"), f"c{n}", Path("/tmp"), embedding), one)
        for n, one in enumerate(chosen)
    ]
    where = Plan(
        settings=chosen[0],
        indexes=indexes,
        vector=vector,
        embedding=embedding,
        calibration=calibration,
    )
    return Search(query="q", framed="q", plan=where, limit=5, scan=20, candidates=50, questions=[])


def _pool(*answered: str, empty: tuple[str, ...] = (), ranked: int = 0) -> Pool:
    """What the retrieval read: a ranking per collection, `empty` ones with no row."""
    rankings: dict[str, list[Any]] = {name: [(name, "a.md", 1)] for name in answered}
    rankings |= {name: [] for name in empty}
    scored = [(("c0", "a.md", n), 1.0) for n in range(ranked)]
    return Pool(rows={}, rankings=rankings, ranked=scored)


def _groups(*seqs: int) -> list[Group]:
    found = ranges(
        [hit(f"chunk {seq}", 1.0, document="a.md", seq=seq) for seq in seqs], how=HARMONIC
    )
    return [Group("c0", "a.md", Section(("A",), 1, 9), found)]


def _read(*columns: str | None) -> Pool:
    """What retrieval read: collection `c<n>` returned one row scored in the column `columns[n]`
    names (`_score` BM25, `_distance` vector, `_relevance_score` fused), or nothing for None."""
    rows = {
        (f"c{n}", "a.md", 1): (None, {column: 1.0}) for n, column in enumerate(columns) if column
    }
    rankings = {
        f"c{n}": [(f"c{n}", "a.md", 1)] if column else [] for n, column in enumerate(columns)
    }
    return Pool(rows=rows, rankings=rankings)  # ty: ignore[invalid-argument-type]


LINEAR = (
    "Hybrid: {} × vector + {} × BM25, each min-max scaled to 0–1 over the candidates, a half that "
    "missed the chunk counting 0. The best candidate of any search scores near 1."
)


@pytest.mark.parametrize(
    ("name", "state", "read", "expected"),
    [
        (
            "full text when the mode asks for it",
            _search(SearchSettings(mode=SearchMode.FTS), vector=None),
            _read("_score"),
            BM25,
        ),
        (
            "full text in any mode without an embedding model",
            _search(SearchSettings(), vector=None, embedding=None),
            _read("_score"),
            f"{BM25} No embedding model, so every mode is BM25.",
        ),
        (
            "vector: the distance, turned into a score",
            _search(SearchSettings(mode=SearchMode.VECTOR)),
            _read("_distance"),
            VECTOR,
        ),
        (
            "hybrid, linear: the weights as shares of their sum",
            _search(SearchSettings(fusion=Fusion.LINEAR, vector_weight=3.0, bm25_weight=1.0)),
            _read("_relevance_score"),
            LINEAR.format("0.75", "0.25"),
        ),
        (
            "hybrid, linear, both weights 0: an even split",
            _search(SearchSettings(fusion=Fusion.LINEAR, vector_weight=0.0, bm25_weight=0.0)),
            _read("_relevance_score"),
            LINEAR.format("0.50", "0.50"),
        ),
        (
            "hybrid, reciprocal rank fusion",
            _search(SearchSettings(fusion=Fusion.RRF, rrf_k=10)),
            _read("_relevance_score"),
            "Hybrid: reciprocal rank fusion of the vector and BM25 ranks, the sum of 1 / (10 + "
            "rank) over the two, at most 0.1818. Rank only.",
        ),
        (
            "hybrid asked of a table written without vectors: it answered by full text",
            _search(SearchSettings(mode=SearchMode.HYBRID)),
            _read("_score"),
            BM25,
        ),
        (
            "a collection that returned nothing: its settings say it",
            _search(SearchSettings(mode=SearchMode.VECTOR)),
            _read(None),
            VECTOR,
        ),
        (
            "and a lexical search's, full text",
            _search(SearchSettings(mode=SearchMode.VECTOR), vector=None),
            _read(None),
            BM25,
        ),
        (
            "collections of one rule say it once",
            _search(SearchSettings(mode=SearchMode.VECTOR), SearchSettings(mode=SearchMode.VECTOR)),
            _read("_distance", "_distance"),
            VECTOR,
        ),
        (
            "collections of other rules each name theirs",
            _search(
                SearchSettings(mode=SearchMode.VECTOR),
                SearchSettings(mode=SearchMode.FTS),
                SearchSettings(mode=SearchMode.VECTOR),
            ),
            _read("_distance", "_score", "_distance"),
            f"c0, c2: {VECTOR} c1: {BM25}",
        ),
    ],
)
def test_retrieval_says_how_each_collection_scored(
    name: str, state: Search, read: Pool, expected: str
) -> None:
    assert RULES["retrieve"](state, None, read) == expected, name


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
            RERANK,
        ),
        (
            "a floor the settings set, over the reranker's calibrated one",
            "rerank",
            _search(SearchSettings(reranker=Reranker.CROSS_ENCODER, min_rerank_score=0.2)),
            None,
            None,
            f"{RERANKS} Chunks it scores under 0.2 (set) are dropped.",
        ),
        (
            "a floor of 0 drops nothing, and says nothing",
            "rerank",
            _search(SearchSettings(reranker=Reranker.CROSS_ENCODER, min_rerank_score=0.0)),
            None,
            None,
            RERANKS,
        ),
        (
            "a shared context: the reranker reads each question alone, by default",
            "rerank",
            msgspec.structs.replace(
                _search(SearchSettings(reranker=Reranker.CROSS_ENCODER)), framed="ddd\n\nq"
            ),
            None,
            None,
            f"{RERANK} It reads each question alone: the shared context only found the candidates.",
        ),
        (
            "or with it in front, when set to",
            "rerank",
            msgspec.structs.replace(
                _search(SearchSettings(reranker=Reranker.CROSS_ENCODER, rerank_with_context=True)),
                framed="ddd\n\nq",
            ),
            None,
            None,
            f"{RERANK} It reads each question with the shared context in front.",
        ),
        (
            "passages that are only judged, by the default sum",
            "judge_thin",
            _search(),
            None,
            None,
            PASSAGE,
        ),
        (
            "by the best chunk",
            "judge_thin",
            _search(SearchSettings(score_fold=ScoreFold.MAX)),
            None,
            None,
            "A passage scores its best chunk's score.",
        ),
        (
            "by the harmonic mean",
            "judge_thin",
            _search(SearchSettings(score_fold=ScoreFold.HARMONIC)),
            None,
            None,
            "A passage scores the harmonic mean of its best chunk and the sum of its matched "
            "chunks, 2·best·sum / (best + sum), between the best and twice it.",
        ),
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
        (
            "several questions with a reranker: each passage scores its best",
            "fold",
            _search(SearchSettings(reranker=Reranker.CROSS_ENCODER)),
            [None, None],
            None,
            "Each chunk of a passage scores its best question, the reranker's score for it in "
            "that question's own ranking, and the passage folds them by its rule. The questions "
            "take turns at the slots, so the list is not in score order.",
        ),
        ("an excerpt", "group", _search(), None, None, "An excerpt scores its best passage."),
        ("a probe that added nothing", "probe_gaps", _search(), _groups(1), _groups(1), None),
        (
            "a probe's passage",
            "probe_gaps",
            _search(),
            _groups(1),
            _groups(1, 5),
            "The passage the probe added scores 0: a full-text search of the missing words alone "
            "found it, on another scale than the passages ranked for the question.",
        ),
        (
            "a probe's passage, judged by the reranker",
            "probe_gaps",
            _search(SearchSettings(reranker=Reranker.CROSS_ENCODER)),
            _groups(1),
            _groups(1, 5),
            "The passage the probe added, found by a full-text search of the missing words, is "
            "scored by the reranker against each question whose words are missing, as the ranked "
            "chunks are, and kept only above its floor.",
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
            "A document scores the sum of its matched chunks' scores. Each section of it scores "
            "the same over its own chunks.",
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
        flow._lineage("retrieve", state, None, _read("_relevance_score"))
        for step in ("merge", "rerank"):
            flow._lineage(step, state, pool, pool)

    assert [one.step for one in trace.scoring] == ["retrieve", "merge", "rerank"]
