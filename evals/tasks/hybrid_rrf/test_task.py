"""Scored against `retrieval.py` as the agent wrote it.

The ordinary assertions check the fusion is right. The discriminating ones are the numbers that
exist only in the benchmark: a default RRF constant is common knowledge, but what one paper's
ablation found best on its own data is not.
"""

import pytest

import retrieval


def test_fuses_by_reciprocal_rank() -> None:
    fused = retrieval.rrf([["a", "b"], ["b", "a"]], k=60)

    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 61 + 1 / 62)


def test_a_document_only_one_retriever_found_still_scores() -> None:
    fused = retrieval.rrf([["a"], ["b"]], k=60)

    assert set(fused) == {"a", "b"}
    assert fused["a"] == pytest.approx(1 / 61)


def test_rank_order_survives_fusion() -> None:
    fused = retrieval.rrf([["a", "b", "c"], ["a", "b", "c"]], k=60)

    assert sorted(fused, key=lambda d: -fused[d]) == ["a", "b", "c"]


def test_the_default_constant_is_the_one_from_the_original_paper() -> None:
    assert retrieval.RRF_K_DEFAULT == 60


@pytest.mark.discriminating
def test_the_ablation_best_constant_comes_from_the_benchmark() -> None:
    """The fusion ablation reports k = 10 as the best RRF variant on this corpus, against 0.695
    for the default 60. Nothing outside that paper says so."""
    assert retrieval.RRF_K_BEST == 10


@pytest.mark.discriminating
def test_the_two_stage_pipeline_uses_the_benchmarks_depths() -> None:
    """Hybrid RRF retrieves 50 candidates, the reranker returns the top 10. Reranking at 20
    candidates is reported as ineffective."""
    assert retrieval.RERANK_CANDIDATES == 50
    assert retrieval.RERANK_TOP_N == 10


def test_bm25_is_parameterised_as_the_benchmark_reports() -> None:
    """Not discriminating: 1.2 and 0.75 are the textbook defaults, and a run that never opened the
    paper will reach for them anyway. Here to catch a broken config, not to separate arms."""
    assert retrieval.BM25_K1 == pytest.approx(1.2)
    assert retrieval.BM25_B == pytest.approx(0.75)
