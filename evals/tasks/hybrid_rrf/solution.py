"""Reference for `hybrid_rrf`, from akarsu-2026 p.3-4 and its configuration table on p.10."""

RRF_K_DEFAULT = 60  # "the value used in the original paper", p.3
RRF_K_BEST = 10  # fusion ablation, p.4: 0.716 against 0.695 for k = 60
RERANK_CANDIDATES = 50  # two-stage pipeline, p.3; 20 candidates is reported ineffective
RERANK_TOP_N = 10
BM25_K1 = 1.2
BM25_B = 0.75


def rrf(rankings: list[list[str]], k: int = RRF_K_DEFAULT) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return fused
