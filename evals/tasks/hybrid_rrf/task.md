Write `retrieval.py` in the current directory, implementing the hybrid retrieval configuration
recommended by the text-and-table RAG benchmark "From BM25 to Corrective RAG: Benchmarking
Retrieval Strategies for Text-and-Table Documents" (Akarsu et al., 2026). Standard library only.

Define exactly these names:

    RRF_K_DEFAULT: int       # the smoothing constant the original RRF paper uses
    RRF_K_BEST: int          # the value that paper's own fusion ablation found best on its data
    RERANK_CANDIDATES: int   # documents the recommended two-stage pipeline passes to the reranker
    RERANK_TOP_N: int        # documents returned after reranking
    BM25_K1: float           # the benchmark's BM25 term-frequency saturation parameter
    BM25_B: float            # the benchmark's BM25 document-length normalisation parameter

    def rrf(rankings: list[list[str]], k: int = RRF_K_DEFAULT) -> dict[str, float]:
        """Fused scores by document id. Each element of `rankings` is one retriever's ranked list
        of document ids, best first, where the best document is at rank 1."""

Use that paper's own numbers rather than the ones you would reach for by default, and say where
each came from. If you cannot confirm a value against a source, write your best estimate rather
than stopping, and mark that line `# unconfirmed`. Write the file in this session either way;
do not stop to ask.
