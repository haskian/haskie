"""The MLX models, behind fastembed's shapes: a reranker answers one score per text, in order.

No model is loaded here (up to 1.2 GB each, Apple Silicon only): Jina's published `MLXReranker`
is stood in for by one that answers the way it does, best first with each text's index.
"""

import threading

import pytest

from haskie.indexing import embed, mlx_models
from haskie.indexing.mlx_models import ListwiseReranker
from haskie.settings import Accelerator

TEXTS = [
    "Basketball is one of the most popular sports in the United States.",
    "Green tea contains antioxidants called catechins that may help reduce inflammation.",
    "Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale.",
]


class Published:
    """`MLXReranker.rerank` as Jina's code answers: sorted best first, each with its index."""

    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        scores = {0: -0.17, 1: 0.298, 2: 0.104}
        order = sorted(scores, key=lambda index: scores[index], reverse=True)
        return [{"document": documents[i], "relevance_score": scores[i], "index": i} for i in order]


def _reranker() -> ListwiseReranker:
    reranker = ListwiseReranker.__new__(ListwiseReranker)  # skip the download and the load
    reranker._model = Published()
    reranker._lock = threading.Lock()
    return reranker


@pytest.mark.parametrize(
    ("name", "texts", "expected"),
    [
        ("scores come back in the order the texts went in", TEXTS, [-0.17, 0.298, 0.104]),
    ],
)
def test_scores_follow_the_texts_not_the_ranking(
    name: str, texts: list[str], expected: list[float]
) -> None:
    assert _reranker().rerank("health benefits of green tea", texts) == expected, name


def test_without_mlx_the_model_says_what_it_needs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mlx_models, "available", lambda: False)

    with pytest.raises(RuntimeError, match="uv sync --extra mlx"):
        mlx_models.reranker("jinaai/jina-reranker-v3-mlx")


def test_every_mlx_model_is_routed_to_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building a pinned MLX reranker or embedder takes the MLX path rather than fastembed's.
    That every pin is in the catalogue is `test_catalogue`'s to say."""
    built: list[str] = []
    monkeypatch.setattr(mlx_models, "reranker", lambda name: built.append(name) or name)
    monkeypatch.setattr(mlx_models, "embedder", lambda name: built.append(name) or name)
    embed._build_cross_encoder.cache_clear()
    embed._build_model.cache_clear()
    rerankers = [*mlx_models.LISTWISE, *mlx_models.PAIRWISE]
    embedders = list(mlx_models.EMBEDDERS)

    for name in rerankers:
        assert embed._build_cross_encoder(name, Accelerator.AUTO) == name
    for name in embedders:
        assert embed._build_model(name, Accelerator.AUTO) == name
    embed._build_cross_encoder.cache_clear()
    embed._build_model.cache_clear()

    assert built == rerankers + embedders


@pytest.mark.skipif(not mlx_models.available(), reason="MLX installs on Apple Silicon only")
def test_xlm_roberta_gets_the_classification_head_bge_weights_load_into() -> None:
    """mlx-embeddings has no head for XLM-RoBERTa: bge-reranker's `classifier.*` weights would
    have nowhere to load, and its missing `pooler.*` would be asked for. The shape here is the
    one the published weights have."""
    from mlx.utils import tree_flatten

    config = {
        "model_type": "xlm-roberta",
        "architectures": ["XLMRobertaForSequenceClassification"],
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 64,
        "vocab_size": 100,
        "max_position_embeddings": 64,
        "type_vocab_size": 1,
        "layer_norm_eps": 1e-5,
    }
    model, args, *_ = mlx_models._with_head(config)

    names = {name for name, _ in tree_flatten(model(args.from_dict(config)).parameters())}

    assert {"classifier.dense.weight", "classifier.out_proj.weight"} <= names
    assert not any(name.startswith("pooler.") for name in names), "classification has no pooler"
