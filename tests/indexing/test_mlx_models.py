"""The MLX models, shaped as every model is: a reranker answers one score per text, an embedder
one vector per text, in order.

No model is loaded here (up to 1.2 GB each, Apple Silicon only): the model and its tokenizer are
stood in for by ones that answer each text's index, so the order of the scores shows.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from haskie.indexing import embed, mlx_models
from haskie.indexing.onnx_models import Pooling
from haskie.settings import Accelerator

TEXTS = [
    "Basketball is one of the most popular sports in the United States.",
    "Green tea contains antioxidants called catechins that may help reduce inflammation.",
    "Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale.",
]
# the catalogue pins no MLX reranker today, but the runtime stays: this one stands in
RERANKER, EMBEDDER = "test/tiny-reranker-mlx", "intfloat/e5-base-v2:mlx"
needs_mlx = pytest.mark.skipif(
    not mlx_models.available(), reason="MLX installs on Apple Silicon only"
)


def test_a_model_loads_and_runs_on_the_mlx_thread_whoever_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pipeline loads a model on one worker thread and embeds on others. MLX aborts the
    process when a thread evaluates an array still lazy on another, so both must land on one."""
    ran_on: list[str] = []

    def load(path: Path, **classes: Any) -> tuple[Any, Any]:
        ran_on.append(threading.current_thread().name)
        return None, lambda queries, texts, **_: texts

    def forward(self: mlx_models.Reranker, texts: list[str]) -> list[float]:
        ran_on.append(threading.current_thread().name)
        return [0.5] * len(texts)

    monkeypatch.setattr(mlx_models, "_download", lambda name: tmp_path)
    monkeypatch.setattr(mlx_models, "_load", load)
    monkeypatch.setattr(mlx_models.Reranker, "_forward", forward)

    with ThreadPoolExecutor(1) as loader, ThreadPoolExecutor(1) as searcher:
        model = loader.submit(mlx_models.reranker, RERANKER).result()
        answered = searcher.submit(model.rerank, "green tea", TEXTS).result()

    assert (answered, ran_on) == ([0.5, 0.5, 0.5], ["mlx_0", "mlx_0"]), "load, then one batch"


@pytest.mark.parametrize(
    ("name", "model", "expected"),
    [
        (
            "a head that puts a sigmoid on its label answers the logit instead",
            SimpleNamespace(is_regression=None),
            True,
        ),
        ("a head that answers a logit already is left alone", SimpleNamespace(), None),
    ],
)
def test_every_reranker_answers_a_logit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, model: Any, expected: bool | None
) -> None:
    """gte (ModernBERT) came back as sigmoid(logit) and bge as the logit, and `cross_encode`
    applies the sigmoid once, to every reranker alike."""
    monkeypatch.setattr(mlx_models, "_load", lambda path, **classes: (model, None))

    mlx_models.Reranker(tmp_path)

    assert getattr(model, "is_regression", None) is expected, name


@needs_mlx
@pytest.mark.parametrize(
    ("name", "count"),
    [
        ("no texts, no scores", 0),
        ("one text, one score", 1),
        ("past one batch, every score in the order the texts went in", mlx_models.PAIR_BATCH + 1),
    ],
)
def test_a_reranker_answers_one_score_per_text_in_order(name: str, count: int) -> None:
    import mlx.core as mx

    def tokenizer(queries: list[str], texts: list[str], **_: object) -> dict[str, np.ndarray]:
        indices = [[int(text.split()[-1])] for text in texts]
        return {"input_ids": np.array(indices), "attention_mask": np.ones((len(texts), 1))}

    def model(input_ids: mx.array, attention_mask: mx.array) -> SimpleNamespace:
        return SimpleNamespace(pooler_output=input_ids.astype(mx.float32) / 2)

    reranker = mlx_models.Reranker.__new__(mlx_models.Reranker)  # skip the download and the load
    reranker._model, reranker._tokenizer = model, tokenizer
    texts = [f"{TEXTS[1]} {index}" for index in range(count)]

    assert reranker.rerank("green tea", texts) == [index / 2 for index in range(count)], name


@needs_mlx
def test_an_array_left_lazy_by_one_caller_evaluates_for_another() -> None:
    """Without the MLX thread this aborts the test process: "There is no Stream(gpu, 0) in
    current thread", which is what Jina v5's lazily loaded weights did to the server."""
    import mlx.core as mx

    def build() -> mx.array:
        return mx.ones((4, 4)) * 2  # not evaluated: it stays lazy on the thread that built it

    with ThreadPoolExecutor(1) as first, ThreadPoolExecutor(1) as second:
        lazy: mx.array = first.submit(mlx_models._on_mlx_thread, build).result()
        total = second.submit(mlx_models._on_mlx_thread, lambda: lazy.sum().item()).result()

    assert total == 32.0


@needs_mlx
@pytest.mark.parametrize(
    ("name", "count"),
    [
        ("no texts, no vectors", 0),
        ("one text, one vector", 1),
        ("past one batch, every vector in order", mlx_models.EMBED_BATCH + 1),
    ],
)
def test_an_embedder_answers_one_vector_per_text(name: str, count: int) -> None:
    import mlx.core as mx
    from tokenizers import Tokenizer
    from tokenizers.models import BPE

    class Encoder:
        """Jina v5's `encode`, answering each text's index as its vector."""

        def encode(self, texts: list[str], tokenizer: object, **_: object) -> mx.array:
            return mx.array([[float(text.split()[-1])] * 4 for text in texts])

    embedder = mlx_models.JinaV5Embedder.__new__(mlx_models.JinaV5Embedder)
    embedder._model, embedder._tokenizer = Encoder(), Tokenizer(BPE())
    texts = [f"{TEXTS[0]} {index}" for index in range(count)]

    vectors = list(embedder.embed(texts))

    assert [vector.tolist() for vector in vectors] == [[float(i)] * 4 for i in range(count)], name


def test_without_mlx_the_model_says_what_it_needs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mlx_models, "available", lambda: False)

    with pytest.raises(RuntimeError, match="mlx-embeddings, which haskie installs there"):
        mlx_models.reranker(RERANKER)


def test_a_pinned_mlx_model_is_routed_to_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building a pinned MLX reranker or embedder takes the MLX path rather than ONNX's.
    That every pin is in the catalogue is `test_catalogue`'s to say."""
    built: list[str] = []
    monkeypatch.setitem(mlx_models.RERANKERS, RERANKER, "0")
    monkeypatch.setattr(mlx_models, "reranker", lambda name: built.append(name) or name)
    monkeypatch.setattr(mlx_models, "embedder", lambda name: built.append(name) or name)
    embed._build_cross_encoder.cache_clear()
    embed._build_model.cache_clear()
    try:
        assert embed._build_cross_encoder(RERANKER, Accelerator.AUTO) == RERANKER
        assert embed._build_model(EMBEDDER, Accelerator.AUTO) == EMBEDDER
    finally:
        embed._build_cross_encoder.cache_clear()
        embed._build_model.cache_clear()

    assert built == [RERANKER, EMBEDDER]


@needs_mlx
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


@needs_mlx
@pytest.mark.parametrize(
    ("name", "pooling", "states", "expected"),
    [
        (
            "token states, mean-pooled over the real tokens and normalized",
            Pooling.MEAN,
            [[[3.0, 0.0], [5.0, 0.0]], [[0.0, 2.0], [9.0, 9.0]]],
            [[1.0, 0.0], [0.0, 1.0]],
        ),
        (
            "token states, the last real token's",
            Pooling.LAST,
            [[[9.0, 9.0], [0.0, 4.0]], [[3.0, 0.0], [9.0, 9.0]]],
            [[0.0, 1.0], [1.0, 0.0]],
        ),
        (
            "a model mlx-embeddings pooled already: its pooled vector as it came",
            Pooling.MEAN,
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.6, 0.8], [0.6, 0.8]],
        ),
    ],
)
def test_a_pooled_embedder_pools_its_token_states_as_its_model_asks(
    name: str, pooling: Pooling, states: list, expected: list
) -> None:
    import mlx.core as mx

    def model(input_ids: mx.array, attention_mask: mx.array) -> SimpleNamespace:
        return SimpleNamespace(
            last_hidden_state=mx.array(states), text_embeds=mx.array([[0.6, 0.8]] * 2)
        )

    def tokenizer(texts: list[str], **_: object) -> dict[str, np.ndarray]:
        return {"input_ids": np.zeros((2, 2)), "attention_mask": np.array([[1, 1], [1, 0]])}

    embedder = mlx_models.PooledEmbedder.__new__(mlx_models.PooledEmbedder)  # no download
    embedder._model, embedder._tokenizer, embedder._pooling = model, tokenizer, pooling

    vectors = [vector.tolist() for vector in embedder.embed(["a", "b"])]

    assert vectors == [pytest.approx(one) for one in expected], name


def test_a_pooled_model_downloads_from_its_own_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its catalogue name is the repository's with `:mlx`; the download takes the pin's."""
    import huggingface_hub

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(mlx_models, "available", lambda: True)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo, revision, allow_patterns: calls.append((repo, revision)) or "/models",
    )

    mlx_models._download(EMBEDDER)

    assert calls == [("intfloat/e5-base-v2", mlx_models.POOLED[EMBEDDER].revision)]
