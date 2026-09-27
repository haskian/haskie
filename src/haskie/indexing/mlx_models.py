"""Models that run on Apple Silicon through MLX, for models fastembed cannot run.

fastembed runs ONNX models only. These are published as MLX builds.

Rerankers (`RERANKERS`: bge-reranker-v2-m3, gte-reranker-modernbert, mMiniLM): ordinary
cross-encoders, one (query, text) pair a pass, run through mlx-embeddings. Its XLM-RoBERTa has no
classification head, so one is added here (`_with_head`): the standard one, dense, tanh, then one
logit, over the first token. It loads an MLX conversion and transformers' own checkpoint alike, so
an XLM-RoBERTa cross-encoder needs no conversion. All Apache-2.0.

Embedders (`EMBEDDERS`), each answering `embed(texts)` and `query_embed(text)` the way fastembed's
`TextEmbedding` does, so `embed` calls them the same way:

- nomic ModernBERT embed, through mlx-embeddings, which mean-pools as the model's config asks and
  normalizes. Its search prefixes are the profile's (`EmbeddingModel.query_prefix`).
- Jina v5 text nano (retrieval): its own `model.py`, run as published, which adds its task
  prefixes itself. CC BY-NC 4.0, non-commercial only.

Every repository is pinned to a revision, so the code and weights that run are the ones reviewed.
Every load and every forward pass runs on one thread (`_on_mlx_thread`), a batch at a time.
Needs the `mlx` extra (mlx-embeddings), which only installs on Apple Silicon.
"""

import functools
import importlib.util
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from pathlib import Path
from typing import Any

# Hugging Face repository -> the revision its code and weights are read at
RERANKERS: dict[str, str] = {
    "soichisumi/bge-reranker-v2-m3-mlx-affine8": "512d2c5984b21da2b134c7f169a9f4176735287c",
    "afanjul/gte-reranker-modernbert-base-mlx": "0b1cfb9141dd1452e07a328a0dec430f2324da12",
    # not an MLX conversion: transformers' own checkpoint, loaded by `_with_head` as it stands
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": "1427fd652930e4ba29e8149678df786c240d8825",
}
MODERNBERT: dict[str, str] = {
    "mlx-community/nomicai-modernbert-embed-base-bf16": "5bfbc2093cbc41d44548a8de02c0c26f801938e1",
}
JINA_V5: dict[str, str] = {
    "jinaai/jina-embeddings-v5-text-nano-retrieval-mlx": "cb07521719bddd48f5647b5531358a8ca2d1b8d0",
}
EMBEDDERS = MODERNBERT | JINA_V5
REVISIONS = RERANKERS | EMBEDDERS
EMBED_BATCH = 16  # texts a forward pass

PAIR_BATCH = 16  # pairs a forward pass: the candidates of one search in a few passes
MAX_PAIR_TOKENS = 512  # what bge-reranker-v2-m3 was tuned at; a chunk and a query fit well inside
MAX_EMBED_TOKENS = 1024  # a chunk with its heading path fits well inside; both models read 8K

# MLX keeps its streams per thread: an array still lazy on one thread aborts the process when
# another evaluates it ("There is no Stream(cpu, 1) in current thread", a C++ exception nothing
# catches). Jina v5's weights stay lazy after loading, and the pipeline loads a model on one worker
# thread and embeds on others, so every MLX call goes to this one thread instead. A job is one
# batch, so a search waits behind at most one batch of indexing, not a whole part.
_MLX_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


def _on_mlx_thread[T](call: Callable[..., T], /, *args: Any) -> T:
    """`call(*args)`, run on the MLX thread; the caller waits for it."""
    return _MLX_THREAD.submit(call, *args).result()


@functools.cache
def available() -> bool:
    """Whether MLX and mlx-embeddings are installed here, so the models above can load at all."""
    return all(importlib.util.find_spec(module) is not None for module in ("mlx", "mlx_embeddings"))


def reranker(name: str) -> "Reranker":
    """The MLX reranker `name` is, loaded (and downloaded on first use)."""
    path = _download(name)  # off the MLX thread: a download must not hold up the other models
    return _on_mlx_thread(Reranker, path)


def embedder(name: str) -> "ModernBertEmbedder | JinaV5Embedder":
    """The MLX embedder `name` is, loaded (and downloaded on first use)."""
    path = _download(name)  # see `reranker`
    if name in JINA_V5:
        return _on_mlx_thread(JinaV5Embedder, path)
    return _on_mlx_thread(ModernBertEmbedder, path)


def _download(name: str) -> Path:
    """Where model `name` is on disk, at its pinned revision. Every MLX model loads through here."""
    if not available():
        raise RuntimeError(
            f"{name} runs on MLX, which is not installed: it needs Apple Silicon and "
            "`uv sync --extra mlx`"
        )
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(name, revision=REVISIONS[name]))


def _load(path: Path, **classes: Any) -> tuple[Any, Any]:
    """mlx-embeddings' model downloaded at `path`, with its transformers tokenizer."""
    from mlx_embeddings.utils import load_model
    from transformers import AutoTokenizer

    # transformers types the tokenizer as maybe None; a downloaded repository always has one
    return load_model(path, **classes), AutoTokenizer.from_pretrained(path)


class Reranker:
    """A cross-encoder converted to MLX: one logit per (query, text) pair, in the order given,
    as every reranker answers (`index.cross_encode` turns it into the score)."""

    def __init__(self, path: Path) -> None:
        self._model, self._tokenizer = _load(path, get_model_classes=_with_head)
        # mlx-embeddings' ModernBERT (gte) puts a sigmoid on its one label unless it is told the
        # head is a regression; the logit is what the other rerankers answer
        if hasattr(self._model, "is_regression"):
            self._model.is_regression = True

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for batch in batched(texts, PAIR_BATCH, strict=False):
            encoded = self._tokenizer(
                [query] * len(batch),
                list(batch),
                padding=True,
                truncation="only_second",
                max_length=MAX_PAIR_TOKENS,
                return_tensors="np",
            )
            scores += _on_mlx_thread(self._forward, encoded)
        return scores

    def _forward(self, encoded: dict[str, Any]) -> list[float]:
        import mlx.core as mx

        out = self._model(
            mx.array(encoded["input_ids"]), attention_mask=mx.array(encoded["attention_mask"])
        )
        return out.pooler_output.reshape(-1).tolist()


def _with_head(config: dict) -> tuple[Any, ...]:
    """mlx-embeddings' model classes for `config`, with a classification head on XLM-RoBERTa,
    which it lacks: without one, bge-reranker's head weights have nowhere to load. The classes
    come as mlx-embeddings' own `_get_classes` gives them (model, args, then two vision configs),
    whatever its docstring says."""
    from mlx_embeddings.utils import _get_classes

    model, *rest = _get_classes(config)
    if config.get("architectures") != ["XLMRobertaForSequenceClassification"]:
        return (model, *rest)
    import mlx.core as mx
    import mlx.nn as nn

    class Head(nn.Module):
        """transformers' `XLMRobertaClassificationHead`, dropout aside (inference only)."""

        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.dense = nn.Linear(hidden, hidden)
            self.out_proj = nn.Linear(hidden, 1)

        def __call__(self, first: Any) -> Any:
            return self.out_proj(mx.tanh(self.dense(first)))

    class WithHead(model):
        def __init__(self, model_args: Any) -> None:
            super().__init__(model_args)
            # built as transformers builds it for classification: no pooler, the head instead
            self.pooler = None
            self.classifier = Head(model_args.hidden_size)

        def __call__(self, input_ids: Any, attention_mask: Any = None, **kwargs: Any) -> Any:
            out = super().__call__(input_ids, attention_mask=attention_mask, **kwargs)
            out.pooler_output = self.classifier(out.last_hidden_state[:, 0])
            return out

        def sanitize(self, weights: dict) -> dict:
            # a checkpoint as transformers saves it names the encoder `roberta.*`; an MLX
            # conversion has already dropped the prefix, so both load
            return super().sanitize({k.removeprefix("roberta."): v for k, v in weights.items()})

    return (WithHead, *rest)


class ModernBertEmbedder:
    """A sentence-transformers ModernBERT converted to MLX: pooled and normalized by
    mlx-embeddings, as its config asks."""

    def __init__(self, path: Path) -> None:
        self._model, self._tokenizer = _load(path)

    def embed(self, texts: Sequence[str]) -> Iterator[Any]:
        for batch in batched(texts, EMBED_BATCH, strict=False):
            encoded = self._tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=MAX_EMBED_TOKENS,
                return_tensors="np",
            )
            yield from _on_mlx_thread(self._forward, encoded)

    def _forward(self, encoded: dict[str, Any]) -> Any:
        import mlx.core as mx
        import numpy as np

        out = self._model(
            mx.array(encoded["input_ids"]), attention_mask=mx.array(encoded["attention_mask"])
        )
        return np.array(out.text_embeds.astype(mx.float32))

    def query_embed(self, text: str) -> Iterator[Any]:
        return self.embed([text])


class JinaV5Embedder:
    """Jina v5 text nano, through the repository's own `model.py`: it adds the task's prefix
    ("Query: ", "Document: ") and pools the last token itself."""

    def __init__(self, path: Path) -> None:
        import json

        import mlx.core as mx
        from tokenizers import Tokenizer

        spec = importlib.util.spec_from_file_location("haskie_mlx_jina_v5", path / "model.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"no model.py at {path}")
        published = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(published)
        self._model = published.JinaEmbeddingModel(json.loads((path / "config.json").read_text()))
        # a .safetensors file always loads as one dict of arrays
        weights: dict[str, Any] = mx.load(str(path / "model.safetensors"))
        self._model.load_weights(list(weights.items()))
        self._tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))

    def _encode(self, texts: Sequence[str], task: str) -> Iterator[Any]:
        for batch in batched(texts, EMBED_BATCH, strict=False):
            yield from _on_mlx_thread(self._forward, list(batch), task)

    def _forward(self, texts: list[str], task: str) -> Any:
        import mlx.core as mx
        import numpy as np

        out = self._model.encode(
            texts, self._tokenizer, max_length=MAX_EMBED_TOKENS, task_type=task
        )
        return np.array(out.astype(mx.float32))

    def embed(self, texts: Sequence[str]) -> Iterator[Any]:
        return self._encode(texts, "retrieval.passage")

    def query_embed(self, text: str) -> Iterator[Any]:
        return self._encode([text], "retrieval.query")
