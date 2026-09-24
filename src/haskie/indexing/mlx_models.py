"""Models that run on Apple Silicon through MLX, for models fastembed cannot run.

fastembed runs ONNX models only. These are published as MLX builds. Rerankers, in two kinds:

- Listwise (Jina v3, v3.5): the query and every candidate go into one prompt, and a candidate
  scores the cosine of two projected hidden states. Each repository ships its own inference code
  (`rerank.py`, and for v3.5 a `modeling.py`), which is run as published rather than ported: it is
  the regime Jina's scores were measured under. The weights are CC BY-NC 4.0, non-commercial only.
- Pairwise (bge-reranker-v2-m3, gte-reranker-modernbert, mMiniLM): ordinary cross-encoders, one
  (query, text) pair a pass, run through mlx-embeddings. Its XLM-RoBERTa has no classification
  head, so one is added here (`_with_head`): the standard one, dense, tanh, then one logit, over
  the first token. It loads an MLX conversion and transformers' own checkpoint alike, so an
  XLM-RoBERTa cross-encoder needs no conversion. All Apache-2.0.

Embedders (`EMBEDDERS`), each answering `embed(texts)` and `query_embed(text)` the way fastembed's
`TextEmbedding` does, so `embed` calls them the same way:

- nomic ModernBERT embed, through mlx-embeddings, which mean-pools as the model's config asks and
  normalizes. Its search prefixes are the profile's (`EmbeddingModel.query_prefix`).
- Jina v5 text nano (retrieval): its own `model.py`, run as published, which adds its task
  prefixes itself. CC BY-NC 4.0, non-commercial only.

Every repository is pinned to a revision, so the code and weights that run are the ones reviewed.
Needs the `mlx` extra (mlx-lm and mlx-embeddings), which only installs on Apple Silicon.
"""

import functools
import importlib.util
import sys
import threading
from collections.abc import Iterator, Sequence
from itertools import batched
from pathlib import Path
from typing import Any

# Hugging Face repository -> the revision its code and weights are read at
LISTWISE: dict[str, str] = {
    "jinaai/jina-reranker-v3-mlx": "1d19fe38ae4e6658221479747c1152d6136dd6ab",
    "jinaai/jina-reranker-v3.5-mlx": "3dd4ac901ccdcac85abe3815df0a0aaaf44e4a21",
}
PAIRWISE: dict[str, str] = {
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
REVISIONS = LISTWISE | PAIRWISE | EMBEDDERS
EMBED_BATCH = 16  # texts a forward pass

PAIR_BATCH = 16  # pairs a forward pass: the candidates of one search in a few passes
MAX_PAIR_TOKENS = 512  # what bge-reranker-v2-m3 was tuned at; a chunk and a query fit well inside
MAX_EMBED_TOKENS = 1024  # a chunk with its heading path fits well inside; both models read 8K


@functools.cache
def available() -> bool:
    """Whether MLX and both runtimes are installed here, so the models above can load at all."""
    return all(
        importlib.util.find_spec(module) is not None
        for module in ("mlx", "mlx_lm", "mlx_embeddings")
    )


def loadable(name: str) -> bool:
    """Whether model `name` can load here: any model that is not MLX's, or MLX's where it runs."""
    return name not in REVISIONS or available()


def reranker(name: str) -> "ListwiseReranker | PairwiseReranker":
    """The MLX reranker `name` is, loaded (and downloaded on first use)."""
    return ListwiseReranker(name) if name in LISTWISE else PairwiseReranker(name)


def embedder(name: str) -> "ModernBertEmbedder | JinaV5Embedder":
    """The MLX embedder `name` is, loaded (and downloaded on first use)."""
    return JinaV5Embedder(name) if name in JINA_V5 else ModernBertEmbedder(name)


def _download(name: str) -> Path:
    """Where model `name` is on disk, at its pinned revision. Every MLX model loads through here."""
    if not available():
        raise RuntimeError(
            f"{name} runs on MLX, which is not installed: it needs Apple Silicon and "
            "`uv sync --extra mlx`"
        )
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(name, revision=REVISIONS[name]))


def _load(name: str, **classes: Any) -> tuple[Any, Any]:
    """mlx-embeddings' model for `name`, with its transformers tokenizer."""
    from mlx_embeddings.utils import load_model
    from transformers import AutoTokenizer

    path = _download(name)
    # transformers types the tokenizer as maybe None; a downloaded repository always has one
    return load_model(path, **classes), AutoTokenizer.from_pretrained(path)


class ListwiseReranker:
    """One of Jina's listwise rerankers, shaped like fastembed's cross-encoder: `rerank` answers
    one score per text, in the order given. A score is a cosine, so it lies in [-1, 1]."""

    def __init__(self, name: str) -> None:
        path = _download(name)
        self._model = _published(path, name).MLXReranker(
            model_path=str(path), projector_path=str(path / "projector.safetensors")
        )
        # one Metal command queue per model: two searches scoring at once would interleave on it
        self._lock = threading.Lock()

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        with self._lock:
            ranked = self._model.rerank(query, list(texts))
        scores = [0.0] * len(texts)
        for found in ranked:  # best first, each with the index it had in `texts`
            scores[found["index"]] = float(found["relevance_score"])
        return scores


def _published(path: Path, name: str, module: str = "rerank") -> Any:
    """The repository's own `rerank.py` (or `module`), imported from where it was downloaded.

    Under a name of its own, so two models' `rerank` never share a module. Its directory is on the
    path while it runs, because v3.5's `rerank.py` imports the `modeling.py` beside it.
    """
    spec = importlib.util.spec_from_file_location(
        f"haskie_mlx_{name.replace('/', '_').replace('.', '_').replace('-', '_')}",
        path / f"{module}.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{name} ships no {module}.py at {path}")
    loaded = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path))
    try:
        spec.loader.exec_module(loaded)
    finally:
        sys.path.remove(str(path))
    return loaded


class PairwiseReranker:
    """A cross-encoder converted to MLX: one score per (query, text) pair, in the order given,
    as the model's own head answers it (bge a logit, gte a probability)."""

    def __init__(self, name: str) -> None:
        self._model, self._tokenizer = _load(name, get_model_classes=_with_head)
        self._lock = threading.Lock()  # see `ListwiseReranker`

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        import mlx.core as mx

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
            with self._lock:
                out = self._model(
                    mx.array(encoded["input_ids"]),
                    attention_mask=mx.array(encoded["attention_mask"]),
                )
                scores += out.pooler_output.reshape(-1).tolist()
        return scores


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

    def __init__(self, name: str) -> None:
        self._model, self._tokenizer = _load(name)
        self._lock = threading.Lock()  # see `ListwiseReranker`

    def embed(self, texts: Sequence[str]) -> Iterator[Any]:
        import mlx.core as mx
        import numpy as np

        for batch in batched(texts, EMBED_BATCH, strict=False):
            encoded = self._tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=MAX_EMBED_TOKENS,
                return_tensors="np",
            )
            with self._lock:
                out = self._model(
                    mx.array(encoded["input_ids"]),
                    attention_mask=mx.array(encoded["attention_mask"]),
                )
                vectors = np.array(out.text_embeds.astype(mx.float32))
            yield from vectors

    def query_embed(self, text: str) -> Iterator[Any]:
        return self.embed([text])


class JinaV5Embedder:
    """Jina v5 text nano, through the repository's own `model.py`: it adds the task's prefix
    ("Query: ", "Document: ") and pools the last token itself."""

    def __init__(self, name: str) -> None:
        import json

        import mlx.core as mx
        from tokenizers import Tokenizer

        path = _download(name)
        self._model = _published(path, name, "model").JinaEmbeddingModel(
            json.loads((path / "config.json").read_text())
        )
        # a .safetensors file always loads as one dict of arrays; mlx types every format's shape
        weights: dict[str, Any] = mx.load(str(path / "model.safetensors"))  # ty: ignore[invalid-assignment]
        self._model.load_weights(list(weights.items()))
        self._tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self._lock = threading.Lock()  # see `ListwiseReranker`

    def _encode(self, texts: Sequence[str], task: str) -> Iterator[Any]:
        import mlx.core as mx
        import numpy as np

        for batch in batched(texts, EMBED_BATCH, strict=False):
            with self._lock:
                out = self._model.encode(
                    list(batch), self._tokenizer, max_length=MAX_EMBED_TOKENS, task_type=task
                )
                vectors = np.array(out.astype(mx.float32))
            yield from vectors

    def embed(self, texts: Sequence[str]) -> Iterator[Any]:
        return self._encode(texts, "retrieval.passage")

    def query_embed(self, text: str) -> Iterator[Any]:
        return self._encode([text], "retrieval.query")
