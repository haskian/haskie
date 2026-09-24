"""Cross-encoders whose published ONNX file stops before the score, finished here.

A sentence-transformers CrossEncoder of the newer kind (ettin-reranker) is an encoder followed by
modules of its own: CLS pooling, Dense with GELU, LayerNorm, Dense to one score. Its `onnx/`
export is the encoder alone, so fastembed would answer hidden states rather than scores. Here the
encoder runs on ONNX Runtime and the head, four small matrix steps, in numpy, from the head's own
weight files. Each model is pinned to a revision, so the weights that run are the ones reviewed.
"""

import json
import math
import struct
import threading
from collections.abc import Sequence
from itertools import batched
from pathlib import Path

import numpy as np

# Hugging Face repository -> the revision its encoder and head are read at
REVISIONS: dict[str, str] = {
    "cross-encoder/ettin-reranker-68m-v1": "d166fa88ddde3c42bc3ee92f7df476d941c8204a",
}
BATCH = 16  # pairs a forward pass
MAX_TOKENS = 512  # a chunk and a query fit well inside; the model reads 8K
_FILES = [
    "onnx/model.onnx",
    "tokenizer.json",
    "2_Dense/model.safetensors",
    "3_LayerNorm/model.safetensors",
    "4_Dense/model.safetensors",
]


def _erf(x: np.ndarray) -> np.ndarray:
    """erf over a whole array, which numpy lacks: Abramowitz and Stegun 7.1.26, within 1.5e-7,
    below float32's own noise. torch's GELU is the exact one, so tanh's shortcut would not do."""
    t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
    tail = -1.453152027 + t * 1.061405429
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * tail)))
    return np.sign(x) * (1.0 - poly * np.exp(-x * x))


class HeadedCrossEncoder:
    """`rerank(query, texts)` answers one score per text, in order, as fastembed's does."""

    def __init__(self, name: str, providers: list) -> None:
        import onnxruntime
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        path = Path(snapshot_download(name, revision=REVISIONS[name], allow_patterns=_FILES))
        self._session = onnxruntime.InferenceSession(
            str(path / "onnx/model.onnx"), providers=providers
        )
        self._tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self._tokenizer.enable_truncation(MAX_TOKENS, strategy="only_second")
        self._tokenizer.enable_padding()
        self._dense = _tensors(path / "2_Dense/model.safetensors")["linear.weight"]
        norm = _tensors(path / "3_LayerNorm/model.safetensors")
        self._norm = norm["norm.weight"], norm["norm.bias"]
        score = _tensors(path / "4_Dense/model.safetensors")
        self._score = score["linear.weight"], score["linear.bias"]
        self._lock = threading.Lock()  # one session, one run at a time

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for batch in batched(texts, BATCH, strict=False):
            pairs = [(query, text) for text in batch]
            encoded = self._tokenizer.encode_batch(pairs)
            feeds = {
                "input_ids": np.array([one.ids for one in encoded], dtype=np.int64),
                "attention_mask": np.array([one.attention_mask for one in encoded], dtype=np.int64),
            }
            with self._lock:
                (hidden,) = self._session.run(["last_hidden_state"], feeds)
            # onnxruntime types an output as any of its kinds; this one is a dense array
            scores += self._head(np.asarray(hidden)[:, 0]).tolist()
        return scores

    def _head(self, cls: np.ndarray) -> np.ndarray:
        """Dense (no bias) with exact GELU, LayerNorm, then Dense to one score."""
        x = cls @ self._dense.T
        x = 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))
        weight, bias = self._norm
        x = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
        x = x * weight + bias
        weight, bias = self._score
        return (x @ weight.T + bias)[:, 0]


def _tensors(path: Path) -> dict[str, np.ndarray]:
    """The F32 tensors of a safetensors file: an 8-byte header length, a JSON header, the data.
    Read here rather than through `safetensors`, which only the mlx extra installs."""
    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + size])
    body = memoryview(raw)[8 + size :]
    found: dict[str, np.ndarray] = {}
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        if spec["dtype"] != "F32":
            raise ValueError(f"{path}: {key} is {spec['dtype']}, only F32 is read")
        start, end = spec["data_offsets"]
        found[key] = np.frombuffer(body[start:end], dtype=np.float32).reshape(spec["shape"])
    return found
