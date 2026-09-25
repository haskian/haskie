"""Embedders that run on the Apple GPU through llama.cpp, from GGUF files.

ONNX Runtime reaches the Apple GPU only through CoreML, which must first compile each model for
minutes (see `embed`). llama.cpp runs a GGUF file on Metal as it loads: a model is ready in at most
0.27 s, after a one-time Metal shader compile per machine (7.6 s, measured once, 0.1 s after).
Measured on 128 real chunks against the same model in ONNX on the CPU (M4 Pro):

- bge-small, ggml-org Q8_0: 2.8 s against 3.75 s (1.3x), worst cosine to the ONNX vectors 0.99989
- bge-m3, ggml-org Q8_0: 4.2 s against 19.6 s (4.7x), worst cosine 0.99904
- jina-v2-base, ggml-org Q8_0: 1.5 s against 8.1 s (5.4x), worst cosine 0.99986
- nomic v1.5, nomic-ai f16: 1.5 s against 7.6 s (5.2x), worst cosine 1.00000; nomic's own Q8_0
  file drifts to 0.9985, so the f16 one is pinned

Only official conversions are pinned (ggml-org, and nomic-ai for its own model), each to a
revision, so the weights that run are the ones measured. Each file carries its model's pooling,
which llama.cpp applies. The files run on the CPU too, but slower than ONNX there (arctic: 27 s
against 6 s), so they run on Metal only: `hardware.device` finds no device for them otherwise.

llama.cpp computes attention over a whole micro-batch, every token against every other, so a
larger micro-batch costs more: 512 tokens ran fastest, 1024 cost 12% more, 8192 was 17x slower.
So texts are read at up to `MAX_TOKENS`, as the MLX embedders read theirs.

The `gguf` extra installs llama-cpp-python on Apple Silicon only. PyPI ships it as source, so the
install compiles llama.cpp with Metal: about 30 s, with the Xcode command-line tools and cmake.
"""

import functools
import importlib.util
import threading
from collections.abc import Iterator, Sequence
from itertools import batched
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

MAX_TOKENS = 1024  # the longest text read, in tokens: see the module docstring
SEQUENCES = 64  # texts one micro-batch holds, as many as fit its tokens


class Pin(NamedTuple):
    """One GGUF file, at the revision measured, and the longest input its model reads."""

    revision: str
    file: str
    tokens: int  # the model's own context: the file holds no positions past it


PINS: dict[str, Pin] = {
    "ggml-org/bge-small-en-v1.5-Q8_0-GGUF": Pin(
        "f2068edd9b54f2a369549ccc71f70ed273a2a801", "bge-small-en-v1.5-q8_0.gguf", 512
    ),
    "ggml-org/bge-m3-Q8_0-GGUF": Pin(
        "9eba04c5d75ba5a1595e45de734d36bef4e5cb98", "bge-m3-q8_0.gguf", 8192
    ),
    "ggml-org/jina-embeddings-v2-base-en-Q8_0-GGUF": Pin(
        "c4e88d968641ee4fa4941e083980160867fd8bf2", "jina-embeddings-v2-base-en-q8_0.gguf", 8192
    ),
    "nomic-ai/nomic-embed-text-v1.5-GGUF": Pin(
        "0188c9bf409793f810680a5a431e7b899c46104c", "nomic-embed-text-v1.5.f16.gguf", 8192
    ),
}


@functools.cache
def available() -> bool:
    """Whether llama.cpp is installed here. The `gguf` extra installs it on Apple Silicon only,
    where it builds with Metal, so installed means it runs on the GPU. A spec lookup, not an
    import: the import starts Metal, which the first time on a machine takes seconds."""
    return importlib.util.find_spec("llama_cpp") is not None


def _download(name: str) -> Path:
    """Where the pinned file of model `name` is on disk, downloaded on first use."""
    if not available():
        raise RuntimeError(
            f"{name} runs on llama.cpp with Metal, which is not installed: it needs Apple Silicon "
            "and `uv sync --extra gguf`"
        )
    from huggingface_hub import hf_hub_download

    pin = PINS[name]
    return Path(hf_hub_download(name, pin.file, revision=pin.revision))


class GgufEmbedder:
    """A GGUF embedder, answering `embed(texts)` and `query_embed(text)` the way fastembed's
    `TextEmbedding` does: one normalized vector per text, pooled as the file says."""

    def __init__(self, name: str) -> None:
        path = _download(name)  # first: it says what to install when llama.cpp is missing
        from llama_cpp import Llama

        tokens = min(PINS[name].tokens, MAX_TOKENS)
        self._model = Llama(
            model_path=str(path),
            embedding=True,
            n_gpu_layers=-1,  # every layer on the GPU: `hardware.device` keeps these off the CPU
            # one micro-batch is one context: texts are packed into it, and cut to it
            n_ctx=tokens,
            n_batch=tokens,
            n_ubatch=tokens,
            n_seq_max=SEQUENCES,
            verbose=False,
        )
        # one context: two calls at once would decode into the same buffers
        self._lock = threading.Lock()

    def embed(self, texts: Sequence[str]) -> Iterator[Any]:
        # locked a batch at a time, so a search's query waits for one batch, not a whole part:
        # 128 bge-m3 chunks took 4.58 s in one call and 4.60 s in batches, with the same vectors
        for batch in batched(texts, SEQUENCES, strict=False):
            with self._lock:  # llama.cpp releases the GIL while it runs (measured: 13 ms stalls)
                vectors = self._model.embed(list(batch), normalize=True, truncate=True)
            yield from (np.asarray(vector, dtype=np.float32) for vector in vectors)

    def query_embed(self, text: str) -> Iterator[Any]:
        return self.embed([text])
