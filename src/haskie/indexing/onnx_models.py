"""The ONNX models, run on ONNX Runtime directly: embedders, pooled as each model asks, and
cross-encoders, scored by the export or by a head of its own.

fastembed ran them once. The loading is little enough to own, and owning it lets each model be
pinned, padded, cut and pooled as it asks: Ettin's head, F2LLM's last-token pooling, gte's fixed
padding. Each model's files come at the revision pinned here, as real files in one directory:
ONNX Runtime refuses an external-data file (`model.onnx_data`) that resolves outside the model's
own directory, and the Hugging Face cache keeps every file as a link into a blob store elsewhere.

On Apple Silicon the GPU is reached through ONNX Runtime's WebGPU plugin (`onnxruntime-ep-webgpu`,
beside the standard package), which runs on Metal. `runtime` registers it, and a session takes it
by device.

Every export was checked against sentence-transformers on the original weights: worst cosine
0.9996 for the embedders, reranker scores within 1e-5. On the WebGPU provider each gave the
CPU's output (worst cosine 0.99999) in about half the time (M4 Pro, 512 real chunks; see
`embed`).

Each model has one session, which ONNX Runtime runs from several threads at once on the CPU or
CUDA (four threads embedding with e5-base-v2: 6.5 s, against 15.5 s one at a time). On WebGPU
every session shares one lock: they share one Metal device, and two runs at once (an embed while
a search reranks) abort the process ("A command encoder is already encoding to this command
buffer", measured). A batch is padded to its own longest row: a model's `tokenizer.json` may pad
to a fixed length (gte's pads every pair to 8,000 tokens, measured at 538 s and 33 GB for 5
pairs).
"""

import contextlib
import importlib.util
import json
import math
import struct
import threading
from collections.abc import Iterator, Sequence
from enum import StrEnum
from functools import cache
from itertools import batched
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

WEBGPU = "WebGpuExecutionProvider"
_WEBGPU_LOCK = threading.Lock()  # one run at a time on the Metal device (see the docstring)
BATCH = 16  # texts or pairs a forward pass: faster than 64 or 256 on the CPU and on WebGPU
MAX_EMBED_TOKENS = 1024  # a chunk with its heading path fits well inside, as for MLX and GGUF
MAX_PAIR_TOKENS = 512  # what a cross-encoder reads of a pair; a chunk and a query fit inside


class Pooling(StrEnum):
    """How an embedder's token states become one vector, as its `1_Pooling` config says."""

    CLS = "cls"  # the first token's
    MEAN = "mean"  # the mean of the real tokens'
    LAST = "last"  # the last real token's: a decoder's (F2LLM-v2), padded on the right


class EmbedderPin(NamedTuple):
    revision: str
    pooling: Pooling
    tokens: int  # the model's own context; it reads at most `MAX_EMBED_TOKENS` of a text
    files: tuple[str, ...] = ("onnx/model.onnx",)  # the export, then any external data


class RerankerPin(NamedTuple):
    revision: str
    # The export stops at the encoder (a sentence-transformers CrossEncoder of the newer kind,
    # Ettin): CLS pooling, Dense with GELU, LayerNorm and Dense to one score run here, in numpy,
    # from the head's own weight files. Otherwise the export answers the score itself.
    headed: bool = False


# Hugging Face repository -> what is read of it
EMBEDDERS: dict[str, EmbedderPin] = {
    "ibm-granite/granite-embedding-97m-multilingual-r2": EmbedderPin(
        "835ad14087e140460703cf0fae09f97d469d65c2", Pooling.CLS, 32768
    ),
    "hotchpotch/bekko-embedding-v1-a8m": EmbedderPin(
        "c721113d59a1d91b447450324f51c4b3332c924a", Pooling.MEAN, 8192
    ),
    "hotchpotch/bekko-embedding-v1-a25m": EmbedderPin(
        "44f0b8af0f487acd0ccf1a7cb7ae7a29a6dfc09c", Pooling.MEAN, 8192
    ),
    # IBM publishes no ONNX of the English R2 pair: onnx-community's and sirasagi62's exports
    "onnx-community/granite-embedding-small-english-r2-ONNX": EmbedderPin(
        "1dc7835ba0cb9c76a3618d0bf0c427c97671b3c8",
        Pooling.CLS,
        8192,
        ("onnx/model.onnx", "onnx/model.onnx_data"),
    ),
    "sirasagi62/granite-embedding-english-r2-ONNX": EmbedderPin(
        "82a4b31078bee981a68203e4119f83cb71fa374e", Pooling.CLS, 8192
    ),
    "intfloat/e5-base-v2": EmbedderPin(
        "f52bf8ec8c7124536f0efb74aca902b2995e5bcd", Pooling.MEAN, 512
    ),
    "onnx-community/F2LLM-v2-160M-ONNX": EmbedderPin(
        "7202b3cd72a7f11b4bea90d6c41b67423281360b", Pooling.LAST, 40960
    ),
}
RERANKERS: dict[str, RerankerPin] = {
    # its export is the whole classifier
    "cross-encoder/ms-marco-MiniLM-L2-v2": RerankerPin("1b5cd67b15209f24824c50370e0397743aa9b787"),
    "cross-encoder/ettin-reranker-17m-v1": RerankerPin(
        "9e4aa35321a6dd1a43ca313f500c4b4f7cfb5cc6", headed=True
    ),
    "cross-encoder/ettin-reranker-32m-v1": RerankerPin(
        "b33e5ceb5110773ea9cf5e00c9bedc83a8c2afdd", headed=True
    ),
    "cross-encoder/ettin-reranker-68m-v1": RerankerPin(
        "d166fa88ddde3c42bc3ee92f7df476d941c8204a", headed=True
    ),
    "cross-encoder/ettin-reranker-150m-v1": RerankerPin(
        "025501c4e0f9bbeb4c5b198318e0089ff061cc14", headed=True
    ),
    "Alibaba-NLP/gte-reranker-modernbert-base": RerankerPin(
        "f7481e6055501a30fb19d090657df9ec1f79ab2c"
    ),
    # IBM publishes no ONNX of its reranker either: jrc2139's export
    "jrc2139/granite-embedding-reranker-english-r2-ONNX": RerankerPin(
        "2ab5a1720e856eb8a8f757e041435f2e43907e0b"
    ),
}
# Ettin's head, in the order it runs: Dense with GELU, LayerNorm, Dense to one score
_HEAD_FILES = (
    "2_Dense/model.safetensors",
    "3_LayerNorm/model.safetensors",
    "4_Dense/model.safetensors",
)


def _download(repo: str, revision: str, files: Sequence[str]) -> Path:
    """`files` and the tokenizer of `repo` at `revision`, as real files in one directory."""
    from huggingface_hub import constants, snapshot_download

    target = Path(constants.HF_HOME) / "haskie-onnx" / repo.replace("/", "--") / revision
    snapshot_download(
        repo, revision=revision, local_dir=target, allow_patterns=[*files, "tokenizer.json"]
    )
    return target


@cache
def runtime() -> Any:
    """ONNX Runtime, with its telemetry off and the WebGPU plugin registered where it is
    installed. Every path to it goes through here first, a session included.

    Its telemetry (Microsoft's 1DS SDK) uploads usage events from a thread of its own, and a
    process that exits mid-upload crashes in that thread: `recursive_mutex lock failed`, or a
    segmentation fault (macOS crash reports of the test workers: 4 of 6 runs; none of 8 with it
    off). A local app has no business sending them either."""
    import onnxruntime

    onnxruntime.disable_telemetry_events()
    if importlib.util.find_spec("onnxruntime_ep_webgpu") is not None:
        import onnxruntime_ep_webgpu

        # a plugin: its provider is listed once registered, and a session takes it by device
        onnxruntime.register_execution_provider_library(
            "webgpu", onnxruntime_ep_webgpu.get_library_path()
        )
    return onnxruntime


class _Session:
    """One model's ONNX session and tokenizer, fed what its export asks for."""

    def __init__(self, path: Path, model_file: str, providers: list, tokens: int):
        from tokenizers import Tokenizer

        onnxruntime = runtime()
        model = str(path / model_file)
        wants_webgpu = bool(providers) and providers[0] == WEBGPU
        devices = (
            [one for one in onnxruntime.get_ep_devices() if one.ep_name == WEBGPU]
            if wants_webgpu
            else []
        )
        if devices:
            # a plugin provider is taken by device: a `providers` list leaves it out unannounced
            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3  # it warns of every node it leaves to the CPU
            options.add_provider_for_devices(devices, {})
            self._session = onnxruntime.InferenceSession(model, sess_options=options)
        else:
            others = [one for one in providers if one != WEBGPU]
            self._session = onnxruntime.InferenceSession(model, providers=others)
        self._inputs = {one.name for one in self._session.get_inputs()}
        outputs = [one.name for one in self._session.get_outputs()]
        self._output = "last_hidden_state" if "last_hidden_state" in outputs else outputs[0]
        self._tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        # the pad token the model ships, padded on the right to the batch's longest row
        padding = self._tokenizer.padding or {}
        self._tokenizer.enable_padding(
            direction="right",
            pad_id=padding.get("pad_id", 0),
            pad_token=padding.get("pad_token", "[PAD]"),
        )
        # the longer side of a pair is cut first: the text, unless the query outgrows it
        self._tokenizer.enable_truncation(tokens, strategy="longest_first")
        on_webgpu = self._session.get_providers()[0] == WEBGPU
        self._lock = _WEBGPU_LOCK if on_webgpu else contextlib.nullcontext()

    def run(self, rows: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        """The export's output for `rows` (texts or pairs), and their attention mask."""
        encoded = self._tokenizer.encode_batch(rows)
        ids = np.array([one.ids for one in encoded], dtype=np.int64)
        mask = np.array([one.attention_mask for one in encoded], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:  # BERT's: which of a pair each token belongs to
            feeds["token_type_ids"] = np.array([one.type_ids for one in encoded], dtype=np.int64)
        if "position_ids" in self._inputs:  # a decoder's, counted from 0: padding is on the right
            feeds["position_ids"] = np.broadcast_to(np.arange(ids.shape[1]), ids.shape).copy()
        with self._lock:
            (output,) = self._session.run([self._output], feeds)
        # onnxruntime types an output as any of its kinds; these are dense arrays
        return np.asarray(output), mask


class Embedder:
    """`embed(texts)` and `query_embed(text)`: one normalized vector per text, in order."""

    def __init__(self, name: str, providers: list) -> None:
        pin = EMBEDDERS[name]
        path = _download(name, pin.revision, pin.files)
        tokens = min(pin.tokens, MAX_EMBED_TOKENS)
        self._session = _Session(path, pin.files[0], providers, tokens)
        self._pooling = pin.pooling

    def embed(self, texts: Sequence[str]) -> Iterator[np.ndarray]:
        for batch in batched(texts, BATCH, strict=False):
            states, mask = self._session.run(list(batch))
            yield from pool(states, mask, self._pooling)

    def query_embed(self, text: str) -> Iterator[np.ndarray]:
        return self.embed([text])


def pool(states: np.ndarray, mask: np.ndarray, pooling: Pooling) -> np.ndarray:
    """One unit vector per row of token states (rows, tokens, dims), its padding left out."""
    match pooling:
        case Pooling.CLS:
            pooled = states[:, 0]
        case Pooling.MEAN:
            # in the states' own float type: the int mask would promote a float64 copy of them all
            weights = mask.astype(states.dtype)
            pooled = (weights[:, None, :] @ states)[:, 0] / weights.sum(axis=1, keepdims=True)
        case Pooling.LAST:
            pooled = states[np.arange(len(states)), mask.sum(axis=1) - 1]
    return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)


class CrossEncoder:
    """`rerank(query, texts)`: one score per text, in order, a logit before any sigmoid."""

    def __init__(self, name: str, providers: list) -> None:
        pin = RERANKERS[name]
        files = ("onnx/model.onnx", *(_HEAD_FILES if pin.headed else ()))
        path = _download(name, pin.revision, files)
        self._session = _Session(path, files[0], providers, MAX_PAIR_TOKENS)
        self._head = Head(path) if pin.headed else None

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for batch in batched(texts, BATCH, strict=False):
            output, _ = self._session.run([(query, text) for text in batch])
            # a headed export answers token states, whose first is the CLS; else one logit a row
            score = self._head(output[:, 0]) if self._head else output.reshape(len(batch))
            scores += score.tolist()
        return scores


class Head:
    """Ettin's scoring head: Dense (no bias) with exact GELU, LayerNorm, then Dense to one score."""

    def __init__(self, path: Path) -> None:
        dense, norm, score = (_tensors(path / file) for file in _HEAD_FILES)
        self._dense = dense["linear.weight"]
        self._norm = norm["norm.weight"], norm["norm.bias"]
        self._score = score["linear.weight"], score["linear.bias"]

    def __call__(self, cls: np.ndarray) -> np.ndarray:
        x = cls @ self._dense.T
        x = 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))
        weight, bias = self._norm
        x = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
        x = x * weight + bias
        weight, bias = self._score
        return (x @ weight.T + bias)[:, 0]


# torch's GELU is the exact one, so it takes the exact erf, which numpy lacks: tanh's shortcut
# would not do (about 0.5 ms a batch of 16, beside the model's own hundreds)
_erf = np.vectorize(math.erf, otypes=[np.float32])


def _tensors(path: Path) -> dict[str, np.ndarray]:
    """The F32 tensors of a safetensors file: an 8-byte header length, a JSON header, the data.
    Read here rather than through `safetensors`, which only Apple Silicon installs."""
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
