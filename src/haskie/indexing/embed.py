"""Text embeddings and reranker scores. Vectors are computed here, never inside LanceDB, so the
embed stage can run in parallel and the index stage only writes rows.

Each model loads through its runtime (`hardware.runtime`): `onnx_models` on ONNX Runtime,
`mlx_models` on MLX, `gguf_models` on llama.cpp. Embedders and rerankers take the same hardware
setting the same way.

Hardware: ONNX Runtime execution providers. `auto` takes CUDA where it runs, else WebGPU, else
the CPU. Linux installs ONNX Runtime's CUDA build, which runs on the CPU where no NVIDIA GPU is.
Apple Silicon adds ONNX Runtime's WebGPU plugin (`onnxruntime-ep-webgpu`), which reaches the GPU
through Metal. Measured on 512 real chunks (M4 Pro, batches of 16), WebGPU against the CPU:
granite-97m 5.6 s against 14.3 s, bekko-a8m 2.0 s against 4.1 s, F2LLM-160M 16.2 s against 32.0 s,
ettin-150m 23.4 s against 51.6 s, every vector the CPU's (worst cosine 0.99999). The MLX and GGUF
profiles (`mlx_models`, `gguf_models`) reach the same GPU faster still, for the models they hold.

CoreML runs only when the settings ask for it (`Accelerator.COREML`), and only for a model it was
measured to run (`hardware.COREML_RUNS`): none of today's catalogue. It runs a transformer on the
GPU only at fixed sizes, and haskie's sizes are dynamic, so CoreML takes a fraction of each
graph, in pieces (arctic-embed-m: 401 of 623 nodes in 74 pieces). Measured on 64 real chunks
against the CPU: bge-small and bge-base ran as fast, jina-v2-small 4x slower, arctic-embed-m 8x
slower. At fixed sizes CoreML ran 3-4x faster, but compiled each model for 8 s to 14 minutes
first. The setting stays for the day CoreML handles dynamic sizes well.

Building a session holds the GIL for its whole duration (measured: the loop that owns the request
freezes for it), and CoreML adds a model compilation to that. CoreML keeps its compiled models in
`home.MODEL_CACHE`, so only the first build of a model pays for it.

Two lifetimes. A search model (the embedding profile's, a reranker) stays loaded for the life of
the process: every search needs it at once. A knowledge model (`gguf_models.ON_DEMAND`: the
describer, the vocabulary's embedder) is loaded when it is first asked, and freed once nobody has
used it for a while (`free_idle`, which `models` calls): it serves indexing alone, and the
describer holds 2.8 GB.
"""

import ctypes
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cache, partial
from pathlib import Path
from typing import Any

import numpy as np

from haskie import home
from haskie.catalogue.catalogue import EmbeddingModel
from haskie.indexing import gguf_models, hardware, mlx_models, onnx_models
from haskie.indexing.hardware import Runtime, runtime
from haskie.settings import Accelerator

# a provider entry as ONNX Runtime takes it: a name, or a (name, options) pair
Provider = str | tuple[str, dict[str, str]]

# Construction (download, session setup) is serialized, using a model is not, and the lock is
# outside the cache so the second caller of a model being built waits and then gets that one:
# a model downloads into one directory and several worker threads ask for the same model at once,
# so without this each of them starts its own download into the same files.
_MODEL_LOCK = threading.Lock()


COREML = "CoreMLExecutionProvider"
# The NVIDIA providers, each by the library of ONNX Runtime's that must load for it to run.
NVIDIA_LIBRARIES = {
    "TensorrtExecutionProvider": "libonnxruntime_providers_tensorrt.so",
    "CUDAExecutionProvider": "libonnxruntime_providers_cuda.so",
}
# what `auto` takes, best first: CoreML is not among them (see the module docstring)
PREFERENCE = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    onnx_models.WEBGPU,  # Metal on Apple Silicon, through ONNX Runtime's WebGPU plugin
    "CPUExecutionProvider",
)


def select_providers(available: list[str], accelerator: Accelerator) -> list[str]:
    """Ordered provider list for ONNX Runtime; CPU is always the final fallback."""
    if accelerator == Accelerator.CPU:
        return ["CPUExecutionProvider"]
    if accelerator == Accelerator.COREML:
        return [COREML, "CPUExecutionProvider"] if COREML in available else ["CPUExecutionProvider"]
    chosen = [p for p in PREFERENCE if p in available]
    return chosen if "CPUExecutionProvider" in chosen else [*chosen, "CPUExecutionProvider"]


def with_options(names: list[str], model_cache: str) -> list[Provider]:
    """Attach the options a provider needs; today only CoreML's compiled-model cache."""
    return [
        (name, {"ModelCacheDirectory": model_cache}) if name == COREML else name for name in names
    ]


def onnx_runtime() -> Any:
    """ONNX Runtime, set up (`onnx_models.runtime`). Every path to it goes through there first."""
    return onnx_models.runtime()


@cache
def nvidia_loads(provider: str) -> bool:
    """Whether NVIDIA provider `provider` runs here: the driver, and the libraries its ONNX
    Runtime library links to (CUDA and cuDNN; TensorRT's own for TensorRT).

    ONNX Runtime's CUDA build lists both on every machine. A session asked for one that cannot
    load logs an error and falls back, while the status would report the GPU. Loading the
    provider's own library follows whichever versions the installed build needs.
    """
    library = Path(onnx_runtime().__file__).parent / "capi" / NVIDIA_LIBRARIES[provider]
    try:
        ctypes.CDLL("libcuda.so.1")  # the driver's, present only with an NVIDIA driver
        ctypes.CDLL(str(library))
    except OSError:
        return False
    return True


def providers(accelerator: Accelerator = Accelerator.AUTO) -> list[Provider]:
    available = onnx_runtime().get_available_providers()
    # Only `auto` may pick an NVIDIA provider, so only it loads their libraries to find out.
    if accelerator == Accelerator.AUTO:
        available = [p for p in available if p not in NVIDIA_LIBRARIES or nvidia_loads(p)]
    return with_options(select_providers(available, accelerator), str(home.MODEL_CACHE))


def provider_name(provider: Provider) -> str:
    return provider if isinstance(provider, str) else provider[0]


def model_providers(name: str, accelerator: Accelerator) -> list[Provider]:
    """The providers ONNX model `name` runs on: `providers`, less CoreML for a model CoreML was
    not measured to run (`hardware.COREML_RUNS`)."""
    chosen = providers(accelerator)
    if name in hardware.COREML_RUNS:
        return chosen
    return [one for one in chosen if provider_name(one) != COREML]


@cache
def _build_model(name: str, accelerator: Accelerator):
    """The embedder `name` is, on its runtime: each answers `embed(texts)` and
    `query_embed(text)` (`onnx_models`, `mlx_models`, `gguf_models`)."""
    match runtime(name):
        case Runtime.MLX:
            return mlx_models.embedder(name)
        case Runtime.GGUF:
            return gguf_models.GgufEmbedder(name)
    return onnx_models.Embedder(name, model_providers(name, accelerator))


@cache
def _build_cross_encoder(name: str, accelerator: Accelerator):
    """The cross-encoder `name` is, on its runtime: each answers `rerank(query, texts)` with one
    score per text, in order (`onnx_models`, `mlx_models`)."""
    if runtime(name) == Runtime.MLX:
        return mlx_models.reranker(name)
    return onnx_models.CrossEncoder(name, model_providers(name, accelerator))


def _model(name: str, accelerator: Accelerator):
    _check_runs(name, accelerator)
    with _MODEL_LOCK:
        return _build_model(name, accelerator)


def _cross_encoder(name: str, accelerator: Accelerator):
    _check_runs(name, accelerator)
    with _MODEL_LOCK:
        return _build_cross_encoder(name, accelerator)


def _check_runs(name: str, accelerator: Accelerator) -> None:
    """Refuse a model `hardware.device` finds no device for, before anything downloads."""
    if hardware.device(name, accelerator) is None:
        raise RuntimeError(hardware.nowhere(name))


type Key = tuple[str, Accelerator]


class OnDemand:
    """Knowledge models, loaded when first asked and freed once idle (see the module). A model in
    use is never freed: `use` counts its users, so a describe batch keeps its describer for its
    whole length, and each use pushes its idle time back."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[Key, Any] = {}
        self._users: Counter[Key] = Counter()
        self._last: dict[Key, float] = {}
        self._building: dict[Key, threading.Lock] = {}

    def _take(self, key: Key, build: Callable[[], Any]) -> Any:
        with self._lock:
            building = self._building.setdefault(key, threading.Lock())
        # One build of a model at a time, a waiter getting the one just built. Not under
        # `_MODEL_LOCK`, which every search takes: a search never waits out a load of seconds.
        with building:
            with self._lock:
                if key in self._models:
                    self._users[key] += 1
                    return self._models[key]
            model = build()
            with self._lock:
                self._models[key] = model
                self._users[key] += 1
            return model

    @contextmanager
    def use(self, key: Key, build: Callable[[], Any]) -> Iterator[Any]:
        model = self._take(key, build)
        try:
            yield model
        finally:
            with self._lock:
                self._users[key] -= 1
                self._last[key] = time.monotonic()

    def loaded(self, name: str) -> bool:
        with self._lock:
            return any(key[0] == name for key in self._models)

    def free_idle(self, idle_seconds: float) -> list[str]:
        """Free every model nobody is using and nobody used for `idle_seconds`; their names. A
        model is closed outside the lock: no caller can take it once it is out of the table."""
        now = time.monotonic()
        with self._lock:
            idle = [
                key
                for key in self._models
                if not self._users[key] and now - self._last[key] >= idle_seconds
            ]
            freed = [self._models.pop(key) for key in idle]
            for key in idle:
                self._last.pop(key, None)
                self._users.pop(key, None)
        for model in freed:
            if (close := getattr(model, "close", None)) is not None:
                close()
        return [name for name, _ in idle]


_on_demand = OnDemand()


def loaded(name: str) -> bool:
    """Whether knowledge model `name` is in memory now (`OnDemand`)."""
    return _on_demand.loaded(name)


def free_idle(idle_seconds: float) -> list[str]:
    """Free the knowledge models idle for `idle_seconds`; their names (`OnDemand.free_idle`)."""
    return _on_demand.free_idle(idle_seconds)


@contextmanager
def _embedder(name: str, accelerator: Accelerator) -> Iterator[Any]:
    """The embedder `name` is, held for the length of the block: a knowledge model is loaded on
    demand and counted as in use, a search model is the one loaded for the process."""
    if name not in gguf_models.ON_DEMAND:
        yield _model(name, accelerator)
        return
    _check_runs(name, accelerator)
    with _on_demand.use((name, accelerator), partial(gguf_models.GgufEmbedder, name)) as model:
        yield model


def embed_texts(model: EmbeddingModel, texts: list[str]) -> list[list[float]]:
    """Document-side embeddings (one vector per text), each after the model's document prefix."""
    if not texts:
        return []
    order = by_length(texts)
    with _embedder(model.name, model.accelerator) as embedder:
        vectors = list(embedder.embed([model.document_prefix + texts[i] for i in order]))
    return in_order(order, [_cut(model, v).tolist() for v in vectors])


def by_length(texts: list[str]) -> list[int]:
    """The positions of `texts`, shortest first. A batch pads to its longest row, so texts of
    like length batched together pad least (measured on 512 real chunks: 94k padded tokens
    against 166k in document order, 17% faster)."""
    return sorted(range(len(texts)), key=lambda i: len(texts[i]))


def in_order[T](order: list[int], results: list[T]) -> list[T]:
    """`results`, answered in `order`, put back in the order of the texts they answer."""
    placed: list[T] = [results[0]] * len(results)
    for position, result in zip(order, results, strict=True):
        placed[position] = result
    return placed


def embed_query(model: EmbeddingModel, text: str) -> list[float]:
    """Query-side embedding, after the model's query prefix. `query_embed` rather than `embed`:
    a multi-task model picks its query adapter there (`mlx_models.JinaV5Embedder`)."""
    with _embedder(model.name, model.accelerator) as embedder:
        vector = next(iter(embedder.query_embed(model.query_prefix + text)))
    return _cut(model, vector).tolist()


def _cut(model: EmbeddingModel, vector: np.ndarray) -> np.ndarray:
    """`vector` as the model's profile stores it: whole, or cut to its Matryoshka size and
    normalized again."""
    if not model.matryoshka:
        return vector
    cut = vector[: model.dims]
    return cut / np.linalg.norm(cut)


def warm(name: str, accelerator: Accelerator) -> None:
    """Load the model now, downloading it on a cold cache, so the first embed does not stall."""
    with _embedder(name, accelerator):
        pass


def warm_reranker(name: str, accelerator: Accelerator) -> None:
    _cross_encoder(name, accelerator)


def warm_generator(name: str, accelerator: Accelerator) -> None:
    with _generator(name, accelerator):
        pass


@contextmanager
def _generator(name: str, accelerator: Accelerator) -> Iterator[gguf_models.GgufGenerator]:
    """The generator `name` is, a knowledge model: loaded on demand, held for the block."""
    _check_runs(name, accelerator)
    with _on_demand.use((name, accelerator), partial(gguf_models.GgufGenerator, name)) as model:
        yield model


def reply(name: str, accelerator: Accelerator, prompt: str, max_tokens: int) -> str:
    """The generator's greedy reply to one prompt (`gguf_models.GgufGenerator`)."""
    with _generator(name, accelerator) as model:
        return model.reply(prompt, max_tokens)


def yes(name: str, accelerator: Accelerator, prompt: str) -> float:
    """How likely the generator's answer to a yes-or-no question is yes
    (`gguf_models.GgufGenerator.yes`)."""
    with _generator(name, accelerator) as model:
        return model.yes(prompt)


def rerank_scores(
    model_name: str, accelerator: Accelerator, query: str, texts: list[str]
) -> list[float]:
    """Cross-encoder relevance of each text to the query (higher = better; not normalized)."""
    if not texts:
        return []
    order = by_length(texts)
    scores = _cross_encoder(model_name, accelerator).rerank(query, [texts[i] for i in order])
    return in_order(order, [float(score) for score in scores])
