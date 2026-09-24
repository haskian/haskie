"""Text embeddings with fastembed (ONNX). Vectors are computed here, never inside LanceDB, so the
embed stage can run in parallel and the index stage only writes rows.

Hardware: ONNX Runtime execution providers, best available first. CUDA needs the `gpu` extra
(onnxruntime-gpu); Apple Silicon uses CoreML from the default wheel; otherwise CPU.

Building a session holds the GIL for its whole duration (measured: the loop that owns the request
freezes for it), and CoreML adds a model compilation to that. Two things keep the freeze short:
CoreML keeps its compiled models in `home.MODEL_CACHE`, so only the first build of a model pays
for the compilation, and cross-encoders always build on CPU: they score at most `candidates` texts
per search, which CPU does in milliseconds, while their CoreML build stalled the app for seconds.
"""

import threading
from functools import cache, lru_cache

from haskie import home
from haskie.settings import Accelerator, EmbeddingModel

# a provider entry as ONNX Runtime takes it: a name, or a (name, options) pair
Provider = str | tuple[str, dict[str, str]]
RERANKER_ACCELERATOR = Accelerator.CPU

# Construction (download, session setup) is serialized, using a model is not, and the lock is
# outside the cache so the second caller of a model being built waits and then gets that one:
# fastembed downloads into one shared cache directory and several worker threads ask for the same
# model at once, so without this each of them starts its own download into the same files.
_MODEL_LOCK = threading.Lock()


PREFERENCE = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)


def select_providers(available: list[str], accelerator: Accelerator) -> list[str]:
    """Ordered provider list for ONNX Runtime; CPU is always the final fallback."""
    if accelerator == Accelerator.CPU:
        return ["CPUExecutionProvider"]
    chosen = [p for p in PREFERENCE if p in available]
    return chosen if "CPUExecutionProvider" in chosen else [*chosen, "CPUExecutionProvider"]


def with_options(names: list[str], model_cache: str) -> list[Provider]:
    """Attach the options a provider needs; today only CoreML's compiled-model cache."""
    return [
        (name, {"ModelCacheDirectory": model_cache}) if name == "CoreMLExecutionProvider" else name
        for name in names
    ]


def providers(accelerator: Accelerator = Accelerator.AUTO) -> list[Provider]:
    import onnxruntime

    names = select_providers(onnxruntime.get_available_providers(), accelerator)
    return with_options(names, str(home.MODEL_CACHE))


def provider_name(provider: Provider) -> str:
    return provider if isinstance(provider, str) else provider[0]


@lru_cache(maxsize=2)
def device_name(accelerator: Accelerator = Accelerator.AUTO) -> str:
    return provider_name(providers(accelerator)[0]).removesuffix("ExecutionProvider")


@cache
def _build_model(name: str, accelerator: Accelerator):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=name, providers=providers(accelerator))


@cache
def _build_cross_encoder(name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=name, providers=providers(RERANKER_ACCELERATOR))


def _model(name: str, accelerator: Accelerator):
    with _MODEL_LOCK:
        return _build_model(name, accelerator)


def _cross_encoder(name: str):
    with _MODEL_LOCK:
        return _build_cross_encoder(name)


def embed_texts(model: EmbeddingModel, texts: list[str]) -> list[list[float]]:
    """Document-side embeddings (one vector per text)."""
    if not texts:
        return []
    embedder = _model(model.name, model.accelerator)
    return [v.tolist() for v in embedder.embed(texts)]


def embed_query(model: EmbeddingModel, text: str) -> list[float]:
    """Query-side embedding (models like bge prepend a query instruction here)."""
    return next(iter(_model(model.name, model.accelerator).query_embed(text))).tolist()


def warm(name: str, accelerator: Accelerator) -> None:
    """Load the model now, downloading it on a cold cache, so the first embed does not stall."""
    _model(name, accelerator)


def warm_reranker(name: str) -> None:
    _cross_encoder(name)


def rerank_scores(model_name: str, query: str, texts: list[str]) -> list[float]:
    """Cross-encoder relevance of each text to the query (higher = better; not normalized)."""
    if not texts:
        return []
    return [float(s) for s in _cross_encoder(model_name).rerank(query, texts)]
