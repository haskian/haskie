"""Text embeddings and reranker scores. Vectors are computed here, never inside LanceDB, so the
embed stage can run in parallel and the index stage only writes rows.

Each model loads through its runtime (`hardware.runtime`): fastembed or `onnx_rerank` on ONNX
Runtime, `mlx_models` on MLX, `gguf_models` on llama.cpp. Embedders and rerankers take the same
hardware setting the same way.

Hardware: ONNX Runtime execution providers. `auto` takes CUDA where it runs, else the CPU. Linux
installs ONNX Runtime's CUDA build, which runs on the CPU where no NVIDIA GPU is. On Apple Silicon
the GPU is reached through other runtimes: the MLX profiles (`mlx_models`) and the GGUF profiles
(`gguf_models`).

CoreML runs only when the settings ask for it (`Accelerator.COREML`). It runs a transformer on the
GPU only at fixed sizes, and fastembed's sizes are dynamic, so CoreML takes a fraction of each
graph, in pieces (arctic-embed-m: 401 of 623 nodes in 74 pieces). Measured on 64 real chunks
against the CPU: bge-small and bge-base ran as fast, jina-v2-small 4x slower, arctic-embed-m 8x
slower. At fixed sizes CoreML ran 3-4x faster, but compiled each model for 8 s to 14 minutes
first. The setting stays for the day CoreML handles dynamic sizes well.

Building a session holds the GIL for its whole duration (measured: the loop that owns the request
freezes for it), and CoreML adds a model compilation to that. Two things keep the freeze short:
CoreML keeps its compiled models in `home.MODEL_CACHE`, so only the first build of a model pays
for the compilation, and cross-encoders always build on CPU: they score at most `candidates` texts
per search, which CPU does in milliseconds, while their CoreML build stalled the app for seconds.

Text embedding does not use CoreML either. ONNX Runtime cannot place a whole bge graph on CoreML
(1223 nodes, 785 supported, split into 146 partitions), and running inference across that many
subgraphs allocates per partition: embedding 30 chunks with bge-large peaked at 35.7 GB of RSS
against 2.5 GB on CPU, which the OS answers with SIGKILL. CPU was also faster for both sizes
measured (bge-large 1.7s vs 10.5s; bge-small 0.3s either way), so nothing is traded away. CUDA and
the other accelerators are unaffected - only CoreML places a partial graph like this.
"""

import threading
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from haskie import home
from haskie.catalogue.catalogue import EmbeddingModel
from haskie.indexing import gguf_models, hardware, mlx_models, onnx_rerank
from haskie.indexing.hardware import COREML_TOO_LARGE, Runtime, runtime
from haskie.settings import Accelerator

# a provider entry as ONNX Runtime takes it: a name, or a (name, options) pair
Provider = str | tuple[str, dict[str, str]]
RERANKER_ACCELERATOR: Accelerator = "cpu"
EMBEDDING_COREML = False  # see the module docstring: CoreML costs 14x the memory and 6x the time

# Construction (download, session setup) is serialized, using a model is not, and the lock is
# outside the cache so the second caller of a model being built waits and then gets that one:
# fastembed downloads into one shared cache directory and several worker threads ask for the same
# model at once, so without this each of them starts its own download into the same files.
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


def providers(accelerator: Accelerator = "auto", *, coreml: bool = True) -> list[Provider]:
    import onnxruntime

    names = select_providers(onnxruntime.get_available_providers(), accelerator)
    if not coreml:
        names = drop_coreml(names)
    return with_options(names, str(home.MODEL_CACHE))


def drop_coreml(names: list[str]) -> list[str]:
    """Without CoreML, keeping CPU as the fallback it always is."""
    kept = [name for name in names if name != "CoreMLExecutionProvider"]
    return kept if kept else ["CPUExecutionProvider"]


def provider_name(provider: Provider) -> str:
    return provider if isinstance(provider, str) else provider[0]


def model_providers(name: str, accelerator: Accelerator) -> list[Provider]:
    """The providers ONNX model `name` runs on: `providers`, less CoreML for a model too large
    for it."""
    chosen = providers(accelerator)
    if name in COREML_TOO_LARGE:
        return [one for one in chosen if provider_name(one) != COREML]
    return chosen


@cache
def _build_model(name: str, accelerator: Accelerator):
    """fastembed's ONNX `TextEmbedding`, or an MLX or GGUF embedder shaped like it
    (`mlx_models`, `gguf_models`)."""
    match runtime(name):
        case Runtime.MLX:
            return mlx_models.embedder(name)
        case Runtime.GGUF:
            return gguf_models.GgufEmbedder(name)
    onnx_runtime()
    from fastembed import TextEmbedding
    from fastembed.text.onnx_text_model import OnnxTextModel

    _register_custom()
    description = next(
        found for found in TextEmbedding._list_supported_models() if found.model == name
    )
    # a model over ONNX's 2 GB keeps its weights in an external file: see `_local_copy`
    local = _local_copy(description) if description.additional_files else None
    model = TextEmbedding(
        model_name=name,
        providers=model_providers(name, accelerator),
        **({"specific_model_path": str(local)} if local else {}),
    )
    # every fastembed text embedder is one; the check narrows the type fastembed declares
    if isinstance(model.model, OnnxTextModel) and model.model.tokenizer is not None:
        pad_to_longest(model.model.tokenizer)
    return model


def pad_to_longest(tokenizer: Any) -> None:
    """Pad every batch to its own longest row. fastembed keeps the padding a model's
    `tokenizer.json` ships, and gte-base's pads to a fixed 128 tokens: a longer row then stays
    longer than the rest, and fastembed fails to stack the batch (measured: every batch that mixed
    a chunk over 128 tokens with a shorter one)."""
    padding = tokenizer.padding
    if padding and padding["length"] is not None:
        # `padding` holds `enable_padding`'s own arguments, so all but the length carry over
        tokenizer.enable_padding(**{**padding, "length": None})


# Embedders fastembed does not list, each from an ONNX export of its own, pinned to a revision.
# bge-m3: BAAI's export, whose first output is the token states; CLS pooling then normalizing them
# is bge-m3's dense vector, the one its `sentence_embedding` output also gives.
CUSTOM_EMBEDDERS: dict[str, dict] = {
    "BAAI/bge-m3": {
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "dim": 1024,
        "model_file": "onnx/model.onnx",
        "additional_files": ["onnx/model.onnx_data"],
    },
}
# what fastembed downloads beside a model's own files (`common.model_management`)
_TOKENIZER_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
]


@cache
def _register_custom() -> None:
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=name, providers=providers(accelerator, coreml=EMBEDDING_COREML))


@cache
def _register_custom_rerankers() -> None:
    from fastembed.common.model_description import ModelSource
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    for name in CUSTOM_RERANKERS:
        TextCrossEncoder.add_custom_model(
            model=name, sources=ModelSource(hf=name), model_file="onnx/model.onnx"
        )


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


LAYER_NORM_EPS = 1e-5  # torch's `layer_norm` default, which nomic's recipe runs with


def embed_texts(model: EmbeddingModel, texts: list[str]) -> list[list[float]]:
    """Document-side embeddings (one vector per text), each after the model's document prefix."""
    if not texts:
        return []
    embedder = _model(model.name, model.accelerator)
    vectors = embedder.embed([model.document_prefix + t for t in texts])
    return [_cut(model, v).tolist() for v in vectors]


def embed_query(model: EmbeddingModel, text: str) -> list[float]:
    """Query-side embedding, after the model's query prefix. `query_embed` rather than `embed`:
    a multi-task model (jina-v3) picks its query adapter there."""
    embedder = _model(model.name, model.accelerator)
    return _cut(model, next(iter(embedder.query_embed(model.query_prefix + text)))).tolist()


def _cut(model: EmbeddingModel, vector: np.ndarray) -> np.ndarray:
    """`vector` as the model's profile stores it: whole, or cut to its Matryoshka size and
    normalized again, after a layer norm over the whole vector where the model asks for one.
    The layer norm does not care that fastembed normalized first: it undoes any scale."""
    if model.matryoshka is None:
        return vector
    if model.matryoshka.layer_norm:
        vector = (vector - vector.mean()) / np.sqrt(vector.var() + LAYER_NORM_EPS)
    cut = vector[: model.dims]
    return cut / np.linalg.norm(cut)


def warm(name: str, accelerator: Accelerator) -> None:
    """Load the model now, downloading it on a cold cache, so the first embed does not stall."""
    _model(name, accelerator)


def warm_reranker(name: str, accelerator: Accelerator) -> None:
    _cross_encoder(name, accelerator)


def rerank_scores(
    model_name: str, accelerator: Accelerator, query: str, texts: list[str]
) -> list[float]:
    """Cross-encoder relevance of each text to the query (higher = better; not normalized)."""
    if not texts:
        return []
    return [float(s) for s in _cross_encoder(model_name, accelerator).rerank(query, texts)]
