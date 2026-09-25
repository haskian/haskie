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
from pathlib import Path
from typing import Any

import numpy as np

from haskie import home
from haskie.catalogue.catalogue import EmbeddingModel
from haskie.indexing import mlx_models, onnx_rerank
from haskie.indexing.hardware import COREML_TOO_LARGE, RERANKER_ACCELERATOR
from haskie.settings import Accelerator

# a provider entry as ONNX Runtime takes it: a name, or a (name, options) pair
Provider = str | tuple[str, dict[str, str]]

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
    """fastembed's ONNX `TextEmbedding`, or an MLX embedder shaped like it (`mlx_models`)."""
    if name in mlx_models.EMBEDDERS:
        return mlx_models.embedder(name)
    from fastembed import TextEmbedding

    _register_custom()
    description = next(
        found for found in TextEmbedding._list_supported_models() if found.model == name
    )
    # a model over ONNX's 2 GB keeps its weights in an external file: see `_local_copy`
    local = _local_copy(description) if description.additional_files else None
    chosen = providers(accelerator)
    if name in COREML_TOO_LARGE:
        chosen = [one for one in chosen if provider_name(one) != "CoreMLExecutionProvider"]
    return TextEmbedding(
        model_name=name,
        providers=chosen,
        **({"specific_model_path": str(local)} if local else {}),
    )


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
    from fastembed.common.model_description import ModelSource, PoolingType

    for name, spec in CUSTOM_EMBEDDERS.items():
        TextEmbedding.add_custom_model(
            model=name,
            pooling=PoolingType.CLS,
            normalization=True,
            sources=ModelSource(hf=name),
            dim=spec["dim"],
            model_file=spec["model_file"],
            additional_files=spec["additional_files"],
        )


def _local_copy(description: Any) -> Path:
    """A model's files as real files in one directory.

    ONNX Runtime refuses an external-data file (`model.onnx_data`) that resolves outside the
    model's own directory, and the Hugging Face cache fastembed downloads into keeps every file as
    a link into a blob store elsewhere. So a model with external data is downloaded into a plain
    directory beside fastembed's cache instead, at its pinned revision where it has one.
    """
    from fastembed.common.utils import define_cache_dir
    from huggingface_hub import snapshot_download

    repo = description.sources.hf
    revision = CUSTOM_EMBEDDERS.get(description.model, {}).get("revision")
    target = define_cache_dir() / "local" / repo.replace("/", "--") / (revision or "main")
    snapshot_download(
        repo,
        revision=revision,
        local_dir=target,
        allow_patterns=[description.model_file, *description.additional_files, *_TOKENIZER_FILES],
    )
    return target


@cache
def _build_cross_encoder(name: str):
    """fastembed's ONNX cross-encoder, an MLX reranker for the models fastembed cannot run
    (`mlx_models`), or an ONNX encoder finished by its own head (`onnx_rerank`). All answer
    `rerank(query, texts)` with one score per text, in order."""
    if name in mlx_models.REVISIONS:
        return mlx_models.reranker(name)
    if name in onnx_rerank.REVISIONS:
        return onnx_rerank.HeadedCrossEncoder(name, providers(RERANKER_ACCELERATOR))
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    _register_custom_rerankers()
    return TextCrossEncoder(model_name=name, providers=providers(RERANKER_ACCELERATOR))


# Cross-encoders fastembed does not list whose ONNX export is the whole classifier, head and all.
CUSTOM_RERANKERS = ["mixedbread-ai/mxbai-rerank-xsmall-v1"]


@cache
def _register_custom_rerankers() -> None:
    from fastembed.common.model_description import ModelSource
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    for name in CUSTOM_RERANKERS:
        TextCrossEncoder.add_custom_model(
            model=name, sources=ModelSource(hf=name), model_file="onnx/model.onnx"
        )


def _model(name: str, accelerator: Accelerator):
    with _MODEL_LOCK:
        return _build_model(name, accelerator)


def _cross_encoder(name: str):
    with _MODEL_LOCK:
        return _build_cross_encoder(name)


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


def warm_reranker(name: str) -> None:
    _cross_encoder(name)


def rerank_scores(model_name: str, query: str, texts: list[str]) -> list[float]:
    """Cross-encoder relevance of each text to the query (higher = better; not normalized)."""
    if not texts:
        return []
    return [float(s) for s in _cross_encoder(model_name).rerank(query, texts)]
