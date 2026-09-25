"""Which runtime loads a model and which hardware it can run on. The loaders in `embed` and
`mlx_models` act on these facts, and the catalogue reports them, so the two cannot disagree."""

from enum import StrEnum

from haskie.indexing import mlx_models
from haskie.settings import Accelerator


class Runtime(StrEnum):
    ONNX = "onnx"  # ONNX Runtime, through fastembed or `onnx_rerank`
    MLX = "mlx"  # Apple's MLX: the `mlx` extra, on Apple Silicon only


class Device(StrEnum):
    CPU = "cpu"
    APPLE_SILICON = "apple_silicon"  # ONNX through CoreML, or MLX on the GPU
    GPU = "gpu"  # ONNX through CUDA, TensorRT or ROCm: the `gpu` extra


# Cross-encoders always run on the CPU: see `embed`.
RERANKER_ACCELERATOR = Accelerator.CPU

# CoreML compiles a whole model into one protobuf, which caps at 2 GB: a model past that fails to
# build on Apple Silicon ("CoreML.Specification.Model exceeded maximum protobuf size of 2GB") and
# runs on the CPU instead. bge-m3 and e5-large pass, because CoreML takes only part of their graph.
COREML_TOO_LARGE = frozenset({"jinaai/jina-embeddings-v3"})


def runtime(name: str) -> Runtime:
    return Runtime.MLX if name in mlx_models.REVISIONS else Runtime.ONNX


def embedder_devices(name: str) -> tuple[Device, ...]:
    """Every device the embedder `name` can run on, when the settings let it choose."""
    if runtime(name) == Runtime.MLX:
        return (Device.APPLE_SILICON,)
    if name in COREML_TOO_LARGE:
        return (Device.CPU, Device.GPU)
    return (Device.CPU, Device.APPLE_SILICON, Device.GPU)


def reranker_devices(name: str) -> tuple[Device, ...]:
    # an ONNX cross-encoder runs on `RERANKER_ACCELERATOR` only
    return (Device.APPLE_SILICON,) if runtime(name) == Runtime.MLX else (Device.CPU,)
