"""Which runtime loads a model and which hardware it can run on. The loaders in `embed` act on
these facts, and the catalogue reports them, so the two cannot disagree. Embedders and rerankers
follow the same facts: a runtime runs both kinds on the same hardware."""

from enum import StrEnum

from haskie.indexing import gguf_models, mlx_models
from haskie.settings import Accelerator


class Runtime(StrEnum):
    ONNX = "onnx"  # ONNX Runtime, through fastembed or `onnx_rerank`
    MLX = "mlx"  # Apple's MLX: the `mlx` extra, on Apple Silicon only
    GGUF = "gguf"  # llama.cpp on Metal: the `gguf` extra, on Apple Silicon only


class Device(StrEnum):
    CPU = "cpu"
    APPLE_SILICON = "apple_silicon"  # MLX or llama.cpp on the GPU, or ONNX through CoreML
    GPU = "gpu"  # ONNX through CUDA, TensorRT or ROCm: Linux, on an NVIDIA GPU


# CoreML compiles a whole model into one protobuf, which caps at 2 GB: a model past that fails to
# build on Apple Silicon ("CoreML.Specification.Model exceeded maximum protobuf size of 2GB") and
# runs on the CPU instead. bge-m3 and e5-large pass, because CoreML takes only part of their graph.
COREML_TOO_LARGE = frozenset({"jinaai/jina-embeddings-v3"})


def runtime(name: str) -> Runtime:
    if name in mlx_models.REVISIONS:
        return Runtime.MLX
    return Runtime.GGUF if name in gguf_models.PINS else Runtime.ONNX


def device(name: str, accelerator: Accelerator) -> Device | None:
    """The device model `name` runs on here under `accelerator`, or None where it cannot run: its
    runtime is not installed, or the setting asks for the CPU, which MLX lacks and where
    llama.cpp runs slower than ONNX (see `gguf_models`). The loaders refuse a model this answers
    None for, the options list only models it answers a device for, and the status reports it."""
    match runtime(name):
        case Runtime.MLX:
            installed = mlx_models.available()
        case Runtime.GGUF:
            installed = gguf_models.available()
        case Runtime.ONNX:
            # imported here: `embed` builds its models on the facts this module holds
            from haskie.indexing import embed

            provider = embed.provider_name(embed.model_providers(name, accelerator)[0])
            if provider == "CPUExecutionProvider":
                return Device.CPU
            return Device.APPLE_SILICON if provider == embed.COREML else Device.GPU
    return Device.APPLE_SILICON if installed and accelerator != Accelerator.CPU else None


def nowhere(name: str) -> str:
    """Why `device` finds no device for model `name`, which only an MLX or GGUF model lacks."""
    extra = runtime(name).value
    return (
        f"{name} runs on {extra} on the Apple GPU: it needs Apple Silicon, where haskie installs "
        f"{extra}, and a hardware setting other than cpu"
    )


def devices(name: str) -> tuple[Device, ...]:
    """Every device model `name` can run on, when the settings let it choose."""
    if runtime(name) != Runtime.ONNX:
        return (Device.APPLE_SILICON,)
    if name in COREML_TOO_LARGE:
        return (Device.CPU, Device.GPU)
    return (Device.CPU, Device.APPLE_SILICON, Device.GPU)
