"""Which runtime loads a model and which hardware it can run on. The loaders in `embed` act on
these facts, and the catalogue reports them, so the two cannot disagree. Embedders and rerankers
follow the same facts: a runtime runs both kinds on the same hardware."""

from enum import StrEnum

from haskie.document import ocr
from haskie.indexing import gguf_models, mlx_models, onnx_models
from haskie.settings import Accelerator


class Runtime(StrEnum):
    ONNX = "onnx"  # ONNX Runtime (`onnx_models`)
    MLX = "mlx"  # Apple's MLX: the `mlx` extra, on Apple Silicon only
    GGUF = "gguf"  # llama.cpp on Metal: the `gguf` extra, on Apple Silicon only
    # OCR (`document.ocr`): its own ONNX Runtime, on its default provider, the CPU; no choice
    PDF_INSPECTOR = "pdf_inspector"


class Device(StrEnum):
    CPU = "cpu"
    APPLE_SILICON = "apple_silicon"  # MLX or llama.cpp on the GPU, or ONNX through WebGPU or CoreML
    GPU = "gpu"  # ONNX through CUDA, TensorRT or ROCm: Linux, on an NVIDIA GPU


# The ONNX models CoreML was measured to run; any other runs on the CPU when the settings ask for
# CoreML, which they need not: `auto` reaches the Apple GPU through WebGPU (`embed`). None of the
# catalogue's does today (M4 Pro, onnxruntime's CoreML provider): every
# ModernBERT and Qwen3 model fails its first batch ("Unable to compute the prediction using a
# neural network model"), and e5-base-v2 got the process killed. Before, a model over CoreML's
# 2 GB protobuf failed to build (jina-embeddings-v3), while bge-m3 and e5-large passed.
COREML_RUNS: frozenset[str] = frozenset()


def runtime(name: str) -> Runtime:
    if name == ocr.MODEL:
        return Runtime.PDF_INSPECTOR
    if mlx_models.pinned(name):
        return Runtime.MLX
    return Runtime.GGUF if gguf_models.pin(name) else Runtime.ONNX


def device(name: str, accelerator: Accelerator) -> Device | None:
    """The device model `name` runs on here under `accelerator`, or None where it cannot run: its
    runtime is not installed, or the setting asks for the CPU, which MLX lacks and where
    llama.cpp runs slower than ONNX (see `gguf_models`). The loaders refuse a model this answers
    None for, the options list only models it answers a device for, and the status reports it."""
    match runtime(name):
        case Runtime.PDF_INSPECTOR:
            return Device.CPU
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
            if provider in (embed.COREML, onnx_models.WEBGPU):
                return Device.APPLE_SILICON
            return Device.GPU
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
    match runtime(name):
        case Runtime.PDF_INSPECTOR:
            return (Device.CPU,)
        case Runtime.ONNX:
            # Apple Silicon through WebGPU, whether or not CoreML runs the model
            return (Device.CPU, Device.APPLE_SILICON, Device.GPU)
    return (Device.APPLE_SILICON,)
