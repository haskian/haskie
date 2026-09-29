"""Which runtime loads a model, the devices it can run on, and the one it runs on here. An
embedder and a reranker of one runtime answer alike."""

import types

import pytest

from haskie.indexing import embed, gguf_models, hardware, mlx_models
from haskie.indexing.hardware import Device, Runtime
from haskie.settings import Accelerator

EVERYWHERE = (Device.CPU, Device.APPLE_SILICON, Device.GPU)


@pytest.mark.parametrize(
    ("name", "model", "runtime", "expected"),
    [
        ("an ONNX embedder runs anywhere", "BAAI/bge-small-en-v1.5", Runtime.ONNX, EVERYWHERE),
        (
            "an ONNX reranker the same",
            "Xenova/ms-marco-MiniLM-L-6-v2",
            Runtime.ONNX,
            EVERYWHERE,
        ),
        (
            "an ONNX embedder too large for CoreML skips Apple Silicon",
            "jinaai/jina-embeddings-v3",
            Runtime.ONNX,
            (Device.CPU, Device.GPU),
        ),
        (
            "an MLX embedder runs on Apple Silicon only",
            "mlx-community/nomicai-modernbert-embed-base-bf16",
            Runtime.MLX,
            (Device.APPLE_SILICON,),
        ),
        (
            "an MLX reranker the same",
            "soichisumi/bge-reranker-v2-m3-mlx-affine8",
            Runtime.MLX,
            (Device.APPLE_SILICON,),
        ),
        (
            "a GGUF embedder the same",
            "ggml-org/bge-small-en-v1.5-Q8_0-GGUF",
            Runtime.GGUF,
            (Device.APPLE_SILICON,),
        ),
    ],
)
def test_where_a_model_runs(
    name: str, model: str, runtime: Runtime, expected: tuple[Device, ...]
) -> None:
    assert (hardware.runtime(model), hardware.devices(model)) == (runtime, expected), name


MLX_RERANKER = "soichisumi/bge-reranker-v2-m3-mlx-affine8"
GGUF_EMBEDDER = "ggml-org/bge-m3-Q8_0-GGUF"
ONNX_EMBEDDER, TOO_LARGE = "BAAI/bge-small-en-v1.5", "jinaai/jina-embeddings-v3"
CPU, CUDA, COREML = "CPUExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider"


@pytest.mark.parametrize(
    ("name", "model", "accelerator", "installed", "providers", "expected"),
    [
        ("ONNX on auto without CUDA: the CPU", ONNX_EMBEDDER, "auto", False, [CPU], Device.CPU),
        ("ONNX on auto with CUDA: the GPU", ONNX_EMBEDDER, "auto", False, [CUDA, CPU], Device.GPU),
        ("ONNX on cpu: the CPU, CUDA or not", ONNX_EMBEDDER, "cpu", False, [CUDA, CPU], Device.CPU),
        (
            "ONNX on coreml: Apple Silicon",
            ONNX_EMBEDDER,
            "coreml",
            False,
            [COREML, CPU],
            Device.APPLE_SILICON,
        ),
        ("too large for CoreML: the CPU", TOO_LARGE, "coreml", False, [COREML, CPU], Device.CPU),
        ("MLX installed: Apple Silicon", MLX_RERANKER, "auto", True, [CPU], Device.APPLE_SILICON),
        ("MLX not installed: nowhere", MLX_RERANKER, "auto", False, [CPU], None),
        ("MLX on cpu: nowhere, MLX has no CPU", MLX_RERANKER, "cpu", True, [CPU], None),
        ("GGUF installed: Apple Silicon", GGUF_EMBEDDER, "auto", True, [CPU], Device.APPLE_SILICON),
        ("GGUF not installed: nowhere", GGUF_EMBEDDER, "auto", False, [CPU], None),
        ("GGUF on cpu: nowhere, slower than ONNX there", GGUF_EMBEDDER, "cpu", True, [CPU], None),
    ],
)
def test_the_device_a_model_runs_on_here(
    name: str,
    model: str,
    accelerator: Accelerator,
    installed: bool,
    providers: list[str],
    expected: Device | None,
    monkeypatch,
) -> None:
    monkeypatch.setattr(mlx_models, "available", lambda: installed)
    monkeypatch.setattr(gguf_models, "available", lambda: installed)
    stand_in = types.SimpleNamespace(get_available_providers=lambda: providers)
    monkeypatch.setattr(embed, "onnx_runtime", lambda: stand_in)
    monkeypatch.setattr(embed, "cuda_loads", lambda: True)  # a machine where CUDA runs

    assert hardware.device(model, accelerator) == expected, name
