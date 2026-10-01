"""Which runtime loads a model, the devices it can run on, and the one it runs on here. An
embedder and a reranker of one runtime answer alike."""

import types

import pytest

from haskie.indexing import embed, gguf_models, hardware, mlx_models, onnx_models
from haskie.indexing.hardware import Device, Runtime
from haskie.settings import DEFAULT_RERANKER, Accelerator

EVERYWHERE = (Device.CPU, Device.APPLE_SILICON, Device.GPU)
ONNX_EMBEDDER = "ibm-granite/granite-embedding-97m-multilingual-r2"
MLX_EMBEDDER, GGUF_EMBEDDER = "intfloat/e5-base-v2:mlx", "ChristianAzinn/e5-base-v2-gguf"
# The catalogue holds no MLX reranker and no model CoreML runs today, but the runtime and the rule
# stay: these stand in for one of each
MLX_RERANKER, ON_COREML = "test/tiny-reranker-mlx", "test/coreml"


@pytest.fixture(autouse=True)
def stand_ins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(mlx_models.RERANKERS, MLX_RERANKER, "0")
    monkeypatch.setattr(hardware, "COREML_RUNS", frozenset({ON_COREML}))


@pytest.mark.parametrize(
    ("name", "model", "runtime", "expected"),
    [
        (
            "an ONNX embedder runs anywhere: Apple Silicon by WebGPU",
            ONNX_EMBEDDER,
            Runtime.ONNX,
            EVERYWHERE,
        ),
        ("an ONNX reranker the same", DEFAULT_RERANKER, Runtime.ONNX, EVERYWHERE),
        (
            "an MLX embedder runs on Apple Silicon only",
            MLX_EMBEDDER,
            Runtime.MLX,
            (Device.APPLE_SILICON,),
        ),
        ("an MLX reranker the same", MLX_RERANKER, Runtime.MLX, (Device.APPLE_SILICON,)),
        ("a GGUF embedder the same", GGUF_EMBEDDER, Runtime.GGUF, (Device.APPLE_SILICON,)),
    ],
)
def test_where_a_model_runs(
    name: str, model: str, runtime: Runtime, expected: tuple[Device, ...]
) -> None:
    assert (hardware.runtime(model), hardware.devices(model)) == (runtime, expected), name


CPU, CUDA, COREML = "CPUExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider"
WEBGPU = onnx_models.WEBGPU


@pytest.mark.parametrize(
    ("name", "model", "accelerator", "installed", "providers", "expected"),
    [
        ("ONNX on auto without CUDA: the CPU", ONNX_EMBEDDER, "auto", False, [CPU], Device.CPU),
        ("ONNX on auto with CUDA: the GPU", ONNX_EMBEDDER, "auto", False, [CUDA, CPU], Device.GPU),
        ("ONNX on cpu: the CPU, CUDA or not", ONNX_EMBEDDER, "cpu", False, [CUDA, CPU], Device.CPU),
        (
            "ONNX on auto with WebGPU: Apple Silicon",
            ONNX_EMBEDDER,
            "auto",
            False,
            [COREML, WEBGPU, CPU],
            Device.APPLE_SILICON,
        ),
        (
            "ONNX on coreml: Apple Silicon",
            ON_COREML,
            "coreml",
            False,
            [COREML, CPU],
            Device.APPLE_SILICON,
        ),
        (
            "not measured on CoreML: the CPU",
            ONNX_EMBEDDER,
            "coreml",
            False,
            [COREML, CPU],
            Device.CPU,
        ),
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
    monkeypatch.setattr(embed, "nvidia_loads", lambda _provider: True)  # CUDA runs here

    assert hardware.device(model, accelerator) == expected, name
