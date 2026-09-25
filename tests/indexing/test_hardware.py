"""Which runtime loads each kind of model, and which devices run it."""

from collections.abc import Callable

import pytest

from haskie.indexing import hardware
from haskie.indexing.hardware import Device, Runtime


@pytest.mark.parametrize(
    ("name", "devices", "model", "runtime", "expected"),
    [
        (
            "an ONNX embedder runs anywhere",
            hardware.embedder_devices,
            "BAAI/bge-small-en-v1.5",
            Runtime.ONNX,
            (Device.CPU, Device.APPLE_SILICON, Device.GPU),
        ),
        (
            "an ONNX embedder too large for CoreML skips Apple Silicon",
            hardware.embedder_devices,
            "jinaai/jina-embeddings-v3",
            Runtime.ONNX,
            (Device.CPU, Device.GPU),
        ),
        (
            "an MLX embedder runs on Apple Silicon only",
            hardware.embedder_devices,
            "mlx-community/nomicai-modernbert-embed-base-bf16",
            Runtime.MLX,
            (Device.APPLE_SILICON,),
        ),
        (
            "an ONNX reranker runs on the CPU only",
            hardware.reranker_devices,
            "Xenova/ms-marco-MiniLM-L-6-v2",
            Runtime.ONNX,
            (Device.CPU,),
        ),
        (
            "an MLX reranker runs on Apple Silicon only",
            hardware.reranker_devices,
            "jinaai/jina-reranker-v3-mlx",
            Runtime.MLX,
            (Device.APPLE_SILICON,),
        ),
    ],
)
def test_where_a_model_runs(
    name: str,
    devices: Callable[[str], tuple[Device, ...]],
    model: str,
    runtime: Runtime,
    expected: tuple[Device, ...],
) -> None:
    assert (hardware.runtime(model), devices(model)) == (runtime, expected)
