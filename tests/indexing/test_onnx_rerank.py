"""An ONNX encoder finished by its own head: the head's weight files read, and its four steps.

The encoder itself is not run here (270 MB); the head is, on weights small enough to follow by hand.
"""

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from haskie.indexing.onnx_rerank import HeadedCrossEncoder, _tensors


def _safetensors(path: Path, tensors: dict[str, np.ndarray], dtype: str = "F32") -> Path:
    """A file the way safetensors writes it: header length, JSON header, then the raw data."""
    header: dict = {"__metadata__": {"format": "pt"}}
    body = b""
    for key, value in tensors.items():
        data = value.astype(np.float32).tobytes()
        header[key] = {
            "dtype": dtype,
            "shape": list(value.shape),
            "data_offsets": [len(body), len(body) + len(data)],
        }
        body += data
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)
    return path


def test_the_head_weights_are_read_as_saved(tmp_path: Path) -> None:
    weight = np.arange(6, dtype=np.float32).reshape(2, 3)
    bias = np.array([0.5], dtype=np.float32)

    found = _tensors(
        _safetensors(tmp_path / "head.safetensors", {"linear.weight": weight, "linear.bias": bias})
    )

    assert set(found) == {"linear.weight", "linear.bias"}, "the metadata is not a tensor"
    assert np.array_equal(found["linear.weight"], weight) and found["linear.weight"].shape == (2, 3)
    assert np.array_equal(found["linear.bias"], bias)


def test_a_head_saved_in_another_precision_is_refused(tmp_path: Path) -> None:
    path = _safetensors(
        tmp_path / "head.safetensors", {"linear.weight": np.ones((1, 1))}, dtype="F16"
    )

    with pytest.raises(ValueError, match="only F32 is read"):
        _tensors(path)


def test_the_head_is_dense_gelu_layernorm_dense() -> None:
    """Two features, the first dense an identity: GELU and LayerNorm are the only transforms, so
    the score can be worked out by hand."""
    head = HeadedCrossEncoder.__new__(HeadedCrossEncoder)  # no download, no session
    head._dense = np.eye(2, dtype=np.float32)
    head._norm = np.ones(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
    head._score = np.array([[1.0, -1.0]], dtype=np.float32), np.array([0.25], dtype=np.float32)

    (score,) = head._head(np.array([[1.0, -1.0]], dtype=np.float32))

    gelu = [0.5 * x * (1 + math.erf(x / math.sqrt(2))) for x in (1.0, -1.0)]
    mean = sum(gelu) / 2
    std = math.sqrt(sum((g - mean) ** 2 for g in gelu) / 2 + 1e-5)
    normed = [(g - mean) / std for g in gelu]
    assert score == pytest.approx(normed[0] - normed[1] + 0.25, rel=1e-5)
