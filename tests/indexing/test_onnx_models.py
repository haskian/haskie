"""The ONNX models haskie runs itself: what a session is fed, how token states pool, how a pair is
cut, and the Ettin head. No export runs here (up to 640 MB): a session stands in that answers each
token's id as its state, so a vector says which tokens were read. The tokenizer is a real
`tokenizers.Tokenizer`, saved as a model ships it."""

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from haskie.indexing import onnx_models
from haskie.indexing.onnx_models import Pooling

WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "retry", "the", "call", "idempotent", "jitter"]
ID = {word: n for n, word in enumerate(WORDS)}


def _tokenizer_file(path: Path, fixed_length: int | None = None) -> Path:
    """A BERT-like `tokenizer.json`: [CLS] text [SEP], and [CLS] a [SEP] b [SEP] for a pair,
    padded to `fixed_length` the way gte's ships padded to 8,000."""
    tokenizer = Tokenizer(WordLevel(ID, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", ID["[CLS]"]), ("[SEP]", ID["[SEP]"])],
    )
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=fixed_length)
    tokenizer.save(str(path / "tokenizer.json"))
    return path


class Session:
    """`onnxruntime.InferenceSession` as `_Session` uses it: the inputs and outputs it declares,
    what it was fed, and a state per token of (its id, 1); `logits` answers one per row."""

    def __init__(
        self, inputs: list[str], outputs: list[str], providers: list | None = None
    ) -> None:
        self.inputs, self.outputs = inputs, outputs
        self.providers = providers or ["CPUExecutionProvider"]
        self.feeds: list[dict[str, np.ndarray]] = []

    def get_providers(self) -> list[str]:
        return self.providers

    def get_inputs(self) -> list[Any]:
        return [type("Input", (), {"name": name}) for name in self.inputs]

    def get_outputs(self) -> list[Any]:
        return [type("Output", (), {"name": name}) for name in self.outputs]

    def run(self, outputs: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append(feeds)
        ids = feeds["input_ids"].astype(np.float32)
        if outputs == ["logits"]:
            return [ids.sum(axis=1, keepdims=True)]
        return [np.stack([ids, np.ones_like(ids)], axis=-1)]


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> list[Session]:
    """Every session built, in order, each declaring BERT's three inputs and token states."""
    built: list[Session] = []

    def build(path: str, providers: list) -> Session:
        inputs = ["input_ids", "attention_mask", "token_type_ids"]
        built.append(Session(inputs, ["last_hidden_state"], providers))
        return built[-1]

    monkeypatch.setattr(onnxruntime, "InferenceSession", build)
    return built


def _unit(*ids: float) -> list[float]:
    vector = np.array(ids, dtype=np.float64)
    return (vector / np.linalg.norm(vector)).tolist()


# --- pooling ------------------------------------------------------------------------------

STATES = np.array(
    [[[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]], [[2.0, 1.0], [4.0, 3.0], [9.0, 9.0]]], dtype=np.float32
)
MASK = np.array([[1, 1, 1], [1, 1, 0]])  # the second row is padded once, on the right


@pytest.mark.parametrize(
    ("name", "pooling", "expected"),
    [
        ("cls: the first token's, padding or not", Pooling.CLS, [_unit(1, 0), _unit(2, 1)]),
        ("mean: of the real tokens only", Pooling.MEAN, [_unit(3, 2), _unit(3, 2)]),
        ("last: the last real token's, not the pad", Pooling.LAST, [_unit(5, 4), _unit(4, 3)]),
    ],
)
def test_token_states_pool_to_unit_vectors_as_the_model_asks(
    name: str, pooling: Pooling, expected: list[list[float]]
) -> None:
    pooled = onnx_models.pool(STATES, MASK, pooling)

    assert pooled.tolist() == [pytest.approx(one) for one in expected], name
    assert pooled.dtype == np.float32, "the states' own type, not a float64 copy"


# --- the session --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "inputs", "outputs", "fed", "read"),
    [
        (
            "BERT's three inputs, its token states by name",
            ["input_ids", "attention_mask", "token_type_ids"],
            ["pooler_output", "last_hidden_state"],
            {"input_ids", "attention_mask", "token_type_ids"},
            "last_hidden_state",
        ),
        (
            "a decoder's positions, and an output of another name read first",
            ["input_ids", "attention_mask", "position_ids"],
            ["token_embeddings", "present.0.key"],
            {"input_ids", "attention_mask", "position_ids"},
            "token_embeddings",
        ),
    ],
)
def test_a_session_is_fed_what_its_export_asks_for(
    name: str,
    inputs: list[str],
    outputs: list[str],
    fed: set[str],
    read: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stand_in = Session(inputs, outputs)
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda path, providers: stand_in)
    model = onnx_models._Session(_tokenizer_file(tmp_path), "m.onnx", [], 16)
    asked: list[list[str]] = []
    run = stand_in.run
    monkeypatch.setattr(stand_in, "run", lambda out, feeds: asked.append(out) or run([], feeds))

    model.run(["retry the call", "jitter"])

    (feeds,) = stand_in.feeds
    assert set(feeds) == fed and asked == [[read]], name
    if "position_ids" in fed:
        assert feeds["position_ids"].tolist() == [[0, 1, 2, 3, 4]] * 2, "from 0, every row"


def test_a_batch_pads_to_its_own_longest_row_though_the_tokenizer_ships_a_fixed_length(
    tmp_path: Path, session: list[Session]
) -> None:
    """gte's `tokenizer.json` pads every pair to 8,000 tokens (538 s and 33 GB for 5 pairs)."""
    model = onnx_models._Session(_tokenizer_file(tmp_path, 8000), "m.onnx", [], 16)

    _, mask = model.run(["retry the call", "jitter"])

    assert mask.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], "five wide, padded on the right"


@pytest.mark.parametrize(
    ("name", "query", "text", "expected"),
    [
        (
            "a short query keeps whole: the text is cut",
            "retry the call",
            "idempotent jitter idempotent jitter",
            ["retry", "the", "call", "[SEP]", "idempotent", "jitter", "idempotent"],
        ),
        (
            "a query longer than the cut is cut too, rather than failing the search",
            "retry the call retry the call retry the call",
            "idempotent jitter",
            ["retry", "the", "call", "retry", "[SEP]", "idempotent", "jitter"],
        ),
    ],
)
def test_a_pair_cuts_its_longer_side_first(
    name: str, query: str, text: str, expected: list[str], tmp_path: Path, session: list[Session]
) -> None:
    model = onnx_models._Session(_tokenizer_file(tmp_path), "m.onnx", [], 9)

    model.run([(query, text)])

    (feeds,) = session[0].feeds
    words = [WORDS[n] for n in feeds["input_ids"][0]]
    assert words[1:-1] == expected, name
    assert len(words) == 9 and words[0] == "[CLS]" and words[-1] == "[SEP]", name


def test_every_webgpu_session_shares_one_lock_and_a_cpu_session_has_its_own(
    tmp_path: Path, session: list[Session]
) -> None:
    """Two runs at once on two WebGPU sessions abort the process (one Metal device)."""
    path = _tokenizer_file(tmp_path)
    on_webgpu = [onnx_models.WEBGPU, "CPUExecutionProvider"]
    first, second = (onnx_models._Session(path, "m.onnx", on_webgpu, 16) for _ in range(2))
    cpu_a, cpu_b = (
        onnx_models._Session(path, "m.onnx", ["CPUExecutionProvider"], 16) for _ in range(2)
    )

    assert first._lock is second._lock is onnx_models._WEBGPU_LOCK
    assert cpu_a._lock is not cpu_b._lock and cpu_a._lock is not first._lock


# --- embedders and cross-encoders ---------------------------------------------------------


@pytest.fixture
def pinned(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """`test/embedder` and two `test/reranker`s pinned, their files in `tmp_path`."""
    monkeypatch.setitem(
        onnx_models.EMBEDDERS, "test/embedder", onnx_models.EmbedderPin("r", Pooling.LAST, 8192)
    )
    monkeypatch.setitem(onnx_models.RERANKERS, "test/reranker", onnx_models.RerankerPin("r"))
    monkeypatch.setitem(
        onnx_models.RERANKERS, "test/headed", onnx_models.RerankerPin("r", headed=True)
    )
    monkeypatch.setattr(onnx_models, "_download", lambda repo, revision, files: tmp_path)
    return _tokenizer_file(tmp_path)


@pytest.mark.parametrize(
    ("name", "texts", "expected", "passes"),
    [
        ("no texts, no vectors", [], [], 0),
        ("one text: its last real token, [SEP]", ["retry the call"], [_unit(3, 1)], 1),
        (
            "past one batch: one vector per text, in order",
            ["jitter"] * onnx_models.BATCH + ["retry"],
            [_unit(3, 1)] * (onnx_models.BATCH + 1),
            2,
        ),
    ],
)
def test_an_embedder_answers_one_normalized_vector_per_text(
    name: str,
    texts: list[str],
    expected: list[list[float]],
    passes: int,
    pinned: Path,
    session: list[Session],
) -> None:
    embedder = onnx_models.Embedder("test/embedder", [])

    vectors = [vector.tolist() for vector in embedder.embed(texts)]

    assert vectors == [pytest.approx(one) for one in expected], name
    assert len(session[0].feeds) == passes, name


def test_an_embedder_reads_at_most_its_cut(pinned: Path, session: list[Session]) -> None:
    embedder = onnx_models.Embedder("test/embedder", [])

    (query,) = embedder.query_embed("retry " * (onnx_models.MAX_EMBED_TOKENS + 10))

    assert session[0].feeds[0]["input_ids"].shape == (1, onnx_models.MAX_EMBED_TOKENS)
    assert query.tolist() == pytest.approx(_unit(3, 1)), "the cut ends with [SEP]"


def test_a_cross_encoder_answers_the_exports_logit_per_pair(
    pinned: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in = Session(["input_ids", "attention_mask"], ["logits"])
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda path, providers: stand_in)
    reranker = onnx_models.CrossEncoder("test/reranker", [])

    scores = reranker.rerank("retry", ["the call", "jitter"])

    # the sum of each pair's ids, pads included: [CLS] retry [SEP] the call [SEP] / ... jitter [SEP]
    assert scores == [2 + 4 + 3 + 5 + 6 + 3, 2 + 4 + 3 + 8 + 3 + 0]
    assert reranker.rerank("retry", []) == []


def test_a_headed_cross_encoder_scores_the_first_tokens_state_with_its_head(
    pinned: Path, session: list[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    read: list[list[float]] = []

    class Head:
        def __init__(self, path: Path) -> None:
            pass

        def __call__(self, cls: np.ndarray) -> np.ndarray:
            read.extend(cls.tolist())
            return cls[:, 0] * 10

    monkeypatch.setattr(onnx_models, "Head", Head)
    reranker = onnx_models.CrossEncoder("test/headed", [])

    assert reranker.rerank("retry", ["the call"]) == [20.0], "[CLS]'s id, 2, through the head"
    assert read == [[2.0, 1.0]]


def test_files_download_at_their_revision_into_a_directory_of_their_own(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Real files in one directory: ONNX Runtime refuses external data that resolves elsewhere."""
    import huggingface_hub
    from huggingface_hub import constants

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(constants, "HF_HOME", str(tmp_path))
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo, **kw: calls.append((repo, kw))
    )

    path = onnx_models._download("org/model", "abc", ["onnx/model.onnx", "onnx/model.onnx_data"])

    assert path == tmp_path / "haskie-onnx" / "org--model" / "abc"
    assert calls == [
        (
            "org/model",
            {
                "revision": "abc",
                "local_dir": path,
                "allow_patterns": ["onnx/model.onnx", "onnx/model.onnx_data", "tokenizer.json"],
            },
        )
    ]


# --- the Ettin head -----------------------------------------------------------------------


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)
    return path


def test_the_head_weights_are_read_as_saved(tmp_path: Path) -> None:
    weight = np.arange(6, dtype=np.float32).reshape(2, 3)
    bias = np.array([0.5], dtype=np.float32)

    found = onnx_models._tensors(
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
        onnx_models._tensors(path)


def test_the_head_is_dense_gelu_layernorm_dense(tmp_path: Path) -> None:
    """Two features, the first dense an identity: GELU and LayerNorm are the only transforms, so
    the score can be worked out by hand. Read from the three files as Ettin ships them."""
    _safetensors(tmp_path / "2_Dense/model.safetensors", {"linear.weight": np.eye(2)})
    _safetensors(
        tmp_path / "3_LayerNorm/model.safetensors",
        {"norm.weight": np.ones(2), "norm.bias": np.zeros(2)},
    )
    _safetensors(
        tmp_path / "4_Dense/model.safetensors",
        {"linear.weight": np.array([[1.0, -1.0]]), "linear.bias": np.array([0.25])},
    )

    (score,) = onnx_models.Head(tmp_path)(np.array([[1.0, -1.0]], dtype=np.float32))

    gelu = [0.5 * x * (1 + math.erf(x / math.sqrt(2))) for x in (1.0, -1.0)]
    mean = sum(gelu) / 2
    std = math.sqrt(sum((g - mean) ** 2 for g in gelu) / 2 + 1e-5)
    normed = [(g - mean) / std for g in gelu]
    assert score == pytest.approx(normed[0] - normed[1] + 0.25, rel=1e-5)
