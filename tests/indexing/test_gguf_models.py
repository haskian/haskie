"""The GGUF embedders, behind fastembed's shapes: where they load, how llama.cpp is set up for
each, and one vector per text. The unit cases stand in for llama.cpp; the `network` case runs the
real bge-small file on Metal against the same model in ONNX."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from haskie.indexing import embed, gguf_models
from haskie.settings import Accelerator

BGE_SMALL = "ggml-org/bge-small-en-v1.5-Q8_0-GGUF"
JINA_BASE = "ggml-org/jina-embeddings-v2-base-en-Q8_0-GGUF"


@pytest.mark.parametrize(
    ("name", "spec", "expected"),
    [("installed", object(), True), ("not installed", None, False)],
)
def test_llama_cpp_is_available_where_its_module_is_found(
    name: str, spec: object, expected: bool, monkeypatch
) -> None:
    gguf_models.available.cache_clear()
    monkeypatch.setattr(importlib.util, "find_spec", lambda module: spec)
    try:
        assert gguf_models.available() is expected, name
    finally:
        gguf_models.available.cache_clear()


class Llama:
    """`llama_cpp.Llama` as the embedder uses it: what it was built with, what it embedded."""

    built: list[dict] = []

    def __init__(self, **options: object) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        Llama.built.append(options)

    def embed(self, texts: list[str], **options: object) -> list[list[float]]:
        self.calls.append((texts, options))
        return [[0.6, 0.8] for _ in texts]

    # as the generator uses it: one token a character, so a prompt's length is its tokens
    def tokenize(self, text: bytes, add_bos: bool) -> list[int]:
        return list(text)

    def detokenize(self, tokens: list[int]) -> bytes:
        return bytes(tokens)

    def create_chat_completion(self, messages: list[dict], **options: object) -> dict:
        self.calls.append(([messages[0]["content"]], options))
        return {"choices": [{"message": {"content": "Topic one | Topic two"}}]}


@pytest.fixture
def llama(monkeypatch) -> type[Llama]:
    """The extra may not be installed where the suite runs: the module is stood in for whole."""
    Llama.built = []
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=Llama))
    monkeypatch.setattr(gguf_models, "_download", lambda repo: Path(f"/models/{repo}.gguf"))
    return Llama


@pytest.mark.parametrize(
    ("name", "model", "tokens"),
    [
        ("cut at the model's own 512", BGE_SMALL, 512),
        ("an 8K model read at 1K", JINA_BASE, gguf_models.MAX_TOKENS),
    ],
)
def test_the_embedder_sets_llama_cpp_up_for_its_model(
    name: str, model: str, tokens: int, llama
) -> None:
    gguf_models.GgufEmbedder(model)

    (options,) = llama.built
    assert options["model_path"] == f"/models/{model}.gguf", name
    assert options["n_gpu_layers"] == -1, f"{name}: every layer on the GPU"
    assert options["n_ctx"] == options["n_batch"] == options["n_ubatch"] == tokens, name
    assert options["embedding"] is True


def test_the_embedder_answers_as_fastembed_does(llama) -> None:
    embedder = gguf_models.GgufEmbedder(BGE_SMALL)

    vectors = list(embedder.embed(["retries", "idempotency"]))
    assert [vector.tolist() for vector in vectors] == [
        pytest.approx([0.6, 0.8]),
        pytest.approx([0.6, 0.8]),
    ]
    assert all(vector.dtype == np.float32 for vector in vectors)
    (query,) = embedder.query_embed("retries")
    assert query.shape == (2,)
    calls = embedder._model.calls
    assert [(texts, opts["normalize"], opts["truncate"]) for texts, opts in calls] == [
        (["retries", "idempotency"], True, True),
        (["retries"], True, True),
    ]


def test_without_llama_cpp_the_model_says_what_it_needs(monkeypatch) -> None:
    monkeypatch.setattr(gguf_models, "available", lambda: False)

    with pytest.raises(RuntimeError, match="llama-cpp-python, which haskie installs there"):
        gguf_models.GgufEmbedder(BGE_SMALL)


def test_the_embed_path_refuses_a_gguf_the_settings_put_on_the_cpu(monkeypatch) -> None:
    monkeypatch.setattr(gguf_models, "available", lambda: True)
    monkeypatch.setattr(gguf_models, "GgufEmbedder", lambda name: pytest.fail("built anyway"))

    with pytest.raises(RuntimeError, match="a hardware setting other than cpu"):
        embed._model(BGE_SMALL, Accelerator.CPU)


def test_the_embed_path_routes_every_gguf_name_to_llama_cpp(monkeypatch) -> None:
    built: list[str] = []
    monkeypatch.setattr(gguf_models, "GgufEmbedder", built.append)
    embed._build_model.cache_clear()
    try:
        for name in gguf_models.PINS:
            embed._build_model(name, Accelerator.AUTO)
    finally:
        embed._build_model.cache_clear()

    assert built == list(gguf_models.PINS)


@pytest.mark.network
@pytest.mark.skipif(not gguf_models.available(), reason="llama.cpp is not installed here")
def test_the_real_file_loads_fast_and_embeds_as_onnx_does() -> None:
    """bge-small in GGUF on Metal against the same model in ONNX on the CPU. The load is timed
    after the download: the 10 s bar is for the app waiting on a model it already has."""
    import time

    gguf_models._download(BGE_SMALL)  # a download is not a load
    texts = [
        "A background job retries a failed HTTP call, so the call has to be idempotent.",
        "Exponential backoff with jitter spreads the retries and avoids a thundering herd.",
        "Never trust a wall clock for ordering: hosts drift apart by milliseconds.",
        "A transactional outbox writes the event in the same transaction as the state change.",
    ]

    started = time.perf_counter()
    gguf = gguf_models.GgufEmbedder(BGE_SMALL)
    loaded = time.perf_counter() - started
    on_gpu = np.array(list(gguf.embed(texts)))
    on_cpu = np.array(list(embed._model("BAAI/bge-small-en-v1.5", Accelerator.CPU).embed(texts)))

    assert loaded < 10, f"loaded in {loaded:.1f} s"
    assert (on_gpu * on_cpu).sum(axis=1).min() > 0.999


@pytest.mark.parametrize(
    ("name", "length", "read"),
    [
        ("a prompt that fits is read whole", 1000, 1000),
        (
            "a longer one is cut at its end to what the context holds beside the reply",
            9000,
            gguf_models.GENERATOR_TOKENS - gguf_models.CHAT_TOKENS - 60,
        ),
    ],
)
def test_the_generator_fits_its_prompt_to_the_context(
    name: str, length: int, read: int, llama
) -> None:
    """An excerpt is bounded in characters, and digits take a token each: llama.cpp refuses a
    prompt past the context, so it is cut, keeping the instructions at its start."""
    generator = gguf_models.GgufGenerator(gguf_models.DESCRIBER)
    prompt = "Write descriptors. " + "7" * (length - 19)

    assert generator.reply(prompt, 60) == "Topic one | Topic two", name

    (options,) = llama.built
    assert (options["n_ctx"], options["n_gpu_layers"]) == (gguf_models.GENERATOR_TOKENS, -1)
    ((sent,), asked) = generator._model.calls[0]
    assert (len(sent), sent.startswith("Write descriptors.")) == (read, True), name
    assert (asked["max_tokens"], asked["temperature"]) == (60, 0), "greedy"
