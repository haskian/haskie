"""The GGUF embedders, shaped as every embedder is: where they load, how llama.cpp is set up for
each, and one vector per text. The cases stand in for llama.cpp, and for the pins: the catalogue
pins no GGUF file today (`gguf_models`), but the runtime stays."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from haskie.indexing import embed, gguf_models
from haskie.settings import Accelerator

SHORT, LONG = "test/short-GGUF", "test/long-GGUF"


@pytest.fixture(autouse=True)
def pins(monkeypatch) -> None:
    """A model that reads 512 tokens and one that reads 8K."""
    monkeypatch.setitem(gguf_models.PINS, SHORT, gguf_models.Pin("0", "short.gguf", 512))
    monkeypatch.setitem(gguf_models.PINS, LONG, gguf_models.Pin("0", "long.gguf", 8192))


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

    # one token a word, a start and an end token around them, as [CLS] and [SEP] (an embedder);
    # or, `by_char`, one token a character with none around it (the generator's prompt)
    by_char = False

    def tokenize(self, text: bytes, add_bos: bool = True) -> list[int]:
        if self.by_char:
            return list(text)
        words = [7] * len(text.split())
        return [1, *words, 2] if add_bos else words

    def detokenize(self, tokens: list[int]) -> bytes:
        if self.by_char:
            return bytes(tokens)
        return b" ".join(b"w" for _ in tokens)

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
        ("cut at the model's own 512", SHORT, 512),
        ("an 8K model read at 1K", LONG, gguf_models.MAX_TOKENS),
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


def test_the_embedder_answers_as_every_embedder_does(llama) -> None:
    embedder = gguf_models.GgufEmbedder(SHORT)

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
        gguf_models.GgufEmbedder(SHORT)


def test_the_embed_path_refuses_a_gguf_the_settings_put_on_the_cpu(monkeypatch) -> None:
    monkeypatch.setattr(gguf_models, "available", lambda: True)
    monkeypatch.setattr(gguf_models, "GgufEmbedder", lambda name: pytest.fail("built anyway"))

    with pytest.raises(RuntimeError, match="a hardware setting other than cpu"):
        embed._model(SHORT, Accelerator.CPU)


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


@pytest.mark.parametrize(
    ("name", "words", "sent"),
    [
        ("a text that fits goes as it is", 500, 500),
        ("one at the model's cut, its two own tokens included, goes as it is", 510, 510),
        ("a longer text is cut so the model's end token fits after it", 2000, 510),
    ],
)
def test_a_text_is_cut_to_keep_the_end_token(name: str, words: int, sent: int, llama) -> None:
    """llama.cpp's own cut drops the end token with the rest, and a vector pooled without it is
    another vector (F2LLM: cosine 0.14 to its ONNX twin)."""
    embedder = gguf_models.GgufEmbedder(SHORT)

    list(embedder.embed(["word " * words]))

    ((texts, _),) = embedder._model.calls
    assert len(texts[0].split()) == sent, name


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
    name: str, length: int, read: int, llama, monkeypatch
) -> None:
    """An excerpt is bounded in characters, and digits take a token each: llama.cpp refuses a
    prompt past the context, so it is cut, keeping the instructions at its start."""
    monkeypatch.setattr(llama, "by_char", True)
    generator = gguf_models.GgufGenerator(gguf_models.DESCRIBER)
    prompt = "Write descriptors. " + "7" * (length - 19)

    assert generator.reply(prompt, 60) == "Topic one | Topic two", name

    (options,) = llama.built
    assert (options["n_ctx"], options["n_gpu_layers"]) == (gguf_models.GENERATOR_TOKENS, -1)
    ((sent,), asked) = generator._model.calls[0]
    assert (len(sent), sent.startswith("Write descriptors.")) == (read, True), name
    assert (asked["max_tokens"], asked["temperature"]) == (60, 0), "greedy"
