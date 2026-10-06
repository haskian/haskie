"""Embedders that run on the Apple GPU through llama.cpp, from GGUF files.

ONNX Runtime reaches the Apple GPU through WebGPU (`embed`). llama.cpp is faster there still, 2x
to 3x for the models pinned here (`catalogue/seed.sql`), and runs a GGUF file on Metal as it
loads: a model is ready in at most 0.27 s, after a one-time Metal shader compile per machine (7.6
s, measured once, 0.1 s after).
Measured on 128 real chunks against the model's ONNX export on the CPU (M4 Pro), whose vectors
match sentence-transformers on the original weights:

- granite-english-r2, mradermacher f16: 2.16 s against 7.89 s (3.7x), worst cosine 1.00000
- e5-base-v2, ChristianAzinn fp16: 1.36 s against 5.50 s (4.0x), worst cosine 0.99999
- F2LLM-v2-160M, mradermacher f16: 2.05 s against 5.37 s (2.6x), worst cosine 1.00000

A file is pinned only where it matches to a worst cosine of 0.9999, each to a revision, so the
weights that run are the ones measured: a community file counts as the ONNX export it was
measured against. Left out: granite-97m and granite-small-english, which match but ran at half the
CPU's speed (mykor Q8_0 and BF16, mradermacher f16), and bekko's own files, which ran 4x faster but
drift on every text past 128 tokens, its local attention window (worst cosine 0.985 at F16). The
rerankers' files (keisuke-miyako's gte and Ettin) were not measured to match: llama-cpp-python
takes no (query, text) pair, and Ettin's head is its own. Each file carries its model's pooling,
which llama.cpp applies. The files run on the CPU too, but slower than ONNX there (arctic: 27 s
against 6 s), so they run on Metal only: `hardware.device` finds no device for them otherwise.

llama.cpp computes attention over a whole micro-batch, every token against every other, so a
larger micro-batch costs more: 512 tokens ran fastest, 1024 cost 12% more, 8192 was 17x slower.
So texts are read at up to `MAX_TOKENS`, as the MLX embedders read theirs.

One GGUF embedder serves no embedding profile: Qwen3-Embedding-0.6B (`VOCABULARY_EMBEDDER`),
which embeds the descriptors of a collection's vocabulary (`sections.vocabulary`). Qwen's own Q8_0
file, on 184 descriptor pairs labelled by hand, told synonyms from other concepts at AUC 0.87 and
from antonyms at 0.96, against 0.85 and 0.43 for granite-97m.

One GGUF file is a generator, not an embedder: Gemma-4-E2B-it (`DESCRIBER`), which writes a
section's descriptors when the settings ask for them (`sections.generated`). Its ggml-org Q4_0 file
was judged blind on 200 book sections against the Q8_0 file and the mlx-community 4-bit build:
4.04, 4.01 and 3.86 of 5, against 2.13 for c-TF-IDF, at 0.51 s a section on an M4 Pro (0.57 s
for Q8_0). It is 2.8 GB.

The `gguf` extra installs llama-cpp-python on Apple Silicon only. PyPI ships it as source, so the
install compiles llama.cpp with Metal: about 30 s, with the Xcode command-line tools and cmake.
"""

import functools
import importlib.util
import threading
from collections.abc import Iterator, Sequence
from itertools import batched
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from haskie.settings import Descriptors

MAX_TOKENS = 1024  # the longest text read, in tokens: see the module docstring
SEQUENCES = 64  # texts one micro-batch holds, as many as fit its tokens


class Pin(NamedTuple):
    """One GGUF file, at the revision measured, and the longest input its model reads."""

    revision: str
    file: str
    tokens: int  # the model's own context: the file holds no positions past it


PINS: dict[str, Pin] = {
    "mradermacher/granite-embedding-english-r2-GGUF": Pin(
        "0b0294d75be1ecfaea0ed5464b7c1cd3e2c15538", "granite-embedding-english-r2.f16.gguf", 8192
    ),
    "ChristianAzinn/e5-base-v2-gguf": Pin(
        "374d123d6f9257d6f056687cb742abe7048bed97", "e5-base-v2_fp16.gguf", 512
    ),
    "mradermacher/F2LLM-v2-160M-GGUF": Pin(
        "a1f45469b2b9b3a2d0df7150fad56f65a37b8937", "F2LLM-v2-160M.f16.gguf", 40960
    ),
}
# what a collection's vocabulary embeds its descriptors with (see above): no embedding profile's
VOCABULARY_EMBEDDER = "Qwen/Qwen3-Embedding-0.6B-GGUF"
HELPERS: dict[str, Pin] = {
    VOCABULARY_EMBEDDER: Pin(
        "370f27d7550e0def9b39c1f16d3fbaa13aa67728", "Qwen3-Embedding-0.6B-Q8_0.gguf", 32768
    ),
}
DESCRIBER = "ggml-org/gemma-4-E2B-it-GGUF"  # the generator that writes descriptors (see above)
GENERATORS: dict[str, Pin] = {
    DESCRIBER: Pin("b4243c156154b6dca9324415f8c7ccc098b4aed1", "gemma-4-E2B-it-Q4_0.gguf", 131072),
}


def describer(by: Descriptors) -> str | None:
    """The model strategy `by` writes its descriptors with; None for one that needs no model."""
    return DESCRIBER if by == Descriptors.LLM else None


# what one prompt and its reply take: a section's excerpt is at most `generated.EXCERPT_CHARS`,
# about 1,500 tokens of prose; a longer prompt is cut to fit (`GgufGenerator.reply`)
GENERATOR_TOKENS = 4096
CHAT_TOKENS = 32  # what the chat template wraps a prompt in (Gemma's takes 10)
ANSWER_TOKENS = 4  # of a yes-or-no reply read for the answer (`GgufGenerator.yes`)
ANSWER_CHOICES = 10  # the likeliest tokens at one of them a yes or a no must be among


def pin(name: str) -> Pin | None:
    """The pinned file of GGUF model `name`, an embedder or a generator; None for any other."""
    return PINS.get(name) or HELPERS.get(name) or GENERATORS.get(name)


@functools.cache
def available() -> bool:
    """Whether llama.cpp is installed here. haskie installs it on Apple Silicon only,
    where it builds with Metal, so installed means it runs on the GPU. A spec lookup, not an
    import: the import starts Metal, which the first time on a machine takes seconds."""
    return importlib.util.find_spec("llama_cpp") is not None


def _download(name: str) -> Path:
    """Where the pinned file of model `name` is on disk, downloaded on first use."""
    if not available():
        raise RuntimeError(
            f"{name} runs on llama.cpp with Metal, which is not installed: it needs Apple Silicon "
            "and llama-cpp-python, which haskie installs there"
        )
    from huggingface_hub import hf_hub_download

    found = pin(name)
    assert found is not None, f"{name} is no GGUF model"
    return Path(hf_hub_download(name, found.file, revision=found.revision))


class GgufEmbedder:
    """A GGUF embedder, answering `embed(texts)` and `query_embed(text)` as every embedder does
    (`onnx_models.Embedder`): one normalized vector per text, pooled as the file says."""

    def __init__(self, name: str) -> None:
        path = _download(name)  # first: it says what to install when llama.cpp is missing
        from llama_cpp import Llama

        found = pin(name)
        assert found is not None, f"{name} is no GGUF model"
        tokens = min(found.tokens, MAX_TOKENS)
        self._model = Llama(
            model_path=str(path),
            embedding=True,
            n_gpu_layers=-1,  # every layer on the GPU: `hardware.device` keeps these off the CPU
            # one micro-batch is one context: texts are packed into it, and cut to it
            n_ctx=tokens,
            n_batch=tokens,
            n_ubatch=tokens,
            n_seq_max=SEQUENCES,
            verbose=False,
        )
        self._tokens = tokens
        # what the model adds around a text ([CLS] and [SEP], or an end token)
        self._specials = len(self._model.tokenize(b""))
        # one context: two calls at once would decode into the same buffers
        self._lock = threading.Lock()

    def embed(self, texts: Sequence[str]) -> Iterator[Any]:
        # locked a batch at a time, so a search's query waits for one batch, not a whole part:
        # 128 bge-m3 chunks took 4.58 s in one call and 4.60 s in batches, with the same vectors
        for batch in batched(texts, SEQUENCES, strict=False):
            fitted = [self._fit(text) for text in batch]
            with self._lock:  # llama.cpp releases the GIL while it runs (measured: 13 ms stalls)
                vectors = self._model.embed(fitted, normalize=True, truncate=True)
            yield from (np.asarray(vector, dtype=np.float32) for vector in vectors)

    def query_embed(self, text: str) -> Iterator[Any]:
        return self.embed([text])

    def _fit(self, text: str) -> str:
        """`text` cut to what the context holds with the model's own tokens around it. llama.cpp's
        cut keeps the first tokens and drops the end token with the rest, and a vector pooled
        without it is another vector (measured past the cut, against ONNX: F2LLM 0.14, e5
        0.82). Cut here, as the ONNX loader cuts, the end token stays."""
        content = self._model.tokenize(text.encode(), add_bos=False)
        keep = self._tokens - self._specials
        if len(content) <= keep:
            return text
        # a detokenized cut can tokenize a little longer, so it shrinks until it fits
        while True:
            cut = self._model.detokenize(content[:keep]).decode(errors="ignore")
            if len(self._model.tokenize(cut.encode())) <= self._tokens:
                return cut
            keep -= 8


def leaning(logits: np.ndarray, yes: np.ndarray, no: np.ndarray) -> float | None:
    """P(yes) / (P(yes) + P(no)) from one token's logits, the likeliest spelling of each word,
    when a yes or a no is among the `ANSWER_CHOICES` likeliest tokens; None when neither is, and
    the reply leans neither way yet."""
    likeliest = np.argpartition(-logits, ANSWER_CHOICES)[:ANSWER_CHOICES]
    if not (np.isin(likeliest, yes).any() or np.isin(likeliest, no).any()):
        return None
    return float(1 / (1 + np.exp(logits[no].max() - logits[yes].max())))


class GgufGenerator:
    """A GGUF chat model, answering one prompt at a time with its greedy reply, or with how
    likely its answer to a yes-or-no question is yes."""

    def __init__(self, name: str) -> None:
        path = _download(name)
        from llama_cpp import Llama

        self._model = Llama(
            model_path=str(path), n_gpu_layers=-1, n_ctx=GENERATOR_TOKENS, verbose=False
        )
        self._lock = threading.Lock()  # one context, as `GgufEmbedder`'s

    def yes(self, prompt: str) -> float:
        """P(yes) / (P(yes) + P(no)) at the first token of the greedy reply to `prompt` that leans
        either way (`leaning`): a reply may open with a mark ("**Yes**"), so up to
        `ANSWER_TOKENS` tokens are read. 0.5 when none did."""
        from llama_cpp import llama_get_logits_ith

        chat, yes, no = self._answers
        text = chat(messages=[{"role": "user", "content": prompt}]).prompt
        with self._lock:
            tokens = self._model.tokenize(text.encode(), add_bos=False, special=True)
            self._model.reset()
            self._model.eval(tokens)
            for _ in range(ANSWER_TOKENS):
                # the last token's, the one row llama.cpp keeps without `logits_all`, which would
                # keep one per prompt token, on every prompt `reply` asks too
                logits = np.ctypeslib.as_array(
                    llama_get_logits_ith(self._model.ctx, -1), shape=(self._model.n_vocab(),)
                )
                found = leaning(logits, yes, no)
                if found is not None:
                    return found
                self._model.eval([int(logits.argmax())])
        return 0.5

    @functools.cached_property
    def _answers(self) -> tuple[Any, np.ndarray, np.ndarray]:
        """The chat template `create_chat_completion` applies, the file's own as llama-cpp-python
        reads it, for the prompts `yes` evaluates itself; and the tokens that read "yes" and "no"
        in any spelling ("Yes", " yes"). Once, on the first question: the vocabulary is read a
        token at a time."""
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        model = self._model
        chat = Jinja2ChatFormatter(
            template=model.metadata["tokenizer.chat_template"],
            eos_token=model._model.token_get_text(model.token_eos()),
            bos_token=model._model.token_get_text(model.token_bos()),
        )
        words = [
            model.detokenize([token]).decode(errors="ignore").strip().lower()
            for token in range(model.n_vocab())
        ]
        yes = np.array([token for token, word in enumerate(words) if word == "yes"])
        no = np.array([token for token, word in enumerate(words) if word == "no"])
        return chat, yes, no

    def reply(self, prompt: str, max_tokens: int) -> str:
        """The greedy reply to `prompt`, cut at its end to what the context holds beside the
        reply: an excerpt is bounded in characters, and digits or symbols take a token each (6,000
        characters of numbers measured 5,672 tokens), which llama.cpp refuses past the context."""
        room = GENERATOR_TOKENS - CHAT_TOKENS - max_tokens
        with self._lock:
            tokens = self._model.tokenize(prompt.encode(), add_bos=False)
            if len(tokens) > room:
                prompt = self._model.detokenize(tokens[:room]).decode(errors="ignore")
            answer = self._model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,  # greedy: a section described twice is described alike
            )
        return answer["choices"][0]["message"]["content"] or ""
