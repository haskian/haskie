"""A collection's controlled vocabulary: one preferred term per concept its sections' descriptors
name, and every other way they name it (a variant) pointing at that term, as a thesaurus's USE
reference does.

The describer writes each section's descriptors on its own, so one concept comes back in several
forms: "Event-driven architecture" and "event-driven architectures", "architecture trade-offs" and
"architectural trade-offs", "performance enhancement" and "performance improvement". Across the
26 books first measured, 33,270 descriptors held 23,122 distinct forms, 87% of them used once.

- **Variants.** A descriptor's variant is its lowercase form with its blanks collapsed
  (`variant`). Each variant keeps the form its collection writes it in most often, for display.
- **Vectors.** Each variant is embedded by Qwen3-Embedding-0.6B with its similarity prompt on
  every text (`INSTRUCTION`). Measured on 184 pairs labelled by hand, 40 of them synonyms: it ranks
  synonyms above other concepts at AUC 0.87, and above antonyms at 0.96, where the collections'
  own granite-97m ranks antonyms above synonyms (0.43): "synchronous calls" and "asynchronous
  calls" score 0.99 there. Neither more dimensions (Qwen3-4B, 2,560) nor another wording of the
  prompt did better, and centring or whitening the vectors did worse.
- **Same words** (`same_words`). Two variants whose words are the same once blanks and hyphens go
  ("time-out", "timeout"), or the same word for word by Snowball stem ("system call", "system
  calls"), are one concept with no model asked. A string distance is no such rule: over the whole
  phrase or word by word, it merged "block ordering" into "lock ordering" and "decoupling" into
  "coupling" at every bar that merged anything else.
- **Judged** (`JUDGE_PROMPT`). Every other pair of neighbours at a cosine of `JUDGE_COSINE` or
  more is asked of the describer, in both orders, and is one concept when the mean of its two
  P(yes) reaches the describer's bar (`gguf_models.Generator.same_concept`). On the labelled pairs
  that keeps 78% of synonyms, no antonym, and 12% (Qwen3.5-4B) or 17% (Gemma-4-E2B) of the other
  concepts, all of them close neighbours ("data validation" and "input validation"). Gemma alone
  said yes to 73% of the other concepts when the prompt did not say what is not the same, and the
  embedder's cosine alone, at the bar that keeps 90% of synonyms, lets 32% of the other concepts
  and 11% of the antonyms through.
- **Clustering** (`cluster`). Variants in order of use, most used first; each joins the nearest
  preferred term it is one concept with, or becomes one itself. Only preferred terms take
  variants, so a cluster never grows by a chain of near neighbours, and every variant points
  straight at its term.

No IO here: the caller passes the vectors and the verdicts.
"""

import re
from collections.abc import Iterable, Mapping, Sequence

import msgspec
import numpy as np

from haskie.search.probe import stem

INSTRUCTION = "Instruct: Retrieve semantically similar text\nQuery:"  # Qwen's own, before a text
NEIGHBOURS = 32  # the nearest variants each is compared with
JUDGE_COSINE = 0.93  # a pair of neighbours at or over it is judged
JUDGE_PROMPT = """Two descriptors, short phrases that say what a section of a technical book is \
about:
A: {a}
B: {b}
Do A and B name the same concept, so that a thesaurus would index both under one preferred term? \
Answer No when one is broader or narrower than the other, when they are related but distinct, or \
when they mean the opposite. Spelling, plural and wording variants of one concept count as the \
same.

Answer Yes or No."""
_WORDS = re.compile(r"\w+|[^\w\s-]")  # retain Unicode and symbols such as C++'s plus signs

type Pair = tuple[str, str]  # two variants, in sorted order


class Term(msgspec.Struct):
    """One preferred term: the form it is shown in, how often the collection's sections use it
    and its variants, and every variant it stands for, itself first."""

    variant: str
    term: str
    uses: int
    variants: list[str]


def variant(phrase: str) -> str:
    """A descriptor's variant: lowercase, its blanks collapsed and trimmed."""
    return " ".join(phrase.lower().split())


def pair(a: str, b: str) -> Pair:
    return (a, b) if a <= b else (b, a)


def same_words(a: str, b: str) -> bool:
    """Whether two variants say the same words (see the module)."""
    one, other = _WORDS.findall(a), _WORDS.findall(b)
    if not one or not other:
        return False
    if "".join(one) == "".join(other):
        return True
    return len(one) == len(other) and all(
        stem(x) == stem(y) for x, y in zip(one, other, strict=True)
    )


def neighbours(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each unit vector's `NEIGHBOURS` nearest others, by index and cosine, by exact search in
    blocks: a collection of 12,754 variants takes a second."""
    count = len(vectors)
    near = max(0, min(NEIGHBOURS, count - 1))
    ids = np.empty((count, near), dtype=np.int64)
    cosines = np.empty((count, near), dtype=np.float32)
    if near == 0:
        return ids, cosines
    block = 2048
    for start in range(0, count, block):
        scores = vectors[start : start + block] @ vectors.T
        rows = np.arange(len(scores))
        scores[rows, rows + start] = -np.inf  # not its own neighbour
        top = np.argpartition(-scores, near - 1, axis=1)[:, :near]
        ids[start : start + block] = top
        cosines[start : start + block] = np.take_along_axis(scores, top, axis=1)
    return ids, cosines


def to_judge(variants: Sequence[str], vectors: np.ndarray) -> list[Pair]:
    """The pairs of neighbours `cluster` asks a verdict on: at `JUDGE_COSINE` or more, and not
    the same words, each once, in sorted order."""
    ids, cosines = neighbours(vectors)
    found = {
        pair(variants[at], variants[other])
        for at in range(len(variants))
        for other, cosine in zip(ids[at], cosines[at], strict=True)
        if cosine >= JUDGE_COSINE and not same_words(variants[at], variants[other])
    }
    return sorted(found)


def cluster(
    variants: Sequence[str],
    uses: Sequence[int],
    vectors: np.ndarray,
    verdicts: Mapping[Pair, float],
    same_concept: float,
) -> list[int]:
    """Each variant's preferred term, by index (see the module). A pair whose verdict reaches
    `same_concept` is one concept; one `verdicts` lacks is not. Deterministic: ties in use go to
    the shorter variant, then the first in order."""
    ids, cosines = neighbours(vectors)
    order = sorted(
        range(len(variants)), key=lambda at: (-uses[at], len(variants[at]), variants[at])
    )
    preferred = [-1] * len(variants)
    for at in order:
        best, best_cosine = at, -np.inf
        for other, cosine in zip(ids[at], cosines[at], strict=True):
            if preferred[other] != other or cosine <= best_cosine:
                continue  # not a preferred term (yet), or no nearer than one found
            one, two = variants[at], variants[other]
            judged = cosine >= JUDGE_COSINE and verdicts.get(pair(one, two), 0.0) >= same_concept
            if same_words(one, two) or judged:
                best, best_cosine = int(other), float(cosine)
        preferred[at] = best
    return preferred


def terms(
    variants: Sequence[str], uses: Sequence[int], shown: Sequence[str], preferred: Sequence[int]
) -> list[Term]:
    """The preferred terms `cluster` chose, most used first: each shown as its own variant is
    shown (`shown`), with the uses of every variant it stands for."""
    members: dict[int, list[int]] = {}
    for at, lead in enumerate(preferred):
        members.setdefault(lead, []).append(at)
    found = [
        Term(
            variant=variants[lead],
            term=shown[lead],
            uses=sum(uses[at] for at in ats),
            variants=[variants[lead], *(variants[at] for at in ats if at != lead)],
        )
        for lead, ats in members.items()
    ]
    return sorted(found, key=lambda one: (-one.uses, one.variant))


def compact(phrases: Iterable[str], preferred: Mapping[str, str]) -> list[str]:
    """A section's descriptors in the vocabulary's terms: each replaced by the term its variant
    stands for, once, in order. A phrase the vocabulary does not hold stays as it is."""
    return list(dict.fromkeys(preferred.get(variant(one), one) for one in phrases))
