"""How a search's scores came to be, step by step, for the person reading them.

The score a result carries is set and reset along the pipeline: retrieval scores each collection's
chunks by its mode, the merge fuses collections by rank, a reranker replaces both, and passages,
excerpts and documents fold chunk scores their own way. Two searches that differ in one setting
give numbers on different scales, and two that differ only in what the reranker overrides give
the same numbers. So each step that sets or changes a score says how, in words, as it runs
(`flow`), and the answer carries that lineage beside its step timings. No IO here.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from haskie.collection.index import chunk_key
from haskie.settings import Fusion, Reranker, SearchMode, SearchSettings

if TYPE_CHECKING:
    from haskie.search.flow import Search
    from haskie.search.section import Group

_HARMONIC = "2·best·sum / (best + sum), between the best and twice it"

# One step's rule: how the step scored what it answered with, from the search, what the step read
# and what it answered; None when it set or changed no score this time.
type Rule = Callable[["Search", Any, Any], str | None]


def _retrieve(state: "Search", *_: Any) -> str | None:
    """Each collection's rows scored by its own mode. From the settings alone, as every rule of the
    ranking is, so each question of a search says the same and is recorded once."""
    where = state.plan
    rules: dict[str, list[str]] = {}
    for index, settings in where.indexes:
        rule = _retrieved(settings, where.vector is not None, where.embedding is not None)
        rules.setdefault(rule, []).append(index.collection)
    if len(rules) == 1:
        return next(iter(rules))
    return " ".join(f"{', '.join(names)}: {rule}" for rule, names in rules.items())


def _retrieved(settings: SearchSettings, embedded: bool, has_model: bool) -> str:
    """How one collection's retrieval scores a chunk. Mirrors `index.row_score` per mode."""
    if not embedded or settings.mode == SearchMode.FTS:
        lexical = (
            "BM25 score from LanceDB's full-text index: unbounded, higher is better, larger for "
            "rarer query words. Comparable within one search only."
        )
        return lexical if has_model else f"{lexical} No embedding model, so every mode is BM25."
    if settings.mode == SearchMode.VECTOR:
        return (
            "1 / (1 + d), d the squared L2 distance between the query and chunk embeddings: "
            "1 / (3 − 2·cosine) for unit vectors, 1 at cosine 1 and 0.33 at cosine 0."
        )
    if settings.fusion == Fusion.LINEAR:
        total = settings.vector_weight + settings.bm25_weight
        weight = settings.vector_weight / total if total > 0 else 0.5
        return (
            f"Hybrid: {weight:.2f} × vector + {1 - weight:.2f} × BM25, each min-max scaled to 0–1 "
            "over the candidates, a half that missed the chunk counting 0. The best candidate of "
            "any search scores near 1."
        )
    k = settings.rrf_k
    return (
        f"Hybrid: reciprocal rank fusion of the vector and BM25 ranks, the sum of 1 / ({k} + "
        f"rank) over the two, at most {2 / (k + 1):.4f}. Rank only."
    )


def _merge(state: "Search", *_: Any) -> str | None:
    """The fusion across collections, or none for one."""
    collections = len(state.plan.indexes)
    if collections == 1:
        return "One collection: its scores are kept."
    k = state.plan.settings.rrf_k
    return (
        f"Reciprocal rank fusion over the {collections} collections' rankings replaces "
        f"their scores: the sum of 1 / ({k} + rank) over the rankings a chunk is in, "
        f"{1 / (k + 1):.4f} for first place in one. Rank only."
    )


def _rerank(state: "Search", *_: Any) -> str | None:
    """The cross-encoder's logit in place of every score before it."""
    settings = state.plan.settings
    if settings.reranker == Reranker.NONE:
        return None
    return (
        f"The cross-encoder {settings.reranker_model} rescores up to {state.candidates} "
        "candidates, and the sigmoid of its logit, 1 / (1 + e^−logit), replaces every score before "
        "it: 0 to 1, 0.5 at logit 0, bounded but not calibrated. It reads the query and the chunk "
        "together, so the mode only decides which candidates it reads, and a chunk scores the same "
        "in every mode that finds it."
    )


_PASSAGE = (
    "A passage scores the harmonic mean of its best chunk and the sum of its matched chunks: "
    f"{_HARMONIC}."
)


def _fill_thin(*_: Any) -> str | None:
    return f"{_PASSAGE} A chunk a short passage grew into scores 0, so it adds nothing."


def _judge_thin(*_: Any) -> str | None:
    return _PASSAGE


def _fold(_: "Search", ranged: list, __: Any) -> str | None:
    """Only several questions change what a score is compared with."""
    if len(ranged) == 1:
        return None
    return (
        "Each question's passages keep the scores of their own search, and the questions take "
        "turns at the slots, so the list is not in score order."
    )


def _group(*_: Any) -> str | None:
    return "An excerpt scores its best passage."


def _new_hits(before: list["Group"], after: list["Group"]) -> bool:
    held = {chunk_key(hit) for one in before for hit in one.hits}
    return any(chunk_key(hit) not in held for one in after for hit in one.hits)


def _probe_gaps(state: "Search", before: list["Group"], after: list["Group"]) -> str | None:
    """The probe's passage, scored by a search of its own, when it added one."""
    if not _new_hits(before, after):
        return None
    fused = " fused by rank across collections" if len(state.plan.indexes) > 1 else ""
    return (
        "The passage the probe added scores by a full-text search of the missing words alone: "
        f"BM25{fused}, on another scale than the passages ranked for the question."
    )


def _fill(_: "Search", before: list["Group"], after: list["Group"]) -> str | None:
    if not _new_hits(before, after):
        return None
    return "Text filled in around and between passages scores 0; each passage keeps its score."


def _shortlist(*_: Any) -> str | None:
    return (
        "A document scores the harmonic mean of its best chunk and the sum of all its matched "
        f"chunks: {_HARMONIC}. Each section of it scores the same over its own chunks."
    )


# The steps that set or change a score, by the step names `flow` runs them under. A step missing
# here passes its scores on as it read them.
RULES: dict[str, Rule] = {
    "retrieve": _retrieve,
    "merge": _merge,
    "rerank": _rerank,
    "fill_thin": _fill_thin,
    "judge_thin": _judge_thin,
    "fold": _fold,
    "group": _group,
    "probe_gaps": _probe_gaps,
    "fill": _fill,
    "shortlist": _shortlist,
}
