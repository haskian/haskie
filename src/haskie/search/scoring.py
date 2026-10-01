"""How a search's scores came to be, step by step, for the person reading them.

The score a result carries is set and reset along the pipeline: retrieval scores each collection's
chunks by its mode, the merge ranks each half over every collection and fuses the two, a reranker
replaces both, and passages, excerpts and documents fold chunk scores their own way. Two searches
that differ in one setting give numbers on different scales, and two that differ only in what the
reranker overrides give the same numbers. So each step that sets or changes a score says how, in
words, as it runs (`flow`), and the answer carries that lineage beside its step timings. No IO
here.
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from haskie.collection.index import ChunkKey, Hit, chunk_key
from haskie.search.passage import HitRange
from haskie.search.retrieval import VECTOR_RANKING
from haskie.settings import Fusion, Reranker, ScoreFold, SearchMode, SearchSettings

if TYPE_CHECKING:
    from haskie.search.flow import Search
    from haskie.search.retrieval import Pool, Ranged, Scanned
    from haskie.search.section import Group

# how each `score_fold` folds chunk scores into one (`passage.fold`), in the lineage's words
_FOLDS = {
    ScoreFold.SUM: "the sum of its matched chunks' scores",
    ScoreFold.MAX: "its best chunk's score",
    ScoreFold.HARMONIC: (
        "the harmonic mean of its best chunk and the sum of its matched chunks, 2·best·sum / "
        "(best + sum), between the best and twice it"
    ),
}

# One step's rule: how the step scored what it answered with, from the search, what the step read
# and what it answered; None when it set or changed no score this time.
type Rule = Callable[["Search", Any, Any], str | None]


def _retrieve(state: "Search", _: None, pool: "Pool") -> str | None:
    """Each collection's rows scored by the search it ran (`_ran`), which is its mode unless its
    table has no vectors. The same for every question of a search, so it is recorded once. Of
    several collections, a hybrid one answers its two halves unfused, for the merge to fuse over
    all of them."""
    where = state.plan
    several = len(where.indexes) > 1
    columns = _columns(pool)
    rules: dict[str, list[str]] = {}
    for index, settings in where.indexes:
        found = columns.get(index.collection, set())
        ran = _ran(found, settings, where.vector is not None)
        rule = UNFUSED if several and ran == SearchMode.HYBRID else _retrieved(settings, ran)
        if where.embedding is None:
            rule = f"{rule} No embedding model, so every mode is BM25."
        rules.setdefault(rule, []).append(index.collection)
    if len(rules) == 1:
        return next(iter(rules))
    return " ".join(f"{', '.join(names)}: {rule}" for rule, names in rules.items())


UNFUSED = (
    "Hybrid, unfused: its nearest chunks by vector distance and its best by BM25, for the merge "
    "to fuse over every collection at once."
)


SCORE_COLUMNS = ("_relevance_score", "_distance", "_score")  # fused, vector, BM25


def _columns(pool: "Pool") -> dict[str, set[str]]:
    """The score columns of each collection's rows, in one pass over the pool."""
    found: dict[str, set[str]] = {}
    for (name, *_), (_, row) in pool.rows.items():
        held = found.setdefault(name, set())
        held.update(column for column in SCORE_COLUMNS if column in row)
    return found


def _ran(columns: set[str], settings: SearchSettings, embedded: bool) -> SearchMode:
    """The search a collection ran, read off the score `columns` of the rows it returned: a table
    written without vectors answers any mode by full text. A hybrid one answers fused scores
    alone, or, among several collections, distances and BM25 scores. A collection that returned
    nothing kept, every span of it another's, says its settings."""
    if "_relevance_score" in columns or {"_distance", "_score"} <= columns:
        return SearchMode.HYBRID
    if columns:
        return SearchMode.VECTOR if "_distance" in columns else SearchMode.FTS
    return settings.mode if embedded else SearchMode.FTS


def _retrieved(settings: SearchSettings, ran: SearchMode) -> str:
    """How one collection's retrieval scored a chunk, by the search it ran."""
    if ran == SearchMode.FTS:
        return (
            "BM25 score from LanceDB's full-text index: unbounded, higher is better, larger for "
            "rarer query words. Comparable within one search only."
        )
    if ran == SearchMode.VECTOR:
        return (
            "1 / (1 + d), d the squared L2 distance between the query and chunk embeddings: "
            "1 / (3 − 2·cosine) for unit vectors, 1 at cosine 1 and 0.33 at cosine 0."
        )
    return f"Hybrid: {_fused(settings)}"


def _fused(settings: SearchSettings) -> str:
    """How the vector and BM25 halves of a hybrid search fuse, by the `fusion` setting."""
    if settings.fusion == Fusion.LINEAR:
        weight = settings.vector_share
        return (
            f"{weight:.2f} × vector + {1 - weight:.2f} × BM25, each min-max scaled to 0–1 "
            "over the candidates, a half that missed the chunk counting 0. The best candidate of "
            "any search scores near 1."
        )
    k = settings.rrf_k
    return (
        f"reciprocal rank fusion of the vector and BM25 ranks, the sum of 1 / ({k} + "
        f"rank) over the two, at most {2 / (k + 1):.4f}. Rank only."
    )


def _merge(state: "Search", pool: "Pool", *_: Any) -> str | None:
    """The fusion over every collection's halves, or none for one ranking."""
    collections = len(state.plan.indexes)
    if collections == 1:
        return "One collection: its scores are kept."
    settings = state.plan.settings
    if len(pool.rankings) == 2:
        rule = _fused(settings)
    elif any(pool.rankings.values()):  # one half alone, named by its key
        nearest = next(iter(pool.rankings)) == VECTOR_RANKING
        rule = _retrieved(settings, SearchMode.VECTOR if nearest else SearchMode.FTS)
    else:
        return None
    return f"Over all {collections} collections at once, as one table: {rule}"


def _rerank(state: "Search", *_: Any) -> str | None:
    """The cross-encoder's logit in place of every score before it."""
    settings = state.plan.settings
    if settings.reranker == Reranker.NONE:
        return None
    rule = (
        f"The cross-encoder {settings.reranker_model} rescores up to {state.candidates} "
        "candidates, and the sigmoid of its logit, 1 / (1 + e^−logit), replaces every score before "
        "it: 0 to 1, 0.5 at logit 0, bounded but not calibrated. It reads the query and the chunk "
        "together, so the mode only decides which candidates it reads, and a chunk scores the same "
        "in every mode that finds it."
    )
    floor, calibrated = state.plan.rerank_floor, state.plan.calibration
    if not state.plan.drops:
        rule = f"{rule} Its scores weigh every chunk, and none is dropped."
    elif floor > 0:
        # where the floor came from: the settings, or the reranker's calibration and its source
        said = "set"
        if settings.min_rerank_score is None and calibrated is not None:
            said = calibrated.source
        rule = f"{rule} Chunks it scores under {floor:g} ({said}) are dropped."
    if state.framed == state.query:
        return rule
    if settings.rerank_with_context:
        return f"{rule} It reads each question with the shared context in front."
    return f"{rule} It reads each question alone: the shared context only found the candidates."


def _passage(state: "Search") -> str:
    return f"A passage scores {_FOLDS[state.plan.settings.score_fold]}."


def _fill_thin(state: "Search", *_: Any) -> str | None:
    return f"{_passage(state)} A chunk a short passage grew into scores 0, so it adds nothing."


def _judge_thin(state: "Search", *_: Any) -> str | None:
    return _passage(state)


def _key(result: Hit | HitRange) -> ChunkKey:
    """What identifies a result through the fold: a hit's chunk, or a range's first chunk."""
    return chunk_key(result.hits[0] if isinstance(result, HitRange) else result)


def _swapped(read: Sequence[Hit | HitRange], kept: Sequence[Hit | HitRange]) -> str | None:
    """The superset swap (`collapse`), when it happened: a kept result scoring other than it did
    before the fold took its score from the slot it took."""
    own = {_key(one): one.score for one in read}
    if all(own.get(_key(one), one.score) == one.score for one in kept):
        return None
    return (
        "A fuller result that takes the slot of one it contains takes that one's score too, and "
        "lists it under it (also_in)."
    )


def _collapse_hits(_: "Search", scanned: "Scanned", kept: list[Hit]) -> str | None:
    return _swapped(scanned.hits, kept)


def _collapse_ranges(_: "Search", ranged: "Ranged", kept: list[HitRange]) -> str | None:
    return _swapped(ranged.ranges, kept)


def _fold(state: "Search", ranged: list["Ranged"], kept: list[HitRange]) -> str | None:
    """One question's ranges fold as `collapse_ranges` does; only several change what a score is
    compared with."""
    if len(ranged) == 1:
        return _collapse_ranges(state, ranged[0], kept)
    if state.plan.settings.reranker != Reranker.NONE:
        return (
            "Each chunk of a passage scores its best question, the reranker's score for it in "
            "that question's own ranking, and the passage folds them by its rule. The questions "
            "take turns at the slots, so the list is not in score order."
        )
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
    """The probe's passage, when it added one: judged by the reranker when one is on, else 0."""
    if not _new_hits(before, after):
        return None
    if state.plan.settings.reranker != Reranker.NONE:
        return (
            "The passage the probe added, found by a full-text search of the missing words, is "
            "scored by the reranker against each question whose words are missing, as the ranked "
            "chunks are, and kept only above its floor."
        )
    return (
        "The passage the probe added scores 0: a full-text search of the missing words alone "
        "found it, on another scale than the passages ranked for the question."
    )


def _joined(before: list["Group"], after: list["Group"]) -> bool:
    """Whether the fill bridged a gap: one passage after it holds the first chunk of several."""
    starts = {chunk_key(one.hits[0]) for group in before for one in group.ranges}
    return any(
        sum(chunk_key(hit) in starts for hit in one.hits) > 1
        for group in after
        for one in group.ranges
    )


def _fill(state: "Search", before: list["Group"], after: list["Group"]) -> str | None:
    if not _new_hits(before, after):
        return None
    rule = (
        "Text filled in around and between passages scores 0, so a passage it grows keeps its "
        "score."
    )
    if not _joined(before, after):
        return rule
    fold = _FOLDS[state.plan.settings.score_fold]
    return f"{rule} Two passages it joins make one, and that one scores {fold}."


def _map_sections(state: "Search", *_: Any) -> str | None:
    return (
        f"A section scores {_FOLDS[state.plan.settings.score_fold]}. The order is the order the "
        "sections were picked in to cover the scan, not the order of their scores. A document "
        "scores the same over every chunk of it the scan holds."
    )


def _rerank_excerpts(state: "Search", before: list, after: list) -> str | None:
    """Whole excerpts reranked, when the experiment ran and changed a score."""
    if [one.score for one in before] == [one.score for one in after]:
        return None
    return (
        "Each excerpt scores the reranker's sigmoid for its whole text, its heading path in "
        "front, against the best of the questions it answers, in place of its passages' scores. "
        "A single question's excerpts are sorted by it (rerank_excerpts, an experiment)."
    )


# The steps that set or change a score, by the step names `flow` runs them under. A step missing
# here passes its scores on as it read them.
RULES: dict[str, Rule] = {
    "retrieve": _retrieve,
    "merge": _merge,
    "rerank": _rerank,
    "collapse_hits": _collapse_hits,
    "fill_thin": _fill_thin,
    "judge_thin": _judge_thin,
    "collapse_ranges": _collapse_ranges,
    "fold": _fold,
    "group": _group,
    "probe_gaps": _probe_gaps,
    "fill": _fill,
    "map_sections": _map_sections,
    "rerank_excerpts": _rerank_excerpts,
}
