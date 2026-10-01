"""What each step of a search does, and the IO it takes to do it.

`flow.py` says in what order the steps run and which search runs which of them; this is what
they call. The pure folds — hits into ranges, ranges into passages, hits into documents — live in
`passage.py`, so what is left here is the IO: reading the collections, checking the models,
reading the markdown a range covers.

`scope` is the one place that decides which collections a search covers: the names the caller
gave, else the session's selection, else every collection.
"""

import asyncio
import statistics
import time
from collections.abc import Awaitable, Callable, Iterable
from contextlib import ExitStack
from itertools import islice
from typing import BinaryIO

import anyio.to_thread
import msgspec
import numpy as np

from haskie import cpu
from haskie.catalogue import catalogue
from haskie.catalogue.catalogue import UNCALIBRATED, EmbeddingModel, RerankerCalibration
from haskie.collection.collection import Collection
from haskie.collection.index import (
    EVERYTHING,
    FTS_COLUMN,
    ChunkKey,
    CollectionIndex,
    Hit,
    RowKey,
    Scope,
    chunk_key,
    cross_encode,
    first_per_span,
    gather_rows,
    logit,
    row_score,
)
from haskie.document import document
from haskie.indexing import embed_cache, hardware, mlx_models, models
from haskie.indexing.embed import embed_query
from haskie.indexing.hardware import Runtime
from haskie.logs import get_logger
from haskie.search import (
    aspects,
    collapse,
    passage,
    probe,
    section,
    section_map,
    session,
    text,
    thin,
)
from haskie.search import fill as filling
from haskie.search.passage import Excerpt, Passage, Sources
from haskie.settings import (
    FillValues,
    Reranker,
    ScoreFold,
    SearchMode,
    SearchSettings,
    load_user_settings,
)

_log = get_logger(__name__)


# --- what a search resolves before it reads anything ------------------------------


def rerank_floor(min_rerank_score: float | None, calibrated: float | None) -> float:
    """The reranker score a chunk must reach to stay: the settings' `min_rerank_score` when set,
    else the reranker's calibrated floor, else 0, which keeps every chunk. One rule for the search
    that drops chunks (`Plan.rerank_floor`) and the gaps that judge it later (`gaps`)."""
    if min_rerank_score is not None:
        return min_rerank_score
    return calibrated if calibrated is not None else 0.0


class Plan(msgspec.Struct):
    """Which collections a search covers and how, settled once for the whole fan-out.

    The ranking-level `settings` are the user's, except where a single collection is searched:
    then its own overrides are what the caller chose, and there is no second collection to
    disagree with.
    """

    settings: SearchSettings
    indexes: list[tuple[CollectionIndex, SearchSettings]]  # in the caller's order
    vector: list[float] | None  # the query embedding, None for a lexical search
    embedding: EmbeddingModel | None  # the model every index of the search embeds with
    # how the reranker's scores read, when one is on (`catalogue.calibration`)
    calibration: RerankerCalibration | None = None
    scope: Scope = EVERYTHING  # the documents and sections every read keeps to
    # the documents' names by id, filled as its reads meet them (`gather_rows`); one map shared
    # by every plan of a search
    document_names: dict[str, str] = msgspec.field(default_factory=dict)

    @property
    def rerank_floor(self) -> float:
        """The reranker score a chunk must reach to stay: the settings' `min_rerank_score` when
        set, else the reranker's calibrated floor, else 0, which keeps every chunk."""
        calibrated = self.calibration.floor if self.calibration is not None else None
        return rerank_floor(self.settings.min_rerank_score, calibrated)

    @property
    def names(self) -> list[str]:
        """The collections this search covers, in the caller's order."""
        return [index.collection for index, _ in self.indexes]


async def plan(
    names: list[str], queries: list[str], reranks: bool = True, scope: Scope = EVERYTHING
) -> list[Plan] | None:
    """One plan per query, or None when there is nothing left to search: the settings resolved
    and each model checked once for all of them, and each query embedded once. The plans differ
    only in their `vector`.

    A collection deleted since the caller chose it is skipped, so one stale name does not break
    every search. `reranks` False plans a search with no rerank step: its settings name no
    reranker, so it waits for none to warm and the log records none. Every read keeps to `scope`.
    """
    user = await load_user_settings()
    embedding = await catalogue.embedding_model(user)
    found = await Collection.for_search(names, embedding, scope)
    for name in names:
        if name not in found:
            _log.warning("session_collection_missing", collection=name)
    indexes = [(index, overrides.resolve_search(user)) for index, overrides in found.values()]
    if not indexes:
        return None
    settings = indexes[0][1] if len(indexes) == 1 else user.search
    if not reranks:  # a search with no rerank step waits for no reranker, and logs none
        settings = msgspec.structs.replace(settings, reranker=Reranker.NONE)

    vectors: list[list[float] | None] = [None] * len(queries)
    if embedding is not None and any(one.mode != SearchMode.FTS for _, one in indexes):
        await models.require_ready(models.ModelKind.EMBEDDING, embedding.name)
        vectors = await cpu.on_cpu(_embed_all, embedding, queries)
    calibrated = None
    if settings.reranker != Reranker.NONE:
        # before the fan-out
        await models.require_ready(models.ModelKind.RERANKER, settings.reranker_model)
        calibrated = await catalogue.calibration(settings.reranker_model)
    await asyncio.gather(*(index.open() for index, _ in indexes))
    document_names: dict[str, str] = {}
    return [
        Plan(
            settings=settings,
            indexes=indexes,
            vector=vector,
            embedding=embedding,
            calibration=calibrated,
            scope=scope,
            document_names=document_names,
        )
        for vector in vectors
    ]


def _embed_all(model: EmbeddingModel, queries: list[str]) -> list[list[float] | None]:
    """Every query embedded in one worker-thread hop."""
    return [embed_query(model, query) for query in queries]


# --- the rows a search works on ---------------------------------------------------


class Pool(msgspec.Struct):
    """The rows a search read and the ranking over them: what flows from step to step.

    The rows are kept whole rather than turned into `Hit`s as they are read, because the score a
    row answers with is decided by the steps after it.
    """

    rows: dict[ChunkKey, tuple[CollectionIndex, dict]]
    rankings: dict[str, list[ChunkKey]]  # one per collection, in that collection's own order
    ranked: list[tuple[ChunkKey, float]] = []  # merged, best first
    # the reranker's scores over the pool, best first, kept before its floor drops any: how close
    # the search came is what the search log records (`log.observe_ranking`)
    rerank_scores: list[float] = []


async def fan_out(where: Plan, query: str, candidates: int, vectors: bool = True) -> Pool:
    """Read every collection concurrently and keep one row per chunk.

    A chunk counts once. The same document may be a member of several of the chosen collections,
    and each of their tables then holds the same chunk. A caller searching them wants one hit per
    chunk, not one per collection that holds it. So the first collection in the caller's order
    that returned a chunk gets the credit, and the later copies are dropped before the ranks are
    counted. Otherwise a document in two collections would be fused with itself and outrank an
    equally good one that sits in a single collection. A chunk is its span of the document, not its
    `seq`: two collections that chunk one document two ways give one `seq` other text.

    A collection that fails to answer fails the search: a silent hole in a merged ranking reads as
    "no match".

    A document on its way out of a collection does not answer from it (`CollectionIndex.leaving`):
    its rows stay in the table until the removal queued for them runs.
    """
    chosen = {index.collection: settings for index, settings in where.indexes}

    async def read(index: CollectionIndex) -> list[dict]:
        settings = chosen[index.collection]
        wanted = None if settings.mode == SearchMode.FTS else where.vector
        try:
            return await index.search_rows(query, wanted, settings, candidates, vectors)
        except Exception:
            _log.exception("session_collection_search_failed", collection=index.collection)
            raise

    retrieved = await gather_rows([index for index, _ in where.indexes], read, where.document_names)
    pool = Pool(rows={}, rankings={index.collection: [] for index, _ in retrieved})
    for index, row in first_per_span((i, r) for i, rows in retrieved for r in rows):
        key = (index.collection, row["document_id"], row["seq"])
        pool.rows[key] = (index, row)
        pool.rankings[index.collection].append(key)
    return pool


def rrf_merge[T](ranked: list[list[T]], k: int) -> list[tuple[T, float]]:
    """Reciprocal rank fusion: every item scores the sum of `1 / (k + rank)` over the rankings it
    appears in, best first. Ties keep the order of first appearance.

    Pure, and the only merge that needs no calibration between the inputs: two LanceDB indexes
    score rows on their own scale, so their ranks are comparable where their scores are not.
    """
    scores: dict[T, float] = {}
    for ranking in ranked:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def merge(pool: Pool, rrf_k: int, candidates: int) -> Pool:
    """One ranking out of the per-collection ones, fused by rank (see `rrf_merge`).

    A single collection keeps its own scores: there is nothing to compare them with, and fusing a
    lone ranking with itself would only replace a real score with a rank.
    """
    if len(pool.rankings) == 1:
        keys = next(iter(pool.rankings.values()))
        merged = [(key, row_score(pool.rows[key][1])) for key in keys]
    else:
        merged = rrf_merge(list(pool.rankings.values()), rrf_k)
    return msgspec.structs.replace(pool, ranked=merged[:candidates])


async def rerank(pool: Pool, query: str, where: Plan) -> Pool:
    """Rescore the merged candidates with a cross-encoder, which reads query and chunk together.

    One pass for the whole search rather than one per collection: the pool it rescores is what
    every collection returned, and its scores are the only ones comparable across them. The model
    is CPU work, so it runs in a worker thread under one slot of the CPU budget.
    """
    settings = where.settings
    if settings.reranker == Reranker.NONE:
        return pool
    keys = [key for key, _ in pool.ranked]
    rows = [pool.rows[key][1] for key in keys]
    await cross_encode(query, rows, settings)  # scores the rows in place
    scored = [(key, row_score(row)) for key, row in zip(keys, rows, strict=True)]
    rescored = sorted(scored, key=lambda pair: pair[1], reverse=True)
    # the reranker's score has a scale: under the floor it judged the chunk no answer, and a
    # search that keeps it would fill a slot, or tag a question, with it
    kept = [(key, score) for key, score in rescored if score >= where.rerank_floor]
    scores = sorted((score for _, score in rescored), reverse=True)
    return msgspec.structs.replace(pool, ranked=kept, rerank_scores=scores)


class Scanned(msgspec.Struct):
    """The ranking as far down as a search scans, as hits, and the vectors of their rows.

    The vectors are kept beside the hits rather than on them: a `Hit` is what a caller reads, and
    a vector on it would be a thousand floats in every answer.
    """

    hits: list[Hit]
    vectors: list[collapse.Vector | None]


def scan(pool: Pool, limit: int) -> Scanned:
    """The best `limit` of the ranking, as the `Hit`s a caller cites and opens, and their vectors
    for `collapse` to compare them by."""
    taken = [(pool.rows[key], score) for key, score in pool.ranked[:limit]]
    return Scanned(
        hits=[index.hit(row, score) for (index, row), score in taken],
        vectors=[row.get("vector") for (_, row), _ in taken],
    )


# --- thin ranges -----------------------------------------------------------------


class Ranged(msgspec.Struct):
    """The hits merged into ranges, the thin ones grown or marked alone (`thin`), and every chunk
    those ranges hold with its vector: what the fold compares them by."""

    ranges: list[passage.HitRange]
    scanned: Scanned


async def fill_thin(
    scanned: Scanned, where: Plan, query: str, rerank_query: str, grows: bool = True
) -> Ranged:
    """The scanned hits as ranges, each thin one grown by the neighbours of its section that
    match `query`, else marked too short to stand alone (see `thin`).

    Its neighbours are read from the collection only when a range is thin, in one query per
    collection, and valued around the scanned hits (`fill.value`): by the reranker when one is on,
    since the scanned hits carry its scores already, else as `_values` does. The valuing and the
    growing run in one worker-thread hop. The reranker reads `rerank_query`
    (`flow.Search.rerank_query`); the word scores read `query`, the question alone. `grows`
    False only judges which thin ranges could grow, for the fill of an excerpts search to grow
    them, so a passage grows once.
    """
    settings = where.settings
    hit_ranges = passage.ranges(scanned.hits, settings.score_fold)
    wanted = thin.around(hit_ranges, settings.min_passage_chars, settings.max_passage_grow)
    rows = await _rows_at(where, wanted) if wanted else {}
    reranked = None
    if rows and settings.reranker != Reranker.NONE:
        read = [row for _, row in rows.values()]
        # scored in place and read back in `rows`' order: a (document, seq) key would let two
        # collections' chunk 5 of one document, cut two ways, share one score
        await cross_encode(rerank_query, read, settings)
        reranked = [row_score(row) for row in read]
    filled, signal = await cpu.on_cpu(
        _thin, hit_ranges, scanned, rows, reranked, where, query, grows
    )
    alone = sum(one.alone for one in filled.ranges)
    if filled.grown or alone:
        _log.info(
            "search_thin",
            signal=signal,
            grown=filled.grown,
            alone=alone,
            added=len(filled.added),
        )
    added = [rows[chunk_key(hit)][1].get("vector") for hit in filled.added]
    return Ranged(
        ranges=filled.ranges,
        scanned=Scanned(hits=[*scanned.hits, *filled.added], vectors=[*scanned.vectors, *added]),
    )


def _thin(
    hit_ranges: list[passage.HitRange],
    scanned: Scanned,
    rows: dict[ChunkKey, tuple[Hit, dict]],
    reranked: list[float] | None,
    where: Plan,
    query: str,
    grows: bool,
) -> tuple[thin.Filled, str]:
    """The neighbours valued (`reranked` are their reranker scores, when one is on) and the thin
    ranges grown by them, or only judged when they do not grow here (`thin.fill`)."""
    if not rows:
        values, signal = [], "none"
    elif reranked is not None and where.settings.fill_values == FillValues.ABSOLUTE:
        curve = where.calibration or UNCALIBRATED
        values = [filling.absolute(one, curve.beta_a, curve.beta_b) for one in reranked]
        signal = "reranker, absolute"
    elif reranked is not None:
        # on the logit scale: a linear value between the median and the best is only fair there
        reference = [logit(hit.score) for hit in scanned.hits]
        values, signal = _valued([logit(one) for one in reranked], reference), "reranker"
    else:
        (values,), signal = _values(
            list(zip((hit.text for hit in scanned.hits), scanned.vectors, strict=True)),
            [(hit.text, row.get("vector")) for hit, row in rows.values()],
            [(query, where.vector)],
        )
    settings = where.settings
    neighbours = filling.biased(
        {
            key: filling.Candidate(hit, worth)
            for (key, (hit, _)), worth in zip(rows.items(), values, strict=True)
        },
        settings.grow_bias,
    )
    found = thin.fill(
        hit_ranges,
        neighbours,
        settings.min_passage_chars,
        settings.max_passage_grow,
        settings.score_fold,
        grows,
    )
    return found, signal


async def _rows_at(where: Plan, wanted: set[ChunkKey]) -> dict[ChunkKey, tuple[Hit, dict]]:
    """The stored rows of these chunks, each with the `Hit` it would be, from the collections
    that hold them. Vectors only when the search has a query vector to compare them with."""
    by_collection: dict[str, list[RowKey]] = {}
    for collection, doc, seq in wanted:
        by_collection.setdefault(collection, []).append((doc, seq))
    vectors = where.vector is not None
    found = await _per_collection(
        where, by_collection, lambda index, keys: index.rows_at(keys, vectors)
    )
    hits = ((index.hit(row, 0.0), row) for index, rows in found for row in rows)
    return {chunk_key(hit): (hit, row) for hit, row in hits}


async def _per_collection[T](
    where: Plan,
    wanted: dict[str, T],
    read: Callable[[CollectionIndex, T], Awaitable[list[dict]]],
) -> list[tuple[CollectionIndex, list[dict]]]:
    """`read` over each collection of the search that `wanted` names, with what it names there,
    all at once (`gather_rows`)."""
    indexes = [index for index, _ in where.indexes if index.collection in wanted]
    return await gather_rows(
        indexes, lambda index: read(index, wanted[index.collection]), where.document_names
    )


def _valued(scores: list[float], reference: list[float]) -> list[float]:
    """Each score around the reference scores (`fill.value`): 0 at their median, 1 at their best."""
    floor, top = statistics.median(reference), max(reference)
    return [filling.value(score, floor, top) for score in scores]


def _values(
    reference: list[tuple[str, collapse.Vector | None]],
    candidates: list[tuple[str, collapse.Vector | None]],
    asked: list[tuple[str, collapse.Vector | None]],
) -> tuple[list[list[float]], str]:
    """What each candidate (text, vector) is worth to each question (one row per question),
    around the reference chunks, the ranked ones it grows next to (`_valued`); and by what: the
    cosine of the vectors when every text and question has one, else the share of the question's
    words each text holds (`thin.overlap`). Reference and candidates are scored together, so they
    are on one scale, and each text is read once however many questions there are."""
    every = [*reference, *candidates]
    vectors = [vector for _, vector in every]
    if all(vector is not None for _, vector in asked) and all(one is not None for one in vectors):
        found = collapse.unit_rows(vectors) @ collapse.unit_rows([one for _, one in asked]).T
        scores, signal = found.T.tolist(), "vector"
    else:
        held = [set(thin.terms(text)) for text, _ in every]
        words = [thin.terms(question) for question, _ in asked]
        scores = [[thin.overlap(each, one) for one in held] for each in words]
        signal = "words"
    cut = len(reference)
    return [_valued(per[cut:], per[:cut]) for per in scores], signal


# --- what the hits are folded into ------------------------------------------------


def _spaces(scanned: Scanned, where: Plan) -> collapse.Scan:
    texts = [hit.text for hit in scanned.hits]
    return collapse.spaces(texts, scanned.vectors, where.embedding, where.settings.mode)


def _fold_hits(scanned: Scanned, where: Plan, limit: int) -> list[Hit]:
    scan = _spaces(scanned, where)
    kept = collapse.hits(scanned.hits, scan, limit)
    _log_collapse(scan.deciding[0].kind, len(scanned.hits), kept, limit)
    return kept


def _fold_ranges(ranged: Ranged, where: Plan, limit: int | None) -> list[passage.HitRange]:
    scan = _spaces(ranged.scanned, where)
    kept = collapse.ranges(ranged.ranges, ranged.scanned.hits, scan, limit)
    _log_collapse(scan.deciding[0].kind, len(ranged.ranges), kept, limit)
    return kept


async def collapse_hits(scanned: Scanned, where: Plan, limit: int) -> list[Hit]:
    """The `limit` best hits, each with the near-duplicates it stands for (see `collapse`).

    CPU work that grows with the square of the scan - tens of milliseconds at the default depth -
    so it runs in a worker thread rather than on the event loop the search came in on, the
    comparison spaces included.
    """
    return await cpu.on_cpu(_fold_hits, scanned, where, limit)


def standing(ranged: Ranged) -> Ranged:
    """The ranges without those too short to stand alone (`thin`): what a passage may be."""
    return msgspec.structs.replace(ranged, ranges=[one for one in ranged.ranges if not one.alone])


async def collapse_ranges(ranged: Ranged, where: Plan, limit: int | None) -> list[passage.HitRange]:
    """The `limit` best hit ranges, each with the near-duplicates it stands for; every range that
    repeats none without a `limit`.

    Folded after `passage.ranges` (in `fill_thin`) rather than before: the chunks of one passage
    sit next to each other and read alike, and folding them would split the passage they make up.
    A worker thread runs the fold, as `collapse_hits` says.
    """
    return await cpu.on_cpu(_fold_ranges, ranged, where, limit)


def _picked(picks: list[aspects.Pick], scanned: list[Scanned]) -> Scanned:
    """The chunks of the picks, with their vectors: all the fold across the parts compares.
    Picks never overlap, so each chunk is in one of them."""
    vectors = {
        chunk_key(hit): vector
        for one in scanned
        for hit, vector in zip(one.hits, one.vectors, strict=True)
    }
    hits = [hit for pick in picks for hit in pick.span.hits]
    return Scanned(hits=hits, vectors=[vectors[chunk_key(hit)] for hit in hits])


def _cover(
    ranged: list[Ranged], labels: list[str], where: Plan, depth: int, cap: int
) -> list[passage.HitRange]:
    how = where.settings.score_fold
    picks = aspects.interleave([one.ranges for one in ranged], depth, cap, how)
    joined = _picked(picks, [one.scanned for one in ranged])
    scan = _spaces(joined, where)
    # none cut: the sections they fall in are what the answer counts (`sections`)
    kept = collapse.ranges([pick.span for pick in picks], joined.hits, scan, None)
    scans = [{chunk_key(hit): hit.score for hit in one.scanned.hits} for one in ranged]
    by_score = where.settings.reranker != Reranker.NONE
    found = aspects.tagged(kept, picks, labels, scans, by_score, how)
    _log_collapse(scan.deciding[0].kind, len(picks), kept, None)
    _log.info(
        "search_questions",
        questions=len(labels),
        picks=len(picks),
        absorbed=sum(pick.taken - 1 for pick in picks),
        kept=len(kept),
        folded=sum(collapse.places(one.also_in) for one in kept),
    )
    return found


async def cover(
    ranged: list[Ranged], labels: list[str], where: Plan, depth: int, cap: int
) -> list[passage.HitRange]:
    """The ranges across the parts of one question, in the order the parts took them, each with
    its near-duplicates folded in and tagged with the parts it answers (see `aspects`). None is
    cut: `sections` counts the answer. `ranged` holds one set of ranges and `labels` one question
    per part; with a reranker on, the tags read its scores (`aspects.tagged`). A worker thread
    runs it, as `collapse_hits` says."""
    return await cpu.on_cpu(_cover, ranged, labels, where, depth, cap)


def _log_collapse(
    space: str, candidates: int, kept: list[Hit] | list[passage.HitRange], limit: int | None
) -> None:
    """One line per search, so how often folding leaves an answer short can be counted. A fold
    without a `limit` cuts nothing, so it is never short (`sections` says whether the answer is)."""
    _log.info(
        "search_collapsed",
        space=space,
        candidates=candidates,
        kept=len(kept),
        folded=sum(collapse.places(item.also_in) for item in kept),
        short=None if limit is None else len(kept) < limit,
    )


async def read(hit_ranges: list[passage.HitRange]) -> list[Passage]:
    """These hit ranges as passages, in the order given.

    A passage is what the chunks of one section that sit next to each other say together (see
    `passage.ranges`), read out of the document by the range's own offsets (`passage.quote`). The
    ranges come cut and folded (`collapse_ranges`), so only the ones answered with are read.
    """
    texts = await _texts_of(hit_ranges)
    return [
        passage.quote(hit_range, text) for hit_range, text in zip(hit_ranges, texts, strict=True)
    ]


async def sections(
    hit_ranges: list[passage.HitRange], where: Plan, limit: int, labels: list[str] | None = None
) -> list[section.Group]:
    """The first `limit` sections the ranges (best first) fall in, each with every range of it
    (see `section`).

    The placements of the documents a section can open in are read first (`section.documents`),
    one query per collection, and only the heading path and char span of each chunk. `labels` are
    the questions of a search that asked several, to log which of them no section kept.
    """
    groups = await _grouped(hit_ranges, where, limit)
    answered = {label for one in groups for kept in one.ranges for label in kept.aspects}
    _log.info(
        "search_sections",
        ranges=len(hit_ranges),
        sections=len(groups),
        grouped=sum(len(one.ranges) for one in groups),
        short=len(groups) < limit,
        uncovered=None if labels is None else sum(1 for label in labels if label not in answered),
    )
    return groups


def budget(groups: list[section.Group], where: Plan) -> list[section.Group]:
    """The groups that fit the answer's budget (`section.within`), the last cut first."""
    kept = section.within(groups, where.settings.max_answer_chars)
    if len(kept) < len(groups):
        _log.info(
            "search_budget", cut=len(groups) - len(kept), chars=sum(one.chars for one in kept)
        )
    return kept


async def fill(
    groups: list[section.Group], questions: list[probe.Question], where: Plan
) -> list[section.Group]:
    """The groups, each with the chunks around and between its passages that answer too (see
    `fill`), while they fit the room the answer's budget leaves.

    The chunks near every kept passage are read in one query per collection, with the passages'
    own, and valued around them for each question (`_values`). Each chunk is worth its best
    question's value, and that question tags it, unless a reranker is on: then only its
    judgement tags (`aspects.tagged`).
    """
    reach = where.settings.max_passage_grow
    # each group's own: sections of one document can nest, and a chunk near two groups is a
    # candidate of each, for the one whose passage it continues
    nears = [
        {(one.collection, one.document_id, seq) for seq in filling.near(one, reach)}
        for one in groups
    ]
    wanted = set().union(*nears, (chunk_key(hit) for one in groups for hit in one.hits))
    rows = await _rows_at(where, wanted)
    budget = where.settings.max_answer_chars
    # with a reranker on, a question tags only what it judged an answer (`aspects.tagged`); the
    # fill weighs by vectors or words, so its chunks bring no tag of their own
    settings = where.settings
    tags = settings.reranker == Reranker.NONE
    absolute = None
    if not tags and settings.fill_values == FillValues.ABSOLUTE:
        absolute = await _absolute(sorted(set().union(*nears)), rows, questions, where)
    how, bias = settings.score_fold, settings.grow_bias
    return await cpu.on_cpu(
        _fill, groups, nears, rows, questions, reach, budget, tags, how, bias, absolute
    )


async def _absolute(
    near: list[ChunkKey],
    rows: dict[ChunkKey, tuple[Hit, dict]],
    questions: list[probe.Question],
    where: Plan,
) -> dict[ChunkKey, filling.Candidate]:
    """Each chunk near a passage valued as dsRAG does (`fill.absolute`): the reranker reads it
    against every question, one pass a question, and its best question's score, spread by the
    reranker's curve and less the penalty, is its value."""
    read = [key for key in near if key in rows]
    curve = where.calibration or UNCALIBRATED
    best: dict[ChunkKey, float] = {}
    for question in questions:
        copies = [dict(rows[key][1]) for key in read]
        await cross_encode(question.asked, copies, where.settings)  # scores the copies in place
        for key, row in zip(read, copies, strict=True):
            best[key] = max(row_score(row), best.get(key, 0.0))
    return {
        key: filling.Candidate(rows[key][0], filling.absolute(score, curve.beta_a, curve.beta_b))
        for key, score in best.items()
    }


def _fill(
    groups: list[section.Group],
    nears: list[set[ChunkKey]],
    rows: dict[ChunkKey, tuple[Hit, dict]],
    questions: list[probe.Question],
    reach: int,
    budget: int,
    tags: bool,
    how: ScoreFold,
    bias: float,
    absolute: dict[ChunkKey, filling.Candidate] | None = None,
) -> list[section.Group]:
    held = [key for one in groups for hit in one.hits if (key := chunk_key(hit)) in rows]
    near = list(dict.fromkeys(key for keys in nears for key in keys if key in rows))
    if absolute is not None:
        weighed, signal = absolute, "reranker, absolute"
    else:
        weighed, signal = _weigh(held, near, rows, questions)
    if not tags:
        weighed = {key: msgspec.structs.replace(one, aspect=None) for key, one in weighed.items()}
    weighed = filling.biased(weighed, bias)
    found = [
        one
        for at, group in enumerate(groups)
        for one in filling.fills(
            at, group, {key: weighed[key] for key in nears[at] if key in weighed}, reach
        )
    ]
    chosen = filling.choose(found, budget - sum(one.chars for one in groups))
    taken: dict[int, list[filling.Candidate]] = {}
    for one in chosen:
        taken.setdefault(one.group, []).extend(one.chunks)
    _log.info(
        "search_fill",
        signal=signal,
        fills=len(chosen),
        added=sum(len(one.chunks) for one in chosen),
        chars=sum(one.chars for one in chosen),
    )
    offered: list[set[ChunkKey]] = [set() for _ in groups]
    for one in found:
        offered[one.group].update(chunk_key(chunk.hit) for chunk in one.chunks)
    filled = [filling.apply(one, taken.get(at, []), how) for at, one in enumerate(groups)]
    settled = thin.settle(filled, offered)
    owed = sum(hit_range.owed for one in groups for hit_range in one.ranges)
    if owed:
        _log.info("search_settle", owed=owed, dropped=len(groups) - len(settled))
    return settled


def _weigh(
    held: list[ChunkKey],
    near: list[ChunkKey],
    rows: dict[ChunkKey, tuple[Hit, dict]],
    questions: list[probe.Question],
) -> tuple[dict[ChunkKey, filling.Candidate], str]:
    """Each chunk near a passage as a candidate: its best question's value around the passages'
    own chunks (`_values`), and that question."""
    if not near or not held:
        return {}, "none"
    values, signal = _values(
        [(rows[key][0].text, rows[key][1].get("vector")) for key in held],
        [(rows[key][0].text, rows[key][1].get("vector")) for key in near],
        [(question.asked, question.vector) for question in questions],
    )
    weighed: dict[ChunkKey, filling.Candidate] = {}
    for at, key in enumerate(near):
        best = max(range(len(questions)), key=lambda question: values[question][at])
        weighed[key] = filling.Candidate(rows[key][0], values[best][at], questions[best].label)
    return weighed, signal


async def _placement_rows(where: Plan, places: Iterable[section.Place]) -> list[tuple[str, dict]]:
    """Where every chunk of these documents sits, as (collection, row) pairs, one query per
    collection (`CollectionIndex.placement_rows`)."""
    wanted: dict[str, set[str]] = {}
    for collection, doc in places:
        wanted.setdefault(collection, set()).add(doc)
    read = await _per_collection(where, wanted, lambda index, docs: index.placement_rows(docs))
    return [(index.collection, row) for index, found in read for row in found]


async def _grouped(
    hit_ranges: list[passage.HitRange], where: Plan, limit: int
) -> list[section.Group]:
    """The first `limit` sections of the ranges, the placements they need read first."""
    rows = await _placement_rows(where, section.documents(hit_ranges, limit))
    return await cpu.on_cpu(
        _group, hit_ranges, rows, where.settings.max_section_chars, limit, where.scope.section_ids
    )


async def probe_gaps(
    groups: list[section.Group], questions: list[probe.Question], where: Plan
) -> list[section.Group]:
    """The groups, with one more passage where the questions use words none of them holds (see
    `probe`): the best passage a full-text search of those words finds that no group holds yet,
    tagged with the questions it helps, in its section. Nothing is searched when no word is
    missing.

    With a reranker on, what the search finds is judged as the ranked chunks were (`_judged`):
    scored against each question whose words are missing, dropped under the floor, tagged with
    the questions it clears it for. Without one, its BM25 score is on another scale than the
    ranked passages', so it scores 0 and is tagged by the words it holds (`probe.tags`)."""
    # inline, not in a worker thread: stems are cached (`probe.stem`), see `probe.vocabulary`
    wanted = probe.missing(questions, probe.covered(groups))
    if not wanted:
        return groups
    held = {chunk_key(hit) for one in groups for hit in one.hits}
    depth = probe.PROBE_SCAN + len(held)  # enough rows that the best new one is among them
    lexical = msgspec.structs.replace(where, vector=None)
    found = await fan_out(lexical, " ".join(wanted), depth, vectors=False)  # it only wants chunks
    pool = merge(found, where.settings.rrf_k, depth)
    hits = [hit for hit in scan(pool, depth).hits if chunk_key(hit) not in held]
    reranked = where.settings.reranker != Reranker.NONE
    scores: dict[str, dict[ChunkKey, float]] = {}
    if reranked and hits:
        hits, scores = await _judged(hits[: probe.PROBE_SCAN], pool, wanted, where)
    fresh = passage.ranges(hits, where.settings.score_fold)
    joined = None
    if fresh and reranked:
        found = aspects.question_scores([chunk_key(hit) for hit in fresh[0].hits], scores)
        best = msgspec.structs.replace(fresh[0], aspects=list(found), aspect_scores=found)
    elif fresh:
        best = msgspec.structs.replace(
            fresh[0],
            hits=[msgspec.structs.replace(hit, score=0.0) for hit in fresh[0].hits],
            score=0.0,
            aspects=probe.tags(fresh[0], wanted),
        )
    if fresh:
        (one,) = await _grouped([best], where, 1)
        placed = probe.placed(groups, one)
        joined, groups = len(placed) == len(groups), placed
    _log.info("search_probe", terms=list(wanted), found=bool(fresh), joined=joined)
    return groups


async def _judged(
    hits: list[Hit], pool: Pool, wanted: dict[str, list[probe.Question]], where: Plan
) -> tuple[list[Hit], dict[str, dict[ChunkKey, float]]]:
    """The probe's hits the reranker judges an answer to a question whose words are missing, each
    scored by its best such question, and each labelled question's scores of the hits that clear
    the floor (`Plan.rerank_floor`). The reranker reads each question alone."""
    # by what was asked: a question with its query vector is no dict key
    asked = list({one.asked: one for questions in wanted.values() for one in questions}.values())
    floor = where.rerank_floor
    best: dict[ChunkKey, float] = {}
    scores: dict[str, dict[ChunkKey, float]] = {}
    for question in asked:
        rows = [dict(pool.rows[chunk_key(hit)][1]) for hit in hits]
        await cross_encode(question.asked, rows, where.settings)  # scores the copies in place
        for hit, row in zip(hits, rows, strict=True):
            score, key = row_score(row), chunk_key(hit)
            if score < floor:
                continue
            best[key] = max(score, best.get(key, score))
            if question.label is not None:
                scores.setdefault(question.label, {})[key] = score
    kept = [
        msgspec.structs.replace(hit, score=best[key])
        for hit in hits
        if (key := chunk_key(hit)) in best
    ]
    return kept, scores


def _group(
    hit_ranges: list[passage.HitRange],
    rows: list[tuple[str, dict]],
    max_chars: int,
    limit: int,
    kept: frozenset[str],
) -> list[section.Group]:
    return section.group(hit_ranges, section.placements(rows), max_chars, limit, kept)


CHARS_PER_TOKEN = 4  # a rough English average: what an excerpt's length is judged against


async def rerank_excerpts(
    excerpts: list[Excerpt], questions: list[probe.Question], where: Plan
) -> list[Excerpt]:
    """Each excerpt scored as one text, its heading path in front, by the reranker against the
    questions it answers (every question when it names none), its score the best of them; a
    single question's excerpts sorted by it. An experiment (`rerank_excerpts`): long-document
    reranking scores the unit it returns rather than folding its chunks (SumRank, arXiv
    2603.24204; EBCAR, arXiv 2510.13329), but nothing here measures it against the fold yet.

    Only when every excerpt fits what the reranker reads (`_reads`): an excerpt cut short would
    be scored on its first part, and one scored so beside others scored whole, or by folded chunk
    scores, would sort on two scales. So the excerpts come back as they were, and it is logged.
    """
    settings = where.settings
    if not (excerpts and settings.rerank_excerpts and settings.reranker != Reranker.NONE):
        return excerpts
    started = time.perf_counter()
    reads = await _reads(settings.reranker_model)
    longest = max(len(one.header) + len(one.text) for one in excerpts) // CHARS_PER_TOKEN
    if longest > reads:
        _log.info("search_rerank_excerpts", applied=False, longest=longest, reads=reads)
        return excerpts
    asked = {question.label or question.asked: question.asked for question in questions}
    best: list[float] = [0.0] * len(excerpts)
    # one reranker pass a question, over the excerpts that answer it
    for label, question in asked.items():
        at = [n for n, one in enumerate(excerpts) if label in one.aspects or not one.aspects]
        rows = [{FTS_COLUMN: f"{excerpts[n].header}\n\n{excerpts[n].text}"} for n in at]
        await cross_encode(question, rows, settings)  # scores the rows in place
        for n, row in zip(at, rows, strict=True):
            best[n] = max(best[n], row_score(row))
    pairs = zip(excerpts, best, strict=True)
    rescored = [msgspec.structs.replace(one, score=score) for one, score in pairs]
    if len(questions) == 1:
        rescored.sort(key=lambda one: -one.score)
    _log.info(
        "search_rerank_excerpts",
        applied=True,
        excerpts=len(rescored),
        ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return rescored


async def _reads(model: str) -> int:
    """How many tokens of a pair `model` reads: what the catalogue says it takes, or less where
    its loader cuts shorter (the MLX rerankers read `mlx_models.MAX_PAIR_TOKENS`)."""
    context = (await catalogue.rerankers())[model].context_tokens
    if hardware.runtime(model) == Runtime.MLX:
        return min(context, mlx_models.MAX_PAIR_TOKENS)
    return context


async def read_excerpts(groups: list[section.Group]) -> list[Excerpt]:
    """Each group as one excerpt, its passages read by their own offsets in one worker-thread
    hop for the whole search."""
    texts = iter(await _texts_of([hit_range for one in groups for hit_range in one.ranges]))
    return [section.excerpt(one, list(islice(texts, len(one.ranges)))) for one in groups]


async def _texts_of(hit_ranges: list[passage.HitRange]) -> list[str]:
    """The markdown each range covers, read by seeking to its byte offsets rather than reading the
    document.

    One worker thread for the whole search and one open file per document, however many ranges
    each holds. Measured: ten ranges cost 166us in a single hop against 717us fanned out one hop
    per document - a hop costs more than the few kilobytes it would overlap.
    """
    return await anyio.to_thread.run_sync(_read_texts, hit_ranges)


def _read_texts(hit_ranges: list[passage.HitRange]) -> list[str]:
    """Every range's text, in the order asked for. Sync: the caller runs it in a worker thread,
    where the seeks and reads are ordinary blocking IO.

    A chunk row's byte offsets fall on character boundaries, so each slice decodes whole. A
    markdown rewritten since it was indexed gives a wrong passage either way, but it must not fail
    the search, so a broken character is replaced rather than raised."""
    with ExitStack() as stack:
        handles: dict[str, BinaryIO] = {}
        texts: list[str] = []
        for hit_range in hit_ranges:
            path = hit_range.hits[0].markdown_file
            if path not in handles:
                handles[path] = stack.enter_context(open(path, "rb"))
            handle = handles[path]
            handle.seek(hit_range.byte_start)
            raw = handle.read(hit_range.byte_end - hit_range.byte_start)
            texts.append(raw.decode(errors="replace"))
        return texts


async def shortlist(
    hits: list[Hit], where: Plan, limit: int, sections: int, how: ScoreFold
) -> Sources:
    """Which documents these hits came from, one row per document, and the collections to select
    to read them.

    Its score folds every chunk it matched by `how` (`passage.fold`), `sections` says where in it
    the answer sits, and `collections` names
    which of the searched collections hold it, but for one it is on its way out of
    (`Collection.holding`), where a follow-up search would not find it. `Sources.collections` is the
    cover: the fewest collections a follow-up search has to select to reach every row.
    """
    # the shortlist is cut first: only a document that made it is worth a membership and a
    # description, and both are one query for the whole of it
    kept = passage.top_documents(hits, limit, how)
    docs = {group[0].document_id for group in kept}
    held, described = await asyncio.gather(
        Collection.holding(docs, where.names), document.descriptions_of(docs)
    )
    found = passage.fold_sources(kept, held, sections, how)
    document.fill_descriptions(found.documents, described)
    return found


async def map_sections(scanned: Scanned, where: Plan, limit: int) -> section_map.SectionMap:
    """The `limit` sections the scanned hits cover the topic with (`section_map`), each with its
    descriptors, and the collections to select to read them.

    Reads at once the chunk placements of every document the scan reached (one query per
    collection) and the corpus mean to centre on; then the picks' descriptors and memberships. The
    selection runs in one worker-thread hop."""
    hits = scanned.hits
    if not hits:
        return section_map.SectionMap(sections=[], collections=[])
    places = {(hit.collection, hit.document_id) for hit in hits}
    vectored = all(one is not None for one in scanned.vectors)
    embedding = where.embedding if vectored else None
    rows, centre = await asyncio.gather(
        _placement_rows(where, places),
        Collection.centre(where.names, embedding.cache_name if embedding else None),
    )
    candidates, picked = await cpu.on_cpu(
        _map, scanned, rows, centre, where.settings, limit, embedding is not None
    )
    # the related sections too: a follow-up search scoped to `collections` has to reach them
    listed = [*picked.picks, *(at for near in picked.related.values() for at, _ in near)]
    picked_docs = {candidates[at].document_id for at in listed}
    described, held = await asyncio.gather(
        _descriptors([candidates[at] for at in picked.picks]),
        Collection.holding(picked_docs, where.names),
    )
    found = section_map.mapped(hits, candidates, picked, described)
    _log.info(
        "search_map",
        chunks=len(hits),
        candidates=len(candidates),
        picked=len(found),
        documents=len(picked_docs),
        coverage=[round(share, 3) for share in picked.coverage],
        centred=centre is not None,
        vectors=embedding is not None,
        lifted=picked.lifted,
        # picks their cache entry holds no section of: a document reconverted since it was indexed
        sections_missing=sorted(
            {one.document for one in found if (one.collection, one.id) not in described}
        ),
    )
    memberships = {
        one.document_id: held.get(one.document_id, [one.collection])
        for one in (candidates[at] for at in listed)
    }
    return section_map.SectionMap(sections=found, collections=passage.min_cover(memberships))


async def _descriptors(picks: list[section_map.Candidate]) -> dict[tuple[str, str], list[str]]:
    """Each pick's descriptors by (collection, section id), read from the cache entry its
    collection indexed the document from: section ids are places in one chunking, so the same id
    in another chunking names another section. Each entry's file is read once."""
    entries = await Collection.indexed_entries((one.collection, one.document_id) for one in picks)
    files = sorted({(doc, id) for (_, doc), id in entries.items()})
    sections = await asyncio.gather(*(embed_cache.read_sections(*one) for one in files))
    read = dict(zip(files, sections, strict=True))
    return {
        (collection, one.id): one.descriptors
        for (collection, doc), id in entries.items()
        for one in read[(doc, id)]
    }


def _map(
    scanned: Scanned,
    rows: list[tuple[str, dict]],
    centre: np.ndarray | None,
    settings: SearchSettings,
    limit: int,
    vectored: bool,
) -> tuple[list[section_map.Candidate], section_map.Picked]:
    """The sections the scan reached, and the ones picked."""
    hits = scanned.hits
    candidates = section_map.candidates(
        hits, section.placements(rows), settings.max_section_chars, settings.score_fold
    )
    if vectored:
        vectors = [vector for vector in scanned.vectors if vector is not None]
        near = section_map.nearness(vectors, centre, candidates)
        weights = np.maximum([hit.score for hit in hits], 0.0)
        picked = section_map.cover(weights, near, candidates, limit)
    else:
        picked = section_map.by_rank([hit.text for hit in hits], candidates, limit)
    return candidates, picked


# --- which collections a search covers --------------------------------------------


async def scope(session_id: str | None, collections: str | None) -> list[str]:
    """Which collections a search covers: the comma-separated `collections` if the caller named
    any, else the session's selection if it has one, else every collection.

    A name nobody owns is a mistake in the request, not an empty result — unlike a session's
    stale name, which `plan` skips, because the caller did not choose it just now.
    """
    named = text.split_collections(collections)
    if named:
        return await text.checked_names(named)
    selected = await session.collections_for(session_id) if session_id else []
    return selected or await text.checked_names(None)
