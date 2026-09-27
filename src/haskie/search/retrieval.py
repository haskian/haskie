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
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from itertools import islice
from typing import BinaryIO

import anyio.to_thread
import msgspec

from haskie import cpu
from haskie.catalogue import catalogue
from haskie.catalogue.catalogue import EmbeddingModel
from haskie.collection.collection import Collection
from haskie.collection.index import (
    ChunkKey,
    CollectionIndex,
    Hit,
    RowKey,
    chunk_key,
    cross_encode,
    first_per_key,
    gather_rows,
    logit,
    row_key,
    row_score,
)
from haskie.document import document
from haskie.indexing import models
from haskie.indexing.embed import embed_query
from haskie.logs import get_logger
from haskie.search import aspects, collapse, passage, probe, section, session, text, thin
from haskie.search import fill as filling
from haskie.search.passage import Excerpt, Passage, Sources
from haskie.settings import Reranker, SearchMode, SearchSettings, load_user_settings

_log = get_logger(__name__)


# --- what a search resolves before it reads anything ------------------------------


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

    @property
    def names(self) -> list[str]:
        """The collections this search covers, in the caller's order."""
        return [index.collection for index, _ in self.indexes]


async def plan(names: list[str], queries: list[str]) -> list[Plan] | None:
    """One plan per query, or None when there is nothing left to search: the settings resolved
    and each model checked once for all of them, and each query embedded once. The plans differ
    only in their `vector`.

    A collection deleted since the caller chose it is skipped, so one stale name does not break
    every search.
    """
    user = await load_user_settings()
    found = await Collection.load_overrides(names)
    for name in names:
        if name not in found:
            _log.warning("session_collection_missing", collection=name)
    plans = [
        (Collection(name), found[name].resolve_search(user)) for name in names if name in found
    ]
    if not plans:
        return None
    settings = plans[0][1] if len(plans) == 1 else user.search

    embedding = await catalogue.embedding_model(user)
    vectors: list[list[float] | None] = [None] * len(queries)
    if embedding is not None and any(one.mode != SearchMode.FTS for _, one in plans):
        await models.require_ready(models.ModelKind.EMBEDDING, embedding.name)
        vectors = await cpu.on_cpu(_embed_all, embedding, queries)
    if settings.reranker != Reranker.NONE:
        # before the fan-out
        await models.require_ready(models.ModelKind.RERANKER, settings.reranker_model)
    indexes = [(one.index_with(embedding), where) for one, where in plans]
    await asyncio.gather(*(index.open() for index, _ in indexes))
    return [
        Plan(settings=settings, indexes=indexes, vector=vector, embedding=embedding)
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

    rows: dict[RowKey, tuple[CollectionIndex, dict]]
    rankings: dict[str, list[RowKey]]  # one per collection, in that collection's own order
    ranked: list[tuple[RowKey, float]] = []  # merged, best first


async def fan_out(where: Plan, query: str, candidates: int, vectors: bool = True) -> Pool:
    """Read every collection concurrently and keep one row per chunk.

    A chunk counts once. The same document may be a member of several of the chosen collections,
    and each of their tables then holds the same chunk. A caller searching them wants one hit per
    chunk, not one per collection that holds it. So the first collection in the caller's order
    that returned a chunk gets the credit, and the later copies are dropped before the ranks are
    counted. Otherwise a document in two collections would be fused with itself and outrank an
    equally good one that sits in a single collection.

    A collection that fails to answer fails the search: a silent hole in a merged ranking reads as
    "no match".
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

    retrieved = await gather_rows([index for index, _ in where.indexes], read)
    pool = Pool(rows={}, rankings={index.collection: [] for index, _ in retrieved})
    for index, row in first_per_key((i, r) for i, rows in retrieved for r in rows):
        key = row_key(row)
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


async def rerank(pool: Pool, query: str, settings: SearchSettings) -> Pool:
    """Rescore the merged candidates with a cross-encoder, which reads query and chunk together.

    One pass for the whole search rather than one per collection: the pool it rescores is what
    every collection returned, and its scores are the only ones comparable across them. The model
    is CPU work, so it runs in a worker thread under one slot of the CPU budget.
    """
    if settings.reranker == Reranker.NONE:
        return pool
    rows = [pool.rows[key][1] for key, _ in pool.ranked]
    rescored = [(row_key(row), row_score(row)) for row in await cross_encode(query, rows, settings)]
    return msgspec.structs.replace(pool, ranked=rescored)


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
    scanned: Scanned, where: Plan, query: str, framed: str, grows: bool = True
) -> Ranged:
    """The scanned hits as ranges, each thin one grown by the neighbours of its section that
    match `query`, else marked too short to stand alone (see `thin`).

    Its neighbours are read from the collection only when a range is thin, in one query per
    collection, and valued around the scanned hits (`fill.value`): by the reranker when one is on,
    since the scanned hits carry its scores already, else as `_values` does. The valuing and the
    growing run in one worker-thread hop. The reranker reads the `framed` question, the context in
    front; the word scores read `query`, the question alone (`aspects.Questions.framed`). `grows`
    False only judges which thin ranges could grow, for the fill of an excerpts search to grow
    them, so a passage grows once.
    """
    settings = where.settings
    hit_ranges = passage.ranges(scanned.hits)
    wanted = thin.around(hit_ranges, settings.min_passage_chars, settings.max_passage_grow)
    rows = await _rows_at(where, wanted) if wanted else {}
    reranked = None
    if rows and settings.reranker != Reranker.NONE:
        read = [row for _, row in rows.values()]
        rescored = {
            row_key(row): row_score(row) for row in await cross_encode(framed, read, settings)
        }
        reranked = [rescored[row_key(row)] for row in read]
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
    neighbours = {
        key: filling.Candidate(hit, worth)
        for (key, (hit, _)), worth in zip(rows.items(), values, strict=True)
    }
    settings = where.settings
    found = thin.fill(
        hit_ranges, neighbours, settings.min_passage_chars, settings.max_passage_grow, grows
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
    return await gather_rows(indexes, lambda index: read(index, wanted[index.collection]))


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


def _spaces(scanned: Scanned, model: EmbeddingModel | None, mode: SearchMode) -> collapse.Scan:
    return collapse.spaces([hit.text for hit in scanned.hits], scanned.vectors, model, mode)


def _fold_hits(
    scanned: Scanned, model: EmbeddingModel | None, mode: SearchMode, limit: int
) -> list[Hit]:
    scan = _spaces(scanned, model, mode)
    kept = collapse.hits(scanned.hits, scan, limit)
    _log_collapse(scan.deciding[0].kind, len(scanned.hits), kept, limit)
    return kept


def _fold_ranges(
    ranged: Ranged, model: EmbeddingModel | None, mode: SearchMode, limit: int | None
) -> list[passage.HitRange]:
    scan = _spaces(ranged.scanned, model, mode)
    kept = collapse.ranges(ranged.ranges, ranged.scanned.hits, scan, limit)
    _log_collapse(scan.deciding[0].kind, len(ranged.ranges), kept, limit)
    return kept


async def collapse_hits(
    scanned: Scanned, model: EmbeddingModel | None, mode: SearchMode, limit: int
) -> list[Hit]:
    """The `limit` best hits, each with the near-duplicates it stands for (see `collapse`).

    CPU work that grows with the square of the scan - tens of milliseconds at the default depth -
    so it runs in a worker thread rather than on the event loop the search came in on, the
    comparison spaces included.
    """
    return await cpu.on_cpu(_fold_hits, scanned, model, mode, limit)


def standing(ranged: Ranged) -> Ranged:
    """The ranges without those too short to stand alone (`thin`): what a passage may be."""
    return msgspec.structs.replace(ranged, ranges=[one for one in ranged.ranges if not one.alone])


async def collapse_ranges(
    ranged: Ranged, model: EmbeddingModel | None, mode: SearchMode, limit: int | None
) -> list[passage.HitRange]:
    """The `limit` best hit ranges, each with the near-duplicates it stands for; every range that
    repeats none without a `limit`.

    Folded after `passage.ranges` (in `fill_thin`) rather than before: the chunks of one passage
    sit next to each other and read alike, and folding them would split the passage they make up.
    A worker thread runs the fold, as `collapse_hits` says.
    """
    return await cpu.on_cpu(_fold_ranges, ranged, model, mode, limit)


def _picked(picks: list[aspects.Pick], scanned: list[Scanned]) -> Scanned:
    """The chunks of the picks, with their vectors: all the fold across the parts compares.
    Picks never overlap, so each chunk is in one of them."""
    vectors = {
        (hit.collection, hit.document, hit.seq): vector
        for one in scanned
        for hit, vector in zip(one.hits, one.vectors, strict=True)
    }
    hits = [hit for pick in picks for hit in pick.span.hits]
    return Scanned(
        hits=hits, vectors=[vectors[(hit.collection, hit.document, hit.seq)] for hit in hits]
    )


def _cover(
    ranged: list[Ranged],
    labels: list[str],
    model: EmbeddingModel | None,
    mode: SearchMode,
    depth: int,
    cap: int,
) -> list[passage.HitRange]:
    picks = aspects.interleave([one.ranges for one in ranged], depth, cap)
    joined = _picked(picks, [one.scanned for one in ranged])
    scan = _spaces(joined, model, mode)
    # none cut: the sections they fall in are what the answer counts (`sections`)
    kept = collapse.ranges([pick.span for pick in picks], joined.hits, scan, None)
    found = aspects.tagged(kept, picks, labels)
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
    ranged: list[Ranged],
    labels: list[str],
    model: EmbeddingModel | None,
    mode: SearchMode,
    depth: int,
    cap: int,
) -> list[passage.HitRange]:
    """The ranges across the parts of one question, in the order the parts took them, each with
    its near-duplicates folded in and tagged with the parts it answers (see `aspects`). None is
    cut: `sections` counts the answer. `ranged` holds one set of ranges and `labels` one question
    per part. A worker thread runs it, as `collapse_hits` says."""
    return await cpu.on_cpu(_cover, ranged, labels, model, mode, depth, cap)


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

    The outlines of the documents a section can open in are read first (`section.documents`),
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
    question's value, and that question tags it.
    """
    reach = where.settings.max_passage_grow
    wanted = {
        (one.collection, one.document, seq)
        for one in groups
        for seq in filling.near(one, reach) | {hit.seq for hit in one.hits}
    }
    rows = await _rows_at(where, wanted)
    budget = where.settings.max_answer_chars
    return await cpu.on_cpu(_fill, groups, rows, questions, reach, budget)


def _fill(
    groups: list[section.Group],
    rows: dict[ChunkKey, tuple[Hit, dict]],
    questions: list[probe.Question],
    reach: int,
    budget: int,
) -> list[section.Group]:
    held = [key for one in groups for hit in one.hits if (key := chunk_key(hit)) in rows]
    near = {
        key: at
        for at, one in enumerate(groups)
        for seq in filling.near(one, reach)
        if (key := (one.collection, one.document, seq)) in rows
    }
    weighed, signal = _weigh(held, list(near), rows, questions)
    candidates: dict[int, dict[ChunkKey, filling.Candidate]] = {}
    for key, chunk in weighed.items():
        candidates.setdefault(near[key], {})[key] = chunk
    found = [
        one
        for at, group in enumerate(groups)
        for one in filling.fills(at, group, candidates.get(at, {}), reach)
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
    return [filling.apply(one, taken.get(at, [])) for at, one in enumerate(groups)]


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


async def _grouped(
    hit_ranges: list[passage.HitRange], where: Plan, limit: int
) -> list[section.Group]:
    """The first `limit` sections of the ranges, the outlines they need read first."""
    wanted: dict[str, set[str]] = {}
    for collection, doc in section.documents(hit_ranges, limit):
        wanted.setdefault(collection, set()).add(doc)
    read = await _per_collection(where, wanted, lambda index, docs: index.outline_rows(docs))
    rows = [(index.collection, row) for index, found in read for row in found]
    return await cpu.on_cpu(_group, hit_ranges, rows, where.settings.max_section_chars, limit)


async def probe_gaps(
    groups: list[section.Group], questions: list[probe.Question], where: Plan
) -> list[section.Group]:
    """The groups, with one more passage where the questions use words none of them holds (see
    `probe`): the best passage a full-text search of those words finds that no group holds yet,
    tagged with the questions it helps, in its section. Nothing is searched when no word is
    missing."""
    wanted = probe.missing(questions, probe.covered(groups))
    if not wanted:
        return groups
    held = {chunk_key(hit) for one in groups for hit in one.hits}
    depth = probe.PROBE_SCAN + len(held)  # enough rows that the best new one is among them
    lexical = msgspec.structs.replace(where, vector=None)
    found = await fan_out(lexical, " ".join(wanted), depth, vectors=False)  # it only wants chunks
    pool = merge(found, where.settings.rrf_k, depth)
    hits = [hit for hit in scan(pool, depth).hits if chunk_key(hit) not in held]
    fresh = passage.ranges(hits)
    joined = None
    if fresh:
        best = msgspec.structs.replace(fresh[0], aspects=probe.tags(fresh[0], wanted))
        (one,) = await _grouped([best], where, 1)
        placed = probe.placed(groups, one)
        joined, groups = len(placed) == len(groups), placed
    _log.info("search_probe", terms=list(wanted), found=bool(fresh), joined=joined)
    return groups


def _group(
    hit_ranges: list[passage.HitRange], rows: list[tuple[str, dict]], max_chars: int, limit: int
) -> list[section.Group]:
    return section.group(hit_ranges, section.outlines(rows), max_chars, limit)


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


async def shortlist(hits: list[Hit], names: list[str], limit: int, sections: int) -> Sources:
    """Which documents these hits came from, one row per document, and the collections to select
    to read them.

    Its score folds its best chunk with the sum of every chunk it matched (see
    `passage.harmonic`), `sections` says where in it the answer sits, and `collections` names
    which of the searched collections hold it. `Sources.collections` is the cover: the fewest
    collections a follow-up search has to select to reach every row.
    """
    # the shortlist is cut first: only a document that made it is worth a membership and a
    # description, and both are one query for the whole of it
    kept = passage.top_documents(hits, limit)
    docs = {group[0].document for group in kept}
    memberships, described = await asyncio.gather(
        document.memberships(docs, names), document.descriptions_of(docs)
    )
    found = passage.fold_sources(kept, memberships, sections)
    document.fill_descriptions(found.documents, described)
    return found


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
