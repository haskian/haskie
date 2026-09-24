"""What each step of a search does, and the IO it takes to do it.

`flow.py` says in what order the steps run and which search runs which of them; this is what
they call. The pure folds — hits into ranges, ranges into passages, hits into documents — live in
`passage.py`, so what is left here is the IO: reading the collections, checking the models,
reading the markdown a range is widened against.

`scope` is the one place that decides which collections a search covers: the names the caller
gave, else the session's selection, else every collection.
"""

import asyncio
from contextlib import ExitStack
from typing import BinaryIO

import anyio.to_thread
import msgspec

from haskie import cpu
from haskie.collection.collection import Collection
from haskie.collection.index import (
    CollectionIndex,
    Hit,
    RowKey,
    cross_encode,
    first_per_key,
    gather_rows,
    row_key,
    row_score,
)
from haskie.document import document
from haskie.indexing import models
from haskie.indexing.embed import embed_query
from haskie.logs import get_logger
from haskie.search import collapse, passage, session, text
from haskie.search.passage import Passage, Sources
from haskie.settings import (
    EmbeddingModel,
    Reranker,
    SearchMode,
    SearchSettings,
    load_user_settings,
)

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


async def plan(names: list[str], query: str) -> Plan | None:
    """Resolve the settings, embed the query once and check each model once, or None when there
    is nothing left to search.

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

    embedding = user.embedding_model
    vector: list[float] | None = None
    if embedding is not None and any(one.mode != SearchMode.FTS for _, one in plans):
        await models.require_ready(models.ModelKind.EMBEDDING, embedding.name)
        vector = await cpu.on_cpu(embed_query, embedding, query)
    if settings.reranker != Reranker.NONE:
        # before the fan-out
        await models.require_ready(models.ModelKind.RERANKER, settings.reranker_model)
    return Plan(
        settings=settings,
        indexes=[(one.index_with(embedding), where) for one, where in plans],
        vector=vector,
        embedding=embedding,
    )


# --- the rows a search works on ---------------------------------------------------


class Pool(msgspec.Struct):
    """The rows a search read and the ranking over them: what flows from step to step.

    The rows are kept whole rather than turned into `Hit`s as they are read, because the score a
    row answers with is decided by the steps after it.
    """

    rows: dict[RowKey, tuple[CollectionIndex, dict]]
    rankings: dict[str, list[RowKey]]  # one per collection, in that collection's own order
    ranked: list[tuple[RowKey, float]] = []  # merged, best first


async def fan_out(where: Plan, query: str, candidates: int) -> Pool:
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
            return await index.search_rows(query, wanted, settings, candidates)
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
    vectors: list[list[float] | None]


def scan(pool: Pool, limit: int) -> Scanned:
    """The best `limit` of the ranking, as the `Hit`s a caller cites and opens, and their vectors
    for `collapse` to compare them by."""
    taken = [(pool.rows[key], score) for key, score in pool.ranked[:limit]]
    return Scanned(
        hits=[index.hit(row, score) for (index, row), score in taken],
        vectors=[row.get("vector") for (_, row), _ in taken],
    )


# --- what the hits are folded into ------------------------------------------------


def _fold_hits(scanned: Scanned, model: EmbeddingModel | None, limit: int) -> list[Hit]:
    where = collapse.spaces([hit.text for hit in scanned.hits], scanned.vectors, model)
    kept = collapse.hits(scanned.hits, where, limit)
    _log_collapse(where[0].kind, len(scanned.hits), kept, limit)
    return kept


def _fold_ranges(
    scanned: Scanned, model: EmbeddingModel | None, limit: int
) -> list[passage.HitRange]:
    where = collapse.spaces([hit.text for hit in scanned.hits], scanned.vectors, model)
    hit_ranges = passage.ranges(scanned.hits)
    kept = collapse.ranges(hit_ranges, scanned.hits, where, limit)
    _log_collapse(where[0].kind, len(hit_ranges), kept, limit)
    return kept


async def collapse_hits(scanned: Scanned, model: EmbeddingModel | None, limit: int) -> list[Hit]:
    """The `limit` best hits, each with the near-duplicates it stands for (see `collapse`).

    CPU work that grows with the square of the scan - tens of milliseconds at the default depth -
    so it runs in a worker thread rather than on the event loop the search came in on, the
    comparison spaces included.
    """
    return await cpu.on_cpu(_fold_hits, scanned, model, limit)


async def collapse_ranges(
    scanned: Scanned, model: EmbeddingModel | None, limit: int
) -> list[passage.HitRange]:
    """The `limit` best hit ranges, each with the near-duplicates it stands for.

    Folded after `passage.ranges` rather than before: the chunks of one passage sit next to each
    other and read alike, and folding them would split the passage they make up. A worker thread
    runs the fold, as `collapse_hits` says.
    """
    return await cpu.on_cpu(_fold_ranges, scanned, model, limit)


def _log_collapse(
    space: str, candidates: int, kept: list[Hit] | list[passage.HitRange], limit: int
) -> None:
    """One line per search, so how often folding leaves an answer short can be counted."""
    _log.info(
        "search_collapsed",
        space=space,
        candidates=candidates,
        kept=len(kept),
        folded=sum(len(item.also_in) for item in kept),
        short=len(kept) < limit,
    )


async def widen[P: Passage](hit_ranges: list[passage.HitRange], cls: type[P]) -> list[P]:
    """These hit ranges as passages of `cls`, in the order given.

    A passage is what the chunks of one document that sit next to each other say together (see
    `passage.ranges`), widened to the line or the whole sentences around them (`passage.widen`).
    So the reader gets text that begins and ends where the author stopped. The ranges come cut
    and folded (`collapse_ranges`), so only the ones answered with are read.
    """
    windows = await _windows_of(hit_ranges)
    pairs = zip(hit_ranges, windows, strict=True)
    return [passage.widen(hit_range, window, cls) for hit_range, window in pairs]


# Bytes read around a range, per side. `MAX_WIDEN` is a count of characters and a character is at
# most four bytes in UTF-8, so this much always covers what the widening may reach.
WINDOW_BYTES = 4 * passage.MAX_WIDEN


async def _windows_of(hit_ranges: list[passage.HitRange]) -> list[passage.Window]:
    """The markdown around each range, read by seeking to it rather than reading the document.

    One worker thread for the whole search and one open file per document, however many ranges
    each holds. Measured: ten ranges cost 166us in a single hop against 717us fanned out one hop
    per document - a hop costs more than the few kilobytes it would overlap.
    """
    return await anyio.to_thread.run_sync(_read_windows, hit_ranges)


def _read_windows(hit_ranges: list[passage.HitRange]) -> list[passage.Window]:
    """Every range's window, in the order asked for. Sync: the caller runs it in a worker thread,
    where the seeks and reads are ordinary blocking IO."""
    with ExitStack() as stack:
        handles: dict[str, BinaryIO] = {}
        windows: list[passage.Window] = []
        for hit_range in hit_ranges:
            path = hit_range.hits[0].markdown_file
            if path not in handles:
                handles[path] = stack.enter_context(open(path, "rb"))
            windows.append(_read_window(handles[path], hit_range))
        return windows


def _read_window(handle: BinaryIO, hit_range: passage.HitRange) -> passage.Window:
    """One range's surroundings: the range itself and `WINDOW_BYTES` either side of it, as much of
    that as the file holds.

    Decoded in two halves so one pass over the bytes answers both questions: how many characters
    sit before the range (which is where the window starts, in the document's own offsets) and
    what the window says. A seek lands on a byte, so the read may open mid-character - decoding
    drops that half character from the prefix and from the text alike, which is what keeps the
    two consistent.
    """
    start = max(0, hit_range.byte_start - WINDOW_BYTES)
    handle.seek(start)
    raw = handle.read(hit_range.byte_end - start + WINDOW_BYTES)
    before = raw[: hit_range.byte_start - start].decode(errors="ignore")
    text = before + raw[hit_range.byte_start - start :].decode(errors="ignore")
    return passage.Window(text=text, char_start=hit_range.char_start - len(before))


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
