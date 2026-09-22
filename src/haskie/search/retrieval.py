"""Retrieval over a set of collections, in the three shapes a caller asks for.

One core and three answers: `chunks` is the hybrid fan-out every search runs, `passages` widens
the chunks it returned into readable text, `excerpts` is what an agent quotes, and `sources` folds
the same hits to one row per document. Everything pure — merging chunks, widening them, folding
them, covering them with collections — lives in `passage.py`; the IO lives here.

`scope` is the one place that decides which collections a search covers: the names the caller
gave, else the session's selection, else every collection.
"""

import asyncio

import anyio
import msgspec

from haskie import cpu
from haskie.collection.collection import Collection
from haskie.collection.index import (
    CollectionIndex,
    Hit,
    RowKey,
    first_per_key,
    gather_rows,
    row_key,
)
from haskie.document import document
from haskie.indexing import models
from haskie.indexing.embed import embed_query, rerank_scores
from haskie.logs import get_logger
from haskie.paging import check_page_size
from haskie.search import passage, session, text
from haskie.search.passage import Excerpt, Passage, Sources
from haskie.settings import load_user_settings

# How deep any of these searches reads. A passage or a document row is folded from several chunks,
# so the scan goes deeper than the answer; this is where that stops.
MAX_SCAN = 200
PASSAGE_SCAN = 4  # chunks scanned per passage asked for: consecutive ones merge into one passage
DEFAULT_SECTIONS = 3  # hot sections per document: where in it the answer is, not an outline
MAX_SECTIONS = 20

_log = get_logger(__name__)


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


async def chunks(names: list[str], query: str, limit: int | None = None) -> list[Hit]:
    """Search every collection in `names` and merge the results into one ranking.

    The query is embedded once and each model is checked once for the whole fan-out, the
    collections are then read concurrently, and the per-collection rankings are fused by rank (see
    `rrf_merge`): the merged `Hit.score` is an RRF score, or the cross-encoder's when a reranker is
    on. A single collection keeps its own scores, because there is nothing to compare them with.

    A passage counts once. The same document may be a member of several of the chosen collections,
    and each of their tables then holds the same chunk; a caller searching them wants one hit per
    passage, not one per collection that happens to hold it. So a chunk enters the fusion from the
    first collection in `names` that returned it, and the later collections' copies are dropped
    before the ranks are counted — otherwise a document in two collections would be fused with
    itself and outrank an equally good one that sits in a single collection.

    A collection deleted since the caller chose it is skipped, so one stale name does not break
    every search. A collection that fails to answer is not: a silent hole in the results would be
    read as "no match".
    """
    user = await load_user_settings()
    base = user.search
    limit = limit or base.limit
    found = await Collection.load_settings(names)
    for name in names:
        if name not in found:
            _log.warning("session_collection_missing", collection=name)
    plans = [
        (Collection(name), found[name].resolve_search(user)) for name in names if name in found
    ]
    if not plans:
        return []
    if len(plans) == 1:
        collection, settings = plans[0]
        index = collection.index_with(user.embedding_model)
        return await index.search(query, msgspec.structs.replace(settings, limit=limit))

    embedding = user.embedding_model
    vector: list[float] | None = None
    if embedding is not None and any(settings.mode != "fts" for _, settings in plans):
        await models.require_ready("embedding", embedding.name)
        vector = await cpu.on_cpu(embed_query, embedding, query)
    if base.reranker != "none":
        await models.require_ready("reranker", base.reranker_model)  # fail before the fan-out
    candidates = max(base.candidates, limit)
    chosen = {collection.name: settings for collection, settings in plans}

    async def retrieve(index: CollectionIndex) -> list[dict]:
        settings = chosen[index.collection]
        wanted = None if settings.mode == "fts" else vector
        try:
            return await index.search_rows(query, wanted, settings, candidates)
        except Exception:
            _log.exception("session_collection_search_failed", collection=index.collection)
            raise

    retrieved = await gather_rows(
        [collection.index_with(embedding) for collection, _ in plans], retrieve
    )

    # `retrieved` is in the order of `plans`, which is the caller's own order, so the first
    # collection that holds a passage is the one it is credited to.
    rows: dict[RowKey, tuple[CollectionIndex, dict]] = {}
    rankings: dict[str, list[RowKey]] = {index.collection: [] for index, _ in retrieved}
    for index, row in first_per_key((i, r) for i, found_rows in retrieved for r in found_rows):
        key = row_key(row)
        rows[key] = (index, row)
        rankings[index.collection].append(key)
    merged = rrf_merge(list(rankings.values()), base.rrf_k)[:candidates]
    if base.reranker != "none":
        scores = await cpu.on_cpu(
            rerank_scores, base.reranker_model, query, [rows[key][1]["text"] for key, _ in merged]
        )
        merged = sorted(zip([key for key, _ in merged], scores, strict=True), key=_by_score)
    return [rows[key][0].hit(rows[key][1], score) for key, score in merged[:limit]]


def _by_score(scored: tuple[RowKey, float]) -> float:
    """Sort key for the merged ranking: best first, so the score is negated rather than the list
    reversed (reversing would also flip the stable tie order)."""
    return -scored[1]


async def scope(session_id: str | None, collections: str | None) -> list[str]:
    """Which collections a search covers: the comma-separated `collections` if the caller named
    any, else the session's selection if it has one, else every collection.

    A name nobody owns is a mistake in the request, not an empty result — unlike a session's
    stale name, which `chunks` skips, because the caller did not choose it just now.
    """
    named = text.split_collections(collections)
    if named:
        return await text.checked_names(named)
    selected = await session.collections_for(session_id) if session_id else []
    return selected or await text.checked_names(None)


async def _markdown_of(spans: list[passage.ChunkRange]) -> dict[str, tuple[str, list[int]]]:
    """The whole markdown behind each span with the offsets of its newlines, keyed by the file it
    was read from and read once per document however many spans came out of it. Concurrent: the
    reads are independent, and a passage cannot be widened before its document is in hand."""
    files = list(dict.fromkeys(one.chunks[0].markdown_file for one in spans))
    read = await asyncio.gather(*(anyio.Path(file).read_text(encoding="utf-8") for file in files))
    return {
        file: (markdown, passage.newline_offsets(markdown))
        for file, markdown in zip(files, read, strict=True)
    }


async def _expanded[P: Passage](
    names: list[str], query: str, limit: int | None, cls: type[P]
) -> list[P]:
    """The `limit` best passages of `names`, as `cls`.

    A passage is what the chunks of one document that sit next to each other say together, widened
    to whole sentences (see `passage.expand`): the reader gets text that begins and ends where the
    author did, and never the overlap between two chunks twice. The scan goes `PASSAGE_SCAN` times
    deeper than `limit`, because consecutive chunks fold into one passage.
    """
    limit = limit or (await load_user_settings()).search.limit
    hits = await chunks(names, query, min(limit * PASSAGE_SCAN, MAX_SCAN))
    # cut before the markdown is read: the spans are already best first, and widening one keeps
    # the score it was ranked by
    spans = passage.ranges(hits)[:limit]
    read = await _markdown_of(spans)
    found: list[P] = []
    for span in spans:
        markdown, newlines = read[span.chunks[0].markdown_file]
        found.append(passage.expand(span, markdown, newlines, cls))
    return found


async def passages(names: list[str], query: str, limit: int | None = None) -> list[Passage]:
    """The passages of `names` that answer the query, best first."""
    return await _expanded(names, query, limit, Passage)


async def excerpts(names: list[str], query: str, limit: int | None = None) -> list[Excerpt]:
    """The passages of `names` that answer the query, as an agent quotes them, best first.

    An excerpt is the whole passage today. Cutting the parts of it that do not answer the query
    is a later step, and this is the one place it goes.
    """
    return await _expanded(names, query, limit, Excerpt)


async def sources(
    names: list[str], query: str, limit: int | None = None, sections: int | None = None
) -> Sources:
    """Which documents of `names` answer the query, and the smallest set of collections holding
    them.

    One row per document rather than per passage: its score folds its best chunk with the sum of
    every chunk it matched (see `passage.harmonic`), `sections` says where in it the answer sits,
    and `collections` names which of the searched collections hold it. `Sources.collections` is
    the cover: the fewest collections a follow-up search has to select to reach every row.
    """
    limit = text.document_limit(limit)
    wanted = check_page_size(
        DEFAULT_SECTIONS if sections is None else sections, MAX_SECTIONS, "sections"
    )
    hits = await chunks(names, query, text.scan_size(limit))
    # the shortlist is cut first: only a document that made it is worth a membership and a
    # description, and both are one query for the whole of it
    kept = passage.top_documents(hits, limit)
    docs = {group[0].doc for group in kept}
    memberships, described = await asyncio.gather(
        document.memberships(docs, names), document.describe_of(docs)
    )
    found = passage.fold_sources(kept, memberships, wanted)
    document.fill_descriptions(found.documents, described)
    return found
