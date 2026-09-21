"""Full-text (BM25) search across every collection at once, one page at a time.

Scores are raw BM25, not fused ranks: one lexical scorer with the same tokenizer and the same
chunk size answers in every collection, so two collections score on one scale and the merge can
sort on the score itself. That is the difference with `session.search`, which has to fuse ranks
because a hybrid ranking has no scale to share. Normalizing per collection would be worse than
either: it would put every collection's rank-1 chunk on page one, whatever it matched.

A passage counts once. The same document may be a member of several collections, whose tables then
hold the same chunk; the merge keeps the best-scoring copy and drops the rest (see `merge`), so a
page is a page of passages rather than of memberships.

Paging is an opaque offset bound to the query, not a keyset. A full-text query cannot be filtered
by score, so resuming a walk means recomputing the same ranking and cutting it again; a keyset
cursor would buy nothing and `MAX_DEPTH` is what bounds the cost. Recomputed pages are only as
stable as the indexes underneath them: a document indexed between two pages can move a result
across a page boundary, and creating or deleting a collection changes the query the cursor was
issued for, so the cursor is rejected rather than quietly cutting a different ranking.
"""

import asyncio
import hashlib

import anyio
import msgspec

from haskie import document
from haskie.collection import Collection, resolve_hit
from haskie.errors import CollectionNotFound, InvalidInput
from haskie.index import SEARCH_CONCURRENCY, CollectionIndex, Hit, RowKey, row_key, row_score
from haskie.paging import DEFAULT_PAGE_SIZE, OffsetCursor, Order, Page
from haskie.settings import load_user_settings

MAX_TEXT_PAGE_SIZE = 200  # a page of chunks is a page of text; 200 is already a lot for an agent
MAX_DOCUMENTS = 100  # a shortlist nobody reads past; `search` is there for the chunks themselves
# Chunks scanned per document asked for. A document can hold many matching chunks, so the scan has
# to go deeper than the answer or the tail of the shortlist would be whichever documents happened
# to crowd the top with chunks.
DOCUMENT_SCAN = 20
MAX_DEPTH = 1000  # every page re-runs the whole ranking, so how deep a walk may go is capped

# A cursor is bound to a sort name and direction (see paging.encode_cursor), so a cursor from a
# listing can never be replayed here. The version lives in `paging`, which owns the wire format.
SORT = "text"
ORDER: Order = "desc"
CURSOR = OffsetCursor(SORT, ORDER)


def query_hash(q: str, collections: list[str], page_size: int) -> str:
    """Identity of one result set: the query, the collections it spans, and the size it is cut
    into.

    The material is JSON rather than a delimiter-joined string, because a query that contains the
    delimiter would otherwise hash like a different (query, collections) pair. Not a secret: this
    tells a cursor apart from another cursor, it does not authenticate one.
    """
    material = msgspec.json.encode([q, sorted(collections), page_size])
    return hashlib.sha256(material).hexdigest()[:16]


def make_cursor(q: str, collections: list[str], page_size: int, offset: int) -> str:
    """The cursor for the page starting at `offset` of this query."""
    return CURSOR.encode(query_hash(q, collections, page_size), offset)


def parse_cursor(cursor: str | None, q: str, collections: list[str], page_size: int) -> int:
    """The offset inside `cursor`, or 0 when there is none.

    Anything this query did not issue is invalid input: a cursor of another query, of another page
    size, or of another listing would cut a ranking it was never measured against.
    """
    if cursor is None:
        return 0
    digest, offset = CURSOR.decode(cursor)
    if digest != query_hash(q, collections, page_size):
        raise InvalidInput("cursor was issued for another query")
    return offset


def split_collections(raw: str | None) -> list[str] | None:
    """The comma-separated `collections` query argument as names, or None for "every collection".

    An empty value is not a filter that matches nothing: it means the caller did not narrow.
    """
    names = [name.strip() for name in (raw or "").split(",")]
    return [name for name in names if name] or None


def _rank_key(pair: tuple[CollectionIndex, dict]) -> tuple[float, str, int, int, str]:
    """Best score first, then the identity of the passage — (doc, part, chunk_id) — and the
    collection last, so two collections holding the same chunk sort next to each other and the
    ranking is the same every time it is recomputed."""
    index, row = pair
    return (-row_score(row), *row_key(row), index.collection)


def merge(
    per_collection: list[list[tuple[CollectionIndex, dict]]],
) -> list[tuple[CollectionIndex, dict]]:
    """One ranking out of the per-collection rankings, with each passage in it once.

    The identity tie-breaker is what makes paging work: two chunks that score the same must land
    in the same order every time the ranking is recomputed, or a page boundary would swap them and
    the walk would show one twice and the other never.

    A document in two collections puts the same (doc, part, chunk_id) in both their rankings. The
    sort above puts those copies next to each other, best score first, so keeping the first of each
    identity keeps the best-scoring copy and drops the duplicates deterministically.
    """
    ranked = sorted((pair for rows in per_collection for pair in rows), key=_rank_key)
    seen: set[RowKey] = set()
    unique: list[tuple[CollectionIndex, dict]] = []
    for index, row in ranked:
        identity = row_key(row)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append((index, row))
    return unique


async def search(
    q: str,
    collections: list[str] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> Page[Hit]:
    """One page of the merged full-text ranking over `collections`, or over every collection.

    `total` is None: counting the whole ranking costs the same as producing it, for a number no
    caller pages to. Searches are not audited, like every other search.
    """
    if not 1 <= page_size <= MAX_TEXT_PAGE_SIZE:
        raise InvalidInput(f"page_size must be 1..{MAX_TEXT_PAGE_SIZE}, got {page_size}")
    known = await Collection.names()
    names = list(dict.fromkeys(collections)) if collections else known
    # an unknown name is a mistake in the request, not an empty page (as when a session picks one)
    unknown = next((name for name in names if name not in set(known)), None)
    if unknown is not None:
        raise CollectionNotFound(f"collection not found: {unknown}")
    chosen = [Collection(name) for name in names]
    offset = parse_cursor(cursor, q, names, page_size)
    depth = offset + page_size
    if depth > MAX_DEPTH:
        raise InvalidInput(f"cannot read past {MAX_DEPTH} results; narrow the query instead")
    if not chosen:
        return Page(items=[], next_cursor=None, total=None)

    embedding = (await load_user_settings()).embedding_model  # read once, not once per collection
    # One semaphore per call, never at import time: an anyio primitive belongs to the loop that
    # first used it, and both the Litestar loop and the DBOS loop run searches (see the plan).
    slots = anyio.Semaphore(SEARCH_CONCURRENCY)

    async def retrieve(collection: Collection) -> list[tuple[CollectionIndex, dict]]:
        index = collection.index_with(embedding)
        async with slots:
            return [(index, row) for row in await index.fts_rows(q, depth)]

    # no `return_exceptions`: the first collection that cannot answer fails the whole page
    retrieved = await asyncio.gather(*(retrieve(collection) for collection in chosen))

    merged = merge(retrieved)
    # a collection that returned exactly `depth` rows is holding back rows that may rank into the
    # next page, so it counts as "more" even when the merge alone would look exhausted
    more = len(merged) > depth or any(len(rows) == depth for rows in retrieved)
    return Page(
        # the index knows the row, `collection.resolve_hit` knows where the files are
        items=[resolve_hit(index.hit(row)) for index, row in merged[offset:depth]],
        next_cursor=make_cursor(q, names, page_size, depth) if more and depth < MAX_DEPTH else None,
        total=None,
    )


class DocumentMatch(msgspec.Struct):
    """One document the query matched, and the best evidence that it did.

    The answer to "which documents should I read", not "which passages answer this": `score` is
    the harmonic mean of the document's best chunk and the sum of every scanned chunk that came
    from it (see `_document_score`), and `chunks` how many there were. The evidence fields are its
    best chunk.
    """

    collection: str  # the collection whose table held the best chunk; the document belongs to none
    doc: str
    score: float
    chunks: int
    description: str
    heading: str
    location: str
    text: str  # the best chunk, so a caller can see why the document is on the list
    # Where the document is on disk, so a tool outside the app can open or grep it. The lines are
    # the best chunk's, in `markdown_file`: somewhere to start reading, not the whole match.
    source_file: str
    markdown_file: str
    line_start: int
    line_end: int


async def search_documents(
    q: str, collections: list[str] | None = None, limit: int = 10
) -> list[DocumentMatch]:
    """The distinct documents a full-text query matches, best first.

    The same BM25 scan as `search`, folded to one row per document — by document name alone, not
    by (collection, document): a document in two collections is one document to read, and the
    passages behind it were already deduplicated by `merge`. `collection` names where its best
    chunk came from. Scores are raw BM25 and therefore comparable, for the reason in the module
    docstring.
    """
    page = await search(q, collections, page_size=_scan_size(limit))
    by_doc: dict[str, list[Hit]] = {}
    for hit in page.items:
        by_doc.setdefault(hit.doc, []).append(hit)
    ranked = sorted(map(_document_match, by_doc.values()), key=lambda m: (-m.score, m.doc))
    ranked = ranked[:limit]
    await _attach_descriptions(ranked)
    return ranked


def _document_match(hits: list[Hit]) -> DocumentMatch:
    """One document's row from its matched chunks, in the order the page ranked them: the first
    hit is its best one, and the passage the row shows."""
    best = hits[0]
    return DocumentMatch(
        collection=best.collection,
        doc=best.doc,
        score=_document_score(best.score, sum(hit.score for hit in hits)),
        chunks=len(hits),
        description="",
        heading=best.heading,
        location=best.location,
        text=best.text,
        source_file=best.source_file,
        markdown_file=best.markdown_file,
        line_start=best.line_start,
        line_end=best.line_end,
    )


def _document_score(best: float, total: float) -> float:
    """How strongly a document matches: the harmonic mean of its best chunk and the sum of all
    its matched chunks. A document matched once scores its one chunk. Every further chunk lifts
    it, but the mean stays under twice the best, so many weak chunks never outrank one strong one,
    and a document with a few strong chunks is not held back for having few."""
    if best <= 0 or total <= 0:
        return 0.0
    return 2 * best * total / (best + total)


def _scan_size(limit: int) -> int:
    """How many chunks the shortlist of `limit` documents is folded from."""
    if not 1 <= limit <= MAX_DOCUMENTS:
        raise InvalidInput(f"limit must be 1..{MAX_DOCUMENTS}, got {limit}")
    return min(limit * DOCUMENT_SCAN, MAX_TEXT_PAGE_SIZE)


async def document_passages(
    q: str, doc: str, collections: list[str] | None = None, limit: int = 10
) -> list[Hit]:
    """The passages of one document behind its row in the shortlist, best first.

    The same scan `search_documents` folds, for the same `limit`, kept to `doc`: exactly the
    `chunks` its row counted, unfolded. A document the scan never reached is an empty list.
    """
    page = await search(q, collections, page_size=_scan_size(limit))
    return [hit for hit in page.items if hit.doc == doc]


async def _attach_descriptions(matches: list[DocumentMatch]) -> None:
    """Fill in each match's description. One query for the whole shortlist: a description belongs
    to the document, so there is nothing to group by collection."""
    described = await document.describe_of({match.doc for match in matches})
    for match in matches:
        match.description = described.get(match.doc, "")
