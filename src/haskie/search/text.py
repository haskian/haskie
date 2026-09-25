"""Full-text (BM25) search across every collection at once, one page at a time.

Scores are raw BM25, not fused ranks: one lexical scorer with the same tokenizer and the same
chunk size answers in every collection, so two collections score on one scale and the merge can
sort on the score itself. That is the difference with `flow.chunks`, which has to fuse ranks
because a hybrid ranking has no scale to share. Normalizing per collection would be worse than
either: it would put every collection's rank-1 chunk on page one, whatever it matched.

A chunk counts once. The same document may be a member of several collections, whose tables then
hold the same chunk. The merge keeps the best-scoring copy and drops the rest (see `merge`), so a
page is a page of chunks rather than of memberships.

Paging is an opaque offset bound to the query, not a keyset. A full-text query cannot be filtered
by score, so resuming a walk means recomputing the same ranking and cutting it again; a keyset
cursor would buy nothing and `MAX_DEPTH` is what bounds the cost. Recomputed pages are only as
stable as the indexes underneath them: a document indexed between two pages can move a result
across a page boundary, and creating or deleting a collection changes the query the cursor was
issued for, so the cursor is rejected rather than quietly cutting a different ranking.
"""

import hashlib

import msgspec

from haskie.catalogue import catalogue
from haskie.collection.collection import Collection
from haskie.collection.index import (
    CollectionIndex,
    Hit,
    first_per_key,
    gather_rows,
    row_key,
    row_score,
)
from haskie.errors import InvalidInput, NotFound
from haskie.paging import DEFAULT_PAGE_SIZE, OffsetCursor, Order, Page, check_page_size
from haskie.settings import load_user_settings

MAX_TEXT_PAGE_SIZE = 200  # a page of chunks is a page of text; 200 is already a lot for an agent
MAX_DEPTH = 1000  # every page re-runs the whole ranking, so how deep a walk may go is capped

# A cursor is bound to a sort name and direction (see paging.encode_cursor), so a cursor from a
# listing can never be replayed here. The version lives in `paging`, which owns the wire format.
SORT = "text"
ORDER = Order.DESC
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


async def checked_names(collections: list[str] | None) -> list[str]:
    """The collections a search covers: the names the caller gave, deduplicated and in its own
    order, or every collection when it named none.

    A name nobody owns is a mistake in the request, not an empty result — unlike a session's
    stale name, which `retrieval.plan` skips, because the caller did not choose it just now.
    """
    known = await Collection.names()
    if not collections:
        return known
    names = list(dict.fromkeys(collections))
    owned = set(known)
    unknown = next((name for name in names if name not in owned), None)
    if unknown is not None:
        raise NotFound(f"collection not found: {unknown}")
    return names


def split_collections(raw: str | None) -> list[str] | None:
    """The comma-separated `collections` query argument as names, or None for "every collection".

    An empty value is not a filter that matches nothing: it means the caller did not narrow.
    """
    names = [name.strip() for name in (raw or "").split(",")]
    return [name for name in names if name] or None


def _rank_key(pair: tuple[CollectionIndex, dict]) -> tuple[float, str, int, str]:
    """Best score first, then the identity of the chunk — (document, seq) — and the
    collection last, so two collections holding the same chunk sort next to each other and the
    ranking is the same every time it is recomputed."""
    index, row = pair
    return (-row_score(row), *row_key(row), index.collection)


def merge(
    retrieved: list[tuple[CollectionIndex, list[dict]]],
) -> list[tuple[CollectionIndex, dict]]:
    """One ranking out of the per-collection rankings, with each chunk in it once.

    The identity tie-breaker is what makes paging work: two chunks that score the same must land
    in the same order every time the ranking is recomputed, or a page boundary would swap them and
    the walk would show one twice and the other never.

    A document in two collections puts the same (document, seq) in both their rankings. The
    sort puts those copies next to each other, best score first, so keeping the first of each
    identity (`first_per_key`) keeps the best-scoring copy and drops the rest deterministically.
    """
    pairs = ((index, row) for index, rows in retrieved for row in rows)
    return first_per_key(sorted(pairs, key=_rank_key))


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
    check_page_size(page_size, MAX_TEXT_PAGE_SIZE)
    names = await checked_names(collections)
    chosen = [Collection(name) for name in names]
    offset = parse_cursor(cursor, q, names, page_size)
    depth = offset + page_size
    if depth > MAX_DEPTH:
        raise InvalidInput(f"cannot read past {MAX_DEPTH} results; narrow the query instead")
    if not chosen:
        return Page(items=[], next_cursor=None, total=None)

    # read once, not once per collection
    embedding = await catalogue.embedding_model(await load_user_settings())
    retrieved = await gather_rows(
        [collection.index_with(embedding) for collection in chosen],
        lambda index: index.fts_rows(q, depth),
    )

    merged = merge(retrieved)
    # a collection that returned exactly `depth` rows is holding back rows that may rank into the
    # next page, so it counts as "more" even when the merge alone would look exhausted
    more = len(merged) > depth or any(len(rows) == depth for _, rows in retrieved)
    return Page(
        items=[index.hit(row) for index, row in merged[offset:depth]],
        next_cursor=make_cursor(q, names, page_size, depth) if more and depth < MAX_DEPTH else None,
        total=None,
    )
