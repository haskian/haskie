"""Full-text (BM25) search across every library at once, one page at a time.

Scores are raw BM25, not fused ranks: one lexical scorer with the same tokenizer and the same
chunk size answers in every library, so two libraries score on one scale and the merge can sort on
the score itself. That is the difference with `session.search`, which has to fuse ranks because a
hybrid ranking has no scale to share. Normalizing per library would be worse than either: it would
put every library's rank-1 chunk on page one, whatever it matched.

Paging is an opaque offset bound to the query, not a keyset. A full-text query cannot be filtered
by score, so resuming a walk means recomputing the same ranking and cutting it again; a keyset
cursor would buy nothing and `MAX_DEPTH` is what bounds the cost. Recomputed pages are only as
stable as the indexes underneath them: a document indexed between two pages can move a result
across a page boundary, and creating or deleting a library changes the query the cursor was issued
for, so the cursor is rejected rather than quietly cutting a different ranking.
"""

import asyncio
import hashlib

import anyio
import msgspec

from haskie.errors import InvalidInput, LibraryNotFound
from haskie.index import SEARCH_CONCURRENCY, Hit, LibraryIndex, row_score
from haskie.library import Library
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


def query_hash(q: str, libraries: list[str], page_size: int) -> str:
    """Identity of one result set: the query, the libraries it spans, and the size it is cut into.

    The material is JSON rather than a delimiter-joined string, because a query that contains the
    delimiter would otherwise hash like a different (query, libraries) pair. Not a secret: this
    tells a cursor apart from another cursor, it does not authenticate one.
    """
    material = msgspec.json.encode([q, sorted(libraries), page_size])
    return hashlib.sha256(material).hexdigest()[:16]


def make_cursor(q: str, libraries: list[str], page_size: int, offset: int) -> str:
    """The cursor for the page starting at `offset` of this query."""
    return CURSOR.encode(query_hash(q, libraries, page_size), offset)


def parse_cursor(cursor: str | None, q: str, libraries: list[str], page_size: int) -> int:
    """The offset inside `cursor`, or 0 when there is none.

    Anything this query did not issue is invalid input: a cursor of another query, of another page
    size, or of another listing would cut a ranking it was never measured against.
    """
    if cursor is None:
        return 0
    digest, offset = CURSOR.decode(cursor)
    if digest != query_hash(q, libraries, page_size):
        raise InvalidInput("cursor was issued for another query")
    return offset


def split_libraries(raw: str | None) -> list[str] | None:
    """The comma-separated `libraries` query argument as names, or None for "every library".

    An empty value is not a filter that matches nothing: it means the caller did not narrow.
    """
    names = [name.strip() for name in (raw or "").split(",")]
    return [name for name in names if name] or None


def _rank_key(pair: tuple[LibraryIndex, dict]) -> tuple[float, str, str, int, int]:
    """Best score first, then the identity of the chunk: (library, doc, part, chunk_id)."""
    index, row = pair
    return (-row_score(row), index.library, row["doc"], row.get("part", 0), row["chunk_id"])


def merge(per_library: list[list[tuple[LibraryIndex, dict]]]) -> list[tuple[LibraryIndex, dict]]:
    """One ranking out of the per-library rankings.

    The identity tie-breaker is what makes paging work: two chunks that score the same must land
    in the same order every time the ranking is recomputed, or a page boundary would swap them and
    the walk would show one twice and the other never.
    """
    return sorted((pair for rows in per_library for pair in rows), key=_rank_key)


async def search(
    q: str,
    libraries: list[str] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> Page[Hit]:
    """One page of the merged full-text ranking over `libraries`, or over every library.

    `total` is None: counting the whole ranking costs the same as producing it, for a number no
    caller pages to. Searches are not audited, like every other search.
    """
    if not 1 <= page_size <= MAX_TEXT_PAGE_SIZE:
        raise InvalidInput(f"page_size must be 1..{MAX_TEXT_PAGE_SIZE}, got {page_size}")
    known = await Library.names()
    names = list(dict.fromkeys(libraries)) if libraries else known
    # an unknown name is a mistake in the request, not an empty page (as when a session picks one)
    unknown = next((name for name in names if name not in set(known)), None)
    if unknown is not None:
        raise LibraryNotFound(f"library not found: {unknown}")
    chosen = [Library(name) for name in names]
    offset = parse_cursor(cursor, q, names, page_size)
    depth = offset + page_size
    if depth > MAX_DEPTH:
        raise InvalidInput(f"cannot read past {MAX_DEPTH} results; narrow the query instead")
    if not chosen:
        return Page(items=[], next_cursor=None, total=None)

    embedding = (await load_user_settings()).embedding_model  # read once, not once per library
    # One semaphore per call, never at import time: an anyio primitive belongs to the loop that
    # first used it, and both the Litestar loop and the DBOS loop run searches (see the plan).
    slots = anyio.Semaphore(SEARCH_CONCURRENCY)

    async def retrieve(library: Library) -> list[tuple[LibraryIndex, dict]]:
        index = library.index_with(embedding)
        async with slots:
            return [(index, row) for row in await index.fts_rows(q, depth)]

    # no `return_exceptions`: the first library that cannot answer fails the whole page
    retrieved = await asyncio.gather(*(retrieve(library) for library in chosen))

    merged = merge(retrieved)
    # a library that returned exactly `depth` rows is holding back rows that may rank into the
    # next page, so it counts as "more" even when the merge alone would look exhausted
    more = len(merged) > depth or any(len(rows) == depth for rows in retrieved)
    by_name = {library.name: library for library in chosen}
    return Page(
        # the index knows the row, the library knows where its files are today (Library.resolve_hit)
        items=[
            by_name[index.library].resolve_hit(index.hit(row))
            for index, row in merged[offset:depth]
        ],
        next_cursor=make_cursor(q, names, page_size, depth) if more and depth < MAX_DEPTH else None,
        total=None,
    )


class DocumentMatch(msgspec.Struct):
    """One document the query matched, and the best evidence that it did.

    The answer to "which documents should I read", not "which passages answer this": `score` is
    the document's best chunk, and `chunks` is how many of the scanned chunks came from it, so a
    document that matches all over ranks above one that matches once as well as it does.
    """

    library: str
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
    q: str, libraries: list[str] | None = None, limit: int = 10
) -> list[DocumentMatch]:
    """The distinct documents a full-text query matches, best first.

    The same BM25 scan as `search`, folded to one row per document. Scores are raw BM25 and
    therefore comparable across libraries, for the reason in the module docstring.
    """
    if not 1 <= limit <= MAX_DOCUMENTS:
        raise InvalidInput(f"limit must be 1..{MAX_DOCUMENTS}, got {limit}")
    page = await search(q, libraries, page_size=min(limit * DOCUMENT_SCAN, MAX_TEXT_PAGE_SIZE))

    best: dict[tuple[str, str], DocumentMatch] = {}
    for hit in page.items:
        key = (hit.library, hit.doc)
        found = best.get(key)
        if found is None:
            best[key] = DocumentMatch(
                library=hit.library,
                doc=hit.doc,
                score=hit.score,
                chunks=1,
                description="",
                heading=hit.heading,
                location=hit.location,
                text=hit.text,
                source_file=hit.source_file,
                markdown_file=hit.markdown_file,
                line_start=hit.line_start,
                line_end=hit.line_end,
            )
        else:
            found.chunks += 1  # the page is already ranked, so the first hit seen is the best one

    ranked = sorted(best.values(), key=lambda m: (-m.score, m.library, m.doc))[:limit]
    await _attach_descriptions(ranked)
    return ranked


async def _attach_descriptions(matches: list[DocumentMatch]) -> None:
    """Fill in each match's description, one query per library rather than one per document."""
    by_library: dict[str, list[DocumentMatch]] = {}
    for match in matches:
        by_library.setdefault(match.library, []).append(match)
    for library, group in by_library.items():
        described = await Library(library).describe_of({m.doc for m in group})
        for match in group:
            match.description = described.get(match.doc, "")
