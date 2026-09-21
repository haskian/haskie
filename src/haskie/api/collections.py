"""Collection routes: the listing, one collection, its settings, its members and its search.

A collection holds documents it does not own: attaching and detaching move a membership and the
rows of this collection's index, never the document itself (see `collection.py`).
"""

import time

import msgspec
from litestar import delete, get, post, put

from haskie import audit, logs, models, session, workflows
from haskie.api.common import BulkStarted, Describe, Limit
from haskie.collection import (
    Collection,
    CollectionInfo,
    CollectionSummary,
    Member,
    MemberStatus,
)
from haskie.index import Hit
from haskie.paging import DEFAULT_PAGE_SIZE, Order, Page, page_request
from haskie.settings import (
    CollectionSettings,
    Fusion,
    Reranker,
    SearchMode,
    SearchOverrides,
    load_user_settings,
    without_none,
)


class CreateCollection(msgspec.Struct):
    name: str
    description: str = ""


class AddDocument(msgspec.Struct):
    """Which already-imported document to attach."""

    document: str


@get("/api/collections", mcp_tool="list_collections")
async def list_collections(
    cursor: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    sort: str | None = None,
    order: Order = "asc",
) -> Page[CollectionSummary]:
    """List collections with their member counts, one page at a time.

    Sort by name or created_at, ascending or descending. Pass the `next_cursor` of a response
    back as `cursor` to continue; it is null on the last page.
    """
    return await Collection.page(page_request(cursor, page_size, sort, order))


@post("/api/collections")
@audit.audited("collection.create")
async def create_collection(data: CreateCollection) -> CollectionInfo:
    found = await Collection.create(data.name, data.description)
    audit.attach(collection=found.name)
    logs.bind(collection=found.name)
    return await found.info()


@get("/api/collections/{collection:str}", mcp_tool="get_collection")
async def get_collection(collection: str) -> CollectionInfo:
    """Describe one collection: its settings and how many documents it holds per status.

    The documents themselves are listed by `list_collection_documents`.
    """
    return await (await Collection.get(collection)).info()


@delete("/api/collections/{collection:str}", status_code=202)
@audit.audited("collection.delete")
async def delete_collection(collection: str) -> BulkStarted:
    """Queue the deletion: every index still running for this collection is cancelled first.

    Accepted, not done: cancelling a busy collection and removing its index takes as long as the
    last running step, which is no time to hold a request open. Poll the job for the outcome.
    The documents survive; only this collection's memberships and index go.
    """
    job_id = await workflows.start_delete_collection(collection)
    audit.attach(job_id=job_id)
    return BulkStarted(job_id=job_id)


@get("/api/collections/{collection:str}/search", mcp_tool="search_collection")
async def search_collection(
    collection: str,
    q: str,
    limit: Limit = None,
    mode: SearchMode | None = None,
    fusion: Fusion | None = None,
    vector_weight: float | None = None,
    bm25_weight: float | None = None,
    reranker: Reranker | None = None,
    candidates: int | None = None,
    session_id: str | None = None,
) -> list[Hit]:
    """Search one collection. Options default to the collection's search settings:
    mode hybrid|vector|fts, fusion rrf|linear, vector_weight/bm25_weight for linear,
    reranker none|cross-encoder (rescoring of `candidates` for any mode).

    Args:
        session_id: The conversation's id; the search then shows in that session's history.
    """
    overrides = without_none(
        SearchOverrides(
            mode=mode,
            fusion=fusion,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            reranker=reranker,
            candidates=candidates,
        )
    )
    found = await Collection.get(collection)
    started = time.perf_counter()
    found_hits = await found.search(q, limit, **overrides)
    await session.record_search(session_id, collection, q, found_hits, started)
    return found_hits


@put("/api/collections/{collection:str}/settings")
@audit.audited("collection.settings.update")
async def put_collection_settings(collection: str, data: CollectionSettings) -> CollectionInfo:
    """Saves, then downloads: a collection may override the reranker model, and a search of this
    collection would otherwise fail with "not loaded yet" for a model nothing ever fetched."""
    found = await Collection.get(collection)
    await found.set_settings(data)
    await models.ensure_models(await load_user_settings())
    return await found.info()


@put("/api/collections/{collection:str}/description")
@audit.audited("collection.describe")
async def describe_collection(collection: str, data: Describe) -> CollectionInfo:
    """Replace what the collection is said to hold. Empty clears it."""
    found = await Collection.get(collection)
    await found.describe(data.description)
    return await found.info()


@post("/api/collections/{collection:str}/index", status_code=202)
@audit.audited("collection.reindex")
async def index_collection(collection: str) -> BulkStarted:
    """(Re)index every member of the collection in the background.

    Accepted, not done: the members are queued by a background job, a page at a time, so a
    collection of ten thousand costs this request one insert. Poll the job for its progress.
    """
    job_id = await workflows.start_index_collection(collection)
    audit.attach(job_id=job_id)
    return BulkStarted(job_id=job_id)


@get("/api/collections/{collection:str}/documents", mcp_tool="list_collection_documents")
async def list_collection_documents(
    collection: str,
    cursor: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    sort: str | None = None,
    order: Order = "asc",
    status: MemberStatus | None = None,
) -> Page[Member]:
    """List the documents of one collection, one page at a time.

    Sort by name, size, status or updated_at; `status` keeps one membership state only (pending,
    indexing, indexed, error, cancelled) — how far this collection got writing the document into
    its index, which is not the document's own import status. Pass the `next_cursor` of a
    response back as `cursor` to continue; it is null on the last page.
    """
    found = await Collection.get(collection)
    return await found.members_page(page_request(cursor, page_size, sort, order), status)


@post(
    "/api/collections/{collection:str}/documents",
    status_code=202,
    mcp_tool="add_document_to_collection",
)
@audit.audited("collection.attach")
async def add_document(
    collection: str, data: AddDocument, session_id: str | None = None
) -> BulkStarted:
    """Attach an imported document to this collection and queue its index.

    Accepted, not done: the document is chunked and embedded once per distinct chunk settings and
    reused from its cache, but the first collection to ask still pays for it. Poll the job.

    Args:
        session_id: The conversation's id; the attach and its job then show in that session.
    """
    audit.attach(doc=data.document)
    logs.bind(doc=data.document)
    job_id = await workflows.attach(collection, data.document)
    audit.attach(job_id=job_id)
    await session.record(
        session_id,
        "attach",
        data.document,
        detail={"collection": collection},
        workflow_id=job_id,
    )
    return BulkStarted(job_id=job_id)


@delete(
    "/api/collections/{collection:str}/documents/{doc:str}",
    mcp_tool="remove_document_from_collection",
)
@audit.audited("collection.detach")
async def remove_document(collection: str, doc: str, session_id: str | None = None) -> None:
    """Take one document out of this collection: its rows here go, the document stays.

    Waited out rather than queued: a detach cancels one index and deletes that collection's rows,
    which is short enough to answer with the outcome instead of a job to poll.

    Args:
        session_id: The conversation's id; the detach then shows in that session's history.
    """
    await workflows.detach(collection, doc)
    await session.record(session_id, "detach", doc, detail={"collection": collection})


@post("/api/collections/{collection:str}/documents/{doc:str}/index", status_code=202)
@audit.audited("collection.reindex")
async def index_collection_document(collection: str, doc: str) -> BulkStarted:
    """(Re)index one member: chunk and embed it if the cache misses, then write it into this
    collection's index. Poll the job for the outcome."""
    job_id = await workflows.start_index_collection_document(collection, doc)
    audit.attach(job_id=job_id)
    return BulkStarted(job_id=job_id)
