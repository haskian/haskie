"""Collection routes: the listing, one collection, its settings and its members.

A collection holds documents it does not own: attaching and detaching move a membership and the
rows of this collection's index, never the document itself (see `collection/collection.py`).
"""

from typing import Annotated

import msgspec
from litestar import delete, get, post, put

from haskie import audit, logs
from haskie.api.common import PAGED, BulkStarted, Describe, checked_session
from haskie.catalogue import catalogue
from haskie.collection.collection import (
    Collection,
    CollectionInfo,
    CollectionSummary,
    Member,
    MemberStatus,
)
from haskie.indexing import models, workflows
from haskie.paging import Page, PageRequest, one_of
from haskie.search import session
from haskie.settings import (
    CollectionOverrides,
    load_user_settings,
)


class CreateCollection(msgspec.Struct):
    name: str
    description: str = ""


class Rename(msgspec.Struct):
    name: str


class AddDocument(msgspec.Struct):
    """Which already-imported document to attach."""

    document: str


@get("/api/collections", mcp_tool="list_collections", dependencies=PAGED)
async def list_collections(page: PageRequest) -> Page[CollectionSummary]:
    """List collections with their member counts, one page at a time.

    Sort by name or created_at, ascending or descending. Pass the `next_cursor` of a response
    back as `cursor` to continue; it is null on the last page.
    """
    return await Collection.page(page)


@post("/api/collections")
@audit.audited("collection.create")
async def create_collection(data: CreateCollection) -> CollectionInfo:
    found = await workflows.create_collection(data.name, data.description)
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
    last running step, which is no time to hold a request open. Poll the operation for the outcome.
    The documents survive; only this collection's memberships and index go.
    """
    operation_id = await workflows.start_delete_collection(collection)
    audit.attach(operation_id=operation_id)
    return BulkStarted(operation_id=operation_id)


@put("/api/collections/{collection:str}/overrides")
@audit.audited("collection.overrides.update")
async def put_collection_overrides(collection: str, data: CollectionOverrides) -> CollectionInfo:
    """Saves, then downloads: a collection may override the reranker model, and a search of this
    collection would otherwise fail with "not loaded yet" for a model nothing ever fetched."""
    await catalogue.check(data)
    found = await Collection.get(collection)
    await found.set_overrides(data)
    await models.ensure_models(await load_user_settings())
    return await found.info()


@put("/api/collections/{collection:str}/description")
@audit.audited("collection.describe")
async def describe_collection(collection: str, data: Describe) -> CollectionInfo:
    """Replace what the collection is said to hold. Empty clears it."""
    found = await Collection.get(collection)
    await found.describe(data.description)
    return await found.info()


@put("/api/collections/{collection:str}/name")
@audit.audited("collection.rename")
async def rename_collection(collection: str, data: Rename) -> CollectionInfo:
    """Rename the collection; its documents, settings, index and every session that chose it
    move with it. Refused while any work of the collection runs. The same name is a no-op."""
    renamed = await workflows.rename_collection(collection, data.name)
    audit.attach(renamed_to=renamed.name)
    return await renamed.info()


@post("/api/collections/{collection:str}/index", status_code=202)
@audit.audited("collection.reindex")
async def index_collection(collection: str) -> BulkStarted:
    """(Re)index every member of the collection in the background.

    Accepted, not done: the members are queued in the background, a page at a time, so a
    collection of ten thousand costs this request one insert. Poll the operation for its progress.
    """
    operation_id = await workflows.start_index_collection(collection)
    audit.attach(operation_id=operation_id)
    return BulkStarted(operation_id=operation_id)


@get(
    "/api/collections/{collection:str}/documents",
    mcp_tool="list_collection_documents",
    dependencies=PAGED,
)
async def list_collection_documents(
    collection: str,
    page: PageRequest,
    status: Annotated[MemberStatus | None, one_of(MemberStatus)] = None,
) -> Page[Member]:
    """List the documents of one collection, one page at a time.

    Sort by name, size, status or updated_at; `status` keeps one membership state only (pending,
    indexing, indexed, error, cancelled, removing) — how far this collection got writing the
    document into its index, or taking it out again, which is not the document's own import
    status. Pass the `next_cursor` of a
    response back as `cursor` to continue; it is null on the last page.
    """
    found = await Collection.get(collection)
    return await found.members_page(page, status)


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
    reused from its cache, but the first collection to ask still pays for it. Poll
    `list_collection_documents` until the document reads `indexed`.

    Args:
        session_id: The conversation's id; the attach and its operation then show in that session.
    """
    checked_session(session_id)
    audit.attach(document=data.document)
    logs.bind(document=data.document)
    operation_id = await workflows.attach(collection, data.document)
    audit.attach(operation_id=operation_id)
    await session.record(
        session_id,
        session.Action.ATTACH,
        data.document,
        detail=session.EventDetail(collection=collection),
        operation_id=operation_id,
    )
    return BulkStarted(operation_id=operation_id)


@delete(
    "/api/collections/{collection:str}/documents/{document:str}",
    mcp_tool="remove_document_from_collection",
)
@audit.audited("collection.detach")
async def remove_document(collection: str, document: str, session_id: str | None = None) -> None:
    """Take one document out of this collection: its rows here go, the document stays.

    Queued, not waited out: the removal runs on the collection's single writer, behind any index
    write, compaction or index build already there. The membership reads `removing` from now on
    and is gone once its rows are; poll `list_collection_documents`. A removal that fails leaves
    it in `error`; detaching again retries it.

    Args:
        session_id: The conversation's id; the detach then shows in that session's history.
    """
    checked_session(session_id)
    await workflows.detach(collection, document)
    await session.record(
        session_id,
        session.Action.DETACH,
        document,
        detail=session.EventDetail(collection=collection),
    )


@post("/api/collections/{collection:str}/documents/{document:str}/index", status_code=202)
@audit.audited("collection.reindex")
async def index_collection_document(collection: str, document: str) -> BulkStarted:
    """(Re)index one member: chunk and embed it if the cache misses, then write it into this
    collection's index. Poll the operation for the outcome."""
    operation_id = await workflows.start_index_collection_document(collection, document)
    audit.attach(operation_id=operation_id)
    return BulkStarted(operation_id=operation_id)
