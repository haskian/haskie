"""Library routes: the collection, one library, its settings, its documents listing."""

import msgspec
from litestar import delete, get, post, put

from haskie import audit, logs, models, workflows
from haskie.api.common import BulkStarted, Limit
from haskie.index import Hit
from haskie.library import (
    DocStatus,
    Document,
    Library,
    LibraryInfo,
    LibrarySummary,
)
from haskie.paging import DEFAULT_PAGE_SIZE, Order, Page, page_request
from haskie.settings import (
    Fusion,
    LibrarySettings,
    Reranker,
    SearchMode,
    SearchOverrides,
    load_user_settings,
    without_none,
)


class CreateLibrary(msgspec.Struct):
    name: str
    description: str = ""


class Describe(msgspec.Struct):
    description: str


@get("/api/libraries", mcp_tool="list_libraries")
async def list_libraries(
    cursor: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    sort: str | None = None,
    order: Order = "asc",
) -> Page[LibrarySummary]:
    """List document libraries with their document counts, one page at a time.

    Sort by name or created_at, ascending or descending. Pass the `next_cursor` of a response
    back as `cursor` to continue; it is null on the last page.
    """
    return await Library.page(page_request(cursor, page_size, sort, order))


@post("/api/libraries")
@audit.audited("library.create")
async def create_library(data: CreateLibrary) -> LibraryInfo:
    lib = await Library.create(data.name, data.description)
    audit.attach(library=lib.name)
    logs.bind(library=lib.name)
    return await lib.info()


@get("/api/libraries/{name:str}", mcp_tool="get_library")
async def get_library(name: str) -> LibraryInfo:
    """Describe one library: its settings and how many documents it holds per status.

    The documents themselves are listed by `list_documents`.
    """
    lib = await Library.get(name)
    return await lib.info()


@get("/api/libraries/{name:str}/documents", mcp_tool="list_documents")
async def list_documents(
    name: str,
    cursor: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    sort: str | None = None,
    order: Order = "asc",
    status: DocStatus | None = None,
) -> Page[Document]:
    """List the documents of one library, one page at a time.

    Sort by name, size, status or updated_at; `status` keeps one lifecycle state only
    (uploaded, queued, converting, embedding, indexing, indexed, error, cancelled). Pass the
    `next_cursor` of a response back as `cursor` to continue; it is null on the last page.
    """
    lib = await Library.get(name)
    return await lib.documents_page(page_request(cursor, page_size, sort, order), status)


@delete("/api/libraries/{name:str}", status_code=202)
@audit.audited("library.delete", library="name")
async def delete_library(name: str) -> BulkStarted:
    """Queue the deletion: every running pipeline of the library is cancelled before the files go.

    Accepted, not done: cancelling a busy library and removing its folder takes as long as the
    last running step, which is no time to hold a request open. Poll the job for the outcome.
    """
    job_id = await workflows.start_delete_library(name)
    audit.attach(job_id=job_id)
    return BulkStarted(job_id=job_id)


@get("/api/libraries/{name:str}/search", mcp_tool="search_library")
async def search_library(
    name: str,
    q: str,
    limit: Limit = None,
    mode: SearchMode | None = None,
    fusion: Fusion | None = None,
    vector_weight: float | None = None,
    bm25_weight: float | None = None,
    reranker: Reranker | None = None,
    candidates: int | None = None,
) -> list[Hit]:
    """Search one library. Options default to the library's search settings:
    mode hybrid|vector|fts, fusion rrf|linear, vector_weight/bm25_weight for linear,
    reranker none|cross-encoder (rescoring of `candidates` for any mode)."""
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
    lib = await Library.get(name)
    return await lib.search(q, limit, **overrides)


@put("/api/libraries/{name:str}/settings")
@audit.audited("library.settings.update", library="name")
async def put_library_settings(name: str, data: LibrarySettings) -> LibraryInfo:
    """Saves, then downloads: a library may override the reranker model, and a search of this
    library would otherwise fail with "not loaded yet" for a model nothing ever fetched."""
    lib = await Library.get(name)
    await lib.set_settings(data)
    await models.ensure_models(await load_user_settings())
    return await lib.info()


@post("/api/libraries/{name:str}/index", status_code=202)
@audit.audited("library.reindex", library="name")
async def index_library(name: str) -> BulkStarted:
    """(Re)index every document in the library in the background.

    Accepted, not done: the documents are queued by a background job, a page at a time, so a
    library of ten thousand costs this request one insert. Poll the job for its progress.
    """
    job_id = await workflows.start_index_library(name)
    audit.attach(job_id=job_id)
    return BulkStarted(job_id=job_id)


@put("/api/libraries/{name:str}/description")
@audit.audited("library.describe", library="name")
async def describe_library(name: str, data: Describe) -> LibraryInfo:
    """Replace what the library is said to hold. Empty clears it."""
    lib = await Library.get(name)
    await lib.describe(data.description)
    return await lib.info()
