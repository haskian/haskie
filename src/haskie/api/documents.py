"""Document routes: the two-phase intake, the listing, the delete, and the two preview panes.

Every route here is collection-independent: a document is imported once, under a name that never
changes, and which collections hold it is a membership the collection routes manage.
"""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Literal

import anyio
import msgspec
from litestar import delete, get, post, put
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.response import File, Stream

from haskie import audit, cpu, logs
from haskie.api.common import PAGED, BulkStarted, Describe
from haskie.document import convert, render
from haskie.document import document as documents
from haskie.document.document import Document, DocumentStatus, ImportOptions, Staged
from haskie.errors import InvalidInput, NotFound
from haskie.indexing import embed_cache, workflows
from haskie.paging import Page, PageRequest, one_of
from haskie.search import session


class ImportRequest(ImportOptions):
    """What to import: either a staged upload or a local file, never both."""

    staging_id: str | None = None
    path: str | None = None


class Head(msgspec.Struct):
    """The first line of a rendered document: everything the pane needs before any page."""

    toc: list[render.Heading]
    preview: convert.Preview | None
    pages: int
    kind: Literal["head"] = "head"


@post("/api/documents/staging")
@audit.audited("document.stage")
async def stage_document(
    data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
) -> Staged:
    """Upload a file and keep it until it is imported. Nothing is committed here: no name, no
    document row. Call `import_document` with the returned `staging_id` to commit it."""
    content = await data.read()
    upload = Path(data.filename)
    audit.attach(name=upload.name, size=len(content), suffix=upload.suffix.lower())
    return await documents.stage(data.filename, content)


@post("/api/documents/import", mcp_tool="add_document")
@audit.audited("document.import")
async def import_document(data: ImportRequest, session_id: str | None = None) -> Document:
    """Import a staged upload (`staging_id`) or a local file by absolute path (`path`).

    The name is fixed here and never changes: `name` renames the document, but the original
    suffix is kept because it decides how the document is parsed. Returns the document at status
    `queued`; convert and embed then run in the background, so poll `get_document` for `imported`.

    Args:
        session_id: The conversation's id; the import and its operation then show in that session.
    """
    if data.staging_id is not None and data.path is None:
        row = await documents.import_staged(data.staging_id, data)
    elif data.path is not None and data.staging_id is None:
        # the audit trail records what was imported, never where it came from
        audit.attach(source=Path(data.path).name)
        row = await documents.import_path(data.path, data)
    else:
        raise InvalidInput("give either staging_id or path")
    audit.attach(document=row.name, size=row.size)
    logs.bind(document=row.name)
    operation_id = await workflows.start_import(row.name)
    audit.attach(operation_id=operation_id)
    await session.record(session_id, session.Action.IMPORT, row.name, operation_id=operation_id)
    return row


@get("/api/documents", mcp_tool="list_documents", dependencies=PAGED)
async def list_documents(
    page: PageRequest, status: Annotated[DocumentStatus | None, one_of(DocumentStatus)] = None
) -> Page[documents.Listed]:
    """List every imported document, one page at a time, with how many collections hold each.

    Sort by name, size, status or updated_at; `status` keeps one lifecycle state only (queued,
    converting, embedding, imported, error, cancelled, deleting). Pass the `next_cursor` of a
    response back as `cursor` to continue; it is null on the last page.
    """
    found = await documents.page(page, status)
    return Page(
        items=await documents.listed(found.items),
        next_cursor=found.next_cursor,
        total=found.total,
    )


@get("/api/documents/{document:str}", mcp_tool="get_document")
async def get_document(document: str) -> documents.Listed:
    """One document: its import status, its size, what it is said to be, and how many collections
    hold it."""
    (found,) = await documents.listed([await documents.get(document)])
    return found


@delete("/api/documents/{document:str}", status_code=202)
@audit.audited("document.delete")
async def delete_document(document: str) -> BulkStarted:
    """Queue the deletion: the document goes from every collection that holds it, then its files,
    its embedding cache and its row go.

    Accepted, not done: each collection's index is cleaned on its own partition, which takes as
    long as the work already queued there. Poll the operation for the outcome.
    """
    operation_id = await workflows.start_delete_document(document)
    audit.attach(operation_id=operation_id)
    return BulkStarted(operation_id=operation_id)


@post("/api/documents/{document:str}/import", status_code=202)
@audit.audited("document.reimport")
async def reimport_document(document: str) -> BulkStarted:
    """Run a document's import again after it failed or was cancelled.

    `start_import` refuses the rest with a conflict. A document already imported has its markdown
    and its cache. One still converting or embedding is being written right now, so a re-run would
    race it. A queued document is let through: the deduplication returns the import already queued
    for it.
    """
    operation_id = await workflows.start_import(document)
    audit.attach(operation_id=operation_id)
    return BulkStarted(operation_id=operation_id)


@get("/api/documents/{document:str}/collections")
async def list_document_collections(document: str) -> list[str]:
    """Which collections hold this document, in name order."""
    await documents.get(document)  # NotFound rather than an empty list for a name nobody owns
    return await documents.collections_of(document)


@get("/api/documents/{document:str}/embeddings")
async def list_document_embeddings(document: str) -> list[embed_cache.Entry]:
    """What the embedding cache holds for this document: one entry per distinct chunk settings
    and embedding model, shared by every collection that indexes it with them."""
    await documents.get(document)
    return await embed_cache.entries(document)


@get("/api/documents/{document:str}/source")
async def get_source(document: str) -> File:
    row = await documents.get(document)
    return File(path=row.source_path(), content_disposition_type="inline")


PREVIEW_MEDIA = {
    convert.PreviewKind.PDF: "application/pdf",
    convert.PreviewKind.HTML: "text/html",
    convert.PreviewKind.TEXT: "text/plain",
}


@get("/api/documents/{document:str}/preview")
async def get_preview(document: str) -> File:
    """Left pane: original (pdf cut to first pages, image, text) or HTML stand-in for office."""
    info, preview = await documents.ensure_preview(document)
    media = PREVIEW_MEDIA.get(preview.kind)
    return File(
        path=info.preview_dir / "source",
        filename=document if media is None else None,
        media_type=media,
        content_disposition_type="inline",
    )


@get("/api/documents/{document:str}/markdown")
async def get_markdown(document: str, full: bool = False) -> Stream:
    """Right pane, as NDJSON: one `head` line, then one `page` line per page of HTML.

    Streamed because a full text is one lump otherwise - a 1200-page book renders to megabytes,
    and the pane could show nothing until all of it had arrived and been parsed. Rendered on the
    server because the browser then inserts HTML instead of parsing markdown, and because raw HTML
    has to be dropped somewhere it cannot be forgotten (see `render`).

    `full=true` is the whole converted text; the default is the preview (only the first pages of a
    PDF).
    """
    info, preview = await documents.ensure_preview(document)
    path = anyio.Path(info.markdown if full else info.preview_dir / "preview.md")
    if not await path.exists():
        raise NotFound(f"document not imported yet: {document}")
    markdown = await path.read_text(encoding="utf-8")

    async def frames() -> AsyncIterator[bytes]:
        # rendering is CPU work on a big document, so it goes through the budget like the rest
        rendered, toc = await cpu.on_cpu(render.pages, markdown)
        head = Head(toc=toc, preview=preview, pages=len(rendered))
        yield msgspec.json.encode(head) + b"\n"
        for page in rendered:
            yield msgspec.json.encode(page) + b"\n"

    return Stream(frames(), media_type="application/x-ndjson")


@put("/api/documents/{document:str}/description", mcp_tool="describe_document")
@audit.audited("document.describe")
async def describe_document(
    document: str, data: Describe, session_id: str | None = None
) -> Document:
    """Replace what the document is said to be. Empty clears it.

    The description is what `search_sources` returns beside each document, so it is worth writing
    for anything an agent is expected to choose between.

    Args:
        session_id: The conversation's id; the change then shows in that session's history.
    """
    described = await documents.describe(document, data.description)
    await session.record(session_id, session.Action.DESCRIBE, document)
    return described
