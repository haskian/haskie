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

from haskie import audit, convert, cpu, document, embed_cache, logs, render, workflows
from haskie.api.common import PAGED, BulkStarted, Describe
from haskie.document import DocStatus, Document, ImportOptions, Staged
from haskie.errors import InvalidInput, NotFound
from haskie.paging import Page, PageRequest


class ImportRequest(ImportOptions):
    """What to import: either a staged upload or a local file, never both."""

    staging_id: str | None = None
    path: str | None = None


class Head(msgspec.Struct):
    """The first frame of a rendered document: everything the pane needs before any page."""

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
    return await document.stage(data.filename, content)


@post("/api/documents/import", mcp_tool="add_document")
@audit.audited("document.import")
async def import_document(data: ImportRequest) -> Document:
    """Import a staged upload (`staging_id`) or a local file by absolute path (`path`).

    The name is fixed here and never changes: `name` renames the document, but the original
    suffix is kept because it decides how the document is parsed. Returns the document at status
    `queued`; convert and embed then run in the background, so poll `get_document` for `imported`.
    """
    if data.staging_id is not None and data.path is None:
        row = await document.import_staged(data.staging_id, data)
    elif data.path is not None and data.staging_id is None:
        # the audit trail records what was imported, never where it came from
        audit.attach(source=Path(data.path).name)
        row = await document.import_path(data.path, data)
    else:
        raise InvalidInput("give either staging_id or path")
    audit.attach(doc=row.name, size=row.size)
    logs.bind(doc=row.name)
    audit.attach(job_id=await workflows.start_import(row.name))
    return row


@get("/api/documents", mcp_tool="list_documents", dependencies=PAGED)
async def list_documents(page: PageRequest, status: DocStatus | None = None) -> Page[Document]:
    """List every imported document, one page at a time, whichever collections hold them.

    Sort by name, size, status or updated_at; `status` keeps one lifecycle state only (queued,
    converting, embedding, imported, error, cancelled, deleting). Pass the `next_cursor` of a
    response back as `cursor` to continue; it is null on the last page.
    """
    return await document.page(page, status)


@get("/api/documents/{doc:str}", mcp_tool="get_document")
async def get_document(doc: str) -> Document:
    """One document: its import status, its size and what it is said to be."""
    return await document.get(doc)


@delete("/api/documents/{doc:str}", status_code=202)
@audit.audited("document.delete")
async def delete_document(doc: str) -> BulkStarted:
    """Queue the deletion: the document goes from every collection that holds it, then its files,
    its embedding cache and its row go.

    Accepted, not done: each collection's index is cleaned on its own partition, which takes as
    long as the work already queued there. Poll the job for the outcome.
    """
    job_id = await workflows.start_delete_document(doc)
    audit.attach(job_id=job_id)
    return BulkStarted(job_id=job_id)


@post("/api/documents/{doc:str}/import", status_code=202)
@audit.audited("document.reimport")
async def reimport_document(doc: str) -> BulkStarted:
    """Run the import of a document that failed or was cancelled again.

    Only those two: a document already imported has its markdown and its cache, and one still in
    the pipeline is being written right now, so re-running would race it. `start_import` is what
    refuses the rest, with a conflict.
    """
    job_id = await workflows.start_import(doc)
    audit.attach(job_id=job_id)
    return BulkStarted(job_id=job_id)


@get("/api/documents/{doc:str}/collections")
async def list_document_collections(doc: str) -> list[str]:
    """Which collections hold this document, in name order."""
    await document.get(doc)  # NotFound rather than an empty list for a name nobody owns
    return await document.collections_of(doc)


@get("/api/documents/{doc:str}/embeddings")
async def list_document_embeddings(doc: str) -> list[embed_cache.Entry]:
    """What the embedding cache holds for this document: one entry per distinct chunk settings
    and embedding model, shared by every collection that indexes it with them."""
    await document.get(doc)
    return await embed_cache.entries(doc)


@get("/api/documents/{doc:str}/source")
async def get_source(doc: str) -> File:
    row = await document.get(doc)
    return File(path=row.source_path(), content_disposition_type="inline")


PREVIEW_MEDIA = {"pdf": "application/pdf", "html": "text/html", "text": "text/plain"}


@get("/api/documents/{doc:str}/preview")
async def get_preview(doc: str) -> File:
    """Left pane: original (pdf cut to first pages, image, text) or HTML stand-in for office."""
    info, preview = await document.ensure_preview(doc)
    media = PREVIEW_MEDIA.get(preview.kind)
    return File(
        path=info.preview_dir / "source",
        filename=doc if media is None else None,
        media_type=media,
        content_disposition_type="inline",
    )


@get("/api/documents/{doc:str}/markdown")
async def get_markdown(doc: str, full: bool = False) -> Stream:
    """Right pane, as NDJSON: one `head` frame, then one `page` frame per page of HTML.

    Streamed because a full text is one lump otherwise - a 1200-page book renders to megabytes,
    and the pane could show nothing until all of it had arrived and been parsed. Rendered on the
    server because the browser then inserts HTML instead of parsing markdown, and because raw HTML
    has to be dropped somewhere it cannot be forgotten (see `render`).

    `full=true` is the whole converted text; the default is the preview, the first pages only.
    """
    info, preview = await document.ensure_preview(doc)
    path = anyio.Path(info.markdown if full else info.preview_dir / "preview.md")
    if not await path.exists():
        raise NotFound(f"document not imported yet: {doc}")
    markdown = await path.read_text(encoding="utf-8")

    async def frames() -> AsyncIterator[bytes]:
        # rendering is CPU work on a big document, so it goes through the budget like the rest
        rendered, toc = await cpu.on_cpu(render.pages, markdown)
        head = Head(toc=toc, preview=preview, pages=len(rendered))
        yield msgspec.json.encode(head) + b"\n"
        for page in rendered:
            yield msgspec.json.encode(page) + b"\n"

    return Stream(frames(), media_type="application/x-ndjson")


@put("/api/documents/{doc:str}/description", mcp_tool="describe_document")
@audit.audited("document.describe")
async def describe_document(doc: str, data: Describe) -> Document:
    """Replace what the document is said to be. Empty clears it.

    The description is what `search_documents` returns beside each match, so it is worth writing
    for anything an agent is expected to choose between."""
    return await document.describe(doc, data.description)
