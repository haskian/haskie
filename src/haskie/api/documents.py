"""Document routes: upload, import, index, delete, and the two preview panes."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

import anyio
import msgspec
from litestar import delete, get, post, put
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.response import File, Stream

from haskie import audit, convert, cpu, logs, render, toc, workflows
from haskie.api.common import BulkStarted
from haskie.errors import DocumentNotFound
from haskie.library import Document, Library


class ImportFile(msgspec.Struct):
    path: str
    rename_to: str | None = None  # store it under this name instead of the file's own
    description: str = ""


class Describe(msgspec.Struct):
    description: str


class Head(msgspec.Struct):
    """The first frame of a rendered document: everything the pane needs before any page."""

    kind: str  # "head"
    toc: list[toc.Heading]
    preview: convert.Preview | None
    pages: int


async def _cancel_running(lib: Library, doc: str) -> None:
    """Writing over a document whose pipeline still runs would race its steps, so cancel and
    wait for it first. Nothing to do for a document that does not exist yet."""
    try:
        await lib.document(doc)
    except DocumentNotFound:
        return
    await workflows.cancel_document(lib.name, doc)


async def _save_upload(
    name: str, filename: str, content: bytes, rename_to: str | None, description: str
) -> Document:
    """The storing half of `upload_document`, including the library lookup."""
    lib = await Library.get(name)
    # the running pipeline to stop is the one under the name this upload will land on
    await _cancel_running(lib, lib._stored_name(filename, rename_to))
    return await lib.save(filename, content, rename_to, description)


@post("/api/libraries/{name:str}/documents")
@audit.audited("document.add", library="name")
async def upload_document(
    name: str,
    data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
    rename_to: str | None = None,
    description: str = "",
) -> Document:
    """Upload a file. `rename_to` stores it under a name of your choosing (the original suffix is
    kept, because it decides how the document is parsed); `description` says what it is."""
    content = await data.read()
    upload = Path(data.filename)
    audit.attach(name=upload.name, size=len(content), suffix=upload.suffix.lower())
    return await _save_upload(name, data.filename, content, rename_to, description)


@post("/api/libraries/{name:str}/documents/import", mcp_tool="add_document")
@audit.audited("document.add", library="name")
async def import_document(name: str, data: ImportFile) -> Document:
    """Add a local file by absolute path (pdf, markdown, office, epub...) as `uploaded`.

    Call `index_document` to make it searchable.
    """
    source = Path(data.path).expanduser()
    # the audit trail records what was added, never where it came from
    audit.attach(name=source.name, suffix=source.suffix.lower())
    lib = await Library.get(name)
    logs.bind(doc=source.name)
    await _cancel_running(lib, lib._stored_name(source.name, data.rename_to))
    document = await lib.save_path(data.path, data.rename_to, data.description)
    audit.attach(size=document.size)
    return document


@post("/api/libraries/{name:str}/documents/{doc:str}/index", mcp_tool="index_document")
@audit.audited("document.reindex", library="name", doc="doc")
async def index_document(name: str, doc: str) -> BulkStarted:
    """Queue convert -> embed -> index for one document; poll `list_documents` for status."""
    job_id = await workflows.start_index(name, doc)
    audit.attach(job_id=job_id)
    return BulkStarted(job_id=job_id)


@delete("/api/libraries/{name:str}/documents/{doc:str}")
@audit.audited("document.delete", library="name", doc="doc")
async def delete_document(name: str, doc: str) -> None:
    """Cancels a running pipeline, then removes index rows, files and the row."""
    await workflows.remove_document(name, doc)


@get("/api/libraries/{name:str}/documents/{doc:str}/source")
async def get_source(name: str, doc: str) -> File:
    lib = await Library.get(name)
    return File(path=lib.source_path(doc), content_disposition_type="inline")


PREVIEW_MEDIA = {"pdf": "application/pdf", "html": "text/html", "text": "text/plain"}


@get("/api/libraries/{name:str}/documents/{doc:str}/preview")
async def get_preview(name: str, doc: str) -> File:
    """Left pane: original (pdf cut to first pages, image, text) or HTML stand-in for office."""
    lib = await Library.get(name)
    info = await lib.ensure_preview(doc)
    if info.preview is None:
        raise RuntimeError(f"preview not stored for {doc}")
    media = PREVIEW_MEDIA.get(info.preview.kind)
    return File(
        path=lib.preview_dir(doc) / "source",
        filename=doc if media is None else None,
        media_type=media,
        content_disposition_type="inline",
    )


@get("/api/libraries/{name:str}/documents/{doc:str}/markdown")
async def get_markdown(name: str, doc: str, full: bool = False) -> Stream:
    """Right pane, as NDJSON: one `head` frame, then one `page` frame per page of HTML.

    Streamed because a full text is one lump otherwise - a 1200-page book renders to megabytes,
    and the pane could show nothing until all of it had arrived and been parsed. Rendered on the
    server because the browser then inserts HTML instead of parsing markdown, and because raw HTML
    has to be dropped somewhere it cannot be forgotten (see `render`).

    `full=true` is the indexed text; the default is the preview, which is the first pages only.
    """
    lib = await Library.get(name)
    info = await lib.ensure_preview(doc)
    path = anyio.Path(lib.markdown_path(doc) if full else lib.preview_dir(doc) / "preview.md")
    if not await path.exists():
        raise DocumentNotFound(f"document not indexed yet: {doc}")
    markdown = await path.read_text(encoding="utf-8")

    async def frames() -> AsyncIterator[bytes]:
        # rendering is CPU work on a big document, so it goes through the budget like the rest
        rendered = await cpu.on_cpu("render", render.pages, markdown)
        head = Head(
            kind="head",
            toc=toc.headings(markdown),
            preview=info.preview,
            pages=len(rendered),
        )
        yield msgspec.json.encode(head) + b"\n"
        for page in rendered:
            yield msgspec.json.encode({"kind": "page", **msgspec.structs.asdict(page)}) + b"\n"

    return Stream(frames(), media_type="application/x-ndjson")


@put("/api/libraries/{name:str}/documents/{doc:str}/description", mcp_tool="describe_document")
@audit.audited("document.describe", library="name", doc="doc")
async def describe_document(name: str, doc: str, data: Describe) -> Document:
    """Replace what the document is said to be. Empty clears it.

    The description is what `search_documents` returns beside each match, so it is worth writing
    for anything an agent is expected to choose between."""
    lib = await Library.get(name)
    return await lib.describe_document(doc, data.description)
