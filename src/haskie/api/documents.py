"""Document routes: the two-phase intake, the listing, the delete, and the two preview panes.

Every route here is collection-independent: a document is imported once, and which collections
hold it is a membership the collection routes manage. A route addresses a document by its name;
a rename changes only that (`rename_document`).
"""

from collections.abc import AsyncIterator
from itertools import islice
from pathlib import Path
from typing import Annotated, Literal

import anyio
import anyio.to_thread
import msgspec
from litestar import delete, get, post, put
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.response import File, Stream

from haskie import audit, cpu, logs
from haskie.api.common import PAGED, BulkStarted, Describe, Rename, SessionId
from haskie.catalogue import catalogue
from haskie.collection.index import location
from haskie.document import convert, render
from haskie.document import document as documents
from haskie.document.document import Document, DocumentStatus, ImportOptions, Staged
from haskie.errors import InvalidInput, NotFound
from haskie.indexing import embed_cache, workflows
from haskie.outline import store
from haskie.paging import Page, PageRequest, one_of
from haskie.search import session
from haskie.settings import load_user_settings


class ImportRequest(ImportOptions):
    """What to import: either a staged upload or a local file, never both."""

    staging_id: str | None = None
    path: str | None = None


class Similar(msgspec.Struct):
    """What a document may repeat: the documents whose content lies closest to it. The same file
    is never imported twice, as its bytes are what a document is."""

    nearest: list[embed_cache.Neighbour]  # empty until it is imported, or with no embedding model


NEAREST = 3  # enough to spot a second edition, few enough to read at a glance


class OutlineSection(msgspec.Struct):
    """One section of a document's outline, and what it is about."""

    header: str  # its heading path, ready to cite; empty for the whole document
    depth: int  # how many headings deep: 0 for the whole document
    location: str
    line_start: int
    line_end: int
    chars: int  # how long it is: what reading it costs at most
    keywords: list[str]  # what it is about, against the other sections of its depth


class Head(msgspec.Struct):
    """The first line of a rendered document: everything the pane needs before any page."""

    toc: list[render.Heading]
    preview: convert.Preview | None
    pages: int
    kind: Literal["head"] = "head"


MAX_RENDERED = 100_000  # characters: an excerpt is at most a few sections (`max_answer_chars`)


class Markdown(msgspec.Struct):
    """Markdown to render, as a search result quotes it."""

    markdown: Annotated[str, msgspec.Meta(max_length=MAX_RENDERED)]


class Rendered(msgspec.Struct):
    html: str


@post("/api/documents/render", status_code=200)
async def render_markdown(data: Markdown) -> Rendered:
    """A search result's text as HTML, rendered as the document viewer renders a page: raw HTML is
    stripped first, so the text a document holds cannot inject markup."""
    return Rendered(html=await cpu.on_cpu(render.fragment_html, data.markdown))


@post("/api/documents/staging")
@audit.audited("document.stage")
async def stage_document(
    data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
) -> Staged:
    """Upload a file and keep it until it is imported. Nothing is committed here: no name, no
    document row. Call `import_document` with the returned `staging_id` to commit it.

    `duplicate` names the document these bytes already are, if any: the bytes are the document,
    so importing them again is refused with 409."""
    content = await data.read()
    upload = Path(data.filename)
    audit.attach(name=upload.name, size=len(content), suffix=upload.suffix.lower())
    return await documents.stage(data.filename, content)


@post("/api/documents/import", mcp_tool="add_document")
@audit.audited("document.import")
async def import_document(data: ImportRequest, session_id: SessionId = None) -> Document:
    """Import a staged upload (`staging_id`) or a local file by absolute path (`path`).

    `name` names the document, else the file does. The name is stored in lowercase-kebab-case
    (`My Notes` becomes `my-notes.md`), with the original suffix, which decides how it is
    parsed: address the document by the `name` of the returned row from then on. Returns the
    document at status `queued`; convert and embed then run in the background, so poll
    `get_document` for `imported`.

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
    operation_id = await workflows.start_import(row)
    audit.attach(operation_id=operation_id)
    await session.record(session_id, session.Action.IMPORT, row.name, operation_id=operation_id)
    return row


@get("/api/documents", mcp_tool="list_documents", dependencies=PAGED)
async def list_documents(
    page: PageRequest, status: Annotated[DocumentStatus | None, one_of(DocumentStatus)] = None
) -> Page[documents.Listed]:
    """List every imported document, one page at a time, with the collections holding each.

    Sort by name, size, status, created_at or updated_at; `status` keeps one lifecycle state only
    (queued, converting, embedding, imported, error, cancelled, deleting). Pass the `next_cursor`
    of a response back as `cursor` to continue; it is null on the last page.
    """
    found = await documents.page(page, status)
    return Page(
        items=await documents.listed(found.items),
        next_cursor=found.next_cursor,
        total=found.total,
    )


@get("/api/documents/{document:str}", mcp_tool="get_document")
async def get_document(document: str) -> documents.Listed:
    """One document: its import status, its size, what it is said to be, and the collections
    holding it."""
    (found,) = await documents.listed([await documents.named(document)])
    return found


@delete("/api/documents/{document:str}", status_code=202)
@audit.audited("document.delete")
async def delete_document(document: str) -> BulkStarted:
    """Queue the deletion: the document goes from every collection that holds it, then its files,
    its embedding cache and its row go.

    Accepted, not done: each collection's index is cleaned on its own partition, which takes as
    long as the work already queued there. Poll the operation for the outcome.
    """
    operation_id = await workflows.start_delete_document(await documents.named(document))
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
    operation_id = await workflows.start_import(await documents.named(document))
    audit.attach(operation_id=operation_id)
    return BulkStarted(operation_id=operation_id)


@get("/api/documents/{document:str}/collections")
async def list_document_collections(document: str) -> list[str]:
    """Which collections hold this document, in name order."""
    # NotFound rather than an empty list for a name nobody owns
    return await documents.collections_of(await documents.id_of(document))


@get("/api/documents/{document:str}/embeddings")
async def list_document_embeddings(document: str) -> list[embed_cache.Entry]:
    """What the embedding cache holds for this document: one entry per distinct chunk settings
    and embedding model, shared by every collection that indexes it with them."""
    return await embed_cache.entries(await documents.id_of(document))


@get("/api/documents/{document:str}/similar")
async def similar_documents(document: str) -> Similar:
    """The three documents nearest to this one by the mean vector of their chunks under the
    current embedding model: a second edition, a near copy."""
    row = await documents.named(document)
    model = await catalogue.embedding_model(await load_user_settings())
    nearest = [] if model is None else await embed_cache.nearest(row.id, model.cache_name, NEAREST)
    return Similar(nearest=nearest)


@get("/api/documents/{document:str}/outline", mcp_tool="document_outline")
async def document_outline(document: str) -> list[OutlineSection]:
    """A document's table of contents, each section with the words that say what it is about.

    Every section in document order, a parent before its children: first the whole document
    (`depth` 0), then each heading's section at its depth. `keywords` are the words a section uses
    more than the other sections of its depth, a chapter against the other chapters, best first,
    and those closest in meaning to the section when there is an embedding model. Cite a section
    by its `header` and `location`; ask `search_excerpts` for its text. Empty until the document
    is imported.
    """
    row = await documents.named(document)
    nodes = (await store.read([row.id])).get(row.id, [])
    return [
        OutlineSection(
            header=node.header,
            depth=node.depth,
            location=location(
                row.name, node.page_start, node.page_end, node.line_start, node.line_end
            ),
            line_start=node.line_start,
            line_end=node.line_end,
            chars=node.char_end - node.char_start,
            keywords=list(node.keywords),
        )
        for node in nodes
    ]


# A document's own bytes are served on haskie's origin, where an HTML or SVG file would run its
# script against an API that authenticates no one. `sandbox` gives the response an opaque origin
# and no script, however it is opened: in the pane, in a new tab, or from a link. PDF is left out:
# the sandbox blocks the browser's PDF viewer, which runs a PDF's script in its own sandbox anyway.
UNTRUSTED_HEADERS = {"Content-Security-Policy": "sandbox", "X-Content-Type-Options": "nosniff"}


def _untrusted_headers(is_pdf: bool) -> dict[str, str]:
    return {} if is_pdf else UNTRUSTED_HEADERS


@get("/api/documents/{document:str}/source")
async def get_source(document: str) -> File:
    row = await documents.named(document)
    # named after the document, not the stored `original.*`: the name is what the media type is
    # guessed from, and what a browser that saves it calls the file
    return File(
        path=row.source_path(),
        filename=row.name,
        content_disposition_type="inline",
        headers=_untrusted_headers(row.suffix == ".pdf"),
    )


PREVIEW_MEDIA = {
    convert.PreviewKind.PDF: "application/pdf",
    convert.PreviewKind.HTML: "text/html",
    convert.PreviewKind.TEXT: "text/plain",
}


@get("/api/documents/{document:str}/preview")
async def get_preview(document: str) -> File:
    """Left pane: original (pdf cut to first pages, image, text) or HTML stand-in for office."""
    info, preview = await documents.ensure_preview(await documents.named(document))
    media = PREVIEW_MEDIA.get(preview.kind)
    return File(
        path=info.preview_dir / "source",
        filename=document if media is None else None,
        media_type=media,
        content_disposition_type="inline",
        headers=_untrusted_headers(preview.kind == convert.PreviewKind.PDF),
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
    info, preview = await documents.ensure_preview(await documents.named(document))
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


MAX_LINES = 400  # a pointer's lines, not a document: `markdown` is there for the whole of it


class Lines(msgspec.Struct):
    """Some lines of a document's markdown, page markers taken out."""

    text: str


def _read_lines(path: Path, line_start: int, line_end: int) -> str:
    """Lines `[line_start, line_end]` of `path`, read up to the last one rather than whole."""
    with path.open(encoding="utf-8") as handle:
        return "".join(islice(handle, line_start - 1, line_end))


@get("/api/documents/{document:str}/lines")
async def get_lines(document: str, line_start: int, line_end: int) -> Lines:
    """Lines of the converted markdown, 1-based and inclusive: what a search result's `also_in`
    points at, read back when the reader opens it. At most `MAX_LINES` at a time."""
    if not 1 <= line_start <= line_end or line_end - line_start >= MAX_LINES:
        raise InvalidInput(
            f"lines must be 1 <= line_start <= line_end, at most {MAX_LINES} of them; "
            f"got {line_start}-{line_end}"
        )
    info = await documents.named(document)
    try:
        raw = await anyio.to_thread.run_sync(_read_lines, info.markdown, line_start, line_end)
    except FileNotFoundError as missing:
        raise NotFound(f"document not imported yet: {document}") from missing
    return Lines(text=convert.without_markers(raw).strip())


@put("/api/documents/{document:str}/name")
@audit.audited("document.rename")
async def rename_document(document: str, data: Rename) -> Document:
    """Rename the document. Its collections, indexes and cached embeddings stay as they are, and
    searches cite it by the new name at once. The suffix stays the original's, since it decides
    how the file was converted, and the name is stored in lowercase-kebab-case, as at import.
    The same name is a no-op; a name taken is 409."""
    renamed = await documents.rename(await documents.named(document), data.name)
    audit.attach(renamed_to=renamed.name)
    return renamed


@put("/api/documents/{document:str}/description", mcp_tool="describe_document")
@audit.audited("document.describe")
async def describe_document(
    document: str, data: Describe, session_id: SessionId = None
) -> Document:
    """Replace what the document is said to be. Empty clears it.

    The description is what `search_sources` returns beside each document, so it is worth writing
    for anything an agent is expected to choose between.

    Args:
        session_id: The conversation's id; the change then shows in that session's history.
    """
    described = await documents.describe(await documents.id_of(document), data.description)
    await session.record(session_id, session.Action.DESCRIBE, document)
    return described
