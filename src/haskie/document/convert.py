"""Document to markdown conversion and side-by-side preview artifacts.

Every function here runs in a worker thread (see `cpu.on_cpu`): the parsers take a path and read
it themselves, so this is the only module besides `home.py` where blocking file IO is allowed.
Its own writes go through `home.atomic_write_sync` for the same reason. Calling any of these from
a coroutine blocks that event loop.

Optical character recognition (`ocr`) reads the PDF pages the converter finds no text on, and
raster images. A page it reads nothing on (blank, or OCR off or unavailable) is left to the
`skip_ocr_pages` policy, as before there was OCR.
"""

import io
import re
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec
import pyromark

from haskie import home
from haskie.document import ocr as ocr_reader
from haskie.errors import PermanentError
from haskie.settings import Parser

if TYPE_CHECKING:  # `bookmarks` parses through `render`, which reads this module's markers
    from haskie.document.bookmarks import Bookmark

PREVIEW_PAGES = 10

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".csv", ".json"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
HTML_SUFFIXES = {".html", ".htm"}
ANYDOC_SUFFIXES = {
    ".pdf", ".doc", ".docx", ".docm", ".ppt", ".pps", ".pot", ".pptx", ".pptm", ".ppsx", ".ppsm",
    ".xls", ".xlsx", ".xlsm", ".xlsb", ".odt", ".ods", ".odp", ".rtf", ".epub",
}  # fmt: skip
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | HTML_SUFFIXES | IMAGE_SUFFIXES | ANYDOC_SUFFIXES
RASTER_SUFFIXES = IMAGE_SUFFIXES - {".svg"}  # images of pixels, which Pillow reads and OCR can
OCR_SUFFIXES = RASTER_SUFFIXES | {".pdf"}  # the files OCR can read anything in

# The page marker written into a PDF's markdown. `segment` reads a marker as whitespace, `chunk`
# reads which page a chunk is on and strips the markers from its text (`without_markers`), and
# `render` cuts the document into pages at them. Written and parsed here so every module shares
# one contract.
PAGE_MARKER = re.compile(r"<!-- page (\d+)[^>]*-->")
# A run of page markers and the whitespace around them, the whitespace either side captured
MARKERS = re.compile(
    rf"(?P<before>\s*){PAGE_MARKER.pattern}(?:\s*{PAGE_MARKER.pattern})*(?P<after>\s*)"
)


def without_markers(text: str) -> str:
    """`text` with its page markers taken out, each run of them with the whitespace around it
    made the larger of the whitespace before and after: a paragraph break stays a paragraph break
    (`A.\n\n<!-- page 2 -->\n\nB.` is `A.\n\nB.`), and a line break or a space stays one."""
    if "<!--" not in text:
        return text
    return MARKERS.sub(lambda m: max(m["before"], m["after"], key=_breaks), text)


def _breaks(whitespace: str) -> tuple[int, int]:
    """How much a run of whitespace separates: its line breaks first, then its length."""
    return whitespace.count("\n"), len(whitespace)


# What a page marker notes about its page: OCR read its text, or it was left out for needing OCR
# that read nothing. A PDF's page counts are read off them (`page_counts`).
OCR_READ = "read by OCR"
OCR_SKIPPED = "needs OCR, skipped"


def page_marker(page: int, note: str | None = None) -> str:
    """The marker ahead of one 1-based page, with what it `note`s about the page."""
    return f"<!-- page {page}: {note} -->" if note else f"<!-- page {page} -->"


class PageCounts(msgspec.Struct, frozen=True):
    """What a PDF's conversion made of its pages."""

    pages: int
    ocr: int  # pages the converter found no text on and OCR read
    unread: int  # pages left out with no text, even after OCR


def page_counts(markdown: str, pages: int) -> PageCounts:
    """The counts of a PDF of `pages` pages, converted to `markdown`, off its markers' notes. An
    unread page the policy did not skip failed the conversion (`check_ocr_policy`), so every
    unread page of a converted PDF is noted skipped."""
    markers = [(int(found[1]), found[0]) for found in PAGE_MARKER.finditer(markdown)]
    ocr = sum(marker == page_marker(page, OCR_READ) for page, marker in markers)
    unread = sum(marker == page_marker(page, OCR_SKIPPED) for page, marker in markers)
    return PageCounts(pages=pages, ocr=ocr, unread=unread)


class PreviewKind(StrEnum):
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    HTML = "html"


class Preview(msgspec.Struct):
    kind: PreviewKind
    truncated: bool = False
    pages: int | None = None
    ocr_pages: list[int] = []  # 1-based pages left unread (within the preview)
    # built from the converted markdown, OCR's text included, rather than from the file alone
    converted: bool = False


def _conversion_error(path: Path, exc: Exception) -> PermanentError:
    """A parser failure is a property of the file, not of the moment: retrying cannot fix it."""
    return PermanentError(home.scrub(f"could not convert {path.name}: {type(exc).__name__}: {exc}"))


def _image_markdown(path: Path) -> str:
    """What OCR reads on a raster image."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            page = ocr_reader.image_page(image)
    except Exception as exc:
        raise _conversion_error(path, exc) from exc
    return ocr_reader.read(page).get(1, "")


def to_markdown(path: Path, parser: Parser, ocr: bool = False) -> str:
    """Whole-file conversion, for everything except PDF: a PDF is converted page-wise, in batches
    (`pdf_pages_markdown`), and its OCR policy is applied over the whole document afterwards.
    `ocr` reads a raster image with OCR; any other image has no text."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return _image_markdown(path) if ocr and suffix in RASTER_SUFFIXES else ""
    if parser == Parser.PLAIN or suffix in TEXT_SUFFIXES | HTML_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        raise ValueError(f"PDFs convert page-wise, through pdf_pages_markdown: {path.name}")
    if suffix not in ANYDOC_SUFFIXES:
        raise PermanentError(f"unsupported file type: {suffix or path.name}")
    import anydoc  # OCR deliberately off; PDFs never reach here

    try:
        return anydoc.to_markdown(str(path))
    except Exception as exc:
        raise _conversion_error(path, exc) from exc


class PdfBookmarks(msgspec.Struct, frozen=True):
    """What a conversion plans its batches of a PDF by, read in one pass over the file."""

    pages: int
    # the bookmarks that set its headings (`bookmarks.read`), None when it has none that do
    headings: list["Bookmark"] | None
    starts: list[int]  # the 0-based page every section starts on, one per bookmark of any level


def pdf_bookmarks(path: Path) -> PdfBookmarks:
    """How many pages the PDF has, the bookmarks that set its headings, and where its sections
    start."""
    from pypdf import PdfReader

    from haskie.document import bookmarks

    try:
        reader = PdfReader(str(path))
        every = bookmarks.every(reader)
        headings = bookmarks.headed(every)
        starts = sorted({mark.page - 1 for mark in every})
        return PdfBookmarks(pages=len(reader.pages), headings=headings, starts=starts)
    except Exception as exc:
        raise _conversion_error(path, exc) from exc


def check_ocr_policy(ocr_pages: int, total_pages: int, skip_ocr_pages: bool) -> None:
    """A document with no text, even after OCR, always fails; otherwise pages OCR read nothing on
    fail unless skipped."""
    if total_pages and ocr_pages == total_pages:
        raise PermanentError(f"all {total_pages} pages need OCR and OCR read no text")
    if ocr_pages and not skip_ocr_pages:
        # a re-import keeps the document's own setting, so only a new import can turn it on
        raise PermanentError(
            f"{ocr_pages} of {total_pages} pages need OCR and OCR read no text "
            "(delete the document and import it again with skip_ocr_pages on)"
        )


def pdf_pages_markdown(
    path: Path,
    pages: list[int] | None = None,
    skip_ocr_pages: bool = False,
    marks: Sequence["Bookmark"] | None = None,
    ocr: bool = False,
) -> tuple[str, list[int], int]:
    """Per-page markdown joined with page markers; returns (markdown, 1-based pages left unread,
    pages converted). A page is left unread when the converter finds no text on it and OCR reads
    none either.

    With `ocr`, a page the converter finds no text on is read by OCR (`ocr.read`), and its marker
    notes it. With skip_ocr_pages an unread page becomes a marker comment; otherwise its (empty)
    text stays.
    Policy decisions (fail or not) belong to check_ocr_policy over the whole document.

    `marks` are the PDF's bookmarks (`pdf_bookmarks`), at least those of these pages and the page
    before them, when the document has bookmarks that set its headings: then they do, page by
    page (`bookmarks`). The converter judges a heading by its font and takes running headers for
    sections. A batch with none of its own makes all its headings text. None keeps the
    converter's headings. The page before a batch is converted too, only to learn which
    bookmarks it claims.
    """
    import pdf_inspector

    from haskie.document import bookmarks

    routed = marks is not None
    before = pages[0] - 1 if routed and pages and pages[0] > 0 else None
    wanted = pages if before is None or pages is None else [before, *pages]
    try:
        result = pdf_inspector.extract_pages_markdown(str(path), pages=wanted)
    except Exception as exc:
        raise _conversion_error(path, exc) from exc
    by_page = bookmarks.pages(marks or ())
    # the page before is read too when a bookmark points at it: what it claims, it would claim in
    # its own batch; its text is dropped, so it is not worth reading for anything else
    scanned = [
        one.page + 1
        for one in result.pages
        if one.needs_ocr and (one.page != before or one.page + 1 in by_page)
    ]
    read = ocr_reader.read(path, scanned) if ocr and scanned else {}
    claimed: set[Bookmark] = set()
    ocr_pages: list[int] = []
    parts: list[str] = []
    for one in result.pages:
        number = one.page + 1
        markdown = read.get(number, one.markdown)
        if routed:
            markdown = bookmarks.apply(markdown, number, by_page, claimed)
        if one.page == before:
            continue
        needs_ocr = one.needs_ocr and number not in read
        if needs_ocr:
            ocr_pages.append(number)
        if needs_ocr and skip_ocr_pages:
            parts.append(page_marker(number, OCR_SKIPPED))
        else:
            note = OCR_READ if number in read else None
            parts.append(f"{page_marker(number, note)}\n\n{markdown}")
    return "\n\n".join(parts), ocr_pages, len(parts)


def build_preview(
    source: Path,
    out_dir: Path,
    parser: Parser,
    skip_ocr_pages: bool = False,
    converted: Path | None = None,
) -> Preview:
    """Write `out_dir/source` (left pane) and `out_dir/preview.md` (right pane).

    `converted` is where the document's markdown is written once its conversion is done. When it
    is there, the right pane shows it, so OCR never reads a page twice, and the preview says so
    (`Preview.converted`). Otherwise the file is converted here, without OCR: PDFs only the first
    PREVIEW_PAGES pages, other formats whole (milliseconds).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = converted.read_text(encoding="utf-8") if converted and converted.exists() else None
    if source.suffix.lower() == ".pdf":
        preview = _pdf_preview(source, out_dir, skip_ocr_pages, markdown)
    else:
        preview = _file_preview(source, out_dir, parser, markdown)
    return msgspec.structs.replace(preview, converted=markdown is not None)


def _file_preview(source: Path, out_dir: Path, parser: Parser, markdown: str | None) -> Preview:
    """Every format but PDF: converted whole."""
    suffix = source.suffix.lower()
    full_markdown = to_markdown(source, parser) if markdown is None else markdown
    home.atomic_write_sync(out_dir / "preview.md", full_markdown)
    if suffix in IMAGE_SUFFIXES:
        home.atomic_write_sync(out_dir / "source", source.read_bytes())
        return Preview(kind=PreviewKind.IMAGE)
    if suffix in TEXT_SUFFIXES:
        home.atomic_write_sync(out_dir / "source", source.read_bytes())
        return Preview(kind=PreviewKind.TEXT)
    # office/epub/rtf/odt: browsers cannot render these; show the markdown as HTML instead
    # an HTML file's "markdown" is its own text (`to_markdown`), so it is not read a second time
    html = full_markdown if suffix in HTML_SUFFIXES else pyromark.html(full_markdown)
    home.atomic_write_sync(out_dir / "source", html)
    return Preview(kind=PreviewKind.HTML)


def _pdf_preview(
    source: Path, out_dir: Path, skip_ocr_pages: bool, converted: str | None
) -> Preview:
    """The first PREVIEW_PAGES pages: from the `converted` markdown when there is one."""
    from pypdf import PdfReader, PdfWriter

    from haskie.document import bookmarks

    try:
        reader = PdfReader(str(source))
        total = len(reader.pages)
    except Exception as exc:
        raise _conversion_error(source, exc) from exc
    marks = bookmarks.read(reader)
    shown = min(total, PREVIEW_PAGES)
    writer = PdfWriter()
    for page in reader.pages[:shown]:
        writer.add_page(page)
    buffer = io.BytesIO()
    writer.write(buffer)
    home.atomic_write_sync(out_dir / "source", buffer.getvalue())

    if converted is None:
        markdown, ocr_pages, _ = pdf_pages_markdown(
            source, list(range(shown)), skip_ocr_pages, marks
        )
    else:
        markdown, ocr_pages = _first_pages(converted, shown)
    home.atomic_write_sync(out_dir / "preview.md", markdown)
    return Preview(kind=PreviewKind.PDF, truncated=total > shown, pages=shown, ocr_pages=ocr_pages)


def _first_pages(markdown: str, pages: int) -> tuple[str, list[int]]:
    """The converted markdown of a PDF's first `pages` pages, and the pages in it left unread
    (`page_marker` noted them skipped). An unread page the policy did not skip failed the
    conversion, so there is no converted markdown to hold one."""
    from haskie.document.render import split_pages  # `render` reads this module's markers

    kept = [(page, body) for page, body in split_pages(markdown) if page and page <= pages]
    unread = [page for page, body in kept if body.startswith(page_marker(page, OCR_SKIPPED))]
    return "".join(body for _, body in kept).rstrip() + "\n", unread
