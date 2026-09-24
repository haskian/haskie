"""Document to markdown conversion and side-by-side preview artifacts.

Every function here runs in a worker thread (see `cpu.on_cpu`): the parsers take a path and read
it themselves, so this is the only module besides `home.py` where blocking file IO is allowed.
Its own writes go through `home.atomic_write_sync` for the same reason. Calling any of these from
a coroutine blocks that event loop.
"""

import io
import re
from enum import StrEnum
from pathlib import Path

import msgspec
import pyromark

from haskie import home
from haskie.errors import PermanentError
from haskie.settings import Parser

PREVIEW_PAGES = 10

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".csv", ".json"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
HTML_SUFFIXES = {".html", ".htm"}
ANYDOC_SUFFIXES = {
    ".pdf", ".doc", ".docx", ".docm", ".ppt", ".pps", ".pot", ".pptx", ".pptm", ".ppsx", ".ppsm",
    ".xls", ".xlsx", ".xlsm", ".xlsb", ".odt", ".ods", ".odp", ".rtf", ".epub",
}  # fmt: skip
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | HTML_SUFFIXES | IMAGE_SUFFIXES | ANYDOC_SUFFIXES

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


def page_marker(page: int, skipped: bool = False) -> str:
    """The marker ahead of one 1-based page; `skipped` notes a page left out for needing OCR."""
    return f"<!-- page {page}: needs OCR, skipped -->" if skipped else f"<!-- page {page} -->"


class PreviewKind(StrEnum):
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    HTML = "html"


class Preview(msgspec.Struct):
    kind: PreviewKind
    truncated: bool = False
    pages: int | None = None
    ocr_pages: list[int] = []  # 1-based pages with no extractable text (within the preview)


def _conversion_error(path: Path, exc: Exception) -> PermanentError:
    """A parser failure is a property of the file, not of the moment: retrying cannot fix it."""
    return PermanentError(home.scrub(f"could not convert {path.name}: {type(exc).__name__}: {exc}"))


def to_markdown(path: Path, parser: Parser) -> str:
    """Whole-file conversion, for everything except PDF: a PDF is converted page-wise, in batches
    (`pdf_pages_markdown`), and its OCR policy is applied over the whole document afterwards."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return ""  # no extractable text without OCR; the preview shows the image itself
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


def pdf_page_count(path: Path) -> int:
    from pypdf import PdfReader

    try:
        return len(PdfReader(str(path)).pages)
    except Exception as exc:
        raise _conversion_error(path, exc) from exc


def check_ocr_policy(ocr_pages: int, total_pages: int, skip_ocr_pages: bool) -> None:
    """A document with no extractable text always fails; otherwise OCR pages fail unless skipped."""
    if total_pages and ocr_pages == total_pages:
        raise PermanentError(f"all {total_pages} pages need OCR")
    if ocr_pages and not skip_ocr_pages:
        raise PermanentError(f"{ocr_pages} of {total_pages} pages need OCR (enable skip_ocr_pages)")


def pdf_pages_markdown(
    path: Path, pages: list[int] | None = None, skip_ocr_pages: bool = False
) -> tuple[str, list[int], int]:
    """Per-page markdown joined with page markers; returns (markdown, 1-based pages needing OCR,
    pages in the file).

    With skip_ocr_pages those pages become a marker comment; otherwise their (empty) text stays.
    Policy decisions (fail or not) belong to check_ocr_policy over the whole document.
    """
    import pdf_inspector

    try:
        result = pdf_inspector.extract_pages_markdown(str(path), pages=pages)
    except Exception as exc:
        raise _conversion_error(path, exc) from exc
    ocr_pages = [p.page + 1 for p in result.pages if p.needs_ocr]
    parts = [
        page_marker(p.page + 1, skipped=True)
        if p.needs_ocr and skip_ocr_pages
        else f"{page_marker(p.page + 1)}\n\n{p.markdown}"
        for p in result.pages
    ]
    return "\n\n".join(parts), ocr_pages, len(result.pages)


def build_preview(
    source: Path, out_dir: Path, parser: Parser, skip_ocr_pages: bool = False
) -> Preview:
    """Write `out_dir/source` (left pane) and `out_dir/preview.md` (right pane).

    PDFs only parse the first PREVIEW_PAGES pages; other formats convert whole (milliseconds).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return _pdf_preview(source, out_dir, skip_ocr_pages)
    if suffix in IMAGE_SUFFIXES:
        # before to_markdown: an image has no text, so the right pane stays empty
        home.atomic_write_sync(out_dir / "preview.md", "")
        home.atomic_write_sync(out_dir / "source", source.read_bytes())
        return Preview(kind=PreviewKind.IMAGE)
    full_markdown = to_markdown(source, parser)
    home.atomic_write_sync(out_dir / "preview.md", full_markdown)
    if suffix in TEXT_SUFFIXES:
        home.atomic_write_sync(out_dir / "source", source.read_bytes())
        return Preview(kind=PreviewKind.TEXT)
    # office/epub/rtf/odt: browsers cannot render these; show the markdown as HTML instead
    html = (
        source.read_text(errors="replace")
        if suffix in HTML_SUFFIXES
        else pyromark.html(full_markdown)
    )
    home.atomic_write_sync(out_dir / "source", html)
    return Preview(kind=PreviewKind.HTML)


def _pdf_preview(source: Path, out_dir: Path, skip_ocr_pages: bool) -> Preview:
    from pypdf import PdfReader, PdfWriter

    try:
        reader = PdfReader(str(source))
        total = len(reader.pages)
    except Exception as exc:
        raise _conversion_error(source, exc) from exc
    shown = min(total, PREVIEW_PAGES)
    writer = PdfWriter()
    for page in reader.pages[:shown]:
        writer.add_page(page)
    buffer = io.BytesIO()
    writer.write(buffer)
    home.atomic_write_sync(out_dir / "source", buffer.getvalue())

    markdown, ocr_pages, _ = pdf_pages_markdown(source, list(range(shown)), skip_ocr_pages)
    home.atomic_write_sync(out_dir / "preview.md", markdown)
    return Preview(kind=PreviewKind.PDF, truncated=total > shown, pages=shown, ocr_pages=ocr_pages)
