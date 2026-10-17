"""The source books as text: their hash, their pages, the segments generation reads one at a time,
and where a quote sits in them.

A PDF's text is pypdf's, page by page, with 1-based page numbers - the physical pages, which is
what haskie's `page_start` counts too. Other formats have no pages: HTML is its visible text,
markdown and plain text are read as they are. Generation shows Claude exactly this text, so a
quote it copies can be found again here, whatever the converter haskie indexes with makes of it.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import cache
from html.parser import HTMLParser
from pathlib import Path

MAX_PAGES = 12  # pages per segment: a chapter longer than this is read in several
MAX_CHARS = 40_000  # characters per segment of a source without pages
MIN_CHARS = 1_500  # a segment with less text than this (a cover, a blank run) is skipped
PAGE = "[page {}]"  # the marker ahead of each page in a segment's text

# pypdf warns on every font it can't fully parse without fontTools; the text comes out regardless
logging.getLogger("pypdf").setLevel(logging.ERROR)


@dataclass(frozen=True)
class Segment:
    label: str  # stable within one source: "p011-018" or "part03"
    first_page: int | None  # 1-based, inclusive; None for a source without pages
    last_page: int | None
    text: str  # what generation shows Claude, pages marked with `PAGE`


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_paged(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def compact(text: str) -> str:
    """`text` as its letters and digits alone, lowercased: what two copies of one passage still
    share after extraction differs in whitespace, line-end hyphenation, ligatures and quotes."""
    return "".join(re.findall(r"\w", unicodedata.normalize("NFKC", text).lower()))


def pages(path: Path) -> list[str]:
    """A PDF's text, one string per page."""
    return list(_pages(path.resolve(), path.stat().st_mtime_ns))


@cache
def _pages(path: Path, _mtime: int) -> tuple[str, ...]:
    from pypdf import PdfReader

    return tuple(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def text(path: Path) -> str:
    """A source without pages, as text: HTML's visible text, anything else as it is."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() not in (".html", ".htm"):
        return raw
    visible = _Visible()
    visible.feed(raw)
    return re.sub(r"\n\s*\n+", "\n\n", "".join(visible.parts)).strip()


class _Visible(HTMLParser):
    BLOCKS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "tr", "br", "section"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self.hidden += tag in ("script", "style")
        if tag in self.BLOCKS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        self.hidden -= tag in ("script", "style") and self.hidden > 0

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def segments(path: Path) -> list[Segment]:
    """The source cut into segments small enough for one generation each, in document order: a
    PDF by its top-level outline (its chapters) where it has one, each chapter cut into runs of at
    most `MAX_PAGES` pages; anything else into runs of whole paragraphs up to `MAX_CHARS`."""
    found = _paged_segments(path) if is_paged(path) else _text_segments(text(path))
    return [s for s in found if len(s.text) >= MIN_CHARS]


def _paged_segments(path: Path) -> list[Segment]:
    texts = pages(path)
    total = len(texts)
    starts = sorted({1, *(p for p in _chapter_starts(path) if 1 <= p <= total)})
    bounds = [*starts[1:], total + 1]
    found = []
    for start, stop in zip(starts, bounds, strict=True):
        for first in range(start, stop, MAX_PAGES):
            last = min(first + MAX_PAGES, stop) - 1
            body = "\n\n".join(f"{PAGE.format(p)}\n{texts[p - 1]}" for p in range(first, last + 1))
            found.append(Segment(f"p{first:03d}-{last:03d}", first, last, body))
    return found


def _chapter_starts(path: Path) -> list[int]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    try:
        found = [
            reader.get_destination_page_number(entry)
            for entry in reader.outline
            if not isinstance(entry, list)
        ]
    except Exception:  # a broken outline is no outline: the pages are still there
        return []
    return [page + 1 for page in found if page is not None]


def _text_segments(body: str) -> list[Segment]:
    found: list[Segment] = []
    current: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body):
        if current and sum(map(len, current)) + len(paragraph) > MAX_CHARS:
            found.append(Segment(f"part{len(found) + 1:02d}", None, None, "\n\n".join(current)))
            current = []
        current.append(paragraph)
    if current:
        found.append(Segment(f"part{len(found) + 1:02d}", None, None, "\n\n".join(current)))
    return found


def quote_pages(path: Path, quote: str) -> list[int]:
    """The 1-based pages a quote starts on in a PDF - it may run on into the next page. For a
    source without pages, `[0]` if the quote is anywhere in it, else `[]`."""
    wanted = compact(quote)
    if not wanted:
        return []
    if not is_paged(path):
        return [0] if wanted in compact(text(path)) else []
    texts = [compact(t) for t in pages(path)]
    found = []
    for number, here in enumerate(texts, start=1):
        after = texts[number] if number < len(texts) else ""
        # on this page, or begun here and run on - not merely whole on the next one
        if wanted in here or (wanted in here + after and wanted not in after):
            found.append(number)
    return found
