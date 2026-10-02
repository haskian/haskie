"""Markdown to HTML for the viewer, one page at a time, and the table of contents beside it.

The browser inserts what this returns, so it must not carry script. `pyromark.html` passes raw
HTML straight through (a `<script>` in an uploaded markdown file would reach the page verbatim),
so the raw HTML is removed before rendering, using the parser's own idea of what is raw HTML
rather than a pattern of our own.

That also matches what the viewer did when React rendered the markdown: `react-markdown` ignores
raw HTML unless asked for it, so nothing that used to appear stops appearing.
"""

import itertools
import re
from typing import Literal

import msgspec
import pyromark

from haskie.document.convert import PAGE_MARKER

HEADING_OPEN = re.compile(r"<h([1-6])>")

OPTIONS = pyromark.Options.ENABLE_TABLES | pyromark.Options.ENABLE_STRIKETHROUGH


class Heading(msgspec.Struct):
    level: int
    text: str
    offset: int  # byte offset of the heading in the markdown


class Page(msgspec.Struct):
    """One page of a document, rendered and ready to insert."""

    number: int | None  # the PDF page; None for a document that has no pages
    html: str
    kind: Literal["page"] = "page"  # tags this line of the NDJSON stream (see `api.documents`)


class HeadingSpan(msgspec.Struct, frozen=True):
    """Where one heading of the markdown is, in bytes: the whole of it, its markers included,
    and the text inside them."""

    level: int
    start: int
    end: int
    text_start: int
    text_end: int
    text: str


def heading_spans(markdown: str) -> list[HeadingSpan]:
    """Every heading of the markdown, in document order, by the parser and options the chunker
    reads it with (`segment`): a `#` line inside a code block is none."""
    found: list[HeadingSpan] = []
    level = start = end = 0
    inner: list[tuple[int, int]] | None = None
    texts: list[str] = []
    for event, span in pyromark.events_with_range(markdown, options=OPTIONS):
        match event:
            case {"Start": {"Heading": {"level": depth}}}:
                level, start, end, inner, texts = (
                    int(str(depth)[1]),
                    span["start"],
                    span["end"],
                    [],
                    [],
                )
            case {"End": {"Heading": _}} if inner is not None:
                text_start = min((one for one, _ in inner), default=start)
                text_end = max((one for _, one in inner), default=start)
                found.append(
                    HeadingSpan(level, start, end, text_start, text_end, "".join(texts).strip())
                )
                inner = None
            case _ if inner is not None:
                inner.append((span["start"], span["end"]))
                if isinstance(event, dict) and ("Text" in event or "Code" in event):
                    texts.append(str(event.get("Text", event.get("Code"))))
    return found


def headings(markdown: str) -> list[Heading]:
    """The table of contents, in document order."""
    return [
        Heading(level=one.level, text=one.text, offset=one.start) for one in heading_spans(markdown)
    ]


def _without_raw_html(markdown: str) -> str:
    """The markdown with every raw HTML span cut out.

    The spans come from `events_with_range`, so "raw HTML" means whatever the CommonMark parser
    calls raw HTML, not whatever a regular expression of ours would match. They are UTF-8 byte
    offsets, so the cut is made on the bytes: on the `str`, every non-ASCII character before a
    span would shift it, and the tag would survive. Cut back to front so each removal leaves the
    earlier offsets alone.
    """
    spans = [
        (span["start"], span["end"])
        for event, span in pyromark.events_with_range(markdown)
        if isinstance(event, dict) and ("Html" in event or "InlineHtml" in event)
    ]
    out = markdown.encode()
    for start, end in sorted(set(spans), reverse=True):
        out = out[:start] + out[end:]
    return out.decode()


def fragment_html(markdown: str) -> str:
    """A piece of markdown as HTML, raw HTML stripped as for a page (`_without_raw_html`), and its
    headings without ids: a quoted excerpt sits beside the whole document, whose anchors it must
    not repeat."""
    return pyromark.html(_without_raw_html(markdown), options=OPTIONS)


def to_html(markdown: str, first_heading: int = 0) -> tuple[str, int]:
    """Render one page of markdown, giving each heading the id its table of contents links to.
    Returns the HTML and how many headings it numbered.

    `first_heading` is how many headings the document has already rendered, because a page is
    rendered on its own but its anchors have to be unique across the whole document. Raw HTML is
    gone by the time this matches `<h1>`..`<h6>`, so the nth opening tag is the nth heading.
    """
    html = pyromark.html(_without_raw_html(markdown), options=OPTIONS)
    counter = itertools.count(first_heading)
    return HEADING_OPEN.subn(lambda m: f'<h{m.group(1)} id="h-{next(counter)}">', html)


def split_pages(markdown: str) -> list[tuple[int | None, str]]:
    """The document cut at its page markers, as (page number, markdown) in order.

    A document with no markers is one page numbered None: markdown and office files are not
    paginated, and the viewer shows them as one body.
    """
    markers = list(PAGE_MARKER.finditer(markdown))
    if not markers:
        return [(None, markdown)]
    bounds = [m.start() for m in markers] + [len(markdown)]
    return [
        (int(marker.group(1)), markdown[start:end])
        for marker, start, end in zip(markers, bounds[:-1], bounds[1:], strict=True)
    ]


def pages(markdown: str) -> tuple[list[Page], list[Heading]]:
    """Every page of a document rendered with anchors numbered across the whole document, plus
    that document's table of contents.

    Both in one call because both are CPU work over the same text, and the caller pays for one
    trip through the CPU budget rather than parsing the whole document again on its event loop.
    """
    rendered: list[Page] = []
    seen = 0
    for number, body in split_pages(markdown):
        html, numbered = to_html(body, first_heading=seen)
        rendered.append(Page(number=number, html=html))
        seen += numbered
    return rendered, headings(markdown)
