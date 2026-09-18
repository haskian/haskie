"""Markdown to HTML for the viewer, one page at a time.

The browser inserts what this returns, so the one thing that matters here is that it cannot carry
script. `pyromark.html` passes raw HTML straight through — a `<script>` in an uploaded markdown
file would reach the page verbatim — so the raw HTML is removed before rendering, using the
parser's own idea of what is raw HTML rather than a pattern of our own.

That also matches what the viewer did when React rendered the markdown: `react-markdown` ignores
raw HTML unless asked for it, so nothing that used to appear stops appearing.
"""

import re

import msgspec
import pyromark

from haskie import toc

# Written by `convert.pdf_pages_markdown` ahead of every page, and the only reason this module
# knows about pages at all.
PAGE_MARKER = re.compile(r"<!-- page (\d+)[^>]*-->")
HEADING_OPEN = re.compile(r"<h([1-6])>")

OPTIONS = pyromark.Options.ENABLE_TABLES | pyromark.Options.ENABLE_STRIKETHROUGH


class Page(msgspec.Struct):
    """One page of a document, rendered and ready to insert."""

    number: int | None  # the PDF page; None for a document that has no pages
    html: str


def _without_raw_html(markdown: str) -> str:
    """The markdown with every raw HTML span cut out.

    The spans come from `events_with_range`, so "raw HTML" means whatever the CommonMark parser
    calls raw HTML, not whatever a regular expression of ours would match. Cut back to front so
    each removal leaves the earlier offsets alone.
    """
    spans = [
        (span["start"], span["end"])
        for event, span in pyromark.events_with_range(markdown)
        if isinstance(event, dict) and ("Html" in event or "InlineHtml" in event)
    ]
    out = markdown
    for start, end in sorted(set(spans), reverse=True):
        out = out[:start] + out[end:]
    return out


def to_html(markdown: str, first_heading: int = 0) -> str:
    """Render one chunk of markdown, giving each heading the id its table of contents links to.

    `first_heading` is how many headings the document has already rendered, because a page is
    rendered on its own but its anchors have to be unique across the whole document. Raw HTML is
    gone by the time this matches `<h1>`..`<h6>`, so the nth opening tag really is the nth heading.
    """
    html = pyromark.html(_without_raw_html(markdown), options=OPTIONS)
    counter = iter(range(first_heading, first_heading + 10_000))
    return HEADING_OPEN.sub(lambda m: f'<h{m.group(1)} id="h-{next(counter)}">', html)


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


def pages(markdown: str) -> list[Page]:
    """Every page of a document, rendered, with anchors numbered across the whole document."""
    rendered: list[Page] = []
    seen = 0
    for number, body in split_pages(markdown):
        rendered.append(Page(number=number, html=to_html(body, first_heading=seen)))
        seen += len(toc.headings(body))
    return rendered
