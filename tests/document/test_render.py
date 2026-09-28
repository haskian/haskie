"""Markdown to HTML for the viewer.

The browser inserts what `render` returns, so the security property is the point: nothing a
document carries may end up as script on the page.
"""

import re

import pytest

from haskie.document import render

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ("name", "markdown", "gone"),
    [
        ("script block", "# T\n\n<script>alert(1)</script>\n", "alert"),
        ("event handler attribute", "# T\n\n<img src=x onerror=alert(1)>\n", "onerror"),
        ("inline html", "# T\n\ntext <b onclick=evil()>bold</b> more\n", "onclick"),
        ("iframe", "# T\n\n<iframe src=//evil></iframe>\n", "iframe"),
        ("svg with script", "# T\n\n<svg><script>alert(1)</script></svg>\n", "svg"),
        (
            "after non-ascii text",
            "# T\n\n" + "é" * 40 + " <img src=x onerror=alert(1)>\n",
            "onerror",
        ),
        ("block after non-ascii heading", "# Čšž ☃\n\n<script>alert(1)</script>\n", "alert"),
    ],
)
def test_raw_html_never_reaches_the_page(name: str, markdown: str, gone: str) -> None:
    """`pyromark.html` passes raw HTML straight through, so it is removed before rendering.

    Removed, not escaped: that is what the viewer did when React rendered the markdown, so no
    document that used to display starts showing its own tags.
    """
    html, _ = render.to_html(markdown)

    assert gone not in html, name
    assert "<h1" in html, f"{name}: the markdown around it still renders"


def test_markdown_still_renders() -> None:
    """Dropping raw HTML must not cost the markdown itself."""
    html, _ = render.to_html("# Title\n\nsome *emphasis* and `code`\n\n- one\n- two\n")

    assert "<em>emphasis</em>" in html
    assert "<code>code</code>" in html
    assert "<li>one</li>" in html


def test_headings_carry_the_id_their_toc_entry_links_to() -> None:
    """The anchors are positional, so the nth heading is `h-{n}` for the nth table entry."""
    markdown = "# One\n\na\n\n## Two\n\nb\n\n### Three\n\nc\n"

    html, numbered = render.to_html(markdown)

    assert '<h1 id="h-0">' in html and '<h2 id="h-1">' in html and '<h3 id="h-2">' in html
    assert numbered == len(render.headings(markdown)) == 3, "one id per table of contents entry"


def test_headings_carry_their_level_text_and_byte_offset() -> None:
    markdown = "# Title\n\nintro\n\n## Alpha\n\nbody\n\n## Beta\n\nbody\n"

    result = render.headings(markdown)

    assert [(h.level, h.text) for h in result] == [(1, "Title"), (2, "Alpha"), (2, "Beta")]
    assert markdown.encode()[result[1].offset :].startswith(b"## Alpha"), "a byte offset"


def test_pages_split_on_the_markers_and_number_anchors_across_the_document() -> None:
    """A page is rendered alone but its anchors have to stay unique in the whole document."""
    markdown = "<!-- page 1 -->\n\n# One\n\na\n\n<!-- page 2 -->\n\n## Two\n\nb\n"

    pages, toc = render.pages(markdown)

    assert [p.number for p in pages] == [1, 2]
    assert '<h1 id="h-0">' in pages[0].html
    assert '<h2 id="h-1">' in pages[1].html, "numbering continues rather than restarting"
    assert "page 1" not in pages[0].html, "the marker is a comment, not content"
    assert [(h.level, h.text) for h in toc] == [(1, "One"), (2, "Two")], "one entry per anchor"


def test_a_document_without_markers_is_one_page() -> None:
    """Markdown and office files are not paginated; the viewer shows them as one body."""
    pages, _ = render.pages("# Only\n\nbody\n")

    assert len(pages) == 1
    assert pages[0].number is None
    assert "<h1" in pages[0].html


def test_page_markers_that_note_skipped_ocr_still_split() -> None:
    """`convert` writes a longer marker for a page it could not read."""
    markdown = (
        "<!-- page 1 -->\n\na\n\n<!-- page 2: needs OCR, skipped -->\n\n<!-- page 3 -->\n\nc\n"
    )

    assert [p.number for p in render.pages(markdown)[0]] == [1, 2, 3]


def test_inline_raw_html_loses_its_tags_but_keeps_its_text() -> None:
    """Block and inline raw HTML differ, and the difference is worth stating.

    A raw HTML *block* goes entirely — tags and the text between them. Inline raw HTML is only the
    tags, so `<script>alert(1)</script>` inside a paragraph leaves the literal `alert(1)` as prose.
    That is inert and it is what React did before, but it means "the word alert survived" is not
    the test; "nothing executable survived" is.
    """
    html, _ = render.to_html("para with <script>alert(1)</script> inside\n")

    assert "<script" not in html and "</script" not in html
    assert html.strip() == "<p>para with alert(1) inside</p>", "the tags are gone, the text is text"


@pytest.mark.parametrize(
    ("name", "markdown"),
    [
        ("script block", "<script>alert(1)</script>"),
        ("inline script", "para <script>alert(1)</script> end"),
        ("event handler", "<img src=x onerror=alert(1)>"),
        ("inline handler", "text <b onclick=evil()>b</b>"),
        ("iframe", "<iframe src=//evil></iframe>"),
        ("handler after multi-byte text", "日本語 " * 20 + "<img src=x onerror=alert(1)>"),
        ("script after emoji", "🙂🙂 para <script>alert(1)</script> end"),
    ],
)
def test_nothing_executable_survives(name: str, markdown: str) -> None:
    """The property that matters: no tag that runs, and no event-handler attribute."""
    html, _ = render.to_html(markdown)

    assert not re.search(r"<\s*(script|iframe|object|embed|svg)", html, re.I), name
    assert not re.search(r"\son[a-z]+\s*=", html, re.I), name
