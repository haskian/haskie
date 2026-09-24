"""The window a passage is widened in: what `retrieval` reads off disk for one span.

A chunk row carries byte offsets, so widening reads a couple of kilobytes around the span instead
of the whole document. These check that the slice is the right one, that it starts on a character
boundary whatever byte the seek landed on, and that a passage built from it is the passage the
whole document would have given.
"""

from pathlib import Path

import pytest

from haskie.collection.index import Hit, location
from haskie.search import retrieval
from haskie.search.passage import Passage, Window, ranges, widen

# Multi-byte on purpose: a char offset is not a file position, and a window that started half a
# character in would shift every offset it reports.
ASCII = "# Retries\n\n" + "\n\n".join(
    f"Paragraph {i} about retrying a failed call." for i in range(40)
)
WIDE = "# Wiederholungen\n\n" + "\n\n".join(
    f"Абзац {i} — über Wiederholungen 🌍 und Zustellung." for i in range(40)
)


def _hit(markdown: str, path: Path, snippet: str) -> Hit:
    """One indexed chunk over `snippet`, with the offsets the index would have stored for it."""
    assert markdown.count(snippet) == 1, f"not unique in the fixture: {snippet!r}"
    char_start = markdown.index(snippet)
    char_end = char_start + len(snippet)
    line_start = markdown.count("\n", 0, char_start) + 1
    line_end = markdown.count("\n", 0, char_end - 1) + 1
    return Hit(
        collection="backend",
        document="doc.md",
        source_path="documents/doc.md",
        markdown_path="documents/doc.md",
        part=0,
        seq=1,
        line_start=line_start,
        line_end=line_end,
        char_start=char_start,
        char_end=char_end,
        byte_start=len(markdown[:char_start].encode()),
        byte_end=len(markdown[:char_end].encode()),
        page_start=None,
        page_end=None,
        headings=["Retries"],
        frame=["Retries"],
        header="Retries",
        location=location("doc.md", None, None, line_start, line_end),
        text=snippet,
        score=1.0,
        source_file=str(path),
        markdown_file=str(path),
    )


@pytest.mark.parametrize(
    ("name", "markdown", "snippet"),
    [
        ("ascii, far enough in to be windowed on both sides", ASCII, "Paragraph 20 about"),
        ("multi-byte, so the seek lands mid-character", WIDE, "Абзац 20 — über"),
        ("the first chunk: the window starts at the file", WIDE, "# Wiederholungen"),
        ("the last chunk: the read stops at the end of the file", WIDE, "Абзац 39 — über"),
        ("a span whose own text is multi-byte", WIDE, "Абзац 7 — über Wiederholungen 🌍"),
    ],
)
def test_a_window_is_the_document_around_one_span(
    name: str, markdown: str, snippet: str, tmp_path: Path
) -> None:
    path = tmp_path / "doc.md"
    path.write_text(markdown, encoding="utf-8")
    (span,) = ranges([_hit(markdown, path, snippet)])

    (window,) = retrieval._read_windows([span])

    assert window.text in markdown, f"{name}: a slice of the document, decoded whole"
    start = window.local(span.char_start)
    assert window.text[start : window.local(span.char_end)] == snippet, f"{name}: the span itself"
    assert markdown[window.char_start : window.char_start + len(window.text)] == window.text, name
    assert widen(span, window, Passage) == widen(span, Window(markdown, 0), Passage), name


def test_a_window_reads_a_window_and_not_the_file(tmp_path: Path) -> None:
    """The point of the byte offsets: a book stays on disk while one paragraph is quoted."""
    markdown = ASCII + "filler paragraph.\n\n" * 5000
    path = tmp_path / "big.md"
    path.write_text(markdown, encoding="utf-8")
    (span,) = ranges([_hit(markdown, path, "Paragraph 20 about")])

    (window,) = retrieval._read_windows([span])

    assert len(window.text) <= 2 * retrieval.WINDOW_BYTES + len("Paragraph 20 about")
    assert len(markdown) > 10 * len(window.text), "the document is far larger than what was read"


@pytest.mark.anyio
async def test_the_windows_of_a_search_come_back_in_order(tmp_path: Path) -> None:
    """A search folds spans of several documents and expands them in rank order, so the windows
    have to line up with the spans they were read for - not with the files they came from."""
    first, second = tmp_path / "one.md", tmp_path / "two.md"
    first.write_text(ASCII, encoding="utf-8")
    second.write_text(WIDE, encoding="utf-8")
    spans = [
        ranges([_hit(ASCII, first, "Paragraph 20 about")])[0],
        ranges([_hit(WIDE, second, "Абзац 20 — über")])[0],
        ranges([_hit(ASCII, first, "Paragraph 31 about")])[0],
    ]

    windows = await retrieval._windows_of(spans)

    assert len(windows) == len(spans)
    for span, window in zip(spans, windows, strict=True):
        start = window.local(span.char_start)
        assert window.text[start : window.local(span.char_end)] == span.chunks[0].text
