"""A PDF's bookmarks setting its headings: which heading a bookmark claims, what becomes text,
and when the converter's own headings stand.

The pages below are what `pdf_inspector` wrote for a typeset book (Implementing Domain-Driven
Design): a running header as a heading of its own, a chapter label and its title at two levels
the table of contents does not have, and letter-spaced small caps.
"""

from pathlib import Path

import pytest
from conftest import text_pdf
from pypdf import PdfWriter

from haskie.document import bookmarks, convert
from haskie.document.bookmarks import Bookmark
from haskie.settings import Parser

CHAPTER = """## <u>Chapter 10</u>

# Aggregates

Clustering Entities and Value Objects into an Aggregate is one of the least understood rules.

## 348 Chapter 10 AGGREGATES

# Using Aggregates in the Scrum Core Domain

We model the Scrum Core Domain with Aggregates.
"""
MARKS = bookmarks.pages(
    [
        Bookmark(level=1, title="Aggregates in C#", page=1),
        Bookmark(level=1, title="10 Aggregates", page=390),
        Bookmark(level=2, title="Using Aggregates in the Scrum Core Domain", page=391),
        Bookmark(level=3, title="First Attempt: Large-Cluster Aggregate", page=392),
        Bookmark(level=2, title="Rule: Design Small Aggregates", page=398),
    ]
)


def _levels(markdown: str) -> list[tuple[int, str]]:
    return [(len(line) - len(line.lstrip("#")), line.lstrip("# ")) for line in _lines(markdown)]


def _lines(markdown: str) -> list[str]:
    return [line for line in markdown.splitlines() if line.startswith("#")]


@pytest.mark.parametrize(
    ("name", "markdown", "page", "expected"),
    [
        (
            "the chapter takes its bookmark's level; its label and a running header become text",
            CHAPTER,
            390,
            [(1, "Aggregates")],
        ),
        (
            "a bookmark of the page before claims a heading at the top of this one",
            "# Using Aggregates in the Scrum Core Domain\n\nText.\n",
            392,
            [(2, "Using Aggregates in the Scrum Core Domain")],
        ),
        (
            "a bookmark two pages back claims nothing",
            "# Using Aggregates in the Scrum Core Domain\n\nText.\n",
            393,
            [],
        ),
        (
            "a running header adds its page number: the section heading claims the bookmark",
            "## R ULE: DESIGN SMALL AGGREGATES 355\n\n# Rule: Design Small Aggregates\n\nText.\n",
            398,
            [(2, "Rule: Design Small Aggregates")],
        ),
        (
            "a running header alone claims no bookmark",
            "## R ULE: DESIGN SMALL AGGREGATES 355\n\nText.\n",
            398,
            [],
        ),
        (
            "a closing sequence goes",
            "### Rule: Design Small Aggregates ##\n",
            398,
            [(2, "Rule: Design Small Aggregates")],
        ),
        (
            "a # that is part of the title stays",
            "## Aggregates in C#\n",
            1,
            [(1, "Aggregates in C#")],
        ),
        (
            "a setext heading is set too",
            "Rule: Design Small Aggregates\n=============================\n",
            398,
            [(2, "Rule: Design Small Aggregates")],
        ),
    ],
)
def test_apply(name: str, markdown: str, page: int, expected: list[tuple[int, str]]) -> None:
    assert _levels(bookmarks.apply(markdown, page, MARKS, set())) == expected, name


def test_apply_leaves_a_hash_line_in_a_code_block_alone() -> None:
    """The chunker reads it as code (`segment`), so it is no heading to set."""
    markdown = "```\n# Rule: Design Small Aggregates\n```\n"

    assert bookmarks.apply(markdown, 398, MARKS, set()) == markdown


def test_apply_keeps_the_text_of_every_heading_it_demotes() -> None:
    found = bookmarks.apply(CHAPTER, 390, MARKS, set())

    assert "\n<u>Chapter 10</u>\n" in f"\n{found}"
    assert "\n348 Chapter 10 AGGREGATES\n" in found
    assert found.count("Clustering Entities") == 1, "the body is untouched"


def test_apply_leaves_a_page_without_headings_as_it_was() -> None:
    assert bookmarks.apply("Just text.\n", 390, MARKS, set()) == "Just text.\n"


def test_apply_keeps_inline_markdown_and_multibyte_text() -> None:
    marks = bookmarks.pages([Bookmark(level=1, title="Don’t Trust Every Use Case", page=1)])

    found = bookmarks.apply("## **Don’t** Trust Every Use Case\n\nÜber.\n", 1, marks, set())

    assert found == "# **Don’t** Trust Every Use Case\n\nÜber.\n"


def test_a_bookmark_claims_one_heading_once() -> None:
    """A bookmark matched on its own page is claimed: a running header on the next page that
    repeats the title bare (the page number in the footer) is text."""
    claimed: set[Bookmark] = set()

    first = bookmarks.apply(CHAPTER, 390, MARKS, claimed)
    second = bookmarks.apply("## AGGREGATES\n\nMore text.\n", 391, MARKS, claimed)

    assert _levels(first)[0] == (1, "Aggregates")
    assert second == "AGGREGATES\n\nMore text.\n"
    assert Bookmark(level=1, title="10 Aggregates", page=390) in claimed


@pytest.mark.parametrize(
    ("name", "markdown", "expected"),
    [
        (
            "a setext heading of two lines keeps both as one heading line",
            "Using Aggregates in the\nScrum Core Domain\n===\n\nText.\n",
            "## Using Aggregates in the Scrum Core Domain\n\nText.\n",
        ),
        (
            "a heading on a page's last line adds no line break",
            "Text.\n\n## 348 Chapter 10 AGGREGATES",
            "Text.\n\n348 Chapter 10 AGGREGATES",
        ),
    ],
)
def test_apply_rewrites_only_the_markers(name: str, markdown: str, expected: str) -> None:
    assert bookmarks.apply(markdown, 391, MARKS, set()) == expected, name


def test_a_bookmark_deeper_than_markdown_sets_the_deepest_heading() -> None:
    marks = bookmarks.pages([Bookmark(level=8, title="Deep", page=1)])

    assert bookmarks.apply("# Deep\n", 1, marks, set()) == "###### Deep\n"


# --- through the conversion -----------------------------------------------------------------

BODY = "A saga runs a compensating step for every step already done when a later one fails."
BOOK: list[str | None | list[tuple[int, str]]] = [
    [(10, "12 Chapter 1 SAGAS"), (24, "Sagas"), *[(10, BODY)] * 6],
    [(18, "Orchestration"), *[(10, BODY)] * 6],
]


def _book(tmp_path: Path, outline: list[tuple[str, int, int]]) -> Path:
    """`BOOK` with bookmarks of (title, 0-based page, parent position or -1)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    plain = tmp_path / "plain.pdf"
    plain.write_bytes(text_pdf(BOOK))
    writer = PdfWriter(clone_from=str(plain))
    added = []
    for title, page, parent in outline:
        added.append(
            writer.add_outline_item(title, page, parent=added[parent] if parent >= 0 else None)
        )
    path = tmp_path / "book.pdf"
    writer.write(path)
    return path


@pytest.mark.parametrize(
    ("name", "outline", "expected"),
    [
        (
            "nested bookmarks set the levels",
            [("1 Sagas", 0, -1), ("Orchestration", 1, 0)],
            ["# Sagas", "## Orchestration"],
        ),
        (
            "no bookmarks: the converter's headings stand",
            [],
            ["# Sagas", "# Orchestration"],
        ),
        (
            "one level of bookmarks: the converter's headings stand",
            [("1 Sagas", 0, -1), ("Orchestration", 1, -1)],
            ["# Sagas", "# Orchestration"],
        ),
    ],
)
def test_the_conversion_routes_by_the_bookmarks(
    tmp_path: Path, name: str, outline: list[tuple[str, int, int]], expected: list[str]
) -> None:
    path = _book(tmp_path, outline)
    total, marks = convert.pdf_outline(path)

    markdown, ocr_pages, count = convert.pdf_pages_markdown(path, marks=marks)

    assert (total, ocr_pages, count) == (2, [], 2), name
    assert _lines(markdown) == expected, name
    assert "12 Chapter 1 SAGAS" in markdown, f"{name}: the running header's text stays"


def test_a_batch_learns_what_the_page_before_it_claimed(tmp_path: Path) -> None:
    """Page 2 repeats page 1's title at its top, as a running header does when the page number
    sits in the footer. A batch of page 2 alone converts page 1 too, only to learn that page 1
    claimed the bookmark: page 2's repeat is text."""
    plain = tmp_path / "plain.pdf"
    plain.write_bytes(text_pdf([[(24, "Sagas"), *[(10, BODY)] * 6]] * 2))
    writer = PdfWriter(clone_from=str(plain))
    writer.add_outline_item("Other", 1, parent=writer.add_outline_item("Sagas", 0))
    path = tmp_path / "book.pdf"
    writer.write(path)
    _, marks = convert.pdf_outline(path)

    first, _, _ = convert.pdf_pages_markdown(path, [0], marks=marks)
    second, _, count = convert.pdf_pages_markdown(path, [1], marks=marks)

    assert _lines(first) == ["# Sagas"]
    assert count == 1 and "<!-- page 1 -->" not in second, "the page before is not written"
    assert _lines(second) == [] and "Sagas" in second, "the repeat stays, as text"


def test_a_batch_without_bookmarks_of_its_own_makes_its_headings_text(tmp_path: Path) -> None:
    """The route is the document's: a routed batch whose pages hold no bookmark (`[]`) sets no
    heading, while a document without bookmarks (None) keeps the converter's."""
    path = _book(tmp_path, [])

    routed, _, _ = convert.pdf_pages_markdown(path, [1], marks=[])
    kept, _, _ = convert.pdf_pages_markdown(path, [1], marks=None)

    assert _lines(routed) == [] and "Orchestration" in routed
    assert _lines(kept) == ["# Orchestration"]


def test_pdf_outline_counts_the_pages_and_reads_the_nested_bookmarks(tmp_path: Path) -> None:
    path = _book(tmp_path, [("1 Sagas", 0, -1), ("Orchestration", 1, 0)])
    flat = _book(tmp_path / "flat", [("1 Sagas", 0, -1), ("Orchestration", 1, -1)])

    assert convert.pdf_outline(flat) == (2, None), "one level sets no heading"
    assert convert.pdf_outline(path) == (
        2,
        [
            Bookmark(level=1, title="1 Sagas", page=1),
            Bookmark(level=2, title="Orchestration", page=2),
        ],
    )


def test_an_outline_that_cannot_be_read_is_no_bookmarks() -> None:
    class Broken:
        @property
        def outline(self) -> list:
            raise ValueError("broken outline")

    assert bookmarks.read(Broken()) is None  # ty: ignore[invalid-argument-type]


def test_the_preview_sets_its_headings_by_the_bookmarks(tmp_path: Path) -> None:
    path = _book(tmp_path, [("1 Sagas", 0, -1), ("Orchestration", 1, 0)])

    convert.build_preview(path, tmp_path / "preview", Parser.ANYDOC)

    assert _lines((tmp_path / "preview" / "preview.md").read_text()) == [
        "# Sagas",
        "## Orchestration",
    ]
