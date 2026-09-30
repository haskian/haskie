"""A PDF's bookmarks, and the headings of its markdown set by them.

The PDF converter (`pdf_inspector`) decides what is a heading by font size, one page at a time.
On a typeset book that misses in two ways. It puts a chapter and its sections at one level, so
the chapter never holds them. And it takes a page's running header ("348 Chapter 10
AGGREGATES", or letter-spaced small caps, "R ULE: DESIGN SMALL AGGREGATES 355") for a heading
of its own, so every page of a section opens a new one: in one 657-page book, 395 of its 462
second-level "sections" were running headers.

A PDF that carries bookmarks already says what its sections are and how they nest: the table
of contents a viewer shows in its sidebar, each entry a title, a depth and a page. So when a PDF
has bookmarks of more than one level (`read`), each page's headings are set by them (`apply`):

- a heading whose title matches a bookmark of its page takes the bookmark's depth as its level;
- every other heading becomes plain text, running headers included.

Otherwise the converter's headings stand: a PDF without bookmarks, or with one level of them
only, which would demote every section heading under its chapters to text.

A heading matches a bookmark by its letters, case aside, on the page the bookmark points to or
the next one (a heading at the foot of a page lands there), when it holds no letter or digit the
title lacks: "# Aggregates" matches "10 Aggregates", while the running header "R ULE: DESIGN
SMALL AGGREGATES 355" does not match "Rule: Design Small Aggregates", as it adds its page number.
Of several headings that match one bookmark, the one closest to its title wins, and a bookmark
matches once: the pages are read in order, each claiming what it matched (`claimed`), so a
running header on the next page that repeats the title bare is text. A bookmark whose title is
on neither page adds nothing.

No IO: the caller opens the PDF (`read` takes its reader).
"""

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import msgspec

from haskie.document.render import heading_spans

if TYPE_CHECKING:
    from pypdf import PdfReader

MAX_LEVEL = 6  # the deepest heading markdown has; a deeper bookmark sets this one


class Bookmark(msgspec.Struct, frozen=True):
    """One entry of a PDF's table of contents."""

    level: int  # 1 for a top-level entry, 2 for one under it, as the heading level it sets
    title: str
    page: int  # 1-based, the page it points to


type Pages = Mapping[int, list[Bookmark]]  # the bookmarks each 1-based page holds


def read(reader: "PdfReader") -> list[Bookmark] | None:
    """The bookmarks that set the PDF's headings, in document order: None when it has none, one
    level of them only, or an outline that cannot be read, since a broken outline is no reason to
    fail a conversion that does not need it."""
    found: list[Bookmark] = []
    try:
        _walk(reader, reader.outline, 1, found)
    except Exception:
        return None
    return found if len({mark.level for mark in found}) > 1 else None


def _walk(reader: "PdfReader", items: list, level: int, found: list[Bookmark]) -> None:
    """Every entry of an outline, a nested list in pypdf: a list after an entry holds its
    children. An entry that points nowhere in the file (an external link) is skipped."""
    for item in items:
        if isinstance(item, list):
            _walk(reader, item, level + 1, found)
            continue
        page = reader.get_destination_page_number(item)
        title = str(item.title or "").strip()
        if page is not None and title:
            found.append(Bookmark(level=level, title=title, page=page + 1))


def pages(marks: Iterable[Bookmark]) -> dict[int, list[Bookmark]]:
    """The bookmarks by the page they point to."""
    found: dict[int, list[Bookmark]] = {}
    for mark in marks:
        found.setdefault(mark.page, []).append(mark)
    return found


def _letters(text: str) -> str:
    return "".join(char for char in text.casefold() if char.isalpha())


def _signs(text: str) -> int:
    return sum(char.isalnum() for char in text)


def apply(markdown: str, page: int, marks: Pages, claimed: set[Bookmark]) -> str:
    """One page's markdown with its headings set by the bookmarks (see the module). The
    bookmarks it matches join `claimed`, which the next page reads."""
    found = heading_spans(markdown)
    if not found:
        return markdown
    keys = [(_letters(one.text), _signs(one.text)) for one in found]
    levels: dict[int, int] = {}  # heading position -> the level its bookmark sets
    for mark in [*marks.get(page, []), *marks.get(page - 1, [])]:
        wanted, most = _letters(mark.title), _signs(mark.title)
        if mark in claimed or not wanted:
            continue
        matching = [
            at
            for at, (letters, signs) in enumerate(keys)
            if at not in levels and letters == wanted and signs <= most
        ]
        if matching:
            levels[max(matching, key=lambda at: keys[at][1])] = mark.level
            claimed.add(mark)
    raw = markdown.encode()
    out: list[bytes] = []
    cursor = 0
    for at, one in enumerate(found):
        text = raw[one.text_start : one.text_end]
        if at in levels:  # an ATX heading is one line
            lines = b" ".join(line.strip() for line in text.splitlines())
            text = b"#" * min(levels[at], MAX_LEVEL) + b" " + lines
        ending = b"\n" if raw[one.start : one.end].endswith(b"\n") else b""
        out += [raw[cursor : one.start], text, ending]
        cursor = one.end
    out.append(raw[cursor:])
    return b"".join(out).decode()
