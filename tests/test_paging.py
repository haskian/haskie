"""Pagination: page request validation, opaque cursors, sort whitelist, keyset walks over sqlite."""

import sqlite3
from base64 import urlsafe_b64encode
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from haskie.errors import InvalidInput
from haskie.paging import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    Keyset,
    Order,
    PageRequest,
    decode_cursor,
    encode_cursor,
    page_request,
    resolve_sort,
)

SORTS = {"name": "name", "size": "size"}  # public name -> SQL expression
ROWS = 25
SIZES = (10, 20, 30, 40, 50)


def _b64(raw: bytes) -> str:
    return urlsafe_b64encode(raw).decode().rstrip("=")


# --- cursors ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("string key", ["doc-01"]),
        ("integer key and tie-breaker", [42, "doc-01"]),
        ("float key and tie-breaker", [1.5, "doc-01"]),
        ("negative and zero values", [0, -7, "doc-01"]),
    ],
)
def test_cursor_round_trip(name: str, key: list[Any]) -> None:
    cursor = encode_cursor(key, "size", Order.ASC)
    assert "=" not in cursor, "padding stripped for a URL"
    assert decode_cursor(cursor, "size", Order.ASC, len(key)) == key, name


@pytest.mark.parametrize(
    ("name", "cursor", "sort", "order", "width", "message"),
    [
        (
            "truncated",
            encode_cursor(["doc-01"], "name", Order.ASC)[:10],
            "name",
            "asc",
            1,
            "invalid",
        ),
        ("not base64", "!!!!", "name", "asc", 1, "invalid cursor"),
        ("base64 of a json array", _b64(b"[]"), "name", "asc", 1, "invalid cursor"),
        ("base64 of random bytes", _b64(b"\xff\xfe\x00"), "name", "asc", 1, "invalid cursor"),
        (
            "unknown version",
            _b64(b'{"k":["doc-01"],"s":"name","o":"asc","v":2}'),
            "name",
            "asc",
            1,
            "does not match sort/order",
        ),
        (
            "wrong keyset width",
            encode_cursor(["doc-01"], "name", Order.ASC),
            "name",
            "asc",
            2,
            "does not match sort/order",
        ),
        (
            "built for another sort",
            encode_cursor(["doc-01"], "name", Order.ASC),
            "size",
            "asc",
            1,
            "does not match sort/order",
        ),
        (
            "built for the other order",
            encode_cursor(["doc-01"], "name", Order.ASC),
            "name",
            "desc",
            1,
            "does not match sort/order",
        ),
    ],
)
def test_decode_cursor_rejects(
    name: str, cursor: str, sort: str, order: Order, width: int, message: str
) -> None:
    with pytest.raises(InvalidInput, match=message):
        decode_cursor(cursor, sort, order, width)


# --- page request -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "page_size"),
    [("zero", 0), ("negative", -1), ("one above the maximum", MAX_PAGE_SIZE + 1)],
)
def test_page_request_rejects_page_size(name: str, page_size: int) -> None:
    with pytest.raises(
        InvalidInput, match=f"page_size must be 1..{MAX_PAGE_SIZE}, got {page_size}"
    ):
        page_request(page_size=page_size)


@pytest.mark.parametrize(
    ("name", "page_size", "expected"),
    [
        ("smallest page", 1, 1),
        ("largest page", MAX_PAGE_SIZE, MAX_PAGE_SIZE),
        ("unset falls back to the default", None, DEFAULT_PAGE_SIZE),
    ],
)
def test_page_request_accepts_page_size(name: str, page_size: int | None, expected: int) -> None:
    request = page_request() if page_size is None else page_request(page_size=page_size)
    assert request.page_size == expected, name
    assert (request.cursor, request.sort, request.order) == (None, None, "asc"), "defaults"


# --- sort whitelist ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "requested", "expected"),
    [
        ("requested name maps to its expression", "added", ("added", "rowid")),
        ("unset falls back to the default", None, ("name", "name")),
    ],
)
def test_resolve_sort(name: str, requested: str | None, expected: tuple[str, str]) -> None:
    allowed = {"name": "name", "size": "size", "added": "rowid"}
    assert resolve_sort(requested, allowed, "name") == expected, name


def test_resolve_sort_rejects_unknown_name() -> None:
    allowed = {"name": "name", "size": "size", "added": "rowid"}
    with pytest.raises(InvalidInput, match="unknown sort 'title'; allowed: added, name, size"):
        resolve_sort("title", allowed, "name")


# --- keyset -----------------------------------------------------------------------


@pytest.fixture
def docs() -> Iterator[sqlite3.Connection]:
    """25 documents over 5 distinct sizes, so every sort by size has ties to break."""
    conn = sqlite3.connect(":memory:")
    conn.execute("create table docs (name text primary key, size integer not null)")
    conn.executemany(
        "insert into docs (name, size) values (?, ?)",
        [(f"doc-{i:02d}", SIZES[i % len(SIZES)]) for i in range(ROWS)],
    )
    yield conn
    conn.close()


def _keyset(
    sort: str, order: Order, request: PageRequest
) -> tuple[Keyset, Callable[[tuple], list[Any]]]:
    """Keyset plus the matching key reader; `name` is the primary key, so it is the tie-breaker
    and needs no second column when it is also the sort."""
    public, expression = resolve_sort(sort, SORTS, "name")
    if public == "name":
        return Keyset(public, [expression], order, request), lambda row: [row[0]]
    return Keyset(public, [expression, "name"], order, request), lambda row: [row[1], row[0]]


def _fetch(conn: sqlite3.Connection, keyset: Keyset) -> list[tuple]:
    where, params = keyset.where()
    clause = f"where {where} " if where else ""
    sql = f"select name, size from docs {clause}{keyset.order_by()} limit {keyset.limit()}"
    return conn.execute(sql, params).fetchall()


def _walk(
    conn: sqlite3.Connection, sort: str, order: Order, page_size: int
) -> tuple[list[str], list[str | None]]:
    """Follow next_cursor to the last page; returns the names collected and every page's cursor."""
    names: list[str] = []
    cursors: list[str | None] = []
    cursor: str | None = None
    while True:
        request = page_request(cursor=cursor, page_size=page_size, sort=sort, order=order)
        keyset, key = _keyset(sort, order, request)
        page = keyset.page(_fetch(conn, keyset), build=lambda row: row[0], key=key)
        names.extend(page.items)
        cursors.append(page.next_cursor)
        cursor = page.next_cursor
        if cursor is None:
            return names, cursors


@pytest.mark.parametrize(
    ("name", "order", "cursor_key", "expected_where", "expected_params"),
    [
        ("no cursor leaves the query unfiltered", "asc", None, "", []),
        (
            "ascending compares greater than",
            "asc",
            [10, "doc-01"],
            "(size, name) > (?, ?)",
            [10, "doc-01"],
        ),
        (
            "descending compares less than",
            "desc",
            [10, "doc-01"],
            "(size, name) < (?, ?)",
            [10, "doc-01"],
        ),
    ],
)
def test_keyset_where(
    name: str,
    order: Order,
    cursor_key: list[Any] | None,
    expected_where: str,
    expected_params: list[Any],
) -> None:
    cursor = encode_cursor(cursor_key, "size", order) if cursor_key else None
    request = page_request(cursor=cursor, page_size=4, sort="size", order=order)
    keyset = Keyset("size", ["size", "name"], order, request)
    assert keyset.where() == (expected_where, expected_params), name
    assert keyset.order_by() == f"order by size {order}, name {order}", (
        "both columns, one direction"
    )
    assert keyset.limit() == 5, "page size plus the look-ahead row"


@pytest.mark.parametrize(
    ("name", "sort", "order"),
    [
        ("by name ascending", "name", "asc"),
        ("by name descending", "name", "desc"),
        ("by size ascending, ties broken by name", "size", "asc"),
        ("by size descending, ties broken by name", "size", "desc"),
    ],
)
def test_keyset_walk_reads_every_row_once(
    docs: sqlite3.Connection, name: str, sort: str, order: Order
) -> None:
    expected = [
        row[0]
        for row in docs.execute(
            f"select name from docs order by {SORTS[sort]} {order}, name {order}"
        )
    ]
    names, cursors = _walk(docs, sort, order, page_size=4)
    assert names == expected, name
    assert len(set(names)) == ROWS, "no row is returned twice"
    assert len(cursors) == 7, "25 rows in pages of 4"
    assert cursors[-1] is None, "the last page ends the walk"
    assert all(c is not None for c in cursors[:-1]), "every earlier page carries a cursor"


@pytest.mark.parametrize(
    ("name", "page_size", "expected_items", "expects_cursor"),
    [
        ("more rows than the page", 4, 4, True),
        ("exactly one page of rows", ROWS, ROWS, False),
        ("fewer rows than the page", ROWS + 5, ROWS, False),
    ],
)
def test_look_ahead_sets_next_cursor_only_when_more_rows_exist(
    docs: sqlite3.Connection, name: str, page_size: int, expected_items: int, expects_cursor: bool
) -> None:
    request = page_request(page_size=page_size, sort="size", order=Order.ASC)
    keyset, key = _keyset("size", Order.ASC, request)
    page = keyset.page(_fetch(docs, keyset), build=lambda row: row[0], key=key, total=ROWS)
    assert len(page.items) == expected_items, name
    assert (page.next_cursor is not None) == expects_cursor, name
    assert page.total == ROWS, "total passes through untouched"


def test_page_of_no_rows_is_empty_and_final() -> None:
    keyset = Keyset("name", ["name"], Order.ASC, page_request(page_size=4))
    page = keyset.page([], build=lambda row: row[0], key=lambda row: [row[0]])
    assert (page.items, page.next_cursor, page.total) == ([], None, None)
