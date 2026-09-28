"""Pagination: page request validation, opaque cursors, sort whitelist, keyset walks over sqlite."""

from base64 import urlsafe_b64encode
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import (
    Column,
    ColumnElement,
    Connection,
    Integer,
    MetaData,
    Row,
    Table,
    Text,
    asc,
    create_engine,
    desc,
    insert,
    literal_column,
    select,
)
from sqlalchemy.dialects import sqlite

from haskie.errors import InvalidInput
from haskie.paging import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    Keyset,
    Order,
    PageRequest,
    decode_cursor,
    encode_cursor,
    keyset,
    page_request,
    resolve_sort,
)

METADATA = MetaData()
DOCS = Table(
    "docs",
    METADATA,
    Column("name", Text, primary_key=True),
    Column("size", Integer, nullable=False),
)
SORTS = {"name": DOCS.c.name, "size": DOCS.c.size}  # public name -> column
ROWID = literal_column("rowid")
ALLOWED = {**SORTS, "added": ROWID}
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


SIZE_DESC = encode_cursor([30, "doc-02"], "size", Order.DESC)


@pytest.mark.parametrize(
    ("name", "cursor", "sort", "order", "expected"),
    [
        ("no cursor: nothing to read, ascending", None, None, None, (None, Order.ASC)),
        ("no cursor: what was passed", None, "size", Order.DESC, ("size", Order.DESC)),
        ("cursor alone: its sort and order", SIZE_DESC, None, None, ("size", Order.DESC)),
        ("an empty sort is an omitted one", SIZE_DESC, "", None, ("size", Order.DESC)),
        ("order omitted: the cursor's", SIZE_DESC, "size", None, ("size", Order.DESC)),
        ("sort omitted: the cursor's", SIZE_DESC, None, Order.DESC, ("size", Order.DESC)),
        # kept as passed, so `decode_cursor` refuses the contradiction (next test)
        ("a contradicting sort stays", SIZE_DESC, "name", None, ("name", Order.DESC)),
        ("a contradicting order stays", SIZE_DESC, None, Order.ASC, ("size", Order.ASC)),
    ],
)
def test_page_request_reads_an_omitted_sort_and_order_from_the_cursor(
    name: str,
    cursor: str | None,
    sort: str | None,
    order: Order | None,
    expected: tuple[str | None, Order],
) -> None:
    request = page_request(cursor=cursor, sort=sort, order=order)
    assert (request.sort, request.order) == expected, name


@pytest.mark.parametrize(
    ("name", "sort", "order"),
    [("another sort", "name", None), ("the other order", None, Order.ASC)],
)
def test_a_cursor_contradicting_a_passed_sort_or_order_is_refused(
    name: str, sort: str | None, order: Order | None
) -> None:
    request = page_request(cursor=SIZE_DESC, sort=sort, order=order)
    with pytest.raises(InvalidInput, match="cursor does not match sort/order"):
        _keyset(request.sort or "name", request)


def test_page_request_refuses_a_cursor_it_cannot_read() -> None:
    with pytest.raises(InvalidInput, match="invalid cursor"):
        page_request(cursor="!!!!")


# --- sort whitelist ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "requested", "expected"),
    [
        ("requested name maps to its column", "added", ("added", ROWID)),
        ("unset falls back to the default", None, ("name", DOCS.c.name)),
    ],
)
def test_resolve_sort(
    name: str, requested: str | None, expected: tuple[str, ColumnElement[Any]]
) -> None:
    public, column = resolve_sort(requested, ALLOWED, "name")
    assert public == expected[0], name
    assert column is expected[1], f"{name}: the whitelisted column itself, never one from input"


def test_resolve_sort_rejects_unknown_name() -> None:
    with pytest.raises(InvalidInput, match="unknown sort 'title'; allowed: added, name, size"):
        resolve_sort("title", ALLOWED, "name")


# --- keyset -----------------------------------------------------------------------


@pytest.fixture
def docs() -> Iterator[Connection]:
    """25 documents over 5 distinct sizes, so every sort by size has ties to break."""
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        METADATA.create_all(conn)
        conn.execute(
            insert(DOCS),
            [{"name": f"doc-{i:02d}", "size": SIZES[i % len(SIZES)]} for i in range(ROWS)],
        )
        yield conn
    engine.dispose()


def _keyset(sort: str, request: PageRequest) -> Keyset:
    """`name` is the primary key, so it is the tie-breaker, and needs no second column when it
    is also the sort."""
    public, column = resolve_sort(sort, SORTS, "name")
    return keyset(public, column, request, DOCS.c.name)


def _fetch(conn: Connection, walk: Keyset) -> list[Row[Any]]:
    return list(conn.execute(walk.apply(select(DOCS.c.name, DOCS.c.size))))


def _walk(
    conn: Connection, sort: str, order: Order, page_size: int
) -> tuple[list[str], list[str | None]]:
    """Follow next_cursor to the last page; returns the names collected and every page's cursor."""
    names: list[str] = []
    cursors: list[str | None] = []
    cursor: str | None = None
    while True:
        request = page_request(cursor=cursor, page_size=page_size, sort=sort, order=order)
        walk = _keyset(sort, request)
        page = walk.page(_fetch(conn, walk), build=lambda row: row.name)
        names.extend(page.items)
        cursors.append(page.next_cursor)
        cursor = page.next_cursor
        if cursor is None:
            return names, cursors


@pytest.mark.parametrize(
    ("name", "order", "cursor_key", "expected_where", "expected_params"),
    [
        ("no cursor leaves the query unfiltered", "asc", None, "", [5, 0]),
        (
            "ascending compares greater than",
            "asc",
            [10, "doc-01"],
            "WHERE (docs.size, docs.name) > (?, ?) ",
            [10, "doc-01", 5, 0],
        ),
        (
            "descending compares less than",
            "desc",
            [10, "doc-01"],
            "WHERE (docs.size, docs.name) < (?, ?) ",
            [10, "doc-01", 5, 0],
        ),
    ],
)
def test_keyset_apply(
    name: str,
    order: Order,
    cursor_key: list[Any] | None,
    expected_where: str,
    expected_params: list[Any],
) -> None:
    cursor = encode_cursor(cursor_key, "size", order) if cursor_key else None
    request = page_request(cursor=cursor, page_size=4, sort="size", order=order)
    walk = Keyset("size", [DOCS.c.size, DOCS.c.name], order, request)
    compiled = walk.apply(select(DOCS.c.name)).compile(dialect=sqlite.dialect())
    direction = order.upper()
    assert " ".join(str(compiled).split()) == " ".join(
        f"SELECT docs.name FROM docs {expected_where}"
        f"ORDER BY docs.size {direction}, docs.name {direction} LIMIT ? OFFSET ?".split()
    ), f"{name}: both columns, one direction"
    params = [compiled.params[bound] for bound in compiled.positiontup or ()]
    assert params == expected_params, f"{name}: the page size plus the look-ahead row is the limit"


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
    docs: Connection, name: str, sort: str, order: Order
) -> None:
    direction = asc if order == Order.ASC else desc
    expected = list(
        docs.scalars(select(DOCS.c.name).order_by(direction(SORTS[sort]), direction(DOCS.c.name)))
    )
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
    docs: Connection, name: str, page_size: int, expected_items: int, expects_cursor: bool
) -> None:
    request = page_request(page_size=page_size, sort="size", order=Order.ASC)
    walk = _keyset("size", request)
    page = walk.page(_fetch(docs, walk), build=lambda row: row.name, total=ROWS)
    assert len(page.items) == expected_items, name
    assert (page.next_cursor is not None) == expects_cursor, name
    assert page.total == ROWS, "total passes through untouched"


def test_page_of_no_rows_is_empty_and_final() -> None:
    walk = Keyset("name", [DOCS.c.name], Order.ASC, page_request(page_size=4))
    page = walk.page([], build=lambda row: row.name)
    assert (page.items, page.next_cursor, page.total) == ([], None, None)
