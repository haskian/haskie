"""Pagination shared by every listing endpoint: page request, opaque cursor, sort whitelist.

Two cursor kinds. A keyset cursor (`Keyset`) carries the sort key of the last row of the previous
page, so a page boundary stays exact while rows are inserted or deleted, and sqlite never counts
rows it skips; the document, collection and member listings use it. An offset cursor
(`OffsetCursor`) is for a listing that cannot be resumed by key: full-text search and the
operation history.

The cursor is not signed: this is a single-user local app, the cursor never leaves the machine,
and nothing inside it reaches SQL — columns come from the caller's whitelist only, the cursor
contributes bound parameters, each typed to be one sqlite can bind.
"""

from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Annotated, Any

import msgspec
from litestar.params import Parameter, ParameterKwarg
from sqlalchemy import ColumnElement, Row, Select, asc, desc, func, tuple_

from haskie.errors import InvalidInput


class Order(StrEnum):
    ASC = "asc"
    DESC = "desc"


DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 1000
CURSOR_VERSION = 1


def check_page_size(size: int, cap: int = MAX_PAGE_SIZE, field: str = "page_size") -> int:
    """The one page-size bound check. `cap` and `field` differ where a listing pages something
    dearer than a metadata row (see `search.text`)."""
    if not 1 <= size <= cap:
        raise InvalidInput(f"{field} must be 1..{cap}, got {size}")
    return size


class PageRequest(msgspec.Struct, frozen=True):
    """The query arguments of one page, validated."""

    cursor: str | None = None
    page_size: int = DEFAULT_PAGE_SIZE
    sort: str | None = None
    order: Order = Order.ASC

    def __post_init__(self) -> None:
        check_page_size(self.page_size)


class Page[T](msgspec.Struct):
    """One page of results. `next_cursor` is None on the last page; `total` is optional because
    counting is a second query the caller may skip."""

    items: list[T]
    next_cursor: str | None = None
    total: int | None = None


# A key value sqlite can bind: a string, a float, or an integer its INTEGER holds. Any other JSON
# value, forged into a cursor, would fail in the driver as a 500; msgspec refuses it while decoding.
_KeyValue = str | Annotated[int, msgspec.Meta(ge=-(2**63), le=2**63 - 1)] | float


class _Cursor(msgspec.Struct):
    """Wire form of a cursor: short field names, because it is encoded into every response."""

    k: list[_KeyValue]  # the sort key of the last row of the page, one value per keyset column
    s: str  # public sort name it was built for
    o: Order
    v: int = CURSOR_VERSION


def one_of(choices: type[StrEnum]) -> ParameterKwarg:
    """A query argument over a closed set, with its values spelled out: litestar-mcp renders an
    enum as an untyped object, so an agent would not learn them from the tool schema."""
    return Parameter(description=f"One of: {', '.join(choices)}.")


def page_request(
    cursor: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    sort: str | None = None,
    order: Annotated[Order | None, one_of(Order)] = None,
) -> PageRequest:
    """The one place handler query arguments become a validated request.

    Registered as a Litestar dependency (`api.common.PAGED`), so these four parameters are what a
    paged listing takes on the wire; a handler asks for the `PageRequest` they produce.

    A cursor was built for one sort and order, so a caller passing `next_cursor` back as `cursor`
    need not repeat them: an omitted one is read from the cursor. One passed that contradicts the
    cursor is still refused, where the listing decodes it (`decode_cursor`).
    """
    if cursor is not None:
        issued = _read_cursor(cursor)
        # an empty sort is an omitted one, as `resolve_sort` reads it
        sort, order = sort or issued.s, order or issued.o
    return PageRequest(cursor=cursor, page_size=page_size, sort=sort, order=order or Order.ASC)


def encode_cursor(key: list[Any], sort: str, order: Order) -> str:
    raw = msgspec.json.encode(_Cursor(k=key, s=sort, o=order))
    return urlsafe_b64encode(raw).decode().rstrip("=")  # padding is noise in a URL


def _read_cursor(cursor: str) -> _Cursor:
    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        # binascii.Error (bad base64) and UnicodeDecodeError are both ValueError
        return msgspec.json.decode(urlsafe_b64decode(padded), type=_Cursor)
    except (msgspec.DecodeError, ValueError) as exc:
        raise InvalidInput("invalid cursor") from exc


def decode_cursor(cursor: str, sort: str, order: Order, width: int) -> list[Any]:
    """The sort key inside `cursor`, rejected unless it was built for this sort, order, version
    and keyset width: a cursor from another listing would compare the wrong columns."""
    decoded = _read_cursor(cursor)
    if (
        decoded.s != sort
        or decoded.o != order
        or decoded.v != CURSOR_VERSION
        or len(decoded.k) != width
    ):
        raise InvalidInput("cursor does not match sort/order")
    return decoded.k


class OffsetCursor:
    """Cursor for a listing that cannot be resumed by key: the identity of the result set it was
    issued for, plus how far into that set the next page starts.

    Keyset paging stays exact while rows move; this does not, and it is all a ranking that has to
    be recomputed (a full-text query) or a history with one fixed order (DBOS's) can offer. The
    identity is what stops a cursor of one result set quietly cutting another.
    """

    def __init__(self, sort: str, order: Order) -> None:
        self.sort = sort
        self.order = order

    def encode(self, identity: str, offset: int) -> str:
        return encode_cursor([identity, offset], self.sort, self.order)

    def decode(self, cursor: str) -> tuple[str, int]:
        """The (identity, offset) inside `cursor`. Callers check the identity: only they know
        which result set they are paging."""
        identity, offset = decode_cursor(cursor, self.sort, self.order, width=2)
        if not isinstance(identity, str) or not isinstance(offset, int) or offset < 0:
            raise InvalidInput("invalid cursor")
        return identity, offset


def resolve_sort[C: ColumnElement[Any]](
    requested: str | None, allowed: dict[str, C], default: str
) -> tuple[str, C]:
    """Map a public sort name to its column. The whitelist is the only source of columns, so a
    request can never name one."""
    name = requested or default
    column = allowed.get(name)
    if column is None:
        raise InvalidInput(f"unknown sort {name!r}; allowed: {', '.join(sorted(allowed))}")
    return name, column


class Keyset:
    """Sort columns plus the primary-key tie-breaker(s), all read in the same direction.

    The page boundary is one row-value comparison, `(size, name) > (?, ?)`, which sqlite resolves
    with the same index it uses for the ordering. It is exact because every whitelisted sort
    column is NOT NULL: SQL's three-valued logic would otherwise drop rows at the boundary.
    """

    def __init__(
        self, sort: str, columns: list[ColumnElement[Any]], order: Order, request: PageRequest
    ) -> None:
        """`columns` is [sort column, *tie-breakers], each whitelisted."""
        self.sort = sort
        self.columns = columns
        self.order = order
        self.request = request
        self.key: list[Any] | None = (
            decode_cursor(request.cursor, sort, order, len(columns)) if request.cursor else None
        )

    def apply[S: Select[Any]](self, statement: S) -> S:
        """`statement` cut to this page: past the cursor's boundary, in keyset order, and one row
        longer than the page, because that row's presence is what tells us another page
        exists."""
        if self.key is not None:
            columns = tuple_(*self.columns)
            boundary = tuple_(*self.key)
            statement = statement.where(
                columns > boundary if self.order == Order.ASC else columns < boundary
            )
        direction = asc if self.order == Order.ASC else desc
        return statement.order_by(*(direction(column) for column in self.columns)).limit(
            self.request.page_size + 1
        )

    def page[T](
        self, rows: Sequence[Row[Any]], build: Callable[[Row[Any]], T], total: int | None = None
    ) -> Page[T]:
        """Cut the look-ahead row off and turn the rest into a page. The next cursor is the
        keyset columns of the last row, which the statement has to select."""
        visible = rows[: self.request.page_size]
        next_cursor = None
        if len(rows) > len(visible):
            key = [visible[-1]._mapping[column] for column in self.columns]
            next_cursor = encode_cursor(key, self.sort, self.order)
        return Page(items=[build(row) for row in visible], next_cursor=next_cursor, total=total)


def keyset(
    sort: str, column: ColumnElement[Any], request: PageRequest, name: ColumnElement[Any]
) -> Keyset:
    """The keyset of a listing whose rows are unique by `name`, so `name` breaks every tie; when
    it is also the sort column it is the whole keyset rather than a column repeated twice."""
    columns = [column] if column is name else [column, name]
    return Keyset(sort, columns, request.order, request)


def count_of(listing: Select[Any]) -> Select[Any]:
    """How many rows `listing` holds before any page cuts it: the same FROM and filters, so a
    page's `total` counts what the page is a page of."""
    return listing.with_only_columns(func.count(), maintain_column_froms=True)
