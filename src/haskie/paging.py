"""Keyset pagination shared by every listing endpoint: page request, opaque cursor, sort whitelist.

Keyset (not offset): the cursor carries the sort key of the last row of the previous page, so a
page boundary stays exact while rows are inserted or deleted, and sqlite never counts rows it
skips. The cursor is not signed: this is a single-user local app, the cursor never leaves the
machine, and nothing inside it reaches SQL — column identifiers come from the caller's whitelist
only, the cursor contributes bound parameters.
"""

from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Any

import msgspec
from litestar.params import Parameter, ParameterKwarg

from haskie import db
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


class _Cursor(msgspec.Struct):
    """Wire form of a cursor: short field names, because it is encoded into every response."""

    k: list[Any]  # the sort key of the last row of the page, one value per keyset column
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
    order: Annotated[Order, one_of(Order)] = Order.ASC,
) -> PageRequest:
    """The one place handler query arguments become a validated request.

    Registered as a Litestar dependency (`api.common.PAGED`), so these four parameters are what a
    paged listing takes on the wire; a handler asks for the `PageRequest` they produce.
    """
    return PageRequest(cursor=cursor, page_size=page_size, sort=sort, order=order)


def encode_cursor(key: list[Any], sort: str, order: Order) -> str:
    raw = msgspec.json.encode(_Cursor(k=key, s=sort, o=order))
    return urlsafe_b64encode(raw).decode().rstrip("=")  # padding is noise in a URL


def decode_cursor(cursor: str, sort: str, order: Order, width: int) -> list[Any]:
    """The sort key inside `cursor`, rejected unless it was built for this sort, order, version
    and keyset width: a cursor from another listing would compare the wrong columns."""
    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        # binascii.Error (bad base64) and UnicodeDecodeError are both ValueError
        decoded = msgspec.json.decode(urlsafe_b64decode(padded), type=_Cursor)
    except (msgspec.DecodeError, ValueError) as exc:
        raise InvalidInput("invalid cursor") from exc
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
        # `type(...) is` rather than isinstance: JSON true/false decode as int subclasses
        if not isinstance(identity, str) or type(offset) is not int or offset < 0:
            raise InvalidInput("invalid cursor")
        return identity, offset


def resolve_sort(requested: str | None, allowed: dict[str, str], default: str) -> tuple[str, str]:
    """Map a public sort name to its SQL expression. The whitelist is the only source of column
    identifiers, so a request can never name a column."""
    name = requested or default
    expression = allowed.get(name)
    if expression is None:
        raise InvalidInput(f"unknown sort {name!r}; allowed: {', '.join(sorted(allowed))}")
    return name, expression


class Keyset:
    """Sort columns plus the primary-key tie-breaker(s), all read in the same direction.

    The page boundary is one row-value comparison, `(size, name) > (?, ?)`, which sqlite resolves
    with the same index it uses for the ordering. It is exact because every whitelisted sort
    column is NOT NULL: SQL's three-valued logic would otherwise drop rows at the boundary.
    """

    def __init__(self, sort: str, columns: list[str], order: Order, request: PageRequest) -> None:
        """`columns` is [sort expression, *tie-breakers], each a whitelisted SQL expression."""
        self.sort = sort
        self.columns = columns
        self.order = order
        self.request = request
        self.key: list[Any] | None = (
            decode_cursor(request.cursor, sort, order, len(columns)) if request.cursor else None
        )

    def where(self) -> tuple[str, list[Any]]:
        """Condition and parameters for the first page boundary; empty without a cursor."""
        if self.key is None:
            return "", []
        comparison = ">" if self.order == Order.ASC else "<"
        columns = ", ".join(self.columns)
        return f"({columns}) {comparison} ({db.placeholders(len(self.columns))})", list(self.key)

    def order_by(self) -> str:
        return "order by " + ", ".join(f"{column} {self.order}" for column in self.columns)

    def limit(self) -> int:
        """One row more than the page: its presence is what tells us another page exists."""
        return self.request.page_size + 1

    def page[T](
        self,
        rows: list[tuple],
        build: Callable[[tuple], T],
        key: Callable[[tuple], list[Any]],
        total: int | None = None,
    ) -> Page[T]:
        """Cut the look-ahead row off and turn the rest into a page. `key` reads the keyset
        columns of a row, in the order given to the constructor."""
        size = self.request.page_size
        visible = rows[:size]
        more = len(rows) > size
        return Page(
            items=[build(row) for row in visible],
            next_cursor=encode_cursor(key(visible[-1]), self.sort, self.order) if more else None,
            total=total,
        )


def keyset(sort: str, expression: str, request: PageRequest) -> Keyset:
    """The keyset of a listing whose rows are unique by `name`, so `name` breaks every tie; when
    it is also the sort column it is the whole keyset rather than a column repeated twice."""
    columns = [expression] if sort == "name" else [expression, "name"]
    return Keyset(sort, columns, request.order, request)


def key_reader(
    sort: str, expression: str, selected: list[str], name_column: str = "name"
) -> Callable[[tuple], list[Any]]:
    """Reads the keyset columns out of a row of `selected`, in the order `keyset` built them.
    `name_column` is how the name is spelled in `selected`, which a joined listing qualifies."""
    name = selected.index(name_column)
    if sort == "name":
        return lambda row: [row[name]]
    value = selected.index(expression)
    return lambda row: [row[value], row[name]]
