"""Paged listings over HTTP: page boundaries, cursor continuity, sorting, filtering, rejections.

No DBOS here, unlike `test_api.py`: every endpoint under test reads the metadata database, and the
rows are written straight through `document` and `Collection` rather than through the intake, so
nothing reaches a workflow. What is under test is the paging, not how a row came to exist.
"""

from pathlib import Path

import pytest
from conftest import get_page, import_row, walk_pages
from litestar.testing import AsyncTestClient

from haskie.collection.collection import Collection, MemberStatus
from haskie.document import document
from haskie.document.document import DocumentStatus
from haskie.paging import Order, encode_cursor

pytestmark = pytest.mark.anyio

DOC = b"# doc\n\nbody\n"
SIZES = (300, 200, 100)  # three distinct document sizes, so every sort by size has ties to break


@pytest.fixture
def sources(tmp_path: Path) -> Path:
    """Where the files imported below come from; outside the home's own directories."""
    directory = tmp_path / "sources"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


async def _create(api_client: AsyncTestClient, collection: str) -> None:
    response = await api_client.post("/api/collections", json={"name": collection})
    assert response.status_code == 201, response.text


async def _import(sources: Path, name: str, content: bytes = DOC) -> document.Document:
    return await import_row(name, content, sources)


async def _member(sources: Path, collection: str, name: str, content: bytes = DOC) -> None:
    """One membership, without the intake: only an imported document joins a collection, so the
    row is marked `imported` here rather than run through a pipeline."""
    row = await _import(sources, name, content)
    await document.set_status(row.name, DocumentStatus.IMPORTED)
    await Collection(collection).add(row.name)


# --- collections --------------------------------------------------------------------


async def test_collections_page_boundaries(api_client: AsyncTestClient, sources: Path) -> None:
    """Seven collections in pages of three: 3 + 3 + 1, the total counts them all, and the counts
    of a row are the documents attached to it."""
    attached = {f"box-{i}": i % 3 for i in range(7)}
    for collection, members in attached.items():
        await _create(api_client, collection)
        for d in range(members):
            await _member(sources, collection, f"{collection}-{d}.md")

    items, sizes, totals, cursors = await walk_pages(api_client, "/api/collections", page_size=3)

    assert sizes == [3, 3, 1], "the last page is the remainder"
    assert totals == [7, 7, 7], "every page reports the whole listing"
    assert cursors[-1] is None, "the last page ends the walk"
    assert all(cursor is not None for cursor in cursors[:-1]), "every earlier page carries one"
    assert [item["name"] for item in items] == sorted(attached), "name ascending by default"
    assert {item["name"]: item["counts"]["total"] for item in items} == attached
    assert all(item["created_at"] > 0 for item in items), "stamped when the collection was created"


async def test_collection_info_has_counts_and_no_documents(
    api_client: AsyncTestClient, sources: Path
) -> None:
    await _create(api_client, "notes")
    for name in ("a.md", "b.md", "c.md"):
        await _member(sources, "notes", name)
    await Collection("notes").set_member_status("a.md", MemberStatus.INDEXED)
    await Collection("notes").set_member_status("b.md", MemberStatus.INDEXING)

    info = await get_page(api_client, "/api/collections/notes")

    assert "documents" not in info, "the members are a paged listing of their own"
    assert info["counts"] == {
        "total": 3,
        "indexed": 1,
        "active": 2,
        "error": 0,
        "by_status": {"indexed": 1, "indexing": 1, "pending": 1},
    }


# --- documents ----------------------------------------------------------------------


async def test_documents_cursor_continuity_when_rows_are_inserted_between_pages(
    api_client: AsyncTestClient, sources: Path
) -> None:
    """Keyset, not offset: a row inserted before the boundary while the caller walks does not
    shift the next page, so no document is returned twice or skipped."""
    for name in "abcdef":
        await _import(sources, f"{name}.md")

    first = await get_page(api_client, "/api/documents", page_size=3, sort="name")
    assert [item["name"] for item in first["items"]] == ["a.md", "b.md", "c.md"]
    assert first["total"] == 6

    await _import(sources, "aa.md")  # before the boundary
    await _import(sources, "zz.md")  # after it

    rest: list[str] = []
    cursor = first["next_cursor"]
    while cursor is not None:
        page = await get_page(api_client, "/api/documents", page_size=3, sort="name", cursor=cursor)
        rest.extend(item["name"] for item in page["items"])
        assert page["total"] == 8, "the total follows the table, the page boundary does not"
        cursor = page["next_cursor"]

    assert rest == ["d.md", "e.md", "f.md", "zz.md"]
    assert "aa.md" not in rest, "a row inserted behind the cursor is not served again"
    assert len(set(rest)) == len(rest), "no document is returned twice"


async def test_documents_sort_by_size_desc_with_ties(
    api_client: AsyncTestClient, sources: Path
) -> None:
    """Nine documents over three sizes: the tie-breaker is the name, read in the same direction
    as the sort, so a walk in pages of four is exactly the ordering of the whole table."""
    sizes = {f"doc-{i}.md": SIZES[i % len(SIZES)] for i in range(9)}
    for name, size in sizes.items():
        await _import(sources, name, b"x" * size)

    items, page_sizes, totals, _ = await walk_pages(
        api_client, "/api/documents", page_size=4, sort="size", order="desc"
    )

    names = [item["name"] for item in items]
    assert page_sizes == [4, 4, 1] and totals == [9, 9, 9]
    assert len(set(names)) == 9, "every document once"
    assert names == sorted(sizes, key=lambda n: (sizes[n], n), reverse=True), (
        "size descending, ties by name descending"
    )
    walked = [item["size"] for item in items]
    assert walked == sorted(walked, reverse=True), "sizes never rise again inside the walk"


@pytest.mark.parametrize(
    ("name", "repeated"),
    [
        ("sort and order omitted: both come from the cursor", {}),
        ("order omitted: it comes from the cursor", {"sort": "size"}),
        ("sort omitted: it comes from the cursor", {"order": "desc"}),
        ("both repeated as the cursor was built", {"sort": "size", "order": "desc"}),
    ],
)
async def test_next_cursor_alone_continues_the_sort_it_was_built_for(
    api_client: AsyncTestClient, sources: Path, name: str, repeated: dict[str, str]
) -> None:
    """The docs say to pass `next_cursor` back as `cursor`; that must be enough for page two of a
    sort that is not the default, whatever the caller repeats of the first request."""
    sizes = {f"doc-{i}.md": SIZES[i % len(SIZES)] for i in range(9)}
    for document_name, size in sizes.items():
        await _import(sources, document_name, b"x" * size)
    first = await get_page(api_client, "/api/documents", page_size=4, sort="size", order="desc")

    names = [item["name"] for item in first["items"]]
    cursor = first["next_cursor"]
    while cursor is not None:
        page = await get_page(api_client, "/api/documents", page_size=4, cursor=cursor, **repeated)
        names.extend(item["name"] for item in page["items"])
        cursor = page["next_cursor"]

    assert names == sorted(sizes, key=lambda n: (sizes[n], n), reverse=True), name


async def test_documents_status_filter_and_total(
    api_client: AsyncTestClient, sources: Path
) -> None:
    for name in ("a.md", "b.md", "c.md"):
        await _import(sources, name)
    await document.set_status("b.md", DocumentStatus.IMPORTED)

    imported = await get_page(api_client, "/api/documents", status="imported")

    assert [item["name"] for item in imported["items"]] == ["b.md"]
    assert imported["total"] == 1, "the total counts the filtered rows, not the table"
    assert imported["next_cursor"] is None
    unfiltered = await get_page(api_client, "/api/documents")
    assert unfiltered["total"] == 3, "unfiltered is the table"

    rejected = await api_client.get("/api/documents", params={"status": "bogus"})
    assert rejected.status_code == 422
    assert "Invalid enum value 'bogus'" in rejected.json()["detail"]


async def test_documents_sorted_by_status_group_the_lifecycle(
    api_client: AsyncTestClient, sources: Path
) -> None:
    """The status is a plain column, so sorting by it groups the lifecycle and the name breaks
    the ties inside each group."""
    for name in ("a.md", "b.md", "c.md"):
        await _import(sources, name)
    await document.set_status("a.md", DocumentStatus.IMPORTED)
    await document.set_status("c.md", DocumentStatus.ERROR)

    items, _, _, _ = await walk_pages(api_client, "/api/documents", page_size=2, sort="status")

    assert [(item["name"], item["status"]) for item in items] == [
        ("c.md", "error"),
        ("a.md", "imported"),
        ("b.md", "queued"),
    ]


async def test_documents_sorted_by_updated_at_follow_the_lifecycle(
    api_client: AsyncTestClient, sources: Path
) -> None:
    """The timestamps themselves are covered in `test_core.py`; this is the sort reading them."""
    for name in ("a.md", "b.md", "c.md"):
        await _import(sources, name)
    await document.set_status("a.md", DocumentStatus.IMPORTED)  # touched last, so it sorts last

    items, _, _, _ = await walk_pages(api_client, "/api/documents", page_size=2, sort="updated_at")

    assert [item["name"] for item in items] == ["b.md", "c.md", "a.md"]


# --- members ------------------------------------------------------------------------


async def test_members_page_boundaries_and_sort_by_size(
    api_client: AsyncTestClient, sources: Path
) -> None:
    """A member row is the document plus how far this collection got indexing it, so the size
    sort reads the document's column and the ties break on its name."""
    await _create(api_client, "notes")
    sizes = {f"doc-{i}.md": SIZES[i % len(SIZES)] for i in range(9)}
    for name, size in sizes.items():
        await _member(sources, "notes", name, b"x" * size)

    items, page_sizes, totals, _ = await walk_pages(
        api_client, "/api/collections/notes/documents", page_size=4, sort="size", order="desc"
    )

    names = [item["document"]["name"] for item in items]
    assert page_sizes == [4, 4, 1] and totals == [9, 9, 9]
    assert names == sorted(sizes, key=lambda n: (sizes[n], n), reverse=True)
    assert {item["status"] for item in items} == {"pending"}, "attached, not indexed yet"


async def test_members_status_filter_and_total(api_client: AsyncTestClient, sources: Path) -> None:
    """The membership status, not the document's: a document `imported` everywhere can still be
    `error` in one collection."""
    await _create(api_client, "notes")
    for name in ("a.md", "b.md", "c.md"):
        await _member(sources, "notes", name)
    await Collection("notes").set_member_status("b.md", MemberStatus.INDEXED)

    indexed = await get_page(api_client, "/api/collections/notes/documents", status="indexed")

    assert [item["document"]["name"] for item in indexed["items"]] == ["b.md"]
    assert indexed["total"] == 1, "the total counts the filtered rows, not the table"
    assert indexed["next_cursor"] is None
    unfiltered = await get_page(api_client, "/api/collections/notes/documents")
    assert unfiltered["total"] == 3, "unfiltered is the table"
    assert {item["document"]["status"] for item in unfiltered["items"]} == {"imported"}

    rejected = await api_client.get("/api/collections/notes/documents", params={"status": "bogus"})
    assert rejected.status_code == 422
    assert "Invalid enum value 'bogus'" in rejected.json()["detail"]


async def test_members_sorted_by_updated_at_follow_the_indexing(
    api_client: AsyncTestClient, sources: Path
) -> None:
    await _create(api_client, "notes")
    for name in ("a.md", "b.md", "c.md"):
        await _member(sources, "notes", name)
    await Collection("notes").set_member_status(
        "a.md", MemberStatus.INDEXED
    )  # touched last, sorts last

    items, _, _, _ = await walk_pages(
        api_client, "/api/collections/notes/documents", page_size=2, sort="updated_at"
    )

    assert [item["document"]["name"] for item in items] == ["b.md", "c.md", "a.md"]


async def test_members_of_one_collection_only(api_client: AsyncTestClient, sources: Path) -> None:
    """The same document in two collections is one row in each listing, and the listing of a
    collection that never held it is empty."""
    for name in ("alpha", "beta", "empty"):
        await _create(api_client, name)
    row = await _import(sources, "shared.md")
    await document.set_status(
        row.name, DocumentStatus.IMPORTED
    )  # only an imported document joins a collection
    for name in ("alpha", "beta"):
        await Collection(name).add(row.name)

    for name in ("alpha", "beta"):
        page = await get_page(api_client, f"/api/collections/{name}/documents")
        assert [item["document"]["name"] for item in page["items"]] == ["shared.md"]
    assert (await get_page(api_client, "/api/collections/empty/documents"))["items"] == []
    assert (await get_page(api_client, "/api/documents"))["total"] == 1, (
        "one document, two memberships"
    )


# --- the paging arguments as a contract -----------------------------------------------

PAGING_PARAMS = frozenset({"cursor", "page_size", "sort", "order"})
PAGED_LISTINGS = ("/api/documents", "/api/collections", "/api/collections/{collection}/documents")


async def test_the_paging_arguments_are_query_parameters_of_every_listing(
    api_client: AsyncTestClient,
) -> None:
    """They come from one dependency now rather than four parameters per handler, so this is what
    says they are still the public contract of a listing."""
    schema = (await api_client.get("/schema/openapi.json")).json()

    for path in PAGED_LISTINGS:
        declared = schema["paths"][path]["get"]["parameters"]
        paging = [p for p in declared if p["name"] in PAGING_PARAMS]
        assert {p["name"] for p in paging} == PAGING_PARAMS, path
        assert all(p["in"] == "query" and not p["required"] for p in paging), path


async def test_the_paging_arguments_are_inputs_of_the_listing_mcp_tools(
    api_client: AsyncTestClient,
) -> None:
    """litestar-mcp walks a provider's own parameters, so a tool keeps them. A route without the
    dependency must not gain them, which is why it is registered per route and not on the app.

    `api_client` is what builds the app: an unregistered handler resolves no dependencies.
    """
    from litestar_mcp.schema_builder import generate_schema_for_handler

    from haskie.api import collections, documents

    for handler in (documents.list_documents, collections.list_collection_documents):
        inputs = generate_schema_for_handler(handler)["properties"]
        assert PAGING_PARAMS <= set(inputs), handler

    one_document = generate_schema_for_handler(documents.get_document)["properties"]
    assert PAGING_PARAMS.isdisjoint(one_document), "only a paged listing advertises them"


# --- rejections ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "path", "params", "detail"),
    [
        ("cursor that is not base64", "/api/collections", {"cursor": "!!!!"}, "invalid cursor"),
        (
            "cursor of another listing",
            "/api/documents",
            {"cursor": "_COLLECTION_CURSOR_", "sort": "size"},
            "cursor does not match sort/order",
        ),
        (
            "cursor of another listing, on the members",
            "/api/collections/notes/documents",
            {"cursor": "_COLLECTION_CURSOR_", "sort": "size"},
            "cursor does not match sort/order",
        ),
        (
            "cursor read in the other direction",
            "/api/collections",
            {"cursor": "_COLLECTION_CURSOR_", "order": "desc"},
            "cursor does not match sort/order",
        ),
        (
            "cursor read by another sort of its own listing",
            "/api/collections",
            {"cursor": "_COLLECTION_CURSOR_", "sort": "created_at"},
            "cursor does not match sort/order",
        ),
        (
            "forged cursor whose key holds a list",
            "/api/documents",
            {"cursor": encode_cursor([[300], "a.md"], "size", Order.ASC)},
            "invalid cursor",
        ),
        (
            "forged cursor whose key holds an object",
            "/api/documents",
            {"cursor": encode_cursor([{"size": 300}, "a.md"], "size", Order.ASC)},
            "invalid cursor",
        ),
        (
            "forged cursor whose key overflows sqlite's integer",
            "/api/documents",
            {"cursor": encode_cursor([2**63, "a.md"], "size", Order.ASC)},
            "invalid cursor",
        ),
        ("unknown collection sort", "/api/collections", {"sort": "bogus"}, "unknown sort 'bogus'"),
        ("unknown document sort", "/api/documents", {"sort": "bogus"}, "unknown sort 'bogus'"),
        (
            "unknown member sort",
            "/api/collections/notes/documents",
            {"sort": "bogus"},
            "unknown sort 'bogus'",
        ),
        ("page size below one", "/api/collections", {"page_size": 0}, "page_size must be 1..1000"),
        (
            "page size over the cap",
            "/api/documents",
            {"page_size": 1001},
            "page_size must be 1..1000",
        ),
        (
            "page size over the cap, on the members",
            "/api/collections/notes/documents",
            {"page_size": 1001},
            "page_size must be 1..1000",
        ),
    ],
)
async def test_bad_cursor_and_sort_are_422(
    api_client: AsyncTestClient, name: str, path: str, params: dict, detail: str
) -> None:
    await _create(api_client, "notes")
    await _create(api_client, "other")  # a second collection, so the first page carries a cursor
    first = await get_page(api_client, "/api/collections", page_size=1)
    cursor = first["next_cursor"]
    params = {k: (cursor if v == "_COLLECTION_CURSOR_" else v) for k, v in params.items()}

    response = await api_client.get(path, params=params)

    assert response.status_code == 422, f"{name}: {response.text}"
    assert detail in response.json()["detail"], name
