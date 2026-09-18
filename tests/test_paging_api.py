"""Paged listings over HTTP: page boundaries, cursor continuity, sorting, filtering, rejections.

No DBOS here, unlike `test_api.py`: every endpoint under test reads the metadata database, and
uploading a document that does not exist yet never reaches a workflow (see `app._cancel_running`).
"""

from pathlib import Path

import pytest
from litestar.testing import AsyncTestClient

from haskie import app as app_module
from haskie.library import Library

pytestmark = pytest.mark.anyio

DOC = b"# doc\n\nbody\n"
SIZES = (300, 200, 100)  # three distinct document sizes, so every sort by size has ties to break


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncTestClient:
    monkeypatch.setattr(app_module, "WEB_DIST", tmp_path / "no-web-build")
    return AsyncTestClient(app_module.create_app())


async def _create(client: AsyncTestClient, library: str) -> None:
    response = await client.post("/api/libraries", json={"name": library})
    assert response.status_code == 201


async def _upload(client: AsyncTestClient, library: str, name: str, content: bytes = DOC) -> dict:
    response = await client.post(
        f"/api/libraries/{library}/documents", files={"data": (name, content, "text/markdown")}
    )
    assert response.status_code == 201, response.text
    return response.json()


async def _page(client: AsyncTestClient, path: str, **params) -> dict:
    """One page; `cursor=None` is dropped, so the same call reads the first page too."""
    response = await client.get(path, params={k: v for k, v in params.items() if v is not None})
    assert response.status_code == 200, response.text
    return response.json()


async def _walk(
    client: AsyncTestClient, path: str, **params
) -> tuple[list[dict], list[int], list[int], list[str | None]]:
    """Follow next_cursor to the last page; returns the items, the size of each page, the total
    reported by each page and every page's cursor."""
    items: list[dict] = []
    sizes: list[int] = []
    totals: list[int] = []
    cursors: list[str | None] = []
    cursor: str | None = None
    while True:
        page = await _page(client, path, cursor=cursor, **params)
        items.extend(page["items"])
        sizes.append(len(page["items"]))
        totals.append(page["total"])
        cursors.append(page["next_cursor"])
        cursor = page["next_cursor"]
        if cursor is None:
            return items, sizes, totals, cursors


# --- libraries ----------------------------------------------------------------------


async def test_libraries_page_boundaries(client: AsyncTestClient) -> None:
    """Seven libraries in pages of three: 3 + 3 + 1, the total counts them all, and the counts
    of a row are the documents uploaded into it."""
    uploaded = {f"lib-{i}": i % 3 for i in range(7)}
    for library, documents in uploaded.items():
        await _create(client, library)
        for d in range(documents):
            await _upload(client, library, f"doc-{d}.md")

    items, sizes, totals, cursors = await _walk(client, "/api/libraries", page_size=3)

    assert sizes == [3, 3, 1], "the last page is the remainder"
    assert totals == [7, 7, 7], "every page reports the whole listing"
    assert cursors[-1] is None, "the last page ends the walk"
    assert all(cursor is not None for cursor in cursors[:-1]), "every earlier page carries one"
    assert [item["name"] for item in items] == sorted(uploaded), "name ascending by default"
    assert {item["name"]: item["counts"]["total"] for item in items} == uploaded
    assert all(item["created_at"] > 0 for item in items), "stamped when the library was created"


async def test_library_info_has_counts_and_no_documents(client: AsyncTestClient) -> None:
    await _create(client, "notes")
    for name in ("a.md", "b.md", "c.md"):
        await _upload(client, "notes", name)
    await Library("notes").set_status("a.md", "indexed")
    await Library("notes").set_status("b.md", "converting")

    info = await _page(client, "/api/libraries/notes")

    assert "documents" not in info, "the documents are a paged listing of their own"
    assert info["counts"] == {
        "total": 3,
        "indexed": 1,
        "active": 1,
        "error": 0,
        "by_status": {"indexed": 1, "converting": 1, "uploaded": 1},
    }


# --- documents ----------------------------------------------------------------------


async def test_documents_cursor_continuity_when_rows_are_inserted_between_pages(
    client: AsyncTestClient,
) -> None:
    """Keyset, not offset: a row inserted before the boundary while the caller walks does not
    shift the next page, so no document is returned twice or skipped."""
    await _create(client, "notes")
    for name in "abcdef":
        await _upload(client, "notes", f"{name}.md")

    first = await _page(client, "/api/libraries/notes/documents", page_size=3, sort="name")
    assert [item["name"] for item in first["items"]] == ["a.md", "b.md", "c.md"]
    assert first["total"] == 6

    await _upload(client, "notes", "aa.md")  # before the boundary
    await _upload(client, "notes", "zz.md")  # after it

    rest: list[str] = []
    cursor = first["next_cursor"]
    while cursor is not None:
        page = await _page(
            client, "/api/libraries/notes/documents", page_size=3, sort="name", cursor=cursor
        )
        rest.extend(item["name"] for item in page["items"])
        assert page["total"] == 8, "the total follows the table, the page boundary does not"
        cursor = page["next_cursor"]

    assert rest == ["d.md", "e.md", "f.md", "zz.md"]
    assert "aa.md" not in rest, "a row inserted behind the cursor is not served again"
    assert len(set(rest)) == len(rest), "no document is returned twice"


async def test_documents_sort_by_size_desc_with_ties(client: AsyncTestClient) -> None:
    """Nine documents over three sizes: the tie-breaker is the name, read in the same direction
    as the sort, so a walk in pages of four is exactly the ordering of the whole table."""
    await _create(client, "notes")
    sizes = {f"doc-{i}.md": SIZES[i % len(SIZES)] for i in range(9)}
    for name, size in sizes.items():
        await _upload(client, "notes", name, b"x" * size)

    items, page_sizes, totals, _ = await _walk(
        client, "/api/libraries/notes/documents", page_size=4, sort="size", order="desc"
    )

    names = [item["name"] for item in items]
    assert page_sizes == [4, 4, 1] and totals == [9, 9, 9]
    assert len(set(names)) == 9, "every document once"
    assert names == sorted(sizes, key=lambda n: (sizes[n], n), reverse=True), (
        "size descending, ties by name descending"
    )
    walked = [item["size"] for item in items]
    assert walked == sorted(walked, reverse=True), "sizes never rise again inside the walk"


async def test_documents_status_filter_and_total(client: AsyncTestClient) -> None:
    await _create(client, "notes")
    for name in ("a.md", "b.md", "c.md"):
        await _upload(client, "notes", name)
    await Library("notes").set_status("b.md", "indexed")

    indexed = await _page(client, "/api/libraries/notes/documents", status="indexed")

    assert [item["name"] for item in indexed["items"]] == ["b.md"]
    assert indexed["total"] == 1, "the total counts the filtered rows, not the table"
    assert indexed["next_cursor"] is None
    unfiltered = await _page(client, "/api/libraries/notes/documents")
    assert unfiltered["total"] == 3, "unfiltered is the table"

    rejected = await client.get("/api/libraries/notes/documents", params={"status": "bogus"})
    assert rejected.status_code == 422
    assert "Invalid enum value 'bogus'" in rejected.json()["detail"]


async def test_documents_sorted_by_updated_at_follow_the_lifecycle(
    client: AsyncTestClient,
) -> None:
    """The timestamps themselves are covered in `test_core.py`; this is the sort reading them."""
    await _create(client, "notes")
    for name in ("a.md", "b.md", "c.md"):
        await _upload(client, "notes", name)
    await Library("notes").set_status("a.md", "indexed")  # touched last, so it sorts last

    items, _, _, _ = await _walk(
        client, "/api/libraries/notes/documents", page_size=2, sort="updated_at"
    )

    assert [item["name"] for item in items] == ["b.md", "c.md", "a.md"]


# --- rejections ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "path", "params", "detail"),
    [
        ("cursor that is not base64", "/api/libraries", {"cursor": "!!!!"}, "invalid cursor"),
        (
            "cursor of another listing",
            "/api/libraries/notes/documents",
            {"cursor": "_LIBRARY_CURSOR_", "sort": "size"},
            "cursor does not match sort/order",
        ),
        (
            "cursor read in the other direction",
            "/api/libraries",
            {"cursor": "_LIBRARY_CURSOR_", "order": "desc"},
            "cursor does not match sort/order",
        ),
        ("unknown library sort", "/api/libraries", {"sort": "bogus"}, "unknown sort 'bogus'"),
        (
            "unknown document sort",
            "/api/libraries/notes/documents",
            {"sort": "bogus"},
            "unknown sort 'bogus'",
        ),
        ("page size below one", "/api/libraries", {"page_size": 0}, "page_size must be 1..1000"),
        (
            "page size over the cap",
            "/api/libraries/notes/documents",
            {"page_size": 1001},
            "page_size must be 1..1000",
        ),
    ],
)
async def test_bad_cursor_and_sort_are_422(
    client: AsyncTestClient, name: str, path: str, params: dict, detail: str
) -> None:
    await _create(client, "notes")
    await _create(client, "other")  # a second library, so the first page carries a cursor
    first = await _page(client, "/api/libraries", page_size=1)
    cursor = first["next_cursor"]
    params = {k: (cursor if v == "_LIBRARY_CURSOR_" else v) for k, v in params.items()}

    response = await client.get(path, params=params)

    assert response.status_code == 422, f"{name}: {response.text}"
    assert detail in response.json()["detail"], name
