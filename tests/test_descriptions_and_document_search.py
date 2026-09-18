"""Descriptions, renaming on upload, and the document shortlist.

A real index, not a fixture row: `search_documents` folds a BM25 ranking, so the ranking has to be
one LanceDB actually produced.
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from litestar.testing import AsyncTestClient

from haskie import app as app_module
from haskie import textsearch
from haskie.errors import InvalidInput
from haskie.library import Library

from conftest import wait_for  # isort: skip

pytestmark = pytest.mark.anyio

# "parsing" is in the compiler and the OS document, so a match count alone cannot order them.
PAPERS = {
    "compilers.md": "# Compilers\n\nparsing, parsing and more parsing of grammars\n",
    "os.md": "# Operating Systems\n\nprocesses, scheduling and a little parsing\n",
    "networks.md": "# Networks\n\nrouting and congestion control\n",
}


@pytest.fixture
async def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dbos
) -> AsyncIterator[AsyncTestClient]:
    monkeypatch.setattr(app_module, "WEB_DIST", tmp_path / "no-web-build")
    async with AsyncTestClient(app_module.create_app()) as client:
        await client.post("/api/init", json={"profile": "none"})
        yield client


@pytest.fixture
async def shelf(client: AsyncTestClient) -> AsyncTestClient:
    """One library of three indexed markdown documents, two of them described."""
    await client.post("/api/libraries", json={"name": "lit", "description": "reading list"})
    described = {"compilers.md": "the dragon book", "os.md": "processes and threads"}
    for name, body in PAPERS.items():
        await client.post(
            f"/api/libraries/lit/documents?description={described.get(name, '')}",
            files={"data": (name, body.encode(), "text/markdown")},
        )
    started = [
        (await client.post(f"/api/libraries/lit/documents/{name}/index")).json()["job_id"]
        for name in PAPERS
    ]
    await asyncio.gather(*(wait_for(job) for job in started))
    return client


async def test_a_library_and_a_document_carry_a_description(shelf: AsyncTestClient) -> None:
    """Set on create and on upload, returned by the reads the UI and an agent use."""
    info = (await shelf.get("/api/libraries/lit")).json()
    assert info["description"] == "reading list"

    listed = (await shelf.get("/api/libraries")).json()["items"]
    assert [row["description"] for row in listed] == ["reading list"], "the sidebar shows it too"

    documents = {
        d["name"]: d["description"]
        for d in (await shelf.get("/api/libraries/lit/documents")).json()["items"]
    }
    assert documents == {
        "compilers.md": "the dragon book",
        "os.md": "processes and threads",
        "networks.md": "",
    }


async def test_a_description_is_editable_after_the_fact(shelf: AsyncTestClient) -> None:
    """Both descriptions are replaceable, and an empty string clears one."""
    library = await shelf.put("/api/libraries/lit/description", json={"description": "CS classics"})
    assert library.json()["description"] == "CS classics"

    document = await shelf.put(
        "/api/libraries/lit/documents/networks.md/description", json={"description": "TCP/IP"}
    )
    assert document.json()["description"] == "TCP/IP"

    cleared = await shelf.put(
        "/api/libraries/lit/documents/networks.md/description", json={"description": ""}
    )
    assert cleared.json()["description"] == "", "empty clears rather than being rejected"


@pytest.mark.parametrize(
    ("name", "filename", "rename_to", "expected"),
    [
        ("no rename keeps the filename", "paper.md", None, "paper.md"),
        (
            "rename without a suffix keeps the original's",
            "aho-v2-FINAL.md",
            "compilers",
            "compilers.md",
        ),
        ("rename with the same suffix is left alone", "a.md", "compilers.md", "compilers.md"),
        # `safe_name` turns each run of punctuation into one dash, suffix included
        ("a rename is sanitised like any name", "a.md", "my paper!", "my-paper-.md"),
        ("a wrong suffix does not choose a parser", "a.md", "compilers.exe", "compilers.exe.md"),
    ],
)
async def test_rename_on_upload(
    client: AsyncTestClient, name: str, filename: str, rename_to: str | None, expected: str
) -> None:
    """`rename_to` names the document; the suffix still decides how it is parsed."""
    await client.post("/api/libraries", json={"name": "box"})
    query = f"?rename_to={rename_to}" if rename_to else ""
    response = await client.post(
        f"/api/libraries/box/documents{query}",
        files={"data": (filename, b"# Body\n\ntext\n", "text/markdown")},
    )

    assert response.status_code == 201, name
    assert response.json()["name"] == expected, name


async def test_search_documents_returns_one_row_per_document_best_first(
    shelf: AsyncTestClient,
) -> None:
    """The shortlist, not the passages: distinct documents, ranked, with their description."""
    matches = (await shelf.get("/api/search/documents", params={"q": "parsing"})).json()

    docs = [m["doc"] for m in matches]
    assert docs == sorted(set(docs), key=docs.index), "one row per document"
    assert set(docs) == {"compilers.md", "os.md"}, "the document that never says it is left out"
    assert docs[0] == "compilers.md", "the stronger match leads"
    assert [m["score"] for m in matches] == sorted((m["score"] for m in matches), reverse=True)

    best = matches[0]
    assert best["description"] == "the dragon book"
    assert best["chunks"] >= 1 and best["text"], "the evidence for the document being listed"
    assert best["heading"] == "Compilers"


async def test_search_documents_honours_limit_and_library_filter(shelf: AsyncTestClient) -> None:
    one = (await shelf.get("/api/search/documents", params={"q": "parsing", "limit": 1})).json()
    assert len(one) == 1

    elsewhere = await shelf.get(
        "/api/search/documents", params={"q": "parsing", "libraries": "lit"}
    )
    assert [m["doc"] for m in elsewhere.json()] == [
        m["doc"] for m in (await shelf.get("/api/search/documents", params={"q": "parsing"})).json()
    ]


@pytest.mark.parametrize(
    ("name", "limit"),
    [("zero", 0), ("negative", -1), ("above the cap", textsearch.MAX_DOCUMENTS + 1)],
)
async def test_search_documents_rejects_a_bad_limit(name: str, limit: int) -> None:
    with pytest.raises(InvalidInput, match="limit must be 1.."):
        await textsearch.search_documents("parsing", None, limit)


async def test_descriptions_are_read_in_one_query_per_library(shelf: AsyncTestClient) -> None:
    """`describe_of` is the batched read the shortlist uses; absent means no description."""
    found = await Library("lit").describe_of({"compilers.md", "networks.md", "gone.md"})

    assert found == {"compilers.md": "the dragon book"}, "no row for a blank or missing document"
    assert await Library("lit").describe_of(set()) == {}, "nothing asked for, nothing queried"


async def test_results_carry_an_absolute_path_and_position(shelf: AsyncTestClient) -> None:
    """A caller outside the app has to be able to open or grep the file the match came from.

    The index stores paths home-relative so a home stays portable, so the absolute ones are
    derived on read (`Library.resolve_hit`) and have to actually exist.
    """
    match = (await shelf.get("/api/search/documents", params={"q": "parsing"})).json()[0]
    markdown, source = Path(match["markdown_file"]), Path(match["source_file"])

    assert markdown.is_absolute() and source.is_absolute()
    assert markdown.is_file(), "the markdown the line numbers index into"
    assert source.is_file(), "the file that was uploaded"

    lines = markdown.read_text().splitlines()
    span = lines[match["line_start"] - 1 : match["line_end"]]
    assert any("parsing" in line for line in span), "the reported lines contain the match"

    hit = (await shelf.get("/api/libraries/lit/search", params={"q": "parsing"})).json()[0]
    assert Path(hit["markdown_file"]).is_file()
    assert hit["markdown_file"].endswith(hit["markdown_path"]), "absolute is home plus relative"
