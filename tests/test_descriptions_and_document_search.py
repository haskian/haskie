"""Descriptions, renaming at import, and the document shortlist.

A real index, not a fixture row: `search_documents` folds a BM25 ranking, so the ranking has to be
one LanceDB actually produced.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from litestar.testing import AsyncTestClient

from haskie import app as app_module
from haskie import document, textsearch
from haskie.errors import InvalidInput

from conftest import attach_via_api, stage_and_import, wait_import  # isort: skip

pytestmark = pytest.mark.anyio

# "parsing" is in the compiler and the OS document, so a match count alone cannot order them.
PAPERS = {
    "compilers.md": "# Compilers\n\nparsing, parsing and more parsing of grammars\n",
    "os.md": "# Operating Systems\n\nprocesses, scheduling and a little parsing\n",
    "networks.md": "# Networks\n\nrouting and congestion control\n",
}
DESCRIBED = {"compilers.md": "the dragon book", "os.md": "processes and threads"}


@pytest.fixture
async def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seeded_home: Path
) -> AsyncIterator[AsyncTestClient]:
    """The lifespan runs here, so DBOS is started and stopped by the app itself."""
    monkeypatch.setattr(app_module, "WEB_DIST", tmp_path / "no-web-build")
    async with AsyncTestClient(app_module.create_app()) as client:
        await client.post("/api/init", json={"profile": "none"})
        yield client


@pytest.fixture
async def shelf(client: AsyncTestClient) -> AsyncTestClient:
    """One collection of three indexed markdown documents, two of them described."""
    await client.post("/api/collections", json={"name": "lit", "description": "reading list"})
    for name, body in PAPERS.items():
        await stage_and_import(
            client, name, body.encode(), wait=False, description=DESCRIBED.get(name, "")
        )
    for name in PAPERS:
        assert (await wait_import(client, name))["status"] == "imported"
        await attach_via_api(client, "lit", name)
    return client


async def test_a_collection_and_a_document_carry_a_description(shelf: AsyncTestClient) -> None:
    """Set on create and at import, returned by the reads the UI and an agent use."""
    info = (await shelf.get("/api/collections/lit")).json()
    assert info["description"] == "reading list"

    listed = (await shelf.get("/api/collections")).json()["items"]
    assert [row["description"] for row in listed] == ["reading list"], "the sidebar shows it too"

    documents = {
        d["name"]: d["description"] for d in (await shelf.get("/api/documents")).json()["items"]
    }
    assert documents == {
        "compilers.md": "the dragon book",
        "os.md": "processes and threads",
        "networks.md": "",
    }

    members = (await shelf.get("/api/collections/lit/documents")).json()["items"]
    assert {m["document"]["name"]: m["document"]["description"] for m in members} == documents, (
        "a member row carries the document's own description, not a copy per collection"
    )


async def test_a_description_is_editable_after_the_fact(shelf: AsyncTestClient) -> None:
    """Both descriptions are replaceable, and an empty string clears one."""
    collection = await shelf.put(
        "/api/collections/lit/description", json={"description": "CS classics"}
    )
    assert collection.json()["description"] == "CS classics"

    described = await shelf.put(
        "/api/documents/networks.md/description", json={"description": "TCP/IP"}
    )
    assert described.json()["description"] == "TCP/IP"

    cleared = await shelf.put("/api/documents/networks.md/description", json={"description": ""})
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
async def test_rename_at_import(
    client: AsyncTestClient,
    tmp_path: Path,
    name: str,
    filename: str,
    rename_to: str | None,
    expected: str,
) -> None:
    """`name` names the document; the suffix still decides how it is parsed.

    Imported by path, so the file's own name is what a missing `name` falls back to."""
    source = tmp_path / "sources" / filename
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"# Body\n\ntext\n")
    body: dict = {"path": str(source)}
    if rename_to:
        body["name"] = rename_to

    started = await client.post("/api/documents/import", json=body)

    assert started.status_code == 201, f"{name}: {started.text}"
    assert started.json()["name"] == expected, name
    assert (await wait_import(client, expected))["status"] == "imported", name


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
    assert best["collection"] == "lit", "where the best chunk came from"
    assert best["description"] == "the dragon book"
    assert best["chunks"] >= 1 and best["text"], "the evidence for the document being listed"
    assert best["heading"] == "Compilers"


async def test_search_documents_folds_a_shared_document_into_one_row(
    shelf: AsyncTestClient,
) -> None:
    """A document in two collections is still one document to read."""
    await shelf.post("/api/collections", json={"name": "extra"})
    await attach_via_api(shelf, "extra", "compilers.md")

    matches = (await shelf.get("/api/search/documents", params={"q": "parsing"})).json()

    assert [m["doc"] for m in matches].count("compilers.md") == 1
    best = next(m for m in matches if m["doc"] == "compilers.md")
    assert best["collection"] in {"extra", "lit"}, "one of the collections that hold it"
    assert best["description"] == "the dragon book", "the description is the document's"


async def test_search_documents_honours_limit_and_collection_filter(
    shelf: AsyncTestClient,
) -> None:
    one = (await shelf.get("/api/search/documents", params={"q": "parsing", "limit": 1})).json()
    assert len(one) == 1

    filtered = await shelf.get(
        "/api/search/documents", params={"q": "parsing", "collections": "lit"}
    )
    assert [m["doc"] for m in filtered.json()] == [
        m["doc"] for m in (await shelf.get("/api/search/documents", params={"q": "parsing"})).json()
    ]

    unknown = await shelf.get(
        "/api/search/documents", params={"q": "parsing", "collections": "ghost"}
    )
    assert unknown.status_code == 404 and "collection not found: ghost" in unknown.text


@pytest.mark.parametrize(
    ("name", "limit"),
    [("zero", 0), ("negative", -1), ("above the cap", textsearch.MAX_DOCUMENTS + 1)],
)
async def test_search_documents_rejects_a_bad_limit(name: str, limit: int) -> None:
    with pytest.raises(InvalidInput, match="limit must be 1.."):
        await textsearch.search_documents("parsing", None, limit)


async def test_descriptions_are_read_in_one_query(shelf: AsyncTestClient) -> None:
    """`describe_of` is the batched read the shortlist uses; absent means no description. It is a
    document read, not a collection one: a description belongs to the document."""
    found = await document.describe_of({"compilers.md", "networks.md", "gone.md"})

    assert found == {"compilers.md": "the dragon book"}, "no row for a blank or missing document"
    assert await document.describe_of(set()) == {}, "nothing asked for, nothing queried"


async def test_results_carry_an_absolute_path_and_position(shelf: AsyncTestClient) -> None:
    """A caller outside the app has to be able to open or grep the file the match came from.

    The index stores paths home-relative so a home stays portable, so the absolute ones are
    derived on read (`CollectionIndex.hit`) and have to actually exist. They point into the
    document's own folder, not into the collection that matched.
    """
    match = (await shelf.get("/api/search/documents", params={"q": "parsing"})).json()[0]
    markdown, source = Path(match["markdown_file"]), Path(match["source_file"])

    assert markdown.is_absolute() and source.is_absolute()
    assert markdown.is_file(), "the markdown the line numbers index into"
    assert source.is_file(), "the file that was imported"

    lines = markdown.read_text().splitlines()
    span = lines[match["line_start"] - 1 : match["line_end"]]
    assert any("parsing" in line for line in span), "the reported lines contain the match"

    hit = (await shelf.get("/api/collections/lit/search", params={"q": "parsing"})).json()[0]
    assert Path(hit["markdown_file"]).is_file()
    assert hit["markdown_file"].endswith(hit["markdown_path"]), "absolute is home plus relative"
