"""`setup.py` is idempotent by checking before writing, not by writing and catching a conflict -
these tests are what proves each check actually skips the write it guards, using a fake server
that records every call it receives so a test can assert on what was and was not sent.
"""

from pathlib import Path

import pytest

from evals import setup


class NotFound(Exception):
    """Stands in for the real `urllib.error.HTTPError` with a 404 status."""


class FakeServer:
    """Enough of the API's shape to drive `setup.py`'s logic without a real haskie instance.

    `documents[name]["in_collections"]` mirrors `GET /api/documents/{name}/collections`.
    `calls` records every request made, so a test can assert a write was skipped rather than
    merely that it didn't crash.
    """

    def __init__(self, collections: dict | None = None, documents: dict | None = None) -> None:
        self.collections = collections or {}
        self.documents = documents or {}
        self.calls: list[tuple[str, str, dict | None]] = []

    def call(self, method: str, path: str, body: dict | None = None):
        self.calls.append((method, path, body))
        if method == "GET" and path.startswith("/api/collections/"):
            name = path.removeprefix("/api/collections/")
            if name not in self.collections:
                raise NotFound(path)
            return self.collections[name]
        if method == "GET" and path.startswith("/api/documents/") and path.endswith("/collections"):
            name = path.removeprefix("/api/documents/").removesuffix("/collections")
            return self.documents[name]["in_collections"]
        if method == "GET" and path.startswith("/api/documents/"):
            name = path.removeprefix("/api/documents/")
            if name not in self.documents:
                raise NotFound(path)
            return self.documents[name]
        if method == "POST":
            assert body is not None, f"POST {path} with no body"
            if path == "/api/collections":
                self.collections[body["name"]] = {"name": body["name"]}
                return {}
            if path == "/api/documents/import":
                name = Path(body["path"]).name
                self.documents[name] = {"name": name, "status": "imported", "in_collections": []}
                return {"name": name}
            if path.endswith("/documents"):
                collection = path.removeprefix("/api/collections/").removesuffix("/documents")
                self.documents[body["document"]]["in_collections"].append(collection)
                return {}
        if method == "PUT" and path.endswith("/description"):
            assert body is not None, f"PUT {path} with no body"
            collection = path.removeprefix("/api/collections/").removesuffix("/description")
            self.collections[collection]["description"] = body["description"]
            return {}
        if method == "PUT" and path.endswith("/overrides"):
            assert body is not None, f"PUT {path} with no body"
            collection = path.removeprefix("/api/collections/").removesuffix("/overrides")
            self.collections[collection].setdefault("overrides", {}).update(body)
            return {}
        raise AssertionError(f"unhandled: {method} {path}")


def _install(monkeypatch, server: FakeServer) -> None:
    monkeypatch.setattr(
        setup, "call", lambda method, path, api, body=None: server.call(method, path, body)
    )

    def get_or_none(path: str, api: str):
        try:
            return server.call("GET", path, None)
        except NotFound:
            return None

    monkeypatch.setattr(setup, "get_or_none", get_or_none)


def _posts(server: FakeServer) -> list[tuple[str, str, dict | None]]:
    return [call for call in server.calls if call[0] == "POST"]


def test_ensure_collection_creates_only_when_missing(monkeypatch) -> None:
    server = FakeServer()
    _install(monkeypatch, server)

    setup.ensure_collection("books", "desc", "http://x")

    assert _posts(server) == [
        ("POST", "/api/collections", {"name": "books", "description": "desc"})
    ]


def test_ensure_collection_skips_the_write_when_it_already_exists(monkeypatch) -> None:
    server = FakeServer(collections={"books": {"name": "books", "description": "desc"}})
    _install(monkeypatch, server)

    setup.ensure_collection("books", "desc", "http://x")

    assert [call for call in server.calls if call[0] != "GET"] == []


def test_ensure_collection_rewrites_a_stale_description(monkeypatch) -> None:
    server = FakeServer(collections={"books": {"name": "books", "description": "old"}})
    _install(monkeypatch, server)

    setup.ensure_collection("books", "new", "http://x")

    assert [call for call in server.calls if call[0] != "GET"] == [
        ("PUT", "/api/collections/books/description", {"description": "new"})
    ]
    assert server.collections["books"]["description"] == "new"


def test_import_all_adopts_an_existing_document_without_reimporting(
    monkeypatch, tmp_path: Path
) -> None:
    server = FakeServer(documents={"a.pdf": {"name": "a.pdf", "status": "imported", "size": 1}})
    _install(monkeypatch, server)
    file = tmp_path / "a.pdf"
    file.write_text("x")

    names = setup.import_all([file], "http://x")

    assert names == ["a.pdf"]
    assert not _posts(server)


def test_import_all_refuses_a_name_imported_earlier_with_different_content(
    monkeypatch, tmp_path: Path
) -> None:
    """Documents are keyed by name, so without this a regenerated synthetic corpus would keep
    being searched as the stale content imported under the same name."""
    server = FakeServer(documents={"a.md": {"name": "a.md", "status": "imported", "size": 99}})
    _install(monkeypatch, server)
    file = tmp_path / "a.md"
    file.write_text("x")

    with pytest.raises(RuntimeError, match="different content"):
        setup.import_all([file], "http://x")
    assert not _posts(server)


def test_ensure_chunking_writes_only_when_the_settings_differ(monkeypatch) -> None:
    server = FakeServer(collections={"c": {"name": "c", "overrides": {"chunk_size": None}}})
    _install(monkeypatch, server)

    setup.ensure_chunking("c", setup.SMALL_CHUNKS, "http://x")
    setup.ensure_chunking("c", setup.SMALL_CHUNKS, "http://x")

    puts = [call for call in server.calls if call[0] == "PUT"]
    assert puts == [("PUT", "/api/collections/c/overrides", setup.SMALL_CHUNKS)]


def test_import_all_imports_a_file_not_seen_before(monkeypatch, tmp_path: Path) -> None:
    server = FakeServer()
    _install(monkeypatch, server)
    file = tmp_path / "b.pdf"
    file.write_text("x")

    names = setup.import_all([file], "http://x")

    assert names == ["b.pdf"]
    assert _posts(server) == [("POST", "/api/documents/import", {"path": str(file.resolve())})]


def test_attach_all_skips_a_document_already_in_the_collection(monkeypatch) -> None:
    server = FakeServer(documents={"a.pdf": {"name": "a.pdf", "in_collections": ["books"]}})
    _install(monkeypatch, server)

    setup.attach_all(["a.pdf"], "books", "http://x")

    assert not _posts(server)


def test_attach_all_attaches_a_document_not_yet_in_the_collection(monkeypatch) -> None:
    server = FakeServer(documents={"a.pdf": {"name": "a.pdf", "in_collections": []}})
    _install(monkeypatch, server)

    setup.attach_all(["a.pdf"], "books", "http://x")

    assert _posts(server) == [("POST", "/api/collections/books/documents", {"document": "a.pdf"})]
