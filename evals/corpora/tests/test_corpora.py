"""`build.py` and `corpus.py` without the network: a fake `download` serves real PDF and EPUB
bytes, an HTML page behind a .pdf link and a dead link."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from evals.corpora import build, corpus
from evals.corpora.corpus import Collection, Document
from evals.synth import to_pdf

LIST = """### Index

* [Operating Systems](#operating-systems)

### <a id="os"></a>Operating Systems

* [A Book](https://example.org/a.pdf) - Ann Author
* [The Web Book](https://example.org/web/) (HTML)
* [Another Book](https://example.org/b) - Bo Author (PDF)
* [An EPUB](https://example.org/c.epub?raw=1)

### Software Architecture

* [Patterns](https://example.org/p.pdf)
"""
TEXT_PDF = to_pdf(
    "\n".join(f"Line {i} of a book about schedulers and their queues." for i in range(60))
)


def _epub() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
    return buffer.getvalue()


def test_subjects_are_read_with_their_anchors_stripped() -> None:
    assert list(build.subjects(LIST)) == ["Index", "Operating Systems", "Software Architecture"]


def test_candidates_are_links_to_one_pdf_or_epub_in_list_order() -> None:
    found = build.candidates("Operating Systems", build.subjects(LIST)["Operating Systems"])

    assert [c.url for c in found] == [
        "https://example.org/a.pdf",
        "https://example.org/b",  # marked (PDF)
        "https://example.org/c.epub?raw=1",
    ]
    assert found[0].title == "A Book"


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (b"<!doctype html><html>", "not a PDF or EPUB"),
        (to_pdf("Too short to be a book."), "a scan?"),
        (b"PK\x03\x04 not really a zip", "not an EPUB"),
    ],
)
def test_a_file_haskie_cannot_index_is_rejected_with_the_reason(data: bytes, reason: str) -> None:
    with pytest.raises(build.Rejected, match=reason):
        build.kind(data)


def test_a_pdf_with_text_and_an_epub_are_kept() -> None:
    assert build.kind(TEXT_PDF) == "pdf"
    assert build.kind(_epub()) == "epub"


def test_collect_keeps_the_first_usable_books_and_records_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served = {
        "https://example.org/a.pdf": TEXT_PDF,
        "https://example.org/b": b"<html>a landing page</html>",
        "https://example.org/c.epub?raw=1": _epub(),
        "https://example.org/d.pdf": TEXT_PDF,  # the same file as a.pdf
    }

    def download(url: str, timeout: float = 120) -> bytes:
        if url not in served:
            raise OSError("no such host")
        return served[url]

    monkeypatch.setattr(corpus, "download", download)
    monkeypatch.setattr(build, "RETRY_SECONDS", 0)
    found = [
        build.Candidate("OS", title, url)
        for title, url in [
            ("A Book", "https://example.org/a.pdf"),
            ("Dead", "https://example.org/gone.pdf"),
            ("Landing", "https://example.org/b"),
            ("Copy", "https://example.org/d.pdf"),
            ("An EPUB", "https://example.org/c.epub?raw=1"),
            ("Never tried", "https://example.org/e.pdf"),
        ]
    ]

    collection, skipped = build.collect("Operating Systems", found, tmp_path, per_collection=2)

    assert collection.name == "open-operating-systems"
    assert [d.name for d in collection.documents] == ["a-book.pdf", "an-epub.epub"]
    assert [(s.title, s.reason.split(":")[0]) for s in skipped] == [
        ("Dead", "download failed"),
        ("Landing", "not a PDF or EPUB"),
        ("Copy", "same file as a kept book"),
    ]
    assert (tmp_path / "open-operating-systems" / "a-book.pdf").read_bytes() == TEXT_PDF


def test_fetch_refuses_a_file_that_no_longer_matches_its_hash(tmp_path: Path) -> None:
    document = Document("a.pdf", "A", "https://example.org/a.pdf", corpus.sha256(b"old"), 3)
    collection = Collection("open-x", "X", [document])
    (tmp_path / "open-x").mkdir()
    (tmp_path / "open-x" / "a.pdf").write_bytes(b"new")

    with pytest.raises(corpus.HashMismatch):
        corpus.fetch(collection, tmp_path)


def test_the_manifest_round_trips(tmp_path: Path) -> None:
    document = Document("a.pdf", "A", "https://example.org/a.pdf", "0" * 64, 3)
    manifest = corpus.Manifest(
        "https://example.org/list.md", [Collection("open-x", "X", [document])]
    )

    corpus.dump(manifest, tmp_path / "manifest.json")

    assert corpus.load(tmp_path / "manifest.json") == manifest


def test_a_landing_page_is_followed_to_the_file_named_like_the_book(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = (
        '<a href="chapters/intro.pdf">Intro</a> <a href="/think-os.pdf?v=2">Whole book</a> '
        '<a href="chapters/intro.pdf">again</a>'
    )
    served = {
        "https://example.org/os/index.html": page.encode(),
        "https://example.org/think-os.pdf?v=2": TEXT_PDF,
    }
    monkeypatch.setattr(corpus, "download", lambda url, timeout=120: served[url])
    candidate = build.Candidate("OS", "Think OS", "https://example.org/os/index.html")

    links = build.linked_files(page, candidate.url)
    data, extension, url = build.resolve(candidate)

    assert links == [
        "https://example.org/os/chapters/intro.pdf",
        "https://example.org/think-os.pdf?v=2",
    ]
    assert (data, extension, url) == (TEXT_PDF, "pdf", "https://example.org/think-os.pdf?v=2")


def test_a_page_with_no_file_on_it_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(corpus, "download", lambda url, timeout=120: b"<html>read online</html>")

    with pytest.raises(build.Rejected, match="not a PDF or EPUB"):
        build.resolve(build.Candidate("OS", "Web Book", "https://example.org/web/"))


def test_an_excluded_book_is_skipped_with_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build, "EXCLUDED", {"https://example.org/a.pdf": "haskie finds no text"})
    monkeypatch.setattr(corpus, "download", lambda url, timeout=120: TEXT_PDF)
    found = [build.Candidate("OS", "A Book", "https://example.org/a.pdf")]

    collection, skipped = build.collect("OS", found, tmp_path)

    assert collection.documents == []
    assert [s.reason for s in skipped] == ["haskie finds no text"]
