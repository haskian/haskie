"""Pure checks: chunking, conversion, settings, document and collection IO, the index and the
pipeline, with HASKIE_HOME pointed at a temp dir (see conftest.py).

Nothing here launches DBOS: the pipeline stages are called straight through, in the order the
workflows call them. Every test that needs the durable runtime lives in `tests/test_workflows.py`;
the HTTP contract lives in `tests/test_api.py`.
"""

import base64
import hashlib
import io
import math
import random
import sqlite3
import sys
import threading
import types
import zipfile
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from datetime import timedelta
from functools import partial
from pathlib import Path

import anyio
import lancedb
import msgspec
import numpy as np
import pyarrow as pa
import pytest
import structlog
from conftest import (
    MD,
    collection_hits,
    document_names,
    events,
    id_of,
    import_row,
    index_hits,
    legacy_index,
    maintenance_state,
    refresh_settled,
    remove_collection,
    text_pdf,
    until,
)
from sqlalchemy import event, insert, select, update
from sqlalchemy.exc import IntegrityError

from haskie import audit, db, home, ids, logs, tables
from haskie.catalogue.catalogue import EmbeddingModel
from haskie.collection import index as index_module
from haskie.collection import maintenance
from haskie.collection.collection import Collection, DocumentCounts, Member, MemberStatus
from haskie.collection.index import (
    EMBEDDING_KEY,
    FTS_COLUMN,
    PLAIN_SCHEMA,
    CollectionIndex,
    Hit,
    IndexStats,
    Row,
    _fusion,
    _partitions,
    row_score,
)
from haskie.document import convert, document
from haskie.document.document import Document, DocumentStatus
from haskie.errors import (
    Conflict,
    HaskieError,
    InvalidInput,
    NotFound,
    NotReady,
    PermanentError,
)
from haskie.indexing import chunk, embed, embed_cache, onnx_models, pipeline
from haskie.indexing.chunk import Chunk, Piece, Position
from haskie.indexing.segment import CutReason, PieceType
from haskie.paging import Order, PageRequest
from haskie.sections.build import Section
from haskie.settings import (
    DEFAULT_RERANKER,
    Accelerator,
    Chunker,
    ChunkSettings,
    CollectionOverrides,
    ConversionSettings,
    Descriptors,
    Fusion,
    Parser,
    PipelineSettings,
    Reranker,
    SearchMode,
    SearchOverrides,
    SearchSettings,
    UserSettings,
    docs,
    init_user_settings,
    load_user_settings,
    load_user_settings_or_none,
    save_user_settings,
)

SMALL = ChunkSettings(chunk_size=40)

# A 1x1 transparent PNG: the smallest real image file, enough to exercise the preview branch.
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

_DOCX_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""  # noqa: E501
_DOCX_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Target="word/document.xml"
 Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"/>
</Relationships>"""
_DOCX_DOCUMENT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Quarterly report</w:t></w:r></w:p>
<w:p><w:r><w:t>Revenue grew by twelve percent.</w:t></w:r></w:p>
</w:body></w:document>"""


def docx_bytes() -> bytes:
    """A real (minimal) OOXML package: the three parts anydoc needs to read a Word document."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", _DOCX_CONTENT_TYPES)
        archive.writestr("_rels/.rels", _DOCX_RELS)
        archive.writestr("word/document.xml", _DOCX_DOCUMENT)
    return buffer.getvalue()


async def _staging_rows() -> list[tuple[str, str, int]]:
    """Every `staging` row as (id, filename, size): what `stage` commits beside the bytes."""
    async with db.connect() as conn:
        rows = await conn.execute(
            select(
                tables.staging.c.staging_id, tables.staging.c.filename, tables.staging.c.size
            ).order_by(tables.staging.c.staging_id)
        )
        return [tuple(row) for row in rows]


async def attachable(name: str, content: bytes | str = MD, **options) -> Document:
    """`imported`, moved on to the status the pipeline ends at: `Collection.add` takes only an
    imported document, so a test that attaches one has to get it there first."""
    doc = await import_row(name, content, **options)
    await document.set_status(doc.id, DocumentStatus.IMPORTED)
    return await document.named(doc.name)


# --- settings ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "overrides", "user", "expected"),
    [
        (
            "no overrides -> user defaults",
            CollectionOverrides(),
            UserSettings(),
            (1200, 66, "m", True),
        ),
        (
            "override size only",
            CollectionOverrides(chunk_size=990),
            UserSettings(),
            (990, 66, "m", True),
        ),
        (
            "override chunker only",
            CollectionOverrides(chunker=Chunker.TEXT),
            UserSettings(),
            (1200, 66, "t", True),
        ),
        (
            "override every field",
            CollectionOverrides(
                chunker=Chunker.TEXT, chunk_size=500, chunk_merge_below=10, chunk_frame=False
            ),
            UserSettings(),
            (500, 10, "t", False),
        ),
        (
            "an unset field follows the user value",
            CollectionOverrides(chunk_merge_below=5),
            UserSettings(conversion=ConversionSettings(chunk_size=300)),
            (300, 5, "m", True),
        ),
        (
            "the heading path switched off for the user, back on for one collection",
            CollectionOverrides(chunk_frame=True),
            UserSettings(conversion=ConversionSettings(chunk_frame=False)),
            (1200, 66, "m", True),
        ),
        (
            "switched off for the user, inherited",
            CollectionOverrides(),
            UserSettings(conversion=ConversionSettings(chunk_frame=False)),
            (1200, 66, "m", False),
        ),
    ],
)
def test_collection_settings_resolve_into_chunk_settings(
    name: str, overrides: CollectionOverrides, user: UserSettings, expected: tuple
) -> None:
    """A collection only overrides how the shared markdown is split: `parser`/`skip_ocr_pages`
    belong to the document, chosen once at import."""
    effective = overrides.resolve(user)

    assert isinstance(effective, ChunkSettings), name
    merge, framed = effective.chunk_merge_below, effective.chunk_frame
    assert (effective.chunk_size, merge, effective.chunker[0], framed) == expected, name


def test_conversion_settings_carry_the_chunking_defaults() -> None:
    user = ConversionSettings(
        chunker=Chunker.TEXT, chunk_size=700, parser=Parser.PLAIN, chunk_frame=False
    )
    assert user.chunking == ChunkSettings(chunker=Chunker.TEXT, chunk_size=700, chunk_frame=False)
    assert CollectionOverrides().resolve(UserSettings(conversion=user)) == user.chunking


def test_search_overrides_resolve_per_field() -> None:
    user = SearchSettings(limit=5, fusion=Fusion.RRF, vector_weight=0.7)
    merged = SearchOverrides(fusion=Fusion.LINEAR, bm25_weight=0.9).resolve(user)
    assert (merged.limit, merged.fusion, merged.vector_weight, merged.bm25_weight) == (
        5,
        "linear",
        0.7,
        0.9,
    )
    assert SearchOverrides().resolve(user) == user


def test_every_setting_has_title_and_definition() -> None:
    user_docs = docs()
    assert set(user_docs) >= {
        "embedding", "conversion.parser", "conversion.chunk_size", "pipeline.cpu_budget",
        "pipeline.batch_pages", "search.limit", "search.reranker", "search.reranker_model",
        "pipeline.maintenance_documents", "pipeline.maintenance_idle_seconds",
        "pipeline.ann_min_rows", "search.nprobes", "search.refine_factor",
    }  # fmt: skip
    assert user_docs["pipeline.maintenance_documents"].title == "Maintenance after documents"
    assert user_docs["search.nprobes"].title == "Vector probes"
    assert user_docs["conversion.chunk_size"].title == "Chunk size (characters)"
    assert "characters" in user_docs["conversion.chunk_size"].description
    assert user_docs["conversion.chunk_frame"].title == "Prepend heading path"
    assert all(d.title and d.description for d in user_docs.values())
    # collection overrides and the chunk settings reuse the same definitions
    collection_docs = docs(CollectionOverrides)
    chunking = {"chunker", "chunk_size", "chunk_merge_below", "chunk_frame"}
    search = {key for key in user_docs if key.startswith("search.")}
    assert set(collection_docs) == chunking | search, "chunking and search, and nothing else"
    assert collection_docs["chunk_size"] == user_docs["conversion.chunk_size"]
    assert collection_docs["search.reranker"] == user_docs["search.reranker"]
    assert docs(ChunkSettings)["chunker"] == user_docs["conversion.chunker"]


def test_docs_rejects_a_non_struct() -> None:
    with pytest.raises(TypeError, match="needs a msgspec Struct"):
        docs(int)  # ty: ignore[invalid-argument-type]


@pytest.mark.anyio
async def test_user_settings_persist_in_db() -> None:
    assert await load_user_settings_or_none() is None, "not initialized yet"
    assert (await load_user_settings()).embedding == "none"
    await save_user_settings(UserSettings(embedding="granite-97m-multilingual"))
    assert await load_user_settings_or_none() is not None
    assert (await load_user_settings()).embedding == "granite-97m-multilingual"


@pytest.mark.anyio
async def test_init_user_settings_creates_the_row_once() -> None:
    assert await init_user_settings(UserSettings(embedding="granite-97m-multilingual")) is True
    assert await init_user_settings(UserSettings(embedding="granite-english")) is False, (
        "second call loses"
    )
    assert (await load_user_settings()).embedding == "granite-97m-multilingual", (
        "the first write stands"
    )


# --- conversion --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "pages", "skip", "expect"),
    [
        ("mixed, skip off -> refused", ["one", None, "two", None], False, "raise"),
        (
            "mixed, skip on -> text pages kept, OCR pages marked",
            ["one", None, "two", None],
            True,
            "ok",
        ),
        ("all OCR, skip on -> still fails", [None, None], True, "raise"),
        ("no OCR pages, skip on -> unchanged", ["one", "two"], True, "ok"),
    ],
)
def test_skip_ocr_pages(tmp_path: Path, name: str, pages: list, skip: bool, expect: str) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(text_pdf(pages))
    markdown, ocr_pages, total = convert.pdf_pages_markdown(pdf, skip_ocr_pages=skip)
    if expect == "raise":
        with pytest.raises(PermanentError, match="need OCR"):
            convert.check_ocr_policy(len(ocr_pages), total, skip)
        return
    convert.check_ocr_policy(len(ocr_pages), total, skip)  # no raise: the policy accepts it
    for i, text in enumerate(pages, start=1):
        if text is None:
            assert f"<!-- page {i}: needs OCR, skipped -->" in markdown, name
        else:
            assert text in markdown, name
    preview = convert.build_preview(pdf, tmp_path / "prev", Parser.ANYDOC)
    assert preview.ocr_pages == [i for i, t in enumerate(pages, start=1) if t is None], name


@pytest.mark.parametrize(
    ("name", "filename", "content", "parser", "expected"),
    [
        ("markdown -> read as text", "n.md", MD.encode(), "anydoc", MD),
        ("plain parser reads any text file", "n.md", b"# H\n", "plain", "# H\n"),
        ("html -> raw source", "p.html", b"<h1>T</h1>", "anydoc", "<h1>T</h1>"),
        ("office file -> anydoc markdown", "r.docx", None, "anydoc", "Quarterly report"),
        ("image -> no extractable text", "pic.png", PNG_1X1, "anydoc", ""),
    ],
)
def test_to_markdown_by_suffix(
    tmp_path: Path, name: str, filename: str, content: bytes | None, parser, expected: str
) -> None:
    path = tmp_path / filename
    path.write_bytes(docx_bytes() if content is None else content)
    markdown = convert.to_markdown(path, parser)
    assert expected in markdown, name


@pytest.mark.parametrize(
    ("name", "filename", "content", "error", "match"),
    [
        ("unknown suffix", "x.zip", b"PK", PermanentError, "unsupported file type: .zip"),
        ("no suffix at all", "README", b"text", PermanentError, "unsupported file type"),
        ("corrupt office file", "broken.docx", b"not a zip", PermanentError, "broken.docx: "),
        ("a pdf, which converts page-wise", "book.pdf", b"%PDF-1.4", ValueError, "page-wise"),
    ],
)
def test_to_markdown_rejects_what_it_cannot_read(
    tmp_path: Path, name: str, filename: str, content: bytes, error: type[Exception], match: str
) -> None:
    path = tmp_path / filename
    path.write_bytes(content)
    with pytest.raises(error, match=match):
        convert.to_markdown(path, Parser.ANYDOC)


def test_a_corrupt_pdf_is_a_conversion_error(tmp_path: Path) -> None:
    """A parser failure is a property of the file, so it must not be retried (see `pipeline`)."""
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf")
    with pytest.raises(PermanentError, match="broken.pdf: "):
        convert.pdf_pages_markdown(path)


@pytest.mark.parametrize(
    ("name", "filename", "content", "kind", "source_starts"),
    [
        ("text file", "n.md", MD.encode(), "text", b"# Title"),
        ("html file", "p.html", b"<h1>T</h1>", "html", b"<h1>T</h1>"),
        ("office file -> html stand-in", "r.docx", None, "html", b"<p>"),
        ("image file -> the image itself", "pic.png", PNG_1X1, "image", b"\x89PNG"),
    ],
)
def test_build_preview_writes_both_panes(
    tmp_path: Path, name: str, filename: str, content: bytes | None, kind: str, source_starts: bytes
) -> None:
    path = tmp_path / filename
    path.write_bytes(docx_bytes() if content is None else content)
    out = tmp_path / "preview"

    info = convert.build_preview(path, out, Parser.ANYDOC)

    assert info.kind == kind, name
    assert (out / "source").read_bytes().startswith(source_starts), name
    assert (out / "preview.md").exists(), name
    assert list(out.glob("*.tmp")) == [], "atomic writes leave no temp file"


def test_build_preview_of_a_pdf_cuts_to_the_first_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "long.pdf"
    pdf.write_bytes(text_pdf([f"page {i}" for i in range(convert.PREVIEW_PAGES + 3)]))

    info = convert.build_preview(pdf, tmp_path / "p", Parser.ANYDOC)

    assert (info.kind, info.truncated, info.pages) == ("pdf", True, convert.PREVIEW_PAGES)
    assert "page 11" not in (tmp_path / "p" / "preview.md").read_text()


def test_build_preview_of_a_corrupt_pdf_raises_conversion_error(tmp_path: Path) -> None:
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(PermanentError, match="broken.pdf"):
        convert.build_preview(bad, tmp_path / "p", Parser.ANYDOC)


def test_pdf_bookmarks_of_a_corrupt_file_raises_conversion_error(tmp_path: Path) -> None:
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(PermanentError, match="broken.pdf"):
        convert.pdf_bookmarks(bad)


# --- documents: staging and import -------------------------------------------------


@pytest.mark.parametrize(
    ("name", "filename", "rename_to", "expected"),
    [
        ("the file name, cleaned", "guide.md", None, "guide.md"),
        ("a path keeps only its last segment", "/tmp/deep/guide.md", None, "guide.md"),
        ("unsafe characters collapse into one dash", "a b  c.md", None, "a-b-c.md"),
        ("a rename with no suffix keeps the original's", "book.pdf", "My Book", "my-book.pdf"),
        ("a rename with the same suffix is taken as is", "book.pdf", "atlas.pdf", "atlas.pdf"),
        ("a rename to another suffix keeps the original's", "book.pdf", "a.txt", "a-txt.pdf"),
        ("upper case is lowered, the suffix too", "BOOK.PDF", None, "book.pdf"),
        (
            "every run of dots, spaces and punctuation in the stem is one dash",
            "2013-Vaughn-Implementing Domain  Driven_Design (v1.2).pdf",
            None,
            "2013-vaughn-implementing-domain-driven-design-v1-2.pdf",
        ),
        ("accents are dropped, and letters fold", "Résumé Straße.md", None, "resume-strasse.md"),
        ("a rename is spelled the same way", "a.md", "My  Notes!", "my-notes.md"),
        ("underscores, dashes and inner dots alike", "A_B--CC.d.f.pdf", None, "a-b-cc-d-f.pdf"),
    ],
)
def test_stored_name_keeps_the_suffix_the_parser_is_chosen_by(
    name: str, filename: str, rename_to: str | None, expected: str
) -> None:
    assert document.stored_name(filename, rename_to) == expected, name


@pytest.mark.parametrize(
    ("name", "filename", "error", "match"),
    [
        ("a type nothing can read", "virus.exe", PermanentError, "unsupported file type"),
        ("no suffix at all", "README", PermanentError, "unsupported file type"),
        ("nothing left after cleaning, so no suffix", "***", PermanentError, "unsupported"),
        ("a stem with no letter or digit", "_.md", InvalidInput, "invalid name"),
        ("a name with no Latin letter or digit", "Отчёт.pdf", InvalidInput, "invalid name"),
    ],
)
def test_stored_name_refuses_what_could_never_be_imported(
    name: str, filename: str, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        document.stored_name(filename)


@pytest.mark.anyio
async def test_stage_writes_the_upload_and_commits_nothing() -> None:
    """Upload is two-phase: the bytes land in `staging/` with a `staging` row beside them, and no
    name is taken and no document row is created until the import."""
    staged = await document.stage("My Guide.md", MD.encode())

    assert staged.filename == "My Guide.md", "the name the user uploaded, for the import form"
    assert staged.size == len(MD.encode())
    assert document.STAGING_ID.match(staged.staging_id), "an id this module can turn into a path"
    path = document.staging_path(staged.staging_id)
    assert path.parent == home.STAGING_ROOT and path.read_text() == MD
    assert await _staging_rows() == [(staged.staging_id, "My Guide.md", len(MD.encode()))]
    assert await document_names() == [], "no document committed yet"


@pytest.mark.anyio
async def test_stage_refuses_what_could_never_be_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(PermanentError, match="unsupported file type"):
        await document.stage("virus.exe", b"x")
    monkeypatch.setattr(document, "UPLOAD_MAX_BYTES", 8)
    with pytest.raises(InvalidInput, match="file larger than 8 bytes: 20"):
        await document.stage("big.md", b"x" * 20)
    assert list(home.STAGING_ROOT.iterdir()) == [], "nothing written for a rejected upload"


@pytest.mark.anyio
async def test_import_staged_moves_the_file_and_creates_the_row() -> None:
    staged = await document.stage("My Guide.md", MD.encode())

    doc = await document.import_staged(
        staged.staging_id,
        document.ImportOptions(
            name="My Guide.md",
            description="the guide",
            parser=Parser.PLAIN,
            skip_ocr_pages=False,
        ),
    )

    assert (doc.name, doc.suffix, doc.size) == ("my-guide.md", ".md", len(MD.encode()))
    assert (doc.status, doc.error, doc.preview) == ("queued", None, None), "the pipeline starts it"
    assert (doc.parser, doc.skip_ocr_pages) == ("plain", False), "conversion is fixed at import"
    assert doc.description == "the guide"
    assert doc.original.read_text() == MD
    assert not document.staging_path(staged.staging_id).exists(), "moved, not copied"
    assert await _staging_rows() == [], "the staging row goes with the bytes"
    assert await document_names() == ["my-guide.md"]


@pytest.mark.anyio
async def test_the_bytes_are_the_document_whichever_way_they_came_in(tmp_path: Path) -> None:
    """A document's id is the MD5 of its bytes, taken at staging and at a path import alike. The
    same bytes again are refused under any name, naming the document they already are; staging
    them says so first. Other bytes are another document."""
    md5 = ids.md5(MD.encode())
    first = await document.stage("guide.md", MD.encode())
    assert first.duplicate is None, "nothing imported yet"
    staged = await document.import_staged(first.staging_id)
    copy = tmp_path / "copy.md"
    copy.write_text(MD)

    with pytest.raises(Conflict, match="this file is already imported as guide.md"):
        await document.import_path(str(copy), document.ImportOptions(name="b-copy.md"))
    again = await document.stage("renamed.md", MD.encode())
    other = await document.stage("other.md", b"# Other\n\nnot the guide\n")

    assert staged.id == md5, "the id is the MD5 of the bytes"
    assert await document_names() == ["guide.md"], "the refused import left no row"
    assert not (document.root(md5) / "original.md.md").exists()
    assert again.duplicate == "guide.md", "staging names the document the bytes already are"
    assert other.duplicate is None, "other bytes are another file"
    with pytest.raises(Conflict, match="this file is already imported as guide.md"):
        await document.import_staged(again.staging_id)
    await document.set_status(md5, DocumentStatus.DELETING)
    assert await document.identical(md5) is None, "one being deleted is no longer a repeat"
    leaving = await document.stage("back.md", MD.encode())
    with pytest.raises(Conflict, match="this file is guide.md, being deleted; import it once"):
        await document.import_staged(leaving.staging_id)


@pytest.mark.anyio
async def test_import_staged_without_a_name_keeps_the_uploaded_file_name() -> None:
    """The staging id carries only the suffix, so the uploaded name comes off the staging row."""
    staged = await document.stage("My Guide.md", MD.encode())

    doc = await document.import_staged(staged.staging_id)

    assert doc.name == "my-guide.md"
    assert await _staging_rows() == [], "the row is consumed with the staged file"


@pytest.mark.anyio
async def test_import_staged_defaults_conversion_to_the_user_settings() -> None:
    await save_user_settings(
        UserSettings(conversion=ConversionSettings(parser=Parser.PLAIN, skip_ocr_pages=False))
    )
    staged = await document.stage("g.md", MD.encode())

    doc = await document.import_staged(staged.staging_id)

    assert (doc.parser, doc.skip_ocr_pages) == ("plain", False)


@pytest.mark.parametrize(
    ("second", "reason"),
    [
        pytest.param("Notes.md", "the same name", id="exact"),
        pytest.param("NOTES.md", "spelled one way, so another case is the same name", id="case"),
        pytest.param("notes .md", "and other punctuation too", id="punctuation"),
    ],
)
@pytest.mark.anyio
async def test_a_name_taken_in_any_spelling_is_refused(second: str, reason: str) -> None:
    """A name is stored in one spelling (`stored_name`), so a second import that spells it
    another way asks for the same name, and is refused with the first one untouched."""
    first = await document.stage("Notes.md", b"# first\n")
    doc = await document.import_staged(first.staging_id)
    clash = await document.stage(second, b"# second\n")

    with pytest.raises(Conflict, match="document already exists: notes.md"):
        await document.import_staged(clash.staging_id)

    assert await document_names() == ["notes.md"], reason
    assert doc.original.read_bytes() == b"# first\n", "the first document's file is untouched"


@pytest.mark.anyio
async def test_import_staged_renames_and_refuses_a_name_already_taken() -> None:
    first = await document.stage("book.pdf", text_pdf(["one"]))
    second = await document.stage("other.pdf", text_pdf(["two"]))

    doc = await document.import_staged(
        first.staging_id, document.ImportOptions(name="Atlas of Maps")
    )

    assert doc.name == "atlas-of-maps.pdf", "the rename names the document, not the parser"
    with pytest.raises(Conflict, match="document already exists"):
        await document.import_staged(
            second.staging_id, document.ImportOptions(name="Atlas of Maps")
        )
    assert document.staging_path(second.staging_id).exists(), "the refused upload is still staged"
    assert [row[0] for row in await _staging_rows()] == [second.staging_id], "and so is its row"
    assert await document_names() == ["atlas-of-maps.pdf"]


@pytest.mark.parametrize(
    ("name", "staging_id", "error", "match"),
    [
        ("a well-formed id nothing was staged under", f"{'0' * 32}.md", NotFound, "staged"),
        ("an id this module never issued", "../../etc/passwd", InvalidInput, "invalid staging id"),
        ("an empty id", "", InvalidInput, "invalid staging id"),
    ],
)
@pytest.mark.anyio
async def test_import_staged_rejects_an_id_it_did_not_issue(
    name: str, staging_id: str, error: type[Exception], match: str
) -> None:
    """The staging id is a trust boundary: only what `stage` produced becomes a path."""
    with pytest.raises(error, match=match):
        await document.import_staged(staging_id)
    assert await document_names() == [], name


@pytest.mark.anyio
async def test_sweep_staging_deletes_only_the_uploads_nobody_imported() -> None:
    """The TTL is read off the staging row; a file with no row at all is an interrupted `stage`,
    so it goes too once it is that old."""
    import os

    old = await document.stage("old.md", MD.encode())
    fresh = await document.stage("fresh.md", MD.encode())
    stale = document.staging_path(old.staging_id)
    async with db.connect() as conn:
        await conn.execute(
            update(tables.staging)
            .where(tables.staging.c.staging_id == old.staging_id)
            .values(created_at=0)
        )
    orphan = home.STAGING_ROOT / f"{'0' * 32}.md"  # bytes written, the row never landed
    orphan.write_text("x")
    os.utime(orphan, (0, 0))  # far older than any cutoff
    (home.STAGING_ROOT / "not-ours.txt").write_text("x")

    assert await document.sweep_staging(max_age_seconds=3600) == 2

    assert not stale.exists(), "an upload nobody imported is only bytes"
    assert not orphan.exists(), "and neither is one whose row never landed"
    assert [row[0] for row in await _staging_rows()] == [fresh.staging_id], "the expired row goes"
    assert document.staging_path(fresh.staging_id).exists(), "the recent one stays"
    assert (home.STAGING_ROOT / "not-ours.txt").exists(), "a file this module never wrote"


@pytest.mark.anyio
async def test_sweep_staging_without_a_staging_directory_is_a_no_op() -> None:
    home.STAGING_ROOT.rmdir()  # made by the test home fixture, still empty
    assert await document.sweep_staging(max_age_seconds=0) == 0


@pytest.mark.parametrize(
    ("name", "path", "error", "match"),
    [
        ("relative path", "notes/a.md", InvalidInput, "path must be absolute"),
        ("absolute but missing", "/definitely/not/here/a.md", InvalidInput, "file not found"),
        ("a directory, not a file", "{tmp}", InvalidInput, "file not found"),
        ("unsupported suffix", "{tmp}/v.exe", PermanentError, "unsupported file type"),
        ("over the upload cap", "{tmp}/big.md", InvalidInput, "file larger than"),
    ],
)
@pytest.mark.anyio
async def test_import_path_validates_the_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    path: str,
    error: type[Exception],
    match: str,
) -> None:
    monkeypatch.setattr(document, "UPLOAD_MAX_BYTES", 16)
    outside = tmp_path / "incoming"
    outside.mkdir(parents=True, exist_ok=True)
    (outside / "v.exe").write_bytes(b"x")
    (outside / "big.md").write_bytes(b"x" * 64)

    with pytest.raises(error, match=match):
        await document.import_path(path.format(tmp=outside))
    assert await document_names() == [], f"nothing stored for a rejected import: {name}"


@pytest.mark.anyio
async def test_import_path_copies_the_file_and_records_its_size() -> None:
    """An import is streamed to its place with `shutil.copyfile`, so the size on the row is the
    size the file was stated at, and the file outside the home is left where it is."""
    source = home.HOME / "incoming" / "outside.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(MD)

    doc = await document.import_path(str(source))

    assert (doc.name, doc.size, doc.status) == ("outside.md", len(MD), "queued")
    assert doc.original.read_text() == MD, "copied, byte for byte"
    assert source.read_text() == MD, "a copy, not a move"
    assert await document_names() == ["outside.md"]


@pytest.mark.anyio
async def test_a_failed_import_leaves_neither_a_row_nor_a_name_it_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row goes in before the file, so a file that cannot be placed has to take the row with
    it: otherwise the name would be held by a document that does not exist."""
    import shutil

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copyfile", refuse)

    with pytest.raises(OSError, match="disk full"):
        await import_row("g.md")

    assert await document_names() == [], "the name is free again"
    assert not document.root("g.md").exists()


# --- documents: rows, listing and preview -------------------------------------------


@pytest.mark.anyio
async def test_get_reads_the_row_and_refuses_a_name_with_none() -> None:
    await import_row("a.md")

    assert (await document.named("a.md")).size == len(MD)
    with pytest.raises(NotFound, match="document not found"):
        await document.named("ghost.md")


@pytest.mark.anyio
async def test_source_path_of_a_document_whose_file_is_gone() -> None:
    doc = await import_row("a.md")
    doc.original.unlink()
    with pytest.raises(NotFound, match="document file missing"):
        doc.source_path()


@pytest.mark.anyio
async def test_set_status_bumps_updated_at_and_the_preview_does_not() -> None:
    """`updated_at` is what the "recently touched" sort reads, so only a change to the document
    may move it: a lifecycle step does, filling in the preview does not."""
    first = await import_row("a.md")
    assert first.created_at > 0, "stamped on import"
    assert first.updated_at == first.created_at, "an import is the document's first change"

    await document.set_status(await id_of("a.md"), DocumentStatus.IMPORTED)
    done = await document.named("a.md")
    assert done.updated_at > first.updated_at, "a lifecycle step is a change"
    assert (done.status, done.error) == ("imported", None)
    assert done.created_at == first.created_at, "the import moment never moves"

    await document.set_status(await id_of("a.md"), DocumentStatus.ERROR, "boom")
    failed = await document.named("a.md")
    assert (failed.status, failed.error) == ("error", "boom"), "the reason is stored with it"

    await document.ensure_preview(await document.named("a.md"))
    previewed = await document.named("a.md")
    assert previewed.preview is not None, "the preview was built"
    assert previewed.updated_at == failed.updated_at, "filling in the preview is not a change"


@pytest.mark.anyio
async def test_describe_replaces_the_description_and_reads_in_batches() -> None:
    a = await import_row("a.md")
    b = await import_row("b.md")

    described = await document.describe(a.id, "the alpha guide")

    assert described.description == "the alpha guide"
    assert await document.descriptions_of({a.id, b.id}) == {a.id: "the alpha guide"}, (
        "a document without one is absent"
    )
    assert await document.descriptions_of(set()) == {}
    assert (await document.describe(a.id, "")).description == "", "empty clears it"
    with pytest.raises(NotFound, match="document not found"):
        await document.describe("0" * 32, "x")


@pytest.mark.anyio
async def test_document_page_sorts_filters_and_resumes_by_keyset() -> None:
    sizes = {"a.md": 30, "b.md": 10, "c.md": 20}
    for name, size in sizes.items():
        await import_row(name, "x" * size)
    await document.set_status(await id_of("a.md"), DocumentStatus.IMPORTED)
    await document.set_status(await id_of("b.md"), DocumentStatus.ERROR, "boom")

    by_name = await document.page(PageRequest(page_size=2))
    assert [d.name for d in by_name.items] == ["a.md", "b.md"]
    assert (by_name.total, by_name.next_cursor is None) == (3, False)
    resumed = await document.page(PageRequest(cursor=by_name.next_cursor, page_size=2))
    assert [d.name for d in resumed.items] == ["c.md"], "the keyset resumes past the last row"
    assert resumed.next_cursor is None, "the last page says so"

    by_size = await document.page(PageRequest(sort="size", order=Order.DESC))
    assert [d.name for d in by_size.items] == ["a.md", "c.md", "b.md"]

    newest_first = await document.page(PageRequest(sort="created_at", order=Order.DESC))
    assert [d.name for d in newest_first.items] == ["c.md", "b.md", "a.md"]

    filtered = await document.page(PageRequest(), status=DocumentStatus.IMPORTED)
    assert [d.name for d in filtered.items] == ["a.md"]
    assert filtered.total == 1, "total counts the filtered rows, not every document"

    with pytest.raises(InvalidInput, match="unknown sort"):
        await document.page(PageRequest(sort="colour"))


@pytest.mark.anyio
async def test_ensure_preview_builds_once_and_then_reads_the_stored_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = await import_row("g.md")
    builds: list[str] = []
    real = convert.build_preview

    def counted(*args, **kwargs):
        builds.append(args[0].name)
        return real(*args, **kwargs)

    monkeypatch.setattr(convert, "build_preview", counted)

    _, first = await document.ensure_preview(doc)
    _, second = await document.ensure_preview(doc)

    assert builds == ["original.md"], "the second call reads the row instead of converting again"
    assert first == second and first.kind == "text"
    assert (doc.preview_dir / "preview.md").exists(), "both panes sit in the document's folder"


@pytest.mark.anyio
async def test_ensure_preview_builds_once_when_two_readers_arrive_together() -> None:
    """B8: the second reader waits on the per-document stripe lock and then finds the stored row."""
    doc = await import_row("g.md")
    real = convert.build_preview
    # threading events, because the gate below is held in the worker thread `cpu.on_cpu` runs the
    # build in; `reader_ready` is awaited on the loop instead, so it is an anyio one
    building, release = threading.Event(), threading.Event()
    reader_ready = anyio.Event()
    builds: list[str] = []
    results: list[tuple[object, str]] = []

    def gated(*args, **kwargs):
        builds.append(args[0].name)
        building.set()
        assert release.wait(timeout=30)
        return real(*args, **kwargs)

    async def first_reader() -> None:
        row, preview = await document.ensure_preview(doc)
        results.append((row, preview.kind))

    async def second_reader() -> None:
        await document.named(doc.name)  # the row is readable while the first reader holds the lock
        reader_ready.set()
        row, preview = await document.ensure_preview(doc)
        results.append((row, preview.kind))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(convert, "build_preview", gated)
        async with anyio.create_task_group() as readers:
            readers.start_soon(first_reader)
            await anyio.to_thread.run_sync(building.wait)  # the first build is under way
            readers.start_soon(second_reader)
            await reader_ready.wait()
            lock = document._preview_locks[doc.id]
            while lock.statistics().tasks_waiting == 0:  # the second reader is on the lock
                await anyio.sleep(0.01)
            release.set()

    assert len(builds) == 1, "one build, whichever reader got there first"
    assert len(results) == 2 and {kind for _, kind in results} == {"text"}


@pytest.mark.anyio
async def test_a_preview_lock_is_dropped_whichever_way_the_build_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One lock per document being built, so the dict is bounded by the builds in flight rather
    than by the documents this process has ever opened."""
    doc = await import_row("g.md")
    assert document._preview_locks == {}, "nothing is allocated before a build"

    await document.ensure_preview(doc)
    assert document._preview_locks == {}, "the build is committed; the next reader needs no lock"

    await document.ensure_preview(doc)
    assert document._preview_locks == {}, "and the early return takes none at all"

    other = await import_row("h.md")
    monkeypatch.setattr(convert, "build_preview", _refuse_to_build)
    with pytest.raises(PermanentError, match="no preview today"):
        await document.ensure_preview(other)

    assert document._preview_locks == {}, "a failed build leaves no lock behind either"


def _refuse_to_build(*args, **kwargs):
    raise PermanentError("no preview today")


@pytest.mark.anyio
async def test_a_failed_preview_build_is_never_retried_side_by_side() -> None:
    """A build that raises stores nothing, so every reader behind it tries again. The lock has to
    outlive the failure while one is queued: dropped there, that reader would hold a lock nobody
    else can find and the next arrival would build the same document at the same time."""
    doc = await import_row("g.md")
    entered = threading.Semaphore(0)  # one release per build entered
    let_first_fail, let_rest_fail = threading.Event(), threading.Event()
    counted = threading.Lock()
    live, peak, builds, failures = 0, 0, 0, 0

    def failing(*args, **kwargs):
        nonlocal live, peak, builds
        with counted:
            live += 1
            peak = max(peak, live)
            builds += 1
            gate = let_first_fail if builds == 1 else let_rest_fail
        entered.release()
        try:
            assert gate.wait(timeout=30)
            raise PermanentError("no preview today")
        finally:
            with counted:
                live -= 1

    async def reader() -> None:
        nonlocal failures
        with pytest.raises(PermanentError, match="no preview today"):
            await document.ensure_preview(doc)
        failures += 1

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(convert, "build_preview", failing)
        async with anyio.create_task_group() as readers:
            readers.start_soon(reader)  # A: the build that will fail
            await anyio.to_thread.run_sync(entered.acquire)
            readers.start_soon(reader)  # B: queued on A's lock
            lock = document._preview_locks[doc.id]
            while lock.statistics().tasks_waiting == 0:
                await anyio.sleep(0.01)
            let_first_fail.set()  # A raises; B rebuilds under the same lock
            await anyio.to_thread.run_sync(entered.acquire)
            readers.start_soon(reader)  # C: arrives after the failure
            while lock.statistics().tasks_waiting == 0 and builds < 3:
                await anyio.sleep(0.01)  # C is queued behind B, or building beside it
            let_rest_fail.set()

    assert peak == 1, "the queued reader and the newcomer never build side by side"
    assert (builds, failures) == (3, 3), "each reader tried once and was told why it failed"
    assert document._preview_locks == {}, "the last one out drops the lock"


class PreviewBuilds:
    """`convert.build_preview`, counted and held: each build signals `entered` as it starts and
    waits for `release`, so a test knows how many parses run at once."""

    def __init__(self) -> None:
        self.real = convert.build_preview
        self.entered = threading.Semaphore(0)  # one release per build entered
        self.release = threading.Event()
        self.counted = threading.Lock()
        self.live = 0
        self.peak = 0
        self.built: list[str] = []

    def __call__(self, *args, **kwargs):
        with self.counted:
            self.live += 1
            self.peak = max(self.peak, self.live)
            self.built.append(args[0])
        self.entered.release()
        assert self.release.wait(timeout=30)
        try:
            return self.real(*args, **kwargs)
        finally:
            with self.counted:
                self.live -= 1

    async def wait_entered(self, builds: int) -> None:
        for _ in range(builds):
            await anyio.to_thread.run_sync(self.entered.acquire)


@pytest.fixture
def preview_builds(monkeypatch: pytest.MonkeyPatch) -> Iterator[PreviewBuilds]:
    """Held preview builds, with the preview pool as the only ceiling on them. The pool size a
    test sets is put back afterwards."""
    from haskie import cpu

    # a build holds a CPU slot too, so the budget must not be the ceiling under test here
    monkeypatch.setattr(cpu, "_cpu_slots", cpu.SlotBudget(4))
    builds = PreviewBuilds()
    monkeypatch.setattr(convert, "build_preview", builds)
    try:
        yield builds
    finally:
        document.configure_preview_slots(PipelineSettings().preview_workers)


@pytest.mark.parametrize(("name", "workers"), [("one at a time", 1), ("two at a time", 2)])
@pytest.mark.anyio
async def test_ensure_preview_bounds_concurrent_builds(
    preview_builds: PreviewBuilds, name: str, workers: int
) -> None:
    """Four readers open four different documents at once; only `preview_workers` parses run.

    The stripe lock is per document, so nothing but the semaphore holds these four apart.
    """
    rows = [await import_row(f"doc-{i}.md") for i in range(4)]
    document.configure_preview_slots(workers)
    async with anyio.create_task_group() as readers:
        for row in rows:
            readers.start_soon(document.ensure_preview, row)
        await preview_builds.wait_entered(workers)  # every slot of the pool is now inside a build
        assert document._preview_slots.available_tokens == 0, f"{name}: no slot left"
        preview_builds.release.set()

    assert len(preview_builds.built) == 4, f"{name}: every document was built, once"
    assert preview_builds.peak == workers, f"{name}: never more parses at once than the pool admits"


@pytest.mark.anyio
async def test_resizing_the_preview_pool_counts_the_builds_already_running(
    preview_builds: PreviewBuilds,
) -> None:
    """Two builds run and two wait; raising the pool from 2 to 3 admits one more, not three."""
    rows = [await import_row(f"doc-{i}.md") for i in range(4)]
    document.configure_preview_slots(2)
    async with anyio.create_task_group() as readers:
        for row in rows:
            readers.start_soon(document.ensure_preview, row)
        await preview_builds.wait_entered(2)

        document.configure_preview_slots(3)

        await preview_builds.wait_entered(1)  # the one it admits
        await anyio.sleep(0.1)  # time enough for any it wrongly admits to enter too
        assert preview_builds.live == 3, "one more build, beside the two already running"
        preview_builds.release.set()

    assert preview_builds.peak == 3, "never more builds at once than the resized pool admits"


@pytest.mark.anyio
async def test_ensure_preview_returns_not_ready_when_the_queue_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader that waited out `PREVIEW_WAIT_SECONDS` is told to retry (503) rather than holding
    its request open until the burst clears."""
    doc = await import_row("g.md")
    monkeypatch.setattr(document, "PREVIEW_WAIT_SECONDS", 0.05)
    slots = anyio.CapacityLimiter(1)
    monkeypatch.setattr(document, "_preview_slots", slots)
    holder = object()  # another build: a limiter tells its borrowers apart
    await slots.acquire_on_behalf_of(holder)  # it holds the only slot, so every reader waits
    try:
        with pytest.raises(NotReady, match="preview queue is full"):
            await document.ensure_preview(doc)
    finally:
        slots.release_on_behalf_of(holder)

    assert (await document.named(doc.name)).preview is None, "nothing built, nothing stored"


# --- collections -------------------------------------------------------------------


@pytest.mark.anyio
async def test_collection_create_get_and_delete() -> None:
    made = await Collection.create("misc", description="odds and ends")

    assert made.name == "misc"
    assert made.root.is_dir(), "the folder is made with the row"
    assert (await made.info()).description == "odds and ends"
    assert await Collection.names() == ["misc"]
    with pytest.raises(Conflict, match="already exists"):
        await Collection.create("misc")
    with pytest.raises(NotFound, match="collection not found"):
        await Collection.get("nope")
    with pytest.raises(InvalidInput, match="invalid name"):
        await Collection.create("***")

    assert (await Collection.get("misc")).root == made.root
    await remove_collection("misc")

    assert await Collection.names() == [] and not made.root.exists()


@pytest.mark.anyio
async def test_collection_describe_replaces_the_description() -> None:
    collection = await Collection.create("notes")
    await collection.describe("everything I read")
    assert (await collection.info()).description == "everything I read"
    await collection.describe("")
    assert (await collection.info()).description == "", "empty clears it"


@pytest.mark.anyio
async def test_collection_settings_are_read_from_the_row_every_time() -> None:
    collection = await Collection.create("stored")
    assert await collection.overrides() == CollectionOverrides(), "no overrides yet"

    await collection.set_overrides(CollectionOverrides(chunker=Chunker.TEXT))
    assert (await (await Collection.get("stored")).overrides()).chunker == "text"

    async with db.connect() as conn:  # a write the setter never saw
        await conn.execute(
            update(tables.collections)
            .where(tables.collections.c.name == "stored")
            .values(overrides='{"chunk_size": 42}')
        )
    assert (await collection.overrides()).chunk_size == 42, "the row owns the value"

    await remove_collection("stored")
    await Collection.create("stored")
    assert await (await Collection.get("stored")).overrides() == CollectionOverrides(), "clean"
    assert await Collection("ghost").overrides() == CollectionOverrides(), "and one with no row"


@pytest.mark.anyio
async def test_chunk_settings_of_a_collection_resolve_against_the_user_settings() -> None:
    await save_user_settings(UserSettings(conversion=ConversionSettings(chunk_size=800)))
    collection = await Collection.create("chunky")
    await collection.set_overrides(CollectionOverrides(chunker=Chunker.TEXT))

    assert await collection.chunk_settings() == ChunkSettings(Chunker.TEXT, 800, 66)
    assert (await collection.info()).search.limit == SearchSettings().limit


@pytest.mark.anyio
async def test_load_settings_reads_every_collection_it_was_asked_for_in_one_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cross-collection search resolves the settings of its whole selection at once, so the cost
    is one SELECT rather than one per collection. A name without a row stays out of the answer."""
    for name in ("alpha", "beta"):
        await Collection.create(name)
    await Collection("alpha").set_overrides(CollectionOverrides(chunker=Chunker.TEXT))
    assert await Collection.load_overrides([]) == {}, "nothing asked for, nothing read"
    # a collection change refreshes the Claude Code installs in the background: counted, its
    # query would land inside the window below
    await until(refresh_settled, "the creates' own refresh ended")
    statements: list[str] = []

    def counted(_conn, _cursor, statement: str, *_args) -> None:
        statements.append(statement)

    event.listen(db.engine().sync_engine, "before_cursor_execute", counted)
    try:
        found = await Collection.load_overrides(["alpha", "beta", "ghost", "alpha"])
    finally:
        event.remove(db.engine().sync_engine, "before_cursor_execute", counted)

    assert set(found) == {"alpha", "beta"}, "a name with no row is absent from the result"
    assert (found["alpha"].chunker, found["beta"]) == ("text", CollectionOverrides())
    selects = [sql for sql in statements if sql.lstrip().lower().startswith("select")]
    assert len(selects) == 1, "one query for every name, duplicates included"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("way_out", "leaving", "held"),
    [
        (
            "nothing",
            {"notes": frozenset(), "other": frozenset()},
            {"guide.md": ["notes", "other"], "keep.md": ["notes"]},
        ),
        (
            "guide.md removing from notes",
            {"notes": frozenset({"guide.md"}), "other": frozenset()},
            {"guide.md": ["other"], "keep.md": ["notes"]},
        ),
        (
            "guide.md deleting",
            {"notes": frozenset({"guide.md"}), "other": frozenset({"guide.md"})},
            {"keep.md": ["notes"]},
        ),
    ],
)
async def test_a_search_reads_one_rule_for_a_document_on_its_way_out(
    way_out: str, leaving: dict[str, frozenset[str]], held: dict[str, list[str]]
) -> None:
    """A membership `removing` or a document `deleting` (`LEAVING`) is what each search index
    leaves out (`for_search`) and what `holding` no longer counts. A collection the search does
    not cover holds nothing, and a document none of them holds is absent."""
    for name in ("notes", "other", "unsearched"):
        await Collection.create(name)
    ids: dict[str, str] = {}
    for name in ("guide.md", "keep.md", "lonely.md"):
        ids[name] = (await import_row(name)).id
        await document.set_status(ids[name], DocumentStatus.IMPORTED)
    for collection, doc in (
        ("notes", "guide.md"),
        ("other", "guide.md"),
        ("notes", "keep.md"),
        ("unsearched", "keep.md"),
    ):
        await Collection(collection).add(ids[doc])
    if way_out == "guide.md removing from notes":
        await Collection("notes").start_removal(ids["guide.md"])
    elif way_out == "guide.md deleting":
        await document.set_status(ids["guide.md"], DocumentStatus.DELETING)
    names = ["notes", "other", "ghost"]
    docs = set(ids.values())
    named = {id: name for name, id in ids.items()}

    found = await Collection.for_search(names, None)

    assert list(found) == ["notes", "other"], f"{way_out}: in the order given, a ghost absent"
    assert {
        name: frozenset(named[doc] for doc in index.leaving) for name, (index, _) in found.items()
    } == leaving, way_out
    holding = await Collection.holding(docs, names)
    assert {named[doc]: held_in for doc, held_in in holding.items()} == held, way_out
    assert await Collection.holding(set(), names) == {}, "no document asked for"
    assert await Collection.holding(docs, []) == {}, "no collection searched"


def test_the_leaving_query_reads_by_index_not_every_membership() -> None:
    """Almost always nothing is leaving, so finding that out must not read every membership of
    the searched collections: each branch of `LEAVING` is bound by its status index."""
    from sqlalchemy.dialects import sqlite as sqlite_dialect

    from haskie.collection.collection import _leaving_query

    conn = sqlite3.connect(":memory:")
    conn.executescript(db.schema_ddl())
    sql = str(
        _leaving_query(["notes", "other"]).compile(
            dialect=sqlite_dialect.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
    plan = [row[3] for row in conn.execute(f"explain query plan {sql}")]

    assert any("idx_collection_documents_status (collection=? AND status=?)" in s for s in plan)
    assert any("idx_documents_status (status=?)" in s for s in plan), plan
    assert not any(s.startswith("SCAN") for s in plan), plan


@pytest.mark.anyio
async def test_reranker_overrides_lists_every_model_a_collection_chose() -> None:
    """The model downloads have to cover the overrides too, so they are read in one query."""
    chosen = "cross-encoder/ettin-reranker-17m-v1"
    for name in ("a", "b", "c"):
        await Collection.create(name)
    await Collection("b").set_overrides(
        CollectionOverrides(search=SearchOverrides(reranker_model=chosen))
    )
    await Collection("c").set_overrides(
        CollectionOverrides(search=SearchOverrides(reranker_model=chosen))
    )

    assert await Collection.reranker_overrides(SearchSettings()) == [chosen], "no duplicates"


@pytest.mark.anyio
async def test_a_collection_that_turns_the_reranker_on_loads_the_users_model() -> None:
    """With the user's reranker off, nothing else downloads the model such a collection's
    search resolves to."""
    await Collection.create("reranked")
    await Collection("reranked").set_overrides(
        CollectionOverrides(search=SearchOverrides(reranker=Reranker.CROSS_ENCODER))
    )

    assert await Collection.reranker_overrides(SearchSettings()) == [DEFAULT_RERANKER]


@pytest.mark.anyio
async def test_collection_page_lists_summaries_with_their_counts() -> None:
    for name in ("alpha", "beta", "gamma"):
        await Collection.create(name, description=f"{name} notes")
    await attachable("a.md")
    await Collection("beta").add(await id_of("a.md"))
    await Collection("beta").set_member_status(await id_of("a.md"), MemberStatus.INDEXED)

    first = await Collection.page(PageRequest(page_size=2))

    assert [c.name for c in first.items] == ["alpha", "beta"]
    assert first.total == 3 and first.next_cursor is not None
    assert first.items[0].counts == DocumentCounts(), "a collection with no members"
    assert (first.items[1].counts.total, first.items[1].counts.indexed) == (1, 1)
    assert first.items[1].description == "beta notes"
    rest = await Collection.page(PageRequest(cursor=first.next_cursor, page_size=2))
    assert [c.name for c in rest.items] == ["gamma"] and rest.next_cursor is None


# --- collection membership ----------------------------------------------------------


@pytest.mark.anyio
async def test_add_is_idempotent_and_member_reads_the_document_with_it() -> None:
    collection = await Collection.create("notes")
    doc = await attachable("a.md")

    await collection.add(doc.id)
    first = await collection.member(doc.id)
    await collection.add(doc.id)
    again = await collection.member(doc.id)

    assert isinstance(first, Member) and first.status == "pending", "indexing moves it along"
    assert first.document.name == doc.name and first.document.size == doc.size
    assert (again.status, again.added_at) == (first.status, first.added_at), "a no-op re-attach"
    assert await collection.member_ids() == [doc.id]

    await collection.set_member_status(doc.id, MemberStatus.ERROR, "boom")
    failed = await collection.member(doc.id)
    assert (failed.status, failed.error) == ("error", "boom")
    assert failed.updated_at >= failed.added_at

    with pytest.raises(NotFound, match="document not found"):
        await collection.add("0" * 32)
    with pytest.raises(NotFound, match="document not in collection notes"):
        await collection.member("0" * 32)


@pytest.mark.parametrize(
    ("name", "status", "to", "refused"),
    [
        ("a member imported: the name moves", "imported", "HEALTH", None),
        (
            "a member being deleted: its removal is queued under the old name",
            "deleting",
            "new",
            "document a.md is being deleted",
        ),
    ],
)
@pytest.mark.anyio
async def test_rename_moves_the_folder_unless_a_member_is_being_deleted(
    name: str, status: str, to: str, refused: str | None
) -> None:
    """`health` and `HEALTH` share a shard, so on a case-insensitive disk the target folder is
    the source itself: it is moved, never swept away as a leftover."""
    collection = await Collection.create("health")
    doc = await attachable("a.md")
    await collection.add(doc.id)
    await document.set_status(doc.id, status)  # ty: ignore

    if refused is not None:
        with pytest.raises(Conflict, match=refused):
            await collection.rename(to)
        assert await Collection.names() == ["health"] and collection.root.is_dir(), name
        return
    renamed = await collection.rename(to)

    assert await Collection.names() == [to], name
    assert renamed.root.is_dir(), f"{name}: the folder moved, not removed"
    assert await renamed.member_ids() == [doc.id], name


@pytest.mark.parametrize(
    "status", ["queued", "converting", "embedding", "describing", "error", "deleting"]
)
@pytest.mark.anyio
async def test_add_refuses_a_document_that_is_not_imported(status: str) -> None:
    """The invariant lives in `add`: one still importing has no markdown to chunk yet, and one
    being deleted must not gain a membership the delete's snapshot missed."""
    collection = await Collection.create("notes")
    doc = await import_row("a.md")
    await document.set_status(doc.id, status)  # ty: ignore

    with pytest.raises(Conflict, match=f"document is {status}; only an imported document"):
        await collection.add(doc.id)

    assert await collection.member_ids() == [], "nothing attached"


@pytest.mark.anyio
async def test_member_ids_walk_one_page_at_a_time() -> None:
    collection = await Collection.create("notes")
    names = ("a.md", "b.md", "c.md")
    first, second, third = sorted([(await attachable(name)).id for name in names])
    for doc in (first, second, third):
        await collection.add(doc)

    assert await collection.member_ids() == [first, second, third]
    assert await collection.member_ids(limit=2) == [first, second]
    assert await collection.member_ids(after=second) == [third]


async def _cache_entry(doc: Document) -> str:
    """A real cache entry of `doc`, one row of its first chunk, written as the embed stage writes
    one (`embed_cache.write`); returns its id."""
    params = embed_cache.params(doc, ChunkSettings(), None)
    rows = home.HOME / "rows" / "000000.rows.json"
    rows.parent.mkdir(parents=True, exist_ok=True)
    rows.write_bytes(msgspec.json.encode([Row(chunk=chunk.split(MD, SMALL)[0], seq=1)]))
    return await embed_cache.write(params, [rows], None)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "status", "entry", "leaving", "answers"),
    [
        ("an indexed member", MemberStatus.INDEXED, True, None, True),
        ("an indexed member, its cache entry forgotten", MemberStatus.INDEXED, False, None, True),
        ("one indexed again: its old rows stand", MemberStatus.INDEXING, True, None, True),
        ("one indexed for the first time: no rows yet", MemberStatus.INDEXING, False, None, False),
        ("a pending member", MemberStatus.PENDING, False, None, False),
        ("a write that failed after its rows were dropped", MemberStatus.ERROR, False, None, False),
        ("a write that failed with rows standing", MemberStatus.ERROR, True, None, True),
        ("a write cancelled, its rows dropped", MemberStatus.CANCELLED, False, None, False),
        ("an indexed member being detached", MemberStatus.INDEXED, True, "removing", False),
        ("an indexed member being deleted", MemberStatus.INDEXED, True, "deleting", False),
    ],
)
async def test_a_collection_answers_a_search_only_from_rows_it_still_has(
    name: str, status: MemberStatus, entry: bool, leaving: str | None, answers: bool
) -> None:
    notes = await Collection.create("notes")
    attached = await attachable("a.md")
    doc = attached.id
    await notes.add(doc)
    if entry:
        await notes.set_member_entry(doc, await _cache_entry(attached))
    await notes.set_member_status(doc, status, "boom" if status == MemberStatus.ERROR else None)
    if leaving == "removing":
        await notes.start_removal(doc)
    elif leaving == "deleting":
        await document.set_status(doc, DocumentStatus.DELETING)

    assert await Collection.searchable(["notes"]) == (["notes"] if answers else []), name


@pytest.mark.anyio
async def test_member_counts_group_by_status() -> None:
    collection = await Collection.create("counts")
    for name in ("a.md", "b.md", "c.md", "d.md", "e.md"):
        await collection.add((await attachable(name)).id)
    await collection.set_member_status(await id_of("a.md"), MemberStatus.INDEXED)
    await collection.set_member_status(await id_of("b.md"), MemberStatus.INDEXING)
    await collection.set_member_status(await id_of("c.md"), MemberStatus.ERROR, "boom")
    await collection.start_removal(await id_of("e.md"))

    counts = await collection.counts()

    assert counts.total == 5
    assert counts.indexed == 1
    assert counts.active == 3, "pending, indexing and removing are all in flight"
    assert counts.error == 1
    assert counts.by_status == {
        "indexed": 1,
        "indexing": 1,
        "error": 1,
        "pending": 1,
        "removing": 1,
    }
    empty = await Collection.create("empty")
    assert await empty.counts() == DocumentCounts(), "a collection without members"


def _mark(status: MemberStatus) -> Callable[[Collection, str], Awaitable[None]]:
    return lambda collection, doc: collection.set_member_status(doc, status)


def _fail_removal(collection: Collection, doc: str) -> Awaitable[None]:
    return collection.fail_removal(doc, "removal failed: boom")


@pytest.mark.parametrize(
    ("detached", "change", "outcome", "status", "error"),
    [
        # a cancelled index still finishing, and the cancel itself, write the status too late
        pytest.param(
            True, _mark(MemberStatus.INDEXED), nullcontext(), "removing", None, id="index"
        ),
        pytest.param(
            True, _mark(MemberStatus.CANCELLED), nullcontext(), "removing", None, id="cancel"
        ),
        # a failed removal is the one write that ends `removing`, short of the row going
        pytest.param(
            True,
            _fail_removal,
            nullcontext(),
            "error",
            "removal failed: boom",
            id="fail the removal",
        ),
        # a delete's removal never marked the member: its failure leaves the status alone
        pytest.param(
            False, _fail_removal, nullcontext(), "indexed", None, id="fail an unmarked removal"
        ),
        # attaching again is refused and leaves the membership as it was
        pytest.param(
            True,
            Collection.add,
            pytest.raises(Conflict, match="being removed from collection going"),
            "removing",
            None,
            id="attach while removing",
        ),
        # attaching a member again is a no-op
        pytest.param(False, Collection.add, nullcontext(), "indexed", None, id="attach again"),
        # detaching again marks it again; a stranger is not found and marks nothing
        pytest.param(
            True, Collection.start_removal, nullcontext(), "removing", None, id="detach again"
        ),
        pytest.param(
            True,
            lambda collection, doc: collection.start_removal("0" * 32),
            pytest.raises(NotFound, match="document not in collection going: 0{32}"),
            "removing",
            None,
            id="detach a stranger",
        ),
    ],
)
@pytest.mark.anyio
async def test_a_removing_membership_keeps_its_status_until_its_removal_ends(
    detached: bool,
    change: Callable[[Collection, str], Awaitable[None]],
    outcome: AbstractContextManager,
    status: str,
    error: str | None,
) -> None:
    """A detach answers once its removal is queued, and the listing shows `removing` until it ran:
    no index status may overwrite that, or a poll would stop following a member still going."""
    collection = await Collection.create("going")
    doc = await attachable("going.md")
    await collection.add(doc.id)
    await collection.set_member_status(doc.id, MemberStatus.INDEXED)
    if detached:
        await collection.start_removal(doc.id)

    with outcome:
        await change(collection, doc.id)

    member = await collection.member(doc.id)
    assert (member.status, member.error) == (status, error)
    assert (await collection.counts()).active == (status == "removing")


@pytest.mark.parametrize(
    ("name", "sort", "order", "expected"),
    [
        ("by name", "name", "asc", ["a.md", "b.md", "c.md"]),
        ("by size, largest first", "size", "desc", ["a.md", "c.md", "b.md"]),
        ("by membership status", "status", "asc", ["c.md", "a.md", "b.md"]),
        ("by when the membership last moved", "updated_at", "asc", ["a.md", "b.md", "c.md"]),
    ],
)
@pytest.mark.anyio
async def test_members_page_sorts_on_the_document_and_on_the_membership(
    name: str, sort: str, order: str, expected: list[str]
) -> None:
    """The listing joins both tables, so `name`/`size` come from the document and
    `status`/`updated_at` from the membership."""
    collection = await Collection.create("notes")
    for doc, size in (("a.md", 30), ("b.md", 10), ("c.md", 20)):
        await collection.add((await attachable(doc, "x" * size)).id)
    for doc, status in (
        ("a.md", MemberStatus.INDEXED),
        ("b.md", MemberStatus.PENDING),
        ("c.md", MemberStatus.ERROR),
    ):
        await collection.set_member_status(await id_of(doc), status)

    page = await collection.members_page(PageRequest(sort=sort, order=order))  # ty: ignore

    assert [member.document.name for member in page.items] == expected, name
    assert page.total == 3, name


@pytest.mark.anyio
async def test_members_page_filters_by_status_and_resumes_by_keyset() -> None:
    collection = await Collection.create("notes")
    for doc in ("a.md", "b.md", "c.md"):
        await collection.add((await attachable(doc)).id)
    await collection.set_member_status(await id_of("b.md"), MemberStatus.INDEXED)

    first = await collection.members_page(PageRequest(page_size=2))
    assert [m.document.name for m in first.items] == ["a.md", "b.md"]
    resumed = await collection.members_page(PageRequest(cursor=first.next_cursor, page_size=2))
    assert [m.document.name for m in resumed.items] == ["c.md"]

    indexed = await collection.members_page(PageRequest(), status=MemberStatus.INDEXED)
    assert [m.document.name for m in indexed.items] == ["b.md"]
    assert indexed.total == 1, "total counts the filtered rows"


@pytest.mark.anyio
async def test_one_document_sits_in_two_collections_and_a_detach_leaves_both_alone() -> None:
    """Membership is many-to-many, so detaching from one collection touches neither the document
    nor the other collection, and never the embedding cache."""
    doc = await attachable("shared.md")
    for name in ("alpha", "beta"):
        await (await Collection.create(name)).add(doc.id)
    cache_id = await _cache_entry(doc)

    assert await document.collections_of(doc.id) == ["alpha", "beta"]

    await Collection("alpha").remove_member(doc.id)

    assert await document.collections_of(doc.id) == ["beta"], "only that membership went"
    assert (await document.named(doc.name)).name == doc.name, "the document stays"
    params = embed_cache.params(doc, ChunkSettings(), None)
    assert await embed_cache.lookup(params) == cache_id, "and so does what it costs to compute"
    assert await Collection("beta").member(doc.id) is not None


@pytest.mark.anyio
async def test_deleting_a_collection_leaves_its_documents() -> None:
    doc = await attachable("kept.md")
    for name in ("alpha", "beta"):
        await (await Collection.create(name)).add(doc.id)

    await remove_collection("alpha")

    assert await Collection.names() == ["beta"]
    assert await document_names() == ["kept.md"], "the document belongs to no collection"
    assert doc.original.exists(), "and keeps its files"
    assert await document.collections_of(doc.id) == ["beta"]


@pytest.mark.anyio
async def test_deleting_a_document_takes_every_membership_and_cache_row_with_it() -> None:
    doc = await attachable("gone.md")
    for name in ("alpha", "beta"):
        await (await Collection.create(name)).add(doc.id)
    await _cache_entry(doc)

    await document.remove_files(doc.id)
    await document.remove_row(doc.id)

    assert await document_names() == []
    assert await document.collections_of(doc.id) == [], "both memberships cascaded"
    assert await embed_cache.entries(doc.id) == [], "and so did the cache rows"
    assert not doc.root.exists()
    assert await Collection("alpha").counts() == DocumentCounts(), "the collections stay, empty"


@pytest.mark.anyio
async def test_an_unattached_document_is_valid_and_listable() -> None:
    doc = await import_row("lonely.md")

    assert await document.collections_of(doc.id) == []
    page = await document.page(PageRequest())
    assert [d.name for d in page.items] == ["lonely.md"]


@pytest.mark.anyio
async def test_deleting_a_collection_drops_it_from_every_session() -> None:
    """One cascade from the `collections` row: the memberships and every session that chose it."""
    for name in ("keep", "drop"):
        await Collection.create(name)
    async with db.connect() as conn:
        await conn.execute(insert(tables.sessions).values(id="s1"))
        await conn.execute(
            insert(tables.session_collections),
            [
                {"session_id": "s1", "collection": "keep", "position": 0},
                {"session_id": "s1", "collection": "drop", "position": 1},
            ],
        )

    await remove_collection("drop")

    async with db.connect() as conn:
        rows = await conn.scalars(select(tables.session_collections.c.collection))
        assert list(rows) == ["keep"]
    assert await Collection.names() == ["keep"]


# --- index -------------------------------------------------------------------------

COMPACT = EmbeddingModel("ibm-granite/granite-embedding-97m-multilingual-r2", 384)
SAME_SIZE = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2", 384)
VECTORS_384 = PLAIN_SCHEMA.append(pa.field("vector", pa.list_(pa.float32(), 384)))


def _table_with(path: Path, schema: pa.Schema) -> None:
    """A table written straight to disk, bypassing `CollectionIndex`: the schema under test is one
    the current build would never write. The sync LanceDB API on purpose, so sync fixtures and
    sync tests can set the stage as well."""
    lancedb.connect(str(path)).create_table("chunks", schema=schema)


async def _aparts(parts: list[tuple[int, list[Row]]]) -> AsyncIterator[tuple[int, list[Row]]]:
    """`add_parts` takes an async iterator: the index stage reads one row group of the embedding
    cache at a time and awaits each read (see `embed_cache.read`)."""
    for part in parts:
        yield part


@pytest.mark.parametrize(
    ("name", "schema", "embedding", "expected"),
    [
        ("no table at all -> nothing to reject", None, COMPACT, True),
        ("plain table, no embedding wanted", PLAIN_SCHEMA, None, True),
        (
            "plain table, embedding wanted -> no vector column",
            PLAIN_SCHEMA,
            COMPACT,
            False,
        ),
        (
            "vectors of the current embedding",
            VECTORS_384.with_metadata({EMBEDDING_KEY: COMPACT.cache_name.encode()}),
            COMPACT,
            True,
        ),
        (
            "vectors of another model of the same size -> outdated",
            VECTORS_384.with_metadata({EMBEDDING_KEY: SAME_SIZE.cache_name.encode()}),
            COMPACT,
            False,
        ),
        (
            "vectors of an unrecorded embedding, from an older build -> outdated",
            VECTORS_384,
            COMPACT,
            False,
        ),
        (
            "vector of other dimensions -> outdated",
            PLAIN_SCHEMA.append(pa.field("vector", pa.list_(pa.float32(), 1024))),
            COMPACT,
            False,
        ),
        (
            "column missing from an older build",
            pa.schema([("doc", pa.string()), ("text", pa.string())]),
            None,
            False,
        ),
    ],
)
@pytest.mark.anyio
async def test_schema_current(
    tmp_path: Path, name: str, schema: pa.Schema | None, embedding, expected: bool
) -> None:
    path = tmp_path / "index"
    if schema is not None:
        _table_with(path, schema)
    index = CollectionIndex(path, "notes", tmp_path, embedding)
    assert await index.schema_current() is expected, name


@pytest.mark.anyio
async def test_schema_current_is_cached_until_a_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index stage is the only writer of a table and runs in this process, so the answer is
    cached per index directory; the write path forgets it again."""
    path = tmp_path / "index"
    _table_with(path, pa.schema([("doc", pa.string()), ("text", pa.string())]))  # older build
    inspected: list[Path] = []
    real = CollectionIndex._existing

    async def counted(self: CollectionIndex):
        inspected.append(self.path)
        return await real(self)

    monkeypatch.setattr(CollectionIndex, "_existing", counted)

    assert await CollectionIndex(path, "notes", tmp_path, None).schema_current() is False
    assert inspected == [path], "the first answer reads the table"
    inspected.clear()
    assert await CollectionIndex(path, "notes", tmp_path, None).schema_current() is False
    assert inspected == [], "a second index instance answers from the process cache"

    await CollectionIndex(path, "notes", tmp_path, None).reset_for_write()  # drops it

    assert await CollectionIndex(path, "notes", tmp_path, None).schema_current() is True, "re-read"


@pytest.mark.anyio
async def test_a_schema_read_that_spans_a_reset_caches_nothing(tmp_path: Path) -> None:
    """A search reads the schema of an outdated table while a document's index stage drops it and
    creates the new one. The search's answer is about the dropped table: cached, it would make the
    next document's write drop the new table with its rows."""
    path = tmp_path / "index"
    _table_with(path, pa.schema([("doc", pa.string()), ("text", pa.string())]))  # older build
    reader = CollectionIndex(path, "notes", tmp_path, None)
    await reader.open()
    old = reader._cached
    assert old is not None

    class ResetMidRead:
        """The reader's table handle, with the index stage's reset landing mid-read."""

        async def schema(self) -> pa.Schema:
            schema = await old.schema()
            await CollectionIndex(path, "notes", tmp_path, None).reset_for_write()
            return schema

    reader._cached = ResetMidRead()  # ty: ignore[invalid-assignment]

    assert await reader.schema_current() is False, "the reader's own table is outdated"
    assert await CollectionIndex(path, "notes", tmp_path, None).schema_current() is True


@pytest.mark.anyio
async def test_a_missing_table_is_never_cached(tmp_path: Path) -> None:
    """Caching "nothing to reject" would hide the table the index stage creates a moment later."""
    path = tmp_path / "index"
    assert await CollectionIndex(path, "notes", tmp_path, COMPACT).schema_current() is True

    _table_with(path, PLAIN_SCHEMA)  # no vector column, so it cannot hold COMPACT rows

    assert await CollectionIndex(path, "notes", tmp_path, COMPACT).schema_current() is False


@pytest.mark.anyio
async def test_deleting_a_collection_forgets_its_cached_schema() -> None:
    collection = await Collection.create("recycled")
    _table_with(collection.index_dir, PLAIN_SCHEMA)
    assert await (await collection.index()).schema_current() is True

    await remove_collection("recycled")
    await Collection.create("recycled")  # same name, same index directory
    _table_with(collection.index_dir, pa.schema([("doc", pa.string())]))  # an older build

    assert await (await (await Collection.get("recycled")).index()).schema_current() is False


@pytest.mark.anyio
async def test_existing_never_creates_the_index_directory(tmp_path: Path) -> None:
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    assert await index._existing() is None
    assert not (tmp_path / "index").exists(), "a read must not create a LanceDB directory"


@pytest.mark.anyio
async def test_existing_is_none_for_a_directory_without_the_table(tmp_path: Path) -> None:
    path = tmp_path / "index"
    lancedb.connect(str(path))  # creates the directory, no table
    assert await CollectionIndex(path, "notes", tmp_path, None)._existing() is None


@pytest.mark.anyio
async def test_search_of_a_never_indexed_collection_is_empty(tmp_path: Path) -> None:
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    assert await index_hits(index, "anything", SearchSettings()) == []


@pytest.mark.anyio
async def test_delete_on_an_outdated_table_removes_nothing_instead_of_dropping_it(
    tmp_path: Path,
) -> None:
    """Deleting one document must never wipe a collection built by an older embedding."""
    path = tmp_path / "index"
    old = legacy_index(path, "a.md", "hello")
    index = CollectionIndex(path, "notes", tmp_path, COMPACT)

    await index.delete_document("a.md")
    await index.delete_parts("a.md", 0, 1)

    assert old.count_rows() == 1, "rows of an unreadable schema are left alone"
    assert "chunks" in lancedb.connect(str(path)).list_tables().tables


@pytest.mark.anyio
async def test_add_parts_without_a_vector_is_rejected(tmp_path: Path) -> None:
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, COMPACT)
    (row,) = [Row(chunk=c, seq=1) for c in chunk.split(MD, SMALL)[:1]]
    with pytest.raises(ValueError, match="carries no vector"):
        await index.add_parts("g.md", "s", "m", _aparts([(0, [row])]))


@pytest.mark.anyio
async def test_add_parts_with_nothing_to_write_creates_no_table(tmp_path: Path) -> None:
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    await index.add_parts("g.md", "s", "m", _aparts([(0, [])]))
    assert not (tmp_path / "index").exists()


@pytest.mark.anyio
async def test_finish_on_an_empty_index_is_a_no_op(tmp_path: Path) -> None:
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    await index.finish()  # no table yet
    _table_with(tmp_path / "index", PLAIN_SCHEMA)
    await CollectionIndex(tmp_path / "index", "notes", tmp_path, None).finish()  # zero rows


# --- index maintenance -------------------------------------------------------------

TINY = EmbeddingModel("test/tiny", 32)  # 32 / 16 = 2 PQ sub-vectors, enough rows per codebook


def _row(text: str, vector: list[float] | None = None, seq: int = 1, char_start: int = 0) -> Row:
    return Row(
        chunk=Chunk(
            headings=["H"],
            frame=["H"],
            pieces=[Piece(PieceType.TEXT, text)],
            line_start=1,
            line_end=1,
            char_start=char_start,
            char_end=char_start + len(text),
            byte_start=char_start,
            byte_end=char_start + len(text.encode()),
        ),
        vector=vector,
        seq=seq,
    )


def _vector(seed: int, dims: int = 32) -> list[float]:
    """A deterministic unit-ish vector: `seed` decides the quadrant, so nearest neighbours of a
    query built the same way are the rows with the same seed."""
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dims)]


async def _fill(
    index: CollectionIndex, doc: str, part: int, count: int, vectors: bool = False
) -> None:
    rows = [
        _row(
            f"{doc} part{part} row{i} lancedb",
            _vector(part * 1000 + i) if vectors else None,
            seq=part * count + i + 1,
        )
        for i in range(count)
    ]
    await index.add_parts(doc, f"documents/{doc}", f"documents/{doc}.md", _aparts([(part, rows)]))


@pytest.mark.anyio
async def test_finish_builds_fts_once_and_later_rows_are_still_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S1: a full-text index costs O(rows) to build, so a collection of n documents must not build
    one per document. Rows added after the build are covered by a scan until maintenance folds
    them in, so nothing is lost by building it once."""
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    await _fill(index, "a.md", 0, 3)
    table = await index._existing()
    assert table is not None
    builds: list[str] = []
    real = type(table).create_index  # the async API builds the FTS index through `create_index`

    async def counted(self, column: str, **kw):
        builds.append(column)
        return await real(self, column, **kw)

    monkeypatch.setattr(type(table), "create_index", counted)

    await index.finish()
    await _fill(index, "b.md", 0, 2)
    await index.finish()

    assert builds == [FTS_COLUMN], "one build across two documents"
    assert await index.has_index(FTS_COLUMN) is True
    assert [list(i.columns) for i in await table.list_indices()] == [[FTS_COLUMN]], "and one index"
    found = {hit.document for hit in await index_hits(index, "lancedb", SearchSettings(limit=10))}
    assert found == {"a.md", "b.md"}, "the rows added after the build are still found"


@pytest.mark.anyio
async def test_stats_reports_rows_fragments_and_indices(tmp_path: Path) -> None:
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    assert await index.stats() is None, "no table, nothing to report"

    await _fill(index, "a.md", 0, 3)
    await _fill(index, "a.md", 1, 2)
    fresh = await index.stats()
    assert fresh is not None
    assert (fresh.num_rows, fresh.num_fragments) == (5, 2), "one fragment per commit"
    assert (fresh.has_fts_index, fresh.has_vector_index) == (False, False)
    assert (fresh.unindexed_rows, fresh.vector_index_rows) == (5, 0), "no index covers any row"

    await index.finish()
    await _fill(index, "b.md", 0, 1)
    grown = await index.stats()
    assert grown is not None
    assert (grown.num_rows, grown.has_fts_index) == (6, True)
    assert grown.unindexed_rows == 1, "the row written after the build is scanned, not indexed"

    await index.optimize(timedelta(minutes=10))
    compacted = await index.stats()
    assert compacted is not None
    assert compacted.num_rows == 6
    assert compacted.num_fragments < grown.num_fragments, "fragments merged"
    assert compacted.unindexed_rows == 0, "and the new row folded into the index"


@pytest.mark.parametrize(
    ("name", "num_rows", "expected"),
    [
        ("empty table", 0, 16),
        ("tiny collection clamps to the floor", 100, 16),
        ("50k rows -> sqrt rounded to a power of two", 50_000, 256),
        ("500k rows", 500_000, 512),
        ("huge collection clamps to the ceiling", 10**12, 4096),
    ],
)
def test_partitions_rule(name: str, num_rows: int, expected: int) -> None:
    assert _partitions(num_rows) == expected, name


@pytest.mark.parametrize(
    ("name", "num_rows", "has_index", "trained_rows", "min_rows", "expected"),
    [
        ("too small to be worth an index", 100, False, 0, 50_000, False),
        ("big enough, none yet", 50_000, False, 0, 50_000, True),
        ("at the threshold, already trained on it", 50_000, True, 50_000, 50_000, False),
        ("grown, but not doubled", 90_000, True, 50_000, 50_000, False),
        ("doubled -> retrain", 100_000, True, 50_000, 50_000, True),
        ("index of unknown age retrains once", 60_000, True, 0, 50_000, True),
        ("a setting under what PQ trains on waits for its rows", 255, False, 0, 10, False),
        ("and trains once they are there", 256, False, 0, 10, True),
    ],
)
def test_ann_due(
    name: str, num_rows: int, has_index: bool, trained_rows: int, min_rows: int, expected: bool
) -> None:
    stats = IndexStats(
        num_rows=num_rows,
        num_fragments=1,
        num_small_fragments=0,
        has_fts_index=True,
        has_vector_index=has_index,
        unindexed_rows=0,
        vector_index_rows=num_rows if has_index else 0,
    )
    settings = PipelineSettings(ann_min_rows=min_rows)
    assert maintenance.ann_due(stats, settings, trained_rows) is expected, name


@pytest.mark.anyio
async def test_build_vector_index_and_search_with_probes(tmp_path: Path) -> None:
    """IVF-PQ needs at least 256 rows per codebook, so this trains on 2000 synthetic 32-d rows.
    `nprobes` and `refine_factor` only reach the index; the nearest row must still come first."""
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, TINY)
    for part in range(2):
        await _fill(index, "a.md", part, 1000, vectors=True)
    await index.finish()

    await index.build_vector_index(2000)

    stats = await index.stats()
    assert stats is not None and stats.has_vector_index is True
    assert stats.vector_index_rows == 2000
    settings = SearchSettings(mode=SearchMode.VECTOR, limit=3, nprobes=4, refine_factor=2)
    rows = await index.search_rows("row7", _vector(7), settings, limit=3)
    assert rows and rows[0]["text"].endswith("row7 lancedb"), "the nearest row, through the index"
    assert "_distance" in rows[0], "a vector query scores by distance, which `row_score` maps"


@pytest.mark.anyio
async def test_search_applies_probes_without_a_vector_index(tmp_path: Path) -> None:
    """The knobs are set on every vector and hybrid query; LanceDB ignores them on a flat scan."""
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, TINY)
    await _fill(index, "a.md", 0, 4, vectors=True)
    await index.finish()
    probed = SearchSettings(nprobes=1, refine_factor=1, candidates=4)

    exact = await index.search_rows(
        "lancedb", _vector(2), msgspec.structs.replace(probed, mode="vector"), limit=2
    )
    hybrid = await index.search_rows("lancedb", _vector(2), probed, limit=2)

    assert [r["text"] for r in exact][0].endswith("row2 lancedb"), "nearest first, scanned exactly"
    assert len(exact) == 2 and len(hybrid) == 4, "hybrid fuses at least `candidates` rows"
    assert "_relevance_score" in hybrid[0], "the fusion reranker scores the merged candidates"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "settings"),
    [
        ("reciprocal rank fusion", SearchSettings(candidates=4, nprobes=4, refine_factor=2)),
        (
            "linear combination of the two rankings",
            SearchSettings(fusion=Fusion.LINEAR, candidates=4, nprobes=4, refine_factor=2),
        ),
    ],
)
async def test_hybrid_search_fuses_both_rankings(
    tmp_path: Path, name: str, settings: SearchSettings
) -> None:
    """Both fusions run over the async hybrid query and score the same column, so `row_score`
    reads one signal whichever one the settings chose."""
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, TINY)
    await _fill(index, "a.md", 0, 6, vectors=True)
    await index.finish()

    rows = await index.search_rows("row3", _vector(3), settings, limit=2)

    assert len(rows) == 4, f"{name}: fused over `candidates` rows, not `limit`"
    assert all("_relevance_score" in row for row in rows), name
    assert all("_score" not in row and "_distance" not in row for row in rows), name
    assert row_score(rows[0]) == pytest.approx(rows[0]["_relevance_score"]), name
    assert rows[0]["text"].endswith("row3 lancedb"), f"{name}: the row both rankings agree on"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "vector", "settings", "leaving", "documents"),
    [
        ("full text, nothing leaving", None, SearchSettings(), frozenset(), {"a.md", "b.md"}),
        ("full text, a.md leaving", None, SearchSettings(), frozenset({"a.md"}), {"b.md"}),
        (
            "vector, a.md leaving",
            _vector(3),
            SearchSettings(mode=SearchMode.VECTOR),
            frozenset({"a.md"}),
            {"b.md"},
        ),
        (
            "hybrid, a.md leaving",
            _vector(3),
            SearchSettings(candidates=3),
            frozenset({"a.md"}),
            {"b.md"},
        ),
        (
            "full text alone (`fts_rows`), a.md leaving",
            "fts_rows",
            SearchSettings(),
            frozenset({"a.md"}),
            {"b.md"},
        ),
    ],
)
async def test_a_leaving_document_takes_no_slot_of_a_search(
    tmp_path: Path,
    name: str,
    vector: list[float] | str | None,
    settings: SearchSettings,
    leaving: frozenset[str],
    documents: set[str],
) -> None:
    """A document on its way out of the collection, as the index was opened with it, is filtered
    before the limit, in every mode: `a.md` holds the best rows for the query, and the slots they
    would take go to `b.md`."""
    writer = CollectionIndex(tmp_path / "index", "notes", tmp_path, TINY)
    await _fill(writer, "a.md", 0, 6, vectors=True)
    await _fill(writer, "b.md", 1, 6, vectors=True)
    await writer.finish()
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, TINY, leaving)

    if vector == "fts_rows":
        rows = await index.fts_rows("row3 lancedb", 3)
    else:
        assert not isinstance(vector, str)
        rows = await index.search_rows("row3 lancedb", vector, settings, 3)

    assert {row["document_id"] for row in rows} == documents, name
    assert len(rows) == 3, f"{name}: filtered before the limit, not after it"


@pytest.mark.anyio
async def test_run_maintenance_compacts_fragments_and_settles_the_counter() -> None:
    """Twenty commits leave twenty fragments; one maintenance pass merges them and clears the
    pending counter without touching the rows."""
    collection = await Collection.create("busy")
    index = collection.index_with(None)
    for part in range(20):
        await _fill(index, "a.md", part, 2)
        await Collection("busy").note_indexed()
    before = await index.stats()
    assert before is not None and before.num_fragments == 20
    assert (await maintenance_state("busy")).pending_documents == 20
    claimed = (await maintenance_state("busy")).pending_documents

    report = await maintenance.run(collection, None, PipelineSettings())
    await Collection("busy").settle_maintenance(claimed, report.ann_trained, report.num_rows)

    assert report.skipped is None
    assert (report.collection, report.num_rows, report.ann_trained) == ("busy", 40, False)
    assert report.fragments_before == 20
    assert report.fragments_after < report.fragments_before
    after = await collection.index_with(None).stats()
    assert after is not None and (after.num_rows, after.num_fragments) == (40, 1)
    state = await maintenance_state("busy")
    assert (state.pending_documents, state.vector_index_rows) == (0, 0)
    assert state.last_maintained_at is not None and state.last_write_at is not None
    assert await Collection.pending_names() == [], "settled, so no boot reschedules it"


@pytest.mark.anyio
async def test_run_maintenance_trains_the_vector_index_once_it_is_big_enough() -> None:
    collection = await Collection.create("vec")
    index = collection.index_with(TINY)
    for part in range(2):
        await _fill(index, "a.md", part, 1000, vectors=True)
    settings = PipelineSettings(ann_min_rows=500)

    report = await maintenance.run(collection, TINY, settings)
    await Collection("vec").settle_maintenance(0, report.ann_trained, report.num_rows)

    assert report.ann_trained is True and report.num_rows == 2000
    assert (await maintenance_state("vec")).vector_index_rows == 2000
    # a fresh handle: the one above is pinned to the table version it opened
    stats = await collection.index_with(TINY).stats()
    assert stats is not None and stats.has_vector_index and stats.has_fts_index
    assert (stats.vector_index_rows, stats.unindexed_rows) == (2000, 0)

    again = await maintenance.run(collection, TINY, settings)

    assert again.ann_trained is False, "the collection has not doubled since it was trained"


@pytest.mark.anyio
async def test_run_maintenance_records_the_corpus_mean_a_search_centres_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under the model it indexes with; a collection without a model sums nothing. The sum
    itself is `embed_cache.corpus_sum`'s, tested with it."""
    asked: list[tuple[str, str]] = []

    async def corpus_sum(name: str, model: str) -> tuple[np.ndarray, int]:
        asked.append((name, model))
        return np.asarray([2.0, 0.0, 0.0, 2.0]), 4

    monkeypatch.setattr(maintenance.embed_cache, "corpus_sum", corpus_sum)
    vec = await Collection.create("vec")
    await _fill(vec.index_with(TINY), "a.md", 0, 3, vectors=True)
    plain = await Collection.create("plain")
    await _fill(plain.index_with(None), "a.md", 0, 3)

    await maintenance.run(vec, TINY, PipelineSettings())
    await maintenance.run(plain, None, PipelineSettings())

    assert asked == [("vec", TINY.cache_name)]
    centre = await Collection.centre(["vec", "plain"], TINY.cache_name)
    assert centre is not None and centre.tolist() == [0.5, 0.0, 0.0, 0.5]


@pytest.mark.parametrize(
    ("name", "reason"),
    [("a collection nobody indexed yet", "no-table"), ("a table an older build wrote", "outdated")],
)
@pytest.mark.anyio
async def test_run_maintenance_skips_what_it_must_not_touch(name: str, reason: str) -> None:
    collection = await Collection.create("skip")
    if reason == "outdated":
        _table_with(collection.index_dir, pa.schema([("doc", pa.string()), ("text", pa.string())]))

    report = await maintenance.run(collection, None, PipelineSettings())

    assert report.skipped == reason, name
    assert (report.num_rows, report.fragments_after, report.ann_trained) == (0, 0, False)


@pytest.mark.anyio
async def test_run_maintenance_of_a_deleted_collection_reports_it_instead_of_raising() -> None:
    """A run may sit in the queue while the collection is deleted, so it never asks for the row it
    needs before it checks that the collection still exists."""
    collection = await Collection.create("gone")
    await _fill(collection.index_with(None), "a.md", 0, 2)
    await remove_collection("gone")

    report = await maintenance.run(collection, None, PipelineSettings())

    assert report.skipped == "no-collection" and report.num_rows == 0
    # settles nothing, raises nothing
    await Collection("gone").settle_maintenance(1, report.ann_trained, report.num_rows)
    assert await Collection("gone").maintenance_state() is None, "no row, not a row of zeroes"


@pytest.mark.anyio
async def test_note_indexed_of_an_unknown_collection_counts_nothing() -> None:
    assert await Collection("ghost").note_indexed() == 0
    assert await Collection.pending_names() == []


@pytest.mark.anyio
async def test_collection_info_reports_the_index_on_demand() -> None:
    collection = await Collection.create("info")
    info = await collection.info()
    assert info.index is None, "no table yet"
    assert (info.overrides, info.effective) == (CollectionOverrides(), ChunkSettings())
    assert info.counts == DocumentCounts() and info.index_outdated is False

    await _fill(collection.index_with(None), "a.md", 0, 3)
    await Collection("info").note_indexed()

    info = await collection.info()
    assert info.index is not None
    assert (info.index.num_rows, info.index.num_fragments) == (3, 1)
    assert (info.index.has_fts_index, info.index.has_vector_index) == (False, False)
    assert (info.maintenance.pending_documents, info.maintenance.last_maintained_at) == (1, None)


@pytest.mark.anyio
async def test_search_falls_back_to_fts_without_an_embedding_model(tmp_path: Path) -> None:
    """A vector query needs something to embed the question with; without a model the mode is
    downgraded rather than failing."""
    path = tmp_path / "index"
    schema = PLAIN_SCHEMA.append(pa.field("vector", pa.list_(pa.float32(), 2)))
    table = lancedb.connect(str(path)).create_table("chunks", schema=schema)
    row = {
        "document_id": "a.md",  # no such row: its id stands in for the name
        "seq": 1,
        "headings": ["H"],
        "text": "hi",
        FTS_COLUMN: "H\n\nhi",
        "vector": [0.1, 0.2],
    }
    table.add([row])
    index = CollectionIndex(path, "notes", tmp_path, None)
    await index.finish()  # build the full-text index the fallback needs

    hits = await index_hits(index, "hi", SearchSettings(mode=SearchMode.VECTOR))

    assert [h.document for h in hits] == ["a.md"]


@pytest.mark.parametrize(
    ("name", "row", "expected"),
    [
        ("cross-encoder score wins", {"_relevance_score": 2.5, "_score": 1.0}, 2.5),
        ("bm25 / fusion score", {"_score": 1.5}, 1.5),
        ("vector distance mapped to 1/(1+d)", {"_distance": 1.0}, 0.5),
        ("nothing scored", {}, 0.0),
    ],
)
def test_score_prefers_the_most_specific_signal(name: str, row: dict, expected: float) -> None:
    assert row_score(row) == expected, name


# One LanceDB row with every `PLAIN_SCHEMA` column at its zero value: what a chunk that carries no
# heading, no ancestry and no pages reads as. `hit` reads every column, so none may be missing.
PLAIN_ROW: dict = dict.fromkeys(PLAIN_SCHEMA.names, "") | {
    "part": 0,
    "seq": 1,
    "line_start": 0,
    "line_end": 0,
    "char_start": 0,
    "char_end": 0,
    "page_start": None,
    "page_end": None,
    "headings": [],
}


def test_hit_of_a_row_with_nothing_optional_set(tmp_path: Path) -> None:
    """Every column of `PLAIN_SCHEMA` is present in any row this build can read, but a chunk may
    carry no heading, no ancestry and no pages: nothing is invented for those."""
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    hit = index.hit(PLAIN_ROW | {"document": "a.md", "seq": 3})
    assert isinstance(hit, Hit)
    assert (hit.source_path, hit.markdown_path, hit.part) == ("", "", 0)
    assert (hit.source_file, hit.markdown_file) == ("", ""), "no path, so nothing to resolve"
    assert (hit.page_start, hit.page_end, hit.headings) == (None, None, [])
    assert (hit.header, hit.location) == ("", "a.md L0-0")


def test_hit_names_the_collection_that_matched_and_builds_a_citation(tmp_path: Path) -> None:
    """A document belongs to no collection, so the hit carries the collection whose table matched
    it; the paths it carries are the document's own."""
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    hit = index.hit(
        {
            "document_id": "1f" + "0" * 30,
            "document": "book.pdf",
            "part": 2,
            "seq": 7,
            "source_path": "documents/1f/book.pdf/original.pdf",
            "markdown_path": "documents/1f/book.pdf/original.pdf.md",
            "line_start": 10,
            "line_end": 20,
            "char_start": 100,
            "char_end": 200,
            "byte_start": 104,
            "byte_end": 210,
            "page_start": 3,
            "page_end": 4,
            "headings": ["Part I", "Chapter 2", "Results"],
            "frame": ["Chapter 2", "Results"],
            "text": "Retries doubled. Latency held. The queue drained by noon.",
            "layout": [
                {"type": "text", "position": 0},
                {"type": "text", "position": 17},
                {"type": "text", "position": 32},
            ],
            "start_reason": "paragraph",
            "end_reason": "length_sentence",
            "_score": 0.5,
        }
    )
    assert hit.collection == "notes"
    assert (hit.part, hit.seq) == (2, 7), "where in the document it sits"
    assert (hit.headings, hit.frame) == (
        ["Part I", "Chapter 2", "Results"],
        ["Chapter 2", "Results"],
    )
    assert hit.header == "Part I > Chapter 2 > Results"
    assert hit.layout == [
        Position(PieceType.TEXT, 0),
        Position(PieceType.TEXT, 17),
        Position(PieceType.TEXT, 32),
    ]
    assert (hit.start_reason, hit.end_reason) == ("paragraph", "length_sentence")
    assert hit.location == "book.pdf p.3-4 L10-20"
    assert (hit.byte_start, hit.byte_end) == (104, 210), "what a search seeks the markdown to"
    assert hit.score == 0.5


@pytest.mark.parametrize(
    ("name", "settings", "expected"),
    [
        ("rrf is the default fusion", SearchSettings(), "RRFReranker"),
        (
            "linear weights the two rankings",
            SearchSettings(fusion=Fusion.LINEAR),
            "LinearCombination",
        ),
        (
            "zero weights fall back to an even split",
            SearchSettings(fusion=Fusion.LINEAR, vector_weight=0.0, bm25_weight=0.0),
            "LinearCombination",
        ),
    ],
)
def test_fusion_reranker_per_setting(name: str, settings: SearchSettings, expected: str) -> None:
    assert type(_fusion(settings)).__name__.startswith(expected), name


@pytest.mark.parametrize(
    ("name", "score", "expected"),
    [
        ("a middling score", 0.5, 0.0),
        ("a strong one", 1 / (1 + math.exp(-3.25)), 3.25),
        ("a weak one", 1 / (1 + math.exp(30.0)), -30.0),
        ("0, where the sigmoid rounds a logit below about -745", 0.0, math.log(math.ulp(0.0))),
        ("1, where it rounds one above about 37", 1.0, 36.7368005696771),
    ],
)
def test_logit_undoes_the_rerankers_sigmoid(name: str, score: float, expected: float) -> None:
    from haskie.collection.index import logit

    assert logit(score) == pytest.approx(expected, abs=1e-9), name


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "logits", "expected"),
    [
        ("nothing retrieved, nothing to rescore", {}, []),
        (
            "each row, in the order given, the sigmoid of its logit, whatever retrieval scored",
            {"short": 5.0, "a longer chunk": 14.0},
            [1 / (1 + math.exp(-5.0)), 1 / (1 + math.exp(-14.0))],
        ),
        (
            "a negative logit scores above 0, where a passage score can use it",
            {"off topic": -9.5, "near": -1.25, "on topic": 0.0},
            [1 / (1 + math.exp(9.5)), 1 / (1 + math.exp(1.25)), 0.5],
        ),
        ("a logit past what exp can take is 0, not an error", {"noise": -800.0}, [0.0]),
    ],
)
async def test_cross_encode_rescores_candidates_in_place(
    monkeypatch: pytest.MonkeyPatch, name: str, logits: dict[str, float], expected: list[float]
) -> None:
    """The cross-encoder is CPU work, so it runs in a worker thread; the sigmoid of its logit
    replaces whatever the retrieval stage put on the row (see `row_score`)."""
    from haskie.collection.index import cross_encode
    from haskie.indexing import models

    checked: list[tuple[str, str]] = []

    async def require_ready(kind: str, model: str) -> None:
        checked.append((kind, model))

    monkeypatch.setattr(models, "require_ready", require_ready)
    hardware: list[Accelerator] = []

    def rerank_scores(model: str, accelerator: Accelerator, q: str, ts: list[str]) -> list[float]:
        hardware.append(accelerator)
        return [logits[t] for t in ts]

    monkeypatch.setattr(embed, "rerank_scores", rerank_scores)
    await save_user_settings(UserSettings(pipeline=PipelineSettings(accelerator=Accelerator.CPU)))
    settings = SearchSettings(reranker=Reranker.CROSS_ENCODER)
    rows = [{"text": text, FTS_COLUMN: text, "frame": [], "_score": 9.0} for text in logits]

    await cross_encode("q", rows, settings)

    assert [row_score(row) for row in rows] == pytest.approx(expected), name
    assert all(0.0 <= row_score(row) < 1.0 for row in rows), f"{name}: bounded"
    assert checked == [("reranker", settings.reranker_model)], name
    assert hardware == [Accelerator.CPU], f"{name}: on the hardware the settings choose"


@pytest.mark.anyio
async def test_an_index_with_an_embedding_stores_a_vector_column(tmp_path: Path) -> None:
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, EmbeddingModel("test/model", 2))
    (chunk_,) = chunk.split("# H\n\n## Sub\n\nbody\n", ChunkSettings())

    row = Row(chunk=chunk_, vector=[0.1, 0.2], seq=1)
    await index.add_parts("g.md", "documents/g.md", "documents/g.md.md", _aparts([(0, [row])]))

    table = await index._existing()
    assert table is not None
    assert (await table.schema()).field("vector").type == pa.list_(pa.float32(), 2)
    (record,) = (await table.to_arrow()).to_pylist()
    assert (record["document_id"], record["headings"], record["part"]) == ("g.md", ["H", "Sub"], 0)
    assert "document" not in record, "the name is read at search time, so a rename moves no row"
    hit = index.hit(record | {"document": "g.md"})
    assert (hit.headings, hit.header) == (["H", "Sub"], "H > Sub"), "the path read back whole"
    assert record["vector"] == pytest.approx([0.1, 0.2])
    assert await index.schema_current() is True


@pytest.mark.anyio
async def test_add_parts_writes_one_fragment_for_many_parts(tmp_path: Path) -> None:
    """Three parts, one commit, one fragment. An empty part inside the group writes nothing but
    does not break the group."""
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    chunks = chunk.split(MD, SMALL)
    numbered = list(enumerate(chunks + chunks, start=1))
    half = len(chunks)
    parts = [
        (0, [Row(chunk=c, seq=seq) for seq, c in numbered[:half]]),
        (1, []),
        (2, [Row(chunk=c, seq=seq) for seq, c in numbered[half:]]),
    ]

    written = await index.add_parts("g.md", "documents/g.md", "documents/g.md.md", _aparts(parts))

    table = await index._existing()
    assert table is not None
    assert written == await table.count_rows() == 2 * len(chunks)
    assert await _fragments(index) == 1, "one commit, however many parts it carried"
    records = (await table.to_arrow()).to_pylist()
    assert sorted({r["part"] for r in records}) == [0, 2], "the empty part is skipped"
    assert sorted(r["seq"] for r in records) == list(range(1, 2 * len(chunks) + 1)), "seq is whole"
    assert {r["markdown_path"] for r in records} == {"documents/g.md.md"}


@pytest.mark.anyio
async def test_delete_parts_removes_only_the_range(tmp_path: Path) -> None:
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    (chunk_,) = chunk.split("# H\n\nbody\n", ChunkSettings())
    for doc in ("a.md", "b.md"):
        await index.add_parts(
            doc,
            "s",
            "m",
            _aparts([(part, [Row(chunk=chunk_, seq=part + 1)]) for part in range(4)]),
        )

    await index.delete_parts("a.md", 1, 3)

    table = await index._existing()
    assert table is not None
    kept = {(r["document_id"], r["part"]) for r in (await table.to_arrow()).to_pylist()}
    assert kept == {("a.md", 0), ("a.md", 3)} | {("b.md", part) for part in range(4)}


@pytest.mark.anyio
async def test_fts_rows_is_empty_without_an_fts_index(tmp_path: Path) -> None:
    """A collection halfway through its first index has rows and no full-text index yet. A
    cross-collection search must not wait for it, so it contributes nothing instead of a scan."""
    missing = CollectionIndex(tmp_path / "missing", "notes", tmp_path, None)
    assert await missing.fts_rows("lancedb", 10) == [], "never indexed, so there is no table"

    _table_with(tmp_path / "empty", PLAIN_SCHEMA)
    empty = CollectionIndex(tmp_path / "empty", "notes", tmp_path, None)
    assert await empty.fts_rows("lancedb", 10) == [], "a table with no rows in it"

    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    rows = [
        Row(chunk=c, seq=seq)
        for seq, c in enumerate(chunk.split("# H\n\nlancedb chapter one\n", ChunkSettings()), 1)
    ]
    await index.add_parts("d.md", "documents/d.md", "documents/d.md.md", _aparts([(0, rows)]))
    assert await index.has_index(FTS_COLUMN) is False, "written, not indexed: the state under test"
    assert await index.fts_rows("lancedb", 10) == [], "rows are there, the full-text index is not"

    await index.finish()

    (row,) = await index.fts_rows("lancedb", 10)
    assert (row["document_id"], row["seq"]) == ("d.md", 1)
    assert row["_score"] > 0, "raw BM25, which is what the cross-collection merge sorts on"


@pytest.mark.anyio
async def test_a_hit_carries_what_the_models_read_and_its_pieces(tmp_path: Path) -> None:
    """The UI shows a chunk as it was embedded: its frame, then its text piece by piece."""
    rules = " ".join(f"Rule {i:02d} raised lancedb costs." for i in range(12))
    text = "# Costs\n\n## Europe\n\n" + rules
    chunks = chunk.split(text, ChunkSettings(chunk_size=120))
    assert len(chunks) > 2
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    rows = [Row(chunk=c, seq=seq) for seq, c in enumerate(chunks, 1)]
    await index.add_parts("d.md", "documents/d.md", "documents/d.md.md", _aparts([(0, rows)]))
    await index.finish()

    hits = {h.seq: h for h in await index_hits(index, "lancedb", SearchSettings(limit=20))}
    middle = hits[2]
    assert middle.header == "Costs > Europe", "the heading path the models read it after"
    assert middle.layout == chunks[1].layout, "stored as the chunk has them"
    starts = [*(p.position for p in middle.layout), len(middle.text)]
    cut = [
        Piece(p.type, middle.text[a:b])
        for p, a, b in zip(middle.layout, starts, starts[1:], strict=False)
    ]
    assert cut == chunks[1].pieces, "they cut the text back into its pieces"
    assert hits[1].text.startswith("Rule 00"), "the first chunk starts at its text"
    assert hits[1].frame == ["Costs", "Europe"], "its headings are the frame"


@pytest.mark.anyio
async def test_search_rows_returns_raw_rows_without_cutting(tmp_path: Path) -> None:
    """Retrieval only: as many rows as the caller asked for, carrying the engine's own score and
    no cross-encoder score."""
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    chunks = [
        c for i in range(6) for c in chunk.split(f"# H\n\nlancedb chapter {i}\n", ChunkSettings())
    ]
    assert len(chunks) == 6, "one chunk per text, or the row counts below mean nothing"
    rows = [Row(chunk=c, seq=seq) for seq, c in enumerate(chunks, 1)]
    await index.add_parts("d.md", "documents/d.md", "documents/d.md.md", _aparts([(0, rows)]))
    await index.finish()
    settings = SearchSettings(limit=2, candidates=4)

    rows = await index.search_rows("lancedb", None, settings, 4)

    assert len(rows) == 4, "the fetch size wins over settings.limit"
    assert all("_score" in row and "_relevance_score" not in row for row in rows)
    assert {row["document_id"] for row in rows} == {"d.md"}
    assert len(await index.search_rows("lancedb", None, settings, 100)) == 6, "no more than exist"
    missing = CollectionIndex(tmp_path / "missing", "notes", tmp_path, None)
    assert await missing.search_rows("lancedb", None, settings, 4) == [], "no table, no rows"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "settings", "vector", "built", "expected"),
    [
        ("lexical before the index: nothing", SearchSettings(), None, False, (0, None)),
        ("hybrid before the index: its vector half", SearchSettings(), 2, False, (2, "_distance")),
        (
            "vector before the index: as ever",
            SearchSettings(mode=SearchMode.VECTOR),
            2,
            False,
            (2, "_distance"),
        ),
        ("lexical after the index: BM25", SearchSettings(), None, True, (2, "_score")),
        ("hybrid after the index: fused", SearchSettings(), 2, True, (4, "_relevance_score")),
    ],
)
async def test_search_rows_answers_what_a_table_without_its_fts_index_can(
    tmp_path: Path,
    name: str,
    settings: SearchSettings,
    vector: int | None,
    built: bool,
    expected: tuple[int, str | None],
) -> None:
    """A collection in the middle of its first index has rows and no full-text index. LanceDB
    refuses any full-text query then, so a search must not send one and fail with it."""
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, TINY)
    await _fill(index, "a.md", 0, 4, vectors=True)
    if built:
        await index.finish()
    assert await index.has_index(FTS_COLUMN) is built, f"{name}: the state under test"
    settings = msgspec.structs.replace(settings, candidates=4)

    rows = await index.search_rows(
        "lancedb", None if vector is None else _vector(vector), settings, limit=2
    )

    count, column = expected
    assert len(rows) == count, name
    assert all(column in row for row in rows), f"{name}: scored by {column}"
    if vector is not None:
        assert rows[0]["text"].endswith(f"row{vector} lancedb"), f"{name}: the nearest row first"


# --- pipeline ----------------------------------------------------------------------
#
# The three stages called straight through, in the order `workflows` calls them: convert once per
# document, embed once per distinct `embed_cache.Params`, index once per collection.


async def _fragments(index: CollectionIndex) -> int:
    """Data fragments of the table: LanceDB writes one per commit that carries rows."""
    table = await index._existing()
    if table is None:
        return 0
    # lancedb annotates stats() as a dataclass but returns plain dicts
    return (await table.stats())["fragment_stats"]["num_fragments"]  # ty: ignore[not-subscriptable]


async def _indexed_rows(collection: Collection) -> int:
    table = await (await collection.index())._existing()
    return 0 if table is None else await table.count_rows()


async def _convert(doc: Document, batch_pages: int = 10) -> list[pipeline.Batch]:
    batches = await pipeline.plan_convert(doc, batch_pages)
    ocr = sum([await pipeline.convert_batch(doc, batch) for batch in batches])
    await pipeline.finalize_convert(doc, batches, ocr)
    return batches


async def _embed(
    doc: Document,
    chunking: ChunkSettings,
    embedding: EmbeddingModel | None = None,
    batch_pages: int = 10,
) -> str:
    params = embed_cache.params(doc, chunking, embedding)
    cache_id = embed_cache.key(params)
    batches = await pipeline.plan_embed(doc, batch_pages)
    for batch in batches:
        await pipeline.embed_batch(doc, batch, cache_id, chunking, embedding)
    await pipeline.finalize_embed(doc, params, embedding.dims if embedding else None, len(batches))
    await _describe(doc, cache_id, embedding, Descriptors.C_TF_IDF, Accelerator.AUTO)
    return cache_id


async def _describe(
    doc: Document,
    cache_id: str,
    embedding: EmbeddingModel | None,
    by: Descriptors,
    accelerator: Accelerator,
) -> int:
    """The describe stage as the workflow runs it: plan, every batch, then the finalizer."""
    batches = await pipeline.plan_describe(doc, cache_id, by)
    for batch in batches:
        await pipeline.describe_batch(doc, cache_id, embedding, by, accelerator, batch)
    return await pipeline.finalize_describe(doc, cache_id, by, len(batches))


async def _index(
    collection: Collection,
    doc: Document,
    cache_id: str,
    group_parts: int = 50,
    embedding: EmbeddingModel | None = None,
) -> int:
    written = 0
    for batch in await pipeline.plan_index(doc, cache_id, group_parts):
        written += await pipeline.index_batch(collection, doc, cache_id, batch, embedding)
    await pipeline.finalize_index(collection, embedding)
    return written


def _parts_of(doc: Document, batches: list[pipeline.Batch]) -> list[str]:
    markdown = doc.markdown.read_bytes()
    return [markdown[b.byte_offset : b.byte_end].decode() for b in batches]


def _rows_of(doc: Document, batches: list[pipeline.Batch]) -> list[list[Row]]:
    """The rows each embed batch wrote, a list per batch."""
    return [
        msgspec.json.decode(
            embed_cache.rows_path(doc.id, "cache", batch.seq).read_bytes(), type=list[Row]
        )
        for batch in batches
    ]


@pytest.mark.anyio
async def test_the_markdown_is_cut_where_its_sections_start_and_packed_up_to_a_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sections go whole into a part, as many as fit a batch of pages; a part never ends inside
    one while a heading is there to end it at, and the parts tile the markdown."""
    monkeypatch.setattr(pipeline, "PAGE_CHARS", 60)
    doc = await import_row("g.md")
    sections = [f"# Chapter {n}\n\n{'Words of the chapter. ' * 2}\n\n" for n in range(5)]
    doc.markdown.write_text("".join(sections))

    batches = await pipeline.plan_embed(doc, 2)

    assert _parts_of(doc, batches) == [
        sections[0] + sections[1],
        sections[2] + sections[3],
        sections[4],
    ], "two sections a part: a third would not fit 120 characters"
    assert [(b.start_reason, b.end_reason) for b in batches] == [
        (CutReason.EDGE, CutReason.HEADING),
        (CutReason.HEADING, CutReason.HEADING),
        (CutReason.HEADING, CutReason.EDGE),
    ]
    assert [b.char_offset for b in batches] == [0, len(sections[0]) * 2, len(sections[0]) * 4]


@pytest.mark.anyio
async def test_a_part_is_embedded_under_the_headings_an_earlier_part_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A section longer than a batch is cut at a page marker inside it: a chapter opened in one
    part still frames the chunks of the next, both in the chunk's headings and in what the model
    embeds. Where the two parts meet is a part boundary, not the document's edge: the section
    goes on across it."""
    monkeypatch.setattr(pipeline, "PAGE_CHARS", 100)
    doc = await import_row("g.md")
    long = "One leader takes writes. " * 6
    markdown = (
        f"# Replication\n\n## Leaders\n\n{long}\n\n<!-- page 2 -->\n\nFollowers apply the log."
    )
    doc.markdown.write_text(markdown)

    batches = await pipeline.plan_embed(doc, 1)

    assert _parts_of(doc, batches) == [
        "# Replication\n\n",
        f"## Leaders\n\n{long}\n\n",
        "<!-- page 2 -->\n\nFollowers apply the log.",
    ], "at the one heading inside the batch, then at the page marker inside the long section"
    assert [[text for _, text in b.opened] for b in batches] == [
        [],
        ["Replication"],
        ["Replication", "Leaders"],
    ]
    assert [(b.start_reason, b.end_reason) for b in batches] == [
        (CutReason.EDGE, CutReason.HEADING),
        (CutReason.HEADING, CutReason.PART),
        (CutReason.PART, CutReason.EDGE),
    ]
    for batch in batches:
        await pipeline.embed_batch(doc, batch, "cache", SMALL, None)
    _, leaders, followers = _rows_of(doc, batches)
    assert {tuple(row.chunk.headings) for row in [*leaders, *followers]} == {
        ("Replication", "Leaders")
    }
    assert followers[0].chunk.char_start == markdown.index("Followers"), "offsets into the file"
    assert (leaders[0].chunk.start_reason, leaders[-1].chunk.end_reason) == (
        CutReason.HEADING,
        CutReason.PART,
    )
    assert followers[0].chunk.start_reason == CutReason.PART


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "second", "meets"),
    [
        ("text goes on: the section crosses the boundary", "More of it.", CutReason.PART),
        (
            "the next part opens with a heading, behind its page marker: the section ends",
            "# Chapter 4\n\nMore.",
            CutReason.HEADING,
        ),
    ],
)
async def test_where_two_parts_meet_is_cut_for_what_comes_next(
    name: str, second: str, meets: CutReason, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PDF chapter often starts on a new page, so a part cut at a page marker often opens with
    its heading. The cut between the two parts is then a heading on both sides, and a short
    section just before it is whole: nothing grows across the heading."""
    monkeypatch.setattr(pipeline, "PAGE_CHARS", 50)
    doc = await import_row("g.md")
    doc.markdown.write_text(f"# Chapter 3\n\n{'A long note. ' * 6}\n\n<!-- page 11 -->\n\n{second}")

    batches = await pipeline.plan_embed(doc, 1)
    for batch in batches:
        await pipeline.embed_batch(doc, batch, "cache", SMALL, None)
    before, after = _rows_of(doc, batches)

    assert [(b.start_reason, b.end_reason) for b in batches] == [
        (CutReason.EDGE, meets),
        (meets, CutReason.EDGE),
    ], name
    assert (before[-1].chunk.end_reason, after[0].chunk.start_reason) == (meets, meets), name


@pytest.mark.anyio
async def test_a_heading_is_cut_ahead_of_its_page_marker() -> None:
    """A PDF chapter starts behind the marker of its page. The cut goes ahead of the marker, so
    the chapter's part knows its page from the first chunk, and no part holds a marker alone."""
    doc = await import_row("g.md")
    pages = [
        f"<!-- page {n} -->\n\n# Chapter {n}\n\n{'Words of the chapter. ' * 4}".strip()
        for n in (1, 2, 3)
    ]
    doc.markdown.write_text("\n\n".join(pages))

    batches = await pipeline.plan_embed(doc, 1)
    for batch in batches:
        await pipeline.embed_batch(doc, batch, "cache", SMALL, None)

    assert [part.strip() for part in _parts_of(doc, batches)] == pages
    assert [(b.start_reason, b.end_reason) for b in batches] == [
        (CutReason.EDGE, CutReason.HEADING),
        (CutReason.HEADING, CutReason.HEADING),
        (CutReason.HEADING, CutReason.EDGE),
    ]
    assert [
        {(row.chunk.page_start, row.chunk.page_end) for row in rows}
        for rows in _rows_of(doc, batches)
    ] == [{(1, 1)}, {(2, 2)}, {(3, 3)}]


@pytest.mark.anyio
async def test_a_part_cut_mid_page_starts_on_the_page_open_there() -> None:
    """A heading partway down a page opens a part with no marker of its own ahead of its text:
    its chunks are on the page an earlier part's marker opened, until the part's first marker."""
    doc = await import_row("g.md")
    words = "Words of the chapter. " * 4
    doc.markdown.write_text(
        f"<!-- page 1 -->\n\n# One\n\n{words}\n\n<!-- page 2 -->\n\n{words}\n\n# Two\n\n{words}"
        f"\n\n<!-- page 3 -->\n\n{words}"
    )

    batches = await pipeline.plan_embed(doc, 1)
    for batch in batches:
        await pipeline.embed_batch(doc, batch, "cache", SMALL, None)

    assert [part.lstrip()[:5] for part in _parts_of(doc, batches)] == [
        "<!-- ",
        "<!-- ",
        "# Two",
        "<!-- ",
    ], "cut at the heading on page 2, then at the marker of page 3"
    assert [b.page for b in batches] == [None, 1, 2, 2]
    two = _rows_of(doc, batches)[2]
    assert {(row.chunk.page_start, row.chunk.page_end) for row in two} == {(2, 2)}


@pytest.mark.anyio
async def test_plan_embed_requires_a_converted_document() -> None:
    doc = await import_row("g.md")
    with pytest.raises(FileNotFoundError, match="markdown missing"):
        await pipeline.plan_embed(doc, 10)


@pytest.mark.anyio
async def test_finalize_embed_requires_the_rows_of_every_part() -> None:
    doc = await import_row("g.md")
    await _convert(doc)

    params = embed_cache.params(doc, ChunkSettings(), None)
    with pytest.raises(FileNotFoundError):
        await pipeline.finalize_embed(doc, params, None, 1)
    assert await embed_cache.lookup(params) is None, "and nothing was published"


@pytest.mark.anyio
async def test_describe_writes_the_descriptors_by_the_strategy_asked_for(
    dbos, monkeypatch: pytest.MonkeyPatch
) -> None:
    """c-TF-IDF needs no model without an embedding one; the llm strategy waits for its describer
    (`ModelLoading`, which a workflow sleeps out) and then asks it once per section with prose,
    and the file says which strategy wrote what it holds."""
    from haskie.indexing import gguf_models, models

    monkeypatch.setattr(gguf_models, "available", lambda: True)  # llama.cpp stood in for

    doc = await import_row("g.md")
    await _convert(doc)
    cache_id = await _embed(doc, SMALL)  # described by c-TF-IDF, without a model
    assert await embed_cache.described_by(doc.id, cache_id) == Descriptors.C_TF_IDF
    by_weight = await embed_cache.read_sections(doc.id, cache_id)
    assert any(one.descriptors for one in by_weight)

    prompts: list[str] = []

    def reply(name: str, accelerator: Accelerator, prompt: str, max_tokens: int) -> str:
        assert (name, accelerator) == (gguf_models.DESCRIBER, Accelerator.AUTO)
        prompts.append(prompt)
        return "Topic one | Topic two"

    monkeypatch.setattr(embed, "reply", reply)
    on_cpu = partial(_describe, doc, cache_id, None, Descriptors.LLM, Accelerator.CPU)
    with pytest.raises(PermanentError, match="runs on gguf on the Apple GPU"):
        await on_cpu()  # its model would never load: no wait, an error
    describe = partial(_describe, doc, cache_id, None, Descriptors.LLM, Accelerator.AUTO)
    with pytest.raises(models.ModelLoading):
        await describe()
    assert prompts == [] and await embed_cache.described_by(doc.id, cache_id) == (
        Descriptors.C_TF_IDF
    ), "nothing asked, nothing written"

    models._mark_ready(models._model_id(models.ModelKind.DESCRIBER, gguf_models.DESCRIBER))
    count = await describe()

    described = await embed_cache.read_sections(doc.id, cache_id)
    assert count == len(described) == len(by_weight)
    assert await embed_cache.described_by(doc.id, cache_id) == Descriptors.LLM
    assert len(prompts) == sum(bool(one.descriptors) for one in described) > 0
    assert {tuple(one.descriptors) for one in described} <= {("Topic one", "Topic two"), ()}
    assert not embed_cache.scratch_dir(doc.id, cache_id).exists(), "the batches' files are gone"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("by", "sections", "expected"),
    [
        pytest.param(Descriptors.LLM, 0, [], id="llm-no-sections-no-batch"),
        pytest.param(Descriptors.LLM, 16, [(0, 16)], id="llm-one-full-batch"),
        pytest.param(Descriptors.LLM, 17, [(0, 16), (16, 17)], id="llm-a-batch-and-one-over"),
        pytest.param(Descriptors.C_TF_IDF, 40, [(0, 40)], id="c-tf-idf-one-batch-for-all"),
    ],
)
async def test_describe_plans_its_batches_by_section(
    monkeypatch: pytest.MonkeyPatch, by: Descriptors, sections: int, expected: list
) -> None:
    """The llm strategy asks about sixteen sections a batch; c-TF-IDF weighs every section against
    the whole document, so it is one batch however many there are."""
    doc = await import_row("plan.md")
    # the sections a real chunking names: a chapter each, one chunk each, on its own lines
    found = [
        Section(
            id=f"s{index}",
            parent_id=None,
            headings=[f"Chapter {index + 1}"],
            seq_start=index + 1,
            seq_end=index + 1,
            line_start=3 * index + 1,
            line_end=3 * index + 3,
            char_start=40 * index,
            char_end=40 * index + 40,
            byte_start=40 * index,
            byte_end=40 * index + 40,
            page_start=None,
            page_end=None,
        )
        for index in range(sections)
    ]

    async def read_sections(doc_id: str, cache_id: str) -> list:
        return found

    monkeypatch.setattr(embed_cache, "read_sections", read_sections)

    batches = await pipeline.plan_describe(doc, "cache", by)

    assert [(one.start, one.end) for one in batches] == expected
    assert [one.seq for one in batches] == list(range(len(expected)))


@pytest.mark.anyio
async def test_plan_index_requires_the_cache_file() -> None:
    doc = await import_row("g.md")
    with pytest.raises(FileNotFoundError):
        await pipeline.plan_index(doc, embed_cache.key(embed_cache.params(doc, SMALL, None)), 50)


@pytest.mark.anyio
async def test_convert_and_embed_write_atomically() -> None:
    doc = await import_row("g.md")
    (batch,) = await pipeline.plan_convert(doc, 10)

    assert await pipeline.convert_batch(doc, batch) == 0, "no OCR pages in markdown"
    await pipeline.finalize_convert(doc, [batch], 0)
    params = embed_cache.params(doc, SMALL, None)
    (part,) = await pipeline.plan_embed(doc, 10)
    chunks = await pipeline.embed_batch(doc, part, embed_cache.key(params), SMALL, None)

    assert chunks == len(chunk.split(MD, SMALL))
    assert doc.markdown.read_text() == MD
    assert doc.part_path(0).read_text() == MD, "the convert part stays"
    assert list(doc.parts_dir.glob("*.tmp")) == [], "no temp file left behind"
    assert not doc.markdown.with_name(doc.markdown.name + ".tmp").exists()


@pytest.mark.anyio
async def test_a_reconversion_starts_the_documents_outputs_over() -> None:
    """The parts and the markdown are outputs of the conversion, so `plan_convert` rebuilds them.
    The cached embeddings were chunked from that markdown, so they go too. `embed_cache.forget`
    drops them, and `workflows.import_document` runs it before the convert stage."""
    doc = await import_row("g.md")
    await _convert(doc)
    cache_id = await _embed(doc, SMALL)
    assert embed_cache.file_path(doc.id, cache_id).exists()

    await embed_cache.forget(doc.id)
    await pipeline.plan_convert(doc, 10)

    assert not doc.markdown.exists(), "the assembled markdown is rebuilt"
    assert list(doc.parts_dir.iterdir()) == [], "and so are the parts"
    assert not doc.embeddings_dir.exists(), "no cache file survives a reconversion"
    assert await embed_cache.entries(doc.id) == [], "and no cache row either"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "marks", "expected"),
    [
        ("no bookmarks: every four pages", [], [(0, 4), (4, 6)]),
        ("a chapter starting on page 4 ends the batch before it", [0, 3], [(0, 3), (3, 6)]),
        ("short chapters packed, as many as fit four pages", [0, 1, 2, 5], [(0, 2), (2, 6)]),
    ],
)
async def test_a_pdf_converts_in_batches_cut_where_its_bookmarks_start(
    name: str, marks: list[int], expected: list[tuple[int, int]], tmp_path: Path
) -> None:
    """At most four pages a batch, each ending where the last section starting inside it does."""
    from pypdf import PdfWriter

    plain = tmp_path / "plain.pdf"
    plain.write_bytes(text_pdf([f"page {n}" for n in range(6)]))
    writer = PdfWriter(clone_from=str(plain))
    for page in marks:
        writer.add_outline_item(f"Chapter {page}", page)
    writer.write(tmp_path / "book.pdf")
    doc = await import_row("book.pdf", (tmp_path / "book.pdf").read_bytes())

    batches = await pipeline.plan_convert(doc, 4)

    assert [(b.start, b.end) for b in batches] == expected, name
    assert [b.seq for b in batches] == list(range(len(expected)))


@pytest.mark.anyio
async def test_convert_batch_of_a_pdf_reports_pages_needing_ocr() -> None:
    doc = await import_row("scan.pdf", text_pdf(["text page", None]))
    batches = await pipeline.plan_convert(doc, 10)

    assert await pipeline.convert_batch(doc, batches[0]) == 1
    assert "needs OCR, skipped" in doc.part_path(0).read_text()


@pytest.mark.anyio
async def test_the_pipeline_indexes_a_markdown_document_into_a_collection() -> None:
    """Convert, embed into the cache, then read the cache into the collection's table: the whole
    path a document takes, without the durable runtime around it."""
    collection = await Collection.create("notes")
    doc = await import_row("guide.md")
    chunking = await collection.chunk_settings()

    await _convert(doc)
    cache_id = await _embed(doc, chunking)
    written = await _index(collection, doc, cache_id)

    assert written == await _indexed_rows(collection) > 0
    (entry,) = await embed_cache.entries(doc.id)
    assert (entry.id, entry.rows) == (cache_id, written)
    (hit,) = await collection_hits(collection.name, "lancedb")
    assert (hit.collection, hit.document) == ("notes", "guide.md")
    assert hit.source_file == str(doc.original), "the hit points at the document's own files"
    assert hit.markdown_file == str(doc.markdown)
    assert Path(hit.markdown_file).read_text() == MD
    sections = await embed_cache.read_sections(doc.id, cache_id)
    by_header = {one.header: one for one in sections}
    assert list(by_header) == ["", "Title", "Title > Alpha", "Title > Beta"], "in document order"
    assert by_header["Title > Alpha"].parent_id == by_header["Title"].id
    assert hit.section_id == by_header["Title > Alpha"].id, "the chunk names its section"
    assert hit.section_ids == [by_header[one].id for one in ("", "Title", "Title > Alpha")]
    assert hit.id == ids.md5(f"{doc.id}/c/{hit.seq}".encode()), "the chunk's document and seq"
    assert [one.id for one in sections] == [
        ids.md5(f"{doc.id}/s/{position}".encode()) for position in range(len(sections))
    ], "each section's document and place among its sections"


@pytest.mark.anyio
async def test_index_batch_group_is_idempotent() -> None:
    """A replay after a crash between the LanceDB commit and the step checkpoint must rewrite the
    group rather than append it a second time."""
    collection = await Collection.create("groups")
    doc = await import_row("p.pdf", text_pdf(["alpha one", "beta two", "gamma three"]))
    await _convert(doc, batch_pages=1)
    cache_id = await _embed(doc, SMALL, batch_pages=1)

    assert await embed_cache.row_groups(doc.id, cache_id) == 3, "one group per embed part"
    assert [(b.seq, b.start, b.end) for b in await pipeline.plan_index(doc, cache_id, 2)] == [
        (0, 0, 2),
        (1, 2, 3),
    ], "the last group holds the remainder"
    (group,) = await pipeline.plan_index(doc, cache_id, 50)
    assert (group.seq, group.start, group.end) == (0, 0, 3), "three parts in one commit"

    written = await pipeline.index_batch(collection, doc, cache_id, group, None)
    assert written == await _indexed_rows(collection) > 0
    assert await _fragments(await collection.index()) == 1

    written_again = await pipeline.index_batch(collection, doc, cache_id, group, None)
    assert written_again == written, "the same group again"
    assert await _indexed_rows(collection) == written, "the range was replaced, not appended"


@pytest.mark.anyio
async def test_two_collections_with_the_same_chunk_settings_share_one_cache_entry() -> None:
    """With the cache, the second collection computes nothing. It reads the parquet file the first
    one left and writes its own table from it."""
    alpha = await Collection.create("alpha")
    beta = await Collection.create("beta")
    # smaller than a section of `MD`: every heading starts a chunk anyway, so only a size below
    # a section's length chunks the same markdown into more of them
    await beta.set_overrides(CollectionOverrides(chunk_size=20))
    doc = await attachable("shared.md")
    await _convert(doc)

    first = await _embed(doc, await alpha.chunk_settings())
    second = await _embed(doc, await beta.chunk_settings())

    assert first != second, "different chunk settings, different entries, no collision"
    assert {entry.id for entry in await embed_cache.entries(doc.id)} == {first, second}
    assert await _embed(doc, await alpha.chunk_settings()) == first, "the same settings, same id"

    await alpha.add(doc.id)
    await beta.add(doc.id)
    rows_alpha = await _index(alpha, doc, first)
    rows_beta = await _index(beta, doc, second)

    assert rows_alpha > 0 and rows_beta > rows_alpha, "beta chunks the same markdown smaller"
    assert await collection_hits("alpha", "lancedb") and await collection_hits("beta", "lancedb")
    assert await document.collections_of(doc.id) == ["alpha", "beta"]


# --- sessions and cross-collection search --------------------------------------------
#
# `CollectionIndex` answers retrieval (`search_rows`, fused or as its two halves) and row-to-Hit
# (`hit`); the search embeds the query once (`retrieval.plan`), fans out and
# rescores once. Over several collections `retrieval.fan_out` ranks each half over all of them
# and `retrieval.merge` fuses the two, as LanceDB fuses one table's. `search.text.merge` merges raw
# BM25 scores too: one lexical scorer with the same tokenizer answers in every collection. Both
# count a passage once, because one document may be a member of several of the collections being
# searched.

# How long one collection of a fan-out may wait for the other before the test calls it sequential.
CONCURRENT_SEARCH_SECONDS = 5.0

# The query one cursor below belongs to: (query, collections, page_size).
TEXT_QUERY = ("lancedb", ["alpha", "beta"], 25)


@pytest.mark.parametrize(
    ("name", "session_id", "collections", "error", "match"),
    [
        ("empty id", "", ["a"], InvalidInput, "session id must be 1..128"),
        ("id too long", "x" * 129, ["a"], InvalidInput, "session id must be 1..128"),
        (
            "more collections than the cap",
            "s",
            [f"c{i}" for i in range(101)],
            InvalidInput,
            "at most 100 collections",
        ),
        ("unknown collection", "s", ["ghost"], NotFound, "collection not found: ghost"),
    ],
)
@pytest.mark.anyio
async def test_set_collections_rejects(
    name: str, session_id: str, collections: list[str], error: type[Exception], match: str
) -> None:
    from haskie.search import session

    await Collection.create("a")
    with pytest.raises(error, match=match):
        await session.set_collections(session_id, collections)
    assert await session.load() == {}, f"nothing stored for a rejected request: {name}"


@pytest.mark.anyio
async def test_set_collections_deduplicates_and_keeps_order() -> None:
    from haskie.search import session

    for name in ("a", "b"):
        await Collection.create(name)
    assert await session.set_collections("s1", ["b", "a", "b"]) == ["b", "a"]
    assert await session.load() == {"s1": ["b", "a"]}
    assert await session.collections_for("s1") == ["b", "a"]
    assert await session.collections_for("unknown") == []


@pytest.mark.anyio
async def test_session_collections_keep_their_order_and_survive_reorder() -> None:
    """The selection is rows with a position, not a JSON list: reordering it rewrites the rows,
    and a session that selected nothing is still a session."""
    from haskie.search import session

    for name in ("a", "b", "c"):
        await Collection.create(name)

    await session.set_collections("s1", ["c", "a", "b"])
    assert await session.collections_for("s1") == ["c", "a", "b"]

    assert await session.set_collections("s1", ["b", "c"]) == ["b", "c"], "the selection is new"
    assert await session.load() == {"s1": ["b", "c"]}
    async with db.connect() as conn:
        chosen = tables.session_collections.c
        rows = await conn.execute(
            select(chosen.collection, chosen.position)
            .where(chosen.session_id == "s1")
            .order_by(chosen.position)
        )
        assert list(rows) == [("b", 0), ("c", 1)], "one row per collection, renumbered from zero"

    await session.set_collections("s1", [])
    assert await session.load() == {"s1": []}, "an empty selection keeps the session itself"


@pytest.mark.anyio
async def test_session_search_skips_a_collection_that_disappeared(caplog) -> None:
    """Deleting a collection drops it from every session (one cascade), so a name without a row
    can only come from a delete between the two reads of the search. It is skipped, not raised."""
    from haskie.search import flow

    with caplog.at_level("WARNING"):
        assert await flow.chunks(["ghost"], "anything") == []
    assert events(caplog) == ["session_collection_missing"]


@pytest.mark.anyio
async def test_session_search_reads_its_collections_concurrently(monkeypatch) -> None:
    """Two collections, two events: each retrieval announces itself and then waits for the other,
    so the search can only answer at all if the fan-out overlapped. A sequential fan-out would
    hold the first retrieval until the wait times out, and the timeout fails the search."""
    import asyncio

    from haskie.search import flow, session

    for name in ("a", "b"):
        await Collection.create(name)
    await session.set_collections("s1", ["a", "b"])
    arrived = {"a": asyncio.Event(), "b": asyncio.Event()}

    async def paired(self, query, vector, settings_, limit, vectors=True, fused=True) -> list[dict]:
        arrived[self.collection].set()
        other = arrived["b" if self.collection == "a" else "a"]
        await asyncio.wait_for(other.wait(), CONCURRENT_SEARCH_SECONDS)
        return []

    monkeypatch.setattr(CollectionIndex, "search_rows", paired)

    assert await flow.chunks(await session.collections_for("s1"), "anything") == []
    assert all(event.is_set() for event in arrived.values()), "both collections were read"


@pytest.mark.anyio
async def test_session_search_counts_a_passage_once_across_collections() -> None:
    """The same document in two chosen collections puts the same chunk in both rankings. A caller
    wants one hit per passage, so it is credited to the first collection that returned it and the
    copy is dropped before the ranks are counted."""
    from haskie.search import flow, session

    for name in ("alpha", "beta"):
        collection = await Collection.create(name)
        index = collection.index_with(None)
        # three chunks that say different things: the copies across the two collections are what
        # merges here, not three chunks near-duplicating one another
        texts = [
            "LanceDB keeps each collection as one table of chunks.",
            "A hybrid lancedb query fuses BM25 with the vector ranking.",
            "Compaction merges the small fragments lancedb writes leave behind.",
        ]
        # each chunk at its own offset, one line after the other, as the chunker cuts them
        starts = [sum(len(before) + 1 for before in texts[:at]) for at in range(len(texts))]
        rows = [
            _row(text, None, seq=seq, char_start=start)
            for seq, (text, start) in enumerate(zip(texts, starts, strict=True), start=1)
        ]
        await index.add_parts(
            "shared.md",
            "documents/shared.md",
            "documents/shared.md.md",
            _aparts([(0, rows)]),
        )
        await index.finish()
    await session.set_collections("s1", ["alpha", "beta"])

    hits = await flow.chunks(await session.collections_for("s1"), "lancedb", limit=10)

    assert len(hits) == 3, "three chunks, not six: the copies are merged away"
    passages = {(hit.document, hit.seq) for hit in hits}
    assert passages == {("shared.md", seq) for seq in range(1, 4)}
    assert {hit.collection for hit in hits} == {"alpha"}, "the first collection that held it"


@pytest.mark.anyio
async def test_fan_out_counts_a_span_once_across_collections_that_chunk_it_two_ways() -> None:
    """Two collections chunk one document with other chunk sizes: alpha a sentence per chunk,
    beta the first two sentences as its chunk 1. Beta's chunk 1 is other text than alpha's, so it
    stays; beta's chunk 2 is the very span of alpha's chunk 3, so it is a copy and goes."""
    from haskie.search import retrieval

    texts = [
        "LanceDB keeps each collection as one table of chunks.",
        "A hybrid lancedb query fuses BM25 with the vector ranking.",
        "Compaction merges the small fragments lancedb writes leave behind.",
    ]
    starts = [sum(len(before) + 1 for before in texts[:at]) for at in range(len(texts))]
    chunked = {
        "alpha": [
            _row(text, None, seq, start)
            for seq, (text, start) in enumerate(zip(texts, starts, strict=True), 1)
        ],
        "beta": [_row("\n".join(texts[:2]), None, 1, 0), _row(texts[2], None, 2, starts[2])],
    }
    for name, rows in chunked.items():
        index = (await Collection.create(name)).index_with(None)
        await index.add_parts(
            "shared.md",
            "documents/shared.md",
            "documents/shared.md.md",
            _aparts([(0, rows)]),
        )
        await index.finish()
    (where,) = await retrieval.plan(["alpha", "beta"], ["lancedb"]) or []

    pool = await retrieval.fan_out(where, "lancedb", 10)

    assert set(pool.rows) == {
        ("alpha", "shared.md", 1),
        ("alpha", "shared.md", 2),
        ("alpha", "shared.md", 3),
        ("beta", "shared.md", 1),
    }
    assert set(pool.rankings) == {retrieval.TEXT_RANKING}, "no embedding: the BM25 half alone"
    ranked = sorted(key for key, _ in pool.rankings[retrieval.TEXT_RANKING])
    assert ranked == sorted(pool.rows), "one ranking of all"
    assert pool.rows[("beta", "shared.md", 1)][1]["text"] == "\n".join(texts[:2])


@pytest.mark.parametrize(
    ("name", "first", "rankings", "answered"),
    [
        (
            "hybrid: each collection answers both halves, ranked over both",
            ("hybrid", True, True, False),
            {"vector", "text"},
            {"_distance", "_score"},
        ),
        (
            "vector mode: no BM25 half from it",
            ("vector", True, True, False),
            {"vector", "text"},
            {"_distance"},
        ),
        (
            "vector mode, its spans shared: no BM25 half, though its rows took the other's",
            ("vector", True, True, True),
            {"vector", "text"},
            {"_distance"},
        ),
        (
            "a table written without vectors answers by full text in any mode",
            ("vector", False, True, False),
            {"vector", "text"},
            {"_score"},
        ),
        (
            "no full-text index yet: the vector half alone",
            ("hybrid", True, False, False),
            {"vector", "text"},
            {"_distance"},
        ),
    ],
)
@pytest.mark.anyio
async def test_fan_out_reads_each_collections_halves_as_its_mode_and_table_allow(
    tmp_path: Path,
    name: str,
    first: tuple[str, bool, bool, bool],
    rankings: set[str],
    answered: set[str],
) -> None:
    """Several collections answer their vector and BM25 halves apart, for the merge to rank each
    over all of them. `first` is (mode, has vectors, has its full-text index, shares its spans
    with the second) of the collection under test; the second is a plain hybrid one, so both
    rankings always exist. A span both found ranks in a half by the score whichever found it
    there, and its row keeps only the scores its own collection ran."""
    from haskie.search import retrieval

    mode, vectored, indexed, shared = first
    alpha = CollectionIndex(tmp_path / "alpha", "alpha", tmp_path, TINY if vectored else None)
    beta = CollectionIndex(tmp_path / "beta", "beta", tmp_path, TINY)
    alpha_rows = [
        _row(f"alpha lancedb row{i}", _vector(i) if vectored else None, i, i * 100) for i in (1, 2)
    ]
    await alpha.add_parts("a.md", "documents/a.md", "documents/a.md.md", _aparts([(0, alpha_rows)]))
    other = "a.md" if shared else "b.md"
    said = "alpha" if shared else "beta"  # a shared span is the same text at the same offsets
    beta_rows = [_row(f"{said} lancedb row{i}", _vector(i + 10), i, i * 100) for i in (1, 2)]
    await beta.add_parts(
        other, f"documents/{other}", f"documents/{other}.md", _aparts([(0, beta_rows)])
    )
    if indexed:
        await alpha.finish()
    await beta.finish()
    settings = SearchSettings(mode=SearchMode(mode))
    where = retrieval.Plan(
        settings=SearchSettings(),
        indexes=[(alpha, settings), (beta, SearchSettings())],
        vector=_vector(1),
        embedding=TINY,
    )

    pool = await retrieval.fan_out(where, "lancedb", 10)

    def columns(collection: str) -> set[str]:
        held = [row for (name, *_), (_, row) in pool.rows.items() if name == collection]
        return {one for row in held for one in ("_distance", "_score") if one in row}

    assert set(pool.rankings) == rankings, name
    assert columns("alpha") == answered, f"{name}: its rows keep the scores it ran"
    holders = {"alpha"} if shared else {"alpha", "beta"}
    assert {key[0] for key in pool.rows} == holders, f"{name}: a shared span counts once"
    if shared:
        texts = {key for key, _ in pool.rankings["text"]}
        assert texts == set(pool.rows), f"{name}: its spans rank by beta's BM25 score"
    else:
        assert columns("beta") == {"_distance", "_score"}, name


@pytest.mark.parametrize(
    ("name", "ranked", "expected"),
    [
        ("nothing to merge", [], []),
        ("one ranking keeps its own order", [["a", "b"]], [("a", 1 / 61), ("b", 1 / 62)]),
        ("an empty ranking contributes nothing", [["a"], []], [("a", 1 / 61)]),
        (
            "a hit in both rankings beats a better hit in one",
            [["a", "b", "c"], ["b", "d"]],
            [("b", 1 / 62 + 1 / 61), ("a", 1 / 61), ("d", 1 / 62), ("c", 1 / 63)],
        ),
        (
            "a tie keeps the order of first appearance",
            [["x"], ["y"]],
            [("x", 1 / 61), ("y", 1 / 61)],
        ),
    ],
)
def test_rrf_merge_orders_by_rank_and_sums_duplicates(
    name: str, ranked: list[list[str]], expected: list[tuple[str, float]]
) -> None:
    from haskie.search import retrieval

    merged = retrieval.rrf_merge(ranked, k=60)

    assert [item for item, _ in merged] == [item for item, _ in expected], name
    assert [score for _, score in merged] == pytest.approx([s for _, s in expected]), name


def _half(collection: str, score: str, found: list[tuple[str, int, float]]) -> list[tuple]:
    """What one half of one collection returned: (index, row) pairs, each row of document `doc`
    chunk `seq` at its own span, scored `value` in the column `score` (`_distance` or
    `_score`)."""
    index = CollectionIndex(Path("/nowhere") / collection, collection, Path("/nowhere"), None)
    return [
        (
            index,
            {"document_id": doc, "seq": seq, "char_start": seq * 10, "char_end": seq * 10 + 9}
            | {score: value},
        )
        for doc, seq, value in found
    ]


@pytest.mark.parametrize(
    ("name", "fusion", "pairs", "expected"),
    [
        (
            "the closer collection takes the top, not one slot each",
            Fusion.RRF,
            [
                *_half("small", "_distance", [("asyncio.md", 1, 0.9), ("asyncio.md", 2, 1.0)]),
                *_half("big", "_distance", [("raft.md", 1, 0.2), ("raft.md", 2, 0.3)]),
            ],
            [
                ("big", "raft.md", 1),
                ("big", "raft.md", 2),
                ("small", "asyncio.md", 1),
                ("small", "asyncio.md", 2),
            ],
        ),
        (
            "a chunk both halves found beats one only the closer half found",
            Fusion.RRF,
            [
                *_half("a", "_distance", [("d.md", 1, 0.1), ("d.md", 2, 0.2)]),
                *_half("b", "_score", [("e.md", 5, 3.0)]),
                *_half("a", "_score", [("d.md", 2, 2.0)]),
            ],
            [("a", "d.md", 2), ("a", "d.md", 1), ("b", "e.md", 5)],
        ),
        (
            "a span two collections return is one row, credited to the first",
            Fusion.RRF,
            [
                *_half("a", "_distance", [("d.md", 1, 0.5)]),
                *_half("b", "_distance", [("d.md", 1, 0.5), ("d.md", 2, 0.6)]),
            ],
            [("a", "d.md", 1), ("b", "d.md", 2)],
        ),
        (
            "one half alone keeps its order and its own scores",
            Fusion.RRF,
            [*_half("a", "_score", [("d.md", 1, 1.0)]), *_half("b", "_score", [("e.md", 1, 4.0)])],
            [("b", "e.md", 1), ("a", "d.md", 1)],
        ),
        (
            "linear: the weighted sum over both halves, scaled over all collections",
            Fusion.LINEAR,
            [
                *_half("a", "_distance", [("d.md", 1, 0.1), ("d.md", 2, 0.9)]),
                *_half("b", "_score", [("d.md", 2, 1.0), ("e.md", 1, 5.0)]),
            ],
            [("a", "d.md", 1), ("b", "e.md", 1), ("a", "d.md", 2)],
        ),
        ("nothing found", Fusion.RRF, [], []),
    ],
)
def test_merge_ranks_every_collection_as_one_table(
    name: str, fusion: Fusion, pairs: list[tuple], expected: list[tuple[str, str, int]]
) -> None:
    """Several collections are ranked per retriever over all of them, then fused: fusing one
    ranking per collection gave each collection's first chunk the same score, however far it
    was from the query."""
    from haskie.search import retrieval

    pool = retrieval.merge(retrieval._ranked_halves(pairs), SearchSettings(fusion=fusion), 10)

    assert [key for key, _ in pool.ranked] == expected, name


@pytest.mark.parametrize(
    ("name", "near", "words", "share", "expected"),
    [
        ("nothing to merge", [], [], 0.7, []),
        (
            "each half scaled to 0-1, a distance turned into a closeness",
            [("a", 0.2), ("b", 0.6)],
            [("b", 8.0), ("a", 4.0)],
            0.5,
            [("a", 0.5), ("b", 0.5)],
        ),
        (
            "a half that missed an item counts 0",
            [("a", 0.2), ("c", 0.4)],
            [("b", 3.0), ("c", 1.0)],
            0.7,
            [("a", 0.7), ("b", 0.3), ("c", 0.0)],
        ),
        (
            "equal distances all count as the closest, as LanceDB scales them",
            [("a", 0.4), ("b", 0.4)],
            [],
            1.0,
            [("a", 1.0), ("b", 1.0)],
        ),
    ],
)
def test_linear_merge_weighs_the_two_halves(
    name: str,
    near: list[tuple[str, float]],
    words: list[tuple[str, float]],
    share: float,
    expected: list[tuple[str, float]],
) -> None:
    from haskie.search import retrieval

    merged = retrieval.linear_merge(near, words, share)

    assert [item for item, _ in merged] == [item for item, _ in expected], name
    assert [score for _, score in merged] == pytest.approx([s for _, s in expected]), name


CHUNK_CHARS = 100  # how long each chunk `_retrieved` writes is


def _retrieved(rows: list[tuple]) -> tuple[CollectionIndex, list[dict]]:
    """What one collection returned, for `merge`: it only ever reads `index.collection` and the
    rows' identity columns. A row is (collection, document, seq, score), chunk `seq` spanning the
    `seq`-th `CHUNK_CHARS` of the document, or (…, char_start) to place it elsewhere. A `score` of
    None writes no `_score` at all, as an unscored row has."""
    name = rows[0][0] if rows else "empty"
    index = CollectionIndex(Path("/nowhere") / name, name, Path("/nowhere"), None)
    placed = [(*row, (row[2] - 1) * CHUNK_CHARS)[:5] for row in rows]
    return index, [
        {
            "document_id": document,
            "seq": seq,
            "char_start": start,
            "char_end": start + CHUNK_CHARS,
        }
        | ({"_score": score} if score is not None else {})
        for _collection, document, seq, score, start in placed
    ]


def _identity(pairs: list[tuple]) -> list[tuple[str, str, int]]:
    return [(index.collection, row["document_id"], row["seq"]) for index, row in pairs]


@pytest.mark.parametrize(
    ("name", "per_collection", "expected"),
    [
        ("nothing to merge", [], []),
        (
            "a collection that matched nothing contributes nothing",
            [[], [("a", "d.md", 1, 1.0)]],
            [("a", "d.md", 1)],
        ),
        (
            "the better score wins, whichever collection it came from",
            [[("b", "x.md", 1, 9.0)], [("a", "y.md", 1, 1.0)]],
            [("b", "x.md", 1), ("a", "y.md", 1)],
        ),
        (
            "the same passage from two collections is kept once, the better copy",
            [[("b", "d.md", 1, 1.0)], [("a", "d.md", 1, 9.0)]],
            [("a", "d.md", 1)],
        ),
        (
            "an equal score falls back to the collection name, and still counts once",
            [[("b", "d.md", 1, 1.0)], [("a", "d.md", 1, 1.0)]],
            [("a", "d.md", 1)],
        ),
        (
            "one document chunked two ways: the same seq over other text is another chunk",
            [[("b", "d.md", 1, 9.0, 0)], [("a", "d.md", 1, 1.0, 50)]],
            [("b", "d.md", 1), ("a", "d.md", 1)],
        ),
        (
            "one document chunked two ways: the same span under another seq is one chunk",
            [[("b", "d.md", 2, 9.0, 0)], [("a", "d.md", 1, 1.0, 0)]],
            [("b", "d.md", 2)],
        ),
        (
            "inside one collection: document, then where the chunk starts",
            [
                [
                    ("a", "z.md", 1, 1.0),
                    ("a", "a.md", 9, 1.0),
                    ("a", "a.md", 6, 1.0),
                    ("a", "a.md", 2, 1.0),
                ]
            ],
            [("a", "a.md", 2), ("a", "a.md", 6), ("a", "a.md", 9), ("a", "z.md", 1)],
        ),
        (
            "a row an older index wrote without a score sorts last",
            [[("a", "d.md", 2, None)], [("a", "d.md", 1, 0.5)]],
            [("a", "d.md", 1), ("a", "d.md", 2)],
        ),
    ],
)
def test_text_merge_orders_by_score_then_identity(
    name: str, per_collection: list[list[tuple]], expected: list[tuple[str, str, int]]
) -> None:
    # aliased: `text` is a chunk field and a parameter name all over this module
    from haskie.search import text as fulltext

    merged = fulltext.merge([_retrieved(rows) for rows in per_collection])

    assert _identity(merged) == expected, name


def _digest() -> str:
    """The query identity `TEXT_QUERY` hashes to, read at collection time by the table below."""
    from haskie.search import text as fulltext

    return fulltext.query_hash(*TEXT_QUERY)


def _wire_cursor(**overrides) -> str:
    """A cursor built straight on the wire format, for the fields `make_cursor` never varies."""
    from haskie.search import text as fulltext

    payload: dict = {"k": [_digest(), 10], "s": fulltext.SORT, "o": fulltext.ORDER, "v": 1}
    payload.update(overrides)
    return base64.urlsafe_b64encode(msgspec.json.encode(payload)).decode().rstrip("=")


@pytest.mark.parametrize(
    ("name", "rejected", "detail"),
    [
        ("another query", ("other", ["alpha", "beta"], 25, 10), "another query"),
        ("another set of collections", ("lancedb", ["alpha"], 25, 10), "another query"),
        ("another page size", ("lancedb", ["alpha", "beta"], 50, 10), "another query"),
        ("a listing's sort", {"s": "name"}, "does not match sort/order"),
        ("the other direction", {"o": "asc"}, "does not match sort/order"),
        ("a version this build does not write", {"v": 2}, "does not match sort/order"),
        ("a keyset of another width", {"k": [_digest()]}, "does not match sort/order"),
        ("a negative offset", {"k": [_digest(), -1]}, "invalid cursor"),
        ("an offset that is not a number", {"k": [_digest(), "10"]}, "invalid cursor"),
        ("an offset that is a boolean", {"k": [_digest(), True]}, "invalid cursor"),
        ("not base64 at all", "not a cursor!!", "invalid cursor"),
        ("base64 that decodes to no cursor", "e30", "invalid cursor"),
    ],
)
def test_text_cursor_roundtrip_and_rejects_other_query(
    name: str, rejected: tuple | dict | str, detail: str
) -> None:
    from haskie.search import text as fulltext

    q, collections, page_size = TEXT_QUERY
    assert fulltext.parse_cursor(None, q, collections, page_size) == 0, "no cursor, first page"
    issued = fulltext.make_cursor(q, collections, page_size, 50)
    assert fulltext.parse_cursor(issued, q, collections, page_size) == 50, "round trip"
    assert fulltext.make_cursor(q, ["beta", "alpha"], page_size, 50) == issued, "order-free"

    if isinstance(rejected, tuple):
        cursor = fulltext.make_cursor(*rejected)  # a cursor this module issued, another page
    elif isinstance(rejected, dict):
        cursor = _wire_cursor(**rejected)
    else:
        cursor = rejected

    with pytest.raises(InvalidInput) as raised:
        fulltext.parse_cursor(cursor, q, collections, page_size)
    assert detail in str(raised.value), name


@pytest.mark.parametrize(
    ("name", "collections", "expected"),
    [
        ("no filter at all", None, None),
        ("an empty filter is not a filter", "", None),
        ("separators alone", " , ,", None),
        ("one name", "alpha", ["alpha"]),
        ("several names, trimmed", " alpha , beta ", ["alpha", "beta"]),
        ("a trailing separator", "alpha,", ["alpha"]),
    ],
)
def test_text_split_collections(name: str, collections: str | None, expected) -> None:
    from haskie.search import text as fulltext

    assert fulltext.split_collections(collections) == expected, name


# --- storage -----------------------------------------------------------------------


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {name for (name,) in conn.execute("select name from sqlite_master where type='table'")}


@pytest.mark.parametrize(
    ("name", "stamped", "outcome"),
    [
        ("a fresh file gets the schema", 0, "created"),
        ("a home this build wrote is opened as it is", db.SCHEMA_VERSION, "kept"),
        ("a home from an older build is refused", db.SCHEMA_VERSION - 1, "refused"),
        ("a home from a newer build is refused too", db.SCHEMA_VERSION + 1, "refused"),
    ],
)
def test_migrate_creates_the_schema_once_and_refuses_every_other_home(
    tmp_path: Path, name: str, stamped: int, outcome: str
) -> None:
    """One schema snapshot and no upgrade path, so a home is created, opened, or refused whole.

    A refused one keeps its rows: the user destroys it rather than losing them to a silent drop.
    """
    conn = sqlite3.connect(str(tmp_path / "m.db"))
    if outcome == "kept":
        db.migrate(conn)
        conn.executescript("drop table sessions;")  # only a second run of the script rebuilds it
    elif stamped:
        conn.executescript("create table libraries (name text primary key);")
        conn.execute("insert into libraries (name) values ('notes')")
        conn.execute(f"pragma user_version = {stamped}")
    conn.commit()

    if outcome == "refused":
        with pytest.raises(HaskieError) as raised:
            db.migrate(conn)
        assert str(raised.value) == db.INCOMPATIBLE_HOME_MESSAGE, name
        assert "haskie destroy" in db.INCOMPATIBLE_HOME_MESSAGE, "the message names the way out"
        assert conn.execute("pragma user_version").fetchone() == (stamped,), f"{name}: untouched"
        assert _tables(conn) == {"libraries"}, f"{name}: nothing created and nothing dropped"
        assert conn.execute("select count(*) from libraries").fetchone() == (1,), f"{name}: rows"
    else:
        db.migrate(conn)
        assert conn.execute("pragma user_version").fetchone() == (db.SCHEMA_VERSION,), name
        assert conn.execute("pragma journal_mode").fetchone() == ("wal",), f"{name}: WAL, for good"
        assert ("sessions" in _tables(conn)) is (outcome == "created"), (
            f"{name}: the script runs on a fresh file and never again"
        )
    conn.close()


# The schema `tables.py` generates at this `SCHEMA_VERSION`: a SHA-256 of its DDL statements,
# sorted, because a table's indexes are a set and come out in no fixed order.
SCHEMA_PIN = (34, "10ceb8cab273eccf29630a23e7510f15bc99c76fa1ea4482890a616dfe6f90c7")


def test_a_table_change_comes_with_a_new_schema_version() -> None:
    """A table change that keeps the version would open an older home as it is, and fail
    mid-query on the first column it lacks, instead of refusing the home at startup."""
    statements = sorted(db.schema_ddl().split(";\n"))
    digest = hashlib.sha256(";\n".join(statements).encode()).hexdigest()

    assert (db.SCHEMA_VERSION, digest) == SCHEMA_PIN, (
        "the schema changed: bump db.SCHEMA_VERSION, then pin both here"
    )


def test_a_fresh_home_has_exactly_the_tables_indexes_and_columns_of_the_metadata(
    tmp_path: Path,
) -> None:
    """The DDL is generated from `tables.metadata`, so the file and the declarations agree."""
    conn = sqlite3.connect(str(tmp_path / "fresh.db"))
    db.migrate(conn)
    master = conn.execute("select type, name from sqlite_master where name not like 'sqlite_%'")
    created = {(kind, name) for kind, name in master}
    declared_tables = tables.metadata.sorted_tables
    index_names = {str(index.name) for table in declared_tables for index in table.indexes}
    declared = {("table", table.name) for table in declared_tables} | {
        ("index", name) for name in index_names
    }
    assert created == declared, "every table and index, and nothing else"
    assert all(name.startswith("idx_") for name in index_names), "every index has the prefix"
    for table in declared_tables:
        columns = [row[1] for row in conn.execute(f"pragma table_info({table.name})")]
        assert columns == [column.name for column in table.columns], table.name
    conn.close()


@pytest.mark.anyio
async def test_connect_rolls_back_a_failed_unit_of_work() -> None:
    with pytest.raises(IntegrityError):
        async with db.connect() as conn:
            await conn.execute(insert(tables.collections).values(name="half"))
            await conn.execute(
                insert(tables.collection_documents).values(collection="ghost", document_id="a.md")
            )
    assert await Collection.names() == [], "the first insert of the failed block is gone too"


# --- embeddings and logging --------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "available", "accelerator", "expected"),
    [
        (
            "cuda first; auto never takes coreml",
            ["CPUExecutionProvider", "CoreMLExecutionProvider", "CUDAExecutionProvider"],
            "auto",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            "apple silicon on auto: webgpu, never coreml",
            [
                "CoreMLExecutionProvider",
                "WebGpuExecutionProvider",
                "AzureExecutionProvider",
                "CPUExecutionProvider",
            ],
            "auto",
            ["WebGpuExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            "apple silicon on auto without the webgpu build: the cpu",
            ["CoreMLExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"],
            "auto",
            ["CPUExecutionProvider"],
        ),
        (
            "cuda before webgpu where both are",
            ["WebGpuExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "auto",
            ["CUDAExecutionProvider", "WebGpuExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            "coreml when asked for, then cpu",
            ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "coreml",
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            "coreml asked for where there is none: the cpu",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "coreml",
            ["CPUExecutionProvider"],
        ),
        ("cpu only", ["CPUExecutionProvider"], "auto", ["CPUExecutionProvider"]),
        (
            "forced cpu ignores gpu",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "cpu",
            ["CPUExecutionProvider"],
        ),
        (
            "cpu always appended",
            ["CUDAExecutionProvider"],
            "auto",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
    ],
)
def test_select_providers(
    name: str, available: list[str], accelerator, expected: list[str]
) -> None:
    assert embed.select_providers(available, accelerator) == expected, name


@pytest.mark.parametrize(
    ("name", "loads", "expected"),
    [
        ("nothing NVIDIA loads: the cpu, never a failing gpu", set(), ["CPUExecutionProvider"]),
        (
            "cuda without tensorrt: cuda, not a tensorrt that fails first",
            {"CUDAExecutionProvider"},
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            "both load: tensorrt first",
            {"TensorrtExecutionProvider", "CUDAExecutionProvider"},
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
    ],
)
def test_the_cuda_build_offers_only_the_nvidia_providers_that_load(
    name: str, loads: set[str], expected: list[str], monkeypatch
) -> None:
    """Linux installs ONNX Runtime's CUDA build, which lists TensorRT and CUDA on every machine.
    Only those whose library loads here are asked for, so the device the status reports is the
    one the models run on."""
    listed = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    stand_in = types.SimpleNamespace(get_available_providers=lambda: listed)
    monkeypatch.setattr(embed, "onnx_runtime", lambda: stand_in)
    monkeypatch.setattr(embed, "nvidia_loads", lambda provider: provider in loads)

    assert embed.providers() == expected, name


def test_an_nvidia_provider_whose_library_is_missing_does_not_load(tmp_path: Path, monkeypatch):
    """No driver or no library, as on any machine without an NVIDIA GPU: not runnable."""
    stand_in = types.SimpleNamespace(__file__=str(tmp_path / "onnxruntime" / "__init__.py"))
    monkeypatch.setattr(embed, "onnx_runtime", lambda: stand_in)
    embed.nvidia_loads.cache_clear()
    try:
        assert embed.nvidia_loads("CUDAExecutionProvider") is False
    finally:
        embed.nvidia_loads.cache_clear()


@pytest.mark.parametrize(
    ("name", "plugin", "expected"),
    [
        ("telemetry off, once per process", False, ["off"]),
        (
            "and the WebGPU plugin registered where it is installed",
            True,
            ["off", "register webgpu /lib/webgpu.dylib"],
        ),
    ],
)
def test_onnx_runtime_comes_with_its_telemetry_off_once(
    name: str, plugin: bool, expected: list[str], monkeypatch
) -> None:
    """Its telemetry thread crashed processes exiting mid-upload (`onnx_models.runtime`)."""
    import importlib.util

    calls: list[str] = []
    stand_in = types.SimpleNamespace(
        disable_telemetry_events=lambda: calls.append("off"),
        register_execution_provider_library=lambda key, path: calls.append(
            f"register {key} {path}"
        ),
    )
    webgpu = types.SimpleNamespace(get_library_path=lambda: "/lib/webgpu.dylib")
    monkeypatch.setitem(sys.modules, "onnxruntime", stand_in)
    monkeypatch.setitem(sys.modules, "onnxruntime_ep_webgpu", webgpu)
    found = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda module: (
            (object() if plugin else None) if module == "onnxruntime_ep_webgpu" else found(module)
        ),
    )
    onnx_models.runtime.cache_clear()
    try:
        assert embed.onnx_runtime() is stand_in
        assert embed.onnx_runtime() is stand_in
    finally:
        onnx_models.runtime.cache_clear()

    assert calls == expected, name


@pytest.mark.parametrize(
    ("name", "names", "expected"),
    [
        ("no coreml", ["CUDAExecutionProvider", "CPUExecutionProvider"], None),
        (
            "coreml gets the compiled-model cache",
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
            [("CoreMLExecutionProvider", {"ModelCacheDirectory": "/c"}), "CPUExecutionProvider"],
        ),
    ],
)
def test_with_options_attaches_the_coreml_cache_only(name, names, expected) -> None:
    got = embed.with_options(names, "/c")
    assert got == (names if expected is None else expected), name
    assert [embed.provider_name(p) for p in got] == names, name


CPU, CUDA = "CPUExecutionProvider", "CUDAExecutionProvider"
ON_COREML = [embed.COREML, CPU]
ON_WEBGPU = [onnx_models.WEBGPU, CPU]
RERANK, EMBED = embed._cross_encoder, embed._model
GTE, ETTIN, GRANITE, F2LLM = (
    "Alibaba-NLP/gte-reranker-modernbert-base",
    "cross-encoder/ettin-reranker-32m-v1",
    "ibm-granite/granite-embedding-97m-multilingual-r2",
    "onnx-community/F2LLM-v2-160M-ONNX",
)
UNMEASURED = "intfloat/e5-base-v2"  # CoreML was measured to run none: the rest stand in


@pytest.mark.parametrize(
    ("name", "build", "model", "accelerator", "available", "expected"),
    [
        ("auto: CUDA if installed", RERANK, GTE, "auto", [CUDA, CPU], [CUDA, CPU]),
        (
            "auto: WebGPU on Apple Silicon",
            RERANK,
            GTE,
            "auto",
            [*ON_COREML, onnx_models.WEBGPU],
            ON_WEBGPU,
        ),
        ("cpu: the CPU, CUDA or not", RERANK, GTE, "cpu", [CUDA, CPU], [CPU]),
        ("coreml: CoreML, with its cache", RERANK, GTE, "coreml", ON_COREML, ON_COREML),
        ("a headed cross-encoder the same", RERANK, ETTIN, "coreml", ON_COREML, ON_COREML),
        ("an embedder the same", EMBED, GRANITE, "coreml", ON_COREML, ON_COREML),
        ("a last-token embedder on WebGPU", EMBED, F2LLM, "auto", ON_WEBGPU, ON_WEBGPU),
        ("not measured on CoreML: without it", EMBED, UNMEASURED, "coreml", ON_COREML, [CPU]),
    ],
)
def test_every_onnx_model_runs_on_the_hardware_the_settings_choose(
    name: str,
    build: Callable[[str, Accelerator], object],
    model: str,
    accelerator: Accelerator,
    available: list[str],
    expected: list,
    monkeypatch,
) -> None:
    """Rerankers once ran on the CPU whatever the setting, on the claim that they score in
    milliseconds: measured, 50 candidates take 0.6 s (MiniLM-L6) to 3.3 s (bge-reranker-base)."""
    from haskie.indexing import hardware, onnx_models

    seen: list[list] = []
    built = lambda model, providers: seen.append(providers)  # noqa: E731
    stand_in = types.SimpleNamespace(get_available_providers=lambda: available)
    monkeypatch.setattr(embed, "onnx_runtime", lambda: stand_in)
    monkeypatch.setattr(embed, "nvidia_loads", lambda _provider: True)  # CUDA runs here
    monkeypatch.setattr(onnx_models, "Embedder", built)
    monkeypatch.setattr(onnx_models, "CrossEncoder", built)
    monkeypatch.setattr(hardware, "COREML_RUNS", frozenset({GTE, ETTIN, GRANITE, F2LLM}))
    embed._build_cross_encoder.cache_clear()
    embed._build_model.cache_clear()
    try:
        build(model, accelerator)
    finally:
        embed._build_cross_encoder.cache_clear()
        embed._build_model.cache_clear()

    # CoreML comes with its compiled-model cache, in this test's home
    cache = {"ModelCacheDirectory": str(home.MODEL_CACHE)}
    assert seen == [[(one, cache) if one == embed.COREML else one for one in expected]], name


@pytest.mark.parametrize(
    ("name", "model", "expected"),
    [
        ("a model whole: its vector as it came", COMPACT, [3.0, 4.0, 0.0, 0.0]),
        (
            "a Matryoshka cut: the first values, normalized again",
            EmbeddingModel("test/cut", 2, matryoshka=True),
            [0.6, 0.8],
        ),
    ],
)
def test_a_vector_is_stored_whole_or_cut_as_the_profile_says(
    name: str, model: EmbeddingModel, expected: list[float]
) -> None:
    vector = np.array([3.0, 4.0, 0.0, 0.0])

    assert embed._cut(model, vector).tolist() == pytest.approx(expected), name


@pytest.mark.parametrize(
    ("name", "texts"),
    [
        ("one text", ["a job retries"]),
        ("texts out of length order run shortest first", ["a long text here", "a", "mid one"]),
        ("ties keep their order", ["bb", "aa", "c"]),
    ],
)
def test_texts_run_shortest_first_and_answer_in_the_order_they_came(
    name: str, texts: list[str], monkeypatch
) -> None:
    """A batch pads to its longest row, so like lengths batch together; each text still gets
    its own vector and score."""
    ran: list[list[str]] = []

    class Model:
        def embed(self, batch: list[str]) -> list[np.ndarray]:
            ran.append(list(batch))
            return [np.array([float(len(text)), 1.0]) for text in batch]

        def rerank(self, query: str, batch: list[str]) -> list[float]:
            ran.append(list(batch))
            return [float(len(text)) for text in batch]

    monkeypatch.setattr(embed, "_model", lambda name, accelerator: Model())
    monkeypatch.setattr(embed, "_cross_encoder", lambda name, accelerator: Model())

    vectors = embed.embed_texts(EmbeddingModel("test/tiny", 2), texts)
    scores = embed.rerank_scores("test/reranker", Accelerator.CPU, "q", texts)

    assert ran == [sorted(texts, key=len)] * 2, name
    assert [vector[0] for vector in vectors] == scores == [float(len(t)) for t in texts], name


def test_embedding_helpers_short_circuit_on_empty_input() -> None:
    """No text means no model, so neither call may download anything."""
    assert embed.embed_texts(COMPACT, []) == []
    assert (
        embed.rerank_scores("cross-encoder/ettin-reranker-32m-v1", Accelerator.AUTO, "q", []) == []
    )


@pytest.mark.parametrize(
    ("name", "log_format", "expected"),
    [
        ("json is the default", "json", structlog.processors.JSONRenderer),
        ("console for a terminal", "console", structlog.dev.ConsoleRenderer),
        ("anything else falls back to json", "yaml", structlog.processors.JSONRenderer),
    ],
)
def test_renderer_per_format(name: str, log_format: str, expected: type) -> None:
    assert isinstance(logs._renderer(log_format), expected), name


def test_attach_outside_an_audited_call_is_a_no_op() -> None:
    audit.attach(collection="notes")  # no record in progress
    assert not audit.path().exists(), "nothing written outside an audited call"


def test_the_audit_record_names_a_collection_and_a_document() -> None:
    """A document belongs to no collection, so a document-scoped action carries `doc` alone."""
    assert {"collection", "document"} <= audit.RECORD_FIELDS
    assert "library" not in audit.RECORD_FIELDS


@pytest.mark.parametrize(
    ("name", "vectors", "expected"),
    [
        (
            "every row a vector: rows of one float32 array",
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
        ),
        (
            "a row without one holds None, the others keep their own",
            [[1.0, 2.0], None, [5.0, 6.0]],
            [[1.0, 2.0], None, [5.0, 6.0]],
        ),
        ("no rows", [], []),
    ],
)
def test_a_read_holds_each_vector_as_an_array_row(name: str, vectors: list, expected: list) -> None:
    """A search compares vectors as arrays (`collapse.unit_rows`), so a read never builds the
    list of floats each row would carry."""
    column = pa.array(vectors, type=pa.list_(pa.float32(), 2))
    found = index_module._rows(pa.table({"seq": list(range(len(vectors))), "vector": column}))

    assert [row["seq"] for row in found] == list(range(len(vectors))), name
    got = [None if row["vector"] is None else row["vector"].tolist() for row in found]
    assert got == expected, name
    assert all(row["vector"] is None or row["vector"].dtype == np.float32 for row in found), name


def test_a_read_without_a_vector_column_is_plain_rows() -> None:
    assert index_module._rows(pa.table({"seq": [1, 2]})) == [{"seq": 1}, {"seq": 2}]
