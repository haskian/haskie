"""Pure checks: chunking, conversion, settings, document and collection IO, the index and the
pipeline, with HASKIE_HOME pointed at a temp dir (see conftest.py).

Nothing here launches DBOS: the pipeline stages are called straight through, in the order the
workflows call them. Every test that needs the durable runtime lives in `tests/test_workflows.py`;
the HTTP contract lives in `tests/test_api.py`.
"""

import base64
import io
import random
import sqlite3
import threading
import zipfile
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import aiosqlite
import anyio
import lancedb
import msgspec
import pyarrow as pa
import pytest
import structlog
from conftest import (
    MD,
    document_names,
    events,
    import_row,
    legacy_index,
    maintenance_state,
    text_pdf,
)

from haskie import (
    audit,
    chunk,
    convert,
    db,
    document,
    embed,
    embed_cache,
    home,
    logs,
    maintenance,
    pipeline,
)
from haskie.chunk import Chunk
from haskie.collection import Collection, DocumentCounts, Member
from haskie.document import Document
from haskie.errors import (
    Conflict,
    HaskieError,
    InvalidInput,
    NotFound,
    NotReady,
    PermanentError,
)
from haskie.index import (
    PLAIN_SCHEMA,
    CollectionIndex,
    Hit,
    IndexStats,
    Row,
    _fusion,
    _partitions,
    row_score,
)
from haskie.paging import PageRequest
from haskie.settings import (
    ChunkSettings,
    CollectionSettings,
    ConversionSettings,
    EmbeddingModel,
    PipelineSettings,
    SearchOverrides,
    SearchSettings,
    UserSettings,
    docs,
    init_user_settings,
    load_user_settings,
    load_user_settings_or_none,
    save_user_settings,
)

SMALL = ChunkSettings(chunk_size=40, chunk_overlap=0)

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
        rows = await conn.execute_fetchall(
            "select staging_id, filename, size from staging order by staging_id"
        )
    return [tuple(row) for row in rows]


async def attachable(name: str, content: bytes | str = MD, **options) -> Document:
    """`imported`, moved on to the status the pipeline ends at: `Collection.add` takes only an
    imported document, so a test that attaches one has to get it there first."""
    doc = await import_row(name, content, **options)
    await document.set_status(doc.name, "imported")
    return await document.get(doc.name)


# --- chunking and table of contents ------------------------------------------------


@pytest.mark.parametrize(
    ("name", "text", "settings", "expected"),
    [
        ("no headings -> one untitled chunk", "plain text", ChunkSettings(), [("", 1, 1, [])]),
        (
            "fits capacity -> one chunk, first heading",
            MD,
            ChunkSettings(),
            [("Title", 1, 11, [])],
        ),
        (
            "small capacity -> one chunk per section with lines and parents",
            MD,
            SMALL,
            [("Title", 1, 3, []), ("Alpha", 5, 7, ["Title"]), ("Beta", 9, 11, ["Title"])],
        ),
        (
            "oversize section -> split keeps heading",
            "# H\n\n" + "word " * 20,
            ChunkSettings(chunk_size=30, chunk_overlap=0),
            [("H", 1, 1, [])] + [("H", 3, 3, [])] * 4,
        ),
        (
            "text before first heading -> untitled",
            "pre\n# H\nbody",
            ChunkSettings(chunk_size=8, chunk_overlap=0),
            [("", 1, 1, []), ("H", 2, 3, [])],
        ),
        (
            "nested headings -> ancestry",
            "# A\n## B\n### C\nc\n## D\nd\n",
            ChunkSettings(chunk_size=6, chunk_overlap=0),
            [
                ("A", 1, 1, []),
                ("B", 2, 2, ["A"]),
                ("C", 3, 3, ["A", "B"]),
                ("C", 4, 4, ["A", "B"]),
                ("D", 5, 6, ["A"]),
            ],
        ),
        (
            "text splitter ignores structure",
            MD,
            ChunkSettings(chunker="text", chunk_size=1000, chunk_overlap=0),
            [("Title", 1, 11, [])],
        ),
        ("empty text -> no chunks", "", ChunkSettings(), []),
        (
            "heading inside first chunk -> used",
            "<!-- page 1 -->\n\n# H\nbody",
            ChunkSettings(),
            [("H", 1, 4, [])],
        ),
    ],
)
def test_chunk(name: str, text: str, settings: ChunkSettings, expected: list) -> None:
    chunks = chunk.split(text, settings)
    got = [(c.heading, c.line_start, c.line_end, c.parents) for c in chunks]
    assert got == expected, name
    assert all(text[c.char_start : c.char_end] == c.text for c in chunks), name


def test_chunk_pages_from_markers() -> None:
    text = "<!-- page 3 -->\n\n# A\nbody a\n\n<!-- page 4 -->\n\n# B\nbody b\n"
    small = [
        (c.heading, c.page_start, c.page_end)
        for c in chunk.split(text, ChunkSettings(chunk_size=30, chunk_overlap=0))
    ]
    assert small == [("A", 3, 3), ("B", 4, 4)]
    (whole,) = chunk.split(text, ChunkSettings())
    assert (whole.page_start, whole.page_end) == (3, 4), "chunk spanning pages reports the range"


def test_chunk_version_is_part_of_the_cache_key() -> None:
    """A change to the splitting logic has to retire the entries it would now produce
    differently, so the version travels with the settings into `embed_cache.Params`."""
    assert chunk.CHUNK_VERSION == embed_cache.CHUNK_VERSION


def test_chunk_rejects_overlap_ge_size() -> None:
    """The splitter itself rejects it; settings validation stops it one layer earlier."""
    bad = ChunkSettings(chunk_size=10, chunk_overlap=9)
    object.__setattr__(bad, "chunk_overlap", 10)  # past __post_init__, as a stored row could be
    with pytest.raises(ValueError, match="overlap"):
        chunk.split(MD, bad)


# --- settings ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "overrides", "user", "expected"),
    [
        ("no overrides -> user defaults", CollectionSettings(), UserSettings(), (1200, 150, "m")),
        (
            "override size only",
            CollectionSettings(chunk_size=990),
            UserSettings(),
            (990, 150, "m"),
        ),
        (
            "override chunker only",
            CollectionSettings(chunker="text"),
            UserSettings(),
            (1200, 150, "t"),
        ),
        (
            "override every field",
            CollectionSettings(chunker="text", chunk_size=500, chunk_overlap=50),
            UserSettings(),
            (500, 50, "t"),
        ),
        (
            "an unset field follows the user value",
            CollectionSettings(chunk_overlap=0),
            UserSettings(conversion=ConversionSettings(chunk_size=300, chunk_overlap=30)),
            (300, 0, "m"),
        ),
    ],
)
def test_collection_settings_resolve_into_chunk_settings(
    name: str, overrides: CollectionSettings, user: UserSettings, expected: tuple
) -> None:
    """A collection only overrides how the shared markdown is split: `parser`/`skip_ocr_pages`
    belong to the document, chosen once at import."""
    effective = overrides.resolve(user)

    assert isinstance(effective, ChunkSettings), name
    assert (effective.chunk_size, effective.chunk_overlap, effective.chunker[0]) == expected, name


def test_conversion_settings_carry_the_chunking_defaults() -> None:
    user = ConversionSettings(chunker="text", chunk_size=700, chunk_overlap=70, parser="plain")
    assert user.chunking == ChunkSettings(chunker="text", chunk_size=700, chunk_overlap=70)
    assert CollectionSettings().resolve(UserSettings(conversion=user)) == user.chunking


def test_search_overrides_resolve_per_field() -> None:
    user = SearchSettings(limit=5, fusion="rrf", vector_weight=0.7)
    merged = SearchOverrides(fusion="linear", bm25_weight=0.9).resolve(user)
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
        "pipeline.maintenance_docs", "pipeline.maintenance_idle_seconds", "pipeline.ann_min_rows",
        "search.nprobes", "search.refine_factor",
    }  # fmt: skip
    assert user_docs["pipeline.maintenance_docs"].title == "Maintenance after documents"
    assert user_docs["search.nprobes"].title == "Vector probes"
    assert user_docs["conversion.chunk_size"].title == "Chunk size (characters)"
    assert "characters" in user_docs["conversion.chunk_size"].description
    assert all(d.title and d.description for d in user_docs.values())
    # collection overrides and the chunk settings reuse the same definitions
    collection_docs = docs(CollectionSettings)
    chunking = {"chunker", "chunk_size", "chunk_overlap"}
    search = {key for key in user_docs if key.startswith("search.")}
    assert set(collection_docs) == chunking | search, "chunking and search, and nothing else"
    assert collection_docs["chunk_size"] == user_docs["conversion.chunk_size"]
    assert collection_docs["search.reranker"] == user_docs["search.reranker"]
    assert docs(ChunkSettings)["chunker"] == user_docs["conversion.chunker"]


def test_docs_rejects_a_non_struct() -> None:
    with pytest.raises(TypeError, match="needs a msgspec Struct"):
        docs(int)  # ty: ignore[invalid-argument-type]


def test_embedding_model_carries_accelerator() -> None:
    user = UserSettings(embedding="compact", pipeline=PipelineSettings(accelerator="cpu"))
    assert user.embedding_model is not None and user.embedding_model.accelerator == "cpu"
    assert UserSettings(embedding="none").embedding_model is None


@pytest.mark.anyio
async def test_user_settings_persist_in_db() -> None:
    assert await load_user_settings_or_none() is None, "not initialized yet"
    assert (await load_user_settings()).embedding == "none"
    await save_user_settings(UserSettings(embedding="compact"))
    assert await load_user_settings_or_none() is not None
    assert (await load_user_settings()).embedding == "compact"


@pytest.mark.anyio
async def test_init_user_settings_creates_the_row_once() -> None:
    assert await init_user_settings(UserSettings(embedding="compact")) is True
    assert await init_user_settings(UserSettings(embedding="quality")) is False, "second call loses"
    assert (await load_user_settings()).embedding == "compact", "the first write stands"


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
    preview = convert.build_preview(pdf, tmp_path / "prev", "anydoc")
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
        convert.to_markdown(path, "anydoc")


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

    info = convert.build_preview(path, out, "anydoc")

    assert info.kind == kind, name
    assert (out / "source").read_bytes().startswith(source_starts), name
    assert (out / "preview.md").exists(), name
    assert list(out.glob("*.tmp")) == [], "atomic writes leave no temp file"


def test_build_preview_of_a_pdf_cuts_to_the_first_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "long.pdf"
    pdf.write_bytes(text_pdf([f"page {i}" for i in range(convert.PREVIEW_PAGES + 3)]))

    info = convert.build_preview(pdf, tmp_path / "p", "anydoc")

    assert (info.kind, info.truncated, info.pages) == ("pdf", True, convert.PREVIEW_PAGES)
    assert "page 11" not in (tmp_path / "p" / "preview.md").read_text()


def test_build_preview_of_a_corrupt_pdf_raises_conversion_error(tmp_path: Path) -> None:
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(PermanentError, match="broken.pdf"):
        convert.build_preview(bad, tmp_path / "p", "anydoc")


def test_pdf_page_count_of_a_corrupt_file_raises_conversion_error(tmp_path: Path) -> None:
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(PermanentError, match="broken.pdf"):
        convert.pdf_page_count(bad)


# --- documents: staging and import -------------------------------------------------


@pytest.mark.parametrize(
    ("name", "filename", "rename_to", "expected"),
    [
        ("the file name, cleaned", "guide.md", None, "guide.md"),
        ("a path keeps only its last segment", "/tmp/deep/guide.md", None, "guide.md"),
        ("unsafe characters collapse into one dash", "a b  c.md", None, "a-b-c.md"),
        ("a rename with no suffix keeps the original's", "book.pdf", "My Book", "My-Book.pdf"),
        ("a rename with the same suffix is taken as is", "book.pdf", "atlas.pdf", "atlas.pdf"),
        ("a rename to another suffix keeps the original's", "book.pdf", "a.txt", "a.txt.pdf"),
        ("the suffix decides the parser, so case is ignored", "BOOK.PDF", None, "BOOK.PDF"),
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
        ("nothing left after cleaning", "***", InvalidInput, "invalid name"),
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
            parser="plain",
            skip_ocr_pages=False,
        ),
    )

    assert (doc.name, doc.suffix, doc.size) == ("My-Guide.md", ".md", len(MD.encode()))
    assert (doc.status, doc.error, doc.preview) == ("queued", None, None), "the pipeline starts it"
    assert (doc.parser, doc.skip_ocr_pages) == ("plain", False), "conversion is fixed at import"
    assert doc.description == "the guide"
    assert doc.original.read_text() == MD
    assert not document.staging_path(staged.staging_id).exists(), "moved, not copied"
    assert await _staging_rows() == [], "the staging row goes with the bytes"
    assert await document_names() == ["My-Guide.md"]


@pytest.mark.anyio
async def test_import_staged_without_a_name_keeps_the_uploaded_file_name() -> None:
    """The staging id carries only the suffix, so the uploaded name comes off the staging row."""
    staged = await document.stage("My Guide.md", MD.encode())

    doc = await document.import_staged(staged.staging_id)

    assert doc.name == "My-Guide.md"
    assert await _staging_rows() == [], "the row is consumed with the staged file"


@pytest.mark.anyio
async def test_import_staged_defaults_conversion_to_the_user_settings() -> None:
    await save_user_settings(
        UserSettings(conversion=ConversionSettings(parser="plain", skip_ocr_pages=False))
    )
    staged = await document.stage("g.md", MD.encode())

    doc = await document.import_staged(staged.staging_id)

    assert (doc.parser, doc.skip_ocr_pages) == ("plain", False)


@pytest.mark.anyio
async def test_import_staged_renames_and_refuses_a_name_already_taken() -> None:
    first = await document.stage("book.pdf", text_pdf(["one"]))
    second = await document.stage("other.pdf", text_pdf(["two"]))

    doc = await document.import_staged(
        first.staging_id, document.ImportOptions(name="Atlas of Maps")
    )

    assert doc.name == "Atlas-of-Maps.pdf", "the rename names the document, not the parser"
    with pytest.raises(Conflict, match="document already exists"):
        await document.import_staged(
            second.staging_id, document.ImportOptions(name="Atlas of Maps")
        )
    assert document.staging_path(second.staging_id).exists(), "the refused upload is still staged"
    assert [row[0] for row in await _staging_rows()] == [second.staging_id], "and so is its row"
    assert await document_names() == ["Atlas-of-Maps.pdf"]


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
            "update staging set created_at = 0 where staging_id = ?", (old.staging_id,)
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

    assert (await document.get("a.md")).size == len(MD)
    with pytest.raises(NotFound, match="document not found"):
        await document.get("ghost.md")


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

    await document.set_status("a.md", "imported")
    done = await document.get("a.md")
    assert done.updated_at > first.updated_at, "a lifecycle step is a change"
    assert (done.status, done.error) == ("imported", None)
    assert done.created_at == first.created_at, "the import moment never moves"

    await document.set_status("a.md", "error", "boom")
    failed = await document.get("a.md")
    assert (failed.status, failed.error) == ("error", "boom"), "the reason is stored with it"

    await document.ensure_preview("a.md")
    previewed = await document.get("a.md")
    assert previewed.preview is not None, "the preview was built"
    assert previewed.updated_at == failed.updated_at, "filling in the preview is not a change"


@pytest.mark.anyio
async def test_describe_replaces_the_description_and_reads_in_batches() -> None:
    await import_row("a.md")
    await import_row("b.md")

    described = await document.describe("a.md", "the alpha guide")

    assert described.description == "the alpha guide"
    assert await document.describe_of({"a.md", "b.md"}) == {"a.md": "the alpha guide"}, (
        "a document without one is absent"
    )
    assert await document.describe_of(set()) == {}
    assert (await document.describe("a.md", "")).description == "", "empty clears it"
    with pytest.raises(NotFound, match="document not found"):
        await document.describe("ghost.md", "x")


@pytest.mark.anyio
async def test_document_page_sorts_filters_and_resumes_by_keyset() -> None:
    sizes = {"a.md": 30, "b.md": 10, "c.md": 20}
    for name, size in sizes.items():
        await import_row(name, "x" * size)
    await document.set_status("a.md", "imported")
    await document.set_status("b.md", "error", "boom")

    by_name = await document.page(PageRequest(page_size=2))
    assert [d.name for d in by_name.items] == ["a.md", "b.md"]
    assert (by_name.total, by_name.next_cursor is None) == (3, False)
    resumed = await document.page(PageRequest(cursor=by_name.next_cursor, page_size=2))
    assert [d.name for d in resumed.items] == ["c.md"], "the keyset resumes past the last row"
    assert resumed.next_cursor is None, "the last page says so"

    by_size = await document.page(PageRequest(sort="size", order="desc"))
    assert [d.name for d in by_size.items] == ["a.md", "c.md", "b.md"]

    filtered = await document.page(PageRequest(), status="imported")
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

    _, first = await document.ensure_preview(doc.name)
    _, second = await document.ensure_preview(doc.name)

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
        row, preview = await document.ensure_preview(doc.name)
        results.append((row, preview.kind))

    async def second_reader() -> None:
        await document.get(doc.name)  # the row is readable while the first reader holds the lock
        reader_ready.set()
        row, preview = await document.ensure_preview(doc.name)
        results.append((row, preview.kind))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(convert, "build_preview", gated)
        async with anyio.create_task_group() as readers:
            readers.start_soon(first_reader)
            await anyio.to_thread.run_sync(building.wait)  # the first build is under way
            readers.start_soon(second_reader)
            await reader_ready.wait()
            lock = document._preview_locks[doc.name]
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

    await document.ensure_preview(doc.name)
    assert document._preview_locks == {}, "the build is committed; the next reader needs no lock"

    await document.ensure_preview(doc.name)
    assert document._preview_locks == {}, "and the early return takes none at all"

    other = await import_row("h.md")
    monkeypatch.setattr(convert, "build_preview", _refuse_to_build)
    with pytest.raises(PermanentError, match="no preview today"):
        await document.ensure_preview(other.name)

    assert document._preview_locks == {}, "a failed build leaves no lock behind either"


def _refuse_to_build(*args, **kwargs):
    raise PermanentError("no preview today")


@pytest.mark.parametrize(("name", "workers"), [("one at a time", 1), ("two at a time", 2)])
@pytest.mark.anyio
async def test_ensure_preview_bounds_concurrent_builds(
    monkeypatch: pytest.MonkeyPatch, name: str, workers: int
) -> None:
    """Four readers open four different documents at once; only `preview_workers` parses run.

    The stripe lock is per document, so nothing but the semaphore holds these four apart.
    """
    from haskie import cpu

    # a build holds a CPU slot too, so the budget must not be the ceiling under test here
    monkeypatch.setattr(cpu, "_cpu_slots", cpu.ResizableSemaphore(threading.BoundedSemaphore, 4))
    names = [(await import_row(f"doc-{i}.md")).name for i in range(4)]
    entered, release, counted = threading.Semaphore(0), threading.Event(), threading.Lock()
    live, peak, builds = 0, 0, []
    real = convert.build_preview

    def gated(*args, **kwargs):
        nonlocal live, peak
        with counted:
            live += 1
            peak = max(peak, live)
            builds.append(args[0])
        entered.release()
        assert release.wait(timeout=30)
        try:
            return real(*args, **kwargs)
        finally:
            with counted:
                live -= 1

    document.configure_preview_slots(workers)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(convert, "build_preview", gated)
            async with anyio.create_task_group() as readers:
                for doc in names:
                    readers.start_soon(document.ensure_preview, doc)
                for _ in range(workers):  # every slot of the pool is now inside a build
                    await anyio.to_thread.run_sync(entered.acquire)
                assert document._preview_slots.current.value == 0, f"{name}: no slot left"
                release.set()
    finally:
        document.configure_preview_slots(PipelineSettings().preview_workers)

    assert len(builds) == 4, f"{name}: every document was built, once"
    assert peak == workers, f"{name}: never more parses at once than the pool admits"


@pytest.mark.anyio
async def test_ensure_preview_returns_not_ready_when_the_queue_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader that waited out `PREVIEW_WAIT_SECONDS` is told to retry (503) rather than holding
    its request open until the burst clears."""
    doc = await import_row("g.md")
    monkeypatch.setattr(document, "PREVIEW_WAIT_SECONDS", 0.05)
    from haskie import cpu

    slots = cpu.ResizableSemaphore(anyio.Semaphore, 1)
    monkeypatch.setattr(document, "_preview_slots", slots)
    await slots.current.acquire()  # the test holds the only slot, so every reader waits it out
    try:
        with pytest.raises(NotReady, match="preview queue is full"):
            await document.ensure_preview(doc.name)
    finally:
        slots.current.release()

    assert (await document.get(doc.name)).preview is None, "nothing built, nothing stored"


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

    await (await Collection.get("misc")).delete()

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
    assert await collection.settings() == CollectionSettings(), "no overrides yet"

    await collection.set_settings(CollectionSettings(chunker="text"))
    assert (await (await Collection.get("stored")).settings()).chunker == "text"

    async with db.connect() as conn:  # a write the setter never saw
        await conn.execute(
            "update collections set settings = '{\"chunk_size\": 42}' where name = ?", ("stored",)
        )
    assert (await collection.settings()).chunk_size == 42, "the row owns the value"

    await collection.delete()
    await Collection.create("stored")
    assert await (await Collection.get("stored")).settings() == CollectionSettings(), "clean"
    assert await Collection("ghost").settings() == CollectionSettings(), "and one with no row"


@pytest.mark.anyio
async def test_chunk_settings_of_a_collection_resolve_against_the_user_settings() -> None:
    await save_user_settings(
        UserSettings(conversion=ConversionSettings(chunk_size=800, chunk_overlap=80))
    )
    collection = await Collection.create("chunky")
    await collection.set_settings(CollectionSettings(chunker="text"))

    assert await collection.chunk_settings() == ChunkSettings("text", 800, 80)
    assert (await collection.search_settings()).limit == SearchSettings().limit


@pytest.mark.anyio
async def test_load_settings_reads_every_collection_it_was_asked_for_in_one_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cross-collection search resolves the settings of its whole selection at once, so the cost
    is one SELECT rather than one per collection. A name without a row stays out of the answer."""
    for name in ("alpha", "beta"):
        await Collection.create(name)
    await Collection("alpha").set_settings(CollectionSettings(chunker="text"))
    assert await Collection.load_settings([]) == {}, "nothing asked for, nothing read"
    statements: list[str] = []
    real_execute = aiosqlite.Connection.execute

    async def counted(self, sql, parameters=None):
        statements.append(sql)
        return await real_execute(self, sql, parameters)

    monkeypatch.setattr(aiosqlite.Connection, "execute", counted)

    found = await Collection.load_settings(["alpha", "beta", "ghost", "alpha"])

    assert set(found) == {"alpha", "beta"}, "a name with no row is absent from the result"
    assert (found["alpha"].chunker, found["beta"]) == ("text", CollectionSettings())
    selects = [sql for sql in statements if sql.lstrip().lower().startswith("select")]
    assert len(selects) == 1, "one query for every name, duplicates included"


@pytest.mark.anyio
async def test_reranker_overrides_lists_every_model_a_collection_chose() -> None:
    """The model downloads have to cover the overrides too, so they are read in one query."""
    from haskie.settings import RERANKER_MODELS

    for name in ("a", "b", "c"):
        await Collection.create(name)
    await Collection("b").set_settings(
        CollectionSettings(search=SearchOverrides(reranker_model=RERANKER_MODELS[1]))
    )
    await Collection("c").set_settings(
        CollectionSettings(search=SearchOverrides(reranker_model=RERANKER_MODELS[1]))
    )

    assert await Collection.reranker_overrides() == [RERANKER_MODELS[1]], "no duplicates"


@pytest.mark.anyio
async def test_collection_page_lists_summaries_with_their_counts() -> None:
    for name in ("alpha", "beta", "gamma"):
        await Collection.create(name, description=f"{name} notes")
    await attachable("a.md")
    await Collection("beta").add("a.md")
    await Collection("beta").set_member_status("a.md", "indexed")

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

    await collection.add(doc.name)
    first = await collection.member(doc.name)
    await collection.add(doc.name)
    again = await collection.member(doc.name)

    assert isinstance(first, Member) and first.status == "pending", "indexing moves it along"
    assert first.document.name == doc.name and first.document.size == doc.size
    assert (again.status, again.added_at) == (first.status, first.added_at), "a no-op re-attach"
    assert await collection.member_names() == ["a.md"]

    await collection.set_member_status(doc.name, "error", "boom")
    failed = await collection.member(doc.name)
    assert (failed.status, failed.error) == ("error", "boom")
    assert failed.updated_at >= failed.added_at

    with pytest.raises(NotFound, match="document not found"):
        await collection.add("ghost.md")
    with pytest.raises(NotFound, match="document not in collection notes"):
        await collection.member("ghost.md")


@pytest.mark.parametrize("status", ["queued", "converting", "embedding", "error", "deleting"])
@pytest.mark.anyio
async def test_add_refuses_a_document_that_is_not_imported(status: str) -> None:
    """The invariant lives in `add`: one still importing has no markdown to chunk yet, and one
    being deleted must not gain a membership the delete's snapshot missed."""
    collection = await Collection.create("notes")
    doc = await import_row("a.md")
    await document.set_status(doc.name, status)  # ty: ignore

    with pytest.raises(Conflict, match=f"document is {status}; only an imported document"):
        await collection.add(doc.name)

    assert await collection.member_names() == [], "nothing attached"


@pytest.mark.anyio
async def test_member_names_walk_one_page_at_a_time() -> None:
    collection = await Collection.create("notes")
    for name in ("a.md", "b.md", "c.md"):
        await collection.add((await attachable(name)).name)

    assert await collection.member_names() == ["a.md", "b.md", "c.md"]
    assert await collection.member_names(limit=2) == ["a.md", "b.md"]
    assert await collection.member_names(after="b.md") == ["c.md"]


@pytest.mark.anyio
async def test_member_counts_group_by_status() -> None:
    collection = await Collection.create("counts")
    for name in ("a.md", "b.md", "c.md", "d.md"):
        await collection.add((await attachable(name)).name)
    await collection.set_member_status("a.md", "indexed")
    await collection.set_member_status("b.md", "indexing")
    await collection.set_member_status("c.md", "error", "boom")

    counts = await collection.counts()

    assert counts.total == 4
    assert counts.indexed == 1
    assert counts.active == 2, "pending and indexing are both in flight"
    assert counts.error == 1
    assert counts.by_status == {"indexed": 1, "indexing": 1, "error": 1, "pending": 1}
    empty = await Collection.create("empty")
    assert await empty.counts() == DocumentCounts(), "a collection without members"


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
        await collection.add((await attachable(doc, "x" * size)).name)
    for doc, status in (("a.md", "indexed"), ("b.md", "pending"), ("c.md", "error")):
        await collection.set_member_status(doc, status)

    page = await collection.members_page(PageRequest(sort=sort, order=order))  # ty: ignore

    assert [member.document.name for member in page.items] == expected, name
    assert page.total == 3, name


@pytest.mark.anyio
async def test_members_page_filters_by_status_and_resumes_by_keyset() -> None:
    collection = await Collection.create("notes")
    for doc in ("a.md", "b.md", "c.md"):
        await collection.add((await attachable(doc)).name)
    await collection.set_member_status("b.md", "indexed")

    first = await collection.members_page(PageRequest(page_size=2))
    assert [m.document.name for m in first.items] == ["a.md", "b.md"]
    resumed = await collection.members_page(PageRequest(cursor=first.next_cursor, page_size=2))
    assert [m.document.name for m in resumed.items] == ["c.md"]

    indexed = await collection.members_page(PageRequest(), status="indexed")
    assert [m.document.name for m in indexed.items] == ["b.md"]
    assert indexed.total == 1, "total counts the filtered rows"


@pytest.mark.anyio
async def test_one_document_sits_in_two_collections_and_a_detach_leaves_both_alone() -> None:
    """Membership is many-to-many, so detaching from one collection touches neither the document
    nor the other collection, and never the embedding cache."""
    doc = await attachable("shared.md")
    for name in ("alpha", "beta"):
        await (await Collection.create(name)).add(doc.name)
    params = embed_cache.params(doc, ChunkSettings(), None)
    rows = home.HOME / "rows" / "000000.rows.json"
    rows.parent.mkdir(parents=True, exist_ok=True)
    rows.write_bytes(msgspec.json.encode([Row(chunk=chunk.split(MD, SMALL)[0])]))
    cache_id = await embed_cache.write(params, [rows], None)

    assert await document.collections_of(doc.name) == ["alpha", "beta"]

    await Collection("alpha").remove_member(doc.name)

    assert await document.collections_of(doc.name) == ["beta"], "only that membership went"
    assert (await document.get(doc.name)).name == doc.name, "the document stays"
    assert await embed_cache.lookup(params) == cache_id, "and so does what it costs to compute"
    assert await Collection("beta").member(doc.name) is not None


@pytest.mark.anyio
async def test_deleting_a_collection_leaves_its_documents() -> None:
    doc = await attachable("kept.md")
    for name in ("alpha", "beta"):
        await (await Collection.create(name)).add(doc.name)

    await (await Collection.get("alpha")).delete()

    assert await Collection.names() == ["beta"]
    assert await document_names() == ["kept.md"], "the document belongs to no collection"
    assert doc.original.exists(), "and keeps its files"
    assert await document.collections_of(doc.name) == ["beta"]


@pytest.mark.anyio
async def test_deleting_a_document_takes_every_membership_and_cache_row_with_it() -> None:
    doc = await attachable("gone.md")
    for name in ("alpha", "beta"):
        await (await Collection.create(name)).add(doc.name)
    params = embed_cache.params(doc, ChunkSettings(), None)
    rows = home.HOME / "rows" / "000000.rows.json"
    rows.parent.mkdir(parents=True, exist_ok=True)
    rows.write_bytes(msgspec.json.encode([Row(chunk=chunk.split(MD, SMALL)[0])]))
    await embed_cache.write(params, [rows], None)

    await document.remove_files(doc.name)
    await document.remove_row(doc.name)

    assert await document_names() == []
    assert await document.collections_of("gone.md") == [], "both memberships cascaded"
    assert await embed_cache.entries("gone.md") == [], "and so did the cache rows"
    assert not doc.root.exists()
    assert await Collection("alpha").counts() == DocumentCounts(), "the collections stay, empty"


@pytest.mark.anyio
async def test_an_unattached_document_is_valid_and_listable() -> None:
    doc = await import_row("lonely.md")

    assert await document.collections_of(doc.name) == []
    page = await document.page(PageRequest())
    assert [d.name for d in page.items] == ["lonely.md"]


@pytest.mark.anyio
async def test_deleting_a_collection_drops_it_from_every_session() -> None:
    """One cascade from the `collections` row: the memberships and every session that chose it."""
    for name in ("keep", "drop"):
        await Collection.create(name)
    async with db.connect() as conn:
        await conn.execute("insert into sessions (id) values ('s1')")
        await conn.executemany(
            "insert into session_collections (session_id, collection, position) values (?, ?, ?)",
            [("s1", "keep", 0), ("s1", "drop", 1)],
        )

    await (await Collection.get("drop")).delete()

    async with db.connect() as conn:
        rows = await conn.execute_fetchall("select collection from session_collections")
    assert list(rows) == [("keep",)]
    assert await Collection.names() == ["keep"]


# --- index -------------------------------------------------------------------------

COMPACT = EmbeddingModel("BAAI/bge-small-en-v1.5", 384)


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
            "vector of the current dimensions",
            PLAIN_SCHEMA.append(pa.field("vector", pa.list_(pa.float32(), 384))),
            COMPACT,
            True,
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

    await collection.delete()
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
    assert await index.search("anything", SearchSettings()) == []


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
    (row,) = [Row(chunk=c) for c in chunk.split(MD, SMALL)[:1]]
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


def _row(text: str, vector: list[float] | None = None) -> Row:
    return Row(
        chunk=Chunk(
            heading="H",
            text=text,
            line_start=1,
            line_end=1,
            char_start=0,
            char_end=len(text),
            parents=[],
        ),
        vector=vector,
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
        _row(f"{doc} part{part} row{i} lancedb", _vector(part * 1000 + i) if vectors else None)
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

    assert builds == ["text"], "one build across two documents"
    assert await index.has_index("text") is True
    assert [list(i.columns) for i in await table.list_indices()] == [["text"]], "and one index"
    found = {hit.doc for hit in await index.search("lancedb", SearchSettings(limit=10))}
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
    settings = SearchSettings(mode="vector", limit=3, nprobes=4, refine_factor=2)
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
            SearchSettings(fusion="linear", candidates=4, nprobes=4, refine_factor=2),
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
    assert (await maintenance_state("busy")).pending_docs == 20
    claimed = (await maintenance_state("busy")).pending_docs

    report = await maintenance.run(collection, None, PipelineSettings())
    await Collection("busy").settle_maintenance(claimed, report.ann_trained, report.num_rows)

    assert report.skipped is None
    assert (report.collection, report.num_rows, report.ann_trained) == ("busy", 40, False)
    assert report.fragments_before == 20
    assert report.fragments_after < report.fragments_before
    after = await collection.index_with(None).stats()
    assert after is not None and (after.num_rows, after.num_fragments) == (40, 1)
    state = await maintenance_state("busy")
    assert (state.pending_docs, state.vector_index_rows) == (0, 0)
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
    await collection.delete()

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
    assert (info.settings, info.effective) == (CollectionSettings(), ChunkSettings())
    assert info.counts == DocumentCounts() and info.index_outdated is False

    await _fill(collection.index_with(None), "a.md", 0, 3)
    await Collection("info").note_indexed()

    info = await collection.info()
    assert info.index is not None
    assert (info.index.num_rows, info.index.num_fragments) == (3, 1)
    assert (info.index.has_fts_index, info.index.has_vector_index) == (False, False)
    assert (info.maintenance.pending_docs, info.maintenance.last_maintained_at) == (1, None)


@pytest.mark.anyio
async def test_search_falls_back_to_fts_without_an_embedding_model(tmp_path: Path) -> None:
    """A vector query needs something to embed the question with; without a model the mode is
    downgraded rather than failing."""
    path = tmp_path / "index"
    schema = PLAIN_SCHEMA.append(pa.field("vector", pa.list_(pa.float32(), 2)))
    table = lancedb.connect(str(path)).create_table("chunks", schema=schema)
    table.add([{"doc": "a.md", "chunk_id": 0, "heading": "H", "text": "hi", "vector": [0.1, 0.2]}])
    index = CollectionIndex(path, "notes", tmp_path, None)
    await index.finish()  # build the full-text index the fallback needs

    hits = await index.search("hi", SearchSettings(mode="vector"))

    assert [h.doc for h in hits] == ["a.md"]


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
    "chunk_id": 0,
    "line_start": 0,
    "line_end": 0,
    "char_start": 0,
    "char_end": 0,
    "page_start": None,
    "page_end": None,
}


def test_hit_of_a_row_with_nothing_optional_set(tmp_path: Path) -> None:
    """Every column of `PLAIN_SCHEMA` is present in any row this build can read, but a chunk may
    carry no heading, no ancestry and no pages: nothing is invented for those."""
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    hit = index.hit(PLAIN_ROW | {"doc": "a.md", "chunk_id": 3})
    assert isinstance(hit, Hit)
    assert (hit.source_path, hit.markdown_path, hit.part) == ("", "", 0)
    assert (hit.source_file, hit.markdown_file) == ("", ""), "no path, so nothing to resolve"
    assert (hit.page_start, hit.page_end, hit.parents) == (None, None, [])
    assert (hit.header, hit.location) == ("", "a.md L0-0")


def test_hit_names_the_collection_that_matched_and_builds_a_citation(tmp_path: Path) -> None:
    """A document belongs to no collection, so the hit carries the collection whose table matched
    it; the paths it carries are the document's own."""
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    hit = index.hit(
        {
            "doc": "book.pdf",
            "chunk_id": 1,
            "part": 2,
            "source_path": "documents/1f/book.pdf/original.pdf",
            "markdown_path": "documents/1f/book.pdf/original.pdf.md",
            "line_start": 10,
            "line_end": 20,
            "char_start": 100,
            "char_end": 200,
            "page_start": 3,
            "page_end": 4,
            "parents": "Part I > Chapter 2",
            "heading": "Results",
            "text": "body",
            "_score": 0.5,
        }
    )
    assert hit.collection == "notes"
    assert hit.parents == ["Part I", "Chapter 2"]
    assert hit.header == "Part I > Chapter 2 > Results"
    assert hit.location == "book.pdf p.3-4 L10-20"
    assert hit.score == 0.5


@pytest.mark.parametrize(
    ("name", "settings", "expected"),
    [
        ("rrf is the default fusion", SearchSettings(), "RRFReranker"),
        ("linear weights the two rankings", SearchSettings(fusion="linear"), "LinearCombination"),
        (
            "zero weights fall back to an even split",
            SearchSettings(fusion="linear", vector_weight=0.0, bm25_weight=0.0),
            "LinearCombination",
        ),
    ],
)
def test_fusion_reranker_per_setting(name: str, settings: SearchSettings, expected: str) -> None:
    assert type(_fusion(settings)).__name__.startswith(expected), name


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "texts", "expected"),
    [
        ("nothing retrieved, nothing to rescore", [], []),
        ("best first, whatever retrieval scored", ["short", "a longer chunk"], [14.0, 5.0]),
    ],
)
async def test_cross_encode_rescores_candidates_best_first(
    monkeypatch: pytest.MonkeyPatch, name: str, texts: list[str], expected: list[float]
) -> None:
    """The cross-encoder is CPU work, so it runs in a worker thread; its score replaces whatever
    the retrieval stage put on the row (see `row_score`)."""
    from haskie import models
    from haskie.index import cross_encode

    checked: list[tuple[str, str]] = []

    async def require_ready(kind: str, model: str) -> None:
        checked.append((kind, model))

    monkeypatch.setattr(models, "require_ready", require_ready)
    monkeypatch.setattr(embed, "rerank_scores", lambda model, q, ts: [float(len(t)) for t in ts])
    settings = SearchSettings(reranker="cross-encoder")
    rows = [{"text": text, "_score": 9.0} for text in texts]

    ranked = await cross_encode("q", rows, settings)

    assert [row_score(row) for row in ranked] == expected, name
    assert checked == [("reranker", settings.reranker_model)], name


@pytest.mark.anyio
async def test_an_index_with_an_embedding_stores_a_vector_column(tmp_path: Path) -> None:
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, EmbeddingModel("test/model", 2))
    (chunk_,) = chunk.split("# H\n\nbody\n", ChunkSettings())

    row = Row(chunk=chunk_, vector=[0.1, 0.2])
    await index.add_parts("g.md", "documents/g.md", "documents/g.md.md", _aparts([(0, [row])]))

    table = await index._existing()
    assert table is not None
    assert (await table.schema()).field("vector").type == pa.list_(pa.float32(), 2)
    (record,) = (await table.to_arrow()).to_pylist()
    assert (record["doc"], record["heading"], record["part"]) == ("g.md", "H", 0)
    assert record["vector"] == pytest.approx([0.1, 0.2])
    assert await index.schema_current() is True


@pytest.mark.anyio
async def test_add_parts_writes_one_fragment_for_many_parts(tmp_path: Path) -> None:
    """Three parts, one commit, one fragment. An empty part inside the group writes nothing but
    does not break the group."""
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    chunks = chunk.split(MD, SMALL)
    parts = [(0, [Row(chunk=c) for c in chunks]), (1, []), (2, [Row(chunk=c) for c in chunks])]

    written = await index.add_parts("g.md", "documents/g.md", "documents/g.md.md", _aparts(parts))

    table = await index._existing()
    assert table is not None
    assert written == await table.count_rows() == 2 * len(chunks)
    assert await _fragments(index) == 1, "one commit, however many parts it carried"
    records = (await table.to_arrow()).to_pylist()
    assert sorted({r["part"] for r in records}) == [0, 2], "the empty part is skipped"
    assert {r["chunk_id"] for r in records} == set(range(len(chunks))), "ids restart per part"
    assert {r["markdown_path"] for r in records} == {"documents/g.md.md"}


@pytest.mark.anyio
async def test_delete_parts_removes_only_the_range(tmp_path: Path) -> None:
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    (chunk_,) = chunk.split("# H\n\nbody\n", ChunkSettings())
    for doc in ("a.md", "b.md"):
        await index.add_parts(
            doc, "s", "m", _aparts([(part, [Row(chunk=chunk_)]) for part in range(4)])
        )

    await index.delete_parts("a.md", 1, 3)

    table = await index._existing()
    assert table is not None
    kept = {(r["doc"], r["part"]) for r in (await table.to_arrow()).to_pylist()}
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
    rows = [Row(chunk=c) for c in chunk.split("# H\n\nlancedb chapter one\n", ChunkSettings())]
    await index.add_parts("d.md", "documents/d.md", "documents/d.md.md", _aparts([(0, rows)]))
    assert await index.has_index("text") is False, "written, not indexed: the state under test"
    assert await index.fts_rows("lancedb", 10) == [], "rows are there, the full-text index is not"

    await index.finish()

    (row,) = await index.fts_rows("lancedb", 10)
    assert (row["doc"], row["chunk_id"]) == ("d.md", 0)
    assert row["_score"] > 0, "raw BM25, which is what the cross-collection merge sorts on"


@pytest.mark.anyio
async def test_search_rows_returns_raw_rows_without_cutting(tmp_path: Path) -> None:
    """Retrieval only: as many rows as the caller asked for, carrying the engine's own score and
    no cross-encoder score. `search` is what cuts to `settings.limit`."""
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, None)
    chunks = [
        c for i in range(6) for c in chunk.split(f"# H\n\nlancedb chapter {i}\n", ChunkSettings())
    ]
    assert len(chunks) == 6, "one chunk per text, or the row counts below mean nothing"
    rows = [Row(chunk=c) for c in chunks]
    await index.add_parts("d.md", "documents/d.md", "documents/d.md.md", _aparts([(0, rows)]))
    await index.finish()
    settings = SearchSettings(limit=2, candidates=4)

    rows = await index.search_rows("lancedb", None, settings, 4)

    assert len(rows) == 4, "the fetch size wins over settings.limit"
    assert all("_score" in row and "_relevance_score" not in row for row in rows)
    assert {row["doc"] for row in rows} == {"d.md"}
    assert len(await index.search_rows("lancedb", None, settings, 100)) == 6, "no more than exist"
    assert len(await index.search("lancedb", settings)) == 2, "the composed search cuts to limit"
    missing = CollectionIndex(tmp_path / "missing", "notes", tmp_path, None)
    assert await missing.search_rows("lancedb", None, settings, 4) == [], "no table, no rows"
    assert await missing.search("lancedb", settings) == [], "and nothing to compose a search from"


@pytest.mark.parametrize(
    ("name", "embedding", "settings", "vector_column", "expected"),
    [
        ("mode fts never embeds", COMPACT, SearchSettings(mode="fts"), True, None),
        ("no embedding profile", None, SearchSettings(mode="hybrid"), True, None),
        ("no vector column", COMPACT, SearchSettings(mode="hybrid"), False, None),
        ("hybrid over a vector table", COMPACT, SearchSettings(mode="hybrid"), True, [0.5] * 384),
        (
            "vector mode over a vector table",
            COMPACT,
            SearchSettings(mode="vector"),
            True,
            [0.5] * 384,
        ),
    ],
)
@pytest.mark.anyio
async def test_query_vector_is_none_for_fts_and_without_embedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    embedding: EmbeddingModel | None,
    settings: SearchSettings,
    vector_column: bool,
    expected: list[float] | None,
) -> None:
    from haskie import models

    path = tmp_path / "index"
    vector_field = pa.field("vector", pa.list_(pa.float32(), 384))
    _table_with(path, PLAIN_SCHEMA.append(vector_field) if vector_column else PLAIN_SCHEMA)
    checked: list[tuple[str, str]] = []

    async def require_ready(kind: str, model: str) -> None:
        checked.append((kind, model))

    monkeypatch.setattr(models, "require_ready", require_ready)
    monkeypatch.setattr(embed, "embed_query", lambda model, text: [0.5] * model.dims)
    index = CollectionIndex(path, "notes", tmp_path, embedding)

    assert await index.query_vector("q", settings) == expected, name
    assert checked == ([("embedding", COMPACT.name)] if expected else []), name


@pytest.mark.anyio
async def test_query_vector_of_a_never_indexed_collection_is_none(tmp_path: Path) -> None:
    """No table means no search, so the model is never asked for (it may not be loaded)."""
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, COMPACT)
    assert await index.query_vector("q", SearchSettings(mode="hybrid")) is None
    assert await index.search("q", SearchSettings(mode="hybrid")) == []


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
    doc: Document, chunking: ChunkSettings, embedding: EmbeddingModel | None = None
) -> str:
    params = embed_cache.params(doc, chunking, embedding)
    cache_id = embed_cache.key(params)
    for batch in await pipeline.plan_embed(doc):
        await pipeline.embed_batch(doc, batch, cache_id, chunking, embedding)
    return await pipeline.finalize_embed(doc, params, embedding)


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


@pytest.mark.anyio
async def test_plan_embed_requires_a_converted_document() -> None:
    doc = await import_row("g.md")
    with pytest.raises(FileNotFoundError, match="markdown parts missing"):
        await pipeline.plan_embed(doc)


@pytest.mark.anyio
async def test_finalize_embed_requires_the_rows_of_every_part() -> None:
    doc = await import_row("g.md")
    await _convert(doc)

    params = embed_cache.params(doc, ChunkSettings(), None)
    with pytest.raises(FileNotFoundError):
        await pipeline.finalize_embed(doc, params, None)
    assert await embed_cache.lookup(params) is None, "and nothing was published"


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
    chunks = await pipeline.embed_batch(doc, batch, embed_cache.key(params), SMALL, None)

    assert chunks == len(chunk.split(MD, SMALL))
    assert doc.markdown.read_text() == MD
    assert doc.part_path(0).read_text() == MD, "the part stays: it is the re-chunking input"
    assert list(doc.parts_dir.glob("*.tmp")) == [], "no temp file left behind"
    assert not doc.markdown.with_name(doc.markdown.name + ".tmp").exists()


@pytest.mark.anyio
async def test_a_reconversion_starts_the_documents_outputs_over() -> None:
    """The parts and the markdown are outputs of the conversion, so `plan_convert` rebuilds them.
    The cached embeddings were chunked from that markdown, so they go too — dropped by
    `embed_cache.forget`, which `workflows.import_document` runs before the convert stage."""
    doc = await import_row("g.md")
    await _convert(doc)
    cache_id = await _embed(doc, SMALL)
    assert embed_cache.file_path(doc.name, cache_id).exists()

    await embed_cache.forget(doc.name)
    await pipeline.plan_convert(doc, 10)

    assert not doc.markdown.exists(), "the assembled markdown is rebuilt"
    assert list(doc.parts_dir.iterdir()) == [], "and so are the parts"
    assert not doc.embeddings_dir.exists(), "no cache file survives a reconversion"
    assert await embed_cache.entries(doc.name) == [], "and no cache row either"


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
    (entry,) = await embed_cache.entries(doc.name)
    assert (entry.id, entry.rows) == (cache_id, written)
    (hit,) = await collection.search("lancedb")
    assert (hit.collection, hit.doc) == ("notes", "guide.md")
    assert hit.source_file == str(doc.original), "the hit points at the document's own files"
    assert hit.markdown_file == str(doc.markdown)
    assert Path(hit.markdown_file).read_text() == MD


@pytest.mark.anyio
async def test_index_batch_group_is_idempotent() -> None:
    """A replay after a crash between the LanceDB commit and the step checkpoint must rewrite the
    group rather than append it a second time."""
    collection = await Collection.create("groups")
    doc = await import_row("p.pdf", text_pdf(["alpha one", "beta two", "gamma three"]))
    await _convert(doc, batch_pages=1)
    cache_id = await _embed(doc, SMALL)

    assert await embed_cache.row_groups(doc.name, cache_id) == 3, "one group per converted part"
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
    """The point of the cache: the second collection computes nothing, it reads the parquet file
    the first one left and writes its own table from it."""
    alpha = await Collection.create("alpha")
    beta = await Collection.create("beta")
    await beta.set_settings(CollectionSettings(chunk_size=40, chunk_overlap=0))
    doc = await attachable("shared.md")
    await _convert(doc)

    first = await _embed(doc, await alpha.chunk_settings())
    second = await _embed(doc, await beta.chunk_settings())

    assert first != second, "different chunk settings, different entries, no collision"
    assert {entry.id for entry in await embed_cache.entries(doc.name)} == {first, second}
    assert await _embed(doc, await alpha.chunk_settings()) == first, "the same settings, same id"

    await alpha.add(doc.name)
    await beta.add(doc.name)
    rows_alpha = await _index(alpha, doc, first)
    rows_beta = await _index(beta, doc, second)

    assert rows_alpha > 0 and rows_beta > rows_alpha, "beta chunks the same markdown smaller"
    assert len(await alpha.search("lancedb")) > 0 and len(await beta.search("lancedb")) > 0
    assert await document.collections_of(doc.name) == ["alpha", "beta"]


# --- sessions and cross-collection search --------------------------------------------
#
# `CollectionIndex` splits search into retrieval (`search_rows`), the query embedding
# (`query_vector`) and row-to-Hit (`hit`), so a session embeds once, fans out and rescores once.
# `session.rrf_merge` fuses the per-collection rankings by rank, because two indexes do not score
# on the same scale. `textsearch.merge` merges raw BM25 scores instead: one lexical scorer with
# the same tokenizer answers in every collection. Both count a passage once, because one document
# may be a member of several of the collections being searched.

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
    from haskie import session

    await Collection.create("a")
    with pytest.raises(error, match=match):
        await session.set_collections(session_id, collections)
    assert await session.load() == {}, f"nothing stored for a rejected request: {name}"


@pytest.mark.anyio
async def test_set_collections_deduplicates_and_keeps_order() -> None:
    from haskie import session

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
    from haskie import session

    for name in ("a", "b", "c"):
        await Collection.create(name)

    await session.set_collections("s1", ["c", "a", "b"])
    assert await session.collections_for("s1") == ["c", "a", "b"]

    assert await session.set_collections("s1", ["b", "c"]) == ["b", "c"], "the selection is new"
    assert await session.load() == {"s1": ["b", "c"]}
    async with db.connect() as conn:
        rows = await conn.execute_fetchall(
            "select collection, position from session_collections where session_id = 's1' "
            "order by position"
        )
    assert list(rows) == [("b", 0), ("c", 1)], "one row per collection, renumbered from zero"

    await session.set_collections("s1", [])
    assert await session.load() == {"s1": []}, "an empty selection keeps the session itself"


@pytest.mark.anyio
async def test_session_search_skips_a_collection_that_disappeared(caplog, monkeypatch) -> None:
    """Deleting a collection drops it from every session (one cascade), so a name without a row
    can only come from a delete between the two reads of the search. It is skipped, not raised."""
    from haskie import session

    async def nothing_found(names: list[str]) -> dict:
        return {}

    await Collection.create("ghost")
    await session.set_collections("s1", ["ghost"])
    monkeypatch.setattr(Collection, "load_settings", staticmethod(nothing_found))

    with caplog.at_level("WARNING"):
        assert await session.search("s1", "anything") == []
    assert events(caplog) == ["session_collection_missing"]


@pytest.mark.anyio
async def test_session_search_reads_its_collections_concurrently(monkeypatch) -> None:
    """Two collections, two events: each retrieval announces itself and then waits for the other,
    so the search can only answer at all if the fan-out overlapped. A sequential fan-out would
    hold the first retrieval until the wait times out, and the timeout fails the search."""
    import asyncio

    from haskie import session

    for name in ("a", "b"):
        await Collection.create(name)
    await session.set_collections("s1", ["a", "b"])
    arrived = {"a": asyncio.Event(), "b": asyncio.Event()}

    async def paired(self, query, vector, settings_, limit) -> list[dict]:
        arrived[self.collection].set()
        other = arrived["b" if self.collection == "a" else "a"]
        await asyncio.wait_for(other.wait(), CONCURRENT_SEARCH_SECONDS)
        return []

    monkeypatch.setattr(CollectionIndex, "search_rows", paired)

    assert await session.search("s1", "anything") == []
    assert all(event.is_set() for event in arrived.values()), "both collections were read"


@pytest.mark.anyio
async def test_session_search_counts_a_passage_once_across_collections() -> None:
    """The same document in two chosen collections puts the same chunk in both rankings. A caller
    wants one hit per passage, so it is credited to the first collection that returned it and the
    copy is dropped before the ranks are counted."""
    from haskie import session

    for name in ("alpha", "beta"):
        collection = await Collection.create(name)
        index = collection.index_with(None)
        await _fill(index, "shared.md", 0, 3)
        await index.finish()
    await session.set_collections("s1", ["alpha", "beta"])

    hits = await session.search("s1", "lancedb", limit=10)

    assert len(hits) == 3, "three chunks, not six: the copies are merged away"
    passages = {(hit.doc, hit.part, hit.chunk_id) for hit in hits}
    assert passages == {("shared.md", 0, i) for i in range(3)}
    assert {hit.collection for hit in hits} == {"alpha"}, "the first collection that held it"


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
    from haskie import session

    merged = session.rrf_merge(ranked, k=60)

    assert [item for item, _ in merged] == [item for item, _ in expected], name
    assert [score for _, score in merged] == pytest.approx([s for _, s in expected]), name


def _retrieved(rows: list[tuple]) -> tuple[CollectionIndex, list[dict]]:
    """What one collection returned, for `merge`: it only ever reads `index.collection` and the
    rows' identity columns. A `score` of None writes no `_score` at all, as an unscored row has."""
    name = rows[0][0] if rows else "empty"
    index = CollectionIndex(Path("/nowhere") / name, name, Path("/nowhere"), None)
    return index, [
        {"doc": doc, "part": part, "chunk_id": chunk_id}
        | ({"_score": score} if score is not None else {})
        for _collection, doc, part, chunk_id, score in rows
    ]


def _identity(pairs: list[tuple]) -> list[tuple[str, str, int, int]]:
    return [(index.collection, row["doc"], row["part"], row["chunk_id"]) for index, row in pairs]


@pytest.mark.parametrize(
    ("name", "per_collection", "expected"),
    [
        ("nothing to merge", [], []),
        (
            "a collection that matched nothing contributes nothing",
            [[], [("a", "d.md", 0, 0, 1.0)]],
            [("a", "d.md", 0, 0)],
        ),
        (
            "the better score wins, whichever collection it came from",
            [[("b", "x.md", 0, 0, 9.0)], [("a", "y.md", 0, 0, 1.0)]],
            [("b", "x.md", 0, 0), ("a", "y.md", 0, 0)],
        ),
        (
            "the same passage from two collections is kept once, the better copy",
            [[("b", "d.md", 0, 0, 1.0)], [("a", "d.md", 0, 0, 9.0)]],
            [("a", "d.md", 0, 0)],
        ),
        (
            "an equal score falls back to the collection name, and still counts once",
            [[("b", "d.md", 0, 0, 1.0)], [("a", "d.md", 0, 0, 1.0)]],
            [("a", "d.md", 0, 0)],
        ),
        (
            "inside one collection: doc, then part, then chunk",
            [
                [
                    ("a", "z.md", 0, 0, 1.0),
                    ("a", "a.md", 1, 0, 1.0),
                    ("a", "a.md", 0, 5, 1.0),
                    ("a", "a.md", 0, 1, 1.0),
                ]
            ],
            [("a", "a.md", 0, 1), ("a", "a.md", 0, 5), ("a", "a.md", 1, 0), ("a", "z.md", 0, 0)],
        ),
        (
            "a row an older index wrote without a score sorts last",
            [[("a", "d.md", 0, 1, None)], [("a", "d.md", 0, 0, 0.5)]],
            [("a", "d.md", 0, 0), ("a", "d.md", 0, 1)],
        ),
    ],
)
def test_text_merge_orders_by_score_then_identity(
    name: str, per_collection: list[list[tuple]], expected: list[tuple[str, str, int, int]]
) -> None:
    from haskie import textsearch

    merged = textsearch.merge([_retrieved(rows) for rows in per_collection])

    assert _identity(merged) == expected, name


def _digest() -> str:
    """The query identity `TEXT_QUERY` hashes to, read at collection time by the table below."""
    from haskie import textsearch

    return textsearch.query_hash(*TEXT_QUERY)


def _wire_cursor(**overrides) -> str:
    """A cursor built straight on the wire format, for the fields `make_cursor` never varies."""
    from haskie import textsearch

    payload: dict = {"k": [_digest(), 10], "s": textsearch.SORT, "o": textsearch.ORDER, "v": 1}
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
    from haskie import textsearch

    q, collections, page_size = TEXT_QUERY
    assert textsearch.parse_cursor(None, q, collections, page_size) == 0, "no cursor, first page"
    issued = textsearch.make_cursor(q, collections, page_size, 50)
    assert textsearch.parse_cursor(issued, q, collections, page_size) == 50, "round trip"
    assert textsearch.make_cursor(q, ["beta", "alpha"], page_size, 50) == issued, "order-free"

    if isinstance(rejected, tuple):
        cursor = textsearch.make_cursor(*rejected)  # a cursor this module issued, another page
    elif isinstance(rejected, dict):
        cursor = _wire_cursor(**rejected)
    else:
        cursor = rejected

    with pytest.raises(InvalidInput) as raised:
        textsearch.parse_cursor(cursor, q, collections, page_size)
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
    from haskie import textsearch

    assert textsearch.split_collections(collections) == expected, name


# --- storage -----------------------------------------------------------------------


def test_migrations_apply_once_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "m.db"
    conn = sqlite3.connect(str(path))
    assert db.migrate(conn) == db.SCHEMA_VERSION
    assert conn.execute("pragma user_version").fetchone() == (db.SCHEMA_VERSION,)
    assert db.migrate(conn) == db.SCHEMA_VERSION, "re-run is a no-op"

    # a DB behind by one version only gets the tail applied
    monkeypatch.setattr(db, "MIGRATIONS", [*db.MIGRATIONS, "create table extra (x);"])
    assert db.migrate(conn) == db.SCHEMA_VERSION + 1
    assert conn.execute("pragma user_version").fetchone() == (db.SCHEMA_VERSION + 1,)
    assert conn.execute("select count(*) from extra").fetchone() == (0,)
    conn.close()


def test_a_home_from_before_the_current_schema_is_refused(tmp_path: Path) -> None:
    """The storage shape changed and nothing is reshaped in place, so an older home is refused
    with its rows untouched: the user destroys it rather than losing them to a silent drop."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.executescript("create table libraries (name text primary key);")
    conn.execute("insert into libraries (name) values ('notes')")
    conn.execute(f"pragma user_version = {db.SCHEMA_VERSION - 1}")
    conn.commit()

    with pytest.raises(HaskieError) as raised:
        db.migrate(conn)

    assert str(raised.value) == db.INCOMPATIBLE_HOME_MESSAGE
    assert "haskie destroy" in db.INCOMPATIBLE_HOME_MESSAGE, "the message names the way out"
    assert conn.execute("pragma user_version").fetchone() == (db.SCHEMA_VERSION - 1,), "untouched"
    tables = {name for (name,) in conn.execute("select name from sqlite_master where type='table'")}
    assert tables == {"libraries"}, "nothing created and nothing dropped"
    assert conn.execute("select count(*) from libraries").fetchone() == (1,), "the rows are there"
    conn.close()


@pytest.mark.anyio
async def test_connect_rolls_back_a_failed_unit_of_work() -> None:
    with pytest.raises(sqlite3.IntegrityError):
        async with db.connect() as conn:
            await conn.execute("insert into collections (name) values ('half')")
            await conn.execute(
                "insert into collection_documents (collection, document) values ('ghost', 'a.md')"
            )
    assert await Collection.names() == [], "the first insert of the failed block is gone too"


# --- embeddings and logging --------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "available", "accelerator", "expected"),
    [
        (
            "cuda wins over coreml",
            ["CPUExecutionProvider", "CoreMLExecutionProvider", "CUDAExecutionProvider"],
            "auto",
            ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            "apple silicon -> coreml then cpu",
            ["CoreMLExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"],
            "auto",
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
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


def test_device_name_is_the_first_provider_without_its_suffix() -> None:
    assert embed.device_name("cpu") == "CPU"


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


def test_cross_encoders_build_on_cpu_whatever_the_accelerator(monkeypatch) -> None:
    """The CoreML build of a reranker stalled the app for seconds; CPU scores `candidates` texts
    in milliseconds, so the accelerator setting does not reach it."""
    import fastembed.rerank.cross_encoder as module

    seen: list[list] = []

    class Recorder:
        def __init__(self, model_name: str, providers: list) -> None:
            seen.append(providers)

    monkeypatch.setattr(module, "TextCrossEncoder", Recorder)
    embed._build_cross_encoder.cache_clear()
    try:
        embed._cross_encoder("Xenova/ms-marco-MiniLM-L-6-v2")
    finally:
        embed._build_cross_encoder.cache_clear()
    assert seen == [["CPUExecutionProvider"]]


def test_embedding_helpers_short_circuit_on_empty_input() -> None:
    """No text means no model, so neither call may download anything."""
    assert embed.embed_texts(COMPACT, []) == []
    assert embed.rerank_scores("Xenova/ms-marco-MiniLM-L-6-v2", "q", []) == []


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
    assert {"collection", "doc"} <= audit.RECORD_FIELDS
    assert "library" not in audit.RECORD_FIELDS
