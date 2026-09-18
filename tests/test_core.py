"""Pure checks: chunking, conversion, settings, library and index IO, with HASKIE_HOME pointed
at a temp dir (see conftest.py).

Nothing here launches DBOS. Every test that needs the durable runtime lives in
`tests/test_workflows.py`; the HTTP contract lives in `tests/test_api.py`.
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
from conftest import maintenance_state, text_pdf

from haskie import audit, chunk, convert, db, embed, logs, maintenance, pipeline, session, toc
from haskie.chunk import Chunk
from haskie.errors import (
    Conflict,
    ConversionError,
    DocumentNotFound,
    InvalidInput,
    LibraryNotFound,
    NeedsOcr,
    UnsupportedFileType,
)
from haskie.index import (
    PLAIN_SCHEMA,
    Hit,
    IndexStats,
    LibraryIndex,
    Row,
    _fusion,
    _partitions,
    row_score,
)
from haskie.library import Library
from haskie.settings import (
    ConversionSettings,
    EmbeddingModel,
    LibrarySettings,
    PipelineSettings,
    SearchOverrides,
    SearchSettings,
    UserSettings,
    docs,
    init_user_settings,
    initialized,
    load_user_settings,
    save_user_settings,
)

MD = "# Title\n\nintro text\n\n## Alpha\n\nalpha body about lancedb\n\n## Beta\n\nbeta body\n"


SMALL = ConversionSettings(chunk_size=40, chunk_overlap=0)

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


# --- chunking and table of contents ------------------------------------------------


@pytest.mark.parametrize(
    ("name", "text", "settings", "expected"),
    [
        ("no headings -> one untitled chunk", "plain text", ConversionSettings(), [("", 1, 1, [])]),
        (
            "fits capacity -> one chunk, first heading",
            MD,
            ConversionSettings(),
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
            ConversionSettings(chunk_size=30, chunk_overlap=0),
            [("H", 1, 1, [])] + [("H", 3, 3, [])] * 4,
        ),
        (
            "text before first heading -> untitled",
            "pre\n# H\nbody",
            ConversionSettings(chunk_size=8, chunk_overlap=0),
            [("", 1, 1, []), ("H", 2, 3, [])],
        ),
        (
            "nested headings -> ancestry",
            "# A\n## B\n### C\nc\n## D\nd\n",
            ConversionSettings(chunk_size=6, chunk_overlap=0),
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
            ConversionSettings(chunker="text", chunk_size=1000, chunk_overlap=0),
            [("Title", 1, 11, [])],
        ),
        ("empty text -> no chunks", "", ConversionSettings(), []),
        (
            "heading inside first chunk -> used",
            "<!-- page 1 -->\n\n# H\nbody",
            ConversionSettings(),
            [("H", 1, 4, [])],
        ),
    ],
)
def test_chunk(name: str, text: str, settings: ConversionSettings, expected: list) -> None:
    chunks = chunk.split(text, settings)
    got = [(c.heading, c.line_start, c.line_end, c.parents) for c in chunks]
    assert got == expected, name
    assert all(text[c.char_start : c.char_end] == c.text for c in chunks), name


def test_chunk_pages_from_markers() -> None:
    text = "<!-- page 3 -->\n\n# A\nbody a\n\n<!-- page 4 -->\n\n# B\nbody b\n"
    small = [
        (c.heading, c.page_start, c.page_end)
        for c in chunk.split(text, ConversionSettings(chunk_size=30, chunk_overlap=0))
    ]
    assert small == [("A", 3, 3), ("B", 4, 4)]
    (whole,) = chunk.split(text, ConversionSettings())
    assert (whole.page_start, whole.page_end) == (3, 4), "chunk spanning pages reports the range"


def test_chunk_rejects_overlap_ge_size() -> None:
    """The splitter itself rejects it; settings validation stops it one layer earlier."""
    bad = ConversionSettings(chunk_size=10, chunk_overlap=9)
    object.__setattr__(bad, "chunk_overlap", 10)  # past __post_init__, as a stored row could be
    with pytest.raises(ValueError, match="overlap"):
        chunk.split(MD, bad)


def test_toc_levels_and_offsets() -> None:
    result = toc.headings(MD)
    assert [(h.level, h.text) for h in result] == [(1, "Title"), (2, "Alpha"), (2, "Beta")]
    assert MD.encode()[result[1].offset :].startswith(b"## Alpha")


# --- settings ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "overrides", "expected"),
    [
        ("no overrides -> user defaults", LibrarySettings(), ("anydoc", 1200, True)),
        ("override parser only", LibrarySettings(parser="plain"), ("plain", 1200, True)),
        ("override size only", LibrarySettings(chunk_size=990), ("anydoc", 990, True)),
        (
            "override skip_ocr_pages off",
            LibrarySettings(skip_ocr_pages=False),
            ("anydoc", 1200, False),
        ),
        (
            "explicit True beats user False",
            LibrarySettings(skip_ocr_pages=True),
            ("anydoc", 1200, True),
        ),
    ],
)
def test_library_settings_resolve(
    name: str, overrides: LibrarySettings, expected: tuple[str, int, bool]
) -> None:
    user = UserSettings()
    if name.startswith("explicit True"):
        user = UserSettings(conversion=ConversionSettings(skip_ocr_pages=False))
    effective = overrides.resolve(user)
    assert (effective.parser, effective.chunk_size, effective.skip_ocr_pages) == expected, name


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
    # library overrides reuse the same definitions
    lib_docs = docs(LibrarySettings)
    assert lib_docs["chunk_size"] == user_docs["conversion.chunk_size"]
    assert lib_docs["search.reranker"] == user_docs["search.reranker"]


def test_docs_rejects_a_non_struct() -> None:
    with pytest.raises(TypeError, match="needs a msgspec Struct"):
        docs(int)  # ty: ignore[invalid-argument-type]


def test_embedding_model_carries_accelerator() -> None:
    user = UserSettings(embedding="compact", pipeline=PipelineSettings(accelerator="cpu"))
    assert user.embedding_model is not None and user.embedding_model.accelerator == "cpu"
    assert UserSettings(embedding="none").embedding_model is None


@pytest.mark.anyio
async def test_user_settings_persist_in_db() -> None:
    assert await initialized() is False
    assert (await load_user_settings()).embedding == "none"
    await save_user_settings(UserSettings(embedding="compact"))
    assert await initialized() is True
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
        ("mixed, skip off -> NeedsOcr", ["one", None, "two", None], False, "raise"),
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
    if expect == "raise":
        with pytest.raises(NeedsOcr, match="need OCR"):
            convert.to_markdown(pdf, "anydoc", skip)
        return
    markdown = convert.to_markdown(pdf, "anydoc", skip)
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
        ("unknown suffix", "x.zip", b"PK", UnsupportedFileType, "unsupported file type: .zip"),
        ("no suffix at all", "README", b"text", UnsupportedFileType, "unsupported file type"),
        ("corrupt pdf", "broken.pdf", b"not a pdf", ConversionError, "broken.pdf: "),
        ("corrupt office file", "broken.docx", b"not a zip", ConversionError, "broken.docx: "),
    ],
)
def test_to_markdown_rejects_what_it_cannot_read(
    tmp_path: Path, name: str, filename: str, content: bytes, error: type[Exception], match: str
) -> None:
    path = tmp_path / filename
    path.write_bytes(content)
    with pytest.raises(error, match=match):
        convert.to_markdown(path, "anydoc")


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
    with pytest.raises(ConversionError, match="broken.pdf"):
        convert.build_preview(bad, tmp_path / "p", "anydoc")


def test_pdf_page_count_of_a_corrupt_file_raises_conversion_error(tmp_path: Path) -> None:
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(ConversionError, match="broken.pdf"):
        convert.pdf_page_count(bad)


# --- library -----------------------------------------------------------------------


@pytest.mark.anyio
async def test_library_rejects_unsupported_and_missing() -> None:
    lib = await Library.create("misc")
    with pytest.raises(UnsupportedFileType, match="unsupported file type"):
        await lib.save("virus.exe", b"")
    with pytest.raises(Conflict, match="already exists"):
        await Library.create("misc")
    with pytest.raises(LibraryNotFound, match="library not found"):
        await Library.get("nope")
    with pytest.raises(DocumentNotFound, match="document file missing"):
        lib.source_path("missing.md")
    with pytest.raises(DocumentNotFound, match="document not found"):
        await lib.document("missing.md")
    with pytest.raises(InvalidInput, match="invalid name"):
        await Library.create("***")


@pytest.mark.parametrize(
    ("name", "path", "error", "match"),
    [
        ("relative path", "notes/a.md", InvalidInput, "path must be absolute"),
        ("absolute but missing", "/definitely/not/here/a.md", InvalidInput, "file not found"),
        ("a directory, not a file", "{tmp}", InvalidInput, "file not found"),
        ("unsupported suffix", "{tmp}/v.exe", UnsupportedFileType, "unsupported file type"),
        ("over the upload cap", "{tmp}/big.md", InvalidInput, "file larger than"),
    ],
)
@pytest.mark.anyio
async def test_save_path_validates_the_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    path: str,
    error: type[Exception],
    match: str,
) -> None:
    from haskie import library as library_module

    monkeypatch.setattr(library_module, "UPLOAD_MAX_BYTES", 16)
    (tmp_path / "v.exe").write_bytes(b"x")
    (tmp_path / "big.md").write_bytes(b"x" * 64)
    lib = await Library.create("imports")

    with pytest.raises(error, match=match):
        await lib.save_path(path.format(tmp=tmp_path))
    assert await lib.document_names() == [], f"nothing stored for a rejected import: {name}"


@pytest.mark.anyio
async def test_save_path_copies_the_file_and_records_its_size(tmp_path: Path) -> None:
    """An import is streamed to its place with `shutil.copyfile`, so the size on the row is the
    size the file was stated at, and the file outside the home is left where it is."""
    source = tmp_path / "outside.md"
    source.write_text(MD)
    lib = await Library.create("imports")

    document = await lib.save_path(str(source))

    assert (document.name, document.size, document.status) == ("outside.md", len(MD), "uploaded")
    assert lib.file_path("outside.md").read_text() == MD, "copied, byte for byte"
    assert source.read_text() == MD, "a copy, not a move"
    assert await lib.document_names() == ["outside.md"]


@pytest.mark.anyio
async def test_save_rejects_content_over_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    from haskie import library as library_module

    monkeypatch.setattr(library_module, "UPLOAD_MAX_BYTES", 8)
    lib = await Library.create("caps")
    with pytest.raises(InvalidInput, match="file larger than 8 bytes: 20"):
        await lib.save("big.md", b"x" * 20)


@pytest.mark.anyio
async def test_ensure_preview_builds_once_and_then_reads_the_stored_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib = await Library.create("prev")
    doc = await lib.save("g.md", MD.encode())
    builds: list[str] = []
    real = convert.build_preview

    def counted(*args, **kwargs):
        builds.append(args[0].name)
        return real(*args, **kwargs)

    monkeypatch.setattr(convert, "build_preview", counted)

    first = await lib.ensure_preview(doc.name)
    second = await lib.ensure_preview(doc.name)

    assert builds == ["g.md"], "the second call reads the row instead of converting again"
    assert first.preview == second.preview and first.preview is not None
    assert first.preview.kind == "text"


@pytest.mark.anyio
async def test_library_settings_cache_invalidates_on_set_and_delete() -> None:
    from haskie import library as library_module

    lib = await Library.create("cached")
    assert await lib.settings() == LibrarySettings(), "no overrides yet"

    await lib.set_settings(LibrarySettings(parser="plain"))
    assert (await (await Library.get("cached")).settings()).parser == "plain", (
        "the setter refreshes"
    )

    async with db.connect() as conn:  # a write the setter never saw
        await conn.execute(
            "update libraries set settings = '{\"chunk_size\": 42}' where name = ?", ("cached",)
        )
    assert (await lib.settings()).chunk_size is None, "the process cache still owns the value"
    library_module.invalidate_library_caches()
    assert (await lib.settings()).chunk_size == 42

    await lib.delete()
    await Library.create("cached")
    assert await (await Library.get("cached")).settings() == LibrarySettings(), "starts clean"


@pytest.mark.anyio
async def test_load_settings_reads_every_library_it_was_asked_for_in_one_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cross-library search resolves the settings of its whole selection at once, so the cost is
    one SELECT rather than one per library. A name without a row stays out of the answer."""
    from haskie import library as library_module

    for name in ("alpha", "beta"):
        await Library.create(name)
    await Library("alpha").set_settings(LibrarySettings(parser="plain"))
    library_module.invalidate_library_caches()
    assert await Library.load_settings([]) == {}, "nothing asked for, nothing read"
    statements: list[str] = []
    real_execute = aiosqlite.Connection.execute

    async def counted(self, sql, parameters=None):
        statements.append(sql)
        return await real_execute(self, sql, parameters)

    monkeypatch.setattr(aiosqlite.Connection, "execute", counted)

    found = await Library.load_settings(["alpha", "beta", "ghost", "alpha"])

    assert set(found) == {"alpha", "beta"}, "a name with no row is absent from the result"
    assert (found["alpha"].parser, found["beta"]) == ("plain", LibrarySettings())
    selects = [sql for sql in statements if sql.lstrip().lower().startswith("select")]
    assert len(selects) == 1, "one query for every name, duplicates included"
    assert set(library_module._settings_cache) == {"alpha", "beta"}, "filled on the way"


@pytest.mark.anyio
async def test_preview_lock_is_striped_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    from haskie import library as library_module

    stripes = library_module._preview_locks
    assert len(stripes) == 64

    assert library_module._preview_lock("a", "g.md") is library_module._preview_lock("a", "g.md")
    assert library_module._preview_lock("a", "g.md") is not library_module._preview_lock(
        "a", "h.md"
    ), "different documents usually take different stripes"

    for i in range(10_000):
        library_module._preview_lock("lib", f"doc-{i}.md")

    assert library_module._preview_locks is stripes and len(stripes) == 64, "nothing allocated"

    fresh: list = [None] * library_module.PREVIEW_LOCK_STRIPES
    monkeypatch.setattr(library_module, "_preview_locks", fresh)
    lock = library_module._preview_lock("a", "g.md")

    assert sum(entry is not None for entry in fresh) == 1, "one lock, made on first use"
    assert library_module._preview_lock("a", "g.md") is lock, "and kept for the next reader"


@pytest.mark.anyio
async def test_delete_drops_the_library_from_every_session() -> None:
    for name in ("keep", "drop"):
        await Library.create(name)
    await session.set_libraries("s1", ["keep", "drop"])
    await session.set_libraries("s2", ["keep"])

    await (await Library.get("drop")).delete()

    assert await session.load() == {"s1": ["keep"], "s2": ["keep"]}
    assert await Library.names() == ["keep"]


def _events(caplog) -> list[str]:
    """Our loggers pass structlog's event dict as the record message."""
    return [r.msg["event"] for r in caplog.records if isinstance(r.msg, dict)]


# --- index -------------------------------------------------------------------------


COMPACT = EmbeddingModel("BAAI/bge-small-en-v1.5", 384)


def _table_with(path: Path, schema: pa.Schema) -> None:
    """A table written straight to disk, bypassing `LibraryIndex`: the schema under test is one
    the current build would never write. The sync LanceDB API on purpose, so sync fixtures and
    sync tests can set the stage as well."""
    lancedb.connect(str(path)).create_table("chunks", schema=schema)


async def _aparts(parts: list[tuple[int, list[Row]]]) -> AsyncIterator[tuple[int, list[Row]]]:
    """`add_parts` takes an async iterator: the pipeline decodes one `rows.json` at a time and
    awaits each read (see `pipeline._grouped_rows`)."""
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
    index = LibraryIndex(path, "lib", tmp_path, embedding)
    assert await index.schema_current() is expected, name


@pytest.mark.anyio
async def test_schema_current_is_cached_until_a_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index stage is the only writer of a table and runs in this process, so the answer is
    cached per index directory; the write path forgets it again."""
    from haskie.index import LibraryIndex as Index

    path = tmp_path / "index"
    _table_with(path, pa.schema([("doc", pa.string()), ("text", pa.string())]))  # older build
    inspected: list[Path] = []
    real = Index._existing

    async def counted(self: Index):
        inspected.append(self.path)
        return await real(self)

    monkeypatch.setattr(Index, "_existing", counted)

    assert await Index(path, "lib", tmp_path, None).schema_current() is False
    assert inspected == [path], "the first answer reads the table"
    inspected.clear()
    assert await Index(path, "lib", tmp_path, None).schema_current() is False
    assert inspected == [], "a second index instance answers from the process cache"

    await Index(path, "lib", tmp_path, None).reset_for_write()  # drops it, creates a current one

    assert await Index(path, "lib", tmp_path, None).schema_current() is True, "re-read"


@pytest.mark.anyio
async def test_a_missing_table_is_never_cached(tmp_path: Path) -> None:
    """Caching "nothing to reject" would hide the table the index stage creates a moment later."""
    path = tmp_path / "index"
    assert await LibraryIndex(path, "lib", tmp_path, COMPACT).schema_current() is True

    _table_with(path, PLAIN_SCHEMA)  # no vector column, so it cannot hold COMPACT rows

    assert await LibraryIndex(path, "lib", tmp_path, COMPACT).schema_current() is False


@pytest.mark.anyio
async def test_deleting_a_library_forgets_its_cached_schema() -> None:
    lib = await Library.create("recycled")
    _table_with(lib.index_dir, PLAIN_SCHEMA)
    assert await (await lib.index()).schema_current() is True

    await lib.delete()
    await Library.create("recycled")  # same name, same index directory
    _table_with(lib.index_dir, pa.schema([("doc", pa.string())]))  # written by an older build

    assert await (await (await Library.get("recycled")).index()).schema_current() is False


@pytest.mark.anyio
async def test_existing_never_creates_the_index_directory(tmp_path: Path) -> None:
    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, None)
    assert await index._existing() is None
    assert not (tmp_path / "index").exists(), "a read must not create a LanceDB directory"


@pytest.mark.anyio
async def test_existing_is_none_for_a_directory_without_the_table(tmp_path: Path) -> None:
    path = tmp_path / "index"
    lancedb.connect(str(path))  # creates the directory, no table
    assert await LibraryIndex(path, "lib", tmp_path, None)._existing() is None


@pytest.mark.anyio
async def test_search_of_a_never_indexed_library_is_empty(tmp_path: Path) -> None:
    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, None)
    assert await index.search("anything", SearchSettings()) == []


@pytest.mark.anyio
async def test_delete_on_an_outdated_table_removes_nothing_instead_of_dropping_it(
    tmp_path: Path,
) -> None:
    """Deleting one document must never wipe a library built by an older embedding (B1)."""
    path = tmp_path / "index"
    old = lancedb.connect(str(path)).create_table(
        "chunks", data=[{"doc": "a.md", "chunk_id": 0, "heading": "H", "text": "hello"}]
    )
    index = LibraryIndex(path, "lib", tmp_path, COMPACT)

    await index.delete_document("a.md")
    await index.delete_parts("a.md", 0, 1)

    assert old.count_rows() == 1, "rows of an unreadable schema are left alone"
    assert "chunks" in lancedb.connect(str(path)).list_tables().tables


@pytest.mark.anyio
async def test_add_parts_without_a_vector_is_rejected(tmp_path: Path) -> None:
    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, COMPACT)
    (row,) = [Row(chunk=c) for c in chunk.split(MD, SMALL)[:1]]
    with pytest.raises(ValueError, match="carries no vector"):
        await index.add_parts("g.md", "s", "m", _aparts([(0, [row])]))


@pytest.mark.anyio
async def test_add_parts_with_nothing_to_write_creates_no_table(tmp_path: Path) -> None:
    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, None)
    await index.add_parts("g.md", "s", "m", _aparts([(0, [])]))
    assert not (tmp_path / "index").exists()


@pytest.mark.anyio
async def test_finish_on_an_empty_index_is_a_no_op(tmp_path: Path) -> None:
    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, None)
    await index.finish()  # no table yet
    _table_with(tmp_path / "index", PLAIN_SCHEMA)
    await LibraryIndex(tmp_path / "index", "lib", tmp_path, None).finish()  # table, zero rows


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
    index: LibraryIndex, doc: str, part: int, count: int, vectors: bool = False
) -> None:
    rows = [
        _row(f"{doc} part{part} row{i} lancedb", _vector(part * 1000 + i) if vectors else None)
        for i in range(count)
    ]
    await index.add_parts(doc, f"files/{doc}", f"markdown/{doc}.md", _aparts([(part, rows)]))


@pytest.mark.anyio
async def test_finish_builds_fts_once_and_later_rows_are_still_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S1: a full-text index costs O(rows) to build, so a library of n documents must not build
    one per document. Rows added after the build are covered by a scan until maintenance folds
    them in, so nothing is lost by building it once."""
    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, None)
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
    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, None)
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
        ("tiny library clamps to the floor", 100, 16),
        ("50k rows -> sqrt rounded to a power of two", 50_000, 256),
        ("500k rows", 500_000, 512),
        ("huge library clamps to the ceiling", 10**12, 4096),
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
    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, TINY)
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
    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, TINY)
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
    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, TINY)
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
    lib = await Library.create("busy")
    index = lib.index_with(None)
    for part in range(20):
        await _fill(index, "a.md", part, 2)
        await Library("busy").note_indexed()
    before = await index.stats()
    assert before is not None and before.num_fragments == 20
    assert (await maintenance_state("busy")).pending_docs == 20
    claimed = (await maintenance_state("busy")).pending_docs

    report = await maintenance.run(lib, None, PipelineSettings())
    await Library("busy").settle_maintenance(claimed, report.ann_trained, report.num_rows)

    assert report.skipped is None
    assert (report.library, report.num_rows, report.ann_trained) == ("busy", 40, False)
    assert report.fragments_before == 20
    assert report.fragments_after < report.fragments_before
    after = await lib.index_with(None).stats()
    assert after is not None and (after.num_rows, after.num_fragments) == (40, 1)
    state = await maintenance_state("busy")
    assert (state.pending_docs, state.vector_index_rows) == (0, 0)
    assert state.last_maintained_at is not None and state.last_write_at is not None
    assert await Library.pending_names() == [], "settled, so no boot reschedules it"


@pytest.mark.anyio
async def test_run_maintenance_trains_the_vector_index_once_the_library_is_big_enough() -> None:
    lib = await Library.create("vec")
    index = lib.index_with(TINY)
    for part in range(2):
        await _fill(index, "a.md", part, 1000, vectors=True)
    settings = PipelineSettings(ann_min_rows=500)

    report = await maintenance.run(lib, TINY, settings)
    await Library("vec").settle_maintenance(0, report.ann_trained, report.num_rows)

    assert report.ann_trained is True and report.num_rows == 2000
    assert (await maintenance_state("vec")).vector_index_rows == 2000
    # a fresh handle: the one above is pinned to the table version it opened
    stats = await lib.index_with(TINY).stats()
    assert stats is not None and stats.has_vector_index and stats.has_fts_index
    assert (stats.vector_index_rows, stats.unindexed_rows) == (2000, 0)

    again = await maintenance.run(lib, TINY, settings)

    assert again.ann_trained is False, "the library has not doubled since it was trained"


@pytest.mark.parametrize(
    ("name", "reason"),
    [("a library nobody indexed yet", "no-table"), ("a table an older build wrote", "outdated")],
)
@pytest.mark.anyio
async def test_run_maintenance_skips_what_it_must_not_touch(name: str, reason: str) -> None:
    lib = await Library.create("skip")
    if reason == "outdated":
        _table_with(lib.index_dir, pa.schema([("doc", pa.string()), ("text", pa.string())]))

    report = await maintenance.run(lib, None, PipelineSettings())

    assert report.skipped == reason, name
    assert (report.num_rows, report.fragments_after, report.ann_trained) == (0, 0, False)


@pytest.mark.anyio
async def test_run_maintenance_of_a_deleted_library_reports_it_instead_of_raising() -> None:
    """A run may sit in the queue while the library is deleted, so it never asks for the row it
    needs before it checks that the library still exists."""
    lib = await Library.create("gone")
    await _fill(lib.index_with(None), "a.md", 0, 2)
    await lib.delete()

    report = await maintenance.run(lib, None, PipelineSettings())

    assert report.skipped == "no-library" and report.num_rows == 0
    # settles nothing, raises nothing
    await Library("gone").settle_maintenance(1, report.ann_trained, report.num_rows)
    assert await Library("gone").maintenance_state() is None, "no row, not a row of zeroes"


@pytest.mark.anyio
async def test_note_indexed_of_an_unknown_library_counts_nothing() -> None:
    assert await Library("ghost").note_indexed() == 0
    assert await Library.pending_names() == []


@pytest.mark.anyio
async def test_library_info_reports_the_index_on_demand() -> None:
    lib = await Library.create("info")
    assert (await lib.info()).index is None, "no table yet"

    await _fill(lib.index_with(None), "a.md", 0, 3)
    await Library("info").note_indexed()

    status = (await lib.info()).index
    assert status is not None
    assert (status.num_rows, status.num_fragments, status.pending_docs) == (3, 1, 1)
    assert (status.has_fts_index, status.has_vector_index) == (False, False)
    assert status.last_maintained_at is None


@pytest.mark.anyio
async def test_search_falls_back_to_fts_without_an_embedding_model(tmp_path: Path) -> None:
    """A vector query needs something to embed the question with; without a model the mode is
    downgraded rather than failing."""
    path = tmp_path / "index"
    schema = PLAIN_SCHEMA.append(pa.field("vector", pa.list_(pa.float32(), 2)))
    table = lancedb.connect(str(path)).create_table("chunks", schema=schema)
    table.add([{"doc": "a.md", "chunk_id": 0, "heading": "H", "text": "hi", "vector": [0.1, 0.2]}])
    index = LibraryIndex(path, "lib", tmp_path, None)
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


def test_hit_tolerates_rows_written_by_an_older_build(tmp_path: Path) -> None:
    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, None)
    hit = index.hit({"doc": "a.md", "chunk_id": 3, "heading": "", "text": "body"})
    assert isinstance(hit, Hit)
    assert (hit.source_path, hit.markdown_path, hit.part) == ("", "", 0)
    assert (hit.page_start, hit.page_end, hit.parents) == (None, None, [])
    assert (hit.header, hit.location) == ("", "a.md L0-0")


def test_hit_builds_a_citable_header_and_location(tmp_path: Path) -> None:
    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, None)
    hit = index.hit(
        {
            "doc": "book.pdf",
            "chunk_id": 1,
            "part": 2,
            "source_path": "library/lib/files/book.pdf",
            "markdown_path": "library/lib/markdown/book.pdf.md",
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
    monkeypatch.setattr(
        embed, "rerank_scores", lambda model, accelerator, q, ts: [float(len(t)) for t in ts]
    )
    settings = SearchSettings(reranker="cross-encoder")
    rows = [{"text": text, "_score": 9.0} for text in texts]

    ranked = await cross_encode("q", rows, settings, "cpu")

    assert [row_score(row) for row in ranked] == expected, name
    assert checked == [("reranker", settings.reranker_model)], name


@pytest.mark.anyio
async def test_an_index_with_an_embedding_stores_a_vector_column(tmp_path: Path) -> None:
    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, EmbeddingModel("test/model", 2))
    (chunk_,) = chunk.split("# H\n\nbody\n", ConversionSettings())

    row = Row(chunk=chunk_, vector=[0.1, 0.2])
    await index.add_parts("g.md", "files/g.md", "markdown/g.md.md", _aparts([(0, [row])]))

    table = await index._existing()
    assert table is not None
    assert (await table.schema()).field("vector").type == pa.list_(pa.float32(), 2)
    (record,) = (await table.to_arrow()).to_pylist()
    assert (record["doc"], record["heading"], record["part"]) == ("g.md", "H", 0)
    assert record["vector"] == pytest.approx([0.1, 0.2])
    assert await index.schema_current() is True


@pytest.mark.anyio
async def test_ensure_preview_builds_once_when_two_readers_arrive_together() -> None:
    """B8: the second reader waits on the per-document stripe lock and then finds the stored row."""
    from haskie import library as library_module

    lib = await Library.create("race")
    doc = await lib.save("g.md", MD.encode())
    real = convert.build_preview
    # threading events, because the gate below is held in the worker thread `cpu.on_cpu` runs the
    # build in; `reader_ready` is awaited on the loop instead, so it is an anyio one
    building, release = threading.Event(), threading.Event()
    reader_ready = anyio.Event()
    builds: list[str] = []
    results: list[object] = []

    def gated(*args, **kwargs):
        builds.append(args[0].name)
        building.set()
        assert release.wait(timeout=30)
        return real(*args, **kwargs)

    async def first_reader() -> None:
        results.append(await lib.ensure_preview(doc.name))

    async def second_reader() -> None:
        await lib.document(doc.name)  # the row is readable while the first reader holds the lock
        reader_ready.set()
        results.append(await lib.ensure_preview(doc.name))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(convert, "build_preview", gated)
        async with anyio.create_task_group() as readers:
            readers.start_soon(first_reader)
            await anyio.to_thread.run_sync(building.wait)  # the first build is under way
            readers.start_soon(second_reader)
            await reader_ready.wait()
            stripe = library_module._preview_lock(lib.name, doc.name)
            while stripe.statistics().tasks_waiting == 0:  # the second reader is on the lock
                await anyio.sleep(0.01)
            release.set()

    assert builds == ["g.md"], "one build, whichever reader got there first"
    assert len(results) == 2 and all(r.preview is not None for r in results)  # ty: ignore


# --- pipeline ----------------------------------------------------------------------


@pytest.mark.anyio
async def test_plan_embed_requires_a_converted_document() -> None:
    lib = await Library.create("plans")
    doc = await lib.save("g.md", MD.encode())
    with pytest.raises(FileNotFoundError, match="markdown parts missing"):
        await pipeline.plan_embed(lib, doc.name)


@pytest.mark.anyio
async def test_plan_index_requires_embedded_rows() -> None:
    lib = await Library.create("plans")
    doc = await lib.save("g.md", MD.encode())
    settings = await lib.effective_settings()
    (batch,) = await pipeline.plan_convert(lib, doc.name, 10)
    await pipeline.convert_batch(lib, doc.name, batch, settings)
    await pipeline.finalize_convert(lib, doc.name, [batch], 0, settings)

    with pytest.raises(FileNotFoundError, match=r"rows missing for parts \[0\]"):
        await pipeline.plan_index(lib, doc.name, 50)


@pytest.mark.anyio
async def test_convert_and_embed_write_atomically() -> None:
    lib = await Library.create("atomic")
    doc = await lib.save("g.md", MD.encode())
    settings = await lib.effective_settings()
    (batch,) = await pipeline.plan_convert(lib, doc.name, 10)

    assert await pipeline.convert_batch(lib, doc.name, batch, settings) == 0, (
        "no OCR pages in markdown"
    )
    await pipeline.finalize_convert(lib, doc.name, [batch], 0, settings)
    chunks = await pipeline.embed_batch(lib, doc.name, batch, settings, None)

    assert chunks == len(chunk.split(MD, settings))
    assert lib.markdown_path(doc.name).read_text() == MD
    assert list(lib.parts_dir(doc.name).glob("*.tmp")) == [], "no temp file left behind"
    assert not lib.markdown_path(doc.name).with_suffix(".md.tmp").exists()


@pytest.mark.anyio
async def test_convert_batch_of_a_pdf_reports_pages_needing_ocr() -> None:
    lib = await Library.create("ocr")
    doc = await lib.save("scan.pdf", text_pdf(["text page", None]))
    settings = await lib.effective_settings()
    batches = await pipeline.plan_convert(lib, doc.name, 10)

    assert await pipeline.convert_batch(lib, doc.name, batches[0], settings) == 1
    assert "needs OCR, skipped" in lib.part_path(doc.name, 0).read_text()


# --- sessions ----------------------------------------------------------------------

# How long one library of a fan-out may wait for the other before the test calls it sequential.
CONCURRENT_SEARCH_SECONDS = 5.0


@pytest.mark.parametrize(
    ("name", "session_id", "libraries", "error", "match"),
    [
        ("empty id", "", ["a"], InvalidInput, "session id must be 1..128"),
        ("id too long", "x" * 129, ["a"], InvalidInput, "session id must be 1..128"),
        (
            "more libraries than the cap",
            "s",
            [f"l{i}" for i in range(101)],
            InvalidInput,
            "at most 100 libraries",
        ),
        ("unknown library", "s", ["ghost"], LibraryNotFound, "library not found: ghost"),
    ],
)
@pytest.mark.anyio
async def test_set_libraries_rejects(
    name: str, session_id: str, libraries: list[str], error: type[Exception], match: str
) -> None:
    await Library.create("a")
    with pytest.raises(error, match=match):
        await session.set_libraries(session_id, libraries)
    assert await session.load() == {}, f"nothing stored for a rejected request: {name}"


@pytest.mark.anyio
async def test_set_libraries_deduplicates_and_keeps_order() -> None:
    for name in ("a", "b"):
        await Library.create(name)
    assert await session.set_libraries("s1", ["b", "a", "b"]) == ["b", "a"]
    assert await session.load() == {"s1": ["b", "a"]}
    assert await session.libraries_for("s1") == ["b", "a"]
    assert await session.libraries_for("unknown") == []


@pytest.mark.anyio
async def test_session_search_skips_a_library_that_disappeared(caplog, monkeypatch) -> None:
    """Deleting a library drops it from every session (one cascade), so a name without a row can
    only come from a delete between the two reads of the search. It is skipped, not raised."""

    async def nothing_found(names: list[str]) -> dict:
        return {}

    await Library.create("ghost")
    await session.set_libraries("s1", ["ghost"])
    monkeypatch.setattr(Library, "load_settings", staticmethod(nothing_found))

    with caplog.at_level("WARNING"):
        assert await session.search("s1", "anything") == []
    assert _events(caplog) == ["session_library_missing"]


@pytest.mark.anyio
async def test_session_search_reads_its_libraries_concurrently(monkeypatch) -> None:
    """Two libraries, two events: each retrieval announces itself and then waits for the other, so
    the search can only answer at all if the fan-out overlapped. A sequential fan-out would hold
    the first retrieval until the wait times out, and the timeout fails the search."""
    import asyncio

    for name in ("a", "b"):
        await Library.create(name)
    await session.set_libraries("s1", ["a", "b"])
    arrived = {"a": asyncio.Event(), "b": asyncio.Event()}

    async def paired(self, query, vector, settings_, limit) -> list[dict]:
        arrived[self.library].set()
        other = arrived["b" if self.library == "a" else "a"]
        await asyncio.wait_for(other.wait(), CONCURRENT_SEARCH_SECONDS)
        return []

    monkeypatch.setattr(LibraryIndex, "search_rows", paired)

    assert await session.search("s1", "anything") == []
    assert all(event.is_set() for event in arrived.values()), "both libraries were read"


# --- storage -----------------------------------------------------------------------


def test_migrations_apply_once_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "m.db"
    conn = sqlite3.connect(str(path))
    assert db.migrate(conn) == len(db.MIGRATIONS)
    assert conn.execute("pragma user_version").fetchone() == (len(db.MIGRATIONS),)
    assert db.migrate(conn) == len(db.MIGRATIONS), "re-run is a no-op"

    # a DB behind by one version only gets the tail applied
    monkeypatch.setattr(db, "MIGRATIONS", [*db.MIGRATIONS, "create table extra (x);"])
    assert db.migrate(conn) == len(db.MIGRATIONS)
    assert conn.execute("select count(*) from extra").fetchone() == (0,)
    conn.close()


@pytest.mark.anyio
async def test_connect_rolls_back_a_failed_unit_of_work() -> None:
    with pytest.raises(sqlite3.IntegrityError):
        async with db.connect() as conn:
            await conn.execute("insert into libraries (name) values ('half')")
            await conn.execute(
                "insert into documents (library, name, size) values ('ghost', 'a', 1)"
            )
    assert await Library.names() == [], "the first insert of the failed block is gone too"


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
        embed._cross_encoder("Xenova/ms-marco-MiniLM-L-6-v2", "auto")
    finally:
        embed._build_cross_encoder.cache_clear()
    assert seen == [["CPUExecutionProvider"]]


def test_embedding_helpers_short_circuit_on_empty_input() -> None:
    """No text means no model, so neither call may download anything."""
    assert embed.embed_texts(COMPACT, []) == []
    assert embed.rerank_scores("Xenova/ms-marco-MiniLM-L-6-v2", "cpu", "q", []) == []


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
    audit.attach(library="notes")  # no record in progress
    assert not audit.path().exists(), "nothing written outside an audited call"


# --- S2: grouped index writes ------------------------------------------------------
#
# One LanceDB commit per group of parts instead of one per micro-batch, and the parts directory
# dropped once a document is indexed.


async def _fragments(index: LibraryIndex) -> int:
    """Data fragments of the table: LanceDB writes one per commit that carries rows."""
    table = await index._existing()
    if table is None:
        return 0
    # lancedb annotates stats() as a dataclass but returns plain dicts
    return (await table.stats())["fragment_stats"]["num_fragments"]  # ty: ignore[not-subscriptable]


async def _indexed_rows(lib: Library) -> int:
    table = await (await lib.index())._existing()
    return 0 if table is None else await table.count_rows()


@pytest.mark.anyio
async def test_add_parts_writes_one_fragment_for_many_parts(tmp_path: Path) -> None:
    """Three parts, one commit, one fragment. An empty part inside the group writes nothing but
    does not break the group."""
    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, None)
    chunks = chunk.split(MD, SMALL)
    parts = [(0, [Row(chunk=c) for c in chunks]), (1, []), (2, [Row(chunk=c) for c in chunks])]

    written = await index.add_parts("g.md", "files/g.md", "markdown/g.md.md", _aparts(parts))

    table = await index._existing()
    assert table is not None
    assert written == await table.count_rows() == 2 * len(chunks)
    assert await _fragments(index) == 1, "one commit, however many parts it carried"
    records = (await table.to_arrow()).to_pylist()
    assert sorted({r["part"] for r in records}) == [0, 2], "the empty part is skipped"
    assert {r["chunk_id"] for r in records} == set(range(len(chunks))), "ids restart per part"
    assert {r["markdown_path"] for r in records} == {"markdown/g.md.md"}


@pytest.mark.anyio
async def test_delete_parts_removes_only_the_range(tmp_path: Path) -> None:
    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, None)
    (chunk_,) = chunk.split("# H\n\nbody\n", ConversionSettings())
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
async def test_index_batch_group_is_idempotent() -> None:
    """A replay after a crash between the LanceDB commit and the step checkpoint must rewrite the
    group rather than append it a second time."""
    lib = await Library.create("groups")
    doc = await lib.save("p.pdf", text_pdf(["alpha one", "beta two", "gamma three"]))
    settings = await lib.effective_settings()
    converts = await pipeline.plan_convert(lib, doc.name, 1)
    for batch in converts:
        await pipeline.convert_batch(lib, doc.name, batch, settings)
    await pipeline.finalize_convert(lib, doc.name, converts, 0, settings)
    for batch in await pipeline.plan_embed(lib, doc.name):
        await pipeline.embed_batch(lib, doc.name, batch, settings, None)

    assert [(b.seq, b.start, b.end) for b in await pipeline.plan_index(lib, doc.name, 2)] == [
        (0, 0, 2),
        (1, 2, 3),
    ], "the last group holds the remainder"
    (group,) = await pipeline.plan_index(lib, doc.name, 50)
    assert (group.seq, group.start, group.end) == (0, 0, 3), "three parts in one commit"

    written = await pipeline.index_batch(lib, doc.name, group, None)
    assert written == await _indexed_rows(lib) > 0
    assert await _fragments(await lib.index()) == 1

    written_again = await pipeline.index_batch(lib, doc.name, group, None)
    assert written_again == written, "the same group again"
    assert await _indexed_rows(lib) == written, "the range was replaced, not appended"


@pytest.mark.anyio
async def test_cleanup_parts_leaves_the_assembled_markdown_and_tolerates_a_missing_directory() -> (
    None
):
    lib = await Library.create("tidy")
    doc = await lib.save("g.md", MD.encode())
    settings = await lib.effective_settings()
    (batch,) = await pipeline.plan_convert(lib, doc.name, 10)
    await pipeline.convert_batch(lib, doc.name, batch, settings)
    await pipeline.finalize_convert(lib, doc.name, [batch], 0, settings)
    await pipeline.embed_batch(lib, doc.name, batch, settings, None)

    await pipeline.cleanup_parts(lib, doc.name)

    assert not lib.parts_dir(doc.name).exists()
    assert lib.markdown_path(doc.name).read_text() == MD
    await pipeline.cleanup_parts(lib, doc.name)  # nothing left to remove, and no raise


# --- document timestamps and counts (paged listings) -------------------------------


@pytest.mark.anyio
async def test_set_status_and_save_bump_updated_at() -> None:
    """`updated_at` is what the "recently touched" sort reads, so only a change to the document
    may move it: a lifecycle step and a re-upload do, building the preview does not."""
    lib = await Library.create("stamps")

    first = await lib.save("a.md", MD.encode())
    assert first.created_at > 0, "stamped on upload"
    assert first.updated_at == first.created_at, "an upload is the document's first change"

    await lib.set_status("a.md", "indexed")
    indexed = await lib.document("a.md")
    assert indexed.updated_at > first.updated_at, "a lifecycle step is a change"
    assert (indexed.status, indexed.error) == ("indexed", None), "and it is the status it set"
    assert indexed.created_at == first.created_at, "the upload moment never moves"

    await lib.set_status("a.md", "error", "boom")
    failed = await lib.document("a.md")
    assert (failed.status, failed.error) == ("error", "boom"), "the reason is stored with it"
    assert failed.updated_at >= indexed.updated_at

    await lib.ensure_preview("a.md")
    previewed = await lib.document("a.md")
    assert previewed.preview is not None, "the preview was built"
    assert previewed.updated_at == failed.updated_at, "filling in the preview is not a change"

    again = await lib.save("a.md", (MD + "more\n").encode())
    assert again.updated_at > indexed.updated_at, "writing over the document is"
    assert again.created_at == first.created_at, "the first upload still owns created_at"
    assert (again.status, again.error, again.preview) == ("uploaded", None, None), "a fresh row"


@pytest.mark.anyio
async def test_document_counts_group_by_status() -> None:
    from haskie.library import DocumentCounts

    lib = await Library.create("counts")
    for name in ("a.md", "b.md", "c.md", "d.md"):
        await lib.save(name, MD.encode())
    await lib.set_status("a.md", "indexed")
    await lib.set_status("b.md", "embedding")
    await lib.set_status("c.md", "error", "boom")

    counts = await lib.counts()

    assert counts.total == 4
    assert counts.indexed == 1
    assert counts.active == 1, "embedding is in the pipeline, uploaded is not"
    assert counts.error == 1
    assert counts.by_status == {"indexed": 1, "embedding": 1, "error": 1, "uploaded": 1}
    empty = await Library.create("empty")
    assert await empty.counts() == DocumentCounts(), "a library without documents"


# --- S3: cross-library search ------------------------------------------------------
#
# `LibraryIndex` splits search into retrieval (`search_rows`), the query embedding (`query_vector`)
# and row-to-Hit (`hit`), so a session embeds once, fans out and rescores once. `session.rrf_merge`
# fuses the per-library rankings by rank, because two indexes do not score on the same scale.


async def _fts_index(path: Path, texts: list[str]) -> LibraryIndex:
    """A small full-text index: one chunk per text, written and finished like the pipeline does."""
    index = LibraryIndex(path, "lib", path.parent, None)
    chunks = [c for text in texts for c in chunk.split(f"# H\n\n{text}\n", ConversionSettings())]
    assert len(chunks) == len(texts), "one chunk per text, or the row counts below mean nothing"
    rows = [Row(chunk=c) for c in chunks]
    await index.add_parts("d.md", "files/d.md", "markdown/d.md.md", _aparts([(0, rows)]))
    await index.finish()
    return index


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
    merged = session.rrf_merge(ranked, k=60)

    assert [item for item, _ in merged] == [item for item, _ in expected], name
    assert [score for _, score in merged] == pytest.approx([s for _, s in expected]), name


@pytest.mark.anyio
async def test_search_rows_returns_raw_rows_without_cutting(tmp_path: Path) -> None:
    """Retrieval only: as many rows as the caller asked for, carrying the engine's own score and
    no cross-encoder score. `search` is what cuts to `settings.limit`."""
    index = await _fts_index(tmp_path / "index", [f"lancedb chapter {i}" for i in range(6)])
    settings = SearchSettings(limit=2, candidates=4)

    rows = await index.search_rows("lancedb", None, settings, 4)

    assert len(rows) == 4, "the fetch size wins over settings.limit"
    assert all("_score" in row and "_relevance_score" not in row for row in rows)
    assert {row["doc"] for row in rows} == {"d.md"}
    assert len(await index.search_rows("lancedb", None, settings, 100)) == 6, "no more than exist"
    assert len(await index.search("lancedb", settings)) == 2, "the composed search cuts to limit"
    missing = LibraryIndex(tmp_path / "missing", "lib", tmp_path, None)
    assert await missing.search_rows("lancedb", None, settings, 4) == [], "no table, no rows"
    assert await missing.search("lancedb", settings) == [], "and nothing to compose a search from"

    _table_with(tmp_path / "blank", PLAIN_SCHEMA)  # a table, no rows, and so no full-text index
    blank = LibraryIndex(tmp_path / "blank", "lib", tmp_path, None)
    assert await blank.search_rows("lancedb", None, settings, 4) == [], "a table with no rows"


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
    index = LibraryIndex(path, "lib", tmp_path, embedding)

    assert await index.query_vector("q", settings) == expected, name
    assert checked == ([("embedding", COMPACT.name)] if expected else []), name


@pytest.mark.anyio
async def test_query_vector_of_a_never_indexed_library_is_none(tmp_path: Path) -> None:
    """No table means no search, so the model is never asked for (it may not be loaded)."""
    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, COMPACT)
    assert await index.query_vector("q", SearchSettings(mode="hybrid")) is None
    assert await index.search("q", SearchSettings(mode="hybrid")) == []


# --- S4: full-text search across libraries -------------------------------------------
#
# `textsearch` merges raw BM25 scores, because one lexical scorer with the same tokenizer answers
# in every library. Its cursor is an opaque offset bound to the query, so every page is a cut of a
# ranking that was recomputed: the merge needs a total order, and the cursor needs to be rejected
# whenever the query it was measured against is not the one being asked now.

# The query one cursor below belongs to: (query, libraries, page_size).
TEXT_QUERY = ("lancedb", ["alpha", "beta"], 25)


def _ranked(library: str, doc: str, part: int, chunk_id: int, score: float | None) -> tuple:
    """One (index, row) pair for `merge`, which only ever reads `index.library` and the row's
    identity columns. `score` None writes no `_score` at all, as a row of an older index has."""
    index = LibraryIndex(Path("/nowhere") / library, library, Path("/nowhere"), None)
    row = {"doc": doc, "part": part, "chunk_id": chunk_id}
    if score is not None:
        row["_score"] = score
    return (index, row)


def _identity(pairs: list[tuple]) -> list[tuple[str, str, int, int]]:
    return [(index.library, row["doc"], row["part"], row["chunk_id"]) for index, row in pairs]


@pytest.mark.parametrize(
    ("name", "per_library", "expected"),
    [
        ("nothing to merge", [], []),
        (
            "a library that matched nothing contributes nothing",
            [[], [("a", "d.md", 0, 0, 1.0)]],
            [("a", "d.md", 0, 0)],
        ),
        (
            "the better score wins, whichever library it came from",
            [[("b", "d.md", 0, 0, 9.0)], [("a", "d.md", 0, 0, 1.0)]],
            [("b", "d.md", 0, 0), ("a", "d.md", 0, 0)],
        ),
        (
            "an equal score falls back to the library name",
            [[("b", "d.md", 0, 0, 1.0)], [("a", "d.md", 0, 0, 1.0)]],
            [("a", "d.md", 0, 0), ("b", "d.md", 0, 0)],
        ),
        (
            "inside one library: doc, then part, then chunk",
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
    name: str, per_library: list[list[tuple]], expected: list[tuple[str, str, int, int]]
) -> None:
    from haskie import textsearch

    merged = textsearch.merge([[_ranked(*row) for row in rows] for rows in per_library])

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
        ("another set of libraries", ("lancedb", ["alpha"], 25, 10), "another query"),
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

    q, libraries, page_size = TEXT_QUERY
    assert textsearch.parse_cursor(None, q, libraries, page_size) == 0, "no cursor, first page"
    issued = textsearch.make_cursor(q, libraries, page_size, 50)
    assert textsearch.parse_cursor(issued, q, libraries, page_size) == 50, "round trip"
    assert textsearch.make_cursor(q, ["beta", "alpha"], page_size, 50) == issued, "order-free"

    if isinstance(rejected, tuple):
        cursor = textsearch.make_cursor(*rejected)  # a cursor this module issued, for another page
    elif isinstance(rejected, dict):
        cursor = _wire_cursor(**rejected)
    else:
        cursor = rejected

    with pytest.raises(InvalidInput) as raised:
        textsearch.parse_cursor(cursor, q, libraries, page_size)
    assert detail in str(raised.value), name


@pytest.mark.parametrize(
    ("name", "libraries", "expected"),
    [
        ("no filter at all", None, None),
        ("an empty filter is not a filter", "", None),
        ("separators alone", " , ,", None),
        ("one name", "alpha", ["alpha"]),
        ("several names, trimmed", " alpha , beta ", ["alpha", "beta"]),
        ("a trailing separator", "alpha,", ["alpha"]),
    ],
)
def test_text_split_libraries(name: str, libraries: str | None, expected: list[str] | None) -> None:
    from haskie import textsearch

    assert textsearch.split_libraries(libraries) == expected, name


@pytest.mark.anyio
async def test_fts_rows_is_empty_without_an_fts_index(tmp_path: Path) -> None:
    """A library halfway through its first index has rows and no full-text index yet. A
    cross-library search must not wait for it, so it contributes nothing instead of a scan."""
    missing = LibraryIndex(tmp_path / "missing", "lib", tmp_path, None)
    assert await missing.fts_rows("lancedb", 10) == [], "never indexed, so there is no table"

    _table_with(tmp_path / "empty", PLAIN_SCHEMA)
    empty = LibraryIndex(tmp_path / "empty", "lib", tmp_path, None)
    assert await empty.fts_rows("lancedb", 10) == [], "a table with no rows in it"

    index = LibraryIndex(tmp_path / "index", "lib", tmp_path, None)
    chunks = chunk.split("# H\n\nlancedb chapter one\n", ConversionSettings())
    rows = [Row(chunk=c) for c in chunks]
    await index.add_parts("d.md", "files/d.md", "markdown/d.md.md", _aparts([(0, rows)]))
    assert await index.has_index("text") is False, "written, not indexed: the state under test"
    assert await index.fts_rows("lancedb", 10) == [], "rows are there, the full-text index is not"

    await index.finish()

    (row,) = await index.fts_rows("lancedb", 10)
    assert (row["doc"], row["chunk_id"]) == ("d.md", 0)
    assert row["_score"] > 0, "raw BM25, which is what the cross-library merge sorts on"


# --- sessions and previews under load (P5) -----------------------------------------


@pytest.mark.anyio
async def test_session_libraries_keep_their_order_and_survive_reorder() -> None:
    """The selection is rows with a position, not a JSON list: reordering it rewrites the rows,
    and a session that selected nothing is still a session."""
    for name in ("a", "b", "c"):
        await Library.create(name)

    await session.set_libraries("s1", ["c", "a", "b"])
    assert await session.libraries_for("s1") == ["c", "a", "b"]

    assert await session.set_libraries("s1", ["b", "c"]) == ["b", "c"], "the selection is replaced"
    assert await session.libraries_for("s1") == ["b", "c"]
    assert await session.load() == {"s1": ["b", "c"]}
    async with db.connect() as conn:
        cursor = await conn.execute(
            "select library, position from session_libraries where session_id = 's1' "
            "order by position"
        )
        rows = await cursor.fetchall()
    assert rows == [("b", 0), ("c", 1)], "one row per library, positions renumbered from zero"

    await session.set_libraries("s1", [])
    assert await session.load() == {"s1": []}, "an empty selection keeps the session itself"


@pytest.mark.parametrize(("name", "workers"), [("one at a time", 1), ("two at a time", 2)])
@pytest.mark.anyio
async def test_ensure_preview_bounds_concurrent_builds(
    monkeypatch: pytest.MonkeyPatch, name: str, workers: int
) -> None:
    """Four readers open four different documents at once; only `preview_workers` parses run.

    The stripe lock is per document, so nothing but the semaphore holds these four apart.
    """
    from haskie import cpu
    from haskie import library as library_module

    # a build holds a CPU slot too, so the budget must not be the ceiling under test here
    monkeypatch.setattr(cpu, "_cpu_slots", threading.BoundedSemaphore(4))
    lib = await Library.create("burst")
    names = [(await lib.save(f"doc-{i}.md", MD.encode())).name for i in range(4)]
    entered, release, counted = threading.Semaphore(0), threading.Event(), threading.Lock()
    live, peak, builds = 0, 0, []
    real = convert.build_preview

    def gated(*args, **kwargs):
        nonlocal live, peak
        with counted:
            live += 1
            peak = max(peak, live)
            builds.append(args[0].name)
        entered.release()
        assert release.wait(timeout=30)
        try:
            return real(*args, **kwargs)
        finally:
            with counted:
                live -= 1

    library_module.configure_preview_slots(workers)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(convert, "build_preview", gated)
            async with anyio.create_task_group() as readers:
                for doc in names:
                    readers.start_soon(lib.ensure_preview, doc)
                for _ in range(workers):  # every slot of the pool is now inside a build
                    await anyio.to_thread.run_sync(entered.acquire)
                assert library_module._preview_slots.value == 0, f"{name}: no slot left"
                release.set()
    finally:
        library_module.configure_preview_slots(PipelineSettings().preview_workers)

    assert sorted(builds) == sorted(names), f"{name}: every document was built, once"
    assert peak == workers, f"{name}: never more parses at once than the pool admits"


@pytest.mark.anyio
async def test_ensure_preview_returns_not_ready_when_the_queue_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader that waited out `PREVIEW_WAIT_SECONDS` is told to retry (503) rather than holding
    its request open until the burst clears."""
    from haskie import library as library_module
    from haskie.errors import NotReady

    lib = await Library.create("queued")
    doc = await lib.save("g.md", MD.encode())
    monkeypatch.setattr(library_module, "PREVIEW_WAIT_SECONDS", 0.05)
    slots = anyio.Semaphore(1)
    monkeypatch.setattr(library_module, "_preview_slots", slots)
    await slots.acquire()  # the test holds the only slot, so every reader waits it out
    try:
        with pytest.raises(NotReady, match="preview queue is full"):
            await lib.ensure_preview(doc.name)
    finally:
        slots.release()

    document = await lib.document(doc.name)
    assert document.preview is None, "nothing was built and nothing was stored"
