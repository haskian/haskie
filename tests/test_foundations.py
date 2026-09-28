"""Foundations: typed errors, atomic writes, migrations, the CPU budget, settings, audit trail."""

import asyncio
import hashlib
import os
import sqlite3
import stat
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import anyio
import msgspec
import pytest
from sqlalchemy import func, insert, select, text, update

from haskie import audit, cpu, db, errors, home, settings, tables
from haskie.audit import Actor, Outcome
from haskie.collection.collection import Collection
from haskie.document import document
from haskie.document.document import Document, DocumentStatus
from haskie.indexing import embed_cache
from haskie.indexing.chunk import Chunk, Piece
from haskie.indexing.segment import CutReason, PieceType
from haskie.settings import (
    MAX_SCAN,
    Chunker,
    ChunkSettings,
    CollectionOverrides,
    ConversionSettings,
    Parser,
    PipelineSettings,
    Reranker,
    RetentionSettings,
    SearchOverrides,
    SearchSettings,
    UserSettings,
    load_user_settings_or_none,
    save_user_settings,
    settings_problem,
    without_none,
)

from conftest import audit_lines, events, forget_settings  # isort: skip

# --- errors -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "template", "expected"),
    [
        (
            "haskie home -> symbolic root",
            "read {home}/documents/a.md failed",
            "read $HASKIE_HOME/documents/a.md failed",
        ),
        ("user home -> tilde", "read {user}/Downloads/a.md failed", "read ~/Downloads/a.md failed"),
        ("both in one line", "{home} and {user}", "$HASKIE_HOME and ~"),
        ("no path -> unchanged", "plain message", "plain message"),
    ],
)
def test_scrub_replaces_absolute_paths(name: str, template: str, expected: str) -> None:
    text = template.format(home=home.HOME, user=Path.home())
    assert home.scrub(text) == expected, name


def test_invalid_input_is_also_a_value_error() -> None:
    """msgspec wraps only a ValueError raised in `__post_init__` as a decode-time error."""
    assert issubclass(errors.InvalidInput, ValueError)


# --- the home layout ---------------------------------------------------------------

DOC = "guide.md"


def test_shard_hashes_the_utf8_bytes_of_the_name() -> None:
    """A name is text, a hash is bytes: the encoding is pinned so the shard never depends on the
    platform or the locale."""
    name = "résumé.md"

    assert home.shard(name) == hashlib.sha1(name.encode("utf-8")).hexdigest()[:2]
    assert len(home.shard(name)) == 2
    assert home.shard(name) == home.shard(name), "the same name always lands in one place"


def test_shard_spreads_names_over_the_whole_byte() -> None:
    """One directory per name would make every listing pay for every document, so the point of
    the shard is the spread: a thousand names have to use most of the 256 directories."""
    shards = {home.shard(f"doc-{i}.md") for i in range(1000)}

    assert len(shards) > 200, "a thousand names fall into more than 200 of the 256 shards"
    assert all(len(prefix) == 2 and int(prefix, 16) >= 0 for prefix in shards), "two hex digits"


def test_every_document_path_sits_under_the_same_shard() -> None:
    """A document owns one folder: the upload, the markdown, the parts, the preview and the
    embedding cache are all inside it, so one `remove_tree` deletes everything it owns."""
    doc = Document(name=DOC, suffix=".md", size=1, status=DocumentStatus.IMPORTED)
    root = home.DOCUMENT_ROOT / home.shard(DOC) / DOC

    assert document.root(DOC) == root
    assert doc.root == root
    assert doc.original == root / "original.md"
    assert doc.markdown == root / "original.md.md"
    assert doc.parts_dir == root / "parts"
    assert doc.preview_dir == root / "preview"
    assert doc.embeddings_dir == root / "embeddings"
    assert doc.part_path(7) == doc.parts_dir / "000007.md"
    assert embed_cache.file_path(DOC, "abc").parent == doc.embeddings_dir
    assert {path.parent.parent for path in (doc.original, doc.markdown)} == {root.parent}


def test_part_numbers_are_wide_enough_for_a_long_document() -> None:
    doc = Document(name=DOC, suffix=".md", size=1, status=DocumentStatus.IMPORTED)

    assert home.PART_DIGITS == 6, "four digits would cap a document at ten thousand parts"
    assert doc.part_path(0).name == "000000.md"
    assert doc.part_path(123456).name == "123456.md"


def test_a_collection_is_sharded_by_its_own_name() -> None:
    collection = Collection("notes")
    root = home.COLLECTION_ROOT / home.shard("notes") / "notes"

    assert collection.root == root
    assert collection.index_dir == root / "index"


@pytest.mark.anyio
async def test_a_document_lands_in_its_shard_and_is_removed_from_it(tmp_path) -> None:
    source = tmp_path / "incoming" / DOC
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# A\n")

    doc = await document.import_path(str(source))

    assert doc.original.read_text() == "# A\n"
    assert doc.source_path() == doc.original
    assert [p.name for p in home.DOCUMENT_ROOT.iterdir()] == [home.shard(DOC)]

    await document.remove_files(doc.name)

    assert not doc.root.exists(), "the whole folder goes, not only the upload"


# --- home.atomic_write ------------------------------------------------------------

WRITERS = ["atomic_write", "atomic_write_sync"]  # the async one and its worker-thread twin


async def _write(writer: str, path: Path, data: bytes | str) -> None:
    """Call whichever of the two writers this case is about, the way its callers do."""
    if writer == "atomic_write":
        await home.atomic_write(path, data)
    else:
        await anyio.to_thread.run_sync(home.atomic_write_sync, path, data)


@pytest.mark.anyio
@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize(
    ("name", "data", "existing", "expected"),
    [
        ("str payload on a new file", "hello", None, b"hello"),
        ("bytes payload on a new file", b"\x00\x01", None, b"\x00\x01"),
        ("replaces existing content", "new", b"old and longer", b"new"),
    ],
)
async def test_atomic_write(
    tmp_path: Path, writer: str, name: str, data, existing, expected: bytes
) -> None:
    target = tmp_path / "out.bin"
    if existing is not None:
        target.write_bytes(existing)
    await _write(writer, target, data)
    assert target.read_bytes() == expected, f"{writer}: {name}"
    assert list(tmp_path.glob("*.tmp")) == [], "temp file removed by the rename"


@pytest.mark.anyio
@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize(
    ("name", "in_missing_directory", "replace_fails"),
    [
        ("the payload cannot be written: no parent directory", True, False),
        ("the rename fails", False, True),
    ],
)
async def test_atomic_write_leaves_no_temp_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: str,
    name: str,
    in_missing_directory: bool,
    replace_fails: bool,
) -> None:
    target = tmp_path / ("sub/out.txt" if in_missing_directory else "out.txt")
    if replace_fails:

        def refuse(src, dst) -> None:
            raise OSError("no rename today")

        monkeypatch.setattr(os, "replace", refuse)

    with pytest.raises(OSError):  # FileNotFoundError is one
        await _write(writer, target, "x")

    assert list(tmp_path.glob("**/*.tmp")) == [], f"{writer}: {name}"
    assert not target.exists(), f"{writer}: the reader still sees no file at all"


@pytest.mark.anyio
@pytest.mark.parametrize("writer", WRITERS)
async def test_atomic_write_puts_the_bytes_on_disk_before_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
) -> None:
    """A rename can reach the disk before the bytes it names, and a power cut then leaves an empty
    file where the old one was. So the whole payload is flushed first."""
    target = tmp_path / "preview.md"
    target.write_text("# Old\n")
    steps: list[tuple[str, int]] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(descriptor: int) -> None:
        steps.append(("fsync", os.fstat(descriptor).st_size))
        real_fsync(descriptor)

    def replace(source: Path, destination: Path) -> None:
        steps.append(("replace", Path(source).stat().st_size))
        real_replace(source, destination)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    await _write(writer, target, "# A longer title\n")

    size = len("# A longer title\n")
    assert steps == [("fsync", size), ("replace", size)], f"{writer}: every byte, then the rename"
    assert target.read_text() == "# A longer title\n", writer


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "already_there"),
    [
        ("a home nobody has made yet", False),
        ("a home an earlier boot already made", True),
    ],
)
async def test_ensure_home_creates_private_directories(name: str, already_there: bool) -> None:
    """The four roots a home is: collections, documents, the staging area uploads wait in, and
    the audit trail. All private to the user running the app."""
    roots = (home.COLLECTION_ROOT, home.DOCUMENT_ROOT, home.STAGING_ROOT, home.AUDIT_DIR)
    if not already_there:
        for directory in roots:
            directory.rmdir()  # made by the test home fixture, still empty

    assert await home.ensure_home() == home.HOME, name

    for directory in roots:
        assert directory.is_dir(), name
        assert directory.parent == home.HOME, name
        assert stat.S_IMODE(directory.stat().st_mode) == home.DIR_MODE, name


@pytest.mark.parametrize(
    ("name", "configured", "expected"),
    [
        ("a relative home is anchored where the process starts", "data", "{cwd}/data"),
        ("a home under ~ is expanded", "~/haskie-home", "{user}/haskie-home"),
    ],
)
def test_the_home_from_the_environment_is_absolute(
    tmp_path: Path, name: str, configured: str, expected: str
) -> None:
    """`litestar run` hands `HASKIE_HOME` over as written, without the CLI's resolving. A relative
    home would follow every later change of directory, and `scrub` would rewrite the bare word
    `data` in every log line. The module reads it once, at import, so a fresh interpreter does."""
    shown = subprocess.run(
        [sys.executable, "-c", "from haskie import home; print(home.HOME)"],
        cwd=tmp_path,
        env={**os.environ, "HASKIE_HOME": configured},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    wanted = expected.format(cwd=tmp_path.resolve(), user=Path.home())
    assert shown == str(Path(wanted).resolve()), name


# --- the holder line in the home lock ---------------------------------------------

HOLDER_ADDRESS = "http://127.0.0.1:8452"


@pytest.mark.parametrize(
    ("name", "left_behind", "written_while_held", "expected_pid"),
    [
        (
            "a longer line an earlier holder left is cut to ours",
            "pid 4194304, http://127.0.0.1:65535 (an older build said more)",
            None,
            "ours",
        ),
        ("a line that is not ours reads as no pid", None, "held, but not by haskie", None),
        ("a pid that is not at the start reads as no pid", None, "holder pid 4194304", None),
    ],
)
def test_the_holder_line(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    left_behind: str | None,
    written_while_held: str | None,
    expected_pid: str | None,
) -> None:
    """`stop` signals the pid it reads off the lock, so the line must be exactly the one
    `claim_home` wrote: a stale tail or a foreign line must never pass for a pid to signal."""
    if left_behind is not None:
        home.LOCK_FILE.write_text(left_behind)
    monkeypatch.setenv(home.ADDRESS_ENV, HOLDER_ADDRESS)

    home.claim_home()
    try:
        if written_while_held is None:
            assert home.LOCK_FILE.read_text() == f"pid {os.getpid()}, {HOLDER_ADDRESS}", name
        else:
            home.LOCK_FILE.write_text(written_while_held)  # the lock is advisory: this is allowed
        pid = home.running_pid()
    finally:
        home.release_home()

    assert pid == (os.getpid() if expected_pid == "ours" else None), name


# --- home.remove_tree -------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "entries"),
    [
        ("a tree of files and subdirectories", ["a.md", "sub/b.md"]),
        ("an empty directory", []),
        ("a path that was never there", None),
    ],
)
async def test_remove_tree_leaves_nothing_behind(
    tmp_path: Path, name: str, entries: list[str] | None
) -> None:
    target = tmp_path / "tree"
    if entries is not None:
        target.mkdir()
        for entry in entries:
            path = target / entry
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")

    await home.remove_tree(target)  # a missing path is not an error

    assert not target.exists(), name


@pytest.mark.anyio
async def test_remove_tree_reports_a_file_it_cannot_delete(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    protected = tmp_path / "locked"
    protected.mkdir()
    (protected / "a.md").write_text("x")
    protected.chmod(0o500)  # read + execute: the entry cannot be unlinked
    try:
        with caplog.at_level("WARNING"):
            await home.remove_tree(protected)

        assert protected.exists(), "the failure is reported, not silently swallowed"
        assert events(caplog) == ["remove_failed", "remove_failed"], "file, then directory"
    finally:
        protected.chmod(0o700)


# --- settings validation ----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "build", "match"),
    [
        ("chunk_size below 1", lambda: ConversionSettings(chunk_size=0), "chunk_size must be >= 1"),
        (
            "chunk_merge_below below 0",
            lambda: ConversionSettings(chunk_merge_below=-1),
            "chunk_merge_below must be 0 to 100",
        ),
        (
            "collection override chunk_merge_below above 100",
            lambda: CollectionOverrides(chunk_merge_below=101),
            "chunk_merge_below must be 0 to 100",
        ),
        (
            "cpu budget below 1",
            lambda: PipelineSettings(cpu_budget=0),
            "cpu_budget must be >= 1",
        ),
        (
            "converting weight below 1",
            lambda: PipelineSettings(converting_weight=0),
            "converting_weight must be >= 1",
        ),
        (
            "embedding weight below 1",
            lambda: PipelineSettings(embedding_weight=0),
            "embedding_weight must be >= 1",
        ),
        (
            "indexing weight below 1",
            lambda: PipelineSettings(indexing_weight=0),
            "indexing_weight must be >= 1",
        ),
        (
            "document_parallelism below 0",
            lambda: PipelineSettings(document_parallelism=-1),
            "document_parallelism must be >= 0",
        ),
        (
            "batch_pages below 1",
            lambda: PipelineSettings(batch_pages=0),
            "batch_pages must be >= 1",
        ),
        (
            "index_group_parts below 1",
            lambda: PipelineSettings(index_group_parts=0),
            "index_group_parts must be >= 1",
        ),
        (
            "task_timeout_seconds below 1",
            lambda: PipelineSettings(task_timeout_seconds=0),
            "task_timeout_seconds must be >= 1",
        ),
        ("limit below 1", lambda: SearchSettings(limit=0), "limit must be >= 1"),
        (
            "limit over the scan depth",
            lambda: SearchOverrides(limit=MAX_SCAN + 1).resolve(SearchSettings()),
            f"limit must be at most {MAX_SCAN}",
        ),
        ("candidates below 1", lambda: SearchSettings(candidates=0), "candidates must be >= 1"),
        ("rrf_k below 1", lambda: SearchSettings(rrf_k=0), "rrf_k must be >= 1"),
        (
            "negative vector weight",
            lambda: SearchSettings(vector_weight=-0.1),
            "vector_weight must be >= 0",
        ),
        (
            "negative bm25 weight",
            lambda: SearchSettings(bm25_weight=-1.0),
            "bm25_weight must be >= 0",
        ),
        (
            "collection override size alone below 1",
            lambda: CollectionOverrides(chunk_size=0),
            "chunk_size must be >= 1",
        ),
        (
            "chunk settings size below 1",
            lambda: ChunkSettings(chunk_size=0),
            "chunk_size must be >= 1",
        ),
        (
            "search override resolves into an invalid value",
            lambda: SearchOverrides(limit=0).resolve(SearchSettings()),
            "limit must be >= 1",
        ),
        (
            "retention days below 1",
            lambda: RetentionSettings(operation_days=0),
            "days must be >= 1",
        ),
        (
            "preview workers below 1",
            lambda: PipelineSettings(preview_workers=0),
            "preview_workers must be >= 1",
        ),
        (
            "negative audit retention",
            lambda: RetentionSettings(audit_days=-1),
            "audit_days must be >= 0",
        ),
        (
            "negative search retention",
            lambda: RetentionSettings(search_days=-1),
            "search_days must be >= 0",
        ),
    ],
)
def test_settings_reject_out_of_bounds(name: str, build, match: str) -> None:
    with pytest.raises(errors.InvalidInput, match=match):
        build()


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("defaults", UserSettings),
        ("the smallest size", lambda: ChunkSettings(chunk_size=1)),
        ("zero weights allowed", lambda: SearchSettings(vector_weight=0.0, bm25_weight=0.0)),
        (
            "collection override size alone: the user merge share is of any size",
            lambda: CollectionOverrides(chunk_size=1).resolve(UserSettings()),
        ),
        ("chunk settings defaults", ChunkSettings),
        ("merging turned off", lambda: ChunkSettings(chunk_merge_below=0)),
        ("merging every paragraph that fits", lambda: ChunkSettings(chunk_merge_below=100)),
        ("the shortest operation history", lambda: RetentionSettings(operation_days=1)),
        ("audit retention of zero keeps everything", lambda: RetentionSettings(audit_days=0)),
        ("search retention of zero keeps every search", lambda: RetentionSettings(search_days=0)),
        ("one preview builder", lambda: PipelineSettings(preview_workers=1)),
        ("one slice per document", lambda: PipelineSettings(document_parallelism=1)),
        (
            "document parallelism follows the stage's share of the budget",
            lambda: PipelineSettings(document_parallelism=0),
        ),
        ("a budget of one, shared equally", lambda: PipelineSettings(1, 1, 1, 1)),
    ],
)
def test_settings_accept_valid_values(name: str, build) -> None:
    assert build() is not None, name


def test_task_timeout_default_and_docs() -> None:
    assert PipelineSettings().task_timeout_seconds == 600
    doc = settings.docs()["pipeline.task_timeout_seconds"]
    assert doc.title and doc.description


def test_document_parallelism_default_and_docs() -> None:
    assert PipelineSettings().document_parallelism == 0, "as many slices as the stage has workers"
    doc = settings.docs()["pipeline.document_parallelism"]
    assert doc.title == "Parallel tasks per document" and doc.description


def test_cpu_budget_defaults_and_docs() -> None:
    """Half the machine by default, so other work keeps the rest; the weights only say how that
    budget is shared out when every stage has work."""
    cores = os.cpu_count() or 2
    indexing = PipelineSettings()
    assert indexing.cpu_budget == max(1, cores // 2)
    weights = (indexing.converting_weight, indexing.embedding_weight, indexing.indexing_weight)
    assert weights == (2, 2, 1), "converting and embedding cost more than the LanceDB write"
    docs = settings.docs()
    keys = ("cpu_budget", "converting_weight", "embedding_weight", "indexing_weight")
    assert [docs[f"pipeline.{key}"].title for key in keys] == [
        "CPU budget",
        "Converting weight",
        "Embedding weight",
        "Indexing weight",
    ]
    assert all(docs[f"pipeline.{key}"].description for key in keys)


@pytest.mark.parametrize(
    ("name", "stored"),
    [
        ("the single `workers` field", '{"pipeline": {"workers": 9, "batch_pages": 4}}'),
        (
            "one `*_workers` field per stage",
            '{"pipeline": {"converting_workers": 9, "embedding_workers": 4, '
            '"indexing_workers": 2, "batch_pages": 4}}',
        ),
    ],
)
def test_settings_stored_before_the_cpu_budget_still_decode(name: str, stored: str) -> None:
    """The worker counts became one budget and three weights. msgspec ignores a key it no longer
    knows, so a row an older build wrote loads with the new defaults rather than breaking the
    boot."""
    loaded = settings._decode(stored)

    assert loaded is not None, name
    assert loaded.pipeline.batch_pages == 4, "the fields that stayed are still read"
    assert loaded.pipeline.cpu_budget == PipelineSettings().cpu_budget, "the new field defaults"


def test_retention_defaults_and_docs() -> None:
    """Four weeks of operation history, and one knob that says so."""
    assert RetentionSettings().operation_days == 28
    assert UserSettings().retention == RetentionSettings()
    docs = settings.docs()
    assert docs["retention.operation_days"].title == "Operation history (days)"
    assert docs["retention.operation_days"].description


def test_maintenance_defaults_and_docs() -> None:
    """Three months of audit files, and two previews built at a time."""
    assert UserSettings().retention.audit_days == 90
    assert PipelineSettings().preview_workers == 2
    docs = settings.docs()
    assert docs["retention.audit_days"].title == "Audit retention (days)"
    assert docs["pipeline.preview_workers"].title == "Preview builds"
    assert all(
        docs[key].description for key in ("retention.audit_days", "pipeline.preview_workers")
    )


@pytest.mark.parametrize(
    ("name", "struct", "expected"),
    [
        ("all unset -> empty", SearchOverrides(), {}),
        ("one set field", SearchOverrides(limit=3), {"limit": 3}),
        (
            "falsy but set values are kept",
            SearchOverrides(vector_weight=0.0, rrf_k=1),
            {"vector_weight": 0.0, "rrf_k": 1},
        ),
        (
            "nested struct is a value, not None",
            CollectionOverrides(chunker=Chunker.TEXT),
            {"chunker": "text", "search": SearchOverrides()},
        ),
    ],
)
def test_without_none(name: str, struct: msgspec.Struct, expected: dict) -> None:
    assert without_none(struct) == expected, name


# --- settings persistence ---------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "stored"),
    [
        ("not json at all", "{not json"),
        ("a profile the catalogue does not hold", '{"embedding": "bogus"}'),
        ("a reranker the catalogue does not hold", '{"search": {"reranker_model": "no/such"}}'),
        ("wrong field type", '{"pipeline": {"cpu_budget": "many"}}'),
        (
            "out of bounds, rejected by __post_init__",
            '{"pipeline": {"embedding_weight": 0}}',
        ),
    ],
)
async def test_unreadable_settings_fall_back_to_defaults(name: str, stored: str) -> None:
    await save_user_settings(UserSettings(embedding="compact"))
    async with db.connect() as conn:
        await conn.execute(update(tables.settings).values(json=stored))
    forget_settings()  # a direct write bypasses the process cache

    loaded = await load_user_settings_or_none()

    assert loaded == UserSettings(), f"defaults, so boot continues: {name}"
    problem = settings_problem()
    assert problem is not None and "unreadable" in problem, name


@pytest.mark.anyio
async def test_settings_problem_clears_after_a_good_load() -> None:
    await save_user_settings(UserSettings(embedding="quality"))
    assert await load_user_settings_or_none() == UserSettings(embedding="quality")
    assert settings_problem() is None


@pytest.mark.anyio
async def test_settings_problem_is_none_before_init() -> None:
    assert await load_user_settings_or_none() is None
    assert settings_problem() is None


# --- the settings process cache ---------------------------------------------------


def _count_connects(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every `db.connect()` from here on, as a list to assert the length of."""
    calls: list[str] = []
    real = db.connect

    def counted():
        calls.append(str(home.DB_FILE))
        return real()

    monkeypatch.setattr(db, "connect", counted)
    return calls


@pytest.mark.anyio
async def test_user_settings_are_read_once_and_refreshed_on_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await save_user_settings(UserSettings(embedding="compact"))
    forget_settings()
    connects = _count_connects(monkeypatch)

    first, second = await load_user_settings_or_none(), await load_user_settings_or_none()

    assert first == second == UserSettings(embedding="compact")
    assert len(connects) == 1, "the second load answers from the process cache"

    await save_user_settings(UserSettings(embedding="quality"))

    assert len(connects) == 2, "the write itself connects"
    assert await settings.load_user_settings() == UserSettings(embedding="quality")
    assert len(connects) == 2, "and refreshes the cache, so the read after it does not"


@pytest.mark.anyio
async def test_forgetting_the_cache_forces_a_reread() -> None:
    await save_user_settings(UserSettings(embedding="compact"))
    async with db.connect() as conn:
        await conn.execute(
            update(tables.settings).values(json=db.dumps(UserSettings(embedding="quality")))
        )

    assert await settings.load_user_settings() == UserSettings(embedding="compact"), "still cached"

    forget_settings()

    assert await settings.load_user_settings() == UserSettings(embedding="quality")


@pytest.mark.anyio
async def test_unreadable_settings_are_not_cached() -> None:
    """A broken row must stay live: the run that repairs it is seen without a second step."""
    await save_user_settings(UserSettings(embedding="compact"))
    async with db.connect() as conn:
        await conn.execute(update(tables.settings).values(json="{not json"))
    forget_settings()

    loaded = await load_user_settings_or_none()

    assert loaded == UserSettings(), "defaults while the row is unreadable"
    assert settings_problem() is not None

    async with db.connect() as conn:  # nothing was cached, so nothing has to be forgotten
        await conn.execute(
            update(tables.settings).values(json=db.dumps(UserSettings(embedding="quality")))
        )

    assert await load_user_settings_or_none() == UserSettings(embedding="quality")
    assert settings_problem() is None


@pytest.mark.anyio
async def test_the_missing_row_before_init_is_not_cached() -> None:
    assert await load_user_settings_or_none() is None

    async with db.connect() as conn:  # first run, straight into the row
        await conn.execute(
            insert(tables.settings).values(id=1, json=db.dumps(UserSettings(embedding="compact")))
        )

    assert await load_user_settings_or_none() == UserSettings(embedding="compact")


@pytest.mark.anyio
async def test_connect_skips_ensure_home_after_the_first_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    real = home.ensure_home

    async def counted() -> Path:
        calls.append(home.HOME)
        return await real()

    monkeypatch.setattr(home, "ensure_home", counted)
    db._migrated.clear()

    async with db.connect():
        pass
    async with db.connect():
        pass

    assert len(calls) == 1, "the home is only made once this process has migrated the file"


@pytest.mark.anyio
async def test_the_settings_cache_ends_on_the_saved_value_under_concurrent_loads() -> None:
    """A load that misses reads the row before it publishes it, so it can still be in flight when
    a save commits. The saved row has to win: the load must not cache the row it read first."""
    await save_user_settings(UserSettings(embedding="compact"))
    saved = UserSettings(embedding="quality")
    loaders, loads = 8, 25
    forget_settings()  # so the first load of every task misses and really reads the row

    async def read() -> list[str]:
        return [(await settings.load_user_settings()).embedding for _ in range(loads)]

    reading = [asyncio.create_task(read()) for _ in range(loaders)]
    await save_user_settings(saved)  # commits while the loads above are in flight
    reads = await asyncio.gather(*reading)

    assert [len(one) for one in reads] == [loads] * loaders, "every load returned a value"
    assert {one for read_back in reads for one in read_back} <= {"compact", "quality"}, (
        "every load saw a stored row, never a default"
    )
    assert settings._state == settings._Loaded(saved), "the cache ends on the saved row"
    assert await settings.load_user_settings() == saved
    forget_settings()
    assert await settings.load_user_settings() == saved, "and the row agrees"


# --- db.connect and db.migrate_once -----------------------------------------------


class _UnitFailed(Exception):
    """Raised inside a `db.connect()` block, to prove the unit of work is rolled back."""


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "fails", "expected"),
    [
        ("the unit of work returns: committed", False, [("notes",)]),
        ("the unit of work raises: rolled back", True, []),
    ],
)
async def test_connect_commits_or_rolls_back_the_whole_unit_of_work(
    name: str, fails: bool, expected: list[tuple[str]]
) -> None:
    async def unit() -> None:
        async with db.connect() as conn:
            await conn.execute(insert(tables.collections).values(name="notes", created_at=1.0))
            if fails:
                raise _UnitFailed(name)

    if fails:
        with pytest.raises(_UnitFailed):
            await unit()
    else:
        await unit()

    async with db.connect() as conn:  # a connection of its own: only committed rows are visible
        rows = await conn.execute(select(tables.collections.c.name))
        assert list(rows) == expected, name


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "concurrently"),
    [
        ("the second call answers from the process set", False),
        ("two callers race for the lock", True),
    ],
)
async def test_migrate_once_creates_the_schema_exactly_once(
    monkeypatch: pytest.MonkeyPatch, name: str, concurrently: bool
) -> None:
    applied: list[int] = []  # the thread each run of the schema script happened on
    real = db.migrate

    def counted(conn: sqlite3.Connection) -> int:
        applied.append(threading.get_ident())
        return real(conn)

    monkeypatch.setattr(db, "migrate", counted)

    if concurrently:
        await asyncio.gather(db.migrate_once(), db.migrate_once())
    else:
        await db.migrate_once()
        await db.migrate_once()

    assert len(applied) == 1, name
    assert applied[0] != threading.get_ident(), "sqlite3 and executescript block: not on the loop"
    async with db.connect() as conn:
        version = await conn.scalar(text("pragma user_version"))
        collections = await conn.scalar(select(func.count()).select_from(tables.collections))
    assert version == db.SCHEMA_VERSION, name
    assert collections == 0, name


def test_two_event_loops_can_open_the_first_connections_at_once() -> None:
    """Litestar's loop and DBOS's both reach a fresh home at boot. SQLAlchemy guards an engine's
    first connection with an asyncio lock bound to one loop, so the loser of that race raised
    "bound to a different event loop" before `migrate_once` made that connection itself."""
    failures: list[BaseException] = []

    async def units_of_work() -> None:
        async def one() -> None:
            async with db.connect() as conn:
                await conn.scalar(select(tables.collections.c.name))

        await asyncio.wait_for(asyncio.gather(*(one() for _ in range(4))), 10)

    def run_loop() -> None:
        try:
            asyncio.run(units_of_work())
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=run_loop) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == [], "every unit of work on both loops got its connection"


# --- the CPU budget ---------------------------------------------------------------

CPU_CALLERS = 4  # more callers than the narrow budget admits, so a leak would show
CPU_WAIT_SECONDS = 10.0


@pytest.fixture
def restored_cpu_budget() -> Iterator[None]:
    """The budget is process state, and the process outlives the test that resized it."""
    yield
    cpu.configure_cpu_budget(PipelineSettings().cpu_budget)


@pytest.mark.anyio
async def test_on_cpu_runs_the_work_in_a_worker_thread() -> None:
    def work(value: int, *, double: bool) -> tuple[int, int]:
        return threading.get_ident(), value * 2 if double else value

    ident, result = await cpu.on_cpu(work, 21, double=True)

    assert ident != threading.get_ident(), "the event loop never runs the CPU work itself"
    assert result == 42, "positional and keyword arguments reach the callable"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "budget"),
    [
        ("a budget of one serialises the callers", 1),
        ("a budget of four runs them side by side", CPU_CALLERS),
    ],
)
async def test_on_cpu_holds_one_slot_of_the_budget(
    restored_cpu_budget: None, name: str, budget: int
) -> None:
    cpu.configure_cpu_budget(budget)
    # every slot the budget admits has to be filled at once, or the barrier times out and the
    # callable raises: that is what proves the budget is the limit, and not the thread pool
    gate = threading.Barrier(budget)
    counter = threading.Lock()
    running, peak = 0, 0

    def work() -> None:
        nonlocal running, peak
        with counter:
            running += 1
            peak = max(peak, running)
        gate.wait(timeout=CPU_WAIT_SECONDS)
        with counter:
            running -= 1

    await asyncio.gather(*(cpu.on_cpu(work) for _ in range(CPU_CALLERS)))

    assert peak == budget, name


@pytest.mark.anyio
async def test_every_event_loop_widens_its_own_thread_limiter() -> None:
    """The app runs two loops, Litestar's and DBOS's background one, each in a thread of its own.
    anyio's default of 40 threads per loop would queue previews behind pipeline work before the
    budget is reached, so each loop widens the limiter it owns and `_cpu_slots` stays the limit."""
    await cpu.on_cpu(lambda: None)
    mine = anyio.to_thread.current_default_thread_limiter()

    assert mine.total_tokens == cpu.THREAD_LIMIT, "wide enough never to be the limit"

    others: list[float] = []

    async def take() -> None:
        await cpu.on_cpu(lambda: None)
        others.append(anyio.to_thread.current_default_thread_limiter().total_tokens)

    thread = threading.Thread(target=lambda: asyncio.run(take()))
    thread.start()
    thread.join(timeout=CPU_WAIT_SECONDS)

    assert others == [cpu.THREAD_LIMIT], "a second loop widens the limiter of its own"


# --- audit ------------------------------------------------------------------------


@pytest.mark.anyio
async def test_record_writes_one_private_json_line_with_every_field() -> None:
    entry = await audit.record(
        "collection.create",
        actor=Actor.MCP,
        outcome=Outcome.OK,
        duration_ms=7,
        request_id="r1",
        session_id="s1",
        operation_id="w1",
        collection="notes",
        document="a.md",
        error=None,
        detail={"size": 12, "suffix": ".md", "cached": False},
    )

    (line,) = audit_lines()
    assert line == {
        "ts": entry.ts,
        "level": "AUDIT",
        "event": "collection.create",
        "actor": "mcp",
        "outcome": "ok",
        "duration_ms": 7,
        "app_version": audit.APP_VERSION,
        "request_id": "r1",
        "session_id": "s1",
        "operation_id": "w1",
        "collection": "notes",
        "document": "a.md",
        "detail": {"size": 12, "suffix": ".md", "cached": False},
    }
    assert stat.S_IMODE(os.stat(audit.path()).st_mode) == audit.FILE_MODE


def _missing_part_error(root: Path) -> str:
    """The text a pipeline failure carries when a part file is gone: `root_cause` of the real
    `FileNotFoundError`, which names the absolute path."""
    missing = root / "documents" / "ab" / "report.pdf" / "parts" / f"{home.part_name(0)}.md"
    try:
        missing.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        return f"{type(exc).__name__}: {exc}"
    raise AssertionError(f"{missing} exists")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "error", "expected"),
    [
        ("no error", None, None),
        (
            "a path under the haskie home",
            lambda: _missing_part_error(home.HOME),
            "FileNotFoundError: [Errno 2] No such file or directory: "
            "'$HASKIE_HOME/documents/ab/report.pdf/parts/000000.md'",
        ),
        (
            "a path under the user's home",
            lambda: _missing_part_error(Path.home() / "private"),
            "FileNotFoundError: [Errno 2] No such file or directory: "
            "'~/private/documents/ab/report.pdf/parts/000000.md'",
        ),
    ],
)
async def test_record_scrubs_the_error_it_is_given(
    name: str, error: Callable[[], str] | None, expected: str | None, caplog
) -> None:
    """A pipeline failure reaches `record` as raw exception text, so the scrub happens there."""
    with caplog.at_level("AUDIT", logger="haskie.audit"):
        entry = await audit.record(
            "import.failed",
            actor=Actor.OPERATION,
            outcome=Outcome.ERROR,
            duration_ms=3,
            document="report.pdf",
            error=None if error is None else error(),
        )

    (line,) = audit_lines()
    assert (entry.error, line.get("error")) == (expected, expected), name
    (logged,) = [r for r in caplog.records if r.name == "haskie.audit"]
    assert getattr(logged, "error", None) == expected, name


@pytest.mark.anyio
async def test_record_appends_rather_than_replacing() -> None:
    await audit.record("a", actor=Actor.WEB, outcome=Outcome.OK, duration_ms=0)
    await audit.record("b", actor=Actor.WEB, outcome=Outcome.OK, duration_ms=0)
    assert [line["event"] for line in audit_lines()] == ["a", "b"]


@pytest.mark.anyio
async def test_audited_handler_records_ok_and_reads_contextvars() -> None:
    from haskie import logs

    @audit.audited("collection.document.add")
    async def handler(collection: str, document: str, size: int = 0) -> str:
        return f"{collection}/{document}/{size}"

    logs.clear()
    logs.bind(actor="mcp", request_id="req-1")
    try:
        assert await handler("notes", document="a.md") == "notes/a.md/0"
    finally:
        logs.clear()

    (line,) = audit_lines()
    assert (line["event"], line["outcome"], line["actor"]) == (
        "collection.document.add",
        "ok",
        "mcp",
    )
    assert (line["request_id"], line["collection"], line["document"]) == ("req-1", "notes", "a.md")
    assert "error" not in line


@pytest.mark.anyio
async def test_audited_records_error_with_a_scrubbed_message_and_re_raises() -> None:
    @audit.audited("collection.document.add")
    async def handler(collection: str) -> None:
        raise errors.InvalidInput(f"bad file {home.HOME}/documents/x")

    with pytest.raises(errors.InvalidInput):
        await handler("notes")

    (line,) = audit_lines()
    assert (line["outcome"], line["actor"], line["collection"]) == ("error", "web", "notes")
    assert line["error"] == "InvalidInput: bad file $HASKIE_HOME/documents/x"
    assert "request_id" not in line, "no request context outside a request"


@pytest.mark.anyio
async def test_audited_records_both_branches() -> None:
    @audit.audited("collection.delete")
    async def handler(collection: str) -> str:
        if collection == "boom":
            raise errors.Conflict("busy")
        return collection

    assert await handler("notes") == "notes"
    with pytest.raises(errors.Conflict, match="busy"):
        await handler("boom")

    ok, failed = audit_lines()
    assert (ok["outcome"], ok["collection"]) == ("ok", "notes")
    assert (failed["outcome"], failed["error"]) == ("error", "Conflict: busy")


def test_audited_preserves_the_wrapped_signature_for_dependency_injection() -> None:
    """Litestar builds its injection from `inspect.signature`, so the wrapper must add nothing."""
    import inspect

    async def handler(collection: str, doc: str, size: int = 0) -> None: ...

    wrapped = audit.audited("collection.document.add")(handler)
    assert inspect.signature(wrapped) == inspect.signature(handler)
    assert getattr(wrapped, "__name__", None) == "handler"


@pytest.mark.anyio
async def test_audited_skips_unset_optional_parameters() -> None:
    @audit.audited("session.collections.set")
    async def handler(session_id: str, collection: str | None = None) -> None: ...

    await handler("s-1")

    (line,) = audit_lines()
    assert line["session_id"] == "s-1"
    assert "collection" not in line


@pytest.mark.anyio
async def test_audited_copies_every_record_field_a_handler_takes() -> None:
    """`operation_id` too: a handler that takes one needs no `attach` for it."""

    @audit.audited("operation.cancel")
    async def handler(operation_id: str) -> None: ...

    await handler("op-1")

    (line,) = audit_lines()
    assert line["operation_id"] == "op-1"
    assert "detail" not in line


# --- audit retention --------------------------------------------------------------

AUDIT_NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)  # 90 days back is 2026-03-03


def _audit_files(names: list[str]) -> None:
    home.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        (home.AUDIT_DIR / name).write_text("{}\n")


@pytest.mark.anyio
async def test_audit_prune_deletes_only_old_daily_files() -> None:
    """Retention is a date comparison on the file name, so only the files this module writes are
    considered, and the cutoff day itself is still inside the window."""
    kept = [
        "audit-2026-02-31.jsonl",  # a well-formed name that is not a real date
        "audit-2026-03-03.jsonl",  # exactly at the cutoff
        "audit-2026-06-01.jsonl",  # today
        "audit-06-01.jsonl",  # not the name `audit.path` writes
        "notes.txt",
    ]
    _audit_files(["audit-2026-01-01.jsonl", *kept])

    assert await audit.prune(90, now=AUDIT_NOW) == 1
    assert sorted(path.name for path in home.AUDIT_DIR.iterdir()) == sorted(kept)
    assert await audit.prune(90, now=AUDIT_NOW) == 0, "nothing older than the cutoff is left"


@pytest.mark.anyio
async def test_audit_prune_measures_from_the_clock_by_default() -> None:
    _audit_files(["audit-2020-01-01.jsonl", audit.path().name])

    assert await audit.prune(1) == 1, "the file from 2020, and only it"
    assert audit.path().exists(), "today's file is inside every window"


@pytest.mark.parametrize(
    ("name", "retention_days", "directory"),
    [
        ("0 keeps everything", 0, True),
        ("a negative value keeps everything", -1, True),
        ("no audit directory yet", 90, False),
    ],
)
@pytest.mark.anyio
async def test_audit_prune_does_nothing(name: str, retention_days: int, directory: bool) -> None:
    old = home.AUDIT_DIR / "audit-2020-01-01.jsonl"
    if directory:
        _audit_files([old.name])
    else:
        home.AUDIT_DIR.rmdir()  # created by the test home fixture, still empty

    assert await audit.prune(retention_days, now=AUDIT_NOW) == 0, name
    assert old.exists() is directory, name


# --- enums on the wire ------------------------------------------------------------

WIRE_CHUNK = Chunk(
    headings=["Replication"],
    frame=["Replication"],
    pieces=[Piece(PieceType.LIST, "- leaders take writes\n")],
    line_start=3,
    line_end=3,
    char_start=15,
    char_end=37,
    byte_start=15,
    byte_end=37,
    start_reason=CutReason.LENGTH_SENTENCE,
)


@pytest.mark.parametrize(
    ("name", "struct", "path", "value"),
    [
        (
            "settings row: the chunker, also a cache key field",
            ChunkSettings(chunker=Chunker.TEXT),
            ["chunker"],
            "text",
        ),
        (
            "settings row: a value with a hyphen",
            SearchSettings(reranker=Reranker.CROSS_ENCODER),
            ["reranker"],
            "cross-encoder",
        ),
        (
            "documents row: the status column",
            Document("guide.md", ".md", 29, DocumentStatus.IMPORTED, parser=Parser.PLAIN),
            ["status"],
            "imported",
        ),
        (
            "embeddings row: the parser column",
            embed_cache.params(
                Document("guide.md", ".md", 29, DocumentStatus.IMPORTED, parser=Parser.PLAIN),
                ChunkSettings(chunker=Chunker.TEXT),
                None,
            ),
            ["parser"],
            "plain",
        ),
        ("parquet and LanceDB: a boundary column", WIRE_CHUNK, ["start_reason"], "length_sentence"),
        ("parquet: the type of a piece", WIRE_CHUNK, ["pieces", 0, "type"], "list"),
        (
            "audit line: the outcome",
            audit.AuditRecord(
                ts="2026-09-24T08:00:00+00:00",
                level=audit.LEVEL_NAME,
                event="document.import",
                actor=Actor.MCP,
                outcome=Outcome.ERROR,
                duration_ms=12,
                app_version="0.5.0",
            ),
            ["outcome"],
            "error",
        ),
    ],
)
def test_an_enum_is_stored_and_sent_as_the_string_it_replaced(
    name: str, struct: msgspec.Struct, path: list[str | int], value: str
) -> None:
    """JSON (settings, the API, audit lines), the builtins every parquet and LanceDB write starts
    from, and a bound SQLite parameter all carry the plain value, and it decodes back to the
    member. The cache URN stays pinned in `test_embed_cache`."""

    def at(tree: Any) -> Any:
        for step in path:
            tree = (
                tree[step]
                if isinstance(step, int) or isinstance(tree, dict)
                else getattr(tree, step)
            )
        return tree

    member = at(struct)
    assert isinstance(member, StrEnum), name
    assert at(msgspec.json.decode(msgspec.json.encode(struct))) == value, name
    built = at(msgspec.to_builtins(struct))
    assert (type(built), built) == (str, value), name
    assert at(msgspec.convert(msgspec.to_builtins(struct), type(struct))) is member, name
    conn = sqlite3.connect(":memory:")
    try:
        assert conn.execute("select ?, typeof(?)", (member, member)).fetchone() == (value, "text")
    finally:
        conn.close()
