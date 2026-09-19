"""Foundations: typed errors, atomic writes, migrations, the CPU budget, settings, audit trail."""

import asyncio
import json
import os
import sqlite3
import stat
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import anyio
import msgspec
import pytest

from haskie import audit, cpu, db, errors, home, settings
from haskie.settings import (
    ChunkSettings,
    CollectionSettings,
    ConversionSettings,
    PipelineSettings,
    RetentionSettings,
    SearchOverrides,
    SearchSettings,
    UserSettings,
    load_user_settings_or_none,
    save_user_settings,
    settings_problem,
    without_none,
)

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
    assert errors.scrub(text) == expected, name


def test_invalid_input_is_also_a_value_error() -> None:
    """Callers written before `errors` (and msgspec's decode-time wrapping) catch ValueError."""
    assert issubclass(errors.InvalidInput, ValueError)
    assert issubclass(errors.CollectionNotFound, errors.NotFound)
    assert issubclass(errors.DocumentNotFound, errors.NotFound)
    assert issubclass(errors.NeedsOcr, errors.PermanentError)


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
        events = [record.msg["event"] for record in caplog.records if isinstance(record.msg, dict)]
        assert events == ["remove_failed", "remove_failed"], "file, then directory"
    finally:
        protected.chmod(0o700)


# --- settings validation ----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "build", "match"),
    [
        ("chunk_size below 1", lambda: ConversionSettings(chunk_size=0), "chunk_size must be >= 1"),
        (
            "chunk_overlap below 0",
            lambda: ConversionSettings(chunk_overlap=-1),
            "chunk_overlap must be >= 0",
        ),
        (
            "chunk_overlap equal to chunk_size",
            lambda: ConversionSettings(chunk_size=10, chunk_overlap=10),
            "chunk_overlap must be <",
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
            "unknown reranker model",
            lambda: SearchSettings(reranker_model="nope/x"),
            "unknown reranker model",
        ),
        (
            "collection overrides both set, overlap wins",
            lambda: CollectionSettings(chunk_size=10, chunk_overlap=10),
            "chunk_overlap must be <",
        ),
        (
            "collection override size alone below 1",
            lambda: CollectionSettings(chunk_size=0),
            "chunk_size must be >= 1",
        ),
        (
            "chunk settings overlap equal to size",
            lambda: ChunkSettings(chunk_size=10, chunk_overlap=10),
            "chunk_overlap must be <",
        ),
        (
            "chunk settings size below 1",
            lambda: ChunkSettings(chunk_size=0),
            "chunk_size must be >= 1",
        ),
        (
            "a collection override that resolves into an invalid pair",
            lambda: CollectionSettings(chunk_size=10).resolve(UserSettings()),
            "chunk_overlap must be <",
        ),
        (
            "search override resolves into an invalid value",
            lambda: SearchOverrides(limit=0).resolve(SearchSettings()),
            "limit must be >= 1",
        ),
        (
            "retention days below 1",
            lambda: RetentionSettings(job_days=0),
            "days must be >= 1",
        ),
        (
            "live window too short to halve",
            lambda: RetentionSettings(job_live_hours=1),
            "job_live_hours must be >= 2",
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
    ],
)
def test_settings_reject_out_of_bounds(name: str, build, match: str) -> None:
    with pytest.raises(errors.InvalidInput, match=match):
        build()


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("defaults", UserSettings),
        ("overlap just below size", lambda: ConversionSettings(chunk_size=2, chunk_overlap=1)),
        ("zero weights allowed", lambda: SearchSettings(vector_weight=0.0, bm25_weight=0.0)),
        (
            "collection override size alone, the user overlap still fits",
            lambda: CollectionSettings(chunk_size=99).resolve(
                UserSettings(conversion=ConversionSettings(chunk_overlap=0))
            ),
        ),
        ("collection override overlap alone", lambda: CollectionSettings(chunk_overlap=0)),
        ("chunk settings defaults", ChunkSettings),
        ("the shortest live window", lambda: RetentionSettings(job_days=1, job_live_hours=2)),
        ("audit retention of zero keeps everything", lambda: RetentionSettings(audit_days=0)),
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


@pytest.mark.parametrize(
    ("name", "stored", "expected"),
    [
        (
            "every renamed section at once",
            {
                "defaults": {"chunk_size": 700},
                "indexing": {"batch_pages": 7},
                "retention": {"days": 3, "live_hours": 9},
                "maintenance": {"audit_retention_days": 5},
            },
            (700, 7, 3, 9, 5),
        ),
        (
            "already written by this build",
            {
                "conversion": {"chunk_size": 700},
                "pipeline": {"batch_pages": 7},
                "retention": {"job_days": 3, "job_live_hours": 9, "audit_days": 5},
            },
            (700, 7, 3, 9, 5),
        ),
        (
            "a section the old build never wrote keeps its default",
            {"indexing": {"batch_pages": 7}},
            (
                ConversionSettings().chunk_size,
                7,
                RetentionSettings().job_days,
                RetentionSettings().job_live_hours,
                RetentionSettings().audit_days,
            ),
        ),
    ],
)
def test_settings_stored_under_the_old_field_names_are_renamed_on_load(
    name: str, stored: dict, expected: tuple[int, ...]
) -> None:
    """`defaults`/`indexing`/`maintenance` became `conversion`/`pipeline`/`retention`. msgspec
    drops a key it does not know, so without the rename a home would come back silently reset."""
    loaded = settings._decode(msgspec.json.encode(stored).decode())

    assert (
        loaded.conversion.chunk_size,
        loaded.pipeline.batch_pages,
        loaded.retention.job_days,
        loaded.retention.job_live_hours,
        loaded.retention.audit_days,
    ) == expected, name


def test_retention_defaults_and_docs() -> None:
    """Four weeks of visible history, two days of it still in the durable-execution tables."""
    assert (RetentionSettings().job_days, RetentionSettings().job_live_hours) == (28, 48)
    assert UserSettings().retention == RetentionSettings()
    docs = settings.docs()
    assert docs["retention.job_days"].title == "Job history (days)"
    assert docs["retention.job_live_hours"].title == "Live job window (hours)"
    assert all(docs[key].description for key in ("retention.job_days", "retention.job_live_hours"))


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
            CollectionSettings(chunker="text"),
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
        ("unknown enum value", '{"embedding": "bogus"}'),
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
        await conn.execute("update settings set json = ? where id = 1", (stored,))
    settings.invalidate()  # a direct write bypasses the process cache (see settings.invalidate)

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
    settings.invalidate()
    connects = _count_connects(monkeypatch)

    first, second = await load_user_settings_or_none(), await load_user_settings_or_none()

    assert first == second == UserSettings(embedding="compact")
    assert len(connects) == 1, "the second load answers from the process cache"

    await save_user_settings(UserSettings(embedding="quality"))

    assert len(connects) == 2, "the write itself connects"
    assert await settings.load_user_settings() == UserSettings(embedding="quality")
    assert len(connects) == 2, "and refreshes the cache, so the read after it does not"


@pytest.mark.anyio
async def test_invalidate_forces_a_reread() -> None:
    await save_user_settings(UserSettings(embedding="compact"))
    async with db.connect() as conn:
        await conn.execute(
            "update settings set json = ? where id = 1",
            (db.dumps(UserSettings(embedding="quality")),),
        )

    assert await settings.load_user_settings() == UserSettings(embedding="compact"), "still cached"

    settings.invalidate()

    assert await settings.load_user_settings() == UserSettings(embedding="quality")


@pytest.mark.anyio
async def test_unreadable_settings_are_not_cached() -> None:
    """A broken row must stay live: the run that repairs it is seen without an `invalidate()`."""
    await save_user_settings(UserSettings(embedding="compact"))
    async with db.connect() as conn:
        await conn.execute("update settings set json = '{not json' where id = 1")
    settings.invalidate()

    loaded = await load_user_settings_or_none()

    assert loaded == UserSettings(), "defaults while the row is unreadable"
    assert settings_problem() is not None

    async with db.connect() as conn:  # no invalidate: nothing was cached
        await conn.execute(
            "update settings set json = ? where id = 1",
            (db.dumps(UserSettings(embedding="quality")),),
        )

    assert await load_user_settings_or_none() == UserSettings(embedding="quality")
    assert settings_problem() is None


@pytest.mark.anyio
async def test_the_missing_row_before_init_is_not_cached() -> None:
    assert await load_user_settings_or_none() is None

    async with db.connect() as conn:  # first run, straight into the row
        await conn.execute(
            "insert into settings (id, json) values (1, ?)",
            (db.dumps(UserSettings(embedding="compact")),),
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
    """A load that misses reads the row outside the cache lock, so it can still be in flight when
    a save commits. The saved row has to win: the load must not cache the row it read first."""
    await save_user_settings(UserSettings(embedding="compact"))
    saved = UserSettings(embedding="quality")
    loaders, loads = 8, 25
    settings.invalidate()  # so the first load of every task misses and really reads the row

    async def read() -> list[str]:
        return [(await settings.load_user_settings()).embedding for _ in range(loads)]

    reading = [asyncio.create_task(read()) for _ in range(loaders)]
    await save_user_settings(saved)  # commits while the loads above are in flight
    reads = await asyncio.gather(*reading)

    assert [len(one) for one in reads] == [loads] * loaders, "every load returned a value"
    assert {one for read_back in reads for one in read_back} <= {"compact", "quality"}, (
        "every load saw a stored row, never a default"
    )
    assert settings._cached == saved, "the cache ends on the saved row, not on one read before it"
    assert await settings.load_user_settings() == saved
    settings.invalidate()
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
            await conn.execute("insert into collections (name, created_at) values ('notes', 1.0)")
            if fails:
                raise _UnitFailed(name)

    if fails:
        with pytest.raises(_UnitFailed):
            await unit()
    else:
        await unit()

    async with db.connect() as conn:  # a connection of its own: only committed rows are visible
        rows = await conn.execute_fetchall("select name from collections")
    assert list(rows) == expected, name


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "concurrently"),
    [
        ("the second call answers from the process set", False),
        ("two callers race for the lock", True),
    ],
)
async def test_migrate_once_applies_every_migration_exactly_once(
    monkeypatch: pytest.MonkeyPatch, name: str, concurrently: bool
) -> None:
    applied: list[int] = []  # the thread each run of the migrations happened on
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
        version = await conn.execute_fetchall("pragma user_version")
        collections = await conn.execute_fetchall("select count(*) from collections")
    assert list(version) == [(len(db.MIGRATIONS),)], name
    assert list(collections) == [(0,)], name


def test_migration_5_adds_the_retention_watermark(tmp_path: Path) -> None:
    """Migration 5 carries the archive watermark, applied to a database built by the previous
    build. It starts at zero, so the first round purges nothing it has not copied."""
    conn = sqlite3.connect(tmp_path / "old.db")
    for number, script in enumerate(db.MIGRATIONS[:4], start=1):
        conn.executescript(script)
        conn.execute(f"pragma user_version = {number}")
    conn.commit()

    assert db.migrate(conn) == len(db.MIGRATIONS)

    assert conn.execute("select key, value from retention_state").fetchall() == [
        ("archive_watermark_ms", "0")
    ]
    assert db.migrate(conn) == len(db.MIGRATIONS), "nothing left to apply"
    conn.close()


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

    ident, result = await cpu.on_cpu("test", work, 21, double=True)

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

    await asyncio.gather(*(cpu.on_cpu("test", work) for _ in range(CPU_CALLERS)))

    assert peak == budget, name


@pytest.mark.anyio
async def test_the_thread_limiter_is_one_per_event_loop() -> None:
    mine = cpu._limiter()

    assert cpu._limiter() is mine, "the running loop keeps the limiter it made"
    assert mine.total_tokens == max(64, 4 * cpu._cpu_budget), "wide enough never to be the limit"

    # the app runs two loops, Litestar's and DBOS's background one, each in a thread of its own
    others: list[anyio.CapacityLimiter] = []

    async def take() -> None:
        others.append(cpu._limiter())

    thread = threading.Thread(target=lambda: asyncio.run(take()))
    thread.start()
    thread.join(timeout=CPU_WAIT_SECONDS)

    assert others and others[0] is not mine, "a second loop gets a limiter of its own"


# --- audit ------------------------------------------------------------------------


def _lines() -> list[dict]:
    return [json.loads(line) for line in audit.path().read_text().splitlines()]


@pytest.mark.anyio
async def test_record_writes_one_private_json_line_with_every_field() -> None:
    entry = await audit.record(
        "collection.create",
        actor="mcp",
        outcome="ok",
        duration_ms=7,
        request_id="r1",
        session_id="s1",
        workflow_id="w1",
        collection="notes",
        doc="a.md",
        error=None,
        detail={"size": 12, "suffix": ".md", "cached": False},
    )

    (line,) = _lines()
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
        "workflow_id": "w1",
        "collection": "notes",
        "doc": "a.md",
        "detail": {"size": 12, "suffix": ".md", "cached": False},
    }
    assert stat.S_IMODE(os.stat(audit.path()).st_mode) == audit.FILE_MODE


@pytest.mark.anyio
async def test_record_appends_rather_than_replacing() -> None:
    await audit.record("a", actor="web", outcome="ok", duration_ms=0)
    await audit.record("b", actor="web", outcome="ok", duration_ms=0)
    assert [line["event"] for line in _lines()] == ["a", "b"]


@pytest.mark.anyio
async def test_audited_handler_records_ok_and_reads_contextvars() -> None:
    from haskie import logs

    @audit.audited("collection.document.add", collection="name", doc="doc")
    async def handler(name: str, doc: str, size: int = 0) -> str:
        return f"{name}/{doc}/{size}"

    logs.clear()
    logs.bind(actor="mcp", request_id="req-1")
    try:
        assert await handler("notes", doc="a.md") == "notes/a.md/0"
    finally:
        logs.clear()

    (line,) = _lines()
    assert (line["event"], line["outcome"], line["actor"]) == (
        "collection.document.add",
        "ok",
        "mcp",
    )
    assert (line["request_id"], line["collection"], line["doc"]) == ("req-1", "notes", "a.md")
    assert "error" not in line


@pytest.mark.anyio
async def test_audited_records_error_with_a_scrubbed_message_and_re_raises() -> None:
    @audit.audited("collection.document.add", collection="name")
    async def handler(name: str) -> None:
        raise errors.InvalidInput(f"bad file {home.HOME}/documents/x")

    with pytest.raises(errors.InvalidInput):
        await handler("notes")

    (line,) = _lines()
    assert (line["outcome"], line["actor"], line["collection"]) == ("error", "web", "notes")
    assert line["error"] == "InvalidInput: bad file $HASKIE_HOME/documents/x"
    assert "request_id" not in line, "no request context outside a request"


@pytest.mark.anyio
async def test_audited_records_both_branches() -> None:
    @audit.audited("collection.delete", collection="name")
    async def handler(name: str) -> str:
        if name == "boom":
            raise errors.Conflict("busy")
        return name

    assert await handler("notes") == "notes"
    with pytest.raises(errors.Conflict, match="busy"):
        await handler("boom")

    ok, failed = _lines()
    assert (ok["outcome"], ok["collection"]) == ("ok", "notes")
    assert (failed["outcome"], failed["error"]) == ("error", "Conflict: busy")


def test_audited_preserves_the_wrapped_signature_for_dependency_injection() -> None:
    """Litestar builds its injection from `inspect.signature`, so the wrapper must add nothing."""
    import inspect

    async def handler(name: str, doc: str, size: int = 0) -> None: ...

    wrapped = audit.audited("collection.document.add", collection="name", doc="doc")(handler)
    assert inspect.signature(wrapped) == inspect.signature(handler)
    assert getattr(wrapped, "__name__", None) == "handler"


@pytest.mark.anyio
async def test_audited_skips_unset_optional_parameters() -> None:
    @audit.audited("session.collections.set", session_id="session", collection="name")
    async def handler(session: str, name: str | None = None) -> None: ...

    await handler("s-1")

    (line,) = _lines()
    assert line["session_id"] == "s-1"
    assert "collection" not in line


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
