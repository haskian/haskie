"""The `haskie` command.

`destroy` deletes a user's whole document store, so its guards are the point of this module: it
refuses a directory that is not a haskie home, and it asks before it deletes. `init` is the other
guard: a home written before documents became collection-independent is refused, not migrated.
"""

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from haskie import db, home
from haskie.cli import cli
from haskie.layout import shard

runner = CliRunner()

# The paths `home.use` rebinds; a test that points the CLI elsewhere must put them all back.
HOME_PATHS = (
    "HOME",
    "COLLECTION_ROOT",
    "DOCUMENT_ROOT",
    "STAGING_ROOT",
    "AUDIT_DIR",
    "DB_FILE",
    "MODEL_CACHE",
)


@pytest.fixture
def restore_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo `home.use` after the test: the CLI rebinds module-level paths for the whole process."""
    for attribute in HOME_PATHS:
        monkeypatch.setattr(home, attribute, getattr(home, attribute), raising=True)
    monkeypatch.setattr(db, "_migrated", set())


@pytest.fixture
def elsewhere(tmp_path: Path, restore_home: None) -> Path:
    """A home of our own, restored afterwards."""
    return tmp_path / "home"


def _text(result) -> str:
    """Everything the command wrote, whichever stream it chose."""
    return result.output + result.stderr


def _shelve(root: Path, name: str) -> None:
    """One entry in the sharded layout, as an import or a create would leave it."""
    (root / shard(name) / name).mkdir(parents=True, exist_ok=True)


def _pre_collection_home(root: Path) -> None:
    """A home as an older build left it: migrations 1..8 applied, with a library row in it.

    Built with sqlite3 rather than through `db.migrate`, because `migrate` is exactly what must
    refuse this file: the guard fires before migration 9 runs (see `db.INCOMPATIBLE_HOME_*`).
    """
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(root / "haskie.db")
    try:
        applied = db.MIGRATIONS[: db.INCOMPATIBLE_HOME_MIGRATION - 1]
        for number, script in enumerate(applied, start=1):
            conn.executescript(script)
            conn.execute(f"pragma user_version = {number}")
        conn.execute("insert into libraries (name) values ('notes')")
        conn.commit()
    finally:
        conn.close()


def test_init_creates_the_home_and_repeats_safely(elsewhere: Path) -> None:
    """Idempotent: `init` is also how an existing home is migrated after an upgrade."""
    first = runner.invoke(cli, ["init", "--home", str(elsewhere)])
    assert first.exit_code == 0, _text(first)
    assert (elsewhere / "haskie.db").is_file()
    for directory in ("documents", "collections", "staging", "audit"):
        assert (elsewhere / directory).is_dir(), directory

    again = runner.invoke(cli, ["init", "--home", str(elsewhere)])
    assert again.exit_code == 0, _text(again)


def test_init_refuses_a_home_from_before_collections(elsewhere: Path) -> None:
    """No migration path exists for the old storage shape, so the user is told to destroy it
    rather than losing rows to a silent drop."""
    _pre_collection_home(elsewhere)

    refused = runner.invoke(cli, ["init", "--home", str(elsewhere)])

    assert refused.exit_code == 1
    assert db.INCOMPATIBLE_HOME_MESSAGE in _text(refused)
    assert refused.stderr.strip() == db.INCOMPATIBLE_HOME_MESSAGE, "a failure goes to stderr"
    with sqlite3.connect(elsewhere / "haskie.db") as conn:
        (version,) = conn.execute("pragma user_version").fetchone()
    assert version == db.INCOMPATIBLE_HOME_MIGRATION - 1, "the refused migration did not run"


def test_destroy_after_a_refused_init_lets_it_start_over(elsewhere: Path) -> None:
    """The recovery path the message prescribes has to actually work."""
    _pre_collection_home(elsewhere)

    assert runner.invoke(cli, ["destroy", "--home", str(elsewhere), "--yes"]).exit_code == 0
    again = runner.invoke(cli, ["init", "--home", str(elsewhere)])

    assert again.exit_code == 0, _text(again)
    assert (elsewhere / "collections").is_dir()


def test_destroy_asks_first_and_leaves_everything_when_refused(elsewhere: Path) -> None:
    runner.invoke(cli, ["init", "--home", str(elsewhere)])
    _shelve(home.COLLECTION_ROOT, "notes")
    _shelve(home.DOCUMENT_ROOT, "guide.md")

    refused = runner.invoke(cli, ["destroy", "--home", str(elsewhere)], input="n\n")

    assert refused.exit_code == 1, "abort is a failure exit, not a silent no-op"
    assert elsewhere.is_dir(), "nothing deleted"
    summary = _text(refused)
    assert "collections: notes" in summary, "the summary names what would be lost"
    assert "1 documents" in summary, "and how many documents go with it"


def test_destroy_deletes_the_home_when_confirmed(elsewhere: Path) -> None:
    runner.invoke(cli, ["init", "--home", str(elsewhere)])

    done = runner.invoke(cli, ["destroy", "--home", str(elsewhere)], input="y\n")

    assert done.exit_code == 0, _text(done)
    assert not elsewhere.exists()


def test_destroy_yes_skips_the_prompt(elsewhere: Path) -> None:
    runner.invoke(cli, ["init", "--home", str(elsewhere)])

    done = runner.invoke(cli, ["destroy", "--home", str(elsewhere), "--yes"])

    assert done.exit_code == 0, _text(done)
    assert not elsewhere.exists()


@pytest.mark.parametrize(
    ("name", "directory"),
    [("the document store", "documents"), ("the collection store", "collections")],
)
def test_destroy_recognises_a_home_without_a_database(
    tmp_path: Path, restore_home: None, name: str, directory: str
) -> None:
    """A crash between the directories and the first migration leaves a home with no haskie.db;
    it is still a home, and still destroyable."""
    root = tmp_path / "half-made"
    (root / directory).mkdir(parents=True)

    done = runner.invoke(cli, ["destroy", "--home", str(root), "--yes"])

    assert done.exit_code == 0, f"{name}: {_text(done)}"
    assert not root.exists(), name


def test_destroy_refuses_a_directory_that_is_not_a_home(tmp_path: Path, restore_home: None) -> None:
    """The guard that stops a mistyped `--home ~/Documents` from deleting the wrong tree."""
    documents = tmp_path / "Documents"
    (documents / "keep").mkdir(parents=True)
    (documents / "keep" / "thesis.pdf").write_bytes(b"%PDF-1.4\n")

    refused = runner.invoke(cli, ["destroy", "--home", str(documents), "--yes"])

    assert refused.exit_code == 1
    assert "does not look like a haskie home" in _text(refused)
    assert (documents / "keep" / "thesis.pdf").is_file(), "untouched"


def test_destroy_on_a_missing_home_says_so(tmp_path: Path, restore_home: None) -> None:
    never = tmp_path / "never-created"

    response = runner.invoke(cli, ["destroy", "--home", str(never), "--yes"])

    assert response.exit_code == 0, "nothing to do is not an error"
    assert "nothing to destroy" in _text(response)


def test_init_after_destroy_migrates_again(elsewhere: Path) -> None:
    """`destroy` clears the per-process "already migrated" set, or the new home would have no
    schema."""
    runner.invoke(cli, ["init", "--home", str(elsewhere)])
    runner.invoke(cli, ["destroy", "--home", str(elsewhere), "--yes"])

    again = runner.invoke(cli, ["init", "--home", str(elsewhere)])

    assert again.exit_code == 0, _text(again)
    assert (elsewhere / "haskie.db").is_file(), "the schema was applied to the new file"


def test_version_reports_the_home_it_would_use(elsewhere: Path) -> None:
    result = runner.invoke(cli, ["version"])

    assert result.exit_code == 0
    assert "haskie" in result.output and "home:" in result.output
