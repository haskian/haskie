"""The `haskie` command.

`destroy` deletes a user's whole document library, so its guards are the point of this module: it
refuses a directory that is not a haskie home, and it asks before it deletes.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from haskie import db, home
from haskie.cli import cli

runner = CliRunner()


@pytest.fixture
def elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home of our own, restored afterwards: `home.use` rebinds module-level paths."""
    root = tmp_path / "home"
    for attribute in ("HOME", "LIBRARY_ROOT", "AUDIT_DIR", "DB_FILE", "MODEL_CACHE"):
        monkeypatch.setattr(home, attribute, getattr(home, attribute), raising=True)
    monkeypatch.setattr(db, "_migrated", set())
    return root


def test_init_creates_the_home_and_repeats_safely(elsewhere: Path) -> None:
    """Idempotent: `init` is also how an existing home is migrated after an upgrade."""
    first = runner.invoke(cli, ["init", "--home", str(elsewhere)])
    assert first.exit_code == 0, first.output
    assert (elsewhere / "haskie.db").is_file()
    assert (elsewhere / "library").is_dir()

    again = runner.invoke(cli, ["init", "--home", str(elsewhere)])
    assert again.exit_code == 0, again.output


def test_destroy_asks_first_and_leaves_everything_when_refused(elsewhere: Path) -> None:
    runner.invoke(cli, ["init", "--home", str(elsewhere)])
    (elsewhere / "library" / "notes").mkdir(parents=True, exist_ok=True)

    refused = runner.invoke(cli, ["destroy", "--home", str(elsewhere)], input="n\n")

    assert refused.exit_code == 1, "abort is a failure exit, not a silent no-op"
    assert elsewhere.is_dir(), "nothing deleted"
    assert "notes" in refused.output, "the summary names what would be lost"


def test_destroy_deletes_the_home_when_confirmed(elsewhere: Path) -> None:
    runner.invoke(cli, ["init", "--home", str(elsewhere)])

    done = runner.invoke(cli, ["destroy", "--home", str(elsewhere)], input="y\n")

    assert done.exit_code == 0, done.output
    assert not elsewhere.exists()


def test_destroy_yes_skips_the_prompt(elsewhere: Path) -> None:
    runner.invoke(cli, ["init", "--home", str(elsewhere)])

    done = runner.invoke(cli, ["destroy", "--home", str(elsewhere), "--yes"])

    assert done.exit_code == 0, done.output
    assert not elsewhere.exists()


def test_destroy_refuses_a_directory_that_is_not_a_home(tmp_path: Path) -> None:
    """The guard that stops a mistyped `--home ~/Documents` from deleting the wrong tree."""
    documents = tmp_path / "Documents"
    (documents / "keep").mkdir(parents=True)
    (documents / "keep" / "thesis.pdf").write_bytes(b"%PDF-1.4\n")

    refused = runner.invoke(cli, ["destroy", "--home", str(documents), "--yes"])

    assert refused.exit_code == 1
    assert "does not look like a haskie home" in refused.output
    assert (documents / "keep" / "thesis.pdf").is_file(), "untouched"


def test_destroy_on_a_missing_home_says_so(tmp_path: Path) -> None:
    never = tmp_path / "never-created"

    response = runner.invoke(cli, ["destroy", "--home", str(never), "--yes"])

    assert response.exit_code == 0, "nothing to do is not an error"
    assert "nothing to destroy" in response.output


def test_init_after_destroy_migrates_again(elsewhere: Path) -> None:
    """`destroy` clears the per-process "already migrated" set, or the new home would have no
    schema."""
    runner.invoke(cli, ["init", "--home", str(elsewhere)])
    runner.invoke(cli, ["destroy", "--home", str(elsewhere), "--yes"])

    again = runner.invoke(cli, ["init", "--home", str(elsewhere)])

    assert again.exit_code == 0, again.output
    assert (elsewhere / "haskie.db").is_file(), "the schema was applied to the new file"


def test_version_reports_the_home_it_would_use(elsewhere: Path) -> None:
    result = runner.invoke(cli, ["version"])

    assert result.exit_code == 0
    assert "haskie" in result.output and "home:" in result.output
