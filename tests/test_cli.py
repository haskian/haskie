"""The `haskie` command.

`destroy` deletes a user's whole document store, so its guards are the point of this module: it
refuses a directory that is not a haskie home, and it asks before it deletes. `init` is the other
guard: a home written before documents became collection-independent is refused, not migrated.

`install claude` and the home lock are the other two: one writes into a user's Claude Code
configuration, the other is what keeps a second haskie off a home a first one is already running.
"""

import asyncio
import json
import re
import sqlite3
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from haskie import claude, db, home
from haskie import cli as cli_module
from haskie.claude import Scope
from haskie.cli import cli
from haskie.collection import Collection
from haskie.errors import Conflict
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
    "LOCK_FILE",
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


# --- one haskie per home ----------------------------------------------------


def test_claim_home_refuses_a_second_holder_and_says_who_has_it(elsewhere: Path) -> None:
    """`flock` is per open file description, so a second claim from this process conflicts exactly
    as a second process would - no subprocess needed to prove the guard."""
    runner.invoke(cli, ["init", "--home", str(elsewhere)])
    home.use(elsewhere)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(home, "ADDRESS", "http://127.0.0.1:8000")

    with home.hold_home():
        monkey.setattr(home, "_holding", None)  # pretend to be a second process
        with pytest.raises(Conflict) as refused:
            home.claim_home()
        monkey.undo()

    assert "already running" in str(refused.value)
    assert str(elsewhere) in str(refused.value), "names the home, not just the port"
    assert "127.0.0.1:8000" in str(refused.value), "names the holder's address"


def test_claim_home_is_reentrant_for_one_process(elsewhere: Path) -> None:
    """`--reload` runs one lifespan per restart in the same process; the second must not refuse."""
    runner.invoke(cli, ["init", "--home", str(elsewhere)])
    home.use(elsewhere)

    with home.hold_home():
        home.claim_home()  # would raise if it took a second descriptor
    assert home._holding is None, "released on the way out"


def test_release_home_lets_the_next_one_in(elsewhere: Path) -> None:
    """A stopped haskie must not lock its home out of the next one."""
    runner.invoke(cli, ["init", "--home", str(elsewhere)])
    home.use(elsewhere)

    with home.hold_home():
        pass
    with home.hold_home():
        assert home.LOCK_FILE.read_text().startswith("pid ")


def test_run_refuses_in_one_line_when_the_home_is_taken(elsewhere: Path) -> None:
    """The app's startup hook is the authority, but its refusal is a lifespan traceback out of
    uvicorn; `run` asks first so the common case reads as one line."""
    runner.invoke(cli, ["init", "--home", str(elsewhere)])
    home.use(elsewhere)

    with home.hold_home():
        refused = runner.invoke(cli, ["run", "--home", str(elsewhere)])

    assert refused.exit_code == 1
    assert "already running" in refused.stderr
    assert refused.stdout == "", "it never got as far as announcing a server"


def test_claim_home_creates_the_home_it_locks(elsewhere: Path) -> None:
    """It is the app's first startup hook, so it runs before anything has made the directory."""
    home.use(elsewhere)

    with home.hold_home():
        assert home.LOCK_FILE.is_file()
    assert stat.S_IMODE(elsewhere.stat().st_mode) == home.DIR_MODE


# --- keeping a server up ----------------------------------------------------

# Nothing ever connects: every case below stubs the probe, so a port is only a string to parse.
DEAD_URL = "http://127.0.0.1:9/mcp"


@dataclass
class EnsureCase:
    serving: bool
    wait: bool
    deadline: float
    exit_code: int
    expect_in_output: str
    expect_spawn: bool


ENSURE_CASES = {
    "a served url costs one probe": EnsureCase(
        serving=True,
        wait=True,
        deadline=60.0,
        exit_code=0,
        expect_in_output="already serving",
        expect_spawn=False,
    ),
    "no-wait returns once it has spawned": EnsureCase(
        serving=False,
        wait=False,
        deadline=60.0,
        exit_code=0,
        expect_in_output="starting haskie",
        expect_spawn=True,
    ),
    "a server that never answers fails": EnsureCase(
        serving=False,
        wait=True,
        deadline=0.0,
        exit_code=1,
        expect_in_output="did not come up",
        expect_spawn=True,
    ),
}


@pytest.mark.parametrize("case", ENSURE_CASES.values(), ids=list(ENSURE_CASES))
def test_ensure(case: EnsureCase, elsewhere: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`ensure` runs on every Claude Code session start, so its fast path spawns nothing and its
    slow path must fail rather than hang the client waiting on it."""
    spawned: list[list[str]] = []
    monkeypatch.setattr(cli_module, "_serving", lambda _url: case.serving)
    monkeypatch.setattr(cli_module, "START_DEADLINE", case.deadline)
    monkeypatch.setattr(subprocess, "Popen", lambda command, **_: spawned.append(command))

    result = runner.invoke(
        cli,
        ["ensure", "--home", str(elsewhere), "--url", DEAD_URL]
        + ([] if case.wait else ["--no-wait"]),
    )

    assert result.exit_code == case.exit_code, result.output
    assert case.expect_in_output in (result.output + result.stderr)
    assert bool(spawned) is case.expect_spawn
    if case.expect_spawn:
        assert spawned[0][1:4] == ["-m", "haskie", "run"], "spawned through this interpreter"
        assert str(elsewhere) in spawned[0], "the child serves the same home"
    if case.exit_code:
        assert str(elsewhere / "server.log") in result.stderr, "names the log to read"


# --- install claude ---------------------------------------------------------


@dataclass
class InstallCase:
    scope: Scope
    claude_on_path: bool
    collections: list[tuple[str, str]]
    expect_in_skill: list[str]
    expect_in_output: str


CLAUDE_STUB = '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$CLAUDE_ARGV"\n'

INSTALL_CASES = {
    "records the argv claude needs": InstallCase(
        scope="project",
        claude_on_path=True,
        collections=[("roasting", "Three books on coffee roasting."), ("adr", "")],
        expect_in_skill=["roasting: Three books on coffee roasting", "adr"],
        expect_in_output="registered the haskie MCP server",
    ),
    "still writes the skill without the claude cli": InstallCase(
        scope="project",
        claude_on_path=False,
        collections=[("roasting", "Coffee.")],
        expect_in_skill=["roasting: Coffee"],
        expect_in_output="register the server by hand",
    ),
    "user scope writes to the user's skills": InstallCase(
        scope="user",
        claude_on_path=True,
        collections=[("adr", "Architecture decisions.")],
        expect_in_skill=["adr: Architecture decisions"],
        expect_in_output="registered the haskie MCP server",
    ),
    "an empty home still installs": InstallCase(
        scope="project",
        claude_on_path=True,
        collections=[],
        expect_in_skill=["list_collections"],
        expect_in_output="collections in the trigger: none yet",
    ),
}


@pytest.mark.parametrize("case", INSTALL_CASES.values(), ids=list(INSTALL_CASES))
def test_install_claude(
    case: InstallCase,
    elsewhere: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner.invoke(cli, ["init", "--home", str(elsewhere)])
    home.use(elsewhere)
    for name, description in case.collections:
        asyncio.run(Collection.create(name, description))

    binaries = tmp_path / "bin"
    binaries.mkdir()
    argv_log = tmp_path / "argv.log"
    if case.claude_on_path:
        stub = binaries / "claude"
        stub.write_text(CLAUDE_STUB)
        stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(binaries))  # only what this case puts there
    monkeypatch.setenv("CLAUDE_ARGV", str(argv_log))
    monkeypatch.setattr(claude, "USER_CLAUDE", tmp_path / "user" / ".claude")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        cli,
        ["install", "claude", "--home", str(elsewhere), "--scope", case.scope, "--no-start"],
    )

    assert result.exit_code == 0, result.output
    assert case.expect_in_output in result.output
    written = claude.skill_path(case.scope).read_text()
    for expected in case.expect_in_skill:
        assert expected in written
    hooks = json.loads(claude.settings_path(case.scope).read_text())["hooks"]["SessionStart"]
    hooked = hooks[0]["hooks"][0]["command"]
    assert cli_module.HOOK_MARKER in hooked, "the hook starts haskie"
    assert hooked.endswith("--no-wait"), "a session start must not wait on a boot"
    if case.claude_on_path:
        recorded = argv_log.read_text().splitlines()
        assert recorded[0] == f"mcp remove -s {case.scope} haskie", "replaced, so re-running works"
        assert recorded[1] == (
            f"mcp add -s {case.scope} --transport http haskie http://127.0.0.1:8000/mcp"
        )
    else:
        assert not argv_log.exists()


def test_install_claude_rewrites_rather_than_accumulates(
    elsewhere: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running is how the trigger is refreshed after a collection is added."""
    runner.invoke(cli, ["init", "--home", str(elsewhere)])
    home.use(elsewhere)
    asyncio.run(Collection.create("roasting", "Coffee."))
    monkeypatch.setenv("PATH", str(tmp_path))  # no claude: the skill is what this asserts
    monkeypatch.chdir(tmp_path)
    arguments = ["install", "claude", "--home", str(elsewhere), "--scope", "project", "--no-start"]

    runner.invoke(cli, arguments)
    asyncio.run(Collection.create("adr", "Architecture decisions."))
    again = runner.invoke(cli, arguments)

    assert again.exit_code == 0, again.output
    written = claude.skill_path("project").read_text()
    assert written.count("name: haskie") == 1, "rewritten, not appended to"
    assert "adr: Architecture decisions" in written, "the new collection reached the trigger"
    matchers = json.loads(claude.settings_path("project").read_text())["hooks"]["SessionStart"]
    assert len(matchers) == 1, "the hook is replaced, not stacked"


def test_install_claude_keeps_the_rest_of_an_existing_settings_file(
    elsewhere: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It writes into a file the user owns, so everything it did not come for stays."""
    runner.invoke(cli, ["init", "--home", str(elsewhere)])
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    settings_file = claude.settings_path("project")
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text(
        json.dumps(
            {
                "model": "opus",
                "hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "mine"}]}]},
            }
        )
    )

    result = runner.invoke(
        cli, ["install", "claude", "--home", str(elsewhere), "--scope", "project", "--no-start"]
    )

    assert result.exit_code == 0, result.output
    kept = json.loads(settings_file.read_text())
    assert kept["model"] == "opus", "an unrelated setting survives"
    commands = [h["command"] for m in kept["hooks"]["SessionStart"] for h in m["hooks"]]
    assert "mine" in commands, "the user's own hook survives"
    assert any(cli_module.HOOK_MARKER in c for c in commands), "ours was added beside it"


def test_install_claude_rejects_an_unknown_scope(elsewhere: Path) -> None:
    """Rejected while the arguments are parsed, so nothing is written before it is."""
    result = runner.invoke(
        cli, ["install", "claude", "--home", str(elsewhere), "--scope", "global"]
    )

    assert result.exit_code == 2
    assert "'global' is not one of" in result.stderr


def test_the_skill_only_names_tools_the_server_actually_serves() -> None:
    """The skill tells Claude which tool to reach for, so a renamed tool makes it wrong in a way
    nothing else catches: the file is prose, and it is written at install time."""
    served = {
        name
        for module in (Path(__file__).parents[1] / "src" / "haskie" / "api").glob("*.py")
        for name in re.findall(r'mcp_tool="([^"]+)"', module.read_text(encoding="utf-8"))
    }
    named = {word for word in re.findall(r"`([a-z_]+)`", claude._BODY) if "_" in word}

    assert named, "the skill is supposed to name the tools"
    assert named <= served, f"the skill names tools that do not exist: {sorted(named - served)}"
