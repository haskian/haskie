"""The `haskie` command.

`destroy` deletes a user's whole document store, so its guards are the point of this module: it
refuses a directory that is not a haskie home, and it asks before it deletes. `init` is the other
guard: a home written before documents became collection-independent is refused, not migrated.

`install claude` and the home lock are the other two: one writes into a user's Claude Code
configuration, the other is what keeps a second haskie off a home a first one is already running.
"""

import asyncio
import json
import os
import re
import shlex
import signal
import sqlite3
import stat
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from haskie import APP_VERSION, claude, db, home
from haskie import cli as cli_module
from haskie.claude import Scope
from haskie.cli import cli
from haskie.collection.collection import Collection, CollectionSummary, DocumentCounts
from haskie.errors import Conflict, InvalidInput

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_home_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every command exports the home it uses (see `cli._use_home`), and `--home` reads it back
    from there: without this a mise shell's `HASKIE_HOME` would reach a test, and a test's own
    home would reach the next one. `setenv` first, so the undo restores even an unset variable."""
    monkeypatch.setenv("HASKIE_HOME", "")
    monkeypatch.delenv("HASKIE_HOME")


@pytest.fixture
def elsewhere(tmp_path: Path) -> Path:
    """A home of our own. The autouse `haskie_home` fixture puts the process back afterwards."""
    return tmp_path / "home"


def _text(result) -> str:
    """Everything the command wrote, whichever stream it chose."""
    return result.output + result.stderr


def _shelve(root: Path, name: str) -> None:
    """One entry in the sharded layout, as an import or a create would leave it."""
    (root / home.shard(name) / name).mkdir(parents=True, exist_ok=True)


def _pre_collection_home(root: Path) -> None:
    """A home as an older build left it: a `libraries` table, stamped with the version before the
    current schema. `db.migrate` is exactly what must refuse this file."""
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(root / "haskie.db")
    try:
        conn.executescript("create table libraries (name text primary key);")
        conn.execute("insert into libraries (name) values ('notes')")
        conn.execute(f"pragma user_version = {db.SCHEMA_VERSION - 1}")
        conn.commit()
    finally:
        conn.close()


def test_init_creates_the_home_and_repeats_safely(elsewhere: Path) -> None:
    """Idempotent: `init` is also how an existing home is migrated after an upgrade."""
    first = runner.invoke(cli, ["init", "--home", str(elsewhere), "--no-browser"])
    assert first.exit_code == 0, _text(first)
    assert (elsewhere / "haskie.db").is_file()
    for directory in ("documents", "collections", "staging", "audit"):
        assert (elsewhere / directory).is_dir(), directory

    again = runner.invoke(cli, ["init", "--home", str(elsewhere), "--no-browser"])
    assert again.exit_code == 0, _text(again)


UI = f"http://{claude.DEFAULT_HOST}:{claude.DEFAULT_PORT}/"
THIS_HOME = None  # the server reached serves the home `init` made


@pytest.mark.parametrize(
    ("name", "picked", "flags", "serving", "exit_code", "served", "opened", "says"),
    [
        (
            "a first run not done: a server, and the UI opened on it",
            False,
            [],
            THIS_HOME,
            0,
            True,
            [UI],
            "pick the embedding model and the search at",
        ),
        (
            "another home serves the port: refused, nothing opened",
            False,
            [],
            "/elsewhere",
            1,
            True,
            [],
            "serves another home (/elsewhere)",
        ),
        (
            "no browser: nothing started, only where to go",
            False,
            ["--no-browser"],
            THIS_HOME,
            0,
            False,
            [],
            "(haskie run serves it)",
        ),
        ("a first run done: the home as it is", True, [], THIS_HOME, 0, False, [], "ready in"),
    ],
)
def test_init_finishes_the_first_run_in_the_browser_once(
    elsewhere: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    picked: bool,
    flags: list[str],
    serving: str | None,
    exit_code: int,
    served: bool,
    opened: list[str],
    says: str,
) -> None:
    """The settings the first run needs are picked in the web UI, so `init` opens it on a server
    it starts, unless they are picked already, the caller has no browser, or the port is taken
    by another home, whose UI would pick that home's settings."""
    import webbrowser

    from haskie import settings

    if picked:
        runner.invoke(cli, ["init", "--home", str(elsewhere), "--no-browser"])
        assert asyncio.run(settings.init_user_settings(settings.UserSettings()))
    serves: list[str] = []
    browsed: list[str] = []
    answers = {"home": serving or str(elsewhere.resolve())}
    monkeypatch.setattr(cli_module, "_serve", lambda url, wait: serves.append(url))
    monkeypatch.setattr(cli_module, "_status", lambda _url: answers)
    monkeypatch.setattr(webbrowser, "open", lambda url: browsed.append(url) or True)

    result = runner.invoke(cli, ["init", "--home", str(elsewhere), *flags])

    assert result.exit_code == exit_code, f"{name}: {_text(result)}"
    assert (elsewhere / "haskie.db").is_file(), name
    assert serves == ([claude.MCP_URL] if served else []), name
    assert browsed == opened, name
    assert says in _text(result), name


@pytest.mark.parametrize(
    "given", [True, False], ids=["--home through a link", "the default home through a link"]
)
def test_init_knows_its_own_server_through_a_linked_home(
    given: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`~/.haskie` may be a link into a synced folder. The server `init` starts is given the
    resolved root and reports that back, so `init` must resolve the home it compares, whether it
    came from `--home` or from the default, or it calls its own server another home's."""
    import webbrowser

    real = tmp_path / "synced" / "haskie"
    real.mkdir(parents=True)
    link = tmp_path / ".haskie"
    link.symlink_to(real)
    home.use(link)  # the default, as `~/.haskie` would be
    monkeypatch.setattr(cli_module, "_serve", lambda _url, wait: None)
    monkeypatch.setattr(cli_module, "_status", lambda _url: {"home": str(real.resolve())})
    monkeypatch.setattr(webbrowser, "open", lambda _url: True)

    result = runner.invoke(cli, ["init", *(["--home", str(link)] if given else [])])

    assert result.exit_code == 0, _text(result)
    assert "serves another home" not in _text(result)
    assert home.HOME == real.resolve()
    assert os.environ["HASKIE_HOME"] == str(real.resolve()), "a --reload child reads the same root"
    assert (real / "haskie.db").is_file()


def test_init_refuses_a_home_from_before_collections(elsewhere: Path) -> None:
    """No migration path exists for the old storage shape, so the user is told to destroy it
    rather than losing rows to a silent drop."""
    _pre_collection_home(elsewhere)

    refused = runner.invoke(cli, ["init", "--home", str(elsewhere), "--no-browser"])

    assert refused.exit_code == 1
    assert db.INCOMPATIBLE_HOME_MESSAGE in _text(refused)
    assert refused.stderr.strip() == db.INCOMPATIBLE_HOME_MESSAGE, "a failure goes to stderr"
    with sqlite3.connect(elsewhere / "haskie.db") as conn:
        (version,) = conn.execute("pragma user_version").fetchone()
    assert version == db.SCHEMA_VERSION - 1, "the refused migration did not run"


def test_destroy_after_a_refused_init_lets_it_start_over(elsewhere: Path) -> None:
    """The recovery path the message prescribes has to actually work."""
    _pre_collection_home(elsewhere)

    assert runner.invoke(cli, ["destroy", "--home", str(elsewhere), "--yes"]).exit_code == 0
    again = runner.invoke(cli, ["init", "--home", str(elsewhere), "--no-browser"])

    assert again.exit_code == 0, _text(again)
    assert (elsewhere / "collections").is_dir()


def test_destroy_asks_first_and_leaves_everything_when_refused(elsewhere: Path) -> None:
    runner.invoke(cli, ["init", "--home", str(elsewhere), "--no-browser"])
    _shelve(home.COLLECTION_ROOT, "notes")
    _shelve(home.DOCUMENT_ROOT, "guide.md")

    refused = runner.invoke(cli, ["destroy", "--home", str(elsewhere)], input="n\n")

    assert refused.exit_code == 1, "abort is a failure exit, not a silent no-op"
    assert elsewhere.is_dir(), "nothing deleted"
    summary = _text(refused)
    assert "collections: notes" in summary, "the summary names what would be lost"
    assert "1 documents" in summary, "and how many documents go with it"


def test_destroy_deletes_the_home_when_confirmed(elsewhere: Path) -> None:
    runner.invoke(cli, ["init", "--home", str(elsewhere), "--no-browser"])

    done = runner.invoke(cli, ["destroy", "--home", str(elsewhere)], input="y\n")

    assert done.exit_code == 0, _text(done)
    assert not elsewhere.exists()


def test_destroy_yes_skips_the_prompt(elsewhere: Path) -> None:
    runner.invoke(cli, ["init", "--home", str(elsewhere), "--no-browser"])

    done = runner.invoke(cli, ["destroy", "--home", str(elsewhere), "--yes"])

    assert done.exit_code == 0, _text(done)
    assert not elsewhere.exists()


@pytest.mark.parametrize(
    ("name", "directory"),
    [("the document store", "documents"), ("the collection store", "collections")],
)
def test_destroy_recognises_a_home_without_a_database(
    tmp_path: Path, name: str, directory: str
) -> None:
    """A crash between the directories and the first migration leaves a home with no haskie.db;
    it is still a home, and still destroyable."""
    root = tmp_path / "half-made"
    (root / directory).mkdir(parents=True)

    done = runner.invoke(cli, ["destroy", "--home", str(root), "--yes"])

    assert done.exit_code == 0, f"{name}: {_text(done)}"
    assert not root.exists(), name


def test_destroy_refuses_a_directory_that_is_not_a_home(tmp_path: Path) -> None:
    """The guard that stops a mistyped `--home ~/Documents` from deleting the wrong tree."""
    documents = tmp_path / "Documents"
    (documents / "keep").mkdir(parents=True)
    (documents / "keep" / "thesis.pdf").write_bytes(b"%PDF-1.4\n")

    refused = runner.invoke(cli, ["destroy", "--home", str(documents), "--yes"])

    assert refused.exit_code == 1
    assert "does not look like a haskie home" in _text(refused)
    assert (documents / "keep" / "thesis.pdf").is_file(), "untouched"


def test_destroy_refuses_a_home_a_server_holds(elsewhere: Path) -> None:
    """The SessionStart hook usually keeps a server up. Deleting under it leaves it serving from
    deleted files and takes the home lock with it, so `destroy` refuses before it asks."""
    runner.invoke(cli, ["init", "--home", str(elsewhere), "--no-browser"])

    with holding():
        refused = runner.invoke(cli, ["destroy", "--home", str(elsewhere)], input="y\n")

    assert refused.exit_code == 1, _text(refused)
    assert "already running" in refused.stderr
    assert "haskie stop" in refused.stderr, "says what to do"
    assert "about to delete" not in refused.stdout, "refused before the prompt"
    assert (elsewhere / "haskie.db").is_file(), "nothing deleted"


@dataclass
class LeftOverCase:
    files: list[str]  # what sits in a directory destroy may not write into
    directories: list[str]  # empty directories beside them
    expect_left: str


LEFT_OVER_CASES = {
    "a file it cannot delete is named": LeftOverCase(
        files=["guide.md"], directories=[], expect_left="still there: documents/pinned/guide.md"
    ),
    "past the first few, a count": LeftOverCase(
        files=[f"part-{n}.md" for n in range(cli_module.LEFT_SHOWN + 2)],
        directories=[],
        expect_left="documents/pinned/part-4.md and 2 more",
    ),
    "only directories left": LeftOverCase(
        files=[], directories=["empty"], expect_left="still there: empty directories"
    ),
}


@pytest.mark.parametrize("case", LEFT_OVER_CASES.values(), ids=list(LEFT_OVER_CASES))
def test_destroy_fails_when_something_is_left(case: LeftOverCase, elsewhere: Path) -> None:
    """`remove_tree` logs what it cannot delete and carries on, so `destroy` looks afterwards: a
    home still on disk is a failure, and the message says what is left."""
    runner.invoke(cli, ["init", "--home", str(elsewhere), "--no-browser"])
    pinned = elsewhere / "documents" / "pinned"
    pinned.mkdir()
    for name in case.files:
        (pinned / name).write_text("# kept\n")
    for name in case.directories:
        (pinned / name).mkdir()
    pinned.chmod(0o500)  # nothing inside can be unlinked
    try:
        result = runner.invoke(cli, ["destroy", "--home", str(elsewhere), "--yes"])
    finally:
        pinned.chmod(0o700)

    assert result.exit_code == 1, _text(result)
    assert f"deleted {elsewhere.resolve()}" not in result.stdout, "never claims it is gone"
    assert f"could not delete all of {elsewhere.resolve()}" in result.stderr
    assert case.expect_left in result.stderr
    assert elsewhere.is_dir()


def test_destroy_on_a_missing_home_says_so(tmp_path: Path) -> None:
    never = tmp_path / "never-created"

    response = runner.invoke(cli, ["destroy", "--home", str(never), "--yes"])

    assert response.exit_code == 0, "nothing to do is not an error"
    assert "nothing to destroy" in _text(response)


def test_init_after_destroy_migrates_again(elsewhere: Path) -> None:
    """`destroy` clears the per-process "already migrated" set, or the new home would have no
    schema."""
    runner.invoke(cli, ["init", "--home", str(elsewhere), "--no-browser"])
    runner.invoke(cli, ["destroy", "--home", str(elsewhere), "--yes"])

    again = runner.invoke(cli, ["init", "--home", str(elsewhere), "--no-browser"])

    assert again.exit_code == 0, _text(again)
    assert (elsewhere / "haskie.db").is_file(), "the schema was applied to the new file"


def test_version_reports_the_home_it_would_use(elsewhere: Path) -> None:
    result = runner.invoke(cli, ["version"])

    assert result.exit_code == 0
    assert "haskie" in result.output and "home:" in result.output


def test_version_flag_answers_without_a_subcommand() -> None:
    """`--version` is eager, so it answers before Typer asks for a command and before anything
    reads the home. The version only: where the data lives is `haskie version`'s job."""
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"haskie {APP_VERSION}"


# --- one haskie per home ----------------------------------------------------


@contextmanager
def holding(address: str = "http://127.0.0.1:8451") -> Iterator[None]:
    """Claim the home for the body, and give it back afterwards. `claim_home` takes the address
    from the environment, the way `run` leaves it there."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(home.ADDRESS_ENV, address)
        home.claim_home()
    try:
        yield
    finally:
        home.release_home()


def test_claim_home_refuses_a_second_holder_and_says_who_has_it(
    elsewhere: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`flock` is per open file description, so a second claim from this process conflicts exactly
    as a second process would - no subprocess needed to prove the guard. No `init` either: the
    lock needs no database, and makes the home itself."""
    home.use(elsewhere)

    with holding():
        monkeypatch.setattr(home, "_holding", None)  # pretend to be a second process
        with pytest.raises(Conflict) as refused:
            home.claim_home()
        monkeypatch.undo()

    assert "already running" in str(refused.value)
    assert str(elsewhere) in str(refused.value), "names the home, not just the port"
    assert "127.0.0.1:8451" in str(refused.value), "names the holder's address"


def test_claim_home_claims_once_and_gives_the_home_back(elsewhere: Path) -> None:
    """`--reload` runs one lifespan per restart in the same process, so a second claim must not
    refuse; and a stopped haskie must not lock its home out of the next one."""
    home.use(elsewhere)

    with holding():
        home.claim_home()  # would raise if it took a second descriptor
        assert home.LOCK_FILE.read_text().startswith("pid ")
    assert home._holding is None, "released on the way out"

    with holding():
        assert home.LOCK_FILE.read_text().startswith("pid "), "the next one gets in"


@pytest.mark.parametrize(
    ("name", "server_pid", "recorded"),
    [
        ("a server that is its own process", None, os.getpid()),
        # under `--reload` the claim runs in a worker; `stop` must signal `run`, which outlives it
        ("a worker under run --reload", 4242, 4242),
    ],
)
def test_the_lock_names_the_process_stop_must_signal(
    elsewhere: Path, monkeypatch: pytest.MonkeyPatch, name: str, server_pid, recorded: int
) -> None:
    home.use(elsewhere)
    if server_pid is not None:
        monkeypatch.setenv(home.SERVER_PID_ENV, str(server_pid))
    else:
        monkeypatch.delenv(home.SERVER_PID_ENV, raising=False)

    with holding():
        line = home.LOCK_FILE.read_text()

    assert line.startswith(f"pid {recorded},"), name


def test_run_exports_its_own_pid_for_the_lock(
    elsewhere: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run` puts its pid in the environment before uvicorn starts, so a `--reload` worker that
    claims the home records the reloader, not itself."""
    seen: dict[str, str | None] = {}

    def serve(*_args, **_kwargs) -> None:
        seen["pid"] = os.environ.get(home.SERVER_PID_ENV)

    monkeypatch.setattr("uvicorn.run", serve)
    monkeypatch.delenv(home.SERVER_PID_ENV, raising=False)

    result = runner.invoke(cli, ["run", "--home", str(elsewhere), "--reload"])

    assert result.exit_code == 0, result.output
    assert seen["pid"] == str(os.getpid())


def test_run_refuses_in_one_line_when_the_home_is_taken(elsewhere: Path) -> None:
    """The app's startup hook is the authority, but its refusal is a lifespan traceback out of
    uvicorn; `run` asks first so the common case reads as one line."""
    home.use(elsewhere)

    with holding():
        refused = runner.invoke(cli, ["run", "--home", str(elsewhere)])

    assert refused.exit_code == 1
    assert "already running" in refused.stderr
    assert refused.stdout == "", "it never got as far as announcing a server"


def test_claim_home_creates_the_home_it_locks(elsewhere: Path) -> None:
    """It is the app's first startup hook, so it runs before anything has made the directory."""
    home.use(elsewhere)

    with holding():
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
    assert case.expect_in_output in _text(result)
    assert bool(spawned) is case.expect_spawn
    if case.expect_spawn:
        prefix = claude.own_command()
        assert spawned[0][: len(prefix) + 1] == [*prefix, "run"], "spawned through this haskie"
        assert str(elsewhere) in spawned[0], "the child serves the same home"
    if case.exit_code:
        assert str(elsewhere / "server.log") in result.stderr, "names the log to read"


@pytest.mark.parametrize(
    "has_script", [True, False], ids=["the script beside the interpreter", "no script: -m"]
)
def test_own_command_is_this_install_not_the_first_on_path(
    has_script: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`haskie-dev` and an installed `haskie` sit on one PATH. Whichever runs `ensure` must spawn
    itself, never the other one's code on its own home."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    if has_script:
        (bin_dir / "haskie").touch()
    (tmp_path / "haskie").touch()  # a PATH `haskie` from another install
    monkeypatch.setattr(claude.sys, "executable", str(python))
    monkeypatch.setenv("PATH", str(tmp_path))

    expected = [str(bin_dir / "haskie")] if has_script else [str(python), "-m", "haskie"]
    assert claude.own_command() == expected


HELD_PID = 4242


@dataclass
class StopCase:
    running: bool  # whether the home is held when `stop` looks
    dies_on: int | None  # the signal the server exits on; None for one that ignores every one
    kill: type[OSError] | None  # what signalling it raises, if anything
    exit_code: int
    expect_in_output: str
    expect_signals: list[int]


STOP_CASES = {
    "nothing running says so and signals nothing": StopCase(
        running=False,
        dies_on=None,
        kill=None,
        exit_code=0,
        expect_in_output="no haskie is running",
        expect_signals=[],
    ),
    "a server that shuts down gracefully is not forced": StopCase(
        running=True,
        dies_on=signal.SIGTERM,
        kill=None,
        exit_code=0,
        expect_in_output=f"stopped haskie (pid {HELD_PID})",
        expect_signals=[signal.SIGTERM],
    ),
    "a server held open by a connection is forced": StopCase(
        running=True,
        dies_on=signal.SIGINT,
        kill=None,
        exit_code=0,
        expect_in_output="forcing it",
        expect_signals=[signal.SIGTERM, signal.SIGINT],
    ),
    "a server stuck past the force is killed": StopCase(
        running=True,
        dies_on=signal.SIGKILL,
        kill=None,
        exit_code=0,
        expect_in_output="ignored the force; killing it",
        expect_signals=[signal.SIGTERM, signal.SIGINT, signal.SIGKILL],
    ),
    "a server that outlives even SIGKILL fails": StopCase(
        running=True,
        dies_on=None,
        kill=None,
        exit_code=1,
        expect_in_output="did not stop; kill it by hand",
        expect_signals=[signal.SIGTERM, signal.SIGINT, signal.SIGKILL],
    ),
    "a server that died first is not an error": StopCase(
        running=True,
        dies_on=None,
        kill=ProcessLookupError,
        exit_code=0,
        expect_in_output="no haskie is running",
        expect_signals=[signal.SIGTERM],
    ),
    "another user's process is refused": StopCase(
        running=True,
        dies_on=None,
        kill=PermissionError,
        exit_code=1,
        expect_in_output=f"cannot stop pid {HELD_PID}",
        expect_signals=[signal.SIGTERM],
    ),
}


@pytest.mark.parametrize("case", STOP_CASES.values(), ids=list(STOP_CASES))
def test_stop(case: StopCase, elsewhere: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`stop` reaches the server through the home lock and reads the lock back for the exit, so
    every case is a lock answer: no holder, one that goes on the graceful signal, one that only
    goes on the forced one, one that only goes on the kill, one that never goes, one already gone,
    one this user may not signal.

    The lock answers off the signals sent rather than off a clock, so no case turns on timing.
    """
    signalled: list[tuple[int, int]] = []

    def fake_kill(pid: int, number: int) -> None:
        signalled.append((pid, number))
        if case.kill is not None:
            raise case.kill

    def fake_running_pid() -> int | None:
        if not case.running:
            return None
        died = case.dies_on is not None and any(number == case.dies_on for _, number in signalled)
        return None if died else HELD_PID

    monkeypatch.setattr(home, "running_pid", fake_running_pid)
    monkeypatch.setattr(os, "kill", fake_kill)
    monkeypatch.setattr(cli_module, "POLL_INTERVAL", 0.01)
    monkeypatch.setattr(cli_module, "STOP_DEADLINE", 0.05)
    monkeypatch.setattr(cli_module, "FORCE_DEADLINE", 0.05)
    monkeypatch.setattr(cli_module, "KILL_DEADLINE", 0.05)

    result = runner.invoke(cli, ["stop", "--home", str(elsewhere)])

    assert result.exit_code == case.exit_code, result.output
    assert case.expect_in_output in _text(result)
    assert [number for _, number in signalled] == case.expect_signals
    assert all(pid == HELD_PID for pid, _ in signalled), "only the holder is signalled"


def test_stop_finds_the_process_holding_the_home(elsewhere: Path) -> None:
    """The pid `stop` signals is the one `claim_home` wrote, read back off a held lock; an
    unheld home has none to read."""
    home.use(elsewhere)

    with holding():
        assert home.running_pid() == os.getpid()
    assert home.running_pid() is None, "a released home holds no pid"


@pytest.mark.parametrize(
    "dbos_stopped",
    [True, False],
    ids=["a runtime that stopped gives the home up", "a hurried stop keeps it until the exit"],
)
def test_the_home_is_released_only_once_the_runtime_stopped(
    dbos_stopped: bool, elsewhere: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hurried stop leaves DBOS running workflows until the exit. A second haskie claiming the
    home meanwhile would run the same ones, so the lock waits for the kernel to drop it."""
    from haskie import app
    from haskie.indexing import workflows

    async def stop() -> bool:
        return dbos_stopped

    monkeypatch.setattr(workflows, "stop", stop)
    home.use(elsewhere)
    home.claim_home()
    try:
        asyncio.run(app.stop_runtime())
        assert (home.running_pid() == os.getpid()) is not dbos_stopped
    finally:
        home.release_home()


@dataclass
class HookInputCase:
    stdin: str
    expect_announced: str | None


HOOK_INPUT_CASES = {
    "a SessionStart payload announces its id": HookInputCase(
        stdin=json.dumps({"session_id": "abc-123", "source": "startup", "cwd": "/tmp"}),
        expect_announced="abc-123",
    ),
    "a payload with no usable id announces nothing": HookInputCase(
        stdin=json.dumps({"session_id": "", "source": "startup"}), expect_announced=None
    ),
    "a session id of the wrong type is not an id": HookInputCase(
        stdin=json.dumps({"session_id": 7}), expect_announced=None
    ),
    "stdin that is not a payload announces nothing": HookInputCase(
        stdin="not a hook payload", expect_announced=None
    ),
}


@pytest.mark.parametrize("case", HOOK_INPUT_CASES.values(), ids=list(HOOK_INPUT_CASES))
def test_ensure_announces_the_hook_session_id(
    case: HookInputCase, elsewhere: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing in an MCP call carries the conversation's id, so the SessionStart hook's own output
    is what puts it in the agent's context. Anything on stdin that is not a payload is ignored:
    `ensure` is also run by hand and by `install claude`."""
    monkeypatch.setattr(cli_module, "_serving", lambda _url: True)  # fast path, spawns nothing

    result = runner.invoke(
        cli, ["ensure", "--home", str(elsewhere), "--url", DEAD_URL], input=case.stdin
    )

    assert result.exit_code == 0, result.output
    output = _text(result)
    assert "already serving" in output, "the announcement does not replace the usual output"
    if case.expect_announced is None:
        assert "session id is" not in output
    else:
        assert f"session id is {case.expect_announced}" in output
        assert "`session_id`" in output, "says what to do with it"


# --- install claude ---------------------------------------------------------

CLAUDE_STUB = '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$CLAUDE_ARGV"\n'


@pytest.fixture
def claude_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory Claude Code's config can be written into, with nothing of the real
    machine in reach: `USER_CLAUDE` is redirected even for project-scope cases, so a test that
    reaches for `user` cannot write into the developer's own `~/.claude`.

    `PATH` holds only `bin`, so `claude` is present exactly when a case puts it there. Returns
    that directory; the recorded argv is `tmp_path/argv.log`.
    """
    binaries = tmp_path / "bin"
    binaries.mkdir()
    monkeypatch.setenv("PATH", str(binaries))
    monkeypatch.setenv("CLAUDE_ARGV", str(tmp_path / "argv.log"))
    monkeypatch.setattr(claude, "USER_CLAUDE", tmp_path / "user" / ".claude")
    monkeypatch.chdir(tmp_path)
    return binaries


def _with_claude(binaries: Path) -> None:
    """Put the stub `claude` CLI on the PATH `claude_workspace` prepared."""
    stub = binaries / "claude"
    stub.write_text(CLAUDE_STUB)
    stub.chmod(0o755)


@dataclass
class InstallCase:
    scope: Scope
    claude_on_path: bool
    collections: list[tuple[str, str]]
    expect_in_skill: list[str]
    expect_in_output: str


INSTALL_CASES = {
    "records the argv claude needs": InstallCase(
        scope=Scope.PROJECT,
        claude_on_path=True,
        collections=[("roasting", "Three books on coffee roasting."), ("adr", "")],
        expect_in_skill=["roasting: Three books on coffee roasting", "adr"],
        expect_in_output="registered the haskie MCP server",
    ),
    "still writes the skill without the claude cli": InstallCase(
        scope=Scope.PROJECT,
        claude_on_path=False,
        collections=[("roasting", "Coffee.")],
        expect_in_skill=["roasting: Coffee"],
        expect_in_output="register the server by hand",
    ),
    "user scope writes to the user's skills": InstallCase(
        scope=Scope.USER,
        claude_on_path=True,
        collections=[("adr", "Architecture decisions.")],
        expect_in_skill=["adr: Architecture decisions"],
        expect_in_output="registered the haskie MCP server",
    ),
    "trailing punctuation gives way to the separator": InstallCase(
        scope=Scope.PROJECT,
        claude_on_path=True,
        collections=[("Software-Architecture", "DDD, event-driven,")],
        expect_in_skill=["Software-Architecture: DDD, event-driven."],
        expect_in_output="registered the haskie MCP server",
    ),
    "an empty home still installs": InstallCase(
        scope=Scope.PROJECT,
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
    claude_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner.invoke(
        cli, ["init", "--home", str(elsewhere), "--no-browser"]
    )  # `init` points the process at it
    for name, description in case.collections:
        asyncio.run(Collection.create(name, description))
    if case.claude_on_path:
        _with_claude(claude_workspace)
    monkeypatch.setattr(cli_module, "_serving", lambda _url: True)  # never start a real server

    result = runner.invoke(
        cli, ["install", "claude", "--home", str(elsewhere), "--scope", case.scope]
    )

    assert result.exit_code == 0, result.output
    assert case.expect_in_output in _text(result)
    written = claude.skill_path(case.scope).read_text()
    rule = claude.rule_path(case.scope).read_text()
    for expected in case.expect_in_skill:
        assert expected in written
        assert expected in rule, "the rule names the same collections as the trigger"
    assert "before answering from memory" in rule, "the rule fires on knowledge questions"
    hooks = json.loads(claude.settings_path(case.scope).read_text())["hooks"]["SessionStart"]
    hooked = hooks[0]["hooks"][0]["command"]
    assert claude.HOOK_MARKER in hooked, "the hook starts haskie"
    assert hooked.endswith("--no-wait"), "a session start must not wait on a boot"
    argv_log = tmp_path / "argv.log"
    if case.claude_on_path:
        recorded = argv_log.read_text().splitlines()
        assert recorded[0] == f"mcp remove -s {case.scope} haskie", "replaced, so re-running works"
        assert recorded[1] == f"mcp add -s {case.scope} --transport http haskie {claude.MCP_URL}"
    else:
        assert not argv_log.exists()


def test_install_claude_refreshes_the_trigger_when_it_is_run_again(
    elsewhere: Path, claude_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running is how the trigger is refreshed after a collection is added."""
    runner.invoke(cli, ["init", "--home", str(elsewhere), "--no-browser"])
    asyncio.run(Collection.create("roasting", "Coffee."))
    monkeypatch.setattr(cli_module, "_serving", lambda _url: True)
    arguments = ["install", "claude", "--home", str(elsewhere), "--scope", "project"]

    runner.invoke(cli, arguments)
    asyncio.run(Collection.create("adr", "Architecture decisions."))
    again = runner.invoke(cli, arguments)

    assert again.exit_code == 0, again.output
    written = claude.skill_path(Scope.PROJECT).read_text()
    assert written.count("name: haskie") == 1, "rewritten, not appended to"
    assert "adr: Architecture decisions" in written, "the new collection reached the trigger"
    rule = claude.rule_path(Scope.PROJECT).read_text()
    assert rule.count("Search the user's own") == 1, "rewritten, not appended to"
    assert "adr: Architecture decisions" in rule, "the new collection reached the rule"


def test_install_claude_leaves_stdin_to_the_script_that_runs_it(
    elsewhere: Path, claude_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a SessionStart hook's `ensure` reads a payload from stdin. `install claude` inside a
    piped script must not read the script's next lines as one, nor wait on a pipe held open."""
    runner.invoke(cli, ["init", "--home", str(elsewhere), "--no-browser"])
    served: list[tuple[str, bool]] = []
    monkeypatch.setattr(cli_module, "_serve", lambda url, wait: served.append((url, wait)))
    payload = json.dumps({"session_id": "abc-123", "source": "startup"})

    result = runner.invoke(
        cli,
        ["install", "claude", "--home", str(elsewhere), "--scope", "project"],
        input=payload,
    )

    assert result.exit_code == 0, _text(result)
    assert "session id is" not in _text(result), "stdin was not read as a hook payload"
    assert served == [(claude.MCP_URL, True)], "brings the server up and waits for it"


SKILL_DESCRIPTION_CAP = 1536  # where Claude Code cuts a skill description in its listing


def test_a_long_trigger_line_keeps_the_fixed_triggers() -> None:
    """Claude Code cuts the description at its cap, so the collections go last: a home with many
    described collections loses its last topics, never "my documents"."""
    essay = "Every book and paper on roasting, brewing and tasting coffee, " * 4
    collections = [
        CollectionSummary(
            name=f"shelf-{n}", counts=DocumentCounts(), created_at=0.0, description=essay
        )
        for n in range(20)
    ]

    skill = claude.render_skill(collections)

    description = skill.split("description: >-\n", 1)[1].split("\n---\n", 1)[0]
    assert len(description) > SKILL_DESCRIPTION_CAP, "long enough to be cut"
    assert '"my documents"' in description[:SKILL_DESCRIPTION_CAP]
    assert "cited to something they own" in description[:SKILL_DESCRIPTION_CAP]


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:8451/mcp", "http://127.0.0.1:8451/mcp?a=1&b=2", "http://host/my mcp"],
    ids=["a plain url", "a url with a shell operator", "a url with a space"],
)
def test_a_url_stays_one_argument(url: str, claude_workspace: Path, tmp_path: Path) -> None:
    """The hook and the by-hand line are both run by a shell, so a URL that holds `&` or a space
    must reach haskie and `claude` as one argument, not split into a second command."""
    home_dir = tmp_path / "my home"

    hooked = shlex.split(claude.hook_command(home_dir, url))
    manual = claude.register_mcp(url, Scope.PROJECT)  # no `claude` on the workspace's PATH

    assert hooked[-6:] == ["ensure", "--home", str(home_dir), "--url", url, "--no-wait"]
    assert manual is not None
    assert shlex.split(manual)[-2:] == ["haskie", url]


@dataclass
class RefusedInstallCase:
    old_home: bool  # a home from before collections, which `read_collections` cannot open
    claude_fails: bool  # a `claude` CLI whose `mcp add` exits non-zero
    settings: str | None  # what `.claude/settings.json` holds before the install
    expect_error: str


REFUSED_INSTALL_CASES = {
    "a home from another schema": RefusedInstallCase(
        old_home=True, claude_fails=False, settings=None, expect_error=db.INCOMPATIBLE_HOME_MESSAGE
    ),
    "claude mcp add fails": RefusedInstallCase(
        old_home=False, claude_fails=True, settings=None, expect_error="no such scope"
    ),
    "a settings file that is not JSON": RefusedInstallCase(
        old_home=False, claude_fails=False, settings="{not json", expect_error="is not valid JSON"
    ),
}


@pytest.mark.parametrize("case", REFUSED_INSTALL_CASES.values(), ids=list(REFUSED_INSTALL_CASES))
def test_install_claude_refuses_in_one_line(
    case: RefusedInstallCase,
    elsewhere: Path,
    claude_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every failure `install claude` can meet is a `HaskieError`, and each one ends as a line on
    stderr and exit 1, never a traceback, wherever in the install it happens."""
    if case.old_home:
        _pre_collection_home(elsewhere)
    else:
        runner.invoke(cli, ["init", "--home", str(elsewhere), "--no-browser"])
    if case.claude_fails:
        stub = claude_workspace / "claude"
        stub.write_text("#!/bin/sh\necho 'error: no such scope' >&2\nexit 1\n")
        stub.chmod(0o755)
    if case.settings is not None:
        settings_file = claude.settings_path(Scope.PROJECT)
        settings_file.parent.mkdir(parents=True)
        settings_file.write_text(case.settings)
    monkeypatch.setattr(cli_module, "_serving", lambda _url: True)

    result = runner.invoke(
        cli, ["install", "claude", "--home", str(elsewhere), "--scope", "project"]
    )

    assert result.exit_code == 1, _text(result)
    assert isinstance(result.exception, SystemExit), "a refusal, not a traceback"
    assert case.expect_error in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1, "one line"
    if case.settings is not None:
        assert claude.settings_path(Scope.PROJECT).read_text() == case.settings, "left as it was"


# `install_hook` merges into a file the user owns, so its branches are worth reaching directly
# rather than through four more end-to-end installs.


@dataclass
class HookCase:
    before: str | None
    added: bool
    expect_commands: int
    keeps: str | None = None


HOOK_CASES = {
    "no settings file yet": HookCase(before=None, added=True, expect_commands=1),
    "a settings file with no hooks": HookCase(
        before='{"model": "opus"}', added=True, expect_commands=1, keeps="model"
    ),
    "a hook of ours already there": HookCase(
        before=(
            '{"hooks": {"SessionStart": [{"hooks": [{"type": "command",'
            ' "command": "/old/haskie ensure --home /old --url u --no-wait"}]}]}}'
        ),
        added=False,
        expect_commands=1,
    ),
    "somebody else's hook": HookCase(
        before='{"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "mine"}]}]}}',
        added=True,
        expect_commands=2,
    ),
}


@pytest.mark.parametrize("case", HOOK_CASES.values(), ids=list(HOOK_CASES))
def test_install_hook(case: HookCase, claude_workspace: Path, tmp_path: Path) -> None:
    settings_file = claude.settings_path(Scope.PROJECT)
    if case.before is not None:
        settings_file.parent.mkdir(parents=True)
        settings_file.write_text(case.before)

    added = claude.install_hook(Scope.PROJECT, tmp_path / "home", "http://127.0.0.1:8451/mcp")

    assert added is case.added
    settings = json.loads(settings_file.read_text())
    commands = [
        hook["command"]
        for matcher in settings["hooks"]["SessionStart"]
        for hook in matcher["hooks"]
    ]
    assert len(commands) == case.expect_commands, "ours is replaced, a stranger's is kept beside"
    assert sum(claude.HOOK_MARKER in command for command in commands) == 1, "exactly one of ours"
    if case.keeps:
        assert settings[case.keeps] == "opus", "an unrelated setting survives"


UNREADABLE_SETTINGS = {
    "not JSON": ("{not json", "is not valid JSON"),
    "not UTF-8": ('{"model": "\xff"}', "is not valid JSON"),
    "not an object": ("[]", "hooks.SessionStart"),
    "hooks is null": ('{"hooks": null}', "hooks.SessionStart"),
    "SessionStart is an object": ('{"hooks": {"SessionStart": {}}}', "hooks.SessionStart"),
    "a matcher is not an object": ('{"hooks": {"SessionStart": ["mine"]}}', "hooks.SessionStart"),
    "a matcher's hooks is not a list": (
        '{"hooks": {"SessionStart": [{"hooks": "mine"}]}}',
        "hooks.SessionStart",
    ),
    "a hook is not an object": (
        '{"hooks": {"SessionStart": [{"hooks": ["mine"]}]}}',
        "hooks.SessionStart",
    ),
}


@pytest.mark.parametrize(
    ("before", "expect_error"), UNREADABLE_SETTINGS.values(), ids=list(UNREADABLE_SETTINGS)
)
def test_install_hook_refuses_a_settings_file_it_cannot_follow(
    before: str, expect_error: str, claude_workspace: Path, tmp_path: Path
) -> None:
    """Rewriting a file we could not read would throw the user's settings away, and every failure
    here must be a `HaskieError`, never a bare `AttributeError`."""
    settings_file = claude.settings_path(Scope.PROJECT)
    settings_file.parent.mkdir(parents=True)
    settings_file.write_bytes(before.encode("latin-1"))

    with pytest.raises(InvalidInput, match=expect_error):
        claude.install_hook(Scope.PROJECT, tmp_path / "home", "http://127.0.0.1:8451/mcp")

    assert settings_file.read_bytes() == before.encode("latin-1"), "left exactly as it was"


@dataclass
class KeptFileCase:
    mode: int | None  # the existing file's mode; None when there is no file yet
    linked: bool  # whether settings.json is a link into a dotfiles directory
    expect_mode: int | None  # None: what a new file gets under the process umask


KEPT_FILE_CASES = {
    "no file yet gets the default mode": KeptFileCase(mode=None, linked=False, expect_mode=None),
    "a private file stays private": KeptFileCase(mode=0o600, linked=False, expect_mode=0o600),
    "a read-only file is still rewritten": KeptFileCase(
        mode=0o400, linked=False, expect_mode=0o400
    ),
    "a linked file stays linked": KeptFileCase(mode=0o600, linked=True, expect_mode=0o600),
}


@pytest.mark.parametrize("case", KEPT_FILE_CASES.values(), ids=list(KEPT_FILE_CASES))
def test_install_hook_keeps_the_settings_file_what_it_was(
    case: KeptFileCase, claude_workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rewrite is a rename, which replaces whatever sits at the path. A settings file that is a
    link into a dotfiles repository must stay one, and one kept private (it can hold API keys) must
    not come back world-readable."""
    settings_file = claude.settings_path(Scope.PROJECT)
    settings_file.parent.mkdir(parents=True)
    real = tmp_path / "dotfiles" / "settings.json" if case.linked else settings_file
    if case.mode is not None:
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_text('{"env": {"API_KEY": "sk-test"}}')
        real.chmod(case.mode)
    if case.linked:
        settings_file.symlink_to(real)

    claude.install_hook(Scope.PROJECT, tmp_path / "home", "http://127.0.0.1:8451/mcp")

    assert settings_file.is_symlink() is case.linked
    umask = os.umask(0)
    os.umask(umask)
    expect_mode = 0o666 & ~umask if case.expect_mode is None else case.expect_mode
    assert stat.S_IMODE(real.stat().st_mode) == expect_mode
    written = json.loads(real.read_text())
    assert claude.HOOK_MARKER in written["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    if case.mode is not None:
        assert written["env"] == {"API_KEY": "sk-test"}, "the rest of the file survives"


def _served_tools() -> set[str]:
    return {
        name
        for module in (Path(__file__).parents[1] / "src" / "haskie" / "api").glob("*.py")
        for name in re.findall(r'mcp_tool="([^"]+)"', module.read_text(encoding="utf-8"))
    }


TOOL_VERBS = ("list_", "get_", "add_", "remove_", "set_", "search_", "describe_")
# A backticked word, with or without an argument list after it: `search_excerpts(q, ...)`.
BACKTICKED = re.compile(r"`([a-z_]+)(?:\([^`]*\))?`")


@pytest.mark.parametrize(
    "prose", [claude.render_skill([]), claude.render_rule([])], ids=["skill", "rule"]
)
def test_the_prose_only_names_tools_the_server_actually_serves(prose: str) -> None:
    """The skill and the rule tell Claude which tool to reach for, so a renamed tool makes them
    wrong in a way nothing else catches: both are prose, written at install time. Field names
    are backticked too, so only words shaped like a tool are held to the served set."""
    named = {word for word in BACKTICKED.findall(prose) if word.startswith(TOOL_VERBS)}

    assert named, "the prose is supposed to name the tools"
    served = _served_tools()
    assert named <= served, f"it names tools that do not exist: {sorted(named - served)}"


def test_the_skill_names_every_tool_the_server_serves() -> None:
    """A new tool that the skill never mentions is one Claude never reaches for."""
    named = set(BACKTICKED.findall(claude.render_skill([])))
    served = _served_tools()

    assert served <= named, f"the skill never mentions: {sorted(served - named)}"


def test_the_default_url_matches_where_mcp_is_mounted() -> None:
    """`claude.MCP_URL` spells the path out rather than importing the app, which would cost every
    `haskie` invocation the whole web stack. This is what keeps the two in step."""
    from haskie import app

    assert cli_module.MCP_URL.endswith(app.MCP_PATH)


@pytest.mark.parametrize(
    ("port", "expected"),
    [(None, "http://127.0.0.1:8451/mcp"), ("8452", "http://127.0.0.1:8452/mcp")],
    ids=["unset: the installed haskie's 8451", "HASKIE_PORT: a development install beside it"],
)
def test_haskie_port_moves_the_default_url(port: str | None, expected: str) -> None:
    """Read once at import, so only a fresh interpreter shows it."""
    env = {key: value for key, value in os.environ.items() if key != "HASKIE_PORT"}
    if port is not None:
        env["HASKIE_PORT"] = port
    printed = subprocess.run(
        [sys.executable, "-c", "from haskie import claude; print(claude.MCP_URL)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert printed == expected


@pytest.mark.parametrize(
    ("config_dir", "expected"),
    [(None, "home/.claude"), ("", "home/.claude"), ("~/work-claude", "home/work-claude")],
    ids=["unset: ~/.claude", "empty: ~/.claude", "CLAUDE_CONFIG_DIR: where Claude Code moved it"],
)
def test_claude_config_dir_moves_the_user_scope(
    config_dir: str | None, expected: str, tmp_path: Path
) -> None:
    """Read once at import, so only a fresh interpreter shows it."""
    env = {key: value for key, value in os.environ.items() if key != "CLAUDE_CONFIG_DIR"}
    env["HOME"] = str(tmp_path / "home")
    if config_dir is not None:
        env["CLAUDE_CONFIG_DIR"] = config_dir
    printed = subprocess.run(
        [sys.executable, "-c", "from haskie import claude; print(claude.USER_CLAUDE)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert printed == str(tmp_path / expected)
