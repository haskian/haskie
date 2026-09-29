"""The `haskie` command.

`destroy` deletes a user's whole document store, so its guards are the point of this module: it
refuses a directory that is not a haskie home, and it asks before it deletes. `run` is the other
guard: a home written before documents became collection-independent is refused, not migrated.

`install claude` and the home lock are the other two: one writes into a user's Claude Code
configuration, the other is what keeps a second haskie off a home a first one is already running.
"""

import asyncio
import contextlib
import json
import os
import re
import shlex
import signal
import sqlite3
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import anyio
import pytest
from conftest import claude_installed, fresh_attribute, holding, refresh_settled, until
from sqlalchemy import select
from typer.testing import CliRunner

from haskie import APP_VERSION, claude, db, home
from haskie import cli as cli_module
from haskie.claude import Scope
from haskie.cli import cli
from haskie.collection.collection import Collection, CollectionSummary, DocumentCounts
from haskie.errors import Conflict, InvalidInput
from haskie.tables import installations

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


def _serves(root: Path, **status: object):
    """A stand-in for `cli._status`: a haskie of `root` answers at every address."""
    return lambda _url: {"home": str(root.resolve()), "web_ui": True, **status}


def _make_home(root: Path) -> None:
    """A home as a first server start leaves it, with the process pointed at it."""
    home.use(root.resolve())
    asyncio.run(db.migrate_once())


def test_run_refuses_a_home_from_before_collections(
    elsewhere: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No migration path exists for the old storage shape, so the user is told to destroy it
    rather than losing rows to a silent drop. In one line, before the server starts: its own
    refusal is a traceback. A detached `run` shows this line from the log (see `test_run`)."""
    _pre_collection_home(elsewhere)
    started: list[object] = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **_: started.append(args))

    refused = runner.invoke(cli, ["run", "--home", str(elsewhere), "--foreground"])

    assert refused.exit_code == 1
    assert refused.stderr.strip() == db.INCOMPATIBLE_HOME_MESSAGE, "one line, on stderr"
    assert started == [], "no server started"
    with sqlite3.connect(elsewhere / "haskie.db") as conn:
        (version,) = conn.execute("pragma user_version").fetchone()
    assert version == db.SCHEMA_VERSION - 1, "the refused home is left as it was"


def test_destroy_after_a_refused_run_lets_it_start_over(elsewhere: Path) -> None:
    """The recovery path the message prescribes has to actually work."""
    _pre_collection_home(elsewhere)

    assert runner.invoke(cli, ["destroy", "--home", str(elsewhere), "--yes"]).exit_code == 0
    _make_home(elsewhere)

    db.check_schema()  # would raise for a home still at the old version
    assert (elsewhere / "collections").is_dir()


@pytest.mark.parametrize(
    "given", [True, False], ids=["--home through a link", "the default home through a link"]
)
def test_run_knows_its_own_server_through_a_linked_home(
    given: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`~/.haskie` may be a link into a synced folder. The server `run` starts is given the
    resolved root and reports that back, so `run` must resolve the home it compares, whether it
    came from `--home` or from the default, or it calls its own server another home's."""
    import webbrowser

    real = tmp_path / "synced" / "haskie"
    real.mkdir(parents=True)
    link = tmp_path / ".haskie"
    link.symlink_to(real)
    home.use(link)  # the default, as `~/.haskie` would be
    monkeypatch.setattr(cli_module, "_status", _serves(real, initialized=False))
    monkeypatch.setattr(webbrowser, "open", lambda _url: True)

    result = runner.invoke(cli, ["run", *(["--home", str(link)] if given else [])])

    assert result.exit_code == 0, _text(result)
    assert "serves another home" not in _text(result)
    assert home.HOME == real.resolve()
    assert os.environ["HASKIE_HOME"] == str(real.resolve()), "a --reload child reads the same root"


def test_destroy_asks_first_and_leaves_everything_when_refused(elsewhere: Path) -> None:
    _make_home(elsewhere)
    _shelve(home.COLLECTION_ROOT, "notes")
    _shelve(home.DOCUMENT_ROOT, "guide.md")

    refused = runner.invoke(cli, ["destroy", "--home", str(elsewhere)], input="n\n")

    assert refused.exit_code == 1, "abort is a failure exit, not a silent no-op"
    assert elsewhere.is_dir(), "nothing deleted"
    summary = _text(refused)
    assert "collections: notes" in summary, "the summary names what would be lost"
    assert "1 documents" in summary, "and how many documents go with it"


def test_destroy_deletes_the_home_when_confirmed(elsewhere: Path) -> None:
    _make_home(elsewhere)

    done = runner.invoke(cli, ["destroy", "--home", str(elsewhere)], input="y\n")

    assert done.exit_code == 0, _text(done)
    assert not elsewhere.exists()


def test_destroy_yes_skips_the_prompt(elsewhere: Path) -> None:
    _make_home(elsewhere)

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
    _make_home(elsewhere)

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
    _make_home(elsewhere)
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


def test_a_home_made_after_destroy_migrates_again(elsewhere: Path) -> None:
    """`destroy` clears the per-process "already migrated" set, or the new home would have no
    schema."""
    _make_home(elsewhere)
    runner.invoke(cli, ["destroy", "--home", str(elsewhere), "--yes"])

    _make_home(elsewhere)

    with sqlite3.connect(elsewhere / "haskie.db") as conn:
        (version,) = conn.execute("pragma user_version").fetchone()
    assert version == db.SCHEMA_VERSION, "the schema was applied to the new file"


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


def test_claim_home_refuses_a_second_holder_and_says_who_has_it(
    elsewhere: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`flock` is per open file description, so a second claim from this process conflicts exactly
    as a second process would - no subprocess needed to prove the guard. No database either: the
    lock needs none, and makes the home itself."""
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
        refused = runner.invoke(cli, ["run", "--home", str(elsewhere), "--foreground"])

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

THIS_HOME = "this home"  # stands for the home the command was given, as the server reports it
DONE: dict[str, object] = {"home": THIS_HOME, "initialized": True, "web_ui": True}
FIRST_RUN: dict[str, object] = {"home": THIS_HOME, "initialized": False, "web_ui": True}


@dataclass
class RunCase:
    before: dict[str, object] | None  # `/api/status` before anything starts; None: nothing serves
    after: dict[str, object] | None  # `/api/status` once a child is spawned
    exit_code: int
    expect_in_output: list[str]
    expect_spawn: bool
    expect_opened: bool = False
    flags: list[str] = field(default_factory=list)
    child_exits: bool = False  # the spawned child is gone by the first check
    held_at: str | None = None  # the address of another haskie holding this home, if one does
    deadline: float = 5.0


RUN_CASES = {
    "a served home costs one probe and starts nothing": RunCase(
        before=DONE,
        after=None,
        exit_code=0,
        expect_in_output=["already serving", "web UI at"],
        expect_spawn=False,
    ),
    "a first run not done opens the UI on it": RunCase(
        before=FIRST_RUN,
        after=None,
        exit_code=0,
        expect_in_output=["pick the embedding model and the search at"],
        expect_spawn=False,
        expect_opened=True,
    ),
    "no browser: only where to go": RunCase(
        before=FIRST_RUN,
        after=None,
        exit_code=0,
        expect_in_output=["pick the embedding model and the search at"],
        expect_spawn=False,
        flags=["--no-browser"],
    ),
    "another home serves the port: refused, nothing opened": RunCase(
        before={"home": "/elsewhere", "initialized": False},
        after=None,
        exit_code=1,
        expect_in_output=["serves another home (/elsewhere)"],
        expect_spawn=False,
    ),
    "a build without the web UI says how to get it": RunCase(
        before={**DONE, "web_ui": False},
        after=None,
        exit_code=0,
        expect_in_output=["no web UI at", "mise run build"],
        expect_spawn=False,
    ),
    "nothing serving: started and waited for": RunCase(
        before=None,
        after=DONE,
        exit_code=0,
        expect_in_output=["starting haskie", "haskie is serving", "web UI at"],
        expect_spawn=True,
    ),
    "the hook returns once it has spawned": RunCase(
        before=None,
        after=None,
        exit_code=0,
        expect_in_output=["starting haskie"],
        expect_spawn=True,
        flags=["--hook"],
    ),
    "a server that never answers fails": RunCase(
        before=None,
        after=None,
        exit_code=1,
        expect_in_output=["did not come up"],
        expect_spawn=True,
        deadline=0.0,
    ),
    "a child that exits while starting shows its last words": RunCase(
        before=None,
        after=None,
        exit_code=1,
        expect_in_output=["exited while starting", "address already in use"],
        expect_spawn=True,
        child_exits=True,
    ),
    "a race lost to a winner that comes up is served": RunCase(
        before=None,
        after=DONE,
        exit_code=0,
        expect_in_output=["haskie is serving"],
        expect_spawn=True,
        child_exits=True,
        held_at="http://127.0.0.1:9",
    ),
    "the home held at another address: the child's refusal, at once": RunCase(
        before=None,
        after=None,
        exit_code=1,
        expect_in_output=["exited while starting", "already running"],
        expect_spawn=True,
        child_exits=True,
        held_at="http://127.0.0.1:8451",
    ),
}


@pytest.mark.parametrize("case", RUN_CASES.values(), ids=list(RUN_CASES))
def test_run(case: RunCase, elsewhere: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`run` starts every Claude Code session and every first run, so its fast path spawns
    nothing, its slow path fails rather than hangs, and it opens the UI only where a first run is
    still to be finished, on a server of this home."""
    import webbrowser

    spawned: list[list[str]] = []
    browsed: list[str] = []
    this = str(elsewhere.resolve())

    class Child:
        """A spawned server: it writes to the log it is given, and may already be gone."""

        def __init__(self, command: list[str], stdout, **_) -> None:
            spawned.append(command)
            if case.child_exits:  # the line its refusal leaves in the log
                stdout.write(
                    b"haskie is already running for this home\n"
                    if case.held_at
                    else b"ERROR: [Errno 48] address already in use\n"
                )

        def poll(self) -> int | None:
            return 1 if case.child_exits else None

    def status(_url: str) -> dict[str, object] | None:
        answer = case.after if spawned else case.before
        if answer is None:
            return None
        return {**answer, "home": this if answer["home"] == THIS_HOME else answer["home"]}

    monkeypatch.setattr(cli_module, "_status", status)
    monkeypatch.setattr(webbrowser, "open", lambda url: browsed.append(url) or True)
    monkeypatch.setattr(subprocess, "Popen", Child)
    monkeypatch.setattr(cli_module, "POLL_INTERVAL", 0.01)
    monkeypatch.setattr(cli_module, "START_DEADLINE", case.deadline)

    home.use(elsewhere.resolve())
    with holding(case.held_at) if case.held_at else contextlib.nullcontext():
        result = runner.invoke(cli, ["run", "--home", str(elsewhere), "--port", "9", *case.flags])

    assert result.exit_code == case.exit_code, _text(result)
    for expected in case.expect_in_output:
        assert expected in _text(result), expected
    assert bool(spawned) is case.expect_spawn
    if case.expect_spawn:
        prefix = claude.own_command()
        assert spawned[0][: len(prefix) + 1] == [*prefix, "run"], "spawned through this haskie"
        assert spawned[0][-1] == "--foreground", "serving in its own process"
        assert this in spawned[0], "the child serves the same home"
    assert browsed == (["http://127.0.0.1:9/"] if case.expect_opened else [])
    if case.child_exits and case.exit_code:
        assert str(elsewhere / "server.log") in result.stderr, "names the log to read"


@pytest.mark.parametrize(
    "has_script", [True, False], ids=["the script beside the interpreter", "no script: -m"]
)
def test_own_command_is_this_install_not_the_first_on_path(
    has_script: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`haskie-dev` and an installed `haskie` sit on one PATH. Whichever runs `run` must spawn
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
    hook: bool = True  # whether the command runs as the hook, the only caller whose stdin is read


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
    "a payload without --hook is never read": HookInputCase(
        stdin=json.dumps({"session_id": "abc-123", "source": "startup"}),
        expect_announced=None,
        hook=False,
    ),
}


@pytest.mark.parametrize("case", HOOK_INPUT_CASES.values(), ids=list(HOOK_INPUT_CASES))
def test_run_announces_the_hook_session_id(
    case: HookInputCase, elsewhere: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing in an MCP call carries the conversation's id, so the SessionStart hook's own output
    is what puts it in the agent's context. Only `--hook` reads stdin: `run` by hand or in a
    script may have a pipe there that nobody closes, and anything that is not a payload is
    ignored."""
    monkeypatch.setattr(cli_module, "_status", _serves(elsewhere, initialized=True))  # fast path
    flags = ["--hook"] if case.hook else []

    result = runner.invoke(
        cli, ["run", "--home", str(elsewhere), "--port", "9", *flags], input=case.stdin
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
    _make_home(elsewhere)
    for name, description in case.collections:
        asyncio.run(Collection.create(name, description))
    if case.claude_on_path:
        _with_claude(claude_workspace)
    monkeypatch.setattr(cli_module, "_status", _serves(elsewhere))  # never start a real server

    result = runner.invoke(
        cli, ["install", "claude", "--home", str(elsewhere), "--scope", case.scope]
    )

    assert result.exit_code == 0, result.output
    assert case.expect_in_output in _text(result)
    directory = claude.claude_dir(case.scope)
    written = claude.skill_path(directory).read_text()
    rule = claude.rule_path(directory).read_text()
    assert asyncio.run(_installations()) == [str(directory)], "recorded, to refresh it later"
    for expected in case.expect_in_skill:
        assert expected in written
        assert expected in rule, "the rule names the same collections as the trigger"
    assert "before answering from memory" in rule, "the rule fires on knowledge questions"
    hooks = json.loads(claude.settings_path(claude.claude_dir(case.scope)).read_text())["hooks"][
        "SessionStart"
    ]
    hooked = hooks[0]["hooks"][0]["command"]
    assert claude.HOOK_MARKERS[0] in hooked, "the hook starts haskie"
    assert hooked.endswith("--hook"), "a session start reads its payload, never waits on a boot"
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
    _make_home(elsewhere)
    asyncio.run(Collection.create("roasting", "Coffee."))
    monkeypatch.setattr(cli_module, "_status", _serves(elsewhere))
    arguments = ["install", "claude", "--home", str(elsewhere), "--scope", "project"]

    runner.invoke(cli, arguments)
    asyncio.run(Collection.create("adr", "Architecture decisions."))
    again = runner.invoke(cli, arguments)

    assert again.exit_code == 0, again.output
    directory = claude.claude_dir(Scope.PROJECT)
    written = claude.skill_path(directory).read_text()
    assert written.count("name: haskie") == 1, "rewritten, not appended to"
    assert "adr: Architecture decisions" in written, "the new collection reached the trigger"
    assert asyncio.run(_installations()) == [str(directory)], "recorded once"
    rule = claude.rule_path(directory).read_text()
    assert rule.count("Search the user's own") == 1, "rewritten, not appended to"
    assert "adr: Architecture decisions" in rule, "the new collection reached the rule"


async def _installations() -> list[str]:
    """The directories recorded for Claude Code, in the current home."""
    async with db.read() as conn:
        return list(await conn.scalars(select(installations.c.directory).order_by("directory")))


@dataclass
class RefreshCase:
    files_kept: bool = True  # False: removed by hand, or with their whole project
    hooked_home: str | None = None  # another home's install took the hook over
    writable: bool = True
    home_name: str = "home"
    expect_written: bool = True


REFRESH_CASES = {
    "an installation still ours is rewritten": RefreshCase(),
    "a home whose path needs quoting still finds its hook": RefreshCase(home_name="my home"),
    "a skill and rule removed stay removed": RefreshCase(files_kept=False, expect_written=False),
    "a directory another home installed into is left to it": RefreshCase(
        hooked_home="other-home", expect_written=False
    ),
    "a directory that refuses the write is tried again next change": RefreshCase(
        writable=False, expect_written=False
    ),
}


@pytest.mark.anyio
@pytest.mark.parametrize("case", REFRESH_CASES.values(), ids=list(REFRESH_CASES))
async def test_refresh_installations(case: RefreshCase, tmp_path: Path) -> None:
    """Every installation is refreshed on its own: the one under test sits beside one that always
    works, which must be rewritten whatever happens to the first. None is ever forgotten: only
    `uninstall` does that."""
    home.use((tmp_path / case.home_name).resolve())
    await Collection.create("roasting", "Three books on coffee roasting.")
    hooked = None if case.hooked_home is None else tmp_path / case.hooked_home
    tested = claude_installed(tmp_path / "tested" / ".claude", hooked)
    beside = claude_installed(tmp_path / "beside" / ".claude")
    if not case.files_kept:
        claude.skill_path(tested).unlink()
        claude.rule_path(tested).unlink()
    locked = claude.skill_path(tested).parent
    if not case.writable:
        locked.chmod(0o500)
    for directory in (tested, beside):
        await claude.record_installation(directory)

    try:
        await claude.refresh_installations()
    finally:
        locked.chmod(0o700)

    assert "roasting: Three books on coffee roasting" in claude.rule_path(beside).read_text()
    skill = claude.skill_path(tested)
    assert skill.is_file() == case.files_kept, "a refresh never makes nor removes a file"
    assert (skill.is_file() and "roasting" in skill.read_text()) == case.expect_written
    assert await _installations() == sorted([str(beside), str(tested)])


@pytest.mark.anyio
async def test_changes_during_a_refresh_coalesce_into_one_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A change landing while a refresh runs gets one more round after it, because the running one
    may have read the collections before the change; three such changes still cost one round."""
    rounds: list[int] = []
    first_entered, release = anyio.Event(), anyio.Event()

    async def refresh() -> None:
        rounds.append(len(rounds))
        if len(rounds) == 1:
            first_entered.set()
            await release.wait()

    monkeypatch.setattr(claude, "refresh_installations", refresh)

    claude.refresh_in_background()
    await first_entered.wait()
    for _ in range(3):
        claude.refresh_in_background()
    release.set()
    await until(refresh_settled, "the refresh task finished")

    assert rounds == [0, 1], "one round for the first request, one for the three during it"


def test_install_claude_leaves_stdin_to_the_script_that_runs_it(
    elsewhere: Path, claude_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a SessionStart hook's `run --hook` reads a payload from stdin. `install claude` inside a
    piped script must not read the script's next lines as one, nor wait on a pipe held open."""
    _make_home(elsewhere)
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


USER_SETTINGS = {"model": "opus"}  # a setting of the user's that no install may touch
USER_HOOK = {"type": "command", "command": "echo hello"}


@dataclass
class UninstallCase:
    installed: bool = True
    claude_on_path: bool = True
    user_hook_beside: bool = False  # the user's own hook in haskie's matcher
    user_file_in_skill_folder: bool = False
    times: int = 1
    expect_in_output: list[str] = field(default_factory=list)
    expect_not_in_output: list[str] = field(default_factory=list)
    expect_settings: dict = field(default_factory=lambda: dict(USER_SETTINGS))


UNINSTALL_CASES = {
    "undoes an install": UninstallCase(
        expect_in_output=[
            "stopped refreshing",
            "removed the haskie MCP server (project scope)",
            "removed the SessionStart hook",
            "SKILL.md",
            "haskie.md",
        ],
    ),
    "a hook of the user's beside haskie's stays": UninstallCase(
        user_hook_beside=True,
        expect_settings={**USER_SETTINGS, "hooks": {"SessionStart": [{"hooks": [USER_HOOK]}]}},
    ),
    "a file of the user's keeps the skill folder": UninstallCase(user_file_in_skill_folder=True),
    "without the claude cli it prints the command": UninstallCase(
        claude_on_path=False,
        expect_in_output=["remove the server by hand", "claude mcp remove -s project haskie"],
    ),
    "a second uninstall finds nothing to do": UninstallCase(
        times=2,
        expect_in_output=["removed the haskie MCP server"],
        expect_not_in_output=["stopped refreshing", "removed the SessionStart hook", "SKILL.md"],
    ),
    "nothing installed does nothing and makes no home": UninstallCase(
        installed=False,
        expect_in_output=["removed the haskie MCP server"],
        expect_not_in_output=["stopped refreshing", "removed the SessionStart hook"],
        expect_settings=USER_SETTINGS,
    ),
}


@pytest.mark.parametrize("case", UNINSTALL_CASES.values(), ids=list(UNINSTALL_CASES))
def test_uninstall_claude(
    case: UninstallCase,
    elsewhere: Path,
    tmp_path: Path,
    claude_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = claude.claude_dir(Scope.PROJECT)
    settings_file = claude.settings_path(claude.claude_dir(Scope.PROJECT))
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text(json.dumps(USER_SETTINGS))
    if case.claude_on_path:
        _with_claude(claude_workspace)
    if case.installed:
        _make_home(elsewhere)
        asyncio.run(Collection.create("roasting", "Coffee."))
        monkeypatch.setattr(cli_module, "_status", _serves(elsewhere))
        installed = runner.invoke(
            cli, ["install", "claude", "--home", str(elsewhere), "--scope", "project"]
        )
        assert installed.exit_code == 0, _text(installed)
    if case.user_hook_beside:
        settings = json.loads(settings_file.read_text())
        settings["hooks"]["SessionStart"][0]["hooks"].append(USER_HOOK)
        settings_file.write_text(json.dumps(settings))
    user_file = claude.skill_path(directory).parent / "notes.md"
    if case.user_file_in_skill_folder:
        user_file.write_text("mine")

    arguments = ["uninstall", "claude", "--home", str(elsewhere), "--scope", "project"]
    results = [runner.invoke(cli, arguments) for _ in range(case.times)]

    for result in results:
        assert result.exit_code == 0, _text(result)
    output = _text(results[-1])
    for expected in case.expect_in_output:
        assert expected in output
    for unexpected in case.expect_not_in_output:
        assert unexpected not in output
    assert json.loads(settings_file.read_text()) == case.expect_settings
    assert not claude.skill_path(directory).exists()
    assert not claude.rule_path(directory).exists()
    assert claude.skill_path(directory).parent.exists() == case.user_file_in_skill_folder
    if case.installed:
        assert asyncio.run(_installations()) == [], "a collection change no longer rewrites it"
    else:
        assert not elsewhere.exists(), "uninstalling never makes a home"
    if case.claude_on_path:
        recorded = (tmp_path / "argv.log").read_text().splitlines()
        assert recorded[-1] == "mcp remove -s project haskie"


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
    """The hook and the by-hand line are both run by a shell, so a home or a URL that holds `&` or
    a space must reach haskie and `claude` as one argument, not split into a second command."""
    home_dir = tmp_path / "my home"

    hooked = shlex.split(claude.hook_command(home_dir, url))
    manual = claude.register_mcp(url, Scope.PROJECT)  # no `claude` on the workspace's PATH

    parts = urlsplit(url)
    assert hooked[-8:] == [
        "run",
        "--home",
        str(home_dir),
        "--host",
        parts.hostname,
        "--port",
        str(parts.port or claude.DEFAULT_PORT),
        "--hook",
    ]
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
        _make_home(elsewhere)
    if case.claude_fails:
        stub = claude_workspace / "claude"
        stub.write_text("#!/bin/sh\necho 'error: no such scope' >&2\nexit 1\n")
        stub.chmod(0o755)
    if case.settings is not None:
        settings_file = claude.settings_path(claude.claude_dir(Scope.PROJECT))
        settings_file.parent.mkdir(parents=True)
        settings_file.write_text(case.settings)
    monkeypatch.setattr(cli_module, "_status", _serves(elsewhere))

    result = runner.invoke(
        cli, ["install", "claude", "--home", str(elsewhere), "--scope", "project"]
    )

    assert result.exit_code == 1, _text(result)
    assert isinstance(result.exception, SystemExit), "a refusal, not a traceback"
    assert case.expect_error in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1, "one line"
    if case.settings is not None:
        assert (
            claude.settings_path(claude.claude_dir(Scope.PROJECT)).read_text() == case.settings
        ), "left as it was"


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
            ' "command": "/old/haskie run --home /old --host h --port 1 --hook"}]}]}}'
        ),
        added=False,
        expect_commands=1,
    ),
    "a hook of ours from before `run`": HookCase(
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
    settings_file = claude.settings_path(claude.claude_dir(Scope.PROJECT))
    if case.before is not None:
        settings_file.parent.mkdir(parents=True)
        settings_file.write_text(case.before)

    added = claude.install_hook(
        claude.claude_dir(Scope.PROJECT), tmp_path / "home", "http://127.0.0.1:8451/mcp"
    )

    assert added is case.added
    settings = json.loads(settings_file.read_text())
    commands = [
        hook["command"]
        for matcher in settings["hooks"]["SessionStart"]
        for hook in matcher["hooks"]
    ]
    assert len(commands) == case.expect_commands, "ours is replaced, a stranger's is kept beside"
    assert sum(claude.HOOK_MARKERS[0] in command for command in commands) == 1, (
        "exactly one of ours"
    )
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
    settings_file = claude.settings_path(claude.claude_dir(Scope.PROJECT))
    settings_file.parent.mkdir(parents=True)
    settings_file.write_bytes(before.encode("latin-1"))

    with pytest.raises(InvalidInput, match=expect_error):
        claude.install_hook(
            claude.claude_dir(Scope.PROJECT), tmp_path / "home", "http://127.0.0.1:8451/mcp"
        )

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
    settings_file = claude.settings_path(claude.claude_dir(Scope.PROJECT))
    settings_file.parent.mkdir(parents=True)
    real = tmp_path / "dotfiles" / "settings.json" if case.linked else settings_file
    if case.mode is not None:
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_text('{"env": {"API_KEY": "sk-test"}}')
        real.chmod(case.mode)
    if case.linked:
        settings_file.symlink_to(real)

    claude.install_hook(
        claude.claude_dir(Scope.PROJECT), tmp_path / "home", "http://127.0.0.1:8451/mcp"
    )

    assert settings_file.is_symlink() is case.linked
    umask = os.umask(0)
    os.umask(umask)
    expect_mode = 0o666 & ~umask if case.expect_mode is None else case.expect_mode
    assert stat.S_IMODE(real.stat().st_mode) == expect_mode
    written = json.loads(real.read_text())
    assert claude.HOOK_MARKERS[0] in written["hooks"]["SessionStart"][0]["hooks"][0]["command"]
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
    printed = fresh_attribute("haskie.claude", "MCP_URL", {"HASKIE_PORT": port})
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
    env = {"HOME": str(tmp_path / "home"), "CLAUDE_CONFIG_DIR": config_dir}
    printed = fresh_attribute("haskie.claude", "USER_CLAUDE", env)
    assert printed == str(tmp_path / expected)
