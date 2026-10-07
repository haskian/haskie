"""The `haskie` command: serve the app, stop it, and install it into a client.

Thin on purpose. The CLI calls what the app already does at startup rather than repeating it. The
server makes the home and its schema, the web UI picks the first-run settings, and `run` is
uvicorn over `app:create_app`. So the CLI adds a way in, never a second way of doing the work.
"""

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from io import BufferedIOBase
from pathlib import Path
from typing import Annotated, Any, cast
from urllib.parse import urlsplit

import typer

from haskie import APP_VERSION, claude, home, shutdown
from haskie.claude import DEFAULT_HOST, DEFAULT_PORT, MCP_URL, Scope
from haskie.errors import HaskieError

cli = typer.Typer(
    name="haskie",
    help="Your documents: markdown conversion, LanceDB search, web UI and MCP server.",
    no_args_is_help=True,
    add_completion=False,
)


VERSION_LINE = f"haskie {APP_VERSION}"


def _print_version(shown: bool) -> None:
    """`--version` before anything else: eager, so it answers without a subcommand and without
    touching the home directory."""
    if shown:
        typer.echo(VERSION_LINE)
        raise typer.Exit()


@cli.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Print the version and exit.",
        ),
    ] = False,
) -> None:
    """Root options. `haskie version` says the same thing and also where the data lives."""


HomeOption = Annotated[
    Path | None,
    typer.Option(
        "--home",
        envvar="HASKIE_HOME",
        help="Where haskie keeps its data. Defaults to ~/.haskie.",
        show_default=False,
    ),
]


def _use_home(path: Path | None) -> None:
    """Point the process at `path`, or at the default home, before anything reads the home layout.
    The environment carries it too, so a `--reload` child re-reads the same root (see `home.HOME`).

    Resolved either way: a server started from here reports the resolved root, so a default reached
    through a link (`~/.haskie` into a synced folder) must compare as the same home.
    """
    resolved = str(Path(home.HOME if path is None else path).expanduser().resolve())
    os.environ["HASKIE_HOME"] = resolved
    home.use(Path(resolved))


SHUTDOWN_DRAIN = 10.0  # what `run` gives uvicorn to finish the requests in flight when it is told


@cli.command()
def run(
    home_dir: HomeOption = None,
    host: Annotated[str, typer.Option(help="Interface to bind.")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port to listen on.")] = DEFAULT_PORT,
    browser: Annotated[
        bool, typer.Option(help="Open the web UI when the first run still needs its settings.")
    ] = True,
    foreground: Annotated[
        bool, typer.Option(help="Serve in this process until stopped, for a supervisor.")
    ] = False,
    reload: Annotated[
        bool, typer.Option(help="Restart on code changes; serves in the foreground (development).")
    ] = False,
    hook: Annotated[
        bool,
        typer.Option(
            help="Run as an agent's SessionStart hook: read its payload, do not wait.",
            hidden=True,
        ),
    ] = False,
    hook_rules: Annotated[
        Path | None, typer.Option(hidden=True, help="Instructions to emit in hook context.")
    ] = None,
) -> None:
    """Serve the web UI, the REST API and the MCP server, and finish the first run in the browser.

    One process serves all three, in the background, and outlives this command: the next Claude
    Code session reuses it, and the web UI stays up. A haskie already serving the address is left
    as it is, so this is safe to repeat. The server makes the home and its database; a home whose
    first run has not picked the embedding model and the search opens the web UI on that page.

    Claude Code's SessionStart hook passes `--hook`, which does not wait: a client connects to the
    MCP endpoint while the hook is still running, so waiting cannot help the session that starts
    it, and a haskie that fails to boot would stall every session start. Only the hook's stdin is
    read: anywhere else stdin may be a pipe that nobody writes to or closes, and reading it would
    hang the command.

    Binds loopback by default: the home directory is one user's documents, and nothing in the app
    authenticates a caller.
    """
    _use_home(home_dir)
    if foreground or reload:
        _serve_here(host, port, reload)
        return
    session_id = _hook_session_id() if hook else None
    if session_id is not None:
        # A SessionStart hook's output becomes context for the session it starts, which is the one
        # place the conversation's own id can reach the tools: nothing in an MCP call carries it,
        # so without this line every search is recorded against no session at all.
        typer.echo(claude.session_announcement(session_id))
    if hook and hook_rules is not None:
        typer.echo(hook_rules.read_text(encoding="utf-8"))
    url = f"http://{host}:{port}"
    status = _serve(url, wait=not hook)
    if status is not None and not hook:
        _greet(url, status, browser)


def _serve_here(host: str, port: int, reload: bool) -> None:
    """uvicorn over the app, in this process, until a signal ends it."""
    import uvicorn

    # The app's first startup hook is what claims the home; this only asks, so that the common
    # refusal is one line here instead of a lifespan traceback out of uvicorn.
    held = home.home_holder()
    if held is not None:
        typer.echo(held, err=True)
        raise typer.Exit(code=1)
    # The server refuses a home another schema wrote as well, but as a traceback in its log. Here
    # it is one line, which `_serve` shows when this runs as its detached child.
    from haskie import db  # the database stack, only once a server is about to start

    try:
        db.check_schema()
    except HaskieError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    # The address the startup hook records, for the next process's message. The environment is the
    # one carrier, so a `--reload` child that re-imports `home` records the same thing.
    os.environ[home.ADDRESS_ENV] = f"http://{host}:{port}"
    os.environ[home.SERVER_PID_ENV] = str(os.getpid())
    if not reload:  # `--reload` watches the working directory, which is the code being edited
        # Not the caller's directory: it may be deleted while the server lives on, and then every
        # new worker process fails on `os.getcwd()`, which breaks every PDF conversion.
        home.ensure_home_sync()  # a first run's home does not exist yet
        os.chdir(home.HOME)
    typer.echo(f"haskie {APP_VERSION} on http://{host}:{port}  (home: {home.HOME})")
    uvicorn.run(
        "haskie.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_config=None,  # haskie configures logging itself (see `logs.configure`)
        # Without it the drain is unbounded: a request in flight when the signal lands holds the
        # whole shutdown open. Here rather than in `stop`, so Ctrl-C and a supervisor's SIGTERM
        # are bounded by the same budget as `haskie stop` is.
        timeout_graceful_shutdown=int(SHUTDOWN_DRAIN),
    )


# --- keeping a server up ----------------------------------------------------


PROBE_TIMEOUT = 1.0  # loopback: either the server answers at once or it is not there
START_DEADLINE = 60.0  # a cold boot runs migrations and launches DBOS before it serves
POLL_INTERVAL = 0.25


def _poll_until(ready: Callable[[], bool], seconds: float) -> bool:
    """Whether `ready` comes true inside `seconds`. Sleeps before the first check, because every
    caller has just asked something to happen and nothing can have happened yet."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL)
        if ready():
            return True
    return False


def _status(url: str) -> dict[str, Any] | None:
    """`/api/status` of the app serving `url`, or None: the cheapest proof that a haskie is up.

    `http.client` rather than `urllib.request`, and imported here rather than at module scope:
    `urlopen` builds a global opener whose `ProxyHandler` reads the system proxy configuration,
    which costs more than the request and can never apply to loopback anyway.
    """
    from http.client import HTTPConnection

    parts = urlsplit(url)
    connection = HTTPConnection(parts.hostname or "", parts.port or 80, timeout=PROBE_TIMEOUT)
    try:
        connection.request("GET", "/api/status")
        response = connection.getresponse()
        return json.loads(response.read()) if response.status == 200 else None
    except (OSError, ValueError):  # refused, timed out, or not haskie's JSON
        return None
    finally:
        connection.close()


# The graceful budget `run` gives uvicorn, and what `stop` allows on top of it: the app's own
# shutdown hook waits for DBOS as well (see `workflows.stop`), which uvicorn's timeout does not
# cover, so the deadline here is the drain plus that wait plus a little slack.
STOP_DEADLINE = SHUTDOWN_DRAIN + shutdown.WORKFLOW_GRACE + 5.0
# By the time `stop` hurries a server, the drain is long over and the hooks are waiting on DBOS.
# The hurry ends that wait at once, so what is left is the interpreter's exit, which
# `shutdown.EXIT_GRACE` bounds, plus slack for the pool shutdown and the process teardown.
FORCE_DEADLINE = shutdown.EXIT_GRACE + 5.0
KILL_DEADLINE = 5.0  # SIGKILL cannot be refused; this is only the kernel tearing the process down


def _signal(pid: int, number: int) -> bool:
    """Send one signal. False when the process is already gone, which is the outcome asked for."""
    try:
        os.kill(pid, number)
    except ProcessLookupError:
        return False
    return True


@cli.command()
def stop(home_dir: HomeOption = None) -> None:
    """Stop the haskie serving this home.

    SIGTERM first, so uvicorn's graceful shutdown runs the app's shutdown hooks: they stop the
    pipeline and hand the home lock back. A server still there after `STOP_DEADLINE` gets a SIGINT.
    By then the drain `run` bounds is over and the hooks are waiting for DBOS to finish running
    workflows, so the SIGINT hurries that wait (see `workflows._destroy_dbos`) rather than skipping
    any hook. A server still there after that is stuck somewhere no signal handler runs, and gets
    a SIGKILL. In-flight operations are durable, so a forced exit costs a recovery at the next
    boot rather than the work.

    The lock is both how this finds the server and how it knows the server is gone; nothing else
    is left behind to clean up (see `home.running_pid`).
    """
    import signal  # only this command signals anything, and `run` starts every session

    def released() -> bool:
        """The kernel drops the lock when the holder exits, so this is the exit itself."""
        return home.running_pid() is None

    _use_home(home_dir)
    pid = home.running_pid()
    try:
        # `_signal` is False when it went away between reading the lock and signalling it
        if pid is None or not _signal(pid, signal.SIGTERM):
            typer.echo(f"no haskie is running for {home.HOME}")
            return
        gone = _poll_until(released, STOP_DEADLINE)
        if not gone:
            typer.echo(f"haskie (pid {pid}) is still finishing; forcing it")
            gone = not _signal(pid, signal.SIGINT) or _poll_until(released, FORCE_DEADLINE)
        if not gone:
            typer.echo(f"haskie (pid {pid}) ignored the force; killing it")
            gone = not _signal(pid, signal.SIGKILL) or _poll_until(released, KILL_DEADLINE)
        if not gone:
            typer.echo(f"haskie (pid {pid}) did not stop; kill it by hand", err=True)
            raise typer.Exit(code=1)
    except PermissionError as exc:  # another user's process: no later signal can go through either
        typer.echo(f"cannot stop pid {pid}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"stopped haskie (pid {pid})")


MAX_HOOK_PAYLOAD = 64 * 1024  # a SessionStart payload is a few hundred bytes of JSON


def _hook_session_id() -> str | None:
    """The conversation's id, out of the SessionStart payload Claude Code writes to our stdin.

    None for a terminal, or for anything on stdin that is not the hook's JSON. One bounded read
    rather than a read to end of file, because a writer that holds the pipe open would otherwise
    stall the session start it is part of.
    """
    import msgspec  # `run` starts every session; only a hook payload needs a decoder

    if sys.stdin is None or sys.stdin.isatty():
        return None
    try:
        # `dict`, not a struct: every other field of the payload is Claude Code's business, and a
        # new one of any type must not make this read as "no hook".
        # `read1`, so one read of whatever arrived: `read` on a pipe waits for the writer to close
        # it, and a hook that keeps stdin open would stall the session start it belongs to.
        buffered = cast("BufferedIOBase", sys.stdin.buffer)
        payload = msgspec.json.decode(buffered.read1(MAX_HOOK_PAYLOAD), type=dict[str, Any])
    except (OSError, ValueError):  # unreadable stdin, or contents that are not a hook payload
        return None
    session_id = payload.get("session_id")
    return session_id if isinstance(session_id, str) and session_id else None


LOG_TAIL_LINES = 20  # enough for the error and the traceback line that raised it


def _serve(url: str, wait: bool) -> dict[str, Any] | None:
    """Start haskie detached if nothing is serving `url`, and wait for it unless told not to.
    Returns the server's status, or None when it was not waited for.

    Refuses a haskie of another home at `url`: its tools and its UI would search and import into
    that home. Racing callers are safe: the app claims the home before it touches the database, so
    a loser exits early while the winner holds the home at this address, and the wait goes on for
    it. A child that exits with the home free or held at another address could not start, and the
    end of its log says why, at once rather than after the whole deadline.
    """
    status = _status(url)
    if status is not None:
        typer.echo(f"haskie is already serving {url}")
    else:
        status = _start(url, wait)
        if status is None:
            return None
    served = status.get("home")
    if served != str(home.HOME):
        typer.echo(
            f"{url} serves another home ({served}); stop it, or pick another --port", err=True
        )
        raise typer.Exit(code=1)
    return status


def _start(url: str, wait: bool) -> dict[str, Any] | None:
    """Spawn `run --foreground` detached, and wait for its status unless told not to."""
    home.ensure_home_sync()
    log_file = home.HOME / "server.log"
    host, port = claude.address(url)
    typer.echo(f"starting haskie on {url} (log: {log_file})")
    with open(log_file, "ab") as stream:
        logged_from = stream.tell()
        child = subprocess.Popen(
            claude.run_command(home.HOME, url, "--foreground"),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=stream,
            start_new_session=True,  # survives the client that spawned it
        )
    if not wait:
        return None

    def failed() -> bool:
        held = home.holder()
        return child.poll() is not None and (
            held is None or held.address != f"http://{host}:{port}"
        )

    status: dict[str, Any] | None = None

    def settled() -> bool:
        nonlocal status
        status = _status(url)
        return status is not None or failed()

    _poll_until(settled, START_DEADLINE)
    if status is not None:
        typer.echo(f"haskie is serving {url}")
        return status
    if failed():
        typer.echo(f"haskie exited while starting; the end of {log_file}:", err=True)
        typer.echo(_tail(log_file, logged_from), err=True)
    else:
        typer.echo(f"haskie did not come up within {START_DEADLINE:.0f}s; see {log_file}", err=True)
    raise typer.Exit(code=1)


def _tail(log_file: Path, start: int) -> str:
    """The last lines written to `log_file` past `start`: what one child said, not the history."""
    with open(log_file, "rb") as stream:
        stream.seek(start)
        written = stream.read().decode("utf-8", "replace")
    return "\n".join(written.rstrip().splitlines()[-LOG_TAIL_LINES:])


def _greet(url: str, status: dict[str, Any], browser: bool) -> None:
    """Say where the web UI is, and open it when the first run is still to be finished there."""
    import webbrowser

    ui = f"{url}/"
    if not status.get("web_ui"):
        # A build product, not source: a checkout that never ran `mise run build` serves the API
        # and MCP alone. Said here, where the user expects a page, not only in the server's log.
        typer.echo(
            f"no web UI at {ui}: this haskie was built without it (mise run build)", err=True
        )
        return
    if status.get("initialized"):
        typer.echo(f"web UI at {ui}")
        return
    typer.echo(f"pick the embedding model and the search at {ui}")
    if browser:
        webbrowser.open(ui)  # False without a browser to open; the line above says where to go


# --- installing into a client -----------------------------------------------

install = typer.Typer(
    name="install",
    help="Register haskie with an MCP client.",
    no_args_is_help=True,
)
cli.add_typer(install)


@install.command("claude")
def install_claude(
    home_dir: HomeOption = None,
    url: Annotated[str, typer.Option(help="MCP endpoint of this haskie.")] = MCP_URL,
    scope: Annotated[Scope, typer.Option(help="Where Claude Code records it.")] = Scope.USER,
) -> None:
    """Register the MCP server with Claude Code and write the haskie skill and rule.

    Four things a client needs that the tool descriptions cannot supply: the endpoint, a server
    running at it, a skill saying how to search the user's own documents, and a rule loaded into
    every session saying when to search: before planning, or answering from memory or the web.
    haskie records the installation and rewrites both whenever a collection changes.
    """
    _use_home(home_dir)
    try:
        _install_claude(url, scope)
    except HaskieError as exc:  # a home too old to read, a settings file that is not JSON, ...
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


def _install_claude(url: str, scope: Scope) -> None:
    """The steps of `install claude`, each reporting as it goes; failures are `HaskieError`."""
    import asyncio

    from haskie import db

    # First, so a home this build cannot read is refused before Claude Code's files are touched.
    # It also makes the home, so there is no prelude to repeat here.
    asyncio.run(db.migrate_once())

    manual = claude.register_mcp(url, scope)
    if manual is None:
        typer.echo(f"registered the haskie MCP server at {url} ({scope} scope)")
    else:
        typer.echo("the `claude` CLI is not on PATH; register the server by hand:")
        typer.echo(f"  {manual}")

    directory = claude.claude_dir(scope)
    added = claude.install_hook(directory, home.HOME, url)
    settings_file = claude.settings_path(directory)
    typer.echo(f"{'added' if added else 'updated'} the SessionStart hook in {settings_file}")

    # read last, so a collection changed during the `claude` calls above still lands
    found = asyncio.run(claude.read_collections())
    for written in claude.write_instructions(directory, found):
        typer.echo(f"wrote {written}")
    named = ", ".join(collection.name for collection in found) or "none yet"
    typer.echo(f"  collections in the trigger: {named}")
    asyncio.run(claude.record_installation(directory))

    # `_serve`, not the `run` command: installing wants the server up, not the first-run page
    # `run` would open in the browser. Already-serving is its fast path, not ours.
    _serve(url, wait=True)
    typer.echo("haskie rewrites the skill and rule whenever a collection changes")


@install.command("codex")
def install_codex(
    home_dir: HomeOption = None,
    url: Annotated[str, typer.Option(help="MCP endpoint of this haskie.")] = MCP_URL,
    scope: Annotated[Scope, typer.Option(help="Where Codex records it.")] = Scope.USER,
) -> None:
    """Register the MCP server with Codex, write search instructions and a SessionStart hook."""
    import asyncio

    from haskie import codex, db

    _use_home(home_dir)
    directory = codex.codex_dir(scope)
    try:
        asyncio.run(db.migrate_once())
        codex.validate(directory, scope)
        codex.register_mcp(directory, url)
        typer.echo(f"registered the haskie MCP server at {url} ({scope} scope)")
        typer.echo("enabled MCP 2026-07-28 support in Codex")
        found = asyncio.run(claude.read_collections())
        for written in claude.write_instructions(directory, found):
            typer.echo(f"wrote {written}")
        typer.echo(f"linked the search rule from {codex.install_rule_reference(directory, scope)}")
        claude.install_hook(
            directory, home.HOME, url, filename="hooks.json", rules=claude.rule_path(directory)
        )
        typer.echo(f"wrote the SessionStart hook in {directory / 'hooks.json'}")
        asyncio.run(claude.record_installation(directory, "codex"))
        _serve(url, wait=True)
    except (HaskieError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("haskie rewrites the skill and rule whenever a collection changes")
    typer.echo("In Codex, review and trust the haskie SessionStart hook, then start a new session.")
    if scope == Scope.PROJECT:
        typer.echo("Codex loads project configuration only in trusted projects.")


uninstall = typer.Typer(
    name="uninstall",
    help="Remove haskie from an MCP client.",
    no_args_is_help=True,
)
cli.add_typer(uninstall)


@uninstall.command("claude")
def uninstall_claude(
    home_dir: HomeOption = None,
    scope: Annotated[Scope, typer.Option(help="Where Claude Code recorded it.")] = Scope.USER,
) -> None:
    """Remove what `install claude` added: the MCP entry, the skill, the rule and the
    SessionStart hook, and stop rewriting them when a collection changes.

    Every haskie hook in the scope goes, whichever home it starts. Documents and collections stay;
    `haskie destroy` removes those. Uninstalling what is not installed does nothing.
    """
    _use_home(home_dir)
    try:
        _uninstall_claude(scope)
    except HaskieError as exc:  # a home too old to read, a settings file that is not JSON, ...
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


def _uninstall_claude(scope: Scope) -> None:
    """The steps of `uninstall claude`, each reporting as it goes; failures are `HaskieError`."""
    import asyncio

    directory = claude.claude_dir(scope)
    # The record first, so a server refreshing meanwhile no longer writes the files back. Only
    # when the home exists: uninstalling must not make one.
    if home.DB_FILE.is_file() and asyncio.run(claude.forget_installation(directory)):
        typer.echo(f"stopped refreshing {directory}")

    manual = claude.unregister_mcp(scope)
    if manual is None:
        typer.echo(f"removed the haskie MCP server ({scope} scope)")
    else:
        typer.echo("the `claude` CLI is not on PATH; remove the server by hand:")
        typer.echo(f"  {manual}")

    settings_file = claude.settings_path(directory)
    if claude.uninstall_hook(directory):
        typer.echo(f"removed the SessionStart hook from {settings_file}")
    for removed in claude.remove_instructions(directory):
        typer.echo(f"removed {removed}")


@uninstall.command("codex")
def uninstall_codex(
    home_dir: HomeOption = None,
    scope: Annotated[Scope, typer.Option(help="Where Codex recorded it.")] = Scope.USER,
) -> None:
    """Remove haskie's Codex MCP entry, instructions and hook. Keep documents and collections."""
    import asyncio

    from haskie import codex

    _use_home(home_dir)
    directory = codex.codex_dir(scope)
    try:
        codex.validate(directory, scope)
        if home.DB_FILE.is_file() and asyncio.run(claude.forget_installation(directory, "codex")):
            typer.echo(f"stopped refreshing {directory}")
        if codex.register_mcp(directory, None):
            typer.echo(f"removed the haskie MCP server ({scope} scope)")
        if claude.uninstall_hook(directory, filename="hooks.json"):
            typer.echo(f"removed the SessionStart hook from {directory / 'hooks.json'}")
        for changed in codex.remove_rule_reference(directory, scope):
            typer.echo(f"removed the search rule reference from {changed}")
        for removed in claude.remove_instructions(directory):
            typer.echo(f"removed {removed}")
    except (HaskieError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@cli.command()
def destroy(
    home_dir: HomeOption = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
) -> None:
    """Delete the home directory and everything in it.

    Every document, collection, index, preview and operation record goes. There is no undo and
    nothing is backed up first. A home a haskie is serving is refused: removing the database under a
    running server leaves it writing into deleted files, and takes the home lock with it. `haskie
    stop` first.
    """
    import asyncio

    from haskie import db

    _use_home(home_dir)
    root = home.HOME
    if not root.exists():
        typer.echo(f"nothing to destroy: {root} does not exist")
        return

    # A home is a database, a document tree and a collection tree. Refusing anything else is what
    # stops a mistyped `--home ~/Documents` from deleting the wrong directory.
    if not (home.DB_FILE.exists() or home.COLLECTION_ROOT.is_dir() or home.DOCUMENT_ROOT.is_dir()):
        typer.echo(
            f"{root} does not look like a haskie home (no haskie.db, no documents/, "
            "no collections/)",
            err=True,
        )
        raise typer.Exit(code=1)

    held = home.home_holder()
    if held is not None:
        typer.echo(f"{held}; stop it first (haskie stop)", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"about to delete {root}")
    typer.echo(f"  {_describe(root)}")
    if not yes:
        typer.confirm("permanently delete it?", abort=True)

    asyncio.run(home.remove_tree(root))
    db.invalidate_migrations()  # the file this process migrated is gone; a new one starts over
    # `remove_tree` logs what it cannot delete and continues, so only a check afterwards knows.
    if os.path.lexists(root):
        typer.echo(f"could not delete all of {root}; still there: {_left_over(root)}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"deleted {root}")


LEFT_SHOWN = 5  # enough to see where the failure is, few enough to stay one line


def _left_over(root: Path) -> str:
    """The files a failed delete left under `root`, the first few by name."""
    left = sorted(str(path.relative_to(root)) for path in root.rglob("*") if not path.is_dir())
    if not left:
        return "empty directories"
    more = f" and {len(left) - LEFT_SHOWN} more" if len(left) > LEFT_SHOWN else ""
    return ", ".join(left[:LEFT_SHOWN]) + more


def _entries(root: Path) -> list[str]:
    """The names one level below each shard (`<root>/<shard>/<name>`), sorted.

    Read off the filesystem rather than the database: this runs before anything opens the home,
    and a home too old to migrate has no readable rows to count anyway.
    """
    if not root.is_dir():
        return []
    return sorted(
        entry.name
        for shard in root.iterdir()
        if shard.is_dir()
        for entry in shard.iterdir()
        if entry.is_dir()
    )


def _describe(root: Path) -> str:
    """What is about to be lost, in one line: enough to recognise the wrong directory."""
    files = [p for p in root.rglob("*") if p.is_file()]
    megabytes = sum(p.stat().st_size for p in files) / 1_048_576
    documents = _entries(home.DOCUMENT_ROOT)
    collections = _entries(home.COLLECTION_ROOT)
    listed = ", ".join(collections) if collections else "none"
    return (
        f"{len(files)} files, {megabytes:.1f} MB, {len(documents)} documents, collections: {listed}"
    )


@cli.command()
def version() -> None:
    """Print the version and where the data lives."""
    typer.echo(VERSION_LINE)
    typer.echo(f"home: {home.HOME}")
