"""The `haskie` command: set the home directory up, and serve the app.

Thin on purpose. Everything it does is something the app already does at startup — `init` is the
migrations, `run` is uvicorn over `app:create_app` — so the CLI adds a way in, never a second way
of doing the work.
"""

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

from haskie import APP_VERSION, claude, home
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
    """Point the process at `path` before anything reads the home layout. The environment carries
    it too, so a `--reload` child re-reads the same root (see `home.HOME`)."""
    if path is None:
        return
    resolved = str(Path(path).expanduser().resolve())
    os.environ["HASKIE_HOME"] = resolved
    home.use(Path(resolved))


@cli.command()
def init(home_dir: HomeOption = None) -> None:
    """Create the home directory and bring its database up to date.

    Safe to repeat: every step is idempotent, so this is also how an existing home is migrated
    after an upgrade. `run` does the same thing at startup; this is for doing it first.

    A home written before documents became collection-independent cannot be migrated, so the
    migration refuses it and says what to do instead (see `db.INCOMPATIBLE_HOME_MESSAGE`).
    """
    import asyncio

    from haskie import db

    _use_home(home_dir)
    try:
        asyncio.run(db.migrate_once())  # makes the home first (see `db.migrate_once`)
    except HaskieError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"haskie {APP_VERSION} ready in {home.HOME}")


SHUTDOWN_DRAIN = 10.0  # what `run` gives uvicorn to finish the requests in flight when it is told


@cli.command()
def run(
    home_dir: HomeOption = None,
    host: Annotated[str, typer.Option(help="Interface to bind.")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port to listen on.")] = DEFAULT_PORT,
    reload: Annotated[bool, typer.Option(help="Restart on code changes (development).")] = False,
) -> None:
    """Run the web UI, the REST API and the MCP server.

    Binds loopback by default: the home directory is one user's documents, and nothing in the app
    authenticates a caller.
    """
    import uvicorn

    _use_home(home_dir)
    # The app's first startup hook is what claims the home; this only asks, so that the common
    # refusal is one line here instead of a lifespan traceback out of uvicorn.
    held = home.home_holder()
    if held is not None:
        typer.echo(held, err=True)
        raise typer.Exit(code=1)
    # The address the startup hook records, for the next process's message. The environment is the
    # one carrier, so a `--reload` child that re-imports `home` records the same thing.
    os.environ["HASKIE_ADDRESS"] = f"http://{host}:{port}"
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


def _serving(url: str) -> bool:
    """`/api/status` of the app serving `url`: the cheapest proof that a haskie is up.

    `http.client` rather than `urllib.request`, and imported here rather than at module scope:
    `urlopen` builds a global opener whose `ProxyHandler` reads the system proxy configuration,
    which costs more than the request and can never apply to loopback anyway.
    """
    from http.client import HTTPConnection

    parts = urlsplit(url)
    connection = HTTPConnection(parts.hostname or "", parts.port or 80, timeout=PROBE_TIMEOUT)
    try:
        connection.request("GET", "/api/status")
        return connection.getresponse().status == 200
    except OSError:  # a refused connection and a timeout are both OSError
        return False
    finally:
        connection.close()


# The graceful budget `run` gives uvicorn, and what `stop` allows on top of it: the app's own
# shutdown hook waits for DBOS as well (see `workflows.stop`), which uvicorn's timeout does not
# cover, so the deadline here is the drain plus that wait plus a little slack.
STOP_DEADLINE = SHUTDOWN_DRAIN + 15.0
FORCE_DEADLINE = 5.0  # after the force signal there is nothing left to wait for but the exit


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
    pipeline and hand the home lock back. A server still there after `STOP_DEADLINE` gets a SIGINT,
    which is what uvicorn force-quits on once it is already shutting down - the hooks themselves
    are unbounded, and DBOS can outlast the drain `run` bounds. In-flight operations are durable,
    so a forced exit costs a recovery at the next boot rather than the work.

    The lock is both how this finds the server and how it knows the server is gone; nothing else
    is left behind to clean up (see `home.running_pid`).
    """
    import signal  # only this command signals anything, and `ensure` runs on every session start

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
        if not _poll_until(released, STOP_DEADLINE):
            typer.echo(f"haskie (pid {pid}) is still finishing; forcing it")
            if _signal(pid, signal.SIGINT) and not _poll_until(released, FORCE_DEADLINE):
                typer.echo(f"haskie (pid {pid}) did not stop; kill it by hand", err=True)
                raise typer.Exit(code=1)
    except PermissionError as exc:  # another user's process: no later signal can go through either
        typer.echo(f"cannot stop pid {pid}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"stopped haskie (pid {pid})")


MAX_HOOK_PAYLOAD = 64 * 1024  # a SessionStart payload is a few hundred bytes of JSON


def _hook_session_id() -> str | None:
    """The conversation's id, out of the SessionStart payload Claude Code writes to our stdin.

    None for every other way of running the command: a terminal, or anything on stdin that is not
    the hook's JSON. One bounded read rather than a read to end of file, because a writer that
    holds the pipe open would otherwise stall the session start it is part of.
    """
    import msgspec  # `ensure` runs on every session start; only a hook payload needs a decoder

    if sys.stdin is None or sys.stdin.isatty():
        return None
    try:
        # `dict`, not a struct: every other field of the payload is Claude Code's business, and a
        # new one of any type must not make this read as "no hook".
        # `read1`, so one read of whatever arrived: `read` on a pipe waits for the writer to close
        # it, and a hook that keeps stdin open would hold up the session start it belongs to.
        buffered = cast("BufferedIOBase", sys.stdin.buffer)
        payload = msgspec.json.decode(buffered.read1(MAX_HOOK_PAYLOAD), type=dict[str, Any])
    except (OSError, ValueError):  # unreadable stdin, or contents that are not a hook payload
        return None
    session_id = payload.get("session_id")
    return session_id if isinstance(session_id, str) and session_id else None


@cli.command()
def ensure(
    home_dir: HomeOption = None,
    url: Annotated[str, typer.Option(help="Where haskie should be serving.")] = MCP_URL,
    wait: Annotated[
        bool, typer.Option(help="Wait for the server to answer before returning.")
    ] = True,
) -> None:
    """Start haskie if nothing is serving `url`, and wait for it unless told not to.

    The server it starts is detached and outlives this command on purpose: the next session reuses
    it, and the web UI stays up. Racing callers are safe - the app claims the home before it
    touches the database, so every loser exits early and the wait below finds the one winner.

    Claude Code's SessionStart hook passes `--no-wait`: a client connects to the MCP endpoint
    while the hook is still running, so waiting cannot help the session that starts it, and a
    haskie that fails to boot would stall every session start.
    """
    _use_home(home_dir)
    session_id = _hook_session_id()
    if session_id is not None:
        # A SessionStart hook's output becomes context for the session it starts, which is the one
        # place the conversation's own id can reach the tools: nothing in an MCP call carries it,
        # so without this line every search is recorded against no session at all.
        typer.echo(claude.session_announcement(session_id))
    if _serving(url):
        typer.echo(f"haskie is already serving {url}")
        return

    parts = urlsplit(url)
    home.ensure_home_sync()
    log_file = home.HOME / "server.log"
    command = [
        *claude.own_command(),
        "run",
        "--home",
        str(home.HOME),
        "--host",
        parts.hostname or DEFAULT_HOST,
        "--port",
        str(parts.port or DEFAULT_PORT),
    ]
    typer.echo(f"starting haskie on {url} (log: {log_file})")
    with open(log_file, "ab") as stream:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=stream,
            start_new_session=True,  # survives the client that spawned it
        )
    if not wait:
        return
    if _poll_until(lambda: _serving(url), START_DEADLINE):
        typer.echo(f"haskie is serving {url}")
        return
    typer.echo(f"haskie did not come up within {START_DEADLINE:.0f}s; see {log_file}", err=True)
    raise typer.Exit(code=1)


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
    scope: Annotated[Scope, typer.Option(help="Where Claude Code records it.")] = "user",
) -> None:
    """Register the MCP server with Claude Code and write the haskie skill.

    Three things a client needs that the tool descriptions cannot supply: the endpoint, a server
    running at it, and a skill saying when the user's own documents beat a web search. Re-run
    after adding a collection to refresh the skill's trigger.
    """
    import asyncio

    _use_home(home_dir)
    # `read_collections` reaches the database through `db.connect`, which migrates the home and
    # makes it first - so there is no prelude to repeat here.
    found = asyncio.run(claude.read_collections())

    try:
        manual = claude.register_mcp(url, scope)
    except HaskieError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    if manual is None:
        typer.echo(f"registered the haskie MCP server at {url} ({scope} scope)")
    else:
        typer.echo("the `claude` CLI is not on PATH; register the server by hand:")
        typer.echo(f"  {manual}")

    destination = claude.write_skill(scope, found)
    named = ", ".join(collection.name for collection in found) or "none yet"
    typer.echo(f"wrote {destination}")
    typer.echo(f"  collections in the trigger: {named}")

    added = claude.install_hook(scope, home.HOME, url)
    settings_file = claude.settings_path(scope)
    typer.echo(f"{'added' if added else 'updated'} the SessionStart hook in {settings_file}")
    ensure(home_dir=home.HOME, url=url)  # already-serving is its fast path, not ours
    typer.echo("re-run `haskie install claude` after adding a collection, to refresh the trigger")


@cli.command()
def destroy(
    home_dir: HomeOption = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
) -> None:
    """Delete the home directory and everything in it.

    Every document, collection, index, preview and operation record goes. There is no undo and
    nothing is backed up first. Stop `haskie run` before this: removing the database under a running
    server leaves it writing into deleted files.
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

    typer.echo(f"about to delete {root}")
    typer.echo(f"  {_describe(root)}")
    if not yes:
        typer.confirm("permanently delete it?", abort=True)

    asyncio.run(home.remove_tree(root))
    db.invalidate_migrations()  # the file this process migrated is gone; a new one starts over
    typer.echo(f"deleted {root}")


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
