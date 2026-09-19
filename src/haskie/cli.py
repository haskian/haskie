"""The `haskie` command: set the home directory up, and serve the app.

Thin on purpose. Everything it does is something the app already does at startup — `init` is the
migrations, `run` is uvicorn over `app:create_app` — so the CLI adds a way in, never a second way
of doing the work.
"""

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any
from urllib.parse import urlsplit

import typer

from haskie import APP_VERSION, home
from haskie.claude import DEFAULT_HOST, DEFAULT_PORT, MCP_URL, Scope

if TYPE_CHECKING:
    from haskie.collection import CollectionSummary

cli = typer.Typer(
    name="haskie",
    help="Your documents: markdown conversion, LanceDB search, web UI and MCP server.",
    no_args_is_help=True,
    add_completion=False,
)


def _print_version(shown: bool) -> None:
    """`--version` before anything else: eager, so it answers without a subcommand and without
    touching the home directory."""
    if shown:
        typer.echo(f"haskie {APP_VERSION}")
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
    """Point the process at `path` before anything reads the home layout.

    `home` resolves its paths at import time from the environment, so a `--home` has to be put
    back into the environment and the module's own paths refreshed.
    """
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
    from haskie import db
    from haskie.errors import HaskieError

    _use_home(home_dir)

    async def prepare() -> None:
        await home.ensure_home()
        await db.migrate_once()

    try:
        asyncio.run(prepare())
    except HaskieError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"haskie {APP_VERSION} ready in {home.HOME}")


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
        typer.echo(f"haskie is already running for {home.HOME} ({held})", err=True)
        raise typer.Exit(code=1)
    # The address the startup hook records, for the next process's message.
    home.ADDRESS = os.environ["HASKIE_ADDRESS"] = f"http://{host}:{port}"
    typer.echo(f"haskie {APP_VERSION} on http://{host}:{port}  (home: {home.HOME})")
    uvicorn.run(
        "haskie.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_config=None,  # haskie configures logging itself (see `logs.configure`)
    )


# --- keeping a server up ----------------------------------------------------

PROBE_TIMEOUT = 1.0  # loopback: either the server answers at once or it is not there
START_DEADLINE = 60.0  # a cold boot runs migrations and launches DBOS before it serves
START_POLL = 0.25


def _status_url(url: str) -> str:
    """`/api/status` of the app serving `url`: the cheapest proof that a haskie is up."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}/api/status"


def _serving(url: str) -> bool:
    try:
        with urllib.request.urlopen(_status_url(url), timeout=PROBE_TIMEOUT) as response:
            return response.status == 200
    except OSError:  # URLError and TimeoutError are both OSError
        return False


@cli.command()
def ensure(
    home_dir: HomeOption = None,
    url: Annotated[str, typer.Option(help="Where haskie should be serving.")] = MCP_URL,
    wait: Annotated[
        bool, typer.Option(help="Wait for the server to answer before returning.")
    ] = True,
) -> None:
    """Start haskie if nothing is serving `url`, and wait for it unless told not to.

    An MCP client reaches haskie over HTTP, so the server has to be up before the client connects.
    This is what `haskie install claude` puts in Claude Code's SessionStart hook, and it is safe to
    run as often as you like: an already-serving haskie costs one loopback request.

    The server it starts is detached and outlives this command on purpose - the next session
    reuses it, and the web UI stays up. Racing callers are safe: the app claims the home before it
    touches the database, so every loser exits early and the wait below finds the one winner.

    `--no-wait` returns as soon as the server is spawned. That is what the hook uses: a client
    connects to the MCP endpoint while the hook is still running, so waiting cannot help the
    session that starts it, and a haskie that fails to boot would stall every session start.

    ponytail: nothing stops the server it starts. A launchd agent is the upgrade path for a
    machine that should always have haskie up.
    """
    _use_home(home_dir)
    if _serving(url):
        typer.echo(f"haskie is already serving {url}")
        return

    parts = urlsplit(url)
    home.HOME.mkdir(parents=True, exist_ok=True, mode=home.DIR_MODE)
    log_file = home.HOME / "server.log"
    command = [
        # `-m haskie`, not the console script: this process's interpreter is always the right one,
        # and a spawned client's PATH may not carry `haskie` at all.
        sys.executable,
        "-m",
        "haskie",
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
    deadline = time.monotonic() + START_DEADLINE
    while time.monotonic() < deadline:
        time.sleep(START_POLL)  # first: the child has not finished exec yet
        if _serving(url):
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


def _register_with_claude(url: str, scope: Scope) -> str | None:
    """Add the HTTP entry to Claude Code, replacing any entry of ours already there.

    HTTP rather than stdio: litestar-mcp serves MCP `2026-07-28`, which replaced `initialize`
    with `server/discover`, and a stdio client that opens with `initialize` never connects.

    Returns the command to run by hand when the `claude` CLI is not installed, so a missing CLI
    costs the user one copy-paste rather than the whole install.
    """
    arguments = ["mcp", "add", "-s", scope, "--transport", "http", "haskie", url]
    claude_cli = shutil.which("claude")
    if claude_cli is None:
        return "claude " + " ".join(arguments)
    # Remove first, so re-running updates the entry instead of failing on the name. No entry is
    # the normal case, so that failure is the expected one.
    subprocess.run(
        [claude_cli, "mcp", "remove", "-s", scope, "haskie"], capture_output=True, check=False
    )
    done = subprocess.run([claude_cli, *arguments], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        typer.echo((done.stderr or done.stdout).strip(), err=True)
        raise typer.Exit(code=1)
    return None


HOOK_MARKER = " ensure --home "  # what identifies a hook of ours, whatever path invoked it


def _install_hook(settings_file: Path, url: str) -> bool:
    """Teach Claude Code to bring haskie up at the start of a session.

    The MCP entry is HTTP, so a session that starts while nothing is serving gets no haskie tools
    at all, and nothing says why. A SessionStart hook running `haskie ensure` fixes that: it costs
    one loopback request when the server is already up, which is the usual case.

    Returns whether this call added the hook. Reads and rewrites the file as a whole, so an
    existing settings file keeps everything else in it.
    """
    command = (
        f"{_own_executable()} ensure --home {shlex.quote(str(home.HOME))} --url {url} --no-wait"
    )
    settings: dict[str, Any] = {}
    if settings_file.is_file():
        try:
            settings = json.loads(settings_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"{settings_file} is not valid JSON: {exc}") from None
    matchers = settings.setdefault("hooks", {}).setdefault("SessionStart", [])
    # Matched on the shape of the command, not on the path `haskie` happens to have today: an
    # upgrade that moves the executable must still replace the hook rather than stack a copy.
    ours = [
        hook
        for matcher in matchers
        for hook in matcher.get("hooks", [])
        if HOOK_MARKER in str(hook.get("command", ""))
    ]
    for hook in ours:
        hook["command"] = command
    if not ours:
        matchers.append({"hooks": [{"type": "command", "command": command, "timeout": 90}]})
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: Claude Code reads this file, and a half-written settings.json is a broken client.
    home.atomic_write_sync(settings_file, json.dumps(settings, indent=2) + "\n")
    return not ours


def _own_executable() -> str:
    """An absolute `haskie`, because a hook and an MCP client both run with their own PATH."""
    return shutil.which("haskie") or f"{sys.executable} -m haskie"


@install.command("claude")
def install_claude(
    home_dir: HomeOption = None,
    url: Annotated[str, typer.Option(help="MCP endpoint of this haskie.")] = MCP_URL,
    scope: Annotated[Scope, typer.Option(help="Where Claude Code records it.")] = "user",
    hook: Annotated[
        bool, typer.Option(help="Add a SessionStart hook that starts haskie when it is down.")
    ] = True,
    start: Annotated[bool, typer.Option(help="Start haskie now if it is not serving.")] = True,
) -> None:
    """Register the MCP server with Claude Code and write the haskie skill.

    Three things a client needs that the tool descriptions cannot supply: the endpoint, a server
    running at it, and a skill saying when the user's own documents beat a web search. Re-run
    after adding a collection to refresh the skill's trigger.
    """
    from haskie import claude

    _use_home(home_dir)
    # `read_collections` reaches the database through `db.connect`, which migrates the home and
    # makes it first - so there is no prelude to repeat here.
    found: list[CollectionSummary] = asyncio.run(claude.read_collections())

    manual = _register_with_claude(url, scope)
    if manual is None:
        typer.echo(f"registered the haskie MCP server at {url} ({scope} scope)")
    else:
        typer.echo("the `claude` CLI is not on PATH; register the server by hand:")
        typer.echo(f"  {manual}")

    destination = claude.write_skill(scope, found)
    named = ", ".join(collection.name for collection in found) or "none yet"
    typer.echo(f"wrote {destination}")
    typer.echo(f"  collections in the trigger: {named}")

    if hook:
        settings_file = claude.settings_path(scope)
        added = _install_hook(settings_file, url)
        typer.echo(f"{'added' if added else 'updated'} the SessionStart hook in {settings_file}")
    if start:
        ensure(home_dir=home.HOME, url=url)  # already-serving is its fast path, not ours
    typer.echo("re-run `haskie install claude` after adding a collection, to refresh the trigger")


@cli.command()
def destroy(
    home_dir: HomeOption = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
) -> None:
    """Delete the home directory and everything in it.

    Every document, collection, index, preview and job record goes. There is no undo and nothing
    is backed up first. Stop `haskie run` before this: removing the database under a running
    server leaves it writing into deleted files.
    """
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
    typer.echo(f"haskie {APP_VERSION}")
    typer.echo(f"home: {home.HOME}")
