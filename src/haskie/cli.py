"""The `haskie` command: set the home directory up, and serve the app.

Thin on purpose. Everything it does is something the app already does at startup — `init` is the
migrations and the layout move, `run` is uvicorn over `app:create_app` — so the CLI adds a way in,
never a second way of doing the work.
"""

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer

from haskie import APP_VERSION, home

cli = typer.Typer(
    name="haskie",
    help="Personal document library: markdown conversion, LanceDB search, web UI and MCP server.",
    no_args_is_help=True,
    add_completion=False,
)

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
    """Create the home directory and bring its database and layout up to date.

    Safe to repeat: every step is idempotent, so this is also how an existing home is migrated
    after an upgrade. `run` does the same thing at startup; this is for doing it first.
    """
    from haskie import db, layout

    _use_home(home_dir)

    async def prepare() -> int:
        await home.ensure_home()
        await db.migrate_once()
        return await layout.migrate_layout()

    moved = asyncio.run(prepare())
    typer.echo(f"haskie {APP_VERSION} ready in {home.HOME}")
    if moved:
        typer.echo(f"moved {moved} entries into the sharded layout")


@cli.command()
def run(
    home_dir: HomeOption = None,
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Restart on code changes (development).")] = False,
) -> None:
    """Run the web UI, the REST API and the MCP server.

    Binds loopback by default: the home directory is one user's documents, and nothing in the app
    authenticates a caller.
    """
    import uvicorn

    _use_home(home_dir)
    typer.echo(f"haskie {APP_VERSION} on http://{host}:{port}  (home: {home.HOME})")
    uvicorn.run(
        "haskie.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_config=None,  # haskie configures logging itself (see `logs.configure`)
    )


@cli.command()
def destroy(
    home_dir: HomeOption = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
) -> None:
    """Delete the home directory and everything in it.

    Every document, index, preview and job record goes. There is no undo and nothing is backed up
    first. Stop `haskie run` before this: removing the database under a running server leaves it
    writing into deleted files.
    """
    from haskie import db

    _use_home(home_dir)
    root = home.HOME
    if not root.exists():
        typer.echo(f"nothing to destroy: {root} does not exist")
        return

    # A home is a database and a library tree. Refusing anything else is what stops a mistyped
    # `--home ~/Documents` from deleting the wrong directory.
    if not (home.DB_FILE.exists() or home.LIBRARY_ROOT.is_dir()):
        typer.echo(f"{root} does not look like a haskie home (no haskie.db, no library/)", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"about to delete {root}")
    typer.echo(f"  {_describe(root)}")
    if not yes:
        typer.confirm("permanently delete it?", abort=True)

    asyncio.run(home.remove_tree(root))
    db.invalidate_migrations()  # the file this process migrated is gone; a new one starts over
    typer.echo(f"deleted {root}")


def _describe(root: Path) -> str:
    """What is about to be lost, in one line: enough to recognise the wrong directory."""
    files = [p for p in root.rglob("*") if p.is_file()]
    megabytes = sum(p.stat().st_size for p in files) / 1_048_576
    libraries = (
        sorted(p.name for p in home.LIBRARY_ROOT.iterdir() if p.is_dir())
        if home.LIBRARY_ROOT.is_dir()
        else []
    )
    listed = ", ".join(libraries) if libraries else "none"
    return f"{len(files)} files, {megabytes:.1f} MB, libraries: {listed}"


@cli.command()
def version() -> None:
    """Print the version and where the data lives."""
    typer.echo(f"haskie {APP_VERSION}")
    typer.echo(f"home: {home.HOME}")
