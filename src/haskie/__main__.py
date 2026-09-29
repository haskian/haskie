"""`python -m haskie`, so a subprocess can reach the CLI through `sys.executable` alone rather
than hoping the console script is on PATH (see `claude.own_command`)."""

from haskie.cli import cli

cli()
