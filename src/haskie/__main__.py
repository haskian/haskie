"""`python -m haskie`, so a subprocess can reach the CLI through `sys.executable` alone rather
than hoping the console script is on PATH (see `cli.ensure`)."""

from haskie.cli import cli

cli()
