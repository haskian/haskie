"""Haskie: personal document library with markdown conversion, LanceDB search, web UI and MCP."""

from importlib.metadata import version

# One source for the version: the OpenAPI document, every audit record, and the DBOS
# application version workflows are recovered under.
APP_VERSION = version("haskie")
