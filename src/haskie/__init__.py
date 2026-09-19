"""Haskie: personal document collections; markdown conversion, LanceDB search, web UI, MCP."""

from typing import Any


# One source for the version: the OpenAPI document, every audit record, and the DBOS
# application version workflows are recovered under.
#
# Read on first use rather than at import: `importlib.metadata` pulls in zipfile, email and
# inspect, and `haskie ensure` runs on every Claude Code session start without ever asking for
# the version. PEP 562, so `from haskie import APP_VERSION` still works.
def __getattr__(name: str) -> Any:
    if name == "APP_VERSION":
        from importlib.metadata import version

        globals()["APP_VERSION"] = version("haskie")
        return globals()["APP_VERSION"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
