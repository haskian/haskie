"""Codex's MCP configuration. Skills and SessionStart hooks share Claude's format.

Keep the skill under the config layer's supported `skills/` root, so CODEX_HOME isolates
the whole installation. The hook loads the rule into context without editing AGENTS.md.
"""

import os
from collections.abc import MutableMapping
from pathlib import Path

import tomlkit
from tomlkit.exceptions import ParseError
from tomlkit.toml_document import TOMLDocument

from haskie import claude
from haskie.claude import Scope
from haskie.errors import InvalidInput


def codex_dir(scope: Scope) -> Path:
    """Resolve at invocation time, including a custom user profile."""
    if scope == Scope.USER:
        directory = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        return directory.expanduser().absolute()
    return (Path.cwd() / ".codex").absolute()


def read_config(directory: Path) -> TOMLDocument:
    """Refuse malformed configuration before changing any installed file."""
    path = directory / "config.toml"
    if not path.exists():
        return tomlkit.document()
    try:
        config = tomlkit.parse(path.read_text(encoding="utf-8"))
    except ParseError as exc:
        raise InvalidInput(f"{path} is not valid TOML: {exc}") from None
    if "mcp_servers" in config and not isinstance(config["mcp_servers"], MutableMapping):
        raise InvalidInput(f"{path} does not hold `mcp_servers` as a table")
    return config


def register_mcp(directory: Path, url: str | None) -> bool:
    """Set or remove only haskie's entry, retaining other servers, settings and comments.

    Editing TOML directly also supports project scope and machines without the Codex CLI.
    `codex mcp add` writes only the user configuration.
    """
    config = read_config(directory)
    servers = config.get("mcp_servers")
    if url is None:
        if servers is None or claude.SKILL_NAME not in servers:
            return False
        del servers[claude.SKILL_NAME]
    else:
        if servers is None:
            config["mcp_servers"] = tomlkit.table()
            servers = config["mcp_servers"]
        servers[claude.SKILL_NAME] = {"url": url}
    claude._write(directory / "config.toml", tomlkit.dumps(config))
    return True


def validate(directory: Path) -> None:
    """Check both user-owned files before the installer writes either one."""
    read_config(directory)
    path = directory / "hooks.json"
    claude._session_start(claude._read_settings(path), path)
