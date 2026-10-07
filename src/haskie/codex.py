"""Codex's MCP configuration. Skills and SessionStart hooks share Claude's format.

Keep the skill under the config layer's supported `skills/` root, so CODEX_HOME isolates
the whole installation. AGENTS.md points to the rule even when the hook does not run.
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

RULE_START = "<!-- haskie:start -->"
RULE_END = "<!-- haskie:end -->"


def instruction_paths(directory: Path, scope: Scope) -> tuple[Path, Path]:
    root = directory if scope == Scope.USER else directory.parent
    return root / "AGENTS.md", root / "AGENTS.override.md"


def _without_rule(path: Path) -> str:
    """Strip only our managed block; refuse ambiguous boundaries before changing files."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if RULE_START not in text and RULE_END not in text:
        return text
    if text.count(RULE_START) != 1 or text.count(RULE_END) != 1:
        raise InvalidInput(f"{path} has an invalid haskie instruction block")
    before, _, rest = text.partition(RULE_START)
    if RULE_END not in rest:
        raise InvalidInput(f"{path} has an invalid haskie instruction block")
    _, _, after = rest.partition(RULE_END)
    return before + after.removeprefix("\n\n")


def install_rule_reference(directory: Path, scope: Scope) -> Path:
    path, override = instruction_paths(directory, scope)
    # Global discovery skips an empty override; project discovery selects it by existence.
    if override.exists() and (scope == Scope.PROJECT or override.read_text().strip()):
        path = override
    original = _without_rule(path)
    block = (
        f"{RULE_START}\n"
        "At the start of every session, read and follow the "
        f"[haskie search rule](<{claude.rule_path(directory)}>).\n"
        "Search the user's collections first whenever they cover the topic, "
        "even when you know the answer.\n"
        "If no haskie session id was announced, choose a short id (at most 128 characters) "
        "and reuse it for this conversation.\n"
        f"{RULE_END}\n\n"
    )
    claude._write(path, block + original)
    return path


def remove_rule_reference(directory: Path, scope: Scope) -> list[Path]:
    changed = []
    for path in instruction_paths(directory, scope):
        original = _without_rule(path)
        if path.exists() and path.read_text(encoding="utf-8") != original:
            claude._write(path, original)
            changed.append(path)
    return changed


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
    for key in ("mcp_servers", "features"):
        if key in config and not isinstance(config[key], MutableMapping):
            raise InvalidInput(f"{path} does not hold `{key}` as a table")
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
        # Codex defaults to the older handshake, which haskie's stateless MCP rejects.
        # Keep this shared capability on uninstall: other servers may depend on it too.
        if "features" not in config:
            config["features"] = tomlkit.table()
        config["features"]["mcp_2026_07_28"] = True
    claude._write(directory / "config.toml", tomlkit.dumps(config))
    return True


def validate(directory: Path, scope: Scope) -> None:
    """Check user-owned files before the installer changes any of them."""
    read_config(directory)
    path = directory / "hooks.json"
    claude._session_start(claude._read_settings(path), path)
    for path in instruction_paths(directory, scope):
        _without_rule(path)
