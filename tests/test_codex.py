"""Codex installation through the real CLI, isolated from the user's configuration."""

import json
import shlex
import sqlite3
import tomllib
from pathlib import Path

import pytest
from conftest import holding_a_document
from sqlalchemy import select
from typer.testing import CliRunner

from haskie import claude, codex, db, home
from haskie import cli as cli_module
from haskie.claude import Scope
from haskie.cli import cli
from haskie.collection.collection import Collection
from haskie.tables import installations


@pytest.mark.parametrize("scope", ["user", "project"])
def test_codex_install_and_uninstall(
    scope: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / ("profile" if scope == "user" else ".codex")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "profile"))
    directory.mkdir()
    config = directory / "config.toml"
    original = '# Keep my settings\nmodel = "gpt-5.4"\n[mcp_servers.docs]\nurl = "https://docs.example/mcp"\n'
    config.write_text(original)
    hooks = directory / "hooks.json"
    user_hook = {"hooks": [{"type": "command", "command": "echo hello"}]}
    hooks.write_text(json.dumps({"hooks": {"SessionStart": [user_hook]}}))
    served = []
    monkeypatch.setattr(cli_module, "_serve", lambda url, wait: served.append((url, wait)))
    runner = CliRunner()
    options = ["--home", str(tmp_path / "data"), "--scope", scope]
    for _ in range(2):
        result = runner.invoke(cli, ["install", "codex", *options])
        assert result.exit_code == 0, result.output
    assert served == [(cli_module.MCP_URL, True)] * 2
    assert tomllib.loads(config.read_text())["mcp_servers"]["haskie"] == {"url": cli_module.MCP_URL}
    assert "# Keep my settings" in config.read_text()
    matchers = json.loads(hooks.read_text())["hooks"]["SessionStart"]
    assert matchers[0] == user_hook
    assert len(matchers) == 2
    assert "--hook" in matchers[1]["hooks"][0]["command"]
    assert "--hook-rules" in matchers[1]["hooks"][0]["command"]
    assert (directory / "skills/haskie/SKILL.md").is_file()
    assert (directory / "rules/haskie.md").is_file()
    assert "trust" in result.output
    for _ in range(2):
        result = runner.invoke(cli, ["uninstall", "codex", *options])
        assert result.exit_code == 0, result.output
    assert tomllib.loads(config.read_text()) == tomllib.loads(original)
    assert json.loads(hooks.read_text()) == {"hooks": {"SessionStart": [user_hook]}}
    assert not (directory / "skills/haskie").exists()
    assert not (directory / "rules/haskie.md").exists()


@pytest.mark.parametrize(
    ("filename", "content", "error"),
    [
        ("config.toml", "[broken", "not valid TOML"),
        ("config.toml", 'mcp_servers = "wrong"', "as a table"),
        ("hooks.json", "{broken", "not valid JSON"),
        ("hooks.json", '{"hooks": null}', "as a list of hooks"),
    ],
)
@pytest.mark.parametrize("command", ["install", "uninstall"])
def test_invalid_configuration_is_never_overwritten(
    filename: str,
    content: str,
    error: str,
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "profile"))
    directory = tmp_path / "profile"
    directory.mkdir()
    path = directory / filename
    path.write_text(content)
    result = CliRunner().invoke(cli, [command, "codex", "--home", str(tmp_path / "data")])
    assert result.exit_code == 1
    assert error in result.output
    assert path.read_text() == content
    assert list(directory.iterdir()) == [path]


def test_uninstall_absent_codex_does_not_create_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "profile"))
    result = CliRunner().invoke(cli, ["uninstall", "codex", "--home", str(tmp_path / "data")])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "profile").exists()
    assert not (tmp_path / "data").exists()


def test_generated_codex_hook_supplies_session_and_search_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "profile with spaces"
    monkeypatch.setenv("CODEX_HOME", str(directory))
    served = []
    monkeypatch.setattr(cli_module, "_serve", lambda url, wait: served.append((url, wait)))
    monkeypatch.setattr(claude, "own_command", lambda: ["haskie"])
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "install",
            "codex",
            "--home",
            str(tmp_path / "data"),
            "--url",
            "http://localhost:9234/mcp",
        ],
    )
    assert result.exit_code == 0, result.output
    settings = json.loads((directory / "hooks.json").read_text())
    hook = settings["hooks"]["SessionStart"][0]["hooks"][0]
    result = runner.invoke(
        cli,
        shlex.split(hook["command"])[1:],
        input=json.dumps({"session_id": "codex-123", "source": "startup", "cwd": str(tmp_path)}),
    )
    assert result.exit_code == 0, result.output
    assert "session id is codex-123" in result.output
    assert claude.render_rule([]) in result.output
    assert served == [("http://localhost:9234/mcp", True), ("http://localhost:9234", False)]


@pytest.mark.anyio
@pytest.mark.parametrize("state", ["installed", "removed", "other-home", "invalid-hooks"])
async def test_codex_refresh_alongside_claude(state: str, tmp_path: Path) -> None:
    await Collection.create("roasting", "Coffee roasting.")
    await holding_a_document("roasting")
    for agent, filename in [("claude", "settings.json"), ("codex", "hooks.json")]:
        directory = tmp_path / agent
        claude.write_instructions(directory, [])
        owner = tmp_path / "other" if agent == "codex" and state == "other-home" else home.HOME
        claude.install_hook(directory, owner, claude.MCP_URL, filename=filename)
        await claude.record_installation(directory, agent)
    directory = tmp_path / "codex"
    if state == "removed":
        claude.remove_instructions(directory)
    elif state == "invalid-hooks":
        (directory / "hooks.json").write_text("{")
    await claude.refresh_installations()
    assert "roasting: Coffee roasting" in claude.rule_path(tmp_path / "claude").read_text()
    skill = claude.skill_path(directory)
    assert skill.exists() == (state != "removed")
    refreshed = skill.exists() and "roasting: Coffee roasting" in skill.read_text()
    assert refreshed == (state == "installed")
    assert await claude.forget_installation(directory, "codex")
    assert not await claude.forget_installation(directory, "codex")
    async with db.read() as conn:
        assert list(await conn.scalars(select(installations.c.agent))) == ["claude"]


def test_upgrade_keeps_claude_installations_and_accepts_codex(tmp_path: Path) -> None:
    with sqlite3.connect(tmp_path / "old.db") as connection:
        connection.executescript(
            "create table installations (agent text check (agent in ('claude')), "
            "directory text, primary key (agent, directory));"
            "insert into installations values ('claude', '/home/user/.claude');"
            "pragma user_version = 35;"
        )
        db.migrate(connection)
        connection.execute("insert into installations values ('codex', '/home/user/.codex')")
        assert connection.execute("select * from installations order by agent").fetchall() == [
            ("claude", "/home/user/.claude"),
            ("codex", "/home/user/.codex"),
        ]
        assert connection.execute("pragma user_version").fetchone() == (db.SCHEMA_VERSION,)


@pytest.mark.parametrize("value", [None, "", "~/codex-profile"])
def test_codex_home(value: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    if value is not None:
        monkeypatch.setenv("CODEX_HOME", value)
    assert codex.codex_dir(Scope.USER) == tmp_path / ("codex-profile" if value else ".codex")
    assert codex.codex_dir(Scope.PROJECT) == tmp_path / ".codex"


@pytest.mark.parametrize(
    "original",
    [
        'mcp_servers = { docs = { url = "https://docs.example/mcp" } }\n',
        '[mcp_servers.docs]\nurl = "https://docs.example/mcp"\n'
        '[profiles.work]\nmodel = "gpt-5.4"\n'
        '[mcp_servers.local]\ncommand = "local-mcp"\n',
    ],
    ids=["inline table", "table split across sections"],
)
def test_mcp_edits_preserve_toml_shapes(original: str, tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(original)
    assert codex.register_mcp(tmp_path, "http://localhost:9234/mcp")
    assert tomllib.loads(config.read_text())["mcp_servers"]["haskie"]["url"] == (
        "http://localhost:9234/mcp"
    )
    assert codex.register_mcp(tmp_path, None)
    assert tomllib.loads(config.read_text()) == tomllib.loads(original)
