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
from haskie.errors import InvalidInput
from haskie.tables import installations


@pytest.mark.parametrize("scope", ["user", "project"])
def test_codex_install_and_uninstall(
    scope: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / ("profile" if scope == "user" else ".codex")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "profile"))
    directory.mkdir()
    instructions = (directory if scope == "user" else tmp_path) / "AGENTS.md"
    original_instructions = "# My instructions\n\nKeep changes small.\n"
    instructions.write_text(original_instructions)
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
    assert tomllib.loads(config.read_text())["features"]["mcp_2026_07_28"] is True
    assert "# Keep my settings" in config.read_text()
    matchers = json.loads(hooks.read_text())["hooks"]["SessionStart"]
    assert matchers[0] == user_hook
    assert len(matchers) == 2
    assert "--hook" in matchers[1]["hooks"][0]["command"]
    assert "--hook-rules" not in matchers[1]["hooks"][0]["command"]
    assert (directory / "skills/haskie/SKILL.md").is_file()
    assert (directory / "rules/haskie.md").is_file()
    assert instructions.read_text().count("<!-- haskie:start -->") == 1
    assert str(directory / "rules/haskie.md") in instructions.read_text()
    assert instructions.read_text().endswith(original_instructions)
    assert "trust" in result.output
    for _ in range(2):
        result = runner.invoke(cli, ["uninstall", "codex", *options])
        assert result.exit_code == 0, result.output
    expected = {**tomllib.loads(original), "features": {"mcp_2026_07_28": True}}
    assert tomllib.loads(config.read_text()) == expected
    assert json.loads(hooks.read_text()) == {"hooks": {"SessionStart": [user_hook]}}
    assert not (directory / "skills/haskie").exists()
    assert not (directory / "rules/haskie.md").exists()
    assert instructions.read_text() == original_instructions


@pytest.mark.parametrize(
    ("filename", "content", "error"),
    [
        ("config.toml", "[broken", "not valid TOML"),
        ("config.toml", 'mcp_servers = "wrong"', "as a table"),
        ("config.toml", 'features = "wrong"', "as a table"),
        ("hooks.json", "{broken", "not valid JSON"),
        ("hooks.json", '{"hooks": null}', "as a list of hooks"),
        ("AGENTS.md", "<!-- haskie:start -->\nunfinished", "haskie instruction block"),
        ("AGENTS.override.md", "<!-- haskie:end -->", "haskie instruction block"),
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


@pytest.mark.parametrize("scope", [Scope.USER, Scope.PROJECT])
@pytest.mark.parametrize("override_text", [None, "", "\n ", "Use project conventions.\n"])
def test_rule_reference_respects_codex_instruction_precedence(
    scope: Scope, override_text: str | None, tmp_path: Path
) -> None:
    directory = tmp_path / "profile with spaces"
    directory.mkdir()
    agents, override = codex.instruction_paths(directory, scope)
    agents.write_text("Existing instructions without a final newline")
    if override_text is not None:
        override.write_text(override_text)
    expected = agents
    if override_text is not None and (scope == Scope.PROJECT or override_text.strip()):
        expected = override
    before = {path: path.read_text() for path in (agents, override) if path.exists()}
    for _ in range(2):
        assert codex.install_rule_reference(directory, scope) == expected
    assert expected.read_text().count(codex.RULE_START) == 1
    assert f"(<{directory / 'rules/haskie.md'}>)" in expected.read_text()
    assert "even when you know the answer" in expected.read_text()
    assert codex.remove_rule_reference(directory, scope) == [expected]
    assert codex.remove_rule_reference(directory, scope) == []
    assert {path: path.read_text() for path in before} == before


def test_rule_reference_preserves_symlink_permissions_and_later_edits(tmp_path: Path) -> None:
    target = tmp_path / "dotfiles.md"
    original = "# Personal instructions\n"
    target.write_text(original)
    target.chmod(0o600)
    agents = tmp_path / "AGENTS.md"
    agents.symlink_to(target)
    codex.install_rule_reference(tmp_path, Scope.USER)
    target.write_text("Before\n" + target.read_text() + "\nAfter")
    codex.remove_rule_reference(tmp_path, Scope.USER)
    assert agents.is_symlink()
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.read_text() == "Before\n" + original + "\nAfter"


@pytest.mark.parametrize(
    "text",
    [
        codex.RULE_START,
        codex.RULE_END,
        codex.RULE_END + codex.RULE_START,
        codex.RULE_START * 2 + codex.RULE_END,
        codex.RULE_START + codex.RULE_END * 2,
    ],
)
def test_rule_reference_rejects_ambiguous_markers(text: str, tmp_path: Path) -> None:
    path = tmp_path / "AGENTS.md"
    path.write_text(text)
    with pytest.raises(InvalidInput, match="haskie instruction block"):
        codex.install_rule_reference(tmp_path, Scope.USER)
    assert path.read_text() == text


def test_rule_reference_removes_blocks_after_override_changes(tmp_path: Path) -> None:
    codex.install_rule_reference(tmp_path, Scope.USER)
    (tmp_path / "AGENTS.override.md").write_text("New override\n")
    codex.install_rule_reference(tmp_path, Scope.USER)
    assert codex.remove_rule_reference(tmp_path, Scope.USER) == [
        tmp_path / "AGENTS.md",
        tmp_path / "AGENTS.override.md",
    ]
    assert (tmp_path / "AGENTS.md").read_text() == ""
    assert (tmp_path / "AGENTS.override.md").read_text() == "New override\n"


@pytest.mark.parametrize("legacy", [False, True], ids=["current hook", "older installed hook"])
def test_codex_hook_supplies_session_and_preserves_older_rule_output(
    legacy: bool,
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
    if legacy:
        hook["command"] += " " + shlex.join(["--hook-rules", str(claude.rule_path(directory))])
    result = runner.invoke(
        cli,
        shlex.split(hook["command"])[1:],
        input=json.dumps({"session_id": "codex-123", "source": "startup", "cwd": str(tmp_path)}),
    )
    assert result.exit_code == 0, result.output
    assert "session id is codex-123" in result.output
    assert (claude.render_rule([]) in result.output) == legacy
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
    expected = {**tomllib.loads(original), "features": {"mcp_2026_07_28": True}}
    assert tomllib.loads(config.read_text()) == expected


@pytest.mark.parametrize("enabled", [None, False, True])
def test_install_enables_codex_protocol_without_changing_other_features(
    enabled: bool | None, tmp_path: Path
) -> None:
    config = tmp_path / "config.toml"
    original = "[features]\n# Keep this setting\nshell_snapshot = false\n"
    if enabled is not None:
        original += f"mcp_2026_07_28 = {str(enabled).lower()}\n"
    config.write_text(original)
    codex.register_mcp(tmp_path, "http://localhost:9234/mcp")
    assert tomllib.loads(config.read_text())["features"] == {
        "shell_snapshot": False,
        "mcp_2026_07_28": True,
    }
    assert "# Keep this setting" in config.read_text()
    codex.register_mcp(tmp_path, None)
    assert tomllib.loads(config.read_text())["features"]["mcp_2026_07_28"] is True
