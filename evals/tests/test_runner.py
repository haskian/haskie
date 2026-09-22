"""What an arm turns into on disk and on the command line. Nothing here starts a run."""

from pathlib import Path

import msgspec

from evals.runner import Arm, allowed, argv, prepare, workspace

SESSION = "3f2a1c4e-0000-4000-8000-000000000000"


def test_the_command_line_shuts_out_everything_the_eval_did_not_set(tmp_path: Path) -> None:
    """Without `--setting-sources project` the user's own CLAUDE.md and settings join the run,
    and the arms stop differing only in what the eval gave them."""
    line = argv(Arm(name="b"), "do the task", tmp_path, session_id=SESSION)
    assert line[1:3] == ["-p", "do the task"]
    assert line[line.index("--setting-sources") + 1] == "project"


def test_user_memory_enables_user_settings_source(tmp_path: Path) -> None:
    line = argv(Arm(name="d", memory="# rules", memory_scope="user"), "do the task", tmp_path, session_id=SESSION)
    assert line[line.index("--setting-sources") + 1] == "user,project"
    assert "--strict-mcp-config" in line
    assert line[line.index("--session-id") + 1] == SESSION


def test_the_mcp_config_is_named_absolutely(tmp_path: Path) -> None:
    """The run starts in the workspace, one level below the directory holding the config, so a
    relative path resolves against the wrong place and every run dies before its first turn."""
    line = argv(Arm(name="b"), "do the task", Path("evals/runs/t/b/0"), session_id=SESSION)

    named = Path(line[line.index("--mcp-config") + 1])

    assert named.is_absolute()
    assert named.name == "mcp.json"


def test_the_no_haskie_arm_cannot_reach_a_server_that_is_running(tmp_path: Path) -> None:
    """The baseline shares a machine with the server, so an empty config is what makes it a
    baseline rather than a run that happened not to search."""
    prepare(Arm(name="a", mcp=False), tmp_path)
    config = msgspec.json.decode((tmp_path / "mcp.json").read_bytes())
    assert config == {"mcpServers": {}}
    assert not any(tool.startswith("mcp__") for tool in allowed(Arm(name="a", mcp=False)))


def test_a_haskie_arm_may_read_the_library_and_not_change_it(tmp_path: Path) -> None:
    tools = allowed(Arm(name="b"))
    assert "mcp__haskie__search_text" in tools
    assert not any(verb in tool for tool in tools for verb in ("add_", "remove_", "describe_"))


def test_the_skill_and_the_memory_land_where_claude_code_looks(tmp_path: Path) -> None:
    arm = Arm(name="d", skill="---\nname: haskie\n---\n\nbody", memory="# rules\n")
    work = prepare(arm, tmp_path)
    assert (work / ".claude/skills/haskie/SKILL.md").read_text().endswith("body")
    assert (work / "CLAUDE.md").read_text() == "# rules\n"


def test_user_memory_is_written_to_the_eval_config(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "claude-config"
    monkeypatch.setenv("EVAL_CLAUDE_CONFIG_DIR", str(config))
    work = prepare(Arm(name="d", memory="# user rules\n", memory_scope="user"), tmp_path / "run")
    assert not (work / "CLAUDE.md").exists()
    assert (config / "CLAUDE.md").read_text() == "# user rules\n"


def test_an_arm_without_guidance_leaves_an_empty_workspace(tmp_path: Path) -> None:
    work = prepare(Arm(name="b"), tmp_path)
    assert list(work.iterdir()) == []


def test_the_runs_own_record_is_outside_what_the_agent_can_see(tmp_path: Path) -> None:
    """A baseline run read `mcp.json` and its own transcript out of the directory it was working
    in, and reported on the arm it was in. The record belongs one level up."""
    work = prepare(Arm(name="b"), tmp_path)

    assert (tmp_path / "mcp.json").is_file()
    assert not (work / "mcp.json").exists()
    assert workspace(tmp_path) == work
