"""The activation suite without Claude: a fake `claude` (`EVAL_CLAUDE_BIN`) that searches haskie
only when it finds the rule installed, and a fake haskie that answers the collection lookup and
the hook's search. These prove what each setup hands the agent and how a run is scored."""

from __future__ import annotations

import json
import stat
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from evals import run
from evals.activation import run as activation

FAKE_CLAUDE = """#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
settings = json.load(open(args[args.index("--settings") + 1]))
ruled = os.path.exists(os.path.join(os.getcwd(), ".claude", "rules", "haskie.md"))
tools = ["ToolSearch", "mcp__haskie__search_excerpts"] if ruled else ["Write"]
for name in tools:
    print(json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": name, "name": name, "input": {}}]}}))
hooked = "hooks" in settings
print(json.dumps({"type": "result", "num_turns": 3 if ruled else 1, "total_cost_usd": 0.02,
                  "result": "hooked" if hooked else "done"}))
"""


class Haskie(BaseHTTPRequestHandler):
    excerpts: list[dict] = []

    def do_GET(self) -> None:
        if self.path.startswith("/api/collections/"):
            body = {"description": "Programming books: Raft, Git"}
        else:
            body = {"excerpts": Haskie.excerpts}
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def haskie() -> Iterator[str]:
    Haskie.excerpts = [{"header": "Raft > Election", "location": "raft.pdf p.8", "text": "x"}]
    server = ThreadingHTTPServer(("127.0.0.1", 0), Haskie)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLAUDE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("EVAL_CLAUDE_BIN", str(fake))
    monkeypatch.setattr(run, "WORK_ROOT", tmp_path / "work")


RAFT = activation.Prompt("raft", "When is a Raft candidate's log up to date?", True, "covered")
MATH = activation.Prompt("math", "What is 17 times 23?", False, "control")


@pytest.mark.parametrize(
    ("setup", "searched", "nudged"),
    [
        ("none", False, False),
        ("rule", True, False),
        ("hook", False, True),
        ("rule+hook", True, True),
    ],
)
def test_each_setup_hands_the_agent_its_rule_or_hook(
    claude: None, haskie: str, tmp_path: Path, setup: str, searched: bool, nudged: bool
) -> None:
    trial = activation.trial(RAFT, setup, 0, "haiku", haskie, "books", tmp_path / "out")

    assert (trial.searched, trial.nudged, trial.consulted) == (searched, nudged, searched or nudged)
    transcript = (tmp_path / "out" / setup / "raft" / "0" / "transcript.jsonl").read_text()
    assert ('"hooked"' in transcript) == nudged, "the hook is in the settings exactly when on"
    work = run.WORK_ROOT / "activation" / setup / "raft" / "0"
    assert not work.exists(), "the working directory, rule and all, is cleaned up"


def test_the_hook_only_counts_as_consulted_when_it_had_passages(
    claude: None, haskie: str, tmp_path: Path
) -> None:
    Haskie.excerpts = []

    trial = activation.trial(MATH, "hook", 0, "haiku", haskie, "books", tmp_path / "out")

    assert (trial.nudged, trial.consulted) == (False, False), "nothing to inject: it stayed quiet"


def test_a_transcript_gives_the_tools_in_order_the_cost_and_the_limit() -> None:
    lines = [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "ToolSearch"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "hm"}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Write"}]}},
        {"type": "result", "num_turns": 2, "total_cost_usd": 0.05, "result": "done"},
    ]
    text = "\n".join(json.dumps(line) for line in lines) + "\nnot json\n"

    assert activation.read(text) == (["ToolSearch", "Write"], 2, 0.05, False)
    limited = text.replace('"done"', '"You\'ve hit your session limit"')
    assert activation.read(limited)[3] is True


def test_the_report_counts_should_trigger_and_controls_apart_and_leaves_out_limited() -> None:
    def trial(pid: str, setup: str, activate: bool, searched: bool, limited: bool = False):
        return activation.Trial(pid, setup, 0, activate, [], searched, False, 1, 0.01, 1.0, limited)

    trials = [
        trial("raft", "none", True, False),
        trial("raft", "rule", True, True),
        trial("math", "rule", False, True),
        trial("math", "none", False, False, limited=True),
    ]

    text = activation.render(trials, ["none", "rule"])

    assert "| none | 0/1 | 0/1 | 0/0 | 0/0 | 0.010 |" in text
    assert "| rule | 1/1 | 1/1 | 1/1 | 1/1 | 0.010 |" in text, "a searched control is a false one"
    assert "1 runs hit the session limit" in text


def test_the_prompts_never_name_haskie_and_hold_both_kinds() -> None:
    prompts = activation.load()

    assert len({p.id for p in prompts}) == len(prompts), "ids are unique"
    assert {p.activate for p in prompts} == {True, False}
    assert not any("haskie" in p.prompt.lower() for p in prompts), "the point is not to be told"
