"""`steer.py`, arm H's hook, and how `run.py` wires arms G and H: what the agent is told, and
what it is not."""

import io
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from evals import run, steer
from evals.run import Task

TASK = Task("synth_s1_n500_l1", "answers", [], Path("corpus"), "eval-synth-s1-n500")
PROMPT = (
    "You are completing the synth_s1_n500_l1 evaluation task. Work only in the current "
    "directory; do not read anything outside it.\n\n"
    "Write `answers.py` in the current directory. Standard library only.\n\n"
    "Our internal documentation describes the service that deduplicates the general ledger "
    "arriving from the carrier API, every night. Record four facts about that service:\n\n"
    "    PORT: int                     # the port it listens on\n"
    "    OWNING_TEAM: str              # the team that owns it\n"
)


def test_the_questions_are_the_prompts_prose_sentences_without_the_header_or_code() -> None:
    assert steer.questions(PROMPT) == [
        "Write `answers.py` in the current directory.",
        "Our internal documentation describes the service that deduplicates the general ledger "
        "arriving from the carrier API, every night.",
        "Record four facts about that service:",
    ]


def test_questions_skip_short_instructions_and_code_and_cap_at_five() -> None:
    asked = steer.questions(PROMPT)

    assert "Standard library only." not in asked, "an instruction, not a topic"
    assert all("PORT" not in q for q in asked), "code is not a question"
    assert not any(q.startswith("You are completing") for q in asked), "the eval's own header"
    many = " ".join(f"Sentence number {i} is about a different topic entirely." for i in range(9))
    assert len(steer.questions(many)) == steer.MAX_QUESTIONS


def test_a_long_sentence_is_cut_at_a_word_to_what_a_search_takes() -> None:
    (asked,) = steer.questions("word " * 200)

    assert len(asked) <= steer.MAX_QUERY and asked.endswith("word")


def test_the_nudge_names_the_collection_and_where_each_passage_is() -> None:
    excerpts = [
        {"header": "finch-agate > Purpose", "location": "doc-1.md L5-5", "text": "Dedupes.\n"},
        {"header": "Retries", "location": "book.pdf p.3", "text": "x " * 400},
    ]

    text = steer.nudge("eval-synth-s1-n500", excerpts)

    assert "collection `eval-synth-s1-n500`" in text
    assert "- finch-agate > Purpose (doc-1.md L5-5): Dedupes." in text
    assert text.count(" ...") == 1, "a long passage is cut, a short one shown whole"
    assert steer.nudge("c", []) == "", "no passage, no nudge"


class Excerpts(BaseHTTPRequestHandler):
    """`/api/search/excerpts` answering in turn order, as haskie does: not best first."""

    asked: list[list[str]] = []

    def do_GET(self) -> None:
        from urllib.parse import parse_qs, urlparse

        params = parse_qs(urlparse(self.path).query)
        Excerpts.asked.append(params["q"])
        assert int(params["limit"][0]) >= len(params["q"]), "a slot per question at least"
        body = json.dumps(
            {
                "excerpts": [
                    {"header": "weak", "location": "a", "text": "w", "score": 0.1},
                    {"header": "best", "location": "b", "text": "b", "score": 0.9},
                    {"header": "good", "location": "c", "text": "g", "score": 0.5},
                    {"header": "worst", "location": "d", "text": "x", "score": 0.05},
                ]
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def haskie() -> Iterator[str]:
    Excerpts.asked = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Excerpts)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_the_hook_prints_the_best_passages_best_first(
    haskie: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"prompt": PROMPT})))

    assert steer.main(["--api", haskie, "--collection", "c"]) == 0

    shown = [
        line.split(" (")[0] for line in capsys.readouterr().out.splitlines() if line[:2] == "- "
    ]
    assert shown == ["- best", "- good", "- weak"]
    assert Excerpts.asked == [steer.questions(PROMPT)], "every question in one search"


@pytest.mark.parametrize("stdin", ["not json", json.dumps({"prompt": PROMPT})])
def test_the_hook_never_fails_the_prompt(
    stdin: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))

    assert steer.main(["--api", "http://127.0.0.1:9", "--collection", "c"]) == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("arm", ["g", "h"])
def test_arms_g_and_h_are_told_nothing_of_haskie_but_have_its_tools(arm: str) -> None:
    prompt = run.prompt_for(TASK, arm, "eval-synth-s1-n500")

    assert "haskie" not in prompt.lower() and "collection" not in prompt.lower()
    assert "mcp__haskie__search_excerpts" in run.allowed_tools(arm)


def test_only_arm_h_gets_a_hook_and_it_runs_the_steering_module() -> None:
    assert all(run.hook_settings(arm, "http://x", "c") is None for arm in "abcdefg")

    settings = run.hook_settings("h", "http://127.0.0.1:8123", "eval-synth-s1-n500")

    assert settings is not None
    (entry,) = settings["hooks"]["UserPromptSubmit"]
    (hook,) = entry["hooks"]
    assert hook["type"] == "command"
    assert (
        "-m evals.steer --api http://127.0.0.1:8123 --collection eval-synth-s1-n500"
        in (hook["command"])
    )


def test_a_plain_run_still_runs_arms_a_to_f() -> None:
    assert run.DEFAULT_ARMS == ("a", "b", "c", "d", "e", "f")
    assert set(run.ARMS) - set(run.DEFAULT_ARMS) == {"g", "h"}
