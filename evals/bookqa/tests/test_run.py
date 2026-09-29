"""`run.py` against a real HTTP server that answers like haskie's search API: what it sends, what
it records, and that it never reaches for generation or the agent eval."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.parse
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import msgspec
import pytest

from evals.bookqa import run
from evals.bookqa.metrics import Outcome
from evals.bookqa.tests import books

ROOT = Path(__file__).resolve().parents[3]


class Haskie:
    """The search API's shape, as much as `run.py` touches: a PUT per mode, a GET per question."""

    def __init__(self, answers: dict[str, list[dict]]) -> None:
        self.answers = answers
        self.overrides: list[tuple[str, dict]] = []
        self.searches: list[dict[str, str]] = []
        self.bodies: dict[str, bytes] = {}

    def handler(self) -> type[BaseHTTPRequestHandler]:
        haskie = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                url = urllib.parse.urlparse(self.path)
                assert url.path == "/api/search/explore", url.path
                params = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
                haskie.searches.append(params)
                body = json.dumps(haskie.answers.get(params["q"], [])).encode()
                haskie.bodies[params["q"]] = body
                self._send(body)

            def do_PUT(self) -> None:
                length = int(self.headers["content-length"])
                haskie.overrides.append((self.path, json.loads(self.rfile.read(length))))
                self._send(b"{}")

            def _send(self, body: bytes) -> None:
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        return Handler


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return books.corpus(tmp_path / "corpus")


@pytest.fixture
def haskie(corpus: Path) -> Iterator[tuple[Haskie, str]]:
    gold = books.record(corpus)
    fake = Haskie(
        {
            gold.query: [
                # haskie's passage rows carry more than `Found` reads; the rest is ignored
                {"document": books.PDF, "text": books.LINES[3], "score": 0.9, "location": "p.1"},
                {"document": books.PDF, "text": books.LINES[65], "score": 0.8, "also_in": []},
            ]
        }
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), fake.handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield fake, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_every_question_is_searched_once_per_mode_on_the_suite_collection(
    corpus: Path, haskie: tuple[Haskie, str]
) -> None:
    fake, api = haskie
    records = [books.record(corpus), books.unanswerable(corpus)]

    outcomes = run.evaluate(records, api, "bookqa-test", ["fts", "hybrid+rerank"])

    assert fake.overrides == [
        ("/api/collections/bookqa-test/overrides", {"search": run.MODES["fts"]}),
        ("/api/collections/bookqa-test/overrides", {"search": run.MODES["hybrid+rerank"]}),
    ]
    assert [s["q"] for s in fake.searches] == [r.query for r in records] * 2
    assert all(
        (s["granularity"], s["limit"], s["collections"]) == ("passage", "10", "bookqa-test")
        for s in fake.searches
    )
    assert [(o.mode, o.id) for o in outcomes] == [
        (mode, r.id) for mode in ("fts", "hybrid+rerank") for r in records
    ]


def test_an_outcome_records_the_ranking_its_scores_size_and_latency(
    corpus: Path, haskie: tuple[Haskie, str]
) -> None:
    fake, api = haskie
    answerable, unanswerable = books.record(corpus), books.unanswerable(corpus)

    found, empty = run.evaluate([answerable, unanswerable], api, "bookqa-test", ["fts"])

    assert [f.text for f in found.results] == [books.LINES[3], books.LINES[65]]
    assert found.scores is not None and found.scores.recall == {1: 0.0, 5: 1.0, 10: 1.0}
    assert found.scores.mrr == 0.5
    assert found.bytes == len(fake.bodies[answerable.query]) and found.seconds > 0
    assert (found.abstained, found.query_type, found.source) == (False, "direct", books.PDF)
    assert (empty.results, empty.scores, empty.abstained) == ([], None, True)


def test_the_outcomes_and_the_report_are_written_side_by_side(
    corpus: Path, haskie: tuple[Haskie, str], tmp_path: Path
) -> None:
    _, api = haskie
    outcomes = run.evaluate([books.record(corpus)], api, "bookqa-test", ["fts"])

    path = run.write(outcomes, tmp_path / "report")

    lines = (tmp_path / "report" / "outcomes.jsonl").read_text().splitlines()
    assert [msgspec.json.decode(line, type=Outcome) for line in lines] == outcomes
    assert path.read_text().startswith("# Book query retrieval")


def test_an_empty_dataset_is_refused_before_anything_is_searched(
    tmp_path: Path, haskie: tuple[Haskie, str]
) -> None:
    fake, api = haskie
    (tmp_path / "dataset.jsonl").write_text("")

    assert run.main(["--dataset", str(tmp_path / "dataset.jsonl"), "--api", api]) == 1
    assert fake.searches == [] and fake.overrides == []


def test_a_dataset_that_fails_review_is_refused_before_anything_is_searched(
    corpus: Path, tmp_path: Path, haskie: tuple[Haskie, str]
) -> None:
    fake, api = haskie
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(msgspec.json.encode(books.record(corpus, expected_facts=[])).decode() + "\n")
    argv = ["--dataset", str(dataset), "--api", api, "--corpus", str(corpus)]

    assert run.main(argv) == 1
    assert fake.searches == [] and fake.overrides == []


@pytest.mark.parametrize(
    ("module", "absent"),
    [
        ("evals.bookqa.run", ["evals.bookqa.generate", "evals.run"]),
        ("evals.run", ["evals.bookqa"]),
    ],
)
def test_the_retrieval_run_and_the_agent_eval_load_none_of_each_other(
    module: str, absent: list[str]
) -> None:
    """The retrieval phase cannot reach generation (so no Claude), and neither suite runs the
    other: importing one loads none of the other's modules."""
    probe = f"import sys, {module}; print(sorted(m for m in sys.modules if m.startswith('evals')))"
    loaded = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout

    assert all(f"'{name}'" not in loaded for name in absent), loaded
