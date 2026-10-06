"""`sections.py` against a fake haskie serving a sections map and the lines behind it: which
section holds the answer, by gold quote or judged passage, and what its descriptors name."""

from __future__ import annotations

import json
import threading
import urllib.parse
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from evals.bookqa import qrels, sections
from evals.bookqa.sections import Section
from evals.bookqa.tests import books

LINES = {i + 1: f"{books.LINES[i]}\n" for i in range(len(books.LINES))}


class Haskie(BaseHTTPRequestHandler):
    map: list[dict] = []
    reads: list[tuple[int, int]] = []

    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)
        params = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        if url.path == "/api/search/sections":
            body = {"sections": Haskie.map, "documents": [], "uncovered": []}
        else:
            start, end = int(params["line_start"]), int(params["line_end"])
            Haskie.reads.append((start, end))
            body = {"text": "".join(LINES.get(n, "") for n in range(start, end + 1))}
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        pass


def _section(start: int, end: int, header: str, descriptors: list[str]) -> dict:
    return {
        "document": books.PDF,
        "header": header,
        "line_start": start,
        "line_end": end,
        "score": 1.0,
        "descriptors": descriptors,
        "id": "x",  # haskie sends more than `Section` reads
    }


@pytest.fixture
def haskie() -> Iterator[str]:
    Haskie.reads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Haskie)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return books.corpus(tmp_path / "corpus")


def test_the_first_section_holding_the_gold_quote_sets_the_rank(corpus: Path, haskie: str) -> None:
    Haskie.map = [
        _section(1, 10, "Workers 0-9", ["retry", "worker"]),
        _section(60, 70, "Workers 59-69", ["budget", "attempts", "455"]),
    ]
    record = books.record(corpus)  # gold: LINES[65], line 66
    found, _, size = sections.search(haskie, "books", record.query)

    gold, judged, either = sections.score(record, found, sections.Texts(haskie), [])

    assert (gold, judged, either) == (2, None, 2)
    assert size > 0 and [s.header for s in found] == ["Workers 0-9", "Workers 59-69"]


def test_a_passage_judged_an_answer_counts_where_no_gold_quote_is(
    corpus: Path, haskie: str
) -> None:
    Haskie.map = [_section(1, 10, "Workers 0-9", [])]
    record = books.record(corpus)
    answer = qrels.Judgment(record.id, books.PDF, "k", 2, books.LINES[4], "m", "v2", "t")

    gold, judged, either = sections.score(
        record, sections.search(haskie, "b", "q")[0], sections.Texts(haskie), [answer]
    )

    assert (gold, judged, either) == (None, 1, 1)


def test_a_long_section_is_read_in_pieces_and_once(haskie: str) -> None:
    text = sections.Texts(haskie)
    long = Section(books.PDF, "all", 1, 900, 1.0)

    text(long)
    text(long)

    assert Haskie.reads == [(1, 400), (401, 800), (801, 900)]


def test_answer_terms_are_the_facts_words_the_question_does_not_use(corpus: Path) -> None:
    record = books.record(
        corpus, query="How many attempts may worker 65 make?", expected_facts=["455 attempts"]
    )

    assert sections.answer_terms(record) == ["455"]
    assert sections.share(["455", "budget"], "a budget of 455") == 1.0
    assert sections.share(["455", "budget"], "Workers 59-69") == 0.0


def test_descriptors_are_scored_against_the_heading_of_the_answering_section(
    corpus: Path, haskie: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    Haskie.map = [_section(60, 70, "Workers 59-69", ["budget", "455 attempts"])]
    monkeypatch.setattr(qrels, "JUDGMENTS", corpus / "none.jsonl")
    record = books.record(corpus, expected_facts=["455 attempts", "a retry budget"])

    (scored,) = sections.evaluate([record], haskie, "books")

    assert scored.terms == ["455", "budget", "retry"]
    assert scored.in_descriptors == pytest.approx(2 / 3)
    assert scored.in_header == 0.0
    text = sections.render([scored])
    assert "| all | 1 | 1.00 | 1.00 | 1.00 |" in text


def test_unanswerable_questions_are_left_out(corpus: Path, haskie: str) -> None:
    Haskie.map = []

    assert sections.evaluate([books.unanswerable(corpus)], haskie, "books") == []


def test_a_relationship_is_complete_only_once_both_sides_are_in_a_section(
    corpus: Path, haskie: str
) -> None:
    Haskie.map = [
        _section(60, 70, "Workers 59-69", []),  # worker 65's side
        _section(1, 10, "Workers 0-9", []),  # worker 3's side
    ]
    record = books.relationship(corpus)
    found = sections.search(haskie, "books", record.query)[0]
    text = sections.Texts(haskie)

    assert sections.complete(record, found, text) == 2
    assert sections.complete(record, found[:1], text) is None
