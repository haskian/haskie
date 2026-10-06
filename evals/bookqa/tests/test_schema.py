"""`schema.py`: what makes a record valid, and every check `review` and `--accept` run on it,
against a real PDF and markdown file (`books.py`)."""

from pathlib import Path

import msgspec
import pytest

from evals.bookqa import schema
from evals.bookqa.schema import Passage, Record
from evals.bookqa.tests import books


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return books.corpus(tmp_path / "corpus")


def _problems(record: Record, corpus: Path) -> list[str]:
    return [issue.problem for issue in schema.validate([record], corpus)]


def test_a_well_formed_answerable_record_passes_every_check(corpus: Path) -> None:
    assert _problems(books.record(corpus), corpus) == []


def test_a_well_formed_unanswerable_record_passes_every_check(corpus: Path) -> None:
    assert _problems(books.unanswerable(corpus), corpus) == []


def test_records_round_trip_through_jsonl(corpus: Path, tmp_path: Path) -> None:
    written = [books.record(corpus), books.unanswerable(corpus)]
    path = tmp_path / "dataset.jsonl"
    path.write_text(schema.dump(written), encoding="utf-8")

    assert schema.load(path) == (written, [])


def test_a_missing_dataset_is_empty_not_an_error(tmp_path: Path) -> None:
    assert schema.load(tmp_path / "absent.jsonl") == ([], [])


@pytest.mark.parametrize(
    ("name", "mangle", "expected"),
    [
        ("a required field missing", lambda d: d.pop("expected_answer"), "not a valid record"),
        ("a field of the wrong type", lambda d: d.update(answerable="yes"), "not a valid record"),
        ("an unknown query type", lambda d: d.update(query_type="trivia"), "not a valid record"),
        ("a field the schema lacks", lambda d: d.update(top_10=[]), "not a valid record"),
        ("metadata missing a field", lambda d: d["meta"].pop("source_sha256"), "not a valid"),
    ],
)
def test_a_line_that_is_not_a_valid_record_is_an_issue_with_its_line_number(
    corpus: Path, tmp_path: Path, name: str, mangle, expected: str
) -> None:
    good = msgspec.to_builtins(books.record(corpus))
    bad = msgspec.to_builtins(books.record(corpus, id="other", query="Another question?"))
    mangle(bad)
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        msgspec.json.encode(good).decode() + "\n" + msgspec.json.encode(bad).decode() + "\n"
    )

    records, issues = schema.load(path)

    assert [r.id for r in records] == ["book-000000-01"], name
    assert [i.record for i in issues] == ["line 2"], name
    assert expected in issues[0].problem, name


def test_a_line_that_is_not_json_is_an_issue(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text("{not json\n")

    assert [(i.record, i.problem[:8]) for i in schema.load(path)[1]] == [("line 1", "not JSON")]


def test_duplicate_ids_are_flagged(corpus: Path) -> None:
    first = books.record(corpus)
    second = books.record(corpus, query="What is the retry budget of worker 65?")

    assert schema.duplicates([first, second]) == [schema.Issue(first.id, "duplicate id")]


def test_the_same_question_worded_apart_only_in_case_and_punctuation_is_a_duplicate(
    corpus: Path,
) -> None:
    first = books.record(corpus)
    second = books.record(
        corpus, id="book-000000-09", query="how many attempts, may worker 65 make"
    )

    assert schema.duplicates([first, second]) == [
        schema.Issue("book-000000-09", "duplicate question of book-000000-01")
    ]


@pytest.mark.parametrize(
    ("name", "changes", "expected"),
    [
        ("no passage", {"relevant_passages": []}, "answerable but cites no passage"),
        ("no fact", {"expected_facts": []}, "answerable but lists no expected fact"),
        (
            "its source not relevant",
            {"relevant_documents": ["notes.md"]},
            "its own source is not among its relevant documents",
        ),
        ("an empty query", {"query": "  "}, "empty query"),
        ("an empty answer", {"expected_answer": ""}, "empty expected_answer"),
    ],
)
def test_an_answerable_record_needs_its_citations(
    corpus: Path, name: str, changes: dict, expected: str
) -> None:
    assert expected in _problems(books.record(corpus, **changes), corpus), name


def test_an_unanswerable_record_cites_nothing(corpus: Path) -> None:
    cited = books.unanswerable(corpus, relevant_documents=[books.PDF])

    assert "unanswerable but cites a document or passage" in _problems(cited, corpus)


def test_a_record_of_another_schema_version_is_flagged(corpus: Path) -> None:
    old = books.record(corpus)
    old = msgspec.structs.replace(old, meta=msgspec.structs.replace(old.meta, schema_version=0))

    assert f"schema version 0, expected {schema.SCHEMA_VERSION}" in _problems(old, corpus)


def test_a_source_changed_since_generation_fails_its_hash(corpus: Path) -> None:
    record = books.record(corpus)
    (corpus / books.PDF).write_bytes((corpus / books.PDF).read_bytes() + b"\n% edited\n")

    assert _problems(record, corpus) == [
        f"source {books.PDF} changed since this record was generated"
    ]


def test_a_source_missing_from_the_corpus_is_flagged(corpus: Path) -> None:
    record = books.record(corpus)
    (corpus / books.PDF).unlink()

    assert _problems(record, corpus) == [f"source {books.PDF} is not in {corpus}"]


@pytest.mark.parametrize(
    ("name", "passage", "expected"),
    [
        ("on its page", Passage(books.PDF, books.LINES[65], 2), None),
        ("on page 1", Passage(books.PDF, books.LINES[3], 1), None),
        (
            "run on from page 1 into page 2",
            Passage(books.PDF, f"{books.LINES[59]} {books.LINES[60]}", 1),
            None,
        ),
        ("in other whitespace and case", Passage(books.PDF, books.LINES[65].upper(), 2), None),
        (
            "on another page",
            Passage(books.PDF, books.LINES[65], 1),
            f"quote is on page 2 of {books.PDF}, not 1",
        ),
        (
            "not in the source",
            Passage(books.PDF, "Line 99 says the retry budget of worker 99 is 693 attempts.", 2),
            "quote not in book.pdf",
        ),
        ("too short to point anywhere", Passage(books.PDF, "worker 65", 2), "quote under 5 words"),
    ],
)
def test_a_quote_must_be_on_the_page_it_names(
    corpus: Path, name: str, passage: Passage, expected: str | None
) -> None:
    problems = _problems(books.record(corpus, relevant_passages=[passage]), corpus)

    if expected is None:
        assert problems == [], name
    else:
        assert any(expected in p for p in problems), (name, problems)


def test_a_source_without_pages_takes_a_quote_with_no_page(corpus: Path) -> None:
    quote = "A lease expires after thirty seconds unless the holder renews it first."
    good = books.record(
        corpus,
        source=books.MARKDOWN,
        meta=books.meta(corpus, books.MARKDOWN),
        relevant_documents=[books.MARKDOWN],
        relevant_passages=[Passage(books.MARKDOWN, quote, None, "Leases")],
    )
    paged = msgspec.structs.replace(
        good, relevant_passages=[Passage(books.MARKDOWN, quote, 3, "Leases")]
    )

    assert _problems(good, corpus) == []
    assert _problems(paged, corpus) == [
        f"{books.MARKDOWN} has no pages, but the quote names page 3"
    ]


def test_a_passage_must_cite_a_relevant_document(corpus: Path) -> None:
    quote = "Every write carries the fencing token of the lease that allowed it."
    record = books.record(
        corpus,
        relevant_passages=[
            Passage(books.PDF, books.LINES[65], 2),
            Passage(books.MARKDOWN, quote, None, "Fencing"),
        ],
    )

    assert _problems(record, corpus) == [
        f"cites {books.MARKDOWN}, not among its relevant documents"
    ]


def test_a_relationship_question_with_a_passage_per_side_passes(corpus: Path) -> None:
    assert _problems(books.relationship(corpus), corpus) == []


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"relation": None}, "a relationship question names no relation"),
        ({"relevant_passages": [Passage(books.PDF, books.LINES[65], 2, "")]}, "under two passages"),
    ],
)
def test_a_relationship_question_needs_a_relation_and_both_sides(
    corpus: Path, changes: dict, expected: str
) -> None:
    problems = _problems(books.relationship(corpus, **changes), corpus)

    assert any(expected in p for p in problems), problems


def test_a_relation_on_another_kind_of_question_is_an_issue(corpus: Path) -> None:
    record = books.record(corpus, relation=schema.Relation.SOLVES)

    assert _problems(record, corpus) == ["names a relation but is a direct question"]


def test_a_record_written_before_relations_still_loads(corpus: Path, tmp_path: Path) -> None:
    line = msgspec.to_builtins(books.record(corpus))
    del line["relation"]
    path = tmp_path / "dataset.jsonl"
    path.write_text(msgspec.json.encode(line).decode() + "\n", encoding="utf-8")

    assert schema.load(path) == ([books.record(corpus)], [])
