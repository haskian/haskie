"""A tiny corpus the tests validate and score against: a real two-page PDF and a markdown book,
with records built from them.

The PDF is written by `synth.to_pdf`, sixty lines a page, so its pages are what pypdf extracts
from any text PDF: `LINES[:60]` on page 1, the rest on page 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec

from evals.bookqa import schema, sources
from evals.bookqa.schema import Generation, Passage, QueryType, Record
from evals.synth import to_pdf

PDF = "book.pdf"
MARKDOWN = "notes.md"
LINES = [f"Line {i} says the retry budget of worker {i} is {i * 7} attempts." for i in range(70)]
NOTES = (
    "# Leases\n\nA lease expires after thirty seconds unless the holder renews it first.\n\n"
    "# Fencing\n\nEvery write carries the fencing token of the lease that allowed it.\n"
)


def corpus(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PDF).write_bytes(to_pdf("\n".join(LINES)))
    (directory / MARKDOWN).write_text(NOTES, encoding="utf-8")
    return directory


def meta(corpus: Path, source: str = PDF) -> Generation:
    return Generation(
        schema_version=schema.SCHEMA_VERSION,
        source_sha256=sources.sha256(corpus / source),
        model="claude-test",
        prompt_version="v1",
        seed=1,
        segment="p001-002",
        generated_at="2026-09-29T12:00:00+00:00",
    )


def record(corpus: Path, **changes: Any) -> Record:
    """An answerable record about `LINES[65]`, on page 2 of the PDF, with `changes` applied."""
    base = Record(
        id="book-000000-01",
        source=PDF,
        query="How many attempts may worker 65 make?",
        query_type=QueryType.DIRECT,
        answerable=True,
        expected_answer="Worker 65 may make 455 attempts.",
        expected_facts=["455 attempts"],
        relevant_documents=[PDF],
        relevant_passages=[Passage(PDF, LINES[65], 2, "")],
        meta=meta(corpus),
    )
    return msgspec.structs.replace(base, **changes)


def unanswerable(corpus: Path, **changes: Any) -> Record:
    base = record(
        corpus,
        id="book-000000-02",
        query="Which worker runs the nightly compaction?",
        answerable=False,
        expected_answer="The book does not say which worker runs compaction.",
        expected_facts=[],
        relevant_documents=[],
        relevant_passages=[],
    )
    return msgspec.structs.replace(base, **changes)
