"""One JSONL record per question, and the checks a record passes before it is trusted.

A record holds one canonical expected answer and the passages that support it, never a ranked
list of expected results: the search under test produces its own ranking, and `metrics` scores
it against these passages. Every record carries how it was made (`Generation`), so a dataset can
say which source bytes, model and prompt each question came from.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

import msgspec

from evals.bookqa import sources

SCHEMA_VERSION = 1
MIN_QUOTE_WORDS = 5  # a shorter quote matches too many places to point at one passage


class QueryType(StrEnum):
    DIRECT = "direct"  # asks what one passage states, in words close to it
    PARAPHRASE = "paraphrase"  # asks the same in other words than the passage's
    MULTI_FACT = "multi_fact"  # needs several facts, possibly from several passages
    NEARBY_SECTIONS = "nearby_sections"  # needs a passage and one in a neighbouring section
    RELATIONSHIP = "relationship"  # asks how two things relate, each side in its own passage


class Relation(StrEnum):
    """How the two sides of a relationship question relate: X is one side, Y the other."""

    SOLVES = "solves"  # X is a solution to the problem Y
    MITIGATES = "mitigates"  # X reduces the impact of Y but does not eliminate it
    CAUSES = "causes"  # X introduces or leads to the problem Y
    TRADES_OFF = "trades_off"  # choosing X sacrifices Y; both cannot be maximized
    ALTERNATIVE_TO = "alternative_to"  # X and Y are competing approaches to the same goal
    REQUIRES = "requires"  # X only works correctly if Y is in place
    IMPLEMENTED_VIA = "implemented_via"  # X is typically implemented using the technique Y
    COMPLEMENTS = "complements"  # X is commonly used together with Y
    FAILS_WHEN = "fails_when"  # X breaks down or becomes a bad choice under condition Y
    CHALLENGES = "challenges"  # X contradicts or undermines the assumptions of Y
    GENERALIZES = "generalizes"  # X is a more general form of Y
    CORRELATES_POSITIVELY = "correlates_positively"  # as X increases, Y tends to increase
    CORRELATES_NEGATIVELY = "correlates_negatively"  # as X increases, Y tends to decrease
    ANALOGOUS = "analogous"  # X plays the same role in one domain as Y does in another


class Passage(msgspec.Struct, forbid_unknown_fields=True):
    document: str
    quote: str  # copied verbatim from the source
    page: int | None = None  # 1-based physical PDF page the quote starts on; None without pages
    section: str = ""  # the heading it sits under, as the source names it


class Generation(msgspec.Struct, forbid_unknown_fields=True):
    schema_version: int
    source_sha256: str  # of the source file the question was generated from
    model: str
    prompt_version: str
    seed: int
    segment: str  # which part of the source was read (`sources.Segment.label`)
    generated_at: str  # ISO 8601, UTC


class Record(msgspec.Struct, forbid_unknown_fields=True):
    id: str
    source: str
    query: str
    query_type: QueryType
    answerable: bool
    expected_answer: str
    expected_facts: list[str]
    relevant_documents: list[str]
    relevant_passages: list[Passage]
    meta: Generation
    relation: Relation | None = None  # a relationship question's only


class Issue(msgspec.Struct, frozen=True):
    record: str  # a record id, or "line N" where a line would not decode
    problem: str


def load(path: Path) -> tuple[list[Record], list[Issue]]:
    """Every record of a JSONL file, and an issue for each line that isn't a valid record. A
    missing file is an empty dataset."""
    if not path.exists():
        return [], []
    records, issues = [], []
    decoder = msgspec.json.Decoder(Record)
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(decoder.decode(line))
        except msgspec.ValidationError as error:
            issues.append(Issue(f"line {number}", f"not a valid record: {error}"))
        except msgspec.DecodeError as error:
            issues.append(Issue(f"line {number}", f"not JSON: {error}"))
    return records, issues


def dump(records: Iterable[Record]) -> str:
    return "".join(msgspec.json.encode(r).decode() + "\n" for r in records)


def question_key(query: str) -> str:
    """What two phrasings of the same question still share: its words, lowercased."""
    return " ".join(re.findall(r"\w+", query.lower()))


def duplicates(records: list[Record]) -> list[Issue]:
    """An issue for every record whose id, or whose question, an earlier record already has."""
    ids: set[str] = set()
    questions: dict[str, str] = {}
    issues = []
    for r in records:
        if r.id in ids:
            issues.append(Issue(r.id, "duplicate id"))
        ids.add(r.id)
        key = question_key(r.query)
        if key in questions:
            issues.append(Issue(r.id, f"duplicate question of {questions[key]}"))
        questions.setdefault(key, r.id)
    return issues


def check(record: Record) -> list[str]:
    """What is wrong with one record on its own, source aside."""
    problems = []
    if record.meta.schema_version != SCHEMA_VERSION:
        problems.append(f"schema version {record.meta.schema_version}, expected {SCHEMA_VERSION}")
    for name in ("id", "source", "query", "expected_answer"):
        if not getattr(record, name).strip():
            problems.append(f"empty {name}")
    if record.answerable:
        if not record.relevant_passages:
            problems.append("answerable but cites no passage")
        if not record.expected_facts:
            problems.append("answerable but lists no expected fact")
        if record.source not in record.relevant_documents:
            problems.append("its own source is not among its relevant documents")
    elif record.relevant_passages or record.relevant_documents:
        problems.append("unanswerable but cites a document or passage")
    if record.query_type is QueryType.RELATIONSHIP:
        if record.relation is None:
            problems.append("a relationship question names no relation")
        if not record.answerable:
            problems.append("a relationship question is unanswerable")
        if len({(p.document, p.quote) for p in record.relevant_passages}) < 2:
            problems.append("a relationship question cites under two passages, one per side")
    elif record.relation is not None:
        problems.append(f"names a relation but is a {record.query_type} question")
    for passage in record.relevant_passages:
        if passage.document not in record.relevant_documents:
            problems.append(f"cites {passage.document}, not among its relevant documents")
        if len(passage.quote.split()) < MIN_QUOTE_WORDS:
            problems.append(f"quote under {MIN_QUOTE_WORDS} words: {passage.quote!r}")
    return problems


def check_sources(record: Record, corpus: Path) -> list[str]:
    """What is wrong with a record against the source files: a source missing or changed since
    the question was generated, and a quote that is not where the record says it is."""
    path = corpus / record.source
    if not path.exists():
        return [f"source {record.source} is not in {corpus}"]
    if sources.sha256(path) != record.meta.source_sha256:
        return [f"source {record.source} changed since this record was generated"]
    problems = []
    for passage in record.relevant_passages:
        cited = corpus / passage.document
        if not cited.exists():
            problems.append(f"cited document {passage.document} is not in {corpus}")
            continue
        found = sources.quote_pages(cited, passage.quote)
        paged = sources.is_paged(cited)
        if not found:
            problems.append(f"quote not in {passage.document}: {passage.quote[:80]!r}")
        elif paged and passage.page not in found:
            problems.append(
                f"quote is on page {found[0]} of {passage.document}, not {passage.page}"
            )
        elif not paged and passage.page is not None:
            problems.append(
                f"{passage.document} has no pages, but the quote names page {passage.page}"
            )
    return problems


def validate(records: list[Record], corpus: Path) -> list[Issue]:
    """Every issue of a dataset: each record's own, against its sources, and the duplicates."""
    issues = [
        Issue(r.id, problem) for r in records for problem in [*check(r), *check_sources(r, corpus)]
    ]
    return issues + duplicates(records)
