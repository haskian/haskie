"""Section search, scored: does `search_sections` point at the section that answers a question,
and do its descriptors say what that section holds?

`search_sections` is haskie's wide search: a map of which sections of which documents touch a
topic, each with a heading, a line range and descriptors - keywords meant to tell an agent what
the section is about without its text, and to put useful terms in front of it. This scores that
map for every answerable question of the reviewed dataset, against the same instance and
collection as `run.py`:

- **S@k, gold**: a section in the top k holds one of the question's gold quotes. A section's text
  is read back through haskie's `/lines`, and a quote counts as in it the way `metrics.matches`
  counts it in a passage.
- **S@k, judged**: a section in the top k holds a passage judged to state the answer
  (`qrels.py`, grade 2), so an answer the book gives outside the gold quotes counts too.
- **descriptors**: the answer's own terms - the words of its expected facts that the question
  does not already use - and the share of them in the descriptors of the first section that holds
  the answer, against the share in its heading. A descriptor earns its place by naming what the
  heading does not.

Retrieval only: no agent, no language model. Unanswerable questions are left out: a map of what a
library holds has no abstention to score.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import msgspec

from evals import setup
from evals.bookqa import metrics, qrels, sources
from evals.bookqa import run as bookqa
from evals.bookqa.metrics import Found
from evals.bookqa.schema import Record

LIMIT = 10
K = (1, 3, 10)
MAX_LINES = 400  # what `/lines` reads at a time
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how in is it its of on or that the "
    "their there these this to was what when which while who why will with you your into than "
    "then they them not only also each per any all".split()
)


class Section(msgspec.Struct):
    document: str
    header: str
    line_start: int
    line_end: int
    score: float
    descriptors: list[str] = []


class Scored(msgspec.Struct):
    id: str
    source: str
    query_type: str
    sections: list[Section]
    seconds: float
    bytes: int
    gold: int | None  # 1-based rank of the first section holding a gold quote
    judged: int | None  # 1-based rank of the first section holding a passage judged an answer
    terms: list[str]  # the answer's terms the question does not use
    in_descriptors: float | None  # share of `terms` in the first answering section's descriptors
    in_header: float | None  # the same share in its heading


def search(api: str, collection: str, query: str) -> tuple[list[Section], float, int]:
    params = urllib.parse.urlencode({"q": query, "collections": collection, "limit": LIMIT})
    started = time.perf_counter()
    with urllib.request.urlopen(f"{api}/api/search/sections?{params}", timeout=120) as response:  # noqa: S310
        body = response.read()
    seconds = time.perf_counter() - started
    found = msgspec.convert(json.loads(body)["sections"], list[Section])  # extra fields ignored
    return found, seconds, len(body)


class Texts:
    """A section's text, read back through `/lines` at most `MAX_LINES` at a time, once each."""

    def __init__(self, api: str) -> None:
        self.api = api
        self.read: dict[tuple[str, int, int], str] = {}

    def __call__(self, section: Section) -> str:
        key = (section.document, section.line_start, section.line_end)
        if key not in self.read:
            parts = []
            for start in range(section.line_start, section.line_end + 1, MAX_LINES):
                end = min(start + MAX_LINES - 1, section.line_end)
                query = urllib.parse.urlencode({"line_start": start, "line_end": end})
                document = urllib.parse.quote(section.document)
                url = f"{self.api}/api/documents/{document}/lines?{query}"
                with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                    parts.append(json.loads(response.read())["text"])
            self.read[key] = "".join(parts)
        return self.read[key]


def words(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9][a-z0-9_.-]*[a-z0-9]|[a-z0-9]", text.lower())
    } - STOPWORDS


def answer_terms(record: Record) -> list[str]:
    """The words of the answer's expected facts the question does not already use."""
    asked = words(record.query)
    return sorted({w for fact in record.expected_facts for w in words(fact)} - asked)


def share(terms: list[str], text: str) -> float:
    found = words(text)
    return sum(term in found for term in terms) / len(terms) if terms else 0.0


def first(ranked: list[bool]) -> int | None:
    return next((rank for rank, hit in enumerate(ranked, start=1) if hit), None)


def score(
    record: Record, sections: list[Section], text: Texts, answers: list[qrels.Judgment]
) -> tuple[int | None, int | None, int | None]:
    """The rank of the first section holding a gold quote, of the first holding a passage judged
    an answer, and the index of the first holding either (the section the descriptors are read
    from)."""
    gold, judged = [], []
    for section in sections:
        body = text(section)
        found = Found(document=section.document, text=body, score=section.score)
        gold.append(any(metrics.matches(found, p) for p in record.relevant_passages))
        here = sources.compact(body)
        judged.append(
            any(
                j.document == section.document and sources.compact(j.excerpt[:200]) in here
                for j in answers
            )
        )
    either = first([g or j for g, j in zip(gold, judged, strict=True)])
    return first(gold), first(judged), either


def evaluate(records: list[Record], api: str, collection: str) -> list[Scored]:
    text = Texts(api)
    judgments = qrels.load(qrels.JUDGMENTS)
    scored = []
    for record in records:
        if not record.answerable:
            continue
        sections, seconds, size = search(api, collection, record.query)
        answers = [j for j in judgments if j.id == record.id and j.grade == qrels.ANSWERS]
        gold, judged, either = score(record, sections, text, answers)
        terms = answer_terms(record)
        hit = sections[either - 1] if either else None
        scored.append(
            Scored(
                id=record.id,
                source=record.source,
                query_type=record.query_type.value,
                sections=sections,
                seconds=seconds,
                bytes=size,
                gold=gold,
                judged=judged,
                terms=terms,
                in_descriptors=share(terms, " ".join(hit.descriptors)) if hit and terms else None,
                in_header=share(terms, hit.header) if hit and terms else None,
            )
        )
    return scored


def _mean(values: list[float]) -> str:
    return f"{sum(values) / len(values):.2f}" if values else "-"


def _row(group: list[Scored]) -> str:
    def hits(rank_of: str, k: int) -> str:
        return _mean([float((getattr(s, rank_of) or 99) <= k) for s in group])

    gold = [hits("gold", k) for k in K]
    judged = [hits("judged", k) for k in K]
    rr = _mean([1 / s.judged if s.judged else 0.0 for s in group])
    described = [s.in_descriptors for s in group if s.in_descriptors is not None]
    headed = [s.in_header for s in group if s.in_header is not None]
    kb = _mean([s.bytes / 1000 for s in group])
    ms = statistics.median(s.seconds * 1000 for s in group)
    cells = [str(len(group)), *gold, *judged, rr, _mean(described), _mean(headed), kb, f"{ms:.0f}"]
    return " | ".join(cells)


def render(scored: list[Scored]) -> str:
    columns = ["n", *(f"S@{k} gold" for k in K), *(f"S@{k} judged" for k in K), "MRR judged"]
    columns += ["terms in descriptors", "terms in heading", "kB", "ms p50"]
    lines = [
        "# Section search",
        "S@k: a section in the top k holds the answer - by its gold quote, or a passage judged to "
        "state it. Terms: the share of the answer's own words (not the question's) in the "
        "descriptors, and in the heading, of the first section that holds it.",
    ]
    for title, field in (("All", None), ("By source", "source"), ("By query type", "query_type")):
        lines += ["", f"## {title}", "", f"| group | {' | '.join(columns)} |"]
        lines.append("|---|" + "---|" * len(columns))
        keys = sorted({getattr(s, field) for s in scored}) if field else ["all"]
        for key in keys:
            group = [s for s in scored if field is None or getattr(s, field) == key]
            lines.append(f"| {key} | {_row(group)} |")
    junk = [d for s in scored for sec in s.sections[:3] for d in sec.descriptors if not words(d)]
    lines += ["", f"Descriptors with no word in them, in top-3 sections: {len(junk)}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=bookqa.DATASET)
    parser.add_argument(
        "--api", default=os.environ.get("HASKIE_EVAL_EMBED_URL", "http://127.0.0.1:8124")
    )
    parser.add_argument("--collection", default=bookqa.COLLECTION)
    parser.add_argument(
        "--profile", default=os.environ.get("HASKIE_EVAL_EMBED_PROFILE", "granite-small-english")
    )
    parser.add_argument("--corpus", type=Path, default=setup.CORPUS_DIR)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--out", type=Path, default=bookqa.REPORTS / f"sections-{stamp}")
    args = parser.parse_args(argv)
    records = bookqa.ready(args.dataset, args.corpus, args.api, args.profile, args.collection)
    if records is None:
        return 1
    scored = evaluate(records, args.api, args.collection)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sections.jsonl").write_text(
        "".join(msgspec.json.encode(s).decode() + "\n" for s in scored), encoding="utf-8"
    )
    text = render(scored)
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
