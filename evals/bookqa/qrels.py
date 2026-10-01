"""Graded relevance judgments: for a question, how well each passage a search returned answers it.

The gold quotes of `dataset.jsonl` name the passages a question was written from; a book often
answers it elsewhere too, and a quote match cannot see that. So every passage any mode returned is
graded on its own (`judge.py`), once, and kept here:

    2  states the answer, or one of the facts the answer needs
    1  on the topic, but states none of them
    0  not relevant

A passage is known by its document and its text, letters and digits only (`key`), so the same
passage returned by another mode or another run finds its grade. A passage no judgment covers is
unjudged: `metrics.judged` counts it as not relevant and reports how much of a top 10 was judged,
and `judge.py` grades it on its next pass.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import msgspec

from evals.bookqa import sources

HERE = Path(__file__).resolve().parent
JUDGMENTS = HERE / "judgments.jsonl"
ANSWERS = 2  # the grade of a passage that states the answer or a fact it needs
EXCERPT = 400  # characters of a passage kept beside its grade, for review


class Judgment(msgspec.Struct, forbid_unknown_fields=True):
    id: str  # the record's
    document: str
    key: str  # `key(document, text)`
    grade: int
    excerpt: str  # the passage's first `EXCERPT` characters, to review a grade by
    model: str
    prompt_version: str
    judged_at: str  # ISO 8601, UTC


type Grades = dict[tuple[str, str], int]  # (record id, passage key) -> grade


def key(document: str, text: str) -> str:
    return hashlib.sha256(f"{document}\0{sources.compact(text)}".encode()).hexdigest()[:20]


def load(path: Path = JUDGMENTS) -> list[Judgment]:
    if not path.exists():
        return []
    decoder = msgspec.json.Decoder(Judgment)
    return [decoder.decode(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def grades(judgments: Iterable[Judgment]) -> Grades:
    return {(j.id, j.key): j.grade for j in judgments}


def dump(judgments: Iterable[Judgment]) -> str:
    return "".join(msgspec.json.encode(j).decode() + "\n" for j in judgments)
