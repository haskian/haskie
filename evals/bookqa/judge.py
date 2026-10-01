"""Grade the passages searches returned, once each, with Claude: the judgments `qrels.py` keeps.

An explicit phase, like `generate.py`, and never part of `run.py`. It reads the outcomes of one or
more runs, pools per question every passage any mode returned in its top 10 (TREC-style pooling),
and asks Claude to grade the ones no judgment covers yet - one call per question, the question
alone, never the expected answer, so a grade does not lean toward the gold quote. A later run that
returns new passages is judged by running this again on its outcomes: only the new ones are asked.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import msgspec

from evals.bookqa import generate, qrels, schema
from evals.bookqa.generate import Ask
from evals.bookqa.metrics import Found, K, Outcome
from evals.bookqa.qrels import Judgment
from evals.bookqa.schema import Record

HERE = Path(__file__).resolve().parent
PROMPT_VERSION = "v1"


class JudgeError(RuntimeError):
    pass


class Grade(msgspec.Struct):
    n: int
    grade: int


def pools(outcomes: list[Outcome]) -> dict[str, dict[str, Found]]:
    """Per question, every distinct passage any mode returned in its top 10, by `qrels.key`, in
    the order they were first returned."""
    pooled: dict[str, dict[str, Found]] = {}
    for outcome in outcomes:
        into = pooled.setdefault(outcome.id, {})
        for found in outcome.results[: max(K)]:
            into.setdefault(qrels.key(found.document, found.text), found)
    return pooled


def prompt(record: Record, passages: list[Found]) -> str:
    template = (HERE / "prompts" / f"judge-{PROMPT_VERSION}.md").read_text(encoding="utf-8")
    listed = "\n\n".join(
        f"[{n}] {p.document}"
        + (f", p.{p.page_start}" if p.page_start else "")
        + (f", {p.header}" if p.header else "")
        + f"\n{p.text}"
        for n, p in enumerate(passages, start=1)
    )
    return template.format(source=record.source, query=record.query, passages=listed)


def parse(reply: str, count: int) -> list[int]:
    """The grade of each of `count` passages, in order."""
    start, end = reply.find("["), reply.rfind("]")
    if start < 0 or end < start:
        raise JudgeError(f"no JSON array in the reply: {reply[:300]!r}")
    try:
        grades = msgspec.json.decode(reply[start : end + 1], type=list[Grade])
    except msgspec.MsgspecError as error:
        raise JudgeError(f"reply is not a list of grades: {error}") from error
    by_n = {g.n: g.grade for g in grades}
    if sorted(by_n) != list(range(1, count + 1)) or not set(by_n.values()) <= {0, 1, 2}:
        raise JudgeError(f"expected grades 0-2 for passages 1-{count}, got {sorted(by_n.items())}")
    return [by_n[n] for n in range(1, count + 1)]


def judge(
    records: list[Record],
    outcomes: list[Outcome],
    known: list[Judgment],
    model: str,
    ask: Ask = generate.ask_claude,
    dry_run: bool = False,
) -> list[Judgment]:
    """The judgments of every pooled passage `known` lacks, one question at a time."""
    graded = {(j.id, j.key) for j in known}
    by_id = {r.id: r for r in records}
    made: list[Judgment] = []
    for rid, pooled in pools(outcomes).items():
        fresh = [(k, p) for k, p in pooled.items() if (rid, k) not in graded]
        if not fresh or rid not in by_id:
            continue
        print(f"{rid}: {len(fresh)} of {len(pooled)} passages to grade", flush=True)
        if dry_run:
            continue
        reply, answered_by = ask(prompt(by_id[rid], [p for _, p in fresh]), model)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        for (k, p), grade in zip(fresh, parse(reply, len(fresh)), strict=True):
            excerpt = p.text[: qrels.EXCERPT]
            made.append(
                Judgment(rid, p.document, k, grade, excerpt, answered_by, PROMPT_VERSION, now)
            )
    return made


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outcomes", nargs="+", type=Path, help="outcomes.jsonl of one or more runs")
    parser.add_argument("--dataset", type=Path, default=generate.DATASET)
    parser.add_argument("--judgments", type=Path, default=qrels.JUDGMENTS)
    parser.add_argument("--model", default=os.environ.get("BOOKQA_JUDGE_MODEL", "opus"))
    parser.add_argument("--dry-run", action="store_true", help="count what would be graded")
    args = parser.parse_args(argv)
    records, broken = schema.load(args.dataset)
    if broken:
        print(f"{args.dataset} has invalid lines (mise run eval:bookqa:review)", file=sys.stderr)
        return 1
    decoder = msgspec.json.Decoder(Outcome)
    outcomes = [
        decoder.decode(line)
        for path in args.outcomes
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    known = qrels.load(args.judgments)
    made: list[Judgment] = []
    try:
        for record in records:  # one question per call, saved as it goes: a failure keeps the rest
            new = judge([record], outcomes, [*known, *made], args.model, dry_run=args.dry_run)
            if new:
                made += new
                args.judgments.write_text(qrels.dump([*known, *made]), encoding="utf-8")
    except (JudgeError, generate.GenerationError) as error:
        print(f"stopped: {error}", file=sys.stderr)
        return 1
    finally:
        total = len(known) + len(made)
        print(f"{len(made)} passages graded, {total} judgments in {args.judgments}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
