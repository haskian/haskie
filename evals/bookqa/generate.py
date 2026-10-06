"""Phase 1, run by hand: ask Claude for question/answer records from the source books.

Never part of an evaluation run. Each source is cut into segments (`sources.segments`); for the
segments chosen, Claude reads the segment's text - pypdf's page text for a PDF, pages marked - and
replies with questions, one canonical answer each, and the verbatim quotes that support it. What
it writes is a candidate, not gold: it lands in `candidates/<source>/`, and only `--accept` moves
the candidates that pass every check (`schema.validate`) into `dataset.jsonl`, the reviewed set
`run.py` reads.

A generation is keyed by everything its output depends on - the source's hash, the segment, the
seed, the model and the prompt version (`generation_key`) - and a key already on disk is never
asked again. The same inputs therefore always give the same records, ids included, and a re-run
only adds what is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import msgspec

from evals import setup
from evals.bookqa import schema, sources
from evals.bookqa.schema import Generation, Passage, QueryType, Record, Relation
from evals.run import claude_binary, subprocess_environment

HERE = Path(__file__).resolve().parent
PROMPT_VERSION = "v1"
RELATE_VERSION = "v2"
CANDIDATES = HERE / "candidates"
DATASET = HERE / "dataset.jsonl"
RELATIONS = HERE / "relations.jsonl"

type Ask = Callable[[str, str], tuple[str, str]]  # (prompt, model) -> (reply, model that answered)


class GenerationError(RuntimeError):
    pass


class Kind(msgspec.Struct, frozen=True):
    """A kind of question: the prompt that asks for it and the dataset it is accepted into. The
    facts prompt's version is the bare `PROMPT_VERSION`, as it was before relations, so its
    generation keys and the candidates already on disk stay valid."""

    name: str
    prompt: str  # the template's stem in `prompts/`, before its version

    @property
    def version(self) -> str:
        return PROMPT_VERSION if self.name == "facts" else f"{self.prompt}-{RELATE_VERSION}"

    @property
    def template(self) -> Path:
        version = PROMPT_VERSION if self.name == "facts" else RELATE_VERSION
        return HERE / "prompts" / f"{self.prompt}-{version}.md"

    @property
    def dataset(self) -> Path:
        return DATASET if self.name == "facts" else RELATIONS


FACTS = Kind("facts", "generate")
RELATIONSHIPS = Kind("relations", "relate")
KINDS = {kind.name: kind for kind in (FACTS, RELATIONSHIPS)}


class DraftPassage(msgspec.Struct):
    quote: str
    page: int | None = None
    section: str = ""


class Draft(msgspec.Struct):
    """One question as Claude replies with it, before it becomes a `Record`."""

    query: str
    query_type: QueryType
    answerable: bool
    expected_answer: str
    expected_facts: list[str] = []
    passages: list[DraftPassage] = []
    relation: Relation | None = None


def prompt(source: str, segment: sources.Segment, count: int, seed: int, kind: Kind = FACTS) -> str:
    template = kind.template.read_text(encoding="utf-8")
    if segment.first_page is None:
        note = f"part {segment.label}, no page numbers"
        page_rule = "null - this document has no pages."
    else:
        note = f"pages {segment.first_page} to {segment.last_page}"
        page_rule = (
            "the number of the `[page N]` marker above the text the quote starts in, as an integer."
        )
    return template.format(
        source=source,
        segment_note=note,
        count=count,
        page_rule=page_rule,
        seed=seed,
        text=segment.text,
    )


def generation_key(
    sha: str, segment: str, seed: int, model: str, count: int, kind: Kind = FACTS
) -> str:
    """Everything a generation's output depends on, as one hash."""
    parts = [sha, segment, str(seed), model, kind.version, str(count)]
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def candidate_path(source: str, segment: str, key: str, kind: Kind = FACTS) -> Path:
    prefix = "" if kind is FACTS else f"{kind.prompt}-"
    return CANDIDATES / Path(source).stem / f"{prefix}{segment}-{key[:8]}.jsonl"


def select(segments: list[sources.Segment], count: int, seed: int) -> list[sources.Segment]:
    """`count` of the segments, drawn by `seed` and kept in document order; all of them for 0."""
    if count <= 0 or count >= len(segments):
        return segments
    chosen = sorted(random.Random(seed).sample(range(len(segments)), count))
    return [segments[i] for i in chosen]


def parse(reply: str) -> list[Draft]:
    """The JSON array in Claude's reply, a code fence or a sentence around it aside."""
    start, end = reply.find("["), reply.rfind("]")
    if start < 0 or end < start:
        raise GenerationError(f"no JSON array in the reply: {reply[:300]!r}")
    try:
        return msgspec.json.decode(reply[start : end + 1], type=list[Draft])
    except msgspec.MsgspecError as error:
        raise GenerationError(f"reply is not a list of questions: {error}") from error


def records(
    drafts: list[Draft],
    source: str,
    sha: str,
    segment: str,
    seed: int,
    model: str,
    key: str,
    generated_at: str,
    kind: Kind = FACTS,
) -> list[Record]:
    meta = Generation(schema.SCHEMA_VERSION, sha, model, kind.version, seed, segment, generated_at)
    stem = re.sub(r"[^a-z0-9]+", "-", Path(source).stem.lower()).strip("-")
    made = []
    for number, draft in enumerate(drafts, start=1):
        passages = [Passage(source, p.quote, p.page, p.section) for p in draft.passages]
        made.append(
            Record(
                id=f"{stem}-{key[:6]}-{number:02d}",
                source=source,
                query=draft.query.strip(),
                query_type=draft.query_type,
                answerable=draft.answerable,
                expected_answer=draft.expected_answer.strip(),
                expected_facts=draft.expected_facts,
                relevant_documents=[source] if draft.answerable or passages else [],
                relevant_passages=passages,
                meta=meta,
                relation=draft.relation,
            )
        )
    return made


def ask_claude(text: str, model: str) -> tuple[str, str]:
    """One `claude -p` turn with no tools: everything it needs is in the prompt. It runs in an
    empty directory, so no project instructions or hooks reach it."""
    argv = [claude_binary(), "-p", "--output-format", "json", "--model", model]
    argv += ["--max-turns", "1", "--setting-sources", "project"]
    argv += ["--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}']
    with tempfile.TemporaryDirectory() as empty:
        completed = subprocess.run(
            argv,
            input=text,
            capture_output=True,
            text=True,
            cwd=empty,
            env=subprocess_environment(),
            check=False,
        )
    try:
        out = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise GenerationError(
            f"claude exited {completed.returncode}: {completed.stderr[-500:]}"
        ) from error
    if out.get("is_error") or completed.returncode != 0:
        raise GenerationError(f"claude failed: {str(out.get('result'))[:500]}")
    return out["result"], next(iter(out.get("modelUsage") or {}), model)


def known_questions(dataset: Path) -> set[str]:
    """The questions already in the dataset or any candidate, as `schema.question_key`s."""
    files = [dataset, *sorted(CANDIDATES.glob("*/*.jsonl"))]
    return {schema.question_key(r.query) for f in files for r in schema.load(f)[0]}


def generate(
    path: Path,
    segments: int,
    per_segment: int,
    seed: int,
    model: str,
    dataset: Path = DATASET,
    ask: Ask = ask_claude,
    dry_run: bool = False,
    kind: Kind = FACTS,
) -> list[Path]:
    """The candidate files of one source for the chosen segments, asking Claude only for those
    not already on disk. A question already in the dataset or another candidate is dropped."""
    sha = sources.sha256(path)
    files = []
    seen = known_questions(dataset)
    for segment in select(sources.segments(path), segments, seed):
        key = generation_key(sha, segment.label, seed, model, per_segment, kind)
        target = candidate_path(path.name, segment.label, key, kind)
        cached = target.exists()
        print(f"{path.name} {segment.label}: {'cached' if cached else 'to generate'} {target.name}")
        if cached or dry_run:
            files += [target] if cached else []
            continue
        reply, answered_by = ask(prompt(path.name, segment, per_segment, seed, kind), model)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        drafts = parse(reply)
        made = records(drafts, path.name, sha, segment.label, seed, answered_by, key, now, kind)
        fresh = [r for r in made if schema.question_key(r.query) not in seen]
        seen |= {schema.question_key(r.query) for r in fresh}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(schema.dump(fresh), encoding="utf-8")
        issues = schema.validate(fresh, path.parent)
        flagged = len({i.record for i in issues})
        dropped = len(made) - len(fresh)
        print(f"  {len(fresh)} candidates, {dropped} duplicates dropped, {flagged} with issues")
        files.append(target)
    return files


def accept(
    files: list[Path], dataset: Path, corpus: Path
) -> tuple[list[Record], list[schema.Issue]]:
    """Move the candidates of `files` that pass every check into the dataset: those with an issue,
    or already there by id or question, stay out. Returns the records accepted and the issues of
    those left out."""
    trusted, broken = schema.load(dataset)
    if broken:
        raise GenerationError(f"{dataset} has invalid lines: {broken}")
    candidates: list[Record] = []
    issues: list[schema.Issue] = []
    for file in files:
        found, bad = schema.load(file)
        candidates += found
        issues += bad
    ids = {r.id for r in trusted}
    questions = {schema.question_key(r.query) for r in trusted}
    fresh = [
        r for r in candidates if r.id not in ids and schema.question_key(r.query) not in questions
    ]
    issues += schema.validate(fresh, corpus)
    rejected = {i.record for i in issues}
    accepted = [r for r in fresh if r.id not in rejected]
    if accepted:
        dataset.write_text(schema.dump([*trusted, *accepted]), encoding="utf-8")
    return accepted, issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    names = [s.name for s in setup.SOURCES]
    parser.add_argument("--source", action="append", choices=names, help="default: every book")
    parser.add_argument("--segments", type=int, default=3, help="segments per source; 0 for all")
    parser.add_argument("--per-segment", type=int, default=5, help="questions per segment")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", default=os.environ.get("BOOKQA_MODEL", "sonnet"))
    parser.add_argument(
        "--kind",
        choices=list(KINDS),
        default=FACTS.name,
        help=f"facts: {DATASET.name}; relations: how two things relate, into {RELATIONS.name}",
    )
    parser.add_argument("--dataset", type=Path, help="default: the kind's dataset")
    parser.add_argument("--corpus", type=Path, default=setup.CORPUS_DIR)
    parser.add_argument("--dry-run", action="store_true", help="list what would be generated")
    parser.add_argument(
        "--accept",
        action="store_true",
        help="move the candidates that pass review into the dataset",
    )
    args = parser.parse_args(argv)
    kind = KINDS[args.kind]
    args.dataset = args.dataset or kind.dataset
    wanted = [s for s in setup.SOURCES if s.name in (args.source or names)]
    setup.fetch(tuple(wanted), args.corpus)
    files = []
    for source in wanted:
        files += generate(
            args.corpus / source.name,
            args.segments,
            args.per_segment,
            args.seed,
            args.model,
            args.dataset,
            dry_run=args.dry_run,
            kind=kind,
        )
    if not args.accept or args.dry_run:
        print(f"candidates are unreviewed; review them, then re-run with --accept ({CANDIDATES})")
        return 0
    accepted, issues = accept(files, args.dataset, args.corpus)
    for issue in issues:
        print(f"  left out {issue.record}: {issue.problem}", file=sys.stderr)
    print(f"{len(accepted)} records accepted into {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
