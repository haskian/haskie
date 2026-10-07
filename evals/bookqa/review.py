"""Check a dataset without searching or generating anything: every line a valid record, no
duplicate id or question, the citations an answerable record needs, the source files' hashes
unchanged, and every quote on the page it claims. Exits 1 on any issue.

Reads `dataset.jsonl` by default; `--candidates` checks the unreviewed candidate files instead.
`--collection` checks an open-corpora collection's dataset, or its books' candidates.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from evals import setup
from evals.bookqa import generate, schema
from evals.corpora import corpus as corpora


def review(files: list[Path], corpus: Path) -> tuple[list[schema.Record], list[schema.Issue]]:
    records: list[schema.Record] = []
    issues: list[schema.Issue] = []
    for file in files:
        found, broken = schema.load(file)
        records += found
        issues += [schema.Issue(f"{file.name} {i.record}", i.problem) for i in broken]
    return records, issues + schema.validate(records, corpus)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", help="an open-corpora collection: its dataset and books")
    parser.add_argument("--dataset", type=Path, help="default: dataset.jsonl, or the collection's")
    parser.add_argument("--candidates", action="store_true", help="check the candidates instead")
    parser.add_argument("--corpus", type=Path, help="default: where the books are downloaded")
    args = parser.parse_args(argv)
    if args.collection:
        collection = corpora.find(args.collection)
        corpus = args.corpus or corpora.DIRECTORY / collection.name
        dataset = args.dataset or corpora.questions(collection.name)
        books = corpora.stems(collection)
    else:
        corpus = args.corpus or setup.CORPUS_DIR
        dataset = args.dataset or generate.DATASET
        books = {Path(s.name).stem for s in setup.SOURCES}
    candidates = [f for f in generate.CANDIDATES.glob("*/*.jsonl") if f.parent.name in books]
    files = sorted(candidates) if args.candidates else [dataset]
    records, issues = review(files, corpus)
    for issue in issues:
        print(f"{issue.record}: {issue.problem}", file=sys.stderr)
    answerable = sum(r.answerable for r in records)
    kinds = Counter(r.query_type.value for r in records)
    print(
        f"{len(records)} records ({answerable} answerable, {len(records) - answerable} "
        f"unanswerable; {', '.join(f'{k} {n}' for k, n in sorted(kinds.items()))}), "
        f"{len(issues)} issues"
    )
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
