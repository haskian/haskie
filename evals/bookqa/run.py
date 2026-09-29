"""Phase 2: search haskie with every question of the reviewed dataset and score the ranking.

Retrieval only. No agent, no language model, no regeneration: the dataset is read as it is
(`dataset.jsonl`, frozen by `generate.py --accept`), each question goes to haskie's HTTP search
once per mode, and the ranked passages are scored against the gold quotes (`metrics.py`).

The search is `/api/search/explore` at passage granularity, top `LIMIT`: haskie's own ranking,
one passage of one section per result. A mode is a collection's search override, set on this
suite's own collection (`COLLECTION`), so switching modes never touches a collection another
eval searches. Vector modes need an instance with an embedding profile: by default the agent
eval's embedding instance (`HASKIE_EVAL_EMBED_URL`).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import msgspec

from evals import setup
from evals.bookqa import metrics, report, schema
from evals.bookqa.metrics import Found, Outcome
from evals.bookqa.schema import Record

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset.jsonl"
REPORTS = HERE / "reports"
COLLECTION = "bookqa-books"
DESCRIPTION = "The source books of the book query retrieval eval (evals/bookqa)."
LIMIT = 10
MODES = {
    "fts": {"mode": "fts", "reranker": "none"},
    "vector": {"mode": "vector", "reranker": "none"},
    "hybrid": {"mode": "hybrid", "reranker": "none"},
    "hybrid+rerank": {"mode": "hybrid", "reranker": "cross-encoder"},
}


def search(api: str, collection: str, query: str) -> tuple[list[Found], float, int]:
    """The ranked passages for `query`, the seconds the request took, and the body's size."""
    params = urllib.parse.urlencode(
        {"q": query, "granularity": "passage", "collections": collection, "limit": LIMIT}
    )
    url = f"{api.rstrip('/')}/api/search/explore?{params}"
    started = time.perf_counter()
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        body = response.read()
    seconds = time.perf_counter() - started
    return msgspec.json.decode(body, type=list[Found]), seconds, len(body)


def set_mode(api: str, collection: str, mode: str) -> None:
    setup.call("PUT", f"/api/collections/{collection}/overrides", api, {"search": MODES[mode]})


def evaluate(records: list[Record], api: str, collection: str, modes: list[str]) -> list[Outcome]:
    outcomes = []
    for mode in modes:
        set_mode(api, collection, mode)
        for record in records:
            found, seconds, size = search(api, collection, record.query)
            scores = metrics.score(record, found) if record.answerable else None
            outcomes.append(
                Outcome(
                    id=record.id,
                    source=record.source,
                    query_type=record.query_type.value,
                    answerable=record.answerable,
                    mode=mode,
                    seconds=seconds,
                    bytes=size,
                    results=found,
                    scores=scores,
                    abstained=metrics.abstained(found),
                )
            )
        print(f"{mode}: {len(records)} questions searched", flush=True)
    return outcomes


def documents(records: list[Record]) -> list[str]:
    """Every document the dataset's questions come from or cite, in order."""
    names = [r.source for r in records]
    names += [p.document for r in records for p in r.relevant_passages]
    return list(dict.fromkeys(names))


def write(outcomes: list[Outcome], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    lines = "".join(msgspec.json.encode(o).decode() + "\n" for o in outcomes)
    (out / "outcomes.jsonl").write_text(lines, encoding="utf-8")
    (out / "report.md").write_text(report.render(outcomes), encoding="utf-8")
    return out / "report.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--api", default=os.environ.get("HASKIE_EVAL_EMBED_URL", "http://127.0.0.1:8124")
    )
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument(
        "--profile",
        default=os.environ.get("HASKIE_EVAL_EMBED_PROFILE", "compact"),
        help="the embedding profile a fresh instance's first run picks; vector modes need one",
    )
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--corpus", type=Path, default=setup.CORPUS_DIR)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--out", type=Path, default=REPORTS / stamp)
    args = parser.parse_args(argv)

    records, broken = schema.load(args.dataset)
    if not records and not broken:
        missing = f"{args.dataset} has no reviewed records: generate, review and --accept them"
        print(f"{missing} (mise run eval:bookqa:generate)", file=sys.stderr)
        return 1
    names = documents(records)
    setup.fetch(tuple(s for s in setup.SOURCES if s.name in names), args.corpus)
    issues = broken + schema.validate(records, args.corpus)
    if issues:
        for issue in issues:
            print(f"{issue.record}: {issue.problem}", file=sys.stderr)
        print("the dataset does not pass review (mise run eval:bookqa:review)", file=sys.stderr)
        return 1
    files = [args.corpus / name for name in names]
    setup.ensure_profile(args.profile, args.api)
    if not setup.load(files, args.collection, DESCRIPTION, args.api):
        print(f"not every source is indexed in {args.collection}", file=sys.stderr)
        return 1
    path = write(evaluate(records, args.api, args.collection, args.modes), args.out)
    print(path.read_text(encoding="utf-8"))
    print(f"written to {path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
