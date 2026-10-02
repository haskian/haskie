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
from evals.bookqa import metrics, qrels, report, schema
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
    # the instance's own reranker model, the one a first run picks
    "hybrid+rerank": {"mode": "hybrid", "reranker": "cross-encoder"},
}
RERANK = "hybrid+rerank:"  # the mode of one named reranker model: `RERANK` + its name
MODEL_WAIT = 1800.0  # seconds a reranker may take to download and load on first use


def rerank_mode(model: str) -> tuple[str, dict]:
    """The mode that reranks hybrid search with `model`, one of `known_rerankers`."""
    return f"{RERANK}{model}", {**MODES["hybrid+rerank"], "reranker_model": model}


def known_rerankers(api: str) -> list[str]:
    """The reranker models the instance's catalogue offers."""
    return setup.call("GET", "/api/options", api)["reranker_models"]


def choose_modes(names: list[str], rerankers: list[str]) -> dict[str, dict]:
    """The modes `names` picks from `MODES`, then one per reranker model, in that order."""
    return {name: MODES[name] for name in names} | dict(map(rerank_mode, rerankers))


def await_reranker(api: str, model: str, limit: float = MODEL_WAIT) -> None:
    """Wait until `model` is loaded: a search with it fails "not loaded yet" until then. Setting
    the override is what starts its download (`PUT .../overrides`)."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        states = {
            m["name"]: m
            for m in setup.call("GET", "/api/status", api)["models"]
            if m["kind"] == "reranker"
        }
        state = states.get(model, {}).get("state")
        if state == "ready":
            return
        if state == "error":
            raise RuntimeError(f"reranker {model} failed to load: {states[model]['error']}")
        time.sleep(setup.POLL_SECONDS)
    raise RuntimeError(f"reranker {model} is still not loaded after {limit:.0f}s")


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


def set_mode(api: str, collection: str, knobs: dict) -> None:
    """Search `collection` with `knobs` from now on. The overrides are replaced whole, so a
    reranker model one mode names does not carry into the next."""
    setup.call("PUT", f"/api/collections/{collection}/overrides", api, {"search": knobs})
    if "reranker_model" in knobs:
        await_reranker(api, knobs["reranker_model"])


def evaluate(
    records: list[Record], api: str, collection: str, modes: dict[str, dict]
) -> list[Outcome]:
    """Every record searched once per mode: `modes` maps a mode's name to its search overrides."""
    outcomes = []
    for mode, knobs in modes.items():
        set_mode(api, collection, knobs)
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
    grades = qrels.grades(qrels.load())
    (out / "report.md").write_text(report.render(outcomes, grades), encoding="utf-8")
    return out / "report.md"


def ready(
    dataset: Path, corpus: Path, api: str, profile: str, collection: str
) -> list[Record] | None:
    """The reviewed records, with every source they cite indexed in `collection`; None, and why
    on stderr, when the dataset is empty or fails review, or a source does not index."""
    records, broken = schema.load(dataset)
    if not records and not broken:
        missing = f"{dataset} has no reviewed records: generate, review and --accept them"
        print(f"{missing} (mise run eval:bookqa:generate)", file=sys.stderr)
        return None
    names = documents(records)
    setup.fetch(tuple(s for s in setup.SOURCES if s.name in names), corpus)
    issues = broken + schema.validate(records, corpus)
    if issues:
        for issue in issues:
            print(f"{issue.record}: {issue.problem}", file=sys.stderr)
        print("the dataset does not pass review (mise run eval:bookqa:review)", file=sys.stderr)
        return None
    setup.ensure_profile(profile, api)
    if not setup.load([corpus / name for name in names], collection, DESCRIPTION, api):
        print(f"not every source is indexed in {collection}", file=sys.stderr)
        return None
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--api", default=os.environ.get("HASKIE_EVAL_EMBED_URL", "http://127.0.0.1:8124")
    )
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument(
        "--profile",
        default=os.environ.get("HASKIE_EVAL_EMBED_PROFILE", "granite-small-english"),
        help="the embedding profile a fresh instance's first run picks; vector modes need one",
    )
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument(
        "--rerankers",
        nargs="+",
        default=[],
        metavar="MODEL",
        help=f"also rerank hybrid search with each of these models, as mode '{RERANK}MODEL'",
    )
    parser.add_argument("--corpus", type=Path, default=setup.CORPUS_DIR)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--out", type=Path, default=REPORTS / stamp)
    args = parser.parse_args(argv)

    offered = known_rerankers(args.api) if args.rerankers else []
    unknown = sorted(set(args.rerankers) - set(offered))
    if unknown:
        listing = "\n  ".join(offered)
        print(f"unknown reranker {', '.join(unknown)}; offered:\n  {listing}", file=sys.stderr)
        return 1
    records = ready(args.dataset, args.corpus, args.api, args.profile, args.collection)
    if records is None:
        return 1
    modes = choose_modes(args.modes, args.rerankers)
    path = write(evaluate(records, args.api, args.collection, modes), args.out)
    print(path.read_text(encoding="utf-8"))
    print(f"written to {path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
