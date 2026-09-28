"""Retrieval alone, no agent: where the synthetic target ranks, per corpus size and reference
level, in each haskie instance - and what a single all-terms grep would have to sift.

An agent run costs tokens and minutes; this costs a few HTTP calls. It's the first thing to
re-run after changing haskie's search, and it says whether an agent sweep is even worth running:
if the target isn't near the top of a search result, no agent will do better than the ranking
lets it. The grep column is a proxy, not a model of what an agent does - one AND of every content
word in the reference, the most direct thing to try - so read it as "how much does the obvious
keyword query narrow things down," not as arm D's score.
"""

from __future__ import annotations

import sys
import time
import urllib.parse

from evals import synth
from evals.setup import POLL_SECONDS, call, get_or_none

LIMIT = 20
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # haskie's default reranker model
INSTANCES = {"fts": "http://127.0.0.1:8123", "hybrid": "http://127.0.0.1:8124"}
# Per-call overrides on the embedding instance - haskie's own search knobs, tried one at a time,
# without changing any collection's stored settings (arm E runs against those).
KNOBS = {
    "fts": {"mode": "fts"},
    "vector": {"mode": "vector"},
    "hybrid": {"mode": "hybrid"},
    "hybrid+rerank": {"mode": "hybrid", "reranker": "cross-encoder", "candidates": 100},
}


def rank(api: str, collection: str, query: str, doc: str) -> int | None:
    params = urllib.parse.urlencode({"q": query, "collections": collection, "limit": LIMIT})
    found = call("GET", f"/api/search/sources?{params}", api)["documents"]
    names = [row["doc"] for row in found]
    return names.index(doc) + 1 if doc in names else None


def ensure_reranker(api: str) -> None:
    """Haskie downloads a reranker model only when some collection's settings ask for one. An
    empty collection of calibration's own asks for it, so the model loads without touching the
    settings of any collection an agent arm searches."""
    name = "calibrate-reranker"
    if get_or_none(f"/api/collections/{name}", api) is None:
        call("POST", "/api/collections", api, {"name": name, "description": "loads the reranker"})
    # The model is named, not left to the default: haskie only downloads reranker models a
    # collection names explicitly (`_required` in src/haskie/indexing/models.py).
    search = {"reranker": "cross-encoder", "reranker_model": RERANKER_MODEL}
    call("PUT", f"/api/collections/{name}/settings", api, {"search": search})
    while True:
        models = call("GET", "/api/status", api)["models"]
        if all(m["state"] == "ready" for m in models):
            return
        if any(m["state"] == "error" for m in models):
            raise RuntimeError(f"model failed to load: {models}")
        time.sleep(POLL_SECONDS)


def chunk_rank(api: str, collection: str, query: str, doc: str, knobs: dict) -> int | None:
    """The target's rank among documents, ordered by each one's best chunk."""
    params = urllib.parse.urlencode({"q": query, "limit": 50, **knobs})
    hits = call("GET", f"/api/collections/{collection}/search?{params}", api)
    docs = list(dict.fromkeys(hit["doc"] for hit in hits))
    return docs.index(doc) + 1 if doc in docs[:LIMIT] else None


def grep_all(seed: int, size: int, query: str, doc: str) -> tuple[int, bool]:
    """How many documents contain every content word of `query`, and whether the target is one."""
    words = synth._words(query)
    hits = [
        path.name
        for path in synth.corpus_dir(seed, size).iterdir()
        if all(word in path.read_text().lower() for word in words)
    ]
    return len(hits), doc in hits


def main(seed: int = synth.DEFAULT_SEED) -> int:
    ordered, target_index = synth.layout(seed)
    target, doc = ordered[target_index], synth.doc_name(seed, target_index)
    print(f"target {target.name} in {doc}; rank = position in search_sources, top {LIMIT}\n")
    heads = "".join(f"{name + ' rank':>13}" for name in INSTANCES)
    print(f"{'size':>5}{'level':>6}{heads}{'grep-all hits':>15}{'has target':>12}")
    for size in synth.SIZES:
        collection = synth.collection_name(seed, size)
        for level in synth.LEVELS:
            # The bare reference, without the task's framing ("a service called ..."): framing
            # words appear in no runbook and would zero the grep column at every level.
            query = target.name if level == 0 else synth.describe(target.purpose, level == 2)
            ranks = "".join(
                f"{rank(api, collection, query, doc) or '-':>13}" for api in INSTANCES.values()
            )
            hits, has = grep_all(seed, size, query, doc)
            print(f"{size:>5}{level:>6}{ranks}{hits:>15}{'yes' if has else 'no':>12}")

    api = INSTANCES["hybrid"]
    ensure_reranker(api)
    print(f"\nsearch knobs on the embedding instance ({api}), rank among documents by best chunk\n")
    print(f"{'size':>5}{'level':>6}" + "".join(f"{name:>15}" for name in KNOBS))
    for size in synth.SIZES:
        collection = synth.collection_name(seed, size)
        for level in synth.LEVELS:
            query = target.name if level == 0 else synth.describe(target.purpose, level == 2)
            ranks = "".join(
                f"{chunk_rank(api, collection, query, doc, knobs) or '-':>15}"
                for knobs in KNOBS.values()
            )
            print(f"{size:>5}{level:>6}{ranks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*map(int, sys.argv[1:])))
