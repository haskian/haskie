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
import urllib.parse

from evals import synth
from evals.setup import call, load

LIMIT = 20
INSTANCES = {"fts": "http://127.0.0.1:8123", "hybrid": "http://127.0.0.1:8124"}
# Haskie's own search knobs, tried one at a time on the embedding instance. Since 0.16 a search
# takes no per-call overrides, so they are set on a collection of calibration's own holding the
# same documents - never on a collection an agent arm searches.
KNOBS = {
    "fts": {"mode": "fts", "reranker": "none"},
    "vector": {"mode": "vector", "reranker": "none"},
    "hybrid": {"mode": "hybrid", "reranker": "none"},
    "hybrid+rerank": {"mode": "hybrid", "reranker": "cross-encoder"},
}


def rank(api: str, collection: str, query: str, doc: str) -> int | None:
    """The target's position among the documents `search_sources` returns, or None past `LIMIT`."""
    params = urllib.parse.urlencode({"q": query, "collections": collection, "limit": LIMIT})
    found = call("GET", f"/api/search/sources?{params}", api)["documents"]
    names = [row["document"] for row in found]
    return names.index(doc) + 1 if doc in names else None


def knob_collection(api: str, seed: int, size: int) -> str:
    name = f"calibrate-s{seed}-n{size}"
    files = sorted(synth.corpus_dir(seed, size).iterdir())
    load(files, name, "Calibration: search knobs are switched on this copy.", api)
    return name


def knob_rank(api: str, collection: str, knobs: dict, query: str, doc: str) -> int | None:
    call("PUT", f"/api/collections/{collection}/overrides", api, {"search": knobs})
    return rank(api, collection, query, doc)


def grep_all(seed: int, size: int, query: str, doc: str) -> tuple[int, bool]:
    """How many documents contain every content word of `query`, and whether the target is one."""
    words = synth._words(query)
    hits = [
        path.name
        for path in synth.corpus_dir(seed, size).iterdir()
        if all(word in path.read_text().lower() for word in words)
    ]
    return len(hits), doc in hits


def reference(target: synth.Service, level: int) -> str:
    # The bare reference, without the task's framing ("a service called ..."): framing words
    # appear in no runbook and would zero the grep column at every level.
    return target.name if level == 0 else synth.describe(target.purpose, level == 2)


def main(seed: int = synth.DEFAULT_SEED) -> int:
    ordered, target_index = synth.layout(seed)
    target, doc = ordered[target_index], synth.doc_name(seed, target_index)
    print(f"target {target.name} in {doc}; rank = position in search_sources, top {LIMIT}\n")
    heads = "".join(f"{name + ' rank':>13}" for name in INSTANCES)
    print(f"{'size':>5}{'level':>6}{heads}{'grep-all hits':>15}{'has target':>12}")
    for size in synth.SIZES:
        collection = synth.collection_name(seed, size)
        for level in synth.LEVELS:
            query = reference(target, level)
            ranks = "".join(
                f"{rank(api, collection, query, doc) or '-':>13}" for api in INSTANCES.values()
            )
            hits, has = grep_all(seed, size, query, doc)
            print(f"{size:>5}{level:>6}{ranks}{hits:>15}{'yes' if has else 'no':>12}")

    api = INSTANCES["hybrid"]
    print(f"\nsearch knobs on the embedding instance ({api})\n")
    print(f"{'size':>5}{'level':>6}" + "".join(f"{name:>15}" for name in KNOBS))
    for size in synth.SIZES:
        collection = knob_collection(api, seed, size)
        for level in synth.LEVELS:
            query = reference(target, level)
            ranks = "".join(
                f"{knob_rank(api, collection, knobs, query, doc) or '-':>15}"
                for knobs in KNOBS.values()
            )
            print(f"{size:>5}{level:>6}{ranks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*map(int, sys.argv[1:])))
