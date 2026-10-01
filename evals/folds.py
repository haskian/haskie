"""Whether haskie's answer leads with the right document or folds it under a decoy, and how much
of the answer is `also_in`. Retrieval only: a few HTTP calls, no agent, no model.

Near-duplicate folding (`search/collapse.py`) files a result under a better-ranked one it repeats,
as an `also_in` place. On the synthetic runbooks it misfires two ways. Templated sections that
differ only in their values (a port, a team) fold as `equivalent`, so one passage carries every
other runbook's - most of an answer's size. And the target itself folds under a decoy: asked for
the service that runs "every night", the answer leads with one that runs "every Monday morning"
and files the target inside it as `contained`. An agent reads the decoy as the result.

For each synthetic collection and reference level this asks `search_excerpts` the reference the
tasks use, on each eval instance, and reports where the target's document ranks among the
excerpts, whether it sits under another document's `also_in` instead (and as what), and the
share of the response that is `also_in`. With `--check` it exits 1 when any target is folded
under another document: the check a fix to the folding has to pass. Without it, it only reports.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from collections.abc import Iterator

import msgspec

from evals import synth
from evals.calibrate import INSTANCES, reference

LIMIT = 10


class Fold(msgspec.Struct):
    rank: int | None  # 1-based, among the excerpts' documents; None when it leads none
    under: str | None  # the document the target is folded under, when it is
    relation: str | None  # how: `contained`, `equivalent` or `duplicate`
    chars: int  # the response body
    also_in: int  # characters of it inside `also_in`


def places(references: list[dict]) -> Iterator[dict]:
    """Every place of an `also_in` tree, at every depth."""
    for place in references:
        yield place
        yield from places(place.get("also_in", []))


def check(api: str, collection: str, question: str, doc: str) -> Fold:
    params = urllib.parse.urlencode({"q": question, "collections": collection, "limit": LIMIT})
    with urllib.request.urlopen(f"{api}/api/search/excerpts?{params}", timeout=120) as response:  # noqa: S310
        body = response.read()
    excerpts = json.loads(body)["excerpts"]
    leading = [e["document"] for e in excerpts]
    rank = leading.index(doc) + 1 if doc in leading else None
    under = relation = None
    folded = 0
    for excerpt in excerpts:
        for span in excerpt.get("spans", []):
            # as compact as haskie sends it, so the share is of the same bytes
            also_in = span.get("also_in", [])
            folded += len(json.dumps(also_in, separators=(",", ":"), ensure_ascii=False).encode())
            if under is None and excerpt["document"] != doc:
                hit = next(
                    (p for p in places(span.get("also_in", [])) if p["document"] == doc), None
                )
                if hit is not None:
                    under, relation = excerpt["document"], hit.get("relation")
    return Fold(rank, under, relation, len(body), folded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", nargs="?", type=int, default=synth.DEFAULT_SEED)
    parser.add_argument("--check", action="store_true", help="exit 1 if any target is misfolded")
    args = parser.parse_args(argv)
    seed = args.seed
    ordered, target_index = synth.layout(seed)
    target, doc = ordered[target_index], synth.doc_name(seed, target_index)
    print(f"target {target.name} in {doc}; rank among the top {LIMIT} excerpts' documents\n")
    head = f"{'instance':9}{'size':>6}{'level':>6}{'rank':>6}  {'folded under':28}"
    print(f"{head}{'kB':>7}{'also_in':>9}")
    misfolded = 0
    for name, api in INSTANCES.items():
        for size in synth.SIZES:
            collection = synth.collection_name(seed, size)
            for level in synth.LEVELS:
                fold = check(api, collection, reference(target, level), doc)
                where = f"{fold.under} ({fold.relation})" if fold.under else "-"
                share = f"{fold.also_in / fold.chars:.0%}" if fold.chars else "-"
                print(
                    f"{name:9}{size:>6}{level:>6}{fold.rank or '-':>6}  {where:28}"
                    f"{fold.chars / 1000:>7.1f}{share:>9}"
                )
                misfolded += fold.under is not None
    verdict = "the folding bug is present" if misfolded else "no target is folded under a decoy"
    print(f"\n{misfolded} searches fold the target under another document: {verdict}")
    return 1 if misfolded and args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
