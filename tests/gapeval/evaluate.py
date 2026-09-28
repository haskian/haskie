"""How well each gap signal tells an answered question from an unanswered one, on labelled shelves.

Run it as `mise run evaluate-gaps` (see `--help`). Not part of `mise run test`: it downloads a book
and runs real models, for minutes.

Two shelves, both free for any use:

- `rust-book`: "The Rust Programming Language" (Apache-2.0 or MIT), fetched at the commit pinned in
  `rust_book.json` into a cache, never committed (see `NOTICE`). Code listings are `{{#include}}`
  lines in its source, so the prose is what gets indexed.
- `haskie-docs`: four of this repository's own docs.

Each shelf is chunked with haskie's own chunker at the default settings and embedded with the
profile asked for. A question's ranking is its `CANDIDATES` nearest chunks by cosine, the pool a
vector search reads, and its score profile is taken with `log.profile`, as the search log takes it.
The reranker scores the same pool. Each feature of `gaps.FEATURES` is then scored by AUROC:
the chance that a random answered question scores above a random unanswered one (0.5 is a coin).
The topic pairs of `topics.json` give the `same_topic` bar the same way.
"""

import argparse
import asyncio
import io
import json
import math
import os
import re
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "haskie" / "gapeval"
CANDIDATES = 50  # the pool a question's ranking reads and the reranker rescores
BATCH = 64


def _book(commit: str) -> str:
    """The Rust book's chapters at `commit`, in `SUMMARY.md` order, fetched once into the cache."""
    root = CACHE / f"rust-book-{commit}"
    if not root.exists():
        url = f"https://codeload.github.com/rust-lang/book/tar.gz/{commit}"
        print(f"fetching {url}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=120) as response:
            archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
        wanted = [m for m in archive.getmembers() if "/src/" in m.name and m.name.endswith(".md")]
        root.mkdir(parents=True)
        for member in wanted:
            source = archive.extractfile(member)
            if source is not None:
                (root / Path(member.name).name).write_bytes(source.read())
    order = re.findall(r"\]\(([^)]+\.md)\)", (root / "SUMMARY.md").read_text())
    chapters = [(root / name).read_text() for name in order if (root / name).exists()]
    # an `{{#include ...}}` line is a code listing the book pulls in at build time
    return "\n\n".join(
        "\n".join(line for line in text.splitlines() if not line.startswith("{{#"))
        for text in chapters
    )


def _shelf(name: str) -> tuple[str, dict[str, Any]]:
    if name == "rust-book":
        labels = json.loads((HERE / "rust_book.json").read_text())
        return _book(labels["commit"]), labels
    labels = json.loads((HERE / "haskie_docs.json").read_text())
    return "\n\n".join((REPO / path).read_text() for path in labels["files"]), labels


def auroc(answered: list[float], unanswered: list[float]) -> float:
    """P(an answered score > an unanswered one), ties counted half: the Mann-Whitney U, scaled."""
    if not answered or not unanswered:
        return math.nan
    a, u = np.asarray(answered), np.asarray(unanswered)
    wins = (a[:, None] > u[None, :]).sum() + 0.5 * (a[:, None] == u[None, :]).sum()
    return float(wins / (len(a) * len(u)))


def _sigmoid(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))


async def _measure(shelf: str, profile: str, reranker: str | None) -> dict[str, Any]:
    from haskie.catalogue import catalogue
    from haskie.indexing import chunk, embed
    from haskie.search import gaps, log
    from haskie.settings import Accelerator, ChunkSettings, UserSettings

    text, labels = _shelf(shelf)
    model = await catalogue.embedding_model(UserSettings(embedding=profile))
    assert model is not None, "a profile with a model"
    chunks = chunk.split(text, ChunkSettings())
    texts = [chunk.framed(one.frame, one.text) for one in chunks]
    print(f"{shelf}: {len(texts)} chunks, embedding with {profile}", file=sys.stderr)
    vectors = np.asarray(
        [
            v
            for at in range(0, len(texts), BATCH)
            for v in embed.embed_texts(model, texts[at : at + BATCH])
        ]
    )
    units = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    groups = {
        "answered": labels["answered"],
        "unanswered": labels["unanswered_near"] + labels["unanswered_far"],
    }
    measured: dict[str, list[log.LoggedQuestion]] = {}
    for group, questions in groups.items():
        measured[group] = []
        for question in questions:
            query = np.asarray(embed.embed_query(model, question))
            pool = np.argsort(-(units @ (query / np.linalg.norm(query))))[:CANDIDATES]
            found = log.profile(query.tolist(), vectors[pool].tolist())
            scores: list[float] = []
            if reranker is not None:
                logits = embed.rerank_scores(
                    reranker, Accelerator.CPU, question, [texts[i] for i in pool]
                )
                scores = sorted((_sigmoid(one) for one in logits), reverse=True)[: log.PROFILE]
            measured[group].append(
                log.LoggedQuestion(
                    question,
                    similarities=found.similarities,
                    rerank_scores=scores,
                    coherence=found.coherence,
                )
            )

    features: dict[str, Any] = {}
    for name, feature in gaps.FEATURES.items():
        a = [v for q in measured["answered"] if (v := feature(q)) is not None]
        u = [v for q in measured["unanswered"] if (v := feature(q)) is not None]
        if not a or not u:
            continue
        features[name] = {
            "auroc": auroc(a, u),
            "answered_min": min(a),
            "unanswered_max": max(u),
        }
    bar = model.weak_match
    floor = (await catalogue.calibration(reranker)).floor if reranker else None

    def under(group: str, head: str, line: float | None) -> int | None:
        """How many questions of `group` a bar on the head of `head` flags as gaps."""
        if line is None:
            return None
        return sum(getattr(q, head)[0] < line for q in measured[group] if getattr(q, head))

    flagged = {
        "cosine_bar": bar,
        "answered_under_bar": under("answered", "similarities", bar),
        "unanswered_under_bar": under("unanswered", "similarities", bar),
        "rerank_floor": floor,
        "answered_under_floor": under("answered", "rerank_scores", floor),
        "unanswered_under_floor": under("unanswered", "rerank_scores", floor),
    }
    return {
        "shelf": shelf,
        "profile": profile,
        "reranker": reranker,
        "chunks": len(texts),
        "answered": len(measured["answered"]),
        "unanswered": len(measured["unanswered"]),
        "features": features,
        "bars": flagged,
        "questions": {
            group: [
                {
                    "question": q.question,
                    "similarities": q.similarities[:5],
                    "rerank_scores": q.rerank_scores[:5],
                    "coherence": q.coherence,
                }
                for q in found
            ]
            for group, found in measured.items()
        },
    }


async def _topics(profile: str) -> dict[str, Any]:
    from haskie.catalogue import catalogue
    from haskie.indexing import embed
    from haskie.settings import UserSettings

    model = await catalogue.embedding_model(UserSettings(embedding=profile))
    assert model is not None
    triples = json.loads((HERE / "topics.json").read_text())
    flat = [(topic, q) for topic, triple in enumerate(triples) for q in triple]
    vectors = np.asarray([embed.embed_query(model, q) for _, q in flat])
    units = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    pairs = units @ units.T
    within, across = [], []
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            (within if flat[i][0] == flat[j][0] else across).append(float(pairs[i, j]))
    bar = model.same_topic
    return {
        "same_topic_bar": bar,
        "within_min": min(within),
        "across_max": max(across),
        "within_joined": (sum(w > bar for w in within) / len(within)) if bar else None,
        "across_joined": sum(a > bar for a in across) if bar else None,
    }


def _report(results: list[dict[str, Any]], topics: dict[str, Any]) -> str:
    lines = []
    for one in results:
        lines.append(
            f"\n## {one['shelf']}: {one['chunks']} chunks, {one['answered']} answered, "
            f"{one['unanswered']} unanswered ({one['profile']}, {one['reranker'] or 'no reranker'})"
        )
        lines.append(f"{'feature':<14} {'AUROC':>6} {'answered min':>13} {'unanswered max':>15}")
        for name, f in sorted(one["features"].items(), key=lambda kv: -abs(kv[1]["auroc"] - 0.5)):
            low, high = f["answered_min"], f["unanswered_max"]
            lines.append(f"{name:<14} {f['auroc']:>6.3f} {low:>13.3f} {high:>15.3f}")
        b = one["bars"]
        if b["cosine_bar"] is not None:
            lines.append(
                f"cosine bar {b['cosine_bar']}: flags {b['answered_under_bar']} answered, "
                f"{b['unanswered_under_bar']} of {one['unanswered']} unanswered"
            )
        if b["rerank_floor"] is not None:
            lines.append(
                f"rerank floor {b['rerank_floor']}: flags {b['answered_under_floor']} answered, "
                f"{b['unanswered_under_floor']} of {one['unanswered']} unanswered"
            )
    lines.append(
        f"\n## topics: same-topic pairs >= {topics['within_min']:.3f}, other pairs <= "
        f"{topics['across_max']:.3f}; bar {topics['same_topic_bar']} joins "
        f"{topics['within_joined']:.0%} of same-topic pairs, {topics['across_joined']} others"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shelf", choices=["rust-book", "haskie-docs", "all"], default="all")
    parser.add_argument("--profile", default="compact", help="an embedding profile key")
    parser.add_argument("--reranker", default="Xenova/ms-marco-MiniLM-L-6-v2", help="or 'none'")
    parser.add_argument("--out", type=Path, help="write the full measurements as JSON here")
    args = parser.parse_args()
    reranker = None if args.reranker == "none" else args.reranker
    shelves = ["rust-book", "haskie-docs"] if args.shelf == "all" else [args.shelf]
    # a throwaway home: the catalogue (profiles, bars, reranker floors) is read from a fresh seed
    os.environ["HASKIE_HOME"] = tempfile.mkdtemp(prefix="haskie-gapeval-")

    async def run() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        results = [await _measure(shelf, args.profile, reranker) for shelf in shelves]
        return results, await _topics(args.profile)

    results, topics = asyncio.run(run())
    if args.out:
        args.out.write_text(json.dumps({"shelves": results, "topics": topics}, indent=1))
    print(_report(results, topics))


if __name__ == "__main__":
    main()
