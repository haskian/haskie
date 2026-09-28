"""How well each gap signal tells an answered question from an unanswered one, on labelled shelves.

Run it as `mise run evaluate-gaps` (see `--help`). Not part of `mise run test`: it downloads a book
and runs real models, for minutes.

Two shelves, both free for any use:

- `rust-book`: "The Rust Programming Language" (Apache-2.0 or MIT), fetched at the commit pinned in
  `rust_book.json` into a cache, never committed (see `NOTICE`). Code listings are `{{#include}}`
  lines in its source, so the prose is what gets indexed.
- `haskie-docs`: four of this repository's own docs, at the commit pinned in `haskie_docs.json`
  (`DOCS_COMMIT` reads another), since every edit to them moves the scores.

Each shelf is chunked with haskie's own chunker at the default settings and embedded with the
profile asked for. A question's ranking is its `CANDIDATES` nearest chunks by cosine, the pool a
vector search reads, and its cosines are taken with `log.similarities`, as the search log takes
them. The reranker scores the same pool. Each of `FEATURES` is then scored by AUROC: the chance that
a random answered question scores above a random unanswered one (0.5 is a coin). The topic pairs of
`topics.json` give the `same_topic` bar the same way.
"""

import argparse
import asyncio
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
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
    commit = os.environ.get("DOCS_COMMIT", labels["commit"])
    return "\n\n".join(_at(commit, path) for path in labels["files"]), labels


def _at(commit: str, path: str) -> str:
    """One of haskie's docs as it read at `commit`: the docs change, the measured bars do not."""
    shown = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return shown.stdout


def auroc(answered: list[float], unanswered: list[float]) -> float:
    """P(an answered score > an unanswered one), ties counted half: the Mann-Whitney U, scaled."""
    if not answered or not unanswered:
        return math.nan
    a, u = np.asarray(answered), np.asarray(unanswered)
    wins = (a[:, None] > u[None, :]).sum() + 0.5 * (a[:, None] == u[None, :]).sum()
    return float(wins / (len(a) * len(u)))


def _gap12(scores: list[float]) -> float | None:
    return scores[0] - scores[1] if len(scores) > 1 else None


def _spread(scores: list[float]) -> float | None:
    return float(np.std(scores[:10])) if len(scores) > 1 else None


def _mean5(scores: list[float]) -> float | None:
    return float(np.mean(scores[:5])) if scores else None


COHERENT = 10  # nearest rows whose likeness to each other `coherence` averages


def _coherence(nearest: np.ndarray) -> float | None:
    """The mean cosine between the nearest rows, pair by pair: the literature's dense
    post-retrieval predictor. An answered question's nearest rows tend to be about one thing."""
    if len(nearest) < 2:
        return None
    from haskie.search import collapse

    units = collapse.unit_rows(list(nearest))
    pairs = units @ units.T
    return float((pairs.sum() - np.trace(pairs)) / (len(units) * (len(units) - 1)))


# The predictors a question's score profile (`log.similarities`, the reranker's scores) can be
# judged by, each a pure function of one `Asked`; None when the profile cannot say. `max` is what
# the Gaps page judges by (`gaps._weak`); the others are measured against it.
FEATURES: dict[str, Callable[[Any], float | None]] = {
    "max": lambda one: one.logged.similarities[0] if one.logged.similarities else None,
    "gap12": lambda one: _gap12(one.logged.similarities),
    "spread": lambda one: _spread(one.logged.similarities),
    "mean5": lambda one: _mean5(one.logged.similarities),
    "coherence": lambda one: one.coherence,
    "rerank_max": lambda one: one.logged.rerank_scores[0] if one.logged.rerank_scores else None,
    "rerank_gap12": lambda one: _gap12(one.logged.rerank_scores),
    "rerank_mean5": lambda one: _mean5(one.logged.rerank_scores),
}


class Asked:
    """One question put to one shelf: its logged score profile, its three nearest chunks (what the
    Gaps page cites as near misses) and the words of it the five nearest chunks do not hold."""

    def __init__(
        self, logged: Any, nearest: list[int], missing: list[str], coherence: float | None
    ) -> None:
        self.logged, self.nearest, self.missing = logged, nearest, missing
        self.coherence = coherence


async def _index(shelf: str, model: Any) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """The shelf's chunks as the models read them, their vectors (cached), and its labels."""
    from haskie.indexing import chunk, embed
    from haskie.settings import ChunkSettings

    text, labels = _shelf(shelf)
    texts = [chunk.framed(one.frame, one.text) for one in chunk.split(text, ChunkSettings())]
    key = hashlib.sha256("\x00".join([model.cache_name, *texts]).encode()).hexdigest()[:16]
    cached = CACHE / f"{shelf}-{key}.npy"
    if cached.exists():
        return texts, np.load(cached), labels
    print(f"{shelf}: {len(texts)} chunks, embedding with {model.name}", file=sys.stderr)
    vectors = np.asarray(
        [
            v
            for at in range(0, len(texts), BATCH)
            for v in embed.embed_texts(model, texts[at : at + BATCH])
        ]
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cached, vectors)
    return texts, vectors, labels


def _ask(
    question: str, model: Any, texts: list[str], vectors: np.ndarray, reranker: str | None
) -> Asked:
    from haskie.collection.index import _sigmoid
    from haskie.indexing import embed
    from haskie.search import collapse, log, probe
    from haskie.settings import Accelerator

    query = np.asarray(embed.embed_query(model, question))
    pool = np.argsort(-(collapse.unit_rows(list(vectors)) @ collapse.unit_rows([query])[0]))[
        :CANDIDATES
    ]
    found = log.similarities(query.tolist(), vectors[pool].tolist())
    scores: list[float] = []
    if reranker is not None:
        logits = embed.rerank_scores(reranker, Accelerator.CPU, question, [texts[i] for i in pool])
        scores = sorted((_sigmoid(one) for one in logits), reverse=True)[: log.PROFILE]
    asked = probe.Question(vector=None, asked=question)
    missing = list(probe.missing([asked], [texts[i] for i in pool[:5]]))
    logged = log.LoggedQuestion(question, similarities=found, rerank_scores=scores)
    return Asked(logged, [int(i) for i in pool[:3]], missing, _coherence(vectors[pool[:COHERENT]]))


async def _measure(
    shelves: list[str], profile: str, reranker: str | None
) -> tuple[dict[str, dict[str, list[Asked]]], Any]:
    """Every labelled question of every shelf, and the topic questions against each shelf."""
    from haskie.catalogue import catalogue
    from haskie.settings import UserSettings

    model = await catalogue.embedding_model(UserSettings(embedding=profile))
    assert model is not None, "a profile with a model"
    triples = json.loads((HERE / "topics.json").read_text())
    measured: dict[str, dict[str, list[Asked]]] = {}
    for shelf in shelves:
        texts, vectors, labels = await _index(shelf, model)
        groups = {
            "answered": labels["answered"],
            "unanswered": labels["unanswered_near"] + labels["unanswered_far"],
            "reworded": labels.get("reworded", []),
            "topics": [q for triple in triples for q in triple],
        }
        measured[shelf] = {
            group: [_ask(q, model, texts, vectors, reranker) for q in questions]
            for group, questions in groups.items()
        }
    return measured, model


def _features(measured: dict[str, dict[str, list[Asked]]]) -> list[str]:
    """AUROC of every feature per shelf, and the step-1 gate: a bar under every answered question
    of every shelf, and how many gaps it catches on each."""
    lines = [
        f"{'feature':<14} "
        + " ".join(f"{s[:11]:>11}" for s in measured)
        + "   bar under all answered: caught"
    ]
    for name, feature in FEATURES.items():
        values = {
            shelf: {
                group: [v for one in found[group] if (v := feature(one)) is not None]
                for group in ("answered", "unanswered")
            }
            for shelf, found in measured.items()
        }
        if any(not v["answered"] or not v["unanswered"] for v in values.values()):
            continue
        aurocs = [auroc(v["answered"], v["unanswered"]) for v in values.values()]
        bar = min(min(v["answered"]) for v in values.values())
        caught = ", ".join(
            f"{sum(u < bar for u in v['unanswered'])}/{len(v['unanswered'])}"
            for v in values.values()
        )
        lines.append(
            f"{name:<14} " + " ".join(f"{a:>11.3f}" for a in aurocs) + f"   {bar:.3f}: {caught}"
        )
    return lines


def _verdict(one: Any, low: float | None, high: float | None) -> str:
    best = one.logged.best_similarity
    if low is None or best is None:
        return "none"
    if best < low:
        return "weak"
    return "borderline" if high is not None and best < high else "answered"


def _bars(
    measured: dict[str, dict[str, list[Asked]]], model: Any, floor: float | None
) -> list[str]:
    """How the catalogue's bars sort each group: weak, borderline or answered; and the reranker's
    floor. The vocabulary rule (step 3) on top: borderline and with words no near miss holds."""
    lines = []
    low, high = model.weak_match, model.answered_match
    lines.append(f"cosine bars: weak under {low}, borderline under {high}")
    for shelf, found in measured.items():
        for group in ("answered", "unanswered", "reworded"):
            asked = found[group]
            if not asked:
                continue
            verdicts = [_verdict(one, low, high) for one in asked]
            vocabulary = sum(
                v in {"weak", "borderline"} and bool(one.missing)
                for v, one in zip(verdicts, asked, strict=True)
            )
            counts = {v: verdicts.count(v) for v in ("weak", "borderline", "answered")}
            floored = (
                sum(
                    one.logged.rerank_scores[0] < floor for one in asked if one.logged.rerank_scores
                )
                if floor is not None
                else None
            )
            lines.append(
                f"  {shelf:<11} {group:<10} {len(asked):>3}: weak {counts['weak']:>2}, borderline "
                f"{counts['borderline']:>2}, answered {counts['answered']:>2}; words missing "
                f"(of weak or borderline) {vocabulary:>2}; under rerank floor {floored}"
            )
    return lines


def _joined(
    pairs: list[tuple[int, int]],
    cosines: np.ndarray,
    nearest: list[set[int]],
    bar: float | None,
    low: float,
    share: float,
) -> int:
    """How many pairs a topic rule joins: query cosine over `bar`, or over `low` with at least
    `share` of their near misses in common (Jaccard of their three nearest chunks)."""

    def join(i: int, j: int) -> bool:
        if bar is not None and cosines[i, j] > bar:
            return True
        union = nearest[i] | nearest[j]
        overlap = len(nearest[i] & nearest[j]) / len(union) if union else 0.0
        return cosines[i, j] > low and overlap >= share

    return sum(join(i, j) for i, j in pairs)


def _topics(measured: dict[str, dict[str, list[Asked]]], model: Any) -> list[str]:
    """The step-4 gate: same-topic pairs joined and other pairs merged, by query cosine alone
    (`same_topic`), and by a lower cosine plus shared near misses."""
    from haskie.indexing import embed
    from haskie.search import collapse

    triples = json.loads((HERE / "topics.json").read_text())
    topic = [n for n, triple in enumerate(triples) for _ in triple]
    asked = [q for triple in triples for q in triple]
    cosines = collapse.unit_rows([embed.embed_query(model, q) for q in asked])
    cosines = cosines @ cosines.T
    pairs = [(i, j) for i in range(len(topic)) for j in range(i + 1, len(topic))]
    same = [(i, j) for i, j in pairs if topic[i] == topic[j]]
    other = [(i, j) for i, j in pairs if topic[i] != topic[j]]
    bar = model.same_topic
    lines = []
    for shelf, found in measured.items():
        nearest = [set(one.nearest) for one in found["topics"]]
        ok = _joined(same, cosines, nearest, bar, 2.0, 2.0) / len(same)
        bad = _joined(other, cosines, nearest, bar, 2.0, 2.0)
        lines.append(f"  {shelf}: cosine > {bar} alone: {ok:.0%} joined, {bad} merged")
        for low in (0.5, 0.55, 0.6, 0.65):
            for share in (0.5, 1.0):
                ok = _joined(same, cosines, nearest, bar, low, share) / len(same)
                bad = _joined(other, cosines, nearest, bar, low, share)
                lines.append(
                    f"    + cosine > {low}, near misses shared >= {share}: "
                    f"{ok:.0%} joined, {bad} merged"
                )
    return lines


def _report(measured: dict[str, dict[str, list[Asked]]], model: Any, floor: float | None) -> str:
    sizes = ", ".join(
        f"{shelf} {len(f['answered'])} answered / {len(f['unanswered'])} unanswered"
        for shelf, f in measured.items()
    )
    return "\n".join(
        [
            f"# {model.profile}: {sizes}",
            "\n## features: AUROC per shelf (answered over unanswered)",
            *_features(measured),
            "\n## bars",
            *_bars(measured, model, floor),
            "\n## topics",
            *_topics(measured, model),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--shelf", choices=["rust-book", "haskie-docs", "all"], default="all")
    parser.add_argument("--profile", default="compact", help="an embedding profile key")
    parser.add_argument("--reranker", default="Xenova/ms-marco-MiniLM-L-6-v2", help="or 'none'")
    parser.add_argument("--out", type=Path, help="write each question's measurements as JSON here")
    args = parser.parse_args()
    reranker = None if args.reranker == "none" else args.reranker
    shelves = ["rust-book", "haskie-docs"] if args.shelf == "all" else [args.shelf]
    # a throwaway home: the catalogue (profiles, bars, reranker floors) is read from a fresh seed
    os.environ["HASKIE_HOME"] = tempfile.mkdtemp(prefix="haskie-gapeval-")

    async def run() -> tuple[Any, Any, float | None]:
        from haskie.catalogue import catalogue

        measured, model = await _measure(shelves, args.profile, reranker)
        floor = (await catalogue.calibration(reranker)).floor if reranker else None
        return measured, model, floor

    measured, model, floor = asyncio.run(run())
    if args.out:
        rows = {
            shelf: {
                group: [
                    {
                        "question": one.logged.question,
                        "similarities": one.logged.similarities[:5],
                        "rerank_scores": one.logged.rerank_scores[:5],
                        "coherence": one.coherence,
                        "nearest": one.nearest,
                        "missing": one.missing,
                    }
                    for one in asked
                ]
                for group, asked in found.items()
            }
            for shelf, found in measured.items()
        }
        args.out.write_text(json.dumps(rows, indent=1))
    print(_report(measured, model, floor))


if __name__ == "__main__":
    main()
