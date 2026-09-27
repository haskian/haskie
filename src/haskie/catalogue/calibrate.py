"""Calibrate each reranker's scores on your own collections: the floor a search drops chunks under
(`min_rerank_score`), and the beta curve that spreads its scores evenly over 0 to 1
(`fill_values = absolute`). Run it through `mise run calibrate-rerankers`, in three steps:

1. `sample`: draws up to 40 distinct questions from the searches this home recorded, searches each
   again, and writes the chunks ranked 10 to 30 as candidates to `eval/candidates.jsonl`.
2. A person reads each question's candidates and copies one that is borderline relevant, as
   `{"query": ..., "text": ...}`, one line a question, into `eval/borderline.jsonl`.
3. `measure`: scores every borderline pair with each reranker named, and every candidate pair
   too. A model's floor is the average score of its borderline pairs, the way Cohere sets a
   relevance threshold ("Select a set of 30-50 representative queries ... borderline relevant
   ... The average ... can then be used as a reference"). Its curve is the beta distribution fitted
   to its candidate scores by moments, whose cumulative function maps them to an even spread:
   dsRAG reshapes its rerankers' scores so for Relevant Segment Extraction. `--write` stores both
   in this home's catalogue; the output also gives the `seed.sql` rows.

The pure parts (`floor`, `fit_beta`) are separate from the IO so they can be tested alone.
"""

import asyncio
import json
import math
import statistics
import time
from datetime import date
from pathlib import Path
from typing import Annotated

import msgspec
import typer
from sqlalchemy import select

from haskie import db
from haskie.catalogue import catalogue
from haskie.catalogue.catalogue import RerankerCalibration
from haskie.search import flow, retrieval
from haskie.search.session import Action
from haskie.settings import load_user_settings
from haskie.tables import reranker_calibration, session_events

QUESTIONS = 40  # Cohere asks for 30 to 50 representative queries
CANDIDATES = range(10, 30)  # past the top, where the borderline chunks sit
MIN_PAIRS = 2  # a variance needs two scores


class Borderline(msgspec.Struct):
    """One question and a chunk a person judged borderline relevant to it."""

    query: str
    text: str


class Candidates(msgspec.Struct):
    """One question and the chunks its search ranked past the top, to judge."""

    query: str
    texts: list[str]


def floor(scores: list[float]) -> float:
    """A reranker's floor: the average score of its borderline pairs (Cohere's reference)."""
    if not scores:
        raise ValueError("no borderline pairs to average")
    return statistics.fmean(scores)


def fit_beta(scores: list[float]) -> tuple[float, float]:
    """The beta distribution matching `scores` (in 0 to 1) by its mean and variance, whose
    cumulative function spreads them evenly over 0 to 1. (1, 1), the identity, when they are too
    few or too alike to fit, or spread wider than any beta can be."""
    if len(scores) < MIN_PAIRS:
        return (1.0, 1.0)
    mean, variance = statistics.fmean(scores), statistics.pvariance(scores)
    # a beta's variance is under mean·(1 − mean): at it the scores sit at 0 and 1 only
    if variance <= 0 or variance >= mean * (1 - mean):
        return (1.0, 1.0)
    common = mean * (1 - mean) / variance - 1
    return (mean * common, (1 - mean) * common)


def _read[T](path: Path, kind: type[T]) -> list[T]:
    return [msgspec.json.decode(line, type=kind) for line in path.read_text().splitlines() if line]


async def _asked(limit: int) -> list[str]:
    """The distinct questions this home searched for, newest first: each part of a search of
    several counts as one."""
    async with db.connect() as conn:
        rows = await conn.execute(
            select(session_events.c.subject, session_events.c.detail)
            .where(session_events.c.action == Action.SEARCH)
            .order_by(session_events.c.ts.desc())
        )
        found: dict[str, None] = {}
        for subject, detail in rows:
            parts = json.loads(detail or "{}").get("questions") or [subject]
            found.update(dict.fromkeys(part for part in parts if part.strip()))
    return list(found)[:limit]


async def _sample(out: Path) -> int:
    names = await retrieval.scope(None, None)
    lines = []
    for query in await _asked(QUESTIONS):
        hits = await flow.chunks(names, query, limit=CANDIDATES.stop)
        texts = [hit.text for hit in hits[CANDIDATES.start : CANDIDATES.stop]]
        if texts:
            lines.append(msgspec.json.encode(Candidates(query, texts)).decode())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return len(lines)


async def _scores(model: str, pairs: list[tuple[str, str]]) -> list[float]:
    """Each pair's reranker score, as a search scores it: the sigmoid of the logit."""
    from haskie.indexing.embed import rerank_scores

    accelerator = (await load_user_settings()).pipeline.accelerator
    found: list[float] = []
    for query, text in pairs:
        (logit,) = rerank_scores(model, accelerator, query, [text])
        found.append(1 / (1 + math.exp(-logit)) if logit > -700 else 0.0)
    return found


async def _measure(
    models: list[str], pairs: list[tuple[str, str]], spread: list[tuple[str, str]], write: bool
) -> list[tuple[str, RerankerCalibration, float]]:
    """Each model's calibration, and the seconds it took; stored in this home when `write`."""
    known = await catalogue.rerankers()
    source = f"calibrated {date.today().isoformat()} on {len(pairs)} borderline pairs"
    found = []
    for name in models:
        if name not in known:
            raise typer.BadParameter(f"not a reranker in the catalogue: {name}")
        started = time.perf_counter()
        beta_a, beta_b = fit_beta(await _scores(name, spread))
        measured = RerankerCalibration(floor(await _scores(name, pairs)), beta_a, beta_b, source)
        if write:
            await _write(name, measured)
        found.append((name, measured, time.perf_counter() - started))
    return found


async def _write(model: str, measured: RerankerCalibration) -> None:
    async with db.connect() as conn:
        await conn.execute(
            reranker_calibration.delete().where(reranker_calibration.c.model == model)
        )
        await conn.execute(
            reranker_calibration.insert().values(model=model, **msgspec.structs.asdict(measured))
        )


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def sample(out: Path = Path("eval/candidates.jsonl")) -> None:
    """Write each searched question's chunks ranked 10 to 30, to judge."""
    written = asyncio.run(_sample(out))
    typer.echo(f"{written} questions to judge in {out}; copy one borderline chunk each into ")
    typer.echo('eval/borderline.jsonl as {"query": ..., "text": ...}')


@app.command()
def measure(
    model: Annotated[list[str], typer.Option(help="A reranker to calibrate; repeat for several.")],
    borderline: Path = Path("eval/borderline.jsonl"),
    candidates: Path = Path("eval/candidates.jsonl"),
    write: bool = False,
) -> None:
    """Each reranker's floor over the borderline pairs and its curve over the candidates."""
    pairs = [(one.query, one.text) for one in _read(borderline, Borderline)]
    spread = [(one.query, text) for one in _read(candidates, Candidates) for text in one.texts]
    for name, measured, seconds in asyncio.run(_measure(model, pairs, spread, write)):
        typer.echo(
            f"('{name}', {measured.floor:.4f}, {measured.beta_a:.3f}, {measured.beta_b:.3f}, "
            f"'{measured.source}'),  -- {seconds:.1f} s"
        )


if __name__ == "__main__":
    app()
