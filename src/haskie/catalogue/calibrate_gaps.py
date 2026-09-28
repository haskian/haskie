"""Calibrate an embedding profile's gap bars on this home's own questions: the cosine under which a
question is a gap (`weak_match`) and the one from which it is answered (`answered_match`; between
the two it is borderline). Run it through `mise run calibrate-gaps`, in three steps:

1. `sample`: writes the questions this home's searches asked under the profile, newest first, to
   `eval/gap-questions.jsonl`, each with its best cosine and the three places that came closest.
   It reads the search log only: nothing is searched again.
2. A person reads each question and its near misses and sets `"answered": true` or `false`, or
   deletes the line when unsure.
3. `measure --profile NAME [--write]`: the highest low bar that flags no answered question, and
   the lowest high bar over every unanswered one. The band is kept only when it flags at most
   `MAX_BAND_ANSWERED` of the answered questions, as the seed's bars are (`catalogue/seed.sql`).
   `--write` stores both in this home's catalogue; the output also gives the values for the seed.

The bars do not travel between shelves (`docs/gaps.md`), which is why a home measures its own.
"""

import asyncio
import math
from pathlib import Path
from typing import Annotated

import msgspec
import typer
from sqlalchemy import update

from haskie import db
from haskie.catalogue import catalogue
from haskie.search import gaps, log
from haskie.tables import embedding_profiles

QUESTIONS = 80  # questions sampled: enough of each kind once a person has labelled them
MIN_EACH = 10  # labelled questions of each kind a bar is measured on, at least
MAX_BAND_ANSWERED = 0.15  # the share of answered questions a borderline band may hold


class Labelled(msgspec.Struct):
    """One logged question, its best cosine, what came closest, and whether it was answered."""

    id: int
    question: str
    profile: str
    best_similarity: float
    near_misses: list[str]  # the citations of its search's best results
    answered: bool | None = None  # set by a person; None is not labelled yet


class Measured(msgspec.Struct, frozen=True):
    """A profile's bars, and what they do to the questions they were measured on."""

    weak_match: float
    answered_match: float | None  # None: no band flags few enough answered questions
    caught: int  # unanswered questions under the low bar
    in_band: int  # unanswered questions in the band
    answered_in_band: int
    answered: int
    unanswered: int


def _down(value: float) -> float:
    return math.floor(value * 1000) / 1000


def _up(value: float) -> float:
    return math.ceil(value * 1000 + 1e-9) / 1000


def bars(answered: list[float], unanswered: list[float]) -> Measured:
    """The bars a profile's labelled best cosines give: the highest low bar under every answered
    question (a gap is a cosine strictly under it), and the lowest high bar over every unanswered
    one, kept only when the band it makes flags at most `MAX_BAND_ANSWERED` of the answered."""
    if len(answered) < MIN_EACH or len(unanswered) < MIN_EACH:
        raise ValueError(
            f"need {MIN_EACH} labelled questions of each kind, got {len(answered)} answered and "
            f"{len(unanswered)} unanswered"
        )
    low, top = _down(min(answered)), _up(max(unanswered))
    answered_in_band = sum(low <= one < top for one in answered)
    # no band when the low bar already catches every unanswered one, or the band costs too much
    banded = top > low and answered_in_band <= MAX_BAND_ANSWERED * len(answered)
    return Measured(
        weak_match=low,
        answered_match=top if banded else None,
        caught=sum(one < low for one in unanswered),
        in_band=sum(low <= one < top for one in unanswered) if banded else 0,
        answered_in_band=answered_in_band if banded else 0,
        answered=len(answered),
        unanswered=len(unanswered),
    )


async def _sample(out: Path, profile: str | None) -> int:
    found: dict[str, Labelled] = {}
    searched = await log.load()
    near = await log.top_results([search.id for search in searched], gaps.NEAR_MISSES)
    for search in searched:
        if search.embedding is None or search.error is not None:
            continue
        if profile is not None and search.embedding != profile:
            continue
        for asked in search.questions:
            if asked.id is None or asked.best_similarity is None or asked.question in found:
                continue
            found[asked.question] = Labelled(
                id=asked.id,
                question=asked.question,
                profile=search.embedding,
                best_similarity=asked.best_similarity,
                near_misses=[result.location for result in near[search.id]],
            )
    lines = [msgspec.json.encode(one).decode() for one in list(found.values())[:QUESTIONS]]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


async def _measure(questions: Path, profile: str, write: bool) -> Measured:
    """The bars the labelled questions of `profile` give, stored in this home when `write`."""
    labelled = [
        one
        for line in questions.read_text().splitlines()
        if line.strip()
        and (one := msgspec.json.decode(line, type=Labelled)).profile == profile
        and one.answered is not None
    ]
    measured = bars(
        [one.best_similarity for one in labelled if one.answered],
        [one.best_similarity for one in labelled if not one.answered],
    )
    if write:
        if profile not in await catalogue.embedders():
            raise ValueError(f"not an embedding profile in the catalogue: {profile}")
        async with db.connect() as conn:
            await conn.execute(
                update(embedding_profiles)
                .where(embedding_profiles.c.profile == profile)
                .values(weak_match=measured.weak_match, answered_match=measured.answered_match)
            )
    return measured


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def sample(
    out: Path = Path("eval/gap-questions.jsonl"),
    profile: Annotated[str | None, typer.Option(help="Only this embedding profile's.")] = None,
) -> None:
    """Write this home's logged questions, with their best cosines and near misses, to label."""
    written = asyncio.run(_sample(out, profile))
    typer.echo(f'{written} questions in {out}; set "answered" to true or false on each')


@app.command()
def measure(
    profile: Annotated[str, typer.Option(help="The embedding profile to calibrate.")],
    questions: Path = Path("eval/gap-questions.jsonl"),
    write: bool = False,
) -> None:
    """A profile's low and high bars from the labelled questions."""
    try:
        measured = asyncio.run(_measure(questions, profile, write))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    high = "null" if measured.answered_match is None else f"{measured.answered_match}"
    typer.echo(f"weak_match {measured.weak_match}, answered_match {high}")
    typer.echo(
        f"catches {measured.caught} of {measured.unanswered} unanswered under the low bar; the "
        f"band holds {measured.in_band} more and {measured.answered_in_band} of "
        f"{measured.answered} answered"
    )
    if write:
        typer.echo("written to this home's catalogue; restart the server to judge by them")


if __name__ == "__main__":
    app()
