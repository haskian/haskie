"""Calibrating a profile's gap bars on this home's own labelled questions: the low bar is the
highest that flags no answered question, the high bar the lowest over every unanswered one, kept
only when its band costs few answered; `sample` reads the log and `measure` stores the bars where
the Gaps page reads them."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from haskie.catalogue import calibrate_gaps, catalogue
from haskie.catalogue.calibrate_gaps import MIN_EACH, bars
from haskie.search import log

pytestmark = pytest.mark.anyio

ANSWERED = [0.81, 0.79, 0.77, 0.76, 0.755, 0.75, 0.74, 0.73, 0.72, 0.7032]
UNANSWERED = [0.45, 0.5, 0.55, 0.6, 0.62, 0.64, 0.66, 0.68, 0.69, 0.7]


@pytest.mark.parametrize(
    ("name", "answered", "unanswered", "expected"),
    [
        (
            "overlap: a band over the last unanswered, costing one answered",
            ANSWERED,
            UNANSWERED[:-1] + [0.7045],
            (0.703, 0.705, 9, 1, 1),
        ),
        (
            "apart: the low bar catches them all, no band is needed",
            ANSWERED,
            UNANSWERED,
            (0.703, None, 10, 0, 0),
        ),
        (
            "a band that would flag too many answered is none",
            ANSWERED,
            UNANSWERED[:-1] + [0.79],
            (0.703, None, 9, 0, 0),
        ),
    ],
)
def test_bars_flag_no_answered_question(
    name: str, answered: list[float], unanswered: list[float], expected: tuple
) -> None:
    found = bars(answered, unanswered)
    low, high, caught, in_band, answered_in_band = expected
    assert (found.weak_match, found.answered_match) == (low, high), name
    assert (found.caught, found.in_band, found.answered_in_band) == (
        caught,
        in_band,
        answered_in_band,
    )
    assert not any(one < found.weak_match for one in answered), "no answered question under it"


@pytest.mark.parametrize(
    ("name", "answered", "unanswered"),
    [
        ("too few answered", ANSWERED[: MIN_EACH - 1], UNANSWERED),
        ("too few unanswered", ANSWERED, UNANSWERED[: MIN_EACH - 1]),
        ("one kind missing", ANSWERED, []),
    ],
)
def test_too_few_labels_measure_nothing(
    name: str, answered: list[float], unanswered: list[float]
) -> None:
    with pytest.raises(ValueError, match=f"need {MIN_EACH}"):
        bars(answered, unanswered)


async def _logged(
    question: str, similarity: float, profile: str = "granite-97m-multilingual"
) -> None:
    """One search as the log keeps it, measured without running one."""
    async with log.capturing(log.Tool.EXCERPTS, [question], None) as capture:
        log.observe_scope(None, ["notes"], log.SearchMode.VECTOR, 10)
        capture.embedding = profile
        (asked,) = capture.asked
        asked.similarities = [similarity]


async def test_sample_label_measure_and_write(seeded_home, tmp_path: Path) -> None:
    """The log's questions of one profile, each once; a person labels them; the bars they give
    land in this home's catalogue."""
    for n, (a, u) in enumerate(zip(ANSWERED, UNANSWERED, strict=True)):
        await _logged(f"answered {n}", a)
        await _logged(f"unanswered {n}", u)
    await _logged("answered 0", 0.99)  # asked again: the newest counts
    await _logged("another profile's", 0.1, profile="bekko-a25m")
    out = tmp_path / "gap-questions.jsonl"

    written = await calibrate_gaps._sample(out, "granite-97m-multilingual")

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert written == len(rows) == 20, "each question once, one profile only"
    assert {row["answered"] for row in rows} == {None}, "left for a person to label"
    newest = next(r for r in rows if r["question"] == "answered 0")["best_similarity"]
    assert newest == pytest.approx(0.99), "float32, as the log stores it"
    for row in rows:
        row["answered"] = row["question"].startswith("answered")
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))

    measured = await calibrate_gaps._measure(out, "granite-97m-multilingual", write=True)

    assert (measured.weak_match, measured.answered_match, measured.caught) == (0.703, None, 10)
    catalogue._embedders.clear()  # the process caches the catalogue, as a server would
    compact = (await catalogue.embedders())["granite-97m-multilingual"]
    assert (compact.weak_match, compact.answered_match) == (0.703, None)

    labels = out.read_text()
    with pytest.raises(FileExistsError):
        await calibrate_gaps._sample(out, "granite-97m-multilingual")
    assert out.read_text() == labels, "a second sample keeps the labels"


async def test_measure_refuses_an_unknown_profile(seeded_home, tmp_path: Path) -> None:
    out = tmp_path / "gap-questions.jsonl"
    rows = [
        {
            "id": n,
            "question": f"q{n}",
            "profile": "ghost",
            "best_similarity": s,
            "near_misses": [],
            "answered": n < 10,
        }
        for n, s in enumerate(ANSWERED + UNANSWERED)
    ]
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="not an embedding profile in the catalogue: ghost"):
        await calibrate_gaps._measure(out, "ghost", write=True)


def test_the_command_reports_the_bars(tmp_path: Path) -> None:
    """The command line prints what it measured, and a refusal as a usage error."""
    out = tmp_path / "gap-questions.jsonl"
    rows = [
        {
            "id": n,
            "question": f"q{n}",
            "profile": "granite-97m-multilingual",
            "best_similarity": s,
            "near_misses": [],
            "answered": n < 10,
        }
        for n, s in enumerate(ANSWERED + UNANSWERED)
    ]
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    runner = CliRunner()

    measured = runner.invoke(
        calibrate_gaps.app,
        ["measure", "--profile", "granite-97m-multilingual", "--questions", str(out)],
    )
    refused = runner.invoke(
        calibrate_gaps.app, ["measure", "--profile", "bekko-a25m", "--questions", str(out)]
    )

    assert measured.exit_code == 0, measured.output
    assert "weak_match 0.703, answered_match null" in measured.output
    assert "catches 10 of 10 unanswered" in measured.output
    assert refused.exit_code != 0 and "need 10" in refused.output


def test_sample_replaces_labels_only_when_forced(seeded_home, tmp_path: Path) -> None:
    out = tmp_path / "gap-questions.jsonl"
    out.write_text('{"answered": true}\n')
    runner = CliRunner()

    kept = runner.invoke(calibrate_gaps.app, ["sample", "--out", str(out)])
    labels = out.read_text()
    forced = runner.invoke(calibrate_gaps.app, ["sample", "--out", str(out), "--force"])

    # a usage error; its text is wrapped and coloured to the terminal's width, so not matched
    assert kept.exit_code == 2, kept.output
    assert labels == '{"answered": true}\n', "the labels are kept"
    assert forced.exit_code == 0, forced.output
    assert "0 questions" in forced.output and out.read_text() == "", "the empty log's sample"
