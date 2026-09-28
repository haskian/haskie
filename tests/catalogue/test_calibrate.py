"""Calibrating a reranker on borderline pairs: its floor is their average score, its curve the
beta fitted to its scores, and `measure` stores both where a search reads them."""

import math
import statistics
from pathlib import Path

import pytest
import typer
from conftest import attach_document, import_document

from haskie.catalogue import calibrate, catalogue
from haskie.catalogue.calibrate import Borderline, Candidates, fit_beta, floor
from haskie.collection.collection import Collection
from haskie.indexing import embed
from haskie.search import log
from haskie.settings import (
    Accelerator,
    Reranker,
    SearchSettings,
    UserSettings,
    save_user_settings,
)

pytestmark = pytest.mark.anyio

MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@pytest.mark.parametrize(
    ("name", "scores", "expected"),
    [
        ("the average of the borderline scores", [0.1, 0.2, 0.6], 0.3),
        ("one pair is its own score", [0.42], 0.42),
    ],
)
def test_a_floor_is_the_average_borderline_score(
    name: str, scores: list[float], expected: float
) -> None:
    assert floor(scores) == pytest.approx(expected), name


def test_no_borderline_pairs_is_no_floor() -> None:
    with pytest.raises(ValueError, match="no borderline pairs"):
        floor([])


@pytest.mark.parametrize(
    ("name", "scores", "expected"),
    [
        ("too few to fit: the identity", [0.3], (1.0, 1.0)),
        ("all alike: the identity", [0.4, 0.4, 0.4], (1.0, 1.0)),
        ("spread wider than a beta can be: the identity", [0.0, 1.0, 0.0, 1.0], (1.0, 1.0)),
    ],
)
def test_a_curve_that_cannot_be_fitted_is_the_identity(
    name: str, scores: list[float], expected: tuple[float, float]
) -> None:
    assert fit_beta(scores) == expected, name


def test_a_fitted_curve_matches_the_scores_mean_and_spread() -> None:
    """Scores bunched low, as a reranker's are for most candidates: the fitted beta has their mean
    and variance, so its cumulative function lifts them toward an even spread."""
    scores = [0.01, 0.02, 0.02, 0.05, 0.1, 0.3, 0.8]

    a, b = fit_beta(scores)

    mean, variance = statistics.fmean(scores), statistics.pvariance(scores)
    assert a / (a + b) == pytest.approx(mean)
    assert a * b / ((a + b) ** 2 * (a + b + 1)) == pytest.approx(variance)
    assert a < 1, "skewed low: the curve rises steeply near 0"


async def test_measuring_stores_the_floor_and_curve_a_search_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reranker gives the borderline pairs logits 0 and 2: the floor is the average of their
    sigmoids, stored for the model and read back as the search reads it."""
    logits = {"borderline one": 0.0, "borderline two": 2.0, "far": -4.0, "near": 1.0}

    def rerank_scores(model: str, accelerator: Accelerator, q: str, ts: list[str]) -> list[float]:
        return [logits[text] for text in ts]

    loaded: list[str] = []
    monkeypatch.setattr(embed, "rerank_scores", rerank_scores)
    monkeypatch.setattr(embed, "warm_reranker", lambda name, accelerator: loaded.append(name))
    pairs = [("why", "borderline one"), ("how", "borderline two")]
    spread = [("why", "far"), ("why", "near"), ("how", "far")]

    ((name, measured, _),) = await calibrate._measure([MODEL], pairs, spread, write=True)

    expected = (0.5 + 1 / (1 + math.exp(-2))) / 2
    assert (name, measured.floor) == (MODEL, pytest.approx(expected))
    assert measured.source.endswith("on 2 borderline pairs")
    assert await catalogue.calibration(MODEL) == measured, "what a search reads next"
    assert loaded == [MODEL, MODEL], (
        "the model is loaded here, where no server does it, and the search's own check passes"
    )


async def test_measuring_no_borderline_pairs_is_refused_before_any_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty borderline file gives no floor, so no candidate is reranked for nothing."""
    scored: list[str] = []

    def rerank_scores(model: str, accelerator: Accelerator, q: str, ts: list[str]) -> list[float]:
        scored.extend(ts)
        return [0.0 for _ in ts]

    monkeypatch.setattr(embed, "rerank_scores", rerank_scores)
    monkeypatch.setattr(embed, "warm_reranker", lambda name, accelerator: None)

    with pytest.raises(typer.BadParameter, match="no borderline pairs"):
        await calibrate._measure([MODEL], [], [("why", "far")], write=True)

    assert scored == [], "refused before the candidates were scored"
    assert await catalogue.calibration(MODEL) == catalogue.UNCALIBRATED, "nothing written"


def test_the_judged_files_read_one_record_a_line(tmp_path: Path) -> None:
    judged = tmp_path / "borderline.jsonl"
    judged.write_text('{"query": "why", "text": "a"}\n\n{"query": "how", "text": "b"}\n')
    sampled = tmp_path / "candidates.jsonl"
    sampled.write_text('{"query": "why", "texts": ["a", "b"]}\n')

    assert calibrate.read_jsonl(judged, Borderline) == [
        Borderline("why", "a"),
        Borderline("how", "b"),
    ]
    assert calibrate.read_jsonl(sampled, Candidates) == [Candidates("why", ["a", "b"])]


async def test_sampling_reads_every_candidate_as_the_reranker_does_under_no_floor(
    dbos, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reranker on, with a floor of 0.9 no chunk reaches: a search keeps nothing, but the sample
    still writes the chunks ranked 10 to 30, since a floor measured on chunks over the old one
    could only rise. Each is written as the reranker reads it, its heading path first."""

    def rerank_scores(model: str, accelerator: Accelerator, q: str, ts: list[str]) -> list[float]:
        return [-3.0 - n / 100 for n, _ in enumerate(ts)]  # every sigmoid under 0.05

    loaded: list[str] = []
    monkeypatch.setattr(embed, "warm_reranker", lambda name, accelerator: loaded.append(name))
    monkeypatch.setattr(embed, "rerank_scores", rerank_scores)
    search = SearchSettings(reranker=Reranker.CROSS_ENCODER, min_rerank_score=0.9)
    await save_user_settings(UserSettings(search=search))
    await Collection.create("notes")
    body = "".join(f"# Part {i}\n\nretry note number {i}\n\n" for i in range(40))
    doc = await import_document(dbos, "guide.md", body, tmp_path)
    await attach_document(dbos, "notes", doc.name)
    out = tmp_path / "candidates.jsonl"
    # the question the home asked: the sample draws it from the search log
    async with log.capturing(log.Tool.EXCERPTS, ["retry"], None):
        pass

    written = await calibrate._sample(out)

    (sampled,) = calibrate.read_jsonl(out, Candidates)
    assert written == 1 and sampled.query == "retry"
    assert len(sampled.texts) == len(calibrate.CANDIDATES), "ranks 10 to 30, none dropped"
    assert all(text.startswith("Part ") and "\n\nretry note" in text for text in sampled.texts)
    assert search.reranker_model in loaded, "loaded here: no server runs to load it"
