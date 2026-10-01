"""`judge.py`, `qrels.py` and the graded scoring in `metrics.judged`, without Claude: a fake `ask`
stands in for it, so these prove what is pooled, what is asked, what is kept and how it scores."""

import json
import math
from pathlib import Path

import msgspec
import pytest

from evals.bookqa import judge, metrics, qrels, report
from evals.bookqa.metrics import Found, Outcome
from evals.bookqa.qrels import Judgment
from evals.bookqa.tests import books


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return books.corpus(tmp_path / "corpus")


def _found(n: int, document: str = books.PDF) -> Found:
    return Found(document=document, text=books.LINES[n], score=1.0 / (n + 1))


def _outcome(rid: str, mode: str, ns: list[int], answerable: bool = True) -> Outcome:
    return Outcome(
        id=rid,
        source=books.PDF,
        query_type="direct",
        answerable=answerable,
        mode=mode,
        seconds=0.01,
        bytes=100,
        results=[_found(n) for n in ns],
        scores=None,
        abstained=not ns,
    )


def _judgment(rid: str, n: int, grade: int, version: str = judge.PROMPT_VERSION) -> Judgment:
    key = qrels.key(books.PDF, books.LINES[n])
    return Judgment(rid, books.PDF, key, grade, books.LINES[n], "m", version, "t")


class FakeJudge:
    def __init__(self, grades: list[int]) -> None:
        self.grades = grades
        self.prompts: list[str] = []

    def __call__(self, prompt: str, model: str) -> tuple[str, str]:
        self.prompts.append(prompt)
        listed = [{"n": n, "grade": g} for n, g in enumerate(self.grades, start=1)]
        return f"```json\n{json.dumps(listed)}\n```", "claude-opus-test"


def test_a_pool_is_every_distinct_passage_any_mode_returned_in_its_top_10() -> None:
    pooled = judge.pools(
        [
            _outcome("q1", "fts", [1, 2, 3]),
            _outcome("q1", "hybrid", [3, 2, 4]),  # 2 and 3 again: pooled once
            _outcome("q1", "rerank", list(range(10, 22))),  # past rank 10: not pooled
            _outcome("q2", "fts", [5]),
        ]
    )

    assert len(pooled["q1"]) == 3 + 1 + 10
    assert [f.text for f in pooled["q2"].values()] == [books.LINES[5]]


def test_the_judge_is_asked_the_question_alone_and_only_about_ungraded_passages(
    corpus: Path,
) -> None:
    record = books.record(corpus)
    ask = FakeJudge([2, 0])

    made = judge.judge(
        [record],
        [_outcome(record.id, "fts", [1, 65, 3])],
        known=[_judgment(record.id, 1, 1)],
        model="opus",
        ask=ask,
    )

    (prompt,) = ask.prompts
    assert record.query in prompt and record.expected_answer not in prompt, "no hint of the gold"
    assert books.LINES[1] not in prompt, "a passage graded already is not asked again"
    assert [(j.excerpt, j.grade) for j in made] == [(books.LINES[65], 2), (books.LINES[3], 0)]
    assert {(j.model, j.prompt_version) for j in made} == {("claude-opus-test", "v2")}


@pytest.mark.parametrize(
    "reply",
    [
        "no grades here",
        '[{"n": 1, "grade": 2}]',  # one passage short
        '[{"n": 1, "grade": 3}, {"n": 2, "grade": 0}]',  # not a grade
    ],
)
def test_a_reply_that_does_not_grade_every_passage_0_to_2_is_an_error(reply: str) -> None:
    with pytest.raises(judge.JudgeError):
        judge.parse(reply, 2)


def test_regrading_takes_back_only_that_grade_from_an_older_prompt() -> None:
    old_two, old_one = _judgment("q", 1, 2, "v1"), _judgment("q", 2, 1, "v1")
    new_two = _judgment("q", 3, 2)

    kept, stale = judge.stale_split([old_two, old_one, new_two], {2})

    assert stale == [old_two]
    assert kept == [old_one, new_two]


def test_judgments_round_trip_and_key_by_document_and_text(tmp_path: Path) -> None:
    made = [_judgment("q", 1, 2), _judgment("q", 2, 0)]
    path = tmp_path / "judgments.jsonl"
    path.write_text(qrels.dump(made))

    assert qrels.load(path) == made
    assert qrels.key(books.PDF, "A  passage.\n") == qrels.key(books.PDF, "a passage")
    assert qrels.key(books.PDF, "a passage") != qrels.key(books.MARKDOWN, "a passage")
    assert qrels.grades(made) == {("q", made[0].key): 2, ("q", made[1].key): 0}


def test_graded_scoring_counts_the_first_answer_and_how_much_was_judged() -> None:
    grades = qrels.grades(
        [_judgment("q", 1, 0), _judgment("q", 2, 1), _judgment("q", 3, 2), _judgment("q", 9, 2)]
    )

    scored = metrics.judged(_outcome("q", "fts", [1, 2, 3, 4]), grades)

    assert scored.success == {1: 0.0, 5: 1.0, 10: 1.0}
    assert scored.mrr == pytest.approx(1 / 3)
    assert scored.judged == 0.75, "passage 4 has no judgment"
    ideal = 3 + 3 / math.log2(3) + 1 / math.log2(4)  # the pool's grades 2, 2, 1, best first
    assert scored.ndcg == pytest.approx((1 / math.log2(3) + 3 / math.log2(4)) / ideal)


def test_a_run_with_no_results_scores_zero_and_counts_as_judged() -> None:
    scored = metrics.judged(_outcome("q", "fts", []), {})

    assert (scored.success[10], scored.mrr, scored.ndcg, scored.judged) == (0.0, 0.0, 0.0, 1.0)


def test_the_report_flags_an_unanswerable_question_the_judge_found_answered() -> None:
    outcomes = [_outcome("q", "fts", [1]), _outcome("u", "fts", [2], answerable=False)]

    text = report.render(outcomes, qrels.grades([_judgment("q", 1, 2), _judgment("u", 2, 2)]))

    assert "## Judged, by mode" in text
    flagged = text.split("## Unanswerable, but a passage was judged to answer them")[1]
    assert "- u" in flagged.split("##")[0]


def test_re_rendering_a_run_drops_questions_removed_from_the_dataset(
    corpus: Path, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(msgspec.json.encode(books.record(corpus)).decode() + "\n")
    outcomes = tmp_path / "outcomes.jsonl"
    kept, gone = _outcome(books.record(corpus).id, "fts", [65]), _outcome("removed", "fts", [1])
    outcomes.write_text("".join(msgspec.json.encode(o).decode() + "\n" for o in (kept, gone)))

    report.main([str(outcomes), "--dataset", str(dataset), "--judgments", str(tmp_path / "none")])

    table = (tmp_path / "report.md").read_text().split("## Answerable, by mode\n")[1]
    assert "| fts | 1 |" in table, "one question scored, the removed one dropped"
