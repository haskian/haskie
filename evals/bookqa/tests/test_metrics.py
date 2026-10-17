"""`metrics.py`: when a search result counts for a gold passage, and the arithmetic of Recall@k,
MRR and nDCG@10 over hand-built rankings."""

import math
from pathlib import Path

import pytest

from evals.bookqa import metrics
from evals.bookqa.metrics import Found
from evals.bookqa.schema import Passage
from evals.bookqa.tests import books

GOLD = books.LINES[65]
OTHER_GOLD = books.LINES[12]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return books.corpus(tmp_path / "corpus")


def _hit(text: str = GOLD, document: str = books.PDF) -> Found:
    return Found(document=document, text=text, score=1.0)


def _miss(n: int) -> Found:
    return Found(document=books.PDF, text=books.LINES[n], score=0.5)


def _ranking(gold_at: dict[int, str], length: int = 12) -> list[Found]:
    """`length` results, the gold text at the given 1-based ranks and misses elsewhere."""
    return [
        _hit(gold_at[rank]) if rank in gold_at else _miss(rank) for rank in range(1, length + 1)
    ]


@pytest.mark.parametrize(
    ("name", "text", "counts"),
    [
        ("the quote itself", GOLD, True),
        ("inside a longer passage", f"{books.LINES[64]} {GOLD} {books.LINES[66]}", True),
        ("hyphenated across a line", GOLD.replace("attempts", "at-\ntempts"), True),
        ("with haskie's markup", GOLD.replace("worker", "<sub>worker</sub>"), True),
        ("with a link", GOLD.replace("retry budget", "[retry budget](#budget)"), True),
        ("cut by a chunk boundary, most of it kept", " ".join(GOLD.split()[3:]), True),
        ("only its first few words", " ".join(GOLD.split()[:4]), False),
        ("a neighbouring line", books.LINES[64], False),
    ],
)
def test_a_result_counts_for_a_passage_that_it_mostly_holds(
    name: str, text: str, counts: bool
) -> None:
    assert metrics.matches(_hit(text), Passage(books.PDF, GOLD, 2)) is counts, name


def test_the_same_text_in_another_document_does_not_count() -> None:
    assert not metrics.matches(_hit(document=books.MARKDOWN), Passage(books.PDF, GOLD, 2))


@pytest.mark.parametrize(
    ("rank", "recall", "mrr"),
    [
        (1, {1: 1.0, 5: 1.0, 10: 1.0}, 1.0),
        (3, {1: 0.0, 5: 1.0, 10: 1.0}, 1 / 3),
        (7, {1: 0.0, 5: 0.0, 10: 1.0}, 1 / 7),
        (11, {1: 0.0, 5: 0.0, 10: 0.0}, 0.0),  # past the top 10: not found
    ],
)
def test_recall_at_k_and_mrr_follow_the_rank_of_the_one_gold_passage(
    corpus: Path, rank: int, recall: dict, mrr: float
) -> None:
    scores = metrics.score(books.record(corpus), _ranking({rank: GOLD}))

    assert scores.recall == recall
    assert scores.mrr == pytest.approx(mrr)
    assert scores.first_match == (rank if rank <= 10 else None)
    assert scores.ndcg == pytest.approx(1 / math.log2(rank + 1) if rank <= 10 else 0.0)


def test_with_several_gold_passages_recall_is_the_share_found_and_mrr_the_first(
    corpus: Path,
) -> None:
    record = books.record(
        corpus,
        relevant_passages=[Passage(books.PDF, GOLD, 2), Passage(books.PDF, OTHER_GOLD, 1)],
    )

    scores = metrics.score(record, _ranking({2: OTHER_GOLD, 7: GOLD}))

    assert scores.recall == {1: 0.0, 5: 0.5, 10: 1.0}
    assert scores.mrr == pytest.approx(1 / 2)
    ideal = 1 + 1 / math.log2(3)
    assert scores.ndcg == pytest.approx((1 / math.log2(3) + 1 / math.log2(8)) / ideal)


def test_a_gold_passage_found_twice_gains_once(corpus: Path) -> None:
    scores = metrics.score(books.record(corpus), _ranking({1: GOLD, 2: GOLD}))

    assert scores.recall[1] == 1.0
    assert scores.ndcg == pytest.approx(1.0), "the repeat at rank 2 adds nothing, and costs nothing"


def test_a_query_with_no_result_scores_zero_and_abstains(corpus: Path) -> None:
    scores = metrics.score(books.record(corpus), [])

    assert scores.recall == {1: 0.0, 5: 0.0, 10: 0.0}
    assert (scores.mrr, scores.ndcg, scores.document_hit, scores.first_match) == (0, 0, False, None)
    assert metrics.abstained([])


def test_the_right_document_without_the_passage_is_a_document_hit_only(corpus: Path) -> None:
    scores = metrics.score(books.record(corpus), [_miss(3), _miss(4)])

    assert scores.document_hit
    assert scores.recall[10] == 0.0


def test_an_unanswerable_question_abstains_only_on_an_empty_result() -> None:
    assert metrics.abstained([])
    assert not metrics.abstained([_miss(1)])
