"""`report.py`: the tables a run prints, from hand-built outcomes."""

from evals.bookqa import report
from evals.bookqa.metrics import Found, Outcome, Scores


def _outcome(
    mode: str,
    source: str,
    answerable: bool = True,
    rank: int | None = 1,
    empty: bool = False,
    query_type: str = "direct",
) -> Outcome:
    results = [] if empty else [Found(document=source, text="text", score=0.25)]
    scores = None
    if answerable:
        found = rank is not None
        scores = Scores(
            recall={1: float(rank == 1), 5: float(found and rank <= 5), 10: float(found)},
            mrr=1 / rank if rank else 0.0,
            ndcg=1.0 if rank == 1 else 0.0,
            document_hit=not empty,
            first_match=rank,
        )
    return Outcome(
        id=f"{source}-{mode}-{rank}",
        source=source,
        query_type=query_type,
        answerable=answerable,
        mode=mode,
        seconds=0.02,
        bytes=3000,
        results=results,
        scores=scores,
        abstained=empty,
    )


def _section(text: str, title: str) -> str:
    """The table under the heading `## {title}`, up to the next heading."""
    return text.split(f"## {title}\n")[1].split("\n## ")[0]


def _row(text: str, starts: str) -> list[str]:
    (line,) = [line for line in text.splitlines() if line.startswith(starts)]
    return [cell.strip() for cell in line.strip("|").split("|")]


def test_answerable_questions_are_averaged_per_mode_source_and_query_type() -> None:
    outcomes = [
        _outcome("fts", "raft.pdf", rank=1),
        _outcome("fts", "raft.pdf", rank=4, query_type="paraphrase"),
        _outcome("fts", "pro-git.pdf", rank=None, empty=True),
        _outcome("hybrid", "raft.pdf", rank=1),
    ]

    text = report.render(outcomes)

    by_mode = _section(text, "Answerable, by mode")
    # n, R@1, R@5, R@10, MRR, nDCG@10, doc@10, empty, kB, ms p50
    expected = ["fts", "3", "0.33", "0.67", "0.67", "0.42", "0.33", "0.67", "1", "3.0", "20"]
    assert _row(by_mode, "| fts |") == expected
    assert _row(by_mode, "| hybrid |")[:3] == ["hybrid", "1", "1.00"]
    by_source = _section(text, "Answerable, by mode and source")
    assert _row(by_source, "| fts | pro-git.pdf |")[:4] == ["fts", "pro-git.pdf", "1", "0.00"]
    by_type = _section(text, "Answerable, by mode and query type")
    assert _row(by_type, "| fts | paraphrase |")[:5] == ["fts", "paraphrase", "1", "0.00", "1.00"]


def test_unanswerable_questions_get_tables_of_their_own_counting_abstentions() -> None:
    outcomes = [
        _outcome("fts", "raft.pdf"),
        _outcome("fts", "raft.pdf", answerable=False, empty=True),
        _outcome("fts", "raft.pdf", answerable=False),
    ]

    text = report.render(outcomes)

    # n, abstained, top score
    assert _row(_section(text, "Unanswerable, by mode"), "| fts |")[:4] == [
        "fts",
        "2",
        "1/2",
        "0.250",
    ]
    assert _row(_section(text, "Answerable, by mode"), "| fts |")[1] == "1", "kept out of recall"


def test_a_run_without_unanswerable_questions_has_no_table_for_them() -> None:
    text = report.render([_outcome("fts", "raft.pdf")])

    assert "## Answerable, by mode" in text and "Unanswerable" not in text
