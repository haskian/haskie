"""Scored against `revision_ranges.py` as the agent wrote it.

Grounded in Pro Git's "Commit Ranges" section (pro-git.pdf).

`diff_from_merge_base` is real, book-grounded material (the book's own example is `git diff
master...contrib`, described as showing only what `contrib` introduced "since its common
ancestor with master") - but it is not a *discriminating* check: since these functions only
build argument lists, and `git diff left...right` and `git log left...right` happen to take the
identical `["<cmd>", "left...right"]` shape, a guess that never distinguished the two would
still pass. The real discriminator below is "Multiple Points": `git log refA refB ^refC` (or
equivalently `--not`), which lets a query name more than two references - the double-dot and
triple-dot shorthands cannot - and it is not something general Git familiarity reliably
reproduces.
"""

import subprocess

import pytest
from revision_ranges import (
    commits_in_three_dot,
    commits_in_two_dot,
    commits_reachable_excluding,
    diff_from_merge_base,
)


def test_two_dot_log_range() -> None:
    assert commits_in_two_dot("main", "feature") == ["log", "main..feature"]


def test_three_dot_log_range() -> None:
    assert commits_in_three_dot("main", "feature") == ["log", "main...feature"]


def test_revision_expressions_are_preserved_exactly() -> None:
    assert commits_in_two_dot("HEAD~2", "feature~1") == ["log", "HEAD~2..feature~1"]


def test_diff_uses_the_books_own_worked_example() -> None:
    assert diff_from_merge_base("master", "contrib") == ["diff", "master...contrib"]


def test_no_git_process_is_started() -> None:
    real_run = subprocess.run
    subprocess.run = lambda *a, **k: (_ for _ in ()).throw(AssertionError("git was invoked"))
    try:
        diff_from_merge_base("a", "b")
    finally:
        subprocess.run = real_run


@pytest.mark.discriminating
def test_multiple_points_excludes_with_a_caret_not_a_flag() -> None:
    """The book's own equivalence: `git log refA refB ^refC` names two include refs and one
    excluded one, using `^ref` rather than a `--not` flag."""
    result = commits_reachable_excluding(["refA", "refB"], ["refC"])
    assert result == ["log", "refA", "refB", "^refC"]


@pytest.mark.discriminating
def test_multiple_points_handles_more_than_two_include_refs() -> None:
    """The book's stated reason this syntax exists at all: "you can specify more than two
    references in your query, which you cannot do with the double-dot syntax." A function that
    only really works for one or two refs has not implemented what the book describes."""
    result = commits_reachable_excluding(["a", "b", "c"], ["d"])
    assert result == ["log", "a", "b", "c", "^d"]
