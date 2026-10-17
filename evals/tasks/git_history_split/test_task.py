"""Scored against `git_history_split.py` as the agent wrote it.

Grounded in git's own `Documentation/git-history.adoc` (indexed here as
`git-history-split.txt`), added to git in 2026 for the new, still-experimental `git history
split` command. This document did not exist during this suite's model's training - the command
is explicitly marked "EXPERIMENTAL. THE BEHAVIOR MAY CHANGE." - so passing the discriminating
tests below cannot come from general git familiarity, only from an actual search of the corpus.

The real discriminator is the doc's own line on `--update-refs`: "Defaults to `branches`." A
guess without the source has two plausible failure modes, both caught below: assuming the safer-
sounding default is to touch only `HEAD` (`test_split_by_default_needs_no_flags`), and guessing
at the flag's spelling for restricting to `HEAD` when it *is* wanted
(`test_restricting_to_head_needs_the_exact_flag_spelling`).
"""

import git_history_split
import pytest


def test_the_git_subcommand_is_two_words() -> None:
    assert git_history_split.split_commit("abc123")[:2] == ["history", "split"]


def test_dry_run_previews_instead_of_updating() -> None:
    assert git_history_split.split_commit_dry_run("abc123") == [
        "history",
        "split",
        "abc123",
        "--dry-run",
    ]


def test_pathspec_is_limited_by_a_double_dash() -> None:
    assert git_history_split.split_commit_limited_to("abc123", ["a.py", "b.py"]) == [
        "history",
        "split",
        "abc123",
        "--",
        "a.py",
        "b.py",
    ]


@pytest.mark.discriminating
def test_split_by_default_needs_no_flags() -> None:
    """The plain, no-argument split already rewrites every descendant branch - that is the
    documented default - so a correct `split_commit` adds no `--update-refs` flag at all. A
    plausible wrong guess would add `--update-refs=head` here on the assumption that touching
    only HEAD is the conservative, presumably-default choice."""
    assert git_history_split.split_commit("abc123") == ["history", "split", "abc123"]


@pytest.mark.discriminating
def test_restricting_to_head_needs_the_exact_flag_spelling() -> None:
    """The doc's exact flag: `--update-refs=(branches|head)`. A plausible wrong guess -
    `--head-only`, `--only-head`, a separate boolean `--head` flag - would not match this."""
    assert git_history_split.split_commit_head_only("abc123") == [
        "history",
        "split",
        "abc123",
        "--update-refs=head",
    ]
