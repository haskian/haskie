"""Scored against `raft_election.py` as the agent wrote it.

Grounded in the Raft paper (raft.pdf), section 5.4.1: "Raft determines which of two logs is more
up-to-date by comparing the index and term of the last entries in the logs. If the logs have
last entries with different terms, then the log with the later term is more up-to-date. If the
logs end with the same term, then whichever log is longer is more up-to-date."

The plausible wrong guess this paper explicitly rules out is comparing log *length* first - "my
log has more entries, so it must be more current." The paper's actual rule compares *term*
first, and only falls back to length as a tiebreaker when the terms match. Raft is widely enough
implemented and taught that this rule may already be common knowledge for a strong model, in
which case this task may not discriminate the way `git_history_split` or
`github_api_breaking_changes` do - unlike those, nothing here postdates any model's training.
"""

import pytest
import raft_election


def test_identical_logs_are_equally_up_to_date() -> None:
    assert raft_election.grant_vote(2, 5, 2, 5) is True


@pytest.mark.discriminating
def test_a_higher_term_wins_even_with_a_much_shorter_log() -> None:
    """The candidate's log is barely started, but from a later term - term is compared first."""
    assert raft_election.grant_vote(2, 100, 3, 1) is True


@pytest.mark.discriminating
def test_a_lower_term_loses_even_with_a_much_longer_log() -> None:
    """The mirror case: the candidate's log is long, but from an earlier term than the voter's -
    length never overrides a strictly lower term."""
    assert raft_election.grant_vote(3, 1, 2, 100) is False


def test_same_term_a_longer_candidate_log_is_more_up_to_date() -> None:
    assert raft_election.grant_vote(2, 5, 2, 10) is True


def test_same_term_a_shorter_candidate_log_is_not_as_up_to_date() -> None:
    assert raft_election.grant_vote(2, 10, 2, 5) is False
