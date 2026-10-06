Write `raft_election.py` in the current directory. Standard library only.

Implement the log-comparison rule from "In Search of an Understandable Consensus Algorithm
(Extended Version)" (the Raft paper), section 5.4.1, "Election restriction":

    def grant_vote(
        voter_last_log_term: int,
        voter_last_log_index: int,
        candidate_last_log_term: int,
        candidate_last_log_index: int,
    ) -> bool:
        """Whether a voter, comparing only the two logs (ignoring term bookkeeping and whether
        it already voted this term), would consider the candidate's log at least as up-to-date
        as its own - the log-comparison half of Raft's voting restriction."""

The paper gives its own precise definition of "more up-to-date" in terms of comparing the last
entries of the two logs - not simply whichever log is longer. Use the paper's own rule, in the
order it states the comparison.
