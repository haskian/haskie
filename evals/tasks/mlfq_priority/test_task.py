"""Scored against `mlfq.py` as the agent wrote it.

Grounded in "Scheduling: The Multi-Level Feedback Queue" (ostep-cpu-sched-mlfq.pdf), which walks
through an early draft - Rule 4a (using the allotment demotes) plus Rule 4b (blocking resets the
allotment) - shows a job gaming that pair by issuing an I/O just before its allotment runs out,
and then replaces both with a single rule:

    "Once a job uses up its time allotment at a given level (regardless of how many times it
    has given up the CPU), its priority is reduced."

The superseded 4a/4b pair is the version most people reconstruct from memory. The two tests
marked below are the ones that fail against that superseded version and pass only against the
chapter's final rule, so they are what tells a run that read the chapter apart from one that
guessed at "how MLFQ generally works".
"""

import mlfq
import pytest


def test_a_new_job_starts_at_the_top() -> None:
    assert mlfq.on_enter(3) == 0


def test_the_boost_moves_everything_to_the_topmost_queue() -> None:
    assert mlfq.boost([0, 1, 2, 2]) == [0, 0, 0, 0]


def test_a_job_below_its_allotment_keeps_its_priority() -> None:
    assert mlfq.after_run(1, 5.0, 10.0, False, 3) == (1, 5.0)


def test_a_job_at_the_bottom_queue_stays_there() -> None:
    priority, _ = mlfq.after_run(2, 10.0, 10.0, False, 3)
    assert priority == 2


def test_using_the_allotment_demotes_and_resets_the_count() -> None:
    assert mlfq.after_run(0, 10.0, 10.0, False, 3) == (1, 0.0)


def test_giving_up_the_cpu_does_not_by_itself_change_anything() -> None:
    """Not the discriminator on its own - a job with time left on its allotment keeps its
    priority whether or not it just blocked. The discriminator is the next test, where the
    allotment has actually run out."""
    assert mlfq.after_run(1, 5.0, 10.0, True, 3) == (1, 5.0)


@pytest.mark.discriminating
def test_giving_up_the_cpu_does_not_save_a_job_from_demotion() -> None:
    """Under the superseded Rule 4b this would stay at priority 0 with its allotment reset,
    because the job blocked before nominally "using up" the allotment in one sitting."""
    assert mlfq.after_run(0, 10.0, 10.0, True, 3) == (1, 0.0)


@pytest.mark.discriminating
def test_blocking_repeatedly_is_not_a_way_to_stay_at_the_top() -> None:
    """The gaming attack the chapter's own figure demonstrates: issue an I/O just before the
    allotment ends, repeatedly. Three short blocking runs still add up to a full allotment under
    the final rule; under 4a/4b the job would never be demoted."""
    priority, used = 0, 0.0
    for _ in range(3):
        used += 4.0
        priority, used = mlfq.after_run(priority, used, 10.0, True, 3)
    assert priority == 1
