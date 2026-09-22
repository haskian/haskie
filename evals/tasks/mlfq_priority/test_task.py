"""Scored against `mlfq.py` as the agent wrote it.

The chapter states Rules 4a and 4b, shows that they let a job game the scheduler by issuing an
I/O just before its allotment runs out, and replaces both with a single Rule 4 that counts total
time at a level. The superseded pair is the version most implementations carry, which is what
makes this task worth running.
"""

import pytest

import mlfq


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


@pytest.mark.discriminating
def test_giving_up_the_cpu_does_not_save_a_job_from_demotion() -> None:
    """Rule 4 counts the allotment "regardless of how many times it has given up the CPU". Under
    the superseded Rule 4b this would stay at priority 0 with its allotment reset."""
    assert mlfq.after_run(0, 10.0, 10.0, True, 3) == (1, 0.0)


@pytest.mark.discriminating
def test_blocking_early_is_not_a_way_to_stay_high_priority() -> None:
    """The gaming attack the chapter's figure demonstrates: issue an I/O before the allotment
    ends. With total-time accounting the job still walks down the queues."""
    priority, used = 0, 0.0
    for _ in range(3):
        used += 4.0
        priority, used = mlfq.after_run(priority, used, 10.0, True, 3)

    assert priority == 1, "three short blocking runs still add up to the allotment"
