"""Scored against `barrier.py` as the agent wrote it.

Grounded in "The Little Book of Semaphores" section 3.7 (little-book-of-semaphores.pdf), which
walks through a design that looks correct - a single turnstile, reset by the last thread to
leave - and then names its exact flaw: "a precocious thread can pass through the second [gate],
then loop around and pass through the first [gate] and the turnstile, effectively getting ahead
of the other threads by a lap." The fix the book gives is two separate turnstiles, one for
arrival and one for departure, so the arrival gate cannot reopen until every thread has left
through the departure gate.

The book also exposes the barrier as `phase1`/`phase2`, callable separately from `wait`. Requiring
that interface is a second, cheap-to-check sign the agent used the book's own design rather than
inventing one.
"""

import threading
import time

import barrier
import pytest


def test_rejects_non_positive_party_count() -> None:
    with pytest.raises(ValueError):
        barrier.Barrier(0)


def test_a_single_generation_lets_every_participant_through() -> None:
    b = barrier.Barrier(3)
    generations = []
    lock = threading.Lock()

    def worker() -> None:
        g = b.wait()
        with lock:
            generations.append(g)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)
        assert not t.is_alive()

    assert generations == [0, 0, 0]


def test_no_one_passes_before_every_participant_has_arrived() -> None:
    b = barrier.Barrier(3)
    passed = []
    lock = threading.Lock()

    def worker() -> None:
        b.wait()
        with lock:
            passed.append(1)

    early = [threading.Thread(target=worker) for _ in range(2)]
    for t in early:
        t.start()
    time.sleep(0.05)
    assert passed == [], "two of three arrived: nobody should have passed yet"

    threading.Thread(target=worker).start()
    for t in early:
        t.join(timeout=2)
        assert not t.is_alive()


def test_exposes_the_books_phase1_and_phase2_separately() -> None:
    """The book: "code that uses a barrier can call phase1 and phase2 separately, if there is
    something else that should be done in between." A bare `wait()` with no way to split it is
    a plausible barrier that was not built from this source."""
    b = barrier.Barrier(1)
    b.phase1()
    b.phase2()


@pytest.mark.discriminating
def test_a_fast_thread_cannot_lap_the_others_into_the_next_generation() -> None:
    """This is a stress test, not a proof: it runs many rapid generations and checks a strict
    ordering invariant, on the premise that the single-turnstile bug the book describes tends to
    surface within a modest number of iterations under contention, not that it is guaranteed to
    on every run.

    The invariant: every generation's cohort of `parties` arrivals is contiguous. A "precocious"
    thread lapping into generation g+1 before generation g's cohort has finished would interleave
    a value from g+1 in among g's arrivals.
    """
    parties = 4
    iterations = 300
    b = barrier.Barrier(parties)
    seen: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(iterations):
            generation = b.wait()
            with lock:
                seen.append(generation)

    threads = [threading.Thread(target=worker) for _ in range(parties)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive(), "a generation never completed: likely deadlocked"

    for start in range(0, len(seen), parties):
        cohort = seen[start : start + parties]
        assert len(set(cohort)) == 1, (
            f"generation {start // parties} was not a clean cohort: {cohort}"
        )
