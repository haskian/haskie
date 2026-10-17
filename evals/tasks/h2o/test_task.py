"""Scored against `h2o.py` as the agent wrote it.

Grounded in "The Little Book of Semaphores" section 5.6 (little-book-of-semaphores.pdf), a much
less commonly discussed puzzle than producer-consumer or the dining philosophers - the kind of
material the earlier, "classical" tasks in this suite turned out not to discriminate on, because
this model already knew them cold. The book's own solution has a genuinely counter-intuitive
feature it calls out explicitly: the thread that releases the shared mutex is not always the one
that acquired it - "there is no rule that says a thread has to hold a lock in order to release
it." That, plus the exact 2:1 hydrogen:oxygen signaling scheme, is not something general
concurrency familiarity reliably reproduces.

Every thread below is a daemon and every join has a timeout, for the same reason as in
`bounded_buffer`: a wrong mutex hand-off can deadlock in a way that would otherwise hang pytest
itself rather than just fail the assertion that caught it.
"""

import threading

import h2o
import pytest


def test_a_single_molecule_bonds_all_three_threads() -> None:
    builder = h2o.H2O()
    bonded: list[str] = []
    lock = threading.Lock()

    def record(label: str) -> None:
        with lock:
            bonded.append(label)

    threads = [
        threading.Thread(target=builder.hydrogen, args=(lambda: record("H"),), daemon=True),
        threading.Thread(target=builder.hydrogen, args=(lambda: record("H"),), daemon=True),
        threading.Thread(target=builder.oxygen, args=(lambda: record("O"),), daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)
        assert not t.is_alive(), "a thread never returned: likely deadlocked"

    assert sorted(bonded) == ["H", "H", "O"]


@pytest.mark.discriminating
def test_high_contention_every_triple_is_one_oxygen_two_hydrogen() -> None:
    """Not a proof: many individual hydrogen and oxygen threads - not a handful of looping
    workers - are all started together, so there is always a pool of threads of both types
    genuinely competing to enter the next group of three. (A pool of exactly one oxygen-loop and
    two hydrogen-loops would trivially force balanced triples through *any* 3-way barrier, type-
    blind or not, since there would be no fourth thread that could ever be admitted out of turn -
    that shape doesn't actually exercise the constraint.) The full sequence of `bond()` calls is
    checked for the book's invariant - every consecutive group of three is exactly one oxygen,
    two hydrogen. A wrong pairing scheme, a wrong signal count, or a broken mutex hand-off tends
    to produce an unbalanced group or a deadlock well within this many molecules, not that it is
    guaranteed to on every run.
    """
    molecules = 100
    builder = h2o.H2O()
    bonded: list[str] = []
    lock = threading.Lock()

    def record(label: str) -> None:
        with lock:
            bonded.append(label)

    threads = [
        threading.Thread(target=builder.hydrogen, args=(lambda: record("H"),), daemon=True)
        for _ in range(2 * molecules)
    ] + [
        threading.Thread(target=builder.oxygen, args=(lambda: record("O"),), daemon=True)
        for _ in range(molecules)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
        assert not t.is_alive(), "a thread never finished: likely deadlocked"

    assert len(bonded) == 3 * molecules
    for start in range(0, len(bonded), 3):
        group = bonded[start : start + 3]
        assert sorted(group) == ["H", "H", "O"], f"molecule {start // 3} was unbalanced: {group}"
