"""Scored against `bounded_buffer.py` as the agent wrote it.

Grounded in "The Little Book of Semaphores" section 4.1 (little-book-of-semaphores.pdf). The
book walks through a naive finite-buffer attempt that checks the item count directly, rejects it
("we can't check the current value of a semaphore"), and gives the real fix: a second counting
semaphore (`spaces`) alongside `items` and `mutex`. It separately names the deadlock risk of
waiting on a counting semaphore while already holding the mutex - the order the first three
tests below don't test directly, but that the discriminating stress test is built to catch: an
implementation that gets the ordering wrong tends to deadlock or lose items under contention
long before it fails a single-threaded check.

Every thread below is a daemon and every join has a timeout: a wrong lock/semaphore ordering can
deadlock a call that looks unrelated to the one under test, not just the one actually being
waited on, and a stuck non-daemon thread would otherwise hang pytest itself at interpreter exit
instead of just failing the one assertion that caught it.
"""

import threading

import bounded_buffer
import pytest


def test_put_then_get_returns_the_item() -> None:
    b = bounded_buffer.BoundedBuffer(3)
    b.put("x")
    assert b.get() == "x"


def test_get_blocks_until_something_is_put() -> None:
    b = bounded_buffer.BoundedBuffer(1)
    result = []

    def consumer() -> None:
        result.append(b.get())

    t = threading.Thread(target=consumer, daemon=True)
    t.start()
    t.join(timeout=0.1)
    assert result == [], "buffer is empty: get() should still be blocked"

    putter = threading.Thread(target=b.put, args=("late",), daemon=True)
    putter.start()
    putter.join(timeout=2)
    assert not putter.is_alive(), "put() into a non-full buffer should not itself block"

    t.join(timeout=2)
    assert not t.is_alive(), "consumer thread never returned: likely deadlocked"
    assert result == ["late"]


def test_put_blocks_when_the_buffer_is_full() -> None:
    b = bounded_buffer.BoundedBuffer(1)

    filler = threading.Thread(target=b.put, args=("first",), daemon=True)
    filler.start()
    filler.join(timeout=2)
    assert not filler.is_alive(), "put() into an empty buffer should not itself block"

    done = []

    def producer() -> None:
        b.put("second")
        done.append(True)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    t.join(timeout=0.1)
    assert done == [], "buffer is full: put() should still be blocked"

    got = []
    getter = threading.Thread(target=lambda: got.append(b.get()), daemon=True)
    getter.start()
    getter.join(timeout=2)
    assert not getter.is_alive(), "get() on a non-empty buffer should not itself block"
    assert got == ["first"]

    t.join(timeout=2)
    assert not t.is_alive(), "producer thread never returned: likely deadlocked"
    assert done == [True]

    got2 = []
    getter2 = threading.Thread(target=lambda: got2.append(b.get()), daemon=True)
    getter2.start()
    getter2.join(timeout=2)
    assert not getter2.is_alive()
    assert got2 == ["second"]


@pytest.mark.discriminating
def test_high_contention_survives_without_deadlock_or_lost_items() -> None:
    """Not a proof, a stress test: many producers and consumers hammering a buffer with only two
    slots, on the premise that a wrong semaphore ordering (checking before acquiring the mutex,
    or the reverse of the book's acquire/release order) tends to surface as a deadlock or a lost
    item well within this many iterations, not that it is guaranteed to on every run.
    """
    producers, consumer_count, per_producer = 4, 4, 200
    total = producers * per_producer
    b = bounded_buffer.BoundedBuffer(2)
    produced = [f"{p}-{i}" for p in range(producers) for i in range(per_producer)]
    consumed: list[str] = []
    lock = threading.Lock()

    def produce(items: list[str]) -> None:
        for item in items:
            b.put(item)

    def consume(n: int) -> None:
        for _ in range(n):
            item = b.get()
            with lock:
                consumed.append(item)

    chunks = [produced[i::producers] for i in range(producers)]
    per_consumer = total // consumer_count
    threads = [threading.Thread(target=produce, args=(chunk,), daemon=True) for chunk in chunks]
    threads += [
        threading.Thread(target=consume, args=(per_consumer,), daemon=True)
        for _ in range(consumer_count)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
        assert not t.is_alive(), "a thread never finished: likely deadlocked"

    assert sorted(consumed) == sorted(produced), "items were lost or duplicated"
