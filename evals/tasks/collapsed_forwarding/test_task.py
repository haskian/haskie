"""Scored against `collapsed_forwarding.py` as the agent wrote it.

Grounded in "Scalable Web Architecture and Distributed Systems" (from *The Architecture of Open
Source Applications*, Vol. 2), which names "collapsed forwarding" and is explicit about what
distinguishes it from a cache: it "optimiz[es] the requests or calls," not the data itself. The
real discriminator is what happens once a fetch finishes: a plausible wrong implementation would
memoize `fetch`'s result forever - the natural first instinct for "don't call this twice" - but
that's a cache, and the chapter says this isn't one.

Every thread below is a daemon and every join has a timeout, matching the pattern used in
`bounded_buffer` and `h2o`: a proxy that gets the in-flight bookkeeping wrong can deadlock a
caller, which should fail the assertion that catches it rather than hang pytest itself.
"""

import threading
import time

import collapsed_forwarding
import pytest


def test_single_caller_gets_the_fetched_value() -> None:
    proxy = collapsed_forwarding.CollapsingProxy(lambda key: f"value-for-{key}")
    assert proxy.get("a") == "value-for-a"


@pytest.mark.discriminating
def test_concurrent_callers_for_the_same_key_share_one_fetch() -> None:
    """Not a proof: 20 threads call `get` for the same key while the backend is deliberately
    slow, on the premise that a 0.3s fetch gives plenty of room for all 20 to arrive and overlap,
    not that overlap is guaranteed on every run."""
    calls: list[str] = []
    calls_lock = threading.Lock()

    def fetch(key: str) -> str:
        with calls_lock:
            calls.append(key)
        time.sleep(0.3)
        return f"value-for-{key}"

    proxy = collapsed_forwarding.CollapsingProxy(fetch)
    results: list[str] = []
    results_lock = threading.Lock()

    def caller() -> None:
        value = proxy.get("littleB")
        with results_lock:
            results.append(value)

    threads = [threading.Thread(target=caller, daemon=True) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive(), "a caller never returned: likely deadlocked"

    assert calls == ["littleB"], f"expected exactly one fetch, backend saw: {calls}"
    assert results == ["value-for-littleB"] * 20


@pytest.mark.discriminating
def test_a_later_non_overlapping_request_fetches_again() -> None:
    """Not a cache: once every caller waiting on a fetch has been served, there is nothing left
    to share. A later, non-overlapping request for the same key calls `fetch` again."""
    calls: list[str] = []
    lock = threading.Lock()

    def fetch(key: str) -> str:
        with lock:
            calls.append(key)
        return f"value-for-{key}"

    proxy = collapsed_forwarding.CollapsingProxy(fetch)
    first = proxy.get("littleB")
    second = proxy.get("littleB")

    assert first == second == "value-for-littleB"
    assert calls == ["littleB", "littleB"], f"expected two separate fetches, backend saw: {calls}"
