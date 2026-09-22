"""Scored against `lru.py` as the agent wrote it.

In-model, but the corpus overlaps it topically: the OSTEP virtual-memory chapter is all about LRU
replacement, so a search for "LRU" does hit the library. That makes this the sharper of the two
controls - not searching here is a choice, not an absence of anything to find.
"""

import lru


def test_a_miss_is_minus_one() -> None:
    assert lru.LRU(2).get(1) == -1


def test_what_was_put_comes_back() -> None:
    cache = lru.LRU(2)
    cache.put(1, 100)

    assert cache.get(1) == 100


def test_the_least_recently_used_key_goes_first() -> None:
    cache = lru.LRU(2)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.put(3, 3)

    assert cache.get(1) == -1
    assert cache.get(2) == 2
    assert cache.get(3) == 3


def test_a_read_counts_as_a_use() -> None:
    cache = lru.LRU(2)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.get(1)
    cache.put(3, 3)

    assert cache.get(1) == 1, "reading key 1 made key 2 the least recently used"
    assert cache.get(2) == -1


def test_updating_a_key_refreshes_it_rather_than_growing_the_cache() -> None:
    cache = lru.LRU(2)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.put(1, 10)
    cache.put(3, 3)

    assert cache.get(1) == 10
    assert cache.get(2) == -1
