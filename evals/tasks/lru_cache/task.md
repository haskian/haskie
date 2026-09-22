Write `lru.py` in the current directory, defining:

    class LRU:
        """A fixed-capacity least-recently-used cache. `get` and `put` are both O(1)."""

        def __init__(self, capacity: int) -> None: ...

        def get(self, key: int) -> int:
            """The value, or -1 if the key is not held. A hit counts as a use."""

        def put(self, key: int, value: int) -> None:
            """Insert or update. At capacity, evict the least recently used key first."""

Standard library only. Write the file in this session; do not stop to ask.
