"""Reference for `lru_cache`. `OrderedDict` gives the O(1) both ends the task asks for."""

from collections import OrderedDict


class LRU:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.held: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.held:
            return -1
        self.held.move_to_end(key)
        return self.held[key]

    def put(self, key: int, value: int) -> None:
        if key in self.held:
            self.held.move_to_end(key)
        self.held[key] = value
        if len(self.held) > self.capacity:
            self.held.popitem(last=False)
