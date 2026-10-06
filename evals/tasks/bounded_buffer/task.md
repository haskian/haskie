Write `bounded_buffer.py` in the current directory. Standard library only - use
`threading.Semaphore`, not `queue.Queue` or any other ready-made concurrent container.

Implement the finite-buffer producer-consumer solution from "The Little Book of Semaphores",
section 4.1, with this interface:

    class BoundedBuffer:
        def __init__(self, size: int) -> None: ...

        def put(self, item) -> None:
            """Block until there is room, then add `item`."""

        def get(self):
            """Block until an item is available, then remove and return one."""

The book first tries checking the count directly - "if items >= bufferSize: block()" - and
rejects it: "we can't check the current value of a semaphore; the only operations are wait and
signal." Its actual fix adds a second counting semaphore, `spaces`, tracking free slots,
alongside `items` (occupied slots) and a `mutex` guarding the buffer itself.

Earlier in the same section the book also shows a "broken consumer solution" that waits on a
counting semaphore from inside the mutex, and names exactly why that's wrong: "any time you wait
for a semaphore while holding a mutex, there is a danger of deadlock." Use the book's actual
ordering instead - acquire the relevant counting semaphore (`spaces` for `put`, `items` for
`get`) before the mutex, and release the mutex before signaling the other counting semaphore.
