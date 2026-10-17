Write `barrier.py` in the current directory. Standard library only - use `threading.Semaphore`,
but not `threading.Barrier`.

Implement the reusable barrier from "The Little Book of Semaphores", section 3.7, with this
interface:

    class Barrier:
        def __init__(self, parties: int) -> None: ...

        def phase1(self) -> None:
            """Block until all `parties` threads have called `phase1` for this generation."""

        def phase2(self) -> int:
            """Block until all `parties` threads have called `phase2` for this generation, then
            return the generation number (0 for the first, 1 for the second, ...) - the same
            value for every thread in that generation."""

        def wait(self) -> int:
            """`phase1()` then `phase2()`, returning what `phase2()` returns."""

`Barrier(n)` is for exactly `n` participating threads, non-positive raises `ValueError`. The
barrier must work correctly across repeated use: the same `n` threads calling `wait()` (or
`phase1`/`phase2`) again for a second, third, ... generation, in a loop.

The book's own text walks through an attempt that looks correct but has a subtle bug: "a
precocious thread" can pass through, loop around, and get ahead of the others "by a lap" before
they have all left. Use the book's actual fix for this, not the first design that occurs to you.
The book also notes that code using a barrier can call `phase1`/`phase2` separately when
something else needs to happen in between them - that is why they are exposed here rather than
folded entirely into `wait`.
