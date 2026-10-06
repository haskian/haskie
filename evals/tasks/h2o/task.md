Write `h2o.py` in the current directory. Standard library only - `threading.Semaphore` and
`threading.Barrier` are both fine here; this task is not about reimplementing a barrier.

Implement "Building H2O" from The Little Book of Semaphores, section 5.6, with this interface:

    class H2O:
        def hydrogen(self, bond) -> None:
            """Call when a hydrogen thread arrives. `bond` is a zero-argument callable. Blocks
            until this thread can join a complete molecule, calls `bond()` at that point, and
            returns only once the whole molecule - two hydrogen threads and one oxygen thread -
            has bonded."""

        def oxygen(self, bond) -> None:
            """Same contract as `hydrogen`, for an oxygen thread."""

The constraint, in the book's own words: "if we examine the sequence of threads that invoke
bond and divide them into groups of three, each group should contain one oxygen and two
hydrogen threads." Threads don't need to know which specific threads they're paired with - only
that each group of three passes as a complete set before the next group starts.

The book's own solution has an unusual feature worth reading before implementing: the thread
that releases the mutex guarding the two counters is not always the same thread that acquired
it. Since there is exactly one oxygen thread in every group, using it as the one that always
releases the mutex - after all three threads have bonded - is deliberate, not an accident.
