Write `collapsed_forwarding.py` in the current directory. Standard library only.

Implement "collapsed forwarding" as described in "Scalable Web Architecture and Distributed
Systems" (from *The Architecture of Open Source Applications*, Volume 2): a proxy that folds
concurrent requests for the same piece of data into a single request to the (slow) backend,
then hands the one result back to every caller that was waiting on it.

    class CollapsingProxy:
        def __init__(self, fetch) -> None:
            """`fetch(key)` is the expensive backend call - reading a piece of data off disk,
            in the chapter's own example. It may be slow, and it may be called concurrently
            from multiple threads for different keys."""

        def get(self, key):
            """Return the value for `key`. If a fetch for this same key is already in flight
            when this is called, join that fetch and share its result rather than starting a
            second one."""

The chapter is explicit about what this is *not*: "This is similar to a cache, but instead of
storing the data/document like a cache, it is optimizing the requests or calls for those
documents." A `CollapsingProxy` only folds together requests that genuinely overlap in time -
it does not remember `fetch`'s result once every caller waiting on it has been served, and a
later, non-overlapping request for the same key calls `fetch` again.
