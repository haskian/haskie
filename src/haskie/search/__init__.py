"""Search: the pipelines in `flow.py`, the steps they run in `retrieval.py`, the pure folds in
`passage.py`, lexical listing in `text.py`, and which collections a session searches.

Five words for five things, all of them a piece of one document. In the order a search meets them:

`Chunk` (`indexing/chunk.py`)
    A cut of the markdown, made when the document was indexed. Cut to a size the embedding model
    likes, overlapping its neighbours, so it starts and ends wherever the size ran out - usually
    mid-sentence. Never shown to anyone: it is what gets embedded and matched, not what gets read.

`Row` (`collection/index.py`)
    A chunk as it is stored: plus its vector and its `seq`, the 1-based position among the
    document's chunks. What the parquet cache and a collection's LanceDB table hold.

`Hit` (`collection/index.py`)
    A chunk that matched, read back out of one collection's table. The same text and offsets, plus
    what only a search knows: the `score` it got, which collection's table matched it, and the
    absolute paths to open it with. One hit is one chunk, so it is still cut on size.

`ChunkRange` (`passage.py`)
    The hits of one document that sit next to each other (`seq`, `seq + 1`, ...), folded into one
    span. Still in index coordinates - offsets and line numbers, no text - because nothing has
    been read yet. It exists so a search can decide what to read before paying for it.

`Passage` (`passage.py`)
    A span widened to where a reader would stop - the line it sits on, or the whole sentences
    around it - and read out of the document. The first of these that carries text, and the first
    anyone outside sees. `markdown[char_start:char_end]` is exactly its `text`, so it is a quote,
    never a stitching-together of chunks.

`Excerpt` (`passage.py`)
    A passage with the parts that do not answer the question removed. Today the passage itself,
    unchanged; the type exists so the trimming has one place to land.

The line between `ChunkRange` and `Passage` is the only file read in a search: `retrieval` seeks
to the span's byte offsets, reads a few kilobytes around it, and `passage.expand` widens inside
that window.
"""
