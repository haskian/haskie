"""Search: the pipelines in `flow.py`, the steps they run in `retrieval.py`, the pure folds in
`passage.py`, the full-text listing in `text.py`, and which collections a session searches in
`session.py`.

Six words for six things, each a piece of one document. In the order a search meets them:

`Chunk` (`indexing/chunk.py`)
    A cut of the markdown, made when the document was indexed: whole sentences and whole blocks,
    packed to a size the embedding model likes. It shares no text with its neighbours, and it
    ends mid-sentence only where one sentence outgrew a chunk. It is what gets embedded and
    matched. What an agent quotes is a passage built from it.

`Row` (`collection/index.py`)
    A chunk as it is stored, plus its vector and its `seq`: its 1-based position among the
    document's chunks. The parquet cache and a collection's LanceDB table hold rows.

`Hit` (`collection/index.py`)
    A chunk that matched, read back out of one collection's table. It has the same text and
    offsets, plus what only a search knows: its `score`, which collection's table matched it, and
    the absolute paths to open it with. One hit is one chunk, so it is still cut to size.

`ChunkRange` (`passage.py`)
    The hits of one document that sit next to each other (`seq`, `seq + 1`, ...), folded into
    one span. It holds offsets and line numbers but no text, because nothing has been read yet.
    It lets a search decide what to read before it pays for the read.

`Passage` (`passage.py`)
    A span widened to where a reader would stop, then read out of the document: to the line it
    sits on, or to whole sentences when the line is long. It is the first of these that carries
    text. Its `text` is `markdown[char_start:char_end]` with the page markers taken out, so it is
    a quote, never chunks stitched together.

`Excerpt` (`passage.py`)
    A passage with the parts that do not answer the question removed. Today it is the passage
    itself, unchanged. The type gives that trimming one place to land.

Between `ChunkRange` and `Passage` sits the only file read in a search. `retrieval` seeks to the
span's byte offsets and reads a few kilobytes around it, and `passage.widen` widens inside that
window.
"""
