"""Search: the pipelines in `flow.py`, the steps they run in `retrieval.py`, the pure folds in
`passage.py` and `collapse.py` (near-duplicates folded into `also_in`), the full-text listing in
`text.py`, and which collections a session searches in `session.py`.

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

`HitRange` (`passage.py`)
    The hits of one section that sit next to each other (`seq`, `seq + 1`, ...), folded into
    one range: a heading between two hits ends it. It holds offsets and line numbers but no
    text, because nothing has been read yet. It lets a search decide what to read before it pays
    for the read.

`Passage` (`passage.py`)
    A range read out of the document by its own offsets. Chunks are cut at headings, blank lines,
    blocks and sentences, so it starts and ends where the author stopped. It is the first of these
    that carries text. Its `text` is `markdown[char_start:char_end]` with the page markers taken
    out, so it is a quote, never chunks stitched together.

`Excerpt` (`passage.py`, built in `section.py`)
    One section of a document with every passage the search kept in it, in document order, each
    listed as a `Span`: where it is and how it matched. Its text is the passages joined under
    their headings, with `[…]` where the document skips text.

Between `HitRange` and `Passage` or `Excerpt` sits the only file read in a search. `retrieval`
seeks to each range's byte offsets and reads exactly its bytes.
"""
