"""A document's sections: each with its id, where it runs, and its descriptors.

A section is a run of the document's chunks under one heading path: the whole document at depth
0, then every heading path's contiguous run at its depth, by the same rule the search groups
passages into sections by (`search.section`), so a section a search names is a section here. A
path that comes back after another section opened is two sections. The sections form a tree:
each names the one it sits in (`parent_id`), and each chunk the sections that hold it, outermost
first (`chunks`).

Ids. A section's id is the MD5 of `<document id>/s/<position>`, its place among the document's
sections, 0 for the whole document. A chunk's id is the MD5 of `<document id>/c/<seq>`, its 1-based
place among the document's chunks. Both name a place in one chunking of the document: sections are
cut by heading paths alone, so other chunk settings mostly give the same sections the same ids,
while a chunk's `seq` names another text under other settings. The `s` and `c` keep the two kinds
apart, so section 2 and chunk 2 of one document are two ids.

Sections are named, and described, once per cached embedding (`embed_cache.write`), which sees
the whole document in order. Descriptors: a `descriptors.Strategy` picks them,
`descriptors.CLASS_TFIDF` unless the caller names another. The strategy reads each chunk's prose
only (`prose`): code blocks and tables name things (identifiers, column headers, values) rather
than say what a section is about. With a model it reranks against each section's vector: the mean
of its chunks' unit vectors, scaled to length one, so a long chunk weighs no more than a short one.

No IO here: the caller passes the function that embeds.
"""

from collections.abc import Sequence
from itertools import groupby

import msgspec
import numpy as np

from haskie import ids
from haskie.indexing.chunk import HEADING_SEP, Chunk
from haskie.indexing.segment import PieceType
from haskie.search.passage import pages
from haskie.sections import descriptors


class Section(msgspec.Struct, frozen=True):
    """One section of a document: which it is, where it sits in the tree, where it runs, by the
    fields a `Chunk` names them with, and what it is about."""

    id: str
    parent_id: str | None  # the section it sits in; None for the whole document
    headings: list[str]  # the headings it sits under, outermost first; [] for the whole document
    seq_start: int  # its first and last chunk, 1-based, in the chunking it was named from
    seq_end: int
    line_start: int  # 1-based, inclusive
    line_end: int
    char_start: int  # 0-based offsets into the markdown, page markers included
    char_end: int
    byte_start: int  # the same span in bytes, what the markdown file can be seeked to
    byte_end: int
    page_start: int | None  # 1-based PDF pages; None for a document without pages
    page_end: int | None
    descriptors: list[str] = []  # what it is about, as written, best first (`describe`)

    @property
    def depth(self) -> int:
        return len(self.headings)

    @property
    def header(self) -> str:
        return HEADING_SEP.join(self.headings)


def runs(paths: Sequence[Sequence[str]]) -> list[tuple[tuple[str, ...], int, int]]:
    """Every section given each chunk's heading path in document order, as (path, first chunk,
    last chunk) positions into `paths`, in document order, a parent before its children."""
    if not paths:
        return []
    found = [((), 0, len(paths) - 1)]
    for depth in range(1, max(len(path) for path in paths) + 1):
        at = 0
        prefixes = (tuple(path[:depth]) if len(path) >= depth else None for path in paths)
        for prefix, run in groupby(prefixes):
            size = sum(1 for _ in run)
            if prefix is not None:
                found.append((prefix, at, at + size - 1))
            at += size
    return sorted(found, key=lambda run: (run[1], len(run[0])))


def sections(document_id: str, chunks: Sequence[Chunk]) -> tuple[list[Section], list[list[int]]]:
    """Every section of a document from its chunks in `seq` order, and for each chunk the
    positions of the sections that hold it, the whole document first and its deepest last (see
    the module)."""
    found = runs([chunk.headings for chunk in chunks])
    named: list[Section] = []
    chains: list[list[int]] = [[] for _ in chunks]
    last_at_depth: dict[int, int] = {}  # the position of the latest section of each depth
    for position, (path, first, last) in enumerate(found):
        held = chunks[first : last + 1]
        start, end = held[0].char_start, held[-1].char_end
        parent = last_at_depth.get(len(path) - 1) if path else None
        page_start, page_end = pages(held)
        named.append(
            Section(
                id=_hashed(f"{document_id}/s/{position}"),
                parent_id=None if parent is None else named[parent].id,
                headings=list(path),
                seq_start=first + 1,
                seq_end=last + 1,
                line_start=held[0].line_start,
                line_end=held[-1].line_end,
                char_start=start,
                char_end=end,
                byte_start=held[0].byte_start,
                byte_end=held[-1].byte_end,
                page_start=page_start,
                page_end=page_end,
            )
        )
        last_at_depth[len(path)] = position
        # sorted by first chunk, then depth: each chunk meets its sections outermost first
        for chain in chains[first : last + 1]:
            chain.append(position)
    return named, chains


def chunk_id(document_id: str, seq: int) -> str:
    """The id of the document's chunk `seq` (see the module)."""
    return _hashed(f"{document_id}/c/{seq}")


def describe(
    found: Sequence[Section],
    texts: Sequence[str],
    vectors: np.ndarray | None,
    embed: descriptors.Embed | None,
    strategy: descriptors.Strategy = descriptors.CLASS_TFIDF,
) -> list[Section]:
    """One document's sections, each with its descriptors. `texts` holds each chunk's `prose` in
    `seq` order, `vectors` each section's unit vector, None without a model."""
    picked = strategy.pick(
        texts,
        [descriptors.Run(tuple(one.headings), one.seq_start - 1, one.seq_end - 1) for one in found],
        vectors,
        embed,
    )
    return [
        msgspec.structs.replace(one, descriptors=words)
        for one, words in zip(found, picked, strict=True)
    ]


NOT_PROSE = frozenset({PieceType.CODE, PieceType.TABLE})


def prose(chunk: Chunk) -> str:
    """The chunk's text without its code blocks and tables, each left as a blank line, so no
    pair of words spans one (`descriptors.terms`)."""
    return "".join("\n\n" if piece.type in NOT_PROSE else piece.text for piece in chunk.pieces)


def _hashed(text: str) -> str:
    return ids.md5(text.encode())
