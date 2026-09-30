"""A document's outline: every section as a node of a tree, each with its keywords.

A node is a run of the document's chunks under one heading path: the whole document at depth 0,
then every heading path's contiguous run at its depth, by the same rule the search groups
passages into sections by (`search.section`), so a section a search names is a node here. A path
that comes back after another section opened is two nodes, as it is two sections.

Keywords: a `keywords.Strategy` picks them, `keywords.CLASS_TFIDF` unless the caller names
another. Each node also gets a vector when a model embedded the chunks: the mean of its chunks'
unit vectors, scaled to length one, so a long chunk weighs no more than a short one. The strategy
reranks against it, and the outline index stores it (`outline.store`).

Built once per document and model (`embed_cache.build_outline`, from
`workflows.ensure_embedding`), from the first cache entry found or written under the model. No IO
here: the caller passes the function that embeds.
"""

from collections.abc import Sequence
from itertools import groupby

import msgspec
import numpy as np

from haskie.collection.index import Row
from haskie.indexing.chunk import HEADING_SEP
from haskie.outline import keywords
from haskie.search.collapse import unit_rows
from haskie.search.passage import pages


class Node(msgspec.Struct, frozen=True):
    """One section of a document: where it runs, by the fields a `Chunk` names them with, and
    what it is about."""

    headings: list[str]  # the headings it sits under, outermost first; [] for the whole document
    line_start: int  # 1-based, inclusive
    line_end: int
    char_start: int  # 0-based offsets into the markdown, page markers included
    char_end: int
    byte_start: int  # the same span in bytes, what the markdown file can be seeked to
    byte_end: int
    page_start: int | None  # 1-based PDF pages; None for a document without pages
    page_end: int | None
    # what it is about, as written, best first, with how often it uses each: the counts are what
    # a search weighs one section's keywords against the others' by (`search.overview.distinct`)
    keywords: dict[str, int]

    @property
    def depth(self) -> int:
        return len(self.headings)

    @property
    def header(self) -> str:
        return HEADING_SEP.join(self.headings)


def runs(paths: Sequence[Sequence[str]]) -> list[tuple[tuple[str, ...], int, int]]:
    """Every node of an outline given each chunk's heading path in document order, as (path,
    first chunk, last chunk) positions into `paths`, in document order, a parent before its
    children."""
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


def describe(
    rows: list[Row],
    vectors: np.ndarray | None,
    embed: keywords.Embed | None,
    strategy: keywords.Strategy = keywords.CLASS_TFIDF,
) -> tuple[list[Node], np.ndarray | None]:
    """The outline of one document from its cached rows in `seq` order, and their vectors, one
    row each, when a model embedded them; with each node's unit vector, one row per node, None
    without chunk vectors (see the module)."""
    found = runs([row.chunk.headings for row in rows])
    pooled = None if vectors is None else _pooled(vectors, found)
    picked = strategy.pick(
        [row.chunk.text for row in rows],
        [keywords.Run(len(path), first, last) for path, first, last in found],
        pooled,
        embed,
    )
    nodes = [
        _node(rows[first : last + 1], list(path), words)
        for (path, first, last), words in zip(found, picked, strict=True)
    ]
    return nodes, pooled


def _pooled(vectors: np.ndarray, found: list[tuple[tuple[str, ...], int, int]]) -> np.ndarray:
    """Each run's unit vector: the mean of its chunks' unit vectors, scaled to length one."""
    if not found:  # a document with no chunk: `unit_rows` needs a matrix
        return np.empty((0, vectors.shape[1]))
    chunks = unit_rows(vectors)
    return unit_rows([chunks[first : last + 1].mean(axis=0) for _, first, last in found])


def _node(rows: list[Row], headings: list[str], found: dict[str, int]) -> Node:
    first, last = rows[0].chunk, rows[-1].chunk
    page_start, page_end = pages([row.chunk for row in rows])
    return Node(
        headings=headings,
        line_start=first.line_start,
        line_end=last.line_end,
        char_start=first.char_start,
        char_end=last.char_end,
        byte_start=first.byte_start,
        byte_end=last.byte_end,
        page_start=page_start,
        page_end=page_end,
        keywords=found,
    )
