"""Chunk markdown with semantic-text-splitter (Rust); attach line range and heading ancestry.

`CHUNK_VERSION` is part of the embedding cache key (`embed_cache.Params`): a cached set of chunks
is only reusable while this module splits the same text the same way. Bump it with any change
here that alters the output for unchanged input and settings, or the cache keeps serving chunks
the current code would no longer produce.
"""

import re
from bisect import bisect_right
from typing import Any

import msgspec
from semantic_text_splitter import MarkdownSplitter, TextSplitter

from haskie import render
from haskie.convert import PAGE_MARKER
from haskie.settings import ChunkSettings

CHUNK_VERSION = 1  # see the module docstring

NEWLINE = re.compile(r"\n")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
TRAILING_COMMENTS = re.compile(r"(\s*<!--.*?-->\s*)+$", re.DOTALL)


class Chunk(msgspec.Struct):
    heading: str
    text: str
    line_start: int  # 1-based, inclusive
    line_end: int  # 1-based, inclusive
    char_start: int  # 0-based offsets into the full markdown
    char_end: int
    parents: list[str]  # enclosing headings, outermost first (excludes `heading`)
    page_start: int | None = None  # 1-based PDF pages, from page markers; None for non-PDF
    page_end: int | None = None


def record(
    chunk: Chunk, vector: list[float] | None, dims: int | None, **columns: Any
) -> dict[str, Any]:
    """One Arrow record for a chunk: its own fields, plus the columns the table adds around it.

    Both tables that hold chunks build their rows here - the parquet cache (`embed_cache`) and a
    collection's LanceDB table (`index`) - so a new field on `Chunk` reaches both. `dims` is the
    width of the table's vector column, or None for a table without one; pyarrow would write a
    null for a missing vector, so a row without one is refused here instead.
    """
    values = msgspec.to_builtins(chunk) | columns
    if dims is not None:
        if vector is None:
            raise ValueError("the table has a vector column but the row carries no vector")
        values["vector"] = vector
    return values


def _heading_ancestry(text: str) -> tuple[list[int], list[tuple[str, list[str]]]]:
    """Char offset of every heading plus (heading, parents) for it. One incremental decode
    turns pyromark's byte offsets into char offsets."""
    data = text.encode()
    offsets: list[int] = []
    ancestry: list[tuple[str, list[str]]] = []
    stack: list[render.Heading] = []
    byte_pos = char_pos = 0
    for mark in render.headings(text):  # already in document order
        char_pos += len(data[byte_pos : mark.offset].decode(errors="ignore"))
        byte_pos = mark.offset
        while stack and stack[-1].level >= mark.level:
            stack.pop()
        offsets.append(char_pos)
        ancestry.append((mark.text, [h.text for h in stack]))
        stack.append(mark)
    return offsets, ancestry


def split(
    text: str, settings: ChunkSettings, line_offset: int = 0, char_offset: int = 0
) -> list[Chunk]:
    """`line_offset` / `char_offset` = lines / chars preceding `text` in the full document
    (batched indexing)."""
    splitter_cls = MarkdownSplitter if settings.chunker == "markdown" else TextSplitter
    splitter = splitter_cls(settings.chunk_size, overlap=settings.chunk_overlap)
    offsets, ancestry = _heading_ancestry(text)
    newlines = [m.start() for m in NEWLINE.finditer(text)]
    markers = [(m.start(), int(m.group(1))) for m in PAGE_MARKER.finditer(text)]
    marker_offsets = [m[0] for m in markers]

    def line_at(pos: int) -> int:
        return line_offset + bisect_right(newlines, pos - 1) + 1

    def page_at(pos: int) -> int | None:
        idx = bisect_right(marker_offsets, pos) - 1
        return markers[idx][1] if idx >= 0 else None

    chunks: list[Chunk] = []
    for start, body in splitter.chunk_indices(text):
        if not HTML_COMMENT.sub("", body).strip():
            continue  # a lone page marker is not content
        end = start + len(body)
        idx = bisect_right(offsets, start) - 1
        if idx < 0 and offsets and offsets[0] < end:
            idx = 0  # no heading before the chunk but one inside it (e.g. after a page marker)
        heading, parents = ancestry[idx] if idx >= 0 else ("", [])
        chunks.append(
            Chunk(
                heading=heading,
                text=body,
                line_start=line_at(start),
                line_end=line_at(end),
                char_start=char_offset + start,
                char_end=char_offset + end,
                parents=parents,
                page_start=page_at(start),
                # a marker at the very end belongs to the next chunk's page, not this one's
                page_end=page_at(start + max(0, len(TRAILING_COMMENTS.sub("", body)) - 1)),
            )
        )
    return chunks
