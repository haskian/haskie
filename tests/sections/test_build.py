"""A document's sections, with their ids, and the descriptors each section gets.

`BOOK` is cut by the real chunker at 200 characters with no merging, so each paragraph is its own
chunk, under the heading path it sits under:

    seq 1  (none)                          a preamble before the first heading
    seq 2  Sagas > Choreography
    seq 3  Sagas > Choreography
    seq 4  Sagas > Orchestration
    seq 5  Sagas > Orchestration
    seq 6  Replication > Leaders
    seq 7  Replication > Leaders
    seq 8  Replication > Quorums
"""

import numpy as np
import pytest

from haskie import ids
from haskie.indexing.chunk import Chunk, split
from haskie.search.collapse import unit_rows
from haskie.sections import build, descriptors
from haskie.settings import ChunkSettings

BOOK = """A preamble before any heading, about nothing in particular.

# Sagas

## Choreography

In a choreographed saga every service listens for events and emits compensating events.

Choreography keeps services loosely coupled: each compensating event undoes one step.

## Orchestration

An orchestrator tells each service which step to run and which compensation to call.

The orchestrator holds the saga state, so a failed step triggers its compensation.

# Replication

## Leaders

A single leader accepts every write and ships its replication log to the followers.

Followers apply the leader log in order, so replication lag shows as stale reads.

## Quorums

A quorum read and a quorum write overlap, so a quorum read sees the latest write.
"""
CHUNKS = split(BOOK, ChunkSettings(chunk_size=200, chunk_merge_below=0))


def test_the_fixture_is_cut_as_the_module_says() -> None:
    assert [tuple(chunk.headings) for chunk in CHUNKS] == [
        (),
        ("Sagas", "Choreography"),
        ("Sagas", "Choreography"),
        ("Sagas", "Orchestration"),
        ("Sagas", "Orchestration"),
        ("Replication", "Leaders"),
        ("Replication", "Leaders"),
        ("Replication", "Quorums"),
    ]


@pytest.mark.parametrize(
    ("name", "paths", "expected"),
    [
        ("no chunk: no section", [], []),
        ("a note with no headings is the document alone", [[], []], [((), 0, 1)]),
        (
            "a parent before its children, in document order",
            [["A", "x"], ["A", "y"], ["B"]],
            [((), 0, 2), (("A",), 0, 1), (("A", "x"), 0, 0), (("A", "y"), 1, 1), (("B",), 2, 2)],
        ),
        (
            "a path that comes back after another opened is two sections",
            [["A"], ["B"], ["A"]],
            [((), 0, 2), (("A",), 0, 0), (("B",), 1, 1), (("A",), 2, 2)],
        ),
        (
            "a chunk under fewer headings ends the deeper run",
            [["A", "x"], ["A"], ["A", "x"]],
            [((), 0, 2), (("A",), 0, 2), (("A", "x"), 0, 0), (("A", "x"), 2, 2)],
        ),
    ],
)
def test_runs(name: str, paths: list[list[str]], expected: list) -> None:
    assert build.runs(paths) == expected, name


DOC = ids.md5(b"book.md")


def _sections(chunks: list[Chunk] = CHUNKS) -> tuple[list[build.Section], list[list[int]]]:
    return build.sections(DOC, chunks)


def _pooled(matrix: np.ndarray, chains: list[list[int]], count: int) -> np.ndarray:
    """Each section's unit vector as the cache merge sums it: every chunk's unit vector into each
    section that holds it (`embed_cache._merge`)."""
    sums = np.zeros((count, matrix.shape[1]))
    for chain, vector in zip(chains, matrix, strict=True):
        sums[chain] += vector / np.linalg.norm(vector)
    return unit_rows(sums)


def _key(written: str) -> str:
    """A descriptor's key, as `descriptors.terms` keys it: a word or a pair, the pair last."""
    return descriptors.terms(written)[-1][0]


def _prose(chunks: list[Chunk]) -> list[str]:
    return [build.prose(chunk) for chunk in chunks]


def _described(embed: descriptors.Embed | None, vectors: bool = True) -> dict[str, build.Section]:
    # each chunk points at its top-level topic: sagas one way, replication the other
    matrix = np.asarray(
        [[1.0, 0.0] if "Sagas" in chunk.headings else [0.0, 1.0] for chunk in CHUNKS]
    )
    found, chains = _sections()
    pooled = _pooled(matrix, chains, len(found)) if vectors else None
    return {one.header: one for one in build.describe(found, _prose(CHUNKS), pooled, embed)}


def test_every_section_has_its_span_its_parent_and_the_chunks_it_holds() -> None:
    found, chains = _sections()
    by_header = {one.header: one for one in found}

    assert list(by_header) == [
        "",
        "Sagas",
        "Sagas > Choreography",
        "Sagas > Orchestration",
        "Replication",
        "Replication > Leaders",
        "Replication > Quorums",
    ]
    sagas = by_header["Sagas"]
    assert sagas.depth == 1
    assert sagas.headings == ["Sagas"], "named as a Chunk names its path"
    assert (sagas.seq_start, sagas.seq_end) == (2, 5)
    assert sagas.char_start == CHUNKS[1].char_start and sagas.char_end == CHUNKS[4].char_end
    assert (sagas.byte_start, sagas.byte_end) == (CHUNKS[1].byte_start, CHUNKS[4].byte_end)
    assert (sagas.line_start, sagas.line_end) == (CHUNKS[1].line_start, CHUNKS[4].line_end)
    assert (sagas.page_start, sagas.page_end) == (None, None), "markdown has no pages"
    whole = by_header[""]
    assert (whole.depth, whole.char_start, whole.char_end) == (0, 0, CHUNKS[-1].char_end)
    assert whole.parent_id is None and sagas.parent_id == whole.id
    assert by_header["Sagas > Orchestration"].parent_id == sagas.id
    assert by_header["Replication > Quorums"].parent_id == by_header["Replication"].id
    assert len({one.id for one in found}) == len(found), "an id names one section"
    headers = [[found[at].header for at in chain] for chain in chains]
    assert headers[0] == [""], "the preamble sits in the whole document alone"
    assert headers[3] == ["", "Sagas", "Sagas > Orchestration"], "outermost first"
    assert headers[7] == ["", "Replication", "Replication > Quorums"]


def _md5(text: str) -> str:
    return ids.md5(text.encode())


def test_a_sections_id_is_its_document_and_its_place_among_its_sections() -> None:
    """The MD5 of `<document id>/<position>`, the whole document 0. Sections are cut by heading
    paths alone: chunked at 1,200 characters with short paragraphs merged, rather than 200 with
    none, the chunks fall elsewhere and the sections keep their ids."""
    big = split(BOOK, ChunkSettings(chunk_size=1200))
    assert len(big) < len(CHUNKS), "the fixture is really chunked another way"
    found, _ = _sections()

    assert [one.id for one in found] == [_md5(f"{DOC}/s/{at}") for at in range(len(found))]
    assert [one.id for one in _sections(big)[0]] == [one.id for one in found]
    elsewhere = build.sections("f" * 32, CHUNKS)[0]
    assert not {one.id for one in found} & {one.id for one in elsewhere}, "another document"


def test_two_sections_of_the_same_content_get_their_own_ids() -> None:
    """The same heading and the same text twice, apart: two sections, two ids."""
    retry = "# Retry\n\nBack off, then retry.\n\n"
    markdown = f"{retry}# Log\n\nKeep it.\n\n{retry}"
    chunks = split(markdown, ChunkSettings(chunk_size=200, chunk_merge_below=0))

    found, _ = build.sections(DOC, chunks)

    retries = [one.id for one in found if one.header == "Retry"]
    assert len(retries) == 2 and retries[0] != retries[1]


def test_a_chunks_id_is_its_document_and_its_seq() -> None:
    assert build.chunk_id(DOC, 1) == _md5(f"{DOC}/c/1")
    found, _ = _sections()
    assert build.chunk_id(DOC, 2) != found[2].id, "chunk 2 and section 2 are two ids"
    assert len({build.chunk_id(DOC, 1), build.chunk_id(DOC, 2), build.chunk_id("f" * 32, 1)}) == 3


def test_each_sections_vector_is_the_unit_mean_of_its_chunks() -> None:
    """A long chunk weighs no more than a short one: each chunk is scaled to length one before
    the sum, and the sum is scaled to length one after."""
    matrix = np.asarray([[3.0, 0.0] if "Sagas" in one.headings else [0.0, 0.5] for one in CHUNKS])
    found, chains = _sections()

    vectors = _pooled(matrix, chains, len(found))

    by_header = {one.header: vector for one, vector in zip(found, vectors, strict=True)}
    assert np.allclose(by_header["Sagas"], [1.0, 0.0])
    assert np.allclose(by_header["Replication > Leaders"], [0.0, 1.0])
    # the whole document: 1 preamble + 3 replication chunks on one axis, 4 sagas on the other
    assert np.allclose(by_header[""], np.asarray([4.0, 4.0]) / np.hypot(4.0, 4.0))


def test_descriptors_weigh_a_section_against_its_siblings() -> None:
    """Without a model: the terms a chapter uses more than the other chapter."""
    described = _described(None)

    sagas, replication = described["Sagas"].descriptors, described["Replication"].descriptors
    assert "compensating" in sagas, "the form written most of compensating/compensation"
    assert "saga" not in replication and "quorum" in replication
    orchestration = described["Sagas > Orchestration"].descriptors
    assert "orchestrator tells" in orchestration, "a section against its siblings at its depth"
    assert "orchestrator" not in orchestration, "its header says it: Orchestration, one stem"
    assert "replication" not in replication, "nor does a chapter repeat its own heading"
    assert described["Sagas"].id == {one.header: one.id for one in _sections()[0]}["Sagas"]


def test_a_model_reranks_the_candidates_against_the_section_vector() -> None:
    """The fake model puts every replication word on the replication axis: the replication
    chapter's descriptors are then the words closest to its own vector, first."""
    calls: list[list[str]] = []
    replication_words = {"leader", "replication", "quorum", "replication log", "followers"}

    def embed(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[0.0, 1.0] if text in replication_words else [1.0, 0.1] for text in texts]

    described = _described(embed)

    assert len(calls) == 1, "every candidate of the document embedded in one call"
    assert len(calls[0]) == len(set(calls[0])), "each candidate once"
    first = next(iter(described["Replication"].descriptors))
    assert first in replication_words


def test_a_model_without_vectors_on_the_rows_ranks_by_weight() -> None:
    """No vectors (no model when the rows were cut): no section vector to rerank against, so the
    embed function is never called."""

    def embed(texts: list[str]) -> list[list[float]]:
        raise AssertionError("no rerank without chunk vectors")

    unranked = _described(None)["Sagas"].descriptors
    assert _described(embed, vectors=False)["Sagas"].descriptors == unranked


def test_no_candidate_leaves_the_descriptors_empty() -> None:
    markdown = "# A\n\nof the and\n"
    chunks = split(markdown, ChunkSettings())
    found, _ = build.sections(DOC, chunks)

    whole, heading = build.describe(
        found, _prose(chunks), np.ones((len(found), 1)), lambda texts: [[1.0] for _ in texts]
    )

    assert whole.descriptors == [] and heading.descriptors == []


def test_a_word_every_section_uses_is_no_sections_descriptor_for_repeating_it() -> None:
    """Every chapter uses "example", the first twice: a word every section uses is the
    document's, not the first chapter's, however often that one repeats it."""
    body = ["# One\n\nsaga, saga, example, example\n"]
    body += [f"# Ch{n}\n\nexample topic{n} topic{n}\n" for n in range(5)]
    markdown = "\n".join(body)
    chunks = split(markdown, ChunkSettings(chunk_size=200, chunk_merge_below=0))
    found, _ = build.sections(DOC, chunks)

    described = {one.header: one for one in build.describe(found, _prose(chunks), None, None)}

    assert list(described["One"].descriptors) == ["saga"]
    assert "example" in described[""].descriptors, "the whole document keeps its own word"


def test_the_strategy_given_picks_the_descriptors() -> None:
    """`describe` hands the strategy each section's headings and chunks, and its vectors."""
    seen: list[tuple[list[descriptors.Run], np.ndarray | None]] = []

    class Headers:
        def pick(self, texts, runs, vectors, embed) -> list[list[str]]:
            seen.append((list(runs), vectors))
            return [[f"depth {run.depth}: {run.last - run.first + 1} chunks"] for run in runs]

    found, _ = _sections()
    vectors = np.ones((len(found), 2))
    described = build.describe(found, _prose(CHUNKS), vectors, None, Headers())

    ((runs, given),) = seen
    assert runs[1] == descriptors.Run(headings=("Sagas",), first=1, last=4), "Sagas: chunks 1 to 4"
    assert given is vectors, "the section vectors it was given"
    assert described[1].descriptors == ["depth 1: 4 chunks"]


def test_a_document_without_chunks_has_no_section() -> None:
    assert build.sections(DOC, []) == ([], [])
    assert build.describe([], [], np.empty((0, 4)), None) == []


def test_code_blocks_and_tables_give_no_descriptors() -> None:
    """The strategy reads prose only: a code block and a table are a blank line in the text it
    gets, so their identifiers and cells are no candidates, and no pair spans one."""
    markdown = (
        "# Sagas\n\n"
        "A saga compensates a failed step. The saga keeps its log.\n\n"
        "```python\ndef compensate(order_id):\n    ledger.refund(order_id)\n```\n\n"
        "| column | value |\n| --- | --- |\n| ledgerRefund | orderId |\n\n"
        "Each compensating step undoes one step of the saga.\n"
    )
    chunks = split(markdown, ChunkSettings(chunk_size=1200))
    found, _ = build.sections(DOC, chunks)
    assert {piece.type for chunk in chunks for piece in chunk.pieces} >= build.NOT_PROSE
    seen: list[list[str]] = []

    class Texts:
        def pick(self, texts, runs, vectors, embed) -> list[list[str]]:
            seen.append(list(texts))
            return [[] for _ in runs]

    build.describe(found, _prose(chunks), None, None, Texts())
    (texts,) = seen
    read = "".join(texts)
    assert "saga compensates a failed step" in read.lower(), "prose kept"
    assert "Each compensating step" in read, "prose after the blocks kept"
    for word in ("refund", "ledger", "orderId", "column"):
        assert word not in read, f"{word}: code and tables left out"
    described = build.describe(found, _prose(chunks), None, None)
    keys = {key for one in described for key in map(_key, one.descriptors)}
    assert _key("saga") in keys
    assert not keys & {_key(word) for word in ("ledger", "refund", "orderid")}
