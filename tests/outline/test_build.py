"""A document's outline: which sections it has, where each runs, and the keywords each gets.

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

from haskie.collection.index import Row
from haskie.indexing.chunk import split
from haskie.outline import build, keywords
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
ROWS = [Row(chunk=chunk, seq=seq) for seq, chunk in enumerate(CHUNKS, start=1)]


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
        ("no chunk: no node", [], []),
        ("a note with no headings is the document alone", [[], []], [((), 0, 1)]),
        (
            "a parent before its children, in document order",
            [["A", "x"], ["A", "y"], ["B"]],
            [((), 0, 2), (("A",), 0, 1), (("A", "x"), 0, 0), (("A", "y"), 1, 1), (("B",), 2, 2)],
        ),
        (
            "a path that comes back after another opened is two nodes",
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


def _outline(embed: keywords.Embed | None, vectors: bool = True) -> dict[str, build.Node]:
    # each chunk points at its top-level topic: sagas one way, replication the other
    matrix = np.asarray(
        [[1.0, 0.0] if "Sagas" in row.chunk.headings else [0.0, 1.0] for row in ROWS]
    )
    found, _ = build.describe(ROWS, matrix if vectors else None, embed)
    return {node.header: node for node in found}


def test_every_section_is_a_node_with_its_span() -> None:
    nodes = _outline(None)

    assert list(nodes) == [
        "",
        "Sagas",
        "Sagas > Choreography",
        "Sagas > Orchestration",
        "Replication",
        "Replication > Leaders",
        "Replication > Quorums",
    ]
    sagas = nodes["Sagas"]
    assert sagas.depth == 1
    assert sagas.headings == ["Sagas"], "named as a Chunk names its path"
    assert sagas.char_start == CHUNKS[1].char_start and sagas.char_end == CHUNKS[4].char_end
    assert (sagas.byte_start, sagas.byte_end) == (CHUNKS[1].byte_start, CHUNKS[4].byte_end)
    assert (sagas.line_start, sagas.line_end) == (CHUNKS[1].line_start, CHUNKS[4].line_end)
    assert (sagas.page_start, sagas.page_end) == (None, None), "markdown has no pages"
    whole = nodes[""]
    assert (whole.depth, whole.char_start, whole.char_end) == (0, 0, CHUNKS[-1].char_end)


def test_keywords_weigh_a_section_against_its_siblings() -> None:
    """Without a model: the terms a chapter uses more than the other chapter, with their
    counts."""
    nodes = _outline(None)

    sagas, replication = nodes["Sagas"].keywords, nodes["Replication"].keywords
    assert "compensating" in sagas, "the form written most of compensating/compensation"
    assert "saga" not in replication and "quorum" in replication
    assert all(uses >= 1 for uses in sagas.values()), "each keyword keeps its count"
    orchestration = nodes["Sagas > Orchestration"].keywords
    assert "orchestrator" in orchestration, "a section against its siblings at its depth"


def test_a_model_reranks_the_candidates_against_the_section_vector() -> None:
    """The fake model puts every replication word on the replication axis: the replication
    chapter's keywords are then the words closest to its own vector, first."""
    calls: list[list[str]] = []
    replication_words = {"leader", "replication", "quorum", "replication log", "followers"}

    def embed(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[0.0, 1.0] if text in replication_words else [1.0, 0.1] for text in texts]

    nodes = _outline(embed)

    assert len(calls) == 1, "every candidate of the document embedded in one call"
    assert len(calls[0]) == len(set(calls[0])), "each candidate once"
    first = next(iter(nodes["Replication"].keywords))
    assert first in replication_words


def test_a_model_without_vectors_on_the_rows_ranks_by_weight() -> None:
    """No vectors (no model when the rows were cut): no section vector to rerank against, so the
    embed function is never called."""

    def embed(texts: list[str]) -> list[list[float]]:
        raise AssertionError("no rerank without chunk vectors")

    assert _outline(embed, vectors=False)["Sagas"].keywords == _outline(None)["Sagas"].keywords


def test_no_candidate_leaves_the_keywords_empty() -> None:
    chunks = split("# A\n\nof the and\n", ChunkSettings())
    rows = [Row(chunk=chunk, seq=1) for chunk in chunks]

    (whole, heading), _ = build.describe(
        rows, np.ones((1, 1)), lambda texts: [[1.0] for _ in texts]
    )

    assert whole.keywords == {} and heading.keywords == {}


def test_a_word_every_section_uses_is_no_sections_keyword_for_repeating_it() -> None:
    """Five chapters use "example" once, the first twice: its uses once are no candidates in the
    others, but they still count in how common it is, so the first chapter's own word wins."""
    body = ["# One\n\nsaga, saga, example, example\n"]
    body += [f"# Ch{n}\n\nexample topic{n} topic{n}\n" for n in range(5)]
    chunks = split("\n".join(body), ChunkSettings(chunk_size=200, chunk_merge_below=0))
    rows = [Row(chunk=chunk, seq=seq) for seq, chunk in enumerate(chunks, start=1)]

    nodes = {node.header: node for node in build.describe(rows, None, None)[0]}

    order = list(nodes["One"].keywords)
    assert order.index("saga") < order.index("example"), order


def test_each_node_gets_the_unit_mean_of_its_chunk_vectors() -> None:
    """A long chunk weighs no more than a short one: each chunk is scaled to length one before
    the mean, and the mean is scaled to length one after."""
    matrix = np.asarray(
        [[3.0, 0.0] if "Sagas" in row.chunk.headings else [0.0, 0.5] for row in ROWS]
    )

    nodes, vectors = build.describe(ROWS, matrix, None)

    assert vectors is not None and vectors.shape == (len(nodes), 2), "one row per node"
    by_header = {node.header: vector for node, vector in zip(nodes, vectors, strict=True)}
    assert np.allclose(by_header["Sagas"], [1.0, 0.0])
    assert np.allclose(by_header["Replication > Leaders"], [0.0, 1.0])
    # the whole document: 1 preamble + 3 replication chunks on one axis, 4 sagas on the other
    assert np.allclose(by_header[""], np.asarray([4.0, 4.0]) / np.hypot(4.0, 4.0))
    assert build.describe(ROWS, None, None)[1] is None, "no chunk vectors: no node vectors"


def test_the_strategy_given_picks_the_keywords() -> None:
    """`describe` hands the strategy each run's depth and chunks, and the node vectors."""
    seen: list[tuple[list[keywords.Run], np.ndarray | None]] = []

    class Headers:
        def pick(self, texts, runs, vectors, embed) -> list[dict[str, int]]:
            seen.append((list(runs), vectors))
            return [{f"depth {run.depth}": run.last - run.first + 1} for run in runs]

    nodes, vectors = build.describe(ROWS, np.ones((len(ROWS), 2)), None, Headers())

    ((runs, given),) = seen
    assert runs[1] == keywords.Run(depth=1, first=1, last=4), "Sagas: chunks 1 to 4"
    assert given is vectors, "the node vectors, not the chunk vectors"
    assert nodes[1].keywords == {"depth 1": 4}


def test_a_document_without_chunks_has_no_node() -> None:
    nodes, vectors = build.describe([], np.empty((0, 4)), None)

    assert nodes == [] and vectors is not None and vectors.shape == (0, 4)
