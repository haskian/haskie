"""Growing passages against a real index with vectors: the neighbours are read back from LanceDB
with their vectors, and valued by cosine around the ranked chunks, the way a hybrid or vector
search does it.

The index holds a real chunking of `MARKDOWN`, one chunk a paragraph, each stored with a
2-dimensional vector that says what it is about: [1, 0] on retries, [0, 1] on something else, and
between for a chunk partly on both. The query vector is [1, 0].

    seq 1-6  Guide > Retries   the section the passages sit in
    seq 7    Guide > Other     another section, never grown into
"""

from pathlib import Path

import msgspec
import numpy as np
import pytest
from conftest import one_part

from haskie.catalogue.catalogue import EmbeddingModel
from haskie.collection.index import ChunkKey, CollectionIndex, Hit, Row, chunk_key, gather_rows
from haskie.indexing.chunk import split
from haskie.search import probe, retrieval, section
from haskie.search.collapse import Vector
from haskie.search.passage import ranges
from haskie.search.retrieval import Plan, Scanned
from haskie.settings import ChunkSettings, FillValues, Reranker, ScoreFold, SearchSettings

HARMONIC = ScoreFold.HARMONIC  # the rule these cases were written against

MARKDOWN = (
    "# Guide\n\n## Retries\n\n"
    + "\n\n".join(
        f"Paragraph {i} of the retries section says one thing about retries and their backoff."
        for i in range(1, 7)
    )
    + "\n\n## Other\n\nThe other section talks about something else entirely, far from here.\n"
)

DOC = "guide.md"
ON, OFF = [1.0, 0.0], [0.0, 1.0]
QUERY = "How do retries back off?"


async def _index(tmp_path: Path, vectors: dict[int, list[float]]) -> tuple[Plan, dict[int, Hit]]:
    """The fixture indexed with `vectors` by seq, and a search plan over it for the query vector,
    with every chunk as the hit the index reads back."""
    model = EmbeddingModel("test/model", 2)
    index = CollectionIndex(tmp_path / "index", "notes", tmp_path, model)
    chunks = split(MARKDOWN, ChunkSettings(chunk_size=150, chunk_merge_below=0))
    assert [tuple(one.headings) for one in chunks] == [("Guide", "Retries")] * 6 + [
        ("Guide", "Other")
    ], "the fixture the docstring names"
    rows = [Row(chunk=one, vector=vectors[seq], seq=seq) for seq, one in enumerate(chunks, 1)]
    await index.add_parts(DOC, f"documents/{DOC}", f"documents/{DOC}.md", one_part(0, rows))
    ((_, stored),) = await gather_rows(
        [index], lambda one: one.rows_at([(DOC, seq) for seq in vectors], vectors=True)
    )
    hits = {row["seq"]: index.hit(row, 1.0) for row in stored}
    settings = SearchSettings(max_passage_grow=2)
    where = Plan(settings=settings, indexes=[(index, settings)], vector=ON, embedding=model)
    return where, hits


def _asked() -> list[probe.Question]:
    return [probe.Question(vector=ON, asked=QUERY)]


NEAR = [0.95, 0.312]  # cosine 0.95 to the query
VECTORS = {1: ON, 2: ON, 3: NEAR, 4: OFF, 5: ON, 6: [0.7, 0.714], 7: ON}


@pytest.mark.anyio
async def test_a_neighbour_is_read_back_with_its_vector(tmp_path: Path) -> None:
    where, _ = await _index(tmp_path, VECTORS)
    (index, _) = where.indexes[0]

    rows = await retrieval._rows_at(where, {("notes", DOC, 4), ("notes", DOC, 99)})
    lexical = await index.rows_at([(DOC, 4)], vectors=False)

    ((hit, row),) = rows.values()
    assert (hit.seq, row["vector"].tolist()) == (4, pytest.approx(OFF)), "the chunk, its vector"
    assert row["vector"].dtype == np.float32, "a row of the read's array, not a list of floats"
    assert hit.score == 0.0, "read, not ranked"
    assert "vector" not in lexical[0], "a lexical search reads no vector"


@pytest.mark.anyio
async def test_a_section_fills_by_what_its_chunks_mean(tmp_path: Path) -> None:
    """Kept: chunks 1 and 3 (cosines 1.0 and 0.95, so the median is 0.975 and the best 1.0).
    Chunk 2 is as good as the best, so the gap fills and the two become one. Past chunk 3, chunk 4
    is off-topic (-1) and chunk 5 on it (+1): the run sums to 0, not above, so neither joins."""
    where, hits = await _index(tmp_path, VECTORS)
    kept = [hits[1], hits[3]]
    groups = await retrieval.sections(ranges(kept, how=HARMONIC), where, limit=5)

    (filled,) = await retrieval.fill(groups, _asked(), where)

    assert filled.section == section.Section(("Guide", "Retries"), 1, 6)
    assert [(one.seq_start, one.seq_end) for one in filled.ranges] == [(1, 3)]
    assert [hit.score for hit in filled.ranges[0].hits] == [1.0, 0.0, 1.0], "chunk 2 is unranked"


def _biased(where: Plan, bias: float) -> Plan:
    return msgspec.structs.replace(
        where, settings=msgspec.structs.replace(where.settings, grow_bias=bias)
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "bias", "expected"),
    [
        ("no bias: chunks 4 (-1) and 5 (+1) sum to 0, so neither joins", 0.0, [(1, 3)]),
        ("an eager bias takes off-topic chunk 4 for chunk 5 past it", 0.5, [(1, 5)]),
        ("-1: chunk 2, as good as the best, is worth 0: the gap stays", -1.0, [(1, 1), (3, 3)]),
    ],
)
async def test_the_grow_bias_decides_how_far_a_section_fills(
    tmp_path: Path, name: str, bias: float, expected: list[tuple[int, int]]
) -> None:
    """The fixture of `test_a_section_fills_by_what_its_chunks_mean`, with `grow_bias` set."""
    where, hits = await _index(tmp_path, VECTORS)
    where = _biased(where, bias)
    groups = await retrieval.sections(ranges([hits[1], hits[3]], how=HARMONIC), where, limit=5)

    (filled,) = await retrieval.fill(groups, _asked(), where)

    assert [(one.seq_start, one.seq_end) for one in filled.ranges] == expected, name


@pytest.mark.anyio
async def test_a_strict_grow_bias_leaves_a_short_passage_as_it_is(tmp_path: Path) -> None:
    """The fixture of the next test, where chunk 1 grows to 3 with no bias. At -1 no neighbour is
    worth taking; chunk 1 is still the best range, so it stays, thin."""
    where, hits = await _index(tmp_path, VECTORS)
    scanned = Scanned(hits=[hits[1], hits[7]], vectors=[ON, [0.8, 0.6]])

    ranged = await retrieval.fill_thin(scanned, _biased(where, -1.0), QUERY, QUERY)

    assert [(one.seq_start, one.seq_end, one.alone) for one in ranged.ranges] == [
        (1, 1, False),
        (7, 7, False),
    ]


@pytest.mark.anyio
async def test_a_short_passage_grows_by_the_neighbours_that_mean_the_same(tmp_path: Path) -> None:
    """Scanned: chunk 1 (cosine 1.0) and chunk 7 (0.8, a whole short section), so the median is
    0.9 and the best 1.0. Chunk 1 is a fragment of its section, so it grows: 2 (1.0) and 3 (0.95)
    are worth more than the median, and the reach of 2 stops it there. Chunk 7 has a heading on
    both sides and is kept as it is."""
    where, hits = await _index(tmp_path, VECTORS)
    scanned = Scanned(hits=[hits[1], hits[7]], vectors=[ON, [0.8, 0.6]])

    ranged = await retrieval.fill_thin(scanned, where, QUERY, QUERY)

    assert [(one.seq_start, one.seq_end) for one in ranged.ranges] == [(1, 3), (7, 7)]
    added: dict[ChunkKey, Vector | None] = {
        chunk_key(hit): vector
        for hit, vector in zip(ranged.scanned.hits, ranged.scanned.vectors, strict=True)
    }
    assert added[("notes", DOC, 3)] == pytest.approx(NEAR), "a neighbour joins the scan"


@pytest.mark.anyio
async def test_a_short_passage_with_no_neighbour_worth_it_stands_alone(tmp_path: Path) -> None:
    """Both scanned hits score cosine 1.0, so only a neighbour as good joins. Chunk 1 grows by
    chunk 2 (1.0) and stops at chunk 3 (0.95). Chunk 5 is on retries but short, and both its
    neighbours are off-topic: nothing joins, and past the best range it stands alone."""
    where, hits = await _index(tmp_path, {**VECTORS, 4: OFF, 6: OFF})
    scanned = Scanned(hits=[hits[1], hits[5]], vectors=[ON, ON])

    ranged = await retrieval.fill_thin(scanned, where, QUERY, QUERY)

    shape = [(one.seq_start, one.seq_end, one.alone) for one in ranged.ranges]
    assert shape == [(1, 2, False), (5, 5, True)]


@pytest.mark.anyio
async def test_in_an_excerpt_a_short_passage_grows_once_by_the_fill(tmp_path: Path) -> None:
    """An excerpts search only judges its short passages (`grows=False`); the fill then grows
    each passage once, by at most `max_passage_grow` chunks a side. Chunk 1 stands, since chunk
    2 is worth taking. The fill values against the kept chunk alone (cosine 1.0): chunk 2 (1.0)
    is worth 1 and chunk 3 (0.95) -1, so it grows by 2 only, where growing twice would have
    reached 4."""
    where, hits = await _index(tmp_path, VECTORS)
    scanned = Scanned(hits=[hits[1], hits[7]], vectors=[ON, [0.8, 0.6]])

    judged = await retrieval.fill_thin(scanned, where, QUERY, QUERY, grows=False)
    groups = await retrieval.sections([judged.ranges[0]], where, limit=1)
    (filled,) = await retrieval.fill(groups, _asked(), where)

    assert [(one.seq_start, one.seq_end, one.alone) for one in judged.ranges] == [
        (1, 1, False),
        (7, 7, False),
    ], "judged, not grown"
    assert [(one.seq_start, one.seq_end) for one in filled.ranges] == [(1, 2)], "grown once"


@pytest.mark.anyio
async def test_a_short_passage_the_fill_grows_by_nothing_goes(tmp_path: Path) -> None:
    """Scanned: chunk 7 (a whole section, the best range, scored 0.3 by its scanned vector) and
    chunk 5 (1.0), so the judge's median is 0.65: chunk 6 (0.7) is worth taking, and chunk 5
    stands, owed its growth. The fill values chunk 6 against the kept chunks' stored vectors
    (both 1.0) instead, where it is worth -1, and takes nothing: chunk 5 goes, as the passages
    answer drops a thin passage, and the best range's section stays."""
    where, hits = await _index(tmp_path, VECTORS)
    best = msgspec.structs.replace(hits[7], score=2.0)
    scanned = Scanned(hits=[best, hits[5]], vectors=[[0.3, 0.954], ON])

    judged = await retrieval.fill_thin(scanned, where, QUERY, QUERY, grows=False)
    groups = await retrieval.sections(judged.ranges, where, limit=5)
    filled = await retrieval.fill(groups, _asked(), where)

    assert [(one.seq_start, one.owed) for one in judged.ranges] == [(7, False), (5, True)]
    assert len(groups) == 2, "both sections placed before the fill"
    assert [one.section.path for one in filled] == [("Guide", "Other")]


@pytest.mark.anyio
async def test_a_lexical_read_leaves_vectors_out_only_when_asked(tmp_path: Path) -> None:
    """The probe's full-text search only wants the chunks, so its rows carry no vector but keep
    their score; any other full-text read keeps the vectors the fold compares by."""
    where, _ = await _index(tmp_path, VECTORS)
    await where.indexes[0][0].finish()  # the full-text index the search reads
    lexical = Plan(settings=where.settings, indexes=where.indexes, vector=None, embedding=None)

    probed = await retrieval.fan_out(lexical, "backoff", 5, vectors=False)
    plain = await retrieval.fan_out(lexical, "backoff", 5)

    assert probed.rows and all("vector" not in row for _, row in probed.rows.values())
    assert all(row["_score"] > 0 for _, row in probed.rows.values()), "scored as before"
    assert all(row["vector"] is not None for _, row in plain.rows.values())
    assert list(probed.rows) == list(plain.rows), "the same chunks, in the same order"


@pytest.mark.anyio
async def test_a_chunk_near_two_groups_is_a_candidate_of_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sections of one document can nest: a group of the whole guide and one of its Retries
    section. Chunk 2 is near a passage of each, and each may take it, the one whose passage it
    continues; one map of chunk to group gave it to whichever came last."""
    where, hits = await _index(tmp_path, VECTORS)
    within = section.Section(("Guide", "Retries"), 1, 6)
    retries = section.Group("notes", DOC, within, ranges([hits[1]], how=HARMONIC))
    guide = section.Group(
        "notes", DOC, section.Section(("Guide",), 1, 7), ranges([hits[3]], how=HARMONIC)
    )
    offered: dict[int, set[int]] = {}
    fills = retrieval.filling.fills

    def seen(at: int, one: section.Group, candidates: dict, reach: int) -> list:
        offered[at] = {seq for _, _, seq in candidates}
        return fills(at, one, candidates, reach)

    monkeypatch.setattr(retrieval.filling, "fills", seen)

    await retrieval.fill([retries, guide], _asked(), where)

    assert offered == {0: {2, 3}, 1: {1, 2, 4, 5}}


@pytest.mark.parametrize(
    ("name", "reranker"),
    [
        ("without a reranker: a fill is valued against the kept passages, not judged", False),
        ("with one: only the reranker tags", True),
    ],
)
@pytest.mark.anyio
async def test_a_filled_chunk_tags_no_question(tmp_path: Path, name: str, reranker: bool) -> None:
    """Chunk 1 is kept untagged; chunk 2 next to it is as close to the first question as it is,
    so the fill takes it. It tags no question either way: a fill's value is relative to this
    search's own passages, so on a question nothing answers it would still mark one answered."""
    where, hits = await _index(tmp_path, VECTORS)
    judge = Reranker.CROSS_ENCODER if reranker else Reranker.NONE
    settings = SearchSettings(max_passage_grow=2, reranker=judge)
    where = msgspec.structs.replace(where, settings=settings)
    groups = await retrieval.sections(ranges([hits[1]], how=HARMONIC), where, limit=1)
    asked = [
        probe.Question(vector=ON, asked=QUERY, label="q1"),
        probe.Question(vector=OFF, asked="What else?", label="q2"),
    ]

    (filled,) = await retrieval.fill(groups, asked, where)

    assert [hit.seq for hit in filled.ranges[0].hits][:2] == [1, 2], f"{name}: chunk 2 joined"
    assert filled.ranges[0].aspects == [], name


@pytest.mark.anyio
async def test_absolute_values_fill_by_the_rerankers_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `fill_values = absolute` the fill asks the reranker about each chunk near a passage,
    once a question, and values it as dsRAG does: chunk 2 at 0.9 is worth 0.72, chunk 3 at 0.05
    costs 0.13, so the passage takes 2 and stops, whatever the vectors say."""
    where, hits = await _index(tmp_path, VECTORS)
    settings = SearchSettings(
        max_passage_grow=2, reranker=Reranker.CROSS_ENCODER, fill_values=FillValues.ABSOLUTE
    )
    where = msgspec.structs.replace(where, settings=settings)
    groups = await retrieval.sections(ranges([hits[1]], how=HARMONIC), where, limit=1)
    asked: list[str] = []

    async def rerank(query: str, rows: list[dict], settings: SearchSettings) -> list[dict]:
        asked.append(query)
        for row in rows:
            row["_relevance_score"] = {2: 0.9, 3: 0.05}.get(row["seq"], 0.0)
        return rows

    monkeypatch.setattr(retrieval, "cross_encode", rerank)

    (filled,) = await retrieval.fill(groups, _asked(), where)

    assert [(one.seq_start, one.seq_end) for one in filled.ranges] == [(1, 2)]
    assert asked == [QUERY], "one reranker pass a question"
