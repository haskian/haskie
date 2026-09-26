"""The text a passage is read from: what `retrieval` reads off disk for one hit range.

A chunk row carries byte offsets, so a search reads exactly the bytes a range covers instead of the
whole document. These check that the slice is the right one, whatever the characters are, and that
the texts line up with the ranges they were read for.
"""

from pathlib import Path

import msgspec
import pytest

from haskie.collection.index import Hit, location
from haskie.search import retrieval
from haskie.search.passage import ranges
from haskie.search.retrieval import Plan, Scanned
from haskie.settings import Reranker, SearchSettings

# Multi-byte on purpose: a char offset is not a file position, so a read by char offsets would
# land mid-character.
ASCII = "# Retries\n\n" + "\n\n".join(
    f"Paragraph {i} about retrying a failed call." for i in range(40)
)
WIDE = "# Wiederholungen\n\n" + "\n\n".join(
    f"Абзац {i} — über Wiederholungen 🌍 und Zustellung." for i in range(40)
)


def _hit(markdown: str, path: Path, snippet: str) -> Hit:
    """One indexed chunk over `snippet`, with the offsets the index would have stored for it."""
    assert markdown.count(snippet) == 1, f"not unique in the fixture: {snippet!r}"
    char_start = markdown.index(snippet)
    char_end = char_start + len(snippet)
    line_start = markdown.count("\n", 0, char_start) + 1
    line_end = markdown.count("\n", 0, char_end - 1) + 1
    return Hit(
        collection="backend",
        document="doc.md",
        source_path="documents/doc.md",
        markdown_path="documents/doc.md",
        part=0,
        seq=1,
        line_start=line_start,
        line_end=line_end,
        char_start=char_start,
        char_end=char_end,
        byte_start=len(markdown[:char_start].encode()),
        byte_end=len(markdown[:char_end].encode()),
        page_start=None,
        page_end=None,
        headings=["Retries"],
        frame=["Retries"],
        header="Retries",
        location=location("doc.md", None, None, line_start, line_end),
        text=snippet,
        score=1.0,
        source_file=str(path),
        markdown_file=str(path),
    )


@pytest.mark.parametrize(
    ("name", "markdown", "snippet"),
    [
        ("ascii", ASCII, "Paragraph 20 about"),
        ("multi-byte text before the range", WIDE, "Абзац 20 — über"),
        ("the first chunk of the file", WIDE, "# Wiederholungen"),
        ("the last chunk of the file", WIDE, "Абзац 39 — über"),
        ("a range whose own text is multi-byte", WIDE, "Абзац 7 — über Wiederholungen 🌍"),
    ],
)
def test_a_read_is_exactly_the_span_of_one_range(
    name: str, markdown: str, snippet: str, tmp_path: Path
) -> None:
    path = tmp_path / "doc.md"
    path.write_text(markdown, encoding="utf-8")
    (hit_range,) = ranges([_hit(markdown, path, snippet)])

    (text,) = retrieval._read_texts([hit_range])

    assert text == snippet, name


@pytest.mark.anyio
async def test_the_texts_of_a_search_come_back_in_order(tmp_path: Path) -> None:
    """A search folds ranges of several documents and reads them in rank order, so the texts
    have to line up with the ranges they were read for - not with the files they came from."""
    first, second = tmp_path / "one.md", tmp_path / "two.md"
    first.write_text(ASCII, encoding="utf-8")
    second.write_text(WIDE, encoding="utf-8")
    snippets = [(ASCII, first, "Paragraph 20 about"), (WIDE, second, "Абзац 20 — über")]
    snippets.append((ASCII, first, "Paragraph 31 about"))
    hit_ranges = [ranges([_hit(markdown, path, one)])[0] for markdown, path, one in snippets]

    texts = await retrieval._texts_of(hit_ranges)

    assert texts == [one for _, _, one in snippets]


# --- how well a neighbour matches the query -----------------------------------------------

QUERY = "Why are retries idempotent?"
# the scanned hits: their scores, vectors and words set the floor a neighbour has to reach
SCANNED = Scanned(
    hits=[
        msgspec.structs.replace(
            _hit(ASCII, Path("doc.md"), f"Paragraph {i} about"), text=text, score=score
        )
        for i, (text, score) in enumerate(
            [("retries are idempotent", 3.0), ("retries", 2.0), ("nothing here", 1.0)]
        )
    ],
    vectors=[[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]],
)
NEIGHBOURS = [
    {"document": "doc.md", "seq": 7, "text": "idempotent retries", "vector": [0.8, 0.6]},
    {"document": "doc.md", "seq": 8, "text": "unrelated", "vector": [0.0, 1.0]},
]


def _read(rows: list[dict]) -> list[tuple[Hit, dict]]:
    """Neighbour rows as `_rows_at` hands them on: each with the hit it would be."""
    hit = SCANNED.hits[0]
    return [(msgspec.structs.replace(hit, seq=row["seq"], text=row["text"]), row) for row in rows]


def _plan(reranker: Reranker = Reranker.NONE, vector: list[float] | None = None) -> Plan:
    return Plan(
        settings=SearchSettings(reranker=reranker), indexes=[], vector=vector, embedding=None
    )


@pytest.mark.parametrize(
    ("name", "where", "rows", "expected", "floor", "signal"),
    [
        ("no neighbour read: nothing to score", _plan(vector=[1.0, 0.0]), [], [], 0.0, "none"),
        (
            "the query vector, against the median scanned hit's cosine",
            _plan(vector=[1.0, 0.0]),
            NEIGHBOURS,
            [0.8, 0.0],
            0.6,
            "vector",
        ),
        (
            "a row without a vector: the question's words instead",
            _plan(vector=[1.0, 0.0]),
            [NEIGHBOURS[0], {**NEIGHBOURS[1], "vector": None}],
            [1.0, 0.0],
            0.5,
            "words",
        ),
        (
            "full text only: the question's words",
            _plan(),
            NEIGHBOURS,
            [1.0, 0.0],
            0.5,
            "words",
        ),
    ],
)
@pytest.mark.anyio
async def test_a_neighbour_is_scored_the_strongest_way_the_search_can(
    name: str,
    where: Plan,
    rows: list[dict],
    expected: list[float],
    floor: float,
    signal: str,
) -> None:
    found = await retrieval._match_scores(SCANNED, _read(rows), where, QUERY)
    scores, found_floor, found_signal = found

    assert scores == pytest.approx(expected), name
    assert found_floor == pytest.approx(floor), f"{name}: the median scanned hit"
    assert found_signal == signal, name


@pytest.mark.anyio
async def test_with_a_reranker_a_neighbour_is_scored_by_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scanned hits already carry the reranker's scores, so the floor is their median and the
    neighbours are rescored by the same model, in whatever order it hands them back."""

    async def rerank(query: str, rows: list[dict], settings: SearchSettings) -> list[dict]:
        assert query == QUERY
        scored = [{**row, "_relevance_score": 4.0 if row["seq"] == 7 else -2.0} for row in rows]
        return sorted(scored, key=lambda row: row["_relevance_score"])

    monkeypatch.setattr(retrieval, "cross_encode", rerank)

    scores, floor, signal = await retrieval._match_scores(
        SCANNED, _read(NEIGHBOURS), _plan(Reranker.CROSS_ENCODER), QUERY
    )

    assert (scores, floor, signal) == ([4.0, -2.0], 2.0, "reranker")
