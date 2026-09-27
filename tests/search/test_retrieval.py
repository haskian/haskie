"""The text a passage is read from: what `retrieval` reads off disk for one hit range.

A chunk row carries byte offsets, so a search reads exactly the bytes a range covers instead of the
whole document. These check that the slice is the right one, whatever the characters are, and that
the texts line up with the ranges they were read for.
"""

import math
from pathlib import Path

import msgspec
import pytest
from conftest import hit

from haskie.collection.index import Hit, chunk_key, location
from haskie.indexing.segment import CutReason
from haskie.search import probe, retrieval, section
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


def _sigmoid(logit: float) -> float:
    """A reranker score, as `cross_encode` stores it."""
    return 1 / (1 + math.exp(-logit))


def _plan(reranker: Reranker = Reranker.NONE, vector: list[float] | None = None) -> Plan:
    return Plan(
        settings=SearchSettings(reranker=reranker), indexes=[], vector=vector, embedding=None
    )


REFERENCE = list(zip((hit.text for hit in SCANNED.hits), SCANNED.vectors, strict=True))


@pytest.mark.parametrize(
    ("name", "vector", "rows", "expected", "signal"),
    [
        (
            # scanned cosines 1.0, 0.6, 0.0: 0.6 is worth 0 and 1.0 is worth 1
            "the query vector, around the scanned hits' cosines",
            [1.0, 0.0],
            NEIGHBOURS,
            [0.5, -1.0],
            "vector",
        ),
        (
            # scanned overlaps 1.0, 0.5, 0.0
            "a row without a vector: the question's words instead",
            [1.0, 0.0],
            [NEIGHBOURS[0], {**NEIGHBOURS[1], "vector": None}],
            [1.0, -1.0],
            "words",
        ),
        ("full text only: the question's words", None, NEIGHBOURS, [1.0, -1.0], "words"),
    ],
)
def test_a_neighbour_is_valued_around_the_scanned_hits(
    name: str, vector: list[float] | None, rows: list[dict], expected: list[float], signal: str
) -> None:
    candidates = [(row["text"], row["vector"]) for row in rows]

    (values,), found = retrieval._values(REFERENCE, candidates, [(QUERY, vector)])

    assert values == pytest.approx(expected), name
    assert found == signal, name


@pytest.mark.parametrize(
    ("name", "where", "rows", "reranked", "signal"),
    [
        ("no neighbour read: nothing to value", _plan(vector=[1.0, 0.0]), [], None, "none"),
        (
            "the reranker's scores when one is on",
            _plan(Reranker.CROSS_ENCODER),
            NEIGHBOURS,
            [_sigmoid(4.0), _sigmoid(-2.0)],
            "reranker",
        ),
        ("else the scanned hits' own signal", _plan(vector=[1.0, 0.0]), NEIGHBOURS, None, "vector"),
    ],
)
def test_thin_values_neighbours_the_strongest_way_the_search_can(
    name: str, where: Plan, rows: list[dict], reranked: list[float] | None, signal: str
) -> None:
    read = {chunk_key(hit): (hit, row) for hit, row in _read(rows)}

    filled, found = retrieval._thin([], SCANNED, read, reranked, where, QUERY, grows=True)

    assert found == signal, name
    assert filled.ranges == [], f"{name}: no thin range, nothing grows"


def test_reranker_scores_are_valued_around_the_scanned_hits_scores() -> None:
    """The scanned hits carry the reranker's scores already (3, 2, 1): 2 is worth 0, 3 is 1."""
    assert retrieval._valued([4.0, -2.0, 2.5], [3.0, 2.0, 1.0]) == [1.0, -1.0, 0.5]


# --- what a chunk near a passage is worth ---------------------------------------------------

# two kept chunks and two near them, each with a vector and its words
WEIGHED = {
    ("backend", "doc.md", seq): (
        msgspec.structs.replace(SCANNED.hits[0], seq=seq, text=text),
        {"document": "doc.md", "seq": seq, "text": text, "vector": vector},
    )
    for seq, text, vector in [
        (1, "retries are idempotent", [1.0, 0.0]),
        (2, "retries back off", [0.8, 0.6]),
        (3, "idempotent retries again", [0.95, 0.3122]),
        (4, "jitter spreads the load", [0.0, 1.0]),
    ]
}
HELD = [("backend", "doc.md", 1), ("backend", "doc.md", 2)]
NEAR = [("backend", "doc.md", 3), ("backend", "doc.md", 4)]


@pytest.mark.parametrize(
    ("name", "questions", "rows", "signal", "values", "aspects"),
    [
        (
            "by the vector: 0 at the median kept chunk, 1 at the best",
            [probe.Question([1.0, 0.0], "retries idempotent")],
            WEIGHED,
            "vector",
            [0.5, -1.0],
            [None, None],
        ),
        (
            "without a query vector, by the question's words",
            [probe.Question(None, "idempotent retries")],
            WEIGHED,
            "words",
            [1.0, -1.0],
            [None, None],
        ),
        (
            "each chunk takes its best question, which tags it",
            [
                probe.Question(None, "idempotent retries", label="a"),
                probe.Question(None, "jitter load", label="b"),
            ],
            WEIGHED,
            "words",
            [1.0, 1.0],
            ["a", "b"],
        ),
    ],
)
def test_a_chunk_near_a_passage_is_weighed_against_the_kept_chunks(
    name: str,
    questions: list[probe.Question],
    rows: dict,
    signal: str,
    values: list[float],
    aspects: list[str | None],
) -> None:
    weighed, found = retrieval._weigh(HELD, NEAR, rows, questions)

    assert found == signal, name
    assert [weighed[key].value for key in NEAR] == pytest.approx(values, abs=1e-3), name
    assert [weighed[key].aspect for key in NEAR] == aspects, name


def test_nothing_near_or_nothing_held_weighs_nothing() -> None:
    question = [probe.Question(None, "retries")]

    assert retrieval._weigh(HELD, [], WEIGHED, question) == ({}, "none")
    assert retrieval._weigh([], NEAR, WEIGHED, question) == ({}, "none")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "others", "after", "grown"),
    [
        (
            "one it rates above the median scanned hit joins",
            [1.0],
            {3: 5.0},
            [(2, 3)],
        ),
        (
            "a weak neighbour is not paid for by a slightly better one past it, as on the logits",
            [5.0, 2.0, 0.0, -2.0, -4.0, -5.0, -5.0, -6.0, -7.0, -8.0, -9.0],
            {3: -10.0, 4: -3.0},
            [(2, 2)],
        ),
    ],
)
async def test_with_a_reranker_a_thin_range_grows_by_the_neighbours_it_scores(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    others: list[float],
    after: dict[int, float],
    grown: list[tuple[int, int]],
) -> None:
    """The reranker scored the ranking, so it scores the neighbours too, valued on its logits
    around the scanned hits': the sigmoid it stores would make every weak one cost nearly 0."""
    thin = msgspec.structs.replace(
        hit("Retries back off.", _sigmoid(8.0 if len(others) > 1 else 3.0), seq=2, char_start=200),
        start_reason=CutReason.PARAGRAPH,
        end_reason=CutReason.PARAGRAPH,
    )
    rest = [
        msgspec.structs.replace(
            hit("Keys dedupe.", _sigmoid(score), document=f"other{n}.md", seq=6, char_start=600),
            start_reason=CutReason.PARAGRAPH,
            end_reason=CutReason.PARAGRAPH,
        )
        for n, score in enumerate(others)
    ]
    scanned = Scanned(hits=[thin, *rest], vectors=[None] * (1 + len(rest)))
    near = {
        seq: msgspec.structs.replace(
            thin, seq=seq, text=f"Jitter spreads them {seq}.", char_start=seq * 100, score=0.0
        )
        for seq in after
    }

    async def rows_at(where: Plan, wanted: set) -> dict:
        found = {
            chunk_key(one): (one, {"document": one.document, "seq": one.seq, "text": one.text})
            for one in near.values()
        }
        return {key: row for key, row in found.items() if key in wanted}

    async def rerank(query: str, rows: list[dict], settings: SearchSettings) -> list[dict]:
        return [{**one, "_relevance_score": _sigmoid(after[one["seq"]])} for one in rows]

    monkeypatch.setattr(retrieval, "_rows_at", rows_at)
    monkeypatch.setattr(retrieval, "cross_encode", rerank)

    ranged = await retrieval.fill_thin(scanned, _plan(Reranker.CROSS_ENCODER), QUERY, QUERY)

    ours = [one for one in ranged.ranges if one.hits[0].document == thin.document]
    assert [(one.seq_start, one.seq_end) for one in ours] == grown, name


def test_the_budget_cuts_the_last_sections_first() -> None:
    groups = [
        section.Group(
            "backend",
            doc,
            section.Section(("Retries",), 1, 1),
            ranges([hit(text, 1.0, document=doc)]),
        )
        for doc, text in [("a.md", "x" * 50), ("b.md", "y" * 50), ("c.md", "z" * 50)]
    ]
    where = Plan(
        settings=SearchSettings(max_answer_chars=120), indexes=[], vector=None, embedding=None
    )

    kept = retrieval.budget(groups, where)

    assert [one.document for one in kept] == ["a.md", "b.md"]
