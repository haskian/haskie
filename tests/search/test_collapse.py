"""Near-duplicates folded into `also_in`: which results fold, which stay, and what the kept one
says about the ones it stands for.

Every hit below is a real sentence of a backend book, placed at its real offsets in its own
document. The word space is the one a search without embeddings compares in; the embedding cases
hand in unit vectors built so the cosines are the ones each case is about.
"""

import msgspec
import numpy as np
import pytest

from haskie.collection.index import Hit, location
from haskie.search import collapse
from haskie.search.collapse import MAX_ALSO, Embedded, Worded
from haskie.search.passage import HitRange, ranges
from haskie.settings import PROFILES, DuplicateCosine, EmbeddingModel, EmbeddingProfile

RETRY = "A background job retries a failed HTTP call, so the call has to be idempotent."
REWORDED = "Make the side effect safe to repeat, because the job may run the request twice."
BACKOFF = "Exponential backoff with jitter spreads the retries and avoids a thundering herd."
CLOCKS = "Never trust a wall clock for ordering: hosts drift apart by milliseconds."
OUTBOX = "A transactional outbox writes the event in the same transaction as the state change."
# RETRY with a sentence either side: a fuller passage that holds the whole of RETRY
FULLER = f"Retries are where most duplicate side effects come from. {RETRY} Key it on a request id."
# RETRY and BACKOFF in one chunk: a fuller passage that holds two results kept apart
BOTH = f"{RETRY} {BACKOFF}"

BGE = DuplicateCosine(chunk=0.92, passage=0.95)
BGE_SMALL = PROFILES[EmbeddingProfile.COMPACT]


def _hit(
    text: str,
    score: float,
    *,
    document: str = "patterns.md",
    collection: str = "backend",
    seq: int = 1,
    char_start: int = 0,
) -> Hit:
    """One indexed chunk: `text` at `char_start` of `document`, on the line its offset gives."""
    line = char_start // 80 + 1
    return Hit(
        collection=collection,
        document=document,
        source_path=f"documents/{document}",
        markdown_path=f"documents/{document}.md",
        part=0,
        seq=seq,
        line_start=line,
        line_end=line,
        char_start=char_start,
        char_end=char_start + len(text),
        byte_start=char_start,
        byte_end=char_start + len(text),
        page_start=None,
        page_end=None,
        headings=["Messaging", "Retries"],
        frame=["Messaging", "Retries"],
        header="Messaging > Retries",
        location=location(document, None, None, line, line),
        text=text,
        score=score,
        source_file=f"/home/documents/{document}",
        markdown_file=f"/home/documents/{document}.md",
    )


def _words(hits: list[Hit]) -> Worded:
    return Worded([hit.text for hit in hits])


def _unit(*rows: list[float]) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float64)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _shape(kept: list[Hit]) -> list[tuple[str, int, float, list[str]]]:
    """What a case asserts: each kept hit's document, position and slot score, and what folded
    into it."""
    return [(h.document, h.seq, h.score, [ref.document for ref in h.also_in]) for h in kept]


# --- hits -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "found", "limit", "expected"),
    [
        ("nothing scanned, nothing kept", [], 3, []),
        (
            "distinct hits are all kept, best first",
            [
                _hit(RETRY, 0.9, document="a.md"),
                _hit(BACKOFF, 0.8, document="b.md"),
                _hit(CLOCKS, 0.7, document="c.md"),
            ],
            3,
            [("a.md", 1, 0.9, []), ("b.md", 1, 0.8, []), ("c.md", 1, 0.7, [])],
        ),
        (
            "a copy in another document folds into the better hit, and the next one takes its slot",
            [
                _hit(RETRY, 0.9, document="a.md"),
                _hit(RETRY, 0.8, document="copy.md"),
                _hit(BACKOFF, 0.7, document="b.md"),
            ],
            2,
            [("a.md", 1, 0.9, ["copy.md"]), ("b.md", 1, 0.7, [])],
        ),
        (
            "a rewording stays: words catch copies, not paraphrases",
            [_hit(RETRY, 0.9, document="a.md"), _hit(REWORDED, 0.8, document="b.md")],
            2,
            [("a.md", 1, 0.9, []), ("b.md", 1, 0.8, [])],
        ),
        (
            "two neighbouring chunks of one document never fold: they are one passage",
            [_hit(RETRY, 0.9, seq=4), _hit(RETRY, 0.8, seq=5)],
            2,
            [("patterns.md", 4, 0.9, []), ("patterns.md", 5, 0.8, [])],
        ),
        (
            "two distant chunks of one document fold when one repeats the other",
            [_hit(RETRY, 0.9, seq=4), _hit(RETRY, 0.8, seq=40, char_start=4000)],
            2,
            [("patterns.md", 4, 0.9, ["patterns.md"])],
        ),
        (
            "one document chunked two ways folds on the characters the two spans share",
            [
                _hit(RETRY, 0.9, collection="backend", seq=3, char_start=200),
                _hit(BACKOFF, 0.8, collection="ops", seq=2, char_start=230),
            ],
            2,
            [("patterns.md", 3, 0.9, ["patterns.md"])],
        ),
        (
            "one document chunked two ways stays apart where the spans barely touch",
            [
                _hit(RETRY, 0.9, collection="backend", seq=3, char_start=200),
                _hit(BACKOFF, 0.8, collection="ops", seq=2, char_start=270),
            ],
            2,
            [("patterns.md", 3, 0.9, []), ("patterns.md", 2, 0.8, [])],
        ),
        (
            "a repeat found past the limit still folds, and a distinct one gets no slot",
            [
                _hit(RETRY, 0.9, document="a.md"),
                _hit(BACKOFF, 0.8, document="b.md"),
                _hit(RETRY, 0.7, document="copy.md"),
            ],
            1,
            [("a.md", 1, 0.9, ["copy.md"])],
        ),
        (
            "a fuller hit further down takes the slot, and the score, of the one it contains",
            [_hit(RETRY, 0.9, document="a.md"), _hit(FULLER, 0.8, document="book.md")],
            2,
            [("book.md", 1, 0.9, ["a.md"])],
        ),
        (
            "a hit holding two kept ones takes the best slot, and the freed slot goes to the next",
            [
                _hit(RETRY, 0.9, document="a.md"),
                _hit(BACKOFF, 0.8, document="b.md"),
                _hit(BOTH, 0.7, document="book.md"),
                _hit(CLOCKS, 0.6, document="c.md"),
            ],
            2,
            [("book.md", 1, 0.9, ["a.md", "b.md"]), ("c.md", 1, 0.6, [])],
        ),
    ],
)
def test_hits_fold_near_duplicates_in_words(
    name: str, found: list[Hit], limit: int, expected: list[tuple[str, int, float, list[str]]]
) -> None:
    kept = collapse.hits(found, _words(found), limit)

    assert _shape(kept) == expected, name
    assert all(hit.also_count == len(hit.also_in) for hit in kept), f"{name}: under the cap"


def test_a_fold_records_where_the_repeat_is_and_how_close_it_was() -> None:
    found = [_hit(RETRY, 0.9, document="a.md"), _hit(RETRY, 0.8, document="copy.md", seq=7)]

    (kept,) = collapse.hits(found, _words(found), 2)

    (reference,) = kept.also_in
    assert (reference.collection, reference.document, reference.seq) == ("backend", "copy.md", 7)
    assert (reference.header, reference.location) == (found[1].header, found[1].location)
    assert reference.score == 0.8, "its own score, before it was folded"
    assert reference.similarity == 1.0, "every word of it is in the kept hit"
    assert kept.score == 0.9, "agreement does not raise the kept hit's score"


def test_also_in_lists_the_best_few_and_counts_them_all() -> None:
    found = [_hit(RETRY, 1.0 - i / 100, document=f"copy{i}.md") for i in range(MAX_ALSO + 3)]

    (kept,) = collapse.hits(found, _words(found), 3)

    assert kept.document == "copy0.md"
    assert [ref.document for ref in kept.also_in] == [f"copy{i}.md" for i in range(1, 6)]
    assert kept.also_count == MAX_ALSO + 2, "the total, past what is listed"


@pytest.mark.parametrize(
    ("name", "cosine", "folds"),
    [
        ("a cosine above the model's bar folds", 0.93, True),
        ("a cosine just under the bar stays", 0.91, False),
        ("a cosine below the bar stays", 0.80, False),
    ],
)
def test_hits_fold_near_duplicates_in_embeddings(name: str, cosine: float, folds: bool) -> None:
    found = [_hit(RETRY, 0.9, document="a.md"), _hit(REWORDED, 0.8, document="b.md")]
    vectors = _unit([1.0, 0.0], [cosine, float(np.sqrt(1 - cosine**2))])

    kept = collapse.hits(found, Embedded(vectors, BGE, [hit.text for hit in found]), 2)

    assert len(kept) == (1 if folds else 2), name


def test_identical_text_under_other_headings_folds_though_its_cosine_misses() -> None:
    """Each chunk is embedded under its heading path, so one paragraph in two books embeds apart:
    0.927 was measured with bge-small. The words still say it is a copy."""
    found = [
        _hit(RETRY, 0.9, document="book.md"),
        msgspec.structs.replace(_hit(RETRY, 0.8, document="notes.md"), header="Delivery"),
    ]
    vectors = _unit([1.0, 0.0], [0.90, float(np.sqrt(1 - 0.90**2))])

    (kept,) = collapse.hits(found, Embedded(vectors, BGE, [hit.text for hit in found]), 2)

    assert [(ref.document, ref.similarity) for ref in kept.also_in] == [("notes.md", 1.0)]


# --- ranges ---------------------------------------------------------------------------


def _passage(texts: list[str], document: str, first_seq: int, score: float) -> list[Hit]:
    """Consecutive chunks of one document: what `passage.ranges` merges into one range."""
    return [
        _hit(text, score, document=document, seq=first_seq + i, char_start=100 * i)
        for i, text in enumerate(texts)
    ]


def _ranges(scanned: list[Hit], vectors: np.ndarray, limit: int) -> list[HitRange]:
    where = Embedded(vectors, BGE, [hit.text for hit in scanned])
    return collapse.ranges(ranges(scanned), scanned, where, limit)


E1, E2, E3, E4 = (list(row) for row in np.eye(4))


@pytest.mark.parametrize(
    ("name", "big_score", "small_score", "small", "small_vectors", "expected"),
    [
        (
            "chunk by chunk: a one-chunk copy of the middle of a three-chunk passage folds into it",
            0.9,
            0.5,
            [REWORDED],
            [E2],
            [("book.md", 1, 3, ["note.md"])],
        ),
        (
            "chunk by chunk: the three-chunk passage takes the slot of the copy that outranked it",
            0.5,
            0.9,
            [REWORDED],
            [E2],
            [("book.md", 1, 3, ["note.md"])],
        ),
        (
            "passage by passage: two chunks that match none singly but sum to the same point fold",
            0.9,
            0.5,
            [REWORDED, OUTBOX],
            [[1.0, 1.0, 0.9, 0.0], [1.0, 1.0, 1.1, 0.0]],
            [("book.md", 1, 3, ["note.md"])],
        ),
        (
            "a passage on something else stays",
            0.9,
            0.5,
            [OUTBOX],
            [E4],
            [("book.md", 1, 3, []), ("note.md", 10, 10, [])],
        ),
    ],
)
def test_ranges_fold_by_containment_or_by_their_mean_vectors(
    name: str,
    big_score: float,
    small_score: float,
    small: list[str],
    small_vectors: list[list[float]],
    expected: list[tuple[str, int, int, list[str]]],
) -> None:
    """The case from the design session: a mean vector of three chunks sits far from any one of
    them, so the passage-level cosine alone would keep a copy of its middle chunk."""
    big = _passage([BACKOFF, RETRY, CLOCKS], "book.md", 1, big_score)
    scanned = big + _passage(small, "note.md", 10, small_score)
    vectors = _unit(E1, E2, E3, *small_vectors)

    kept = _ranges(scanned, vectors, 2)

    shape = [
        (r.hits[0].document, r.seq_start, r.seq_end, [ref.document for ref in r.also_in])
        for r in kept
    ]
    assert shape == expected, name


def test_a_folded_range_points_at_its_own_lines() -> None:
    big = _passage([BACKOFF, RETRY, CLOCKS], "book.md", 1, 0.9)
    small = _passage([RETRY], "note.md", 10, 0.5)
    scanned = big + small

    (kept,) = _ranges(scanned, _unit(E1, E2, E3, E2), 2)

    (reference,) = kept.also_in
    (small_range,) = ranges(small)
    assert (reference.document, reference.seq_start, reference.seq_end) == ("note.md", 10, 10)
    assert reference.location == location(
        "note.md", None, None, small_range.line_start, small_range.line_end
    )
    assert reference.similarity == pytest.approx(1.0), "its one chunk is the passage's middle"
    assert kept.also_count == 1


# --- the space ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "vectors", "model", "expected"),
    [
        (
            "every row carries a vector under a model with thresholds",
            [E1, E2],
            BGE_SMALL,
            "embedding",
        ),
        (
            "one row without a vector sends the whole scan to words",
            [E1, None],
            BGE_SMALL,
            "words",
        ),
        ("no embedding model at all", [None, None], None, "words"),
        ("a model without thresholds", [E1, E2], EmbeddingModel("test/tiny", 2), "words"),
        ("nothing scanned", [], BGE_SMALL, "words"),
    ],
)
def test_the_space_is_embeddings_only_when_every_row_can_be_compared(
    name: str, vectors: list, model: EmbeddingModel | None, expected: str
) -> None:
    texts = [RETRY, BACKOFF][: len(vectors)]

    assert collapse.space(texts, vectors, model).kind == expected, name


def test_folding_leaves_the_hits_it_was_given_alone() -> None:
    found = [_hit(RETRY, 0.9, document="a.md"), _hit(RETRY, 0.8, document="copy.md")]
    before = msgspec.json.encode(found)

    collapse.hits(found, _words(found), 2)

    assert msgspec.json.encode(found) == before, "a pure fold: the scan is not rewritten"
