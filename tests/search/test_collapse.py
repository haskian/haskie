"""Near-duplicates folded into `also_in`: which results fold, which stay, and what the kept one
says about the ones it stands for.

Every hit below is a real sentence of a backend book, placed at its real offsets in its own
document. The word space is the one a search without embeddings compares in; the embedding cases
hand in unit vectors built so the cosines are the ones each case is about.
"""

from collections.abc import Callable

import msgspec
import numpy as np
import pytest
from conftest import compact_model, hit, words_scan

from haskie.catalogue.catalogue import EmbeddingModel
from haskie.collection.index import Hit, Overlap, Relation, location
from haskie.indexing import chunk
from haskie.search import collapse, section
from haskie.search.passage import HitRange, ranges
from haskie.settings import Chunker, ChunkSettings, ScoreFold, SearchMode

HARMONIC = ScoreFold.HARMONIC  # the rule these cases were written against

RETRY = "A background job retries a failed HTTP call, so the call has to be idempotent."
REWORDED = "Make the side effect safe to repeat, because the job may run the request twice."
BACKOFF = "Exponential backoff with jitter spreads the retries and avoids a thundering herd."
CLOCKS = "Never trust a wall clock for ordering: hosts drift apart by milliseconds."
OUTBOX = "A transactional outbox writes the event in the same transaction as the state change."
# RETRY with a sentence either side: a fuller passage that holds the whole of RETRY
FULLER = f"Retries are where most duplicate side effects come from. {RETRY} Key it on a request id."
# A chapter that mentions an appendix, and a chunk that is nothing but the appendix's heading,
# as the converter left it: the case where single words made the heading "contained" at 1.00
MENTIONS = (
    "The domain is the set of activities that those processes support. We show the Docker "
    "configuration in Appendix D, and you'd find the rest of the setup there too."
)

HEADING = "## <u>APPENDIX D</u>"
# RETRY and BACKOFF in one chunk: a fuller passage that holds two results kept apart
BOTH = f"{RETRY} {BACKOFF}"
# RETRY with one word changed: 10 of its 13 word 3-grams either way, 12 of 14 distinct words
NEAR = RETRY.replace("failed", "broken")
# a span of one document around RETRY's, cut by another collection, whose words share nothing
# with it: only the spans can say that one holds the other
AROUND = f"{OUTBOX} {CLOCKS}"


@pytest.fixture
async def bge_small() -> EmbeddingModel:
    return await compact_model()


def _embedded(
    hits: list[Hit],
    vectors: np.ndarray,
    model: EmbeddingModel,
    mode: SearchMode = SearchMode.HYBRID,
) -> collapse.Scan:
    """The spaces of a `mode` search under `model` whose rows carry `vectors`."""
    return collapse.spaces([hit.text for hit in hits], vectors.tolist(), model, mode)


def _heading(text: str, score: float, **fields) -> Hit:
    """A chunk of one heading line and nothing else, as the chunker really cuts it. Only the text
    chunker does: the markdown one reads the line as a heading and makes no chunk of a section
    without text (`segment.pack`), so a heading reaches the fold as text of its own."""
    (alone,) = chunk.split(text, ChunkSettings(chunker=Chunker.TEXT))
    return msgspec.structs.replace(hit(alone.text, score, **fields), layout=alone.layout)


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
                hit(RETRY, 0.9, document="a.md"),
                hit(BACKOFF, 0.8, document="b.md"),
                hit(CLOCKS, 0.7, document="c.md"),
            ],
            3,
            [("a.md", 1, 0.9, []), ("b.md", 1, 0.8, []), ("c.md", 1, 0.7, [])],
        ),
        (
            "a copy in another document folds into the better hit, and the next one takes its slot",
            [
                hit(RETRY, 0.9, document="a.md"),
                hit(RETRY, 0.8, document="copy.md"),
                hit(BACKOFF, 0.7, document="b.md"),
            ],
            2,
            [("a.md", 1, 0.9, ["copy.md"]), ("b.md", 1, 0.7, [])],
        ),
        (
            "a rewording stays: words catch copies, not paraphrases",
            [hit(RETRY, 0.9, document="a.md"), hit(REWORDED, 0.8, document="b.md")],
            2,
            [("a.md", 1, 0.9, []), ("b.md", 1, 0.8, [])],
        ),
        (
            "two neighbouring chunks of one document never fold: they are one passage",
            [hit(RETRY, 0.9, seq=4), hit(RETRY, 0.8, seq=5)],
            2,
            [("patterns.md", 4, 0.9, []), ("patterns.md", 5, 0.8, [])],
        ),
        (
            "two distant chunks of one document fold when one repeats the other",
            [hit(RETRY, 0.9, seq=4), hit(RETRY, 0.8, seq=40, char_start=4000)],
            2,
            [("patterns.md", 4, 0.9, ["patterns.md"])],
        ),
        (
            "one document chunked two ways: a partial overlap is no fold on its own",
            [
                hit(RETRY, 0.9, collection="backend", seq=3, char_start=200),
                hit(BACKOFF, 0.8, collection="ops", seq=2, char_start=230),
            ],
            2,
            [("patterns.md", 3, 0.9, []), ("patterns.md", 2, 0.8, [])],
        ),
        (
            "one document chunked two ways stays apart where the spans barely touch",
            [
                hit(RETRY, 0.9, collection="backend", seq=3, char_start=200),
                hit(BACKOFF, 0.8, collection="ops", seq=2, char_start=270),
            ],
            2,
            [("patterns.md", 3, 0.9, []), ("patterns.md", 2, 0.8, [])],
        ),
        (
            "a repeat found past the limit still folds, and a distinct one gets no slot",
            [
                hit(RETRY, 0.9, document="a.md"),
                hit(BACKOFF, 0.8, document="b.md"),
                hit(RETRY, 0.7, document="copy.md"),
            ],
            1,
            [("a.md", 1, 0.9, ["copy.md"])],
        ),
        (
            "a fuller hit further down takes the slot, and the score, of the one it contains",
            [hit(RETRY, 0.9, document="a.md"), hit(FULLER, 0.8, document="book.md")],
            2,
            [("book.md", 1, 0.9, ["a.md"])],
        ),
        (
            "a bare heading is never contained, though a passage holds every word of it",
            [hit(MENTIONS, 0.9, document="book.md"), _heading(HEADING, 0.8, document="book.pdf")],
            2,
            [("book.md", 1, 0.9, []), ("book.pdf", 1, 0.8, [])],
        ),
        (
            "one heading in two books is no duplicate: too short to be a point",
            [
                _heading("## Summary", 0.9, document="a.md"),
                _heading("## Summary", 0.8, document="b.md"),
            ],
            2,
            [("a.md", 1, 0.9, []), ("b.md", 1, 0.8, [])],
        ),
        (
            "a hit holding two kept ones takes the best slot, and the freed slot goes to the next",
            [
                hit(RETRY, 0.9, document="a.md"),
                hit(BACKOFF, 0.8, document="b.md"),
                hit(BOTH, 0.7, document="book.md"),
                hit(CLOCKS, 0.6, document="c.md"),
            ],
            2,
            [("book.md", 1, 0.9, ["a.md", "b.md"]), ("c.md", 1, 0.6, [])],
        ),
    ],
)
def test_hits_fold_near_duplicates_in_words(
    name: str, found: list[Hit], limit: int, expected: list[tuple[str, int, float, list[str]]]
) -> None:
    kept = collapse.hits(found, words_scan(found), limit)

    assert _shape(kept) == expected, name


def test_a_fold_records_where_the_repeat_is_and_how_close_it_was() -> None:
    found = [hit(RETRY, 0.9, document="a.md"), hit(RETRY, 0.8, document="copy.md", seq=7)]

    (kept,) = collapse.hits(found, words_scan(found), 2)

    (reference,) = kept.also_in
    assert (reference.collection, reference.document, reference.seq) == ("backend", "copy.md", 7)
    assert (reference.header, reference.location) == (found[1].header, found[1].location)
    assert (reference.line_start, reference.line_end) == (found[1].line_start, found[1].line_end)
    assert reference.score == 0.8, "its own score, before it was folded"
    assert reference.to_parent.words.contained == 1.0, "every word of it is in the kept hit"
    assert reference.to_root == reference.to_parent, "listed under the hit, its parent is the root"
    assert (reference.to_parent.embedding, reference.to_parent.chars) == (None, None), (
        "no vectors in a words-only scan, and no shared characters across two documents"
    )
    assert kept.score == 0.9, "agreement does not raise the kept hit's score"


def _tree(references: list) -> list:
    """What a tree case asserts: each place's document and relation to its parent, and what sits
    under it."""
    return [(r.document, r.relation, _tree(r.also_in)) for r in references]


@pytest.mark.parametrize(
    ("name", "found", "expected"),
    [
        (
            "a copy in another document is a duplicate: the same text",
            [hit(RETRY, 0.9, document="a.md"), hit(RETRY, 0.8, document="copy.md")],
            [("a.md", [("copy.md", Relation.DUPLICATE, [])])],
        ),
        (
            "a copy laid out another way is still a duplicate: whitespace is not text",
            [
                hit(RETRY, 0.9, document="a.md"),
                hit(RETRY.replace(", so ", ",\n  so "), 0.8, document="copy.md"),
            ],
            [("a.md", [("copy.md", Relation.DUPLICATE, [])])],
        ),
        (
            "one word changed is equivalent: the same point, other wording",
            [hit(RETRY, 0.9, document="a.md"), hit(NEAR, 0.8, document="near.md")],
            [("a.md", [("near.md", Relation.EQUIVALENT, [])])],
        ),
        (
            "a hit inside a fuller one ranked above it is contained",
            [hit(FULLER, 0.9, document="book.md"), hit(RETRY, 0.8, document="a.md")],
            [("book.md", [("a.md", Relation.CONTAINED, [])])],
        ),
        (
            "a hit a fuller one below it swapped out is contained in it",
            [hit(RETRY, 0.9, document="a.md"), hit(FULLER, 0.8, document="book.md")],
            [("book.md", [("a.md", Relation.CONTAINED, [])])],
        ),
        (
            "after a swap, what the old hit held stays under it",
            [
                hit(RETRY, 0.9, document="a.md"),
                hit(RETRY, 0.85, document="copy.md"),
                hit(FULLER, 0.8, document="book.md"),
            ],
            [("book.md", [("a.md", Relation.CONTAINED, [("copy.md", Relation.DUPLICATE, [])])])],
        ),
        (
            "two swaps nest two levels: each old hit under the one that took its slot",
            [
                hit(RETRY, 0.9, document="a.md"),
                hit(FULLER, 0.85, document="book.md"),
                hit(f"{FULLER} {OUTBOX}", 0.8, document="guide.md"),
            ],
            [
                (
                    "guide.md",
                    [("book.md", Relation.CONTAINED, [("a.md", Relation.CONTAINED, [])])],
                )
            ],
        ),
        (
            "one document chunked two ways: a span inside another is contained, by the spans alone",
            [
                hit(RETRY, 0.9, collection="backend", seq=3, char_start=200),
                hit(AROUND, 0.8, collection="ops", seq=2, char_start=150),
            ],
            [("patterns.md", [("patterns.md", Relation.CONTAINED, [])])],
        ),
    ],
)
def test_a_place_sits_under_the_result_it_was_folded_into(
    name: str, found: list[Hit], expected: list[tuple[str, list]]
) -> None:
    kept = collapse.hits(found, words_scan(found), 3)

    assert [(h.document, _tree(h.also_in)) for h in kept] == expected, name


@pytest.mark.parametrize(
    ("name", "found", "expected"),
    [
        (
            "a sentence inside a fuller paragraph: all of it in there, under half of that in it",
            [hit(FULLER, 0.9, document="book.md"), hit(RETRY, 0.8, document="a.md")],
            # 13 of RETRY's 13 word 3-grams, 13 of FULLER's 28; 13 of 26 distinct words shared;
            # the score is the Dice coefficient of the 3-grams, 2 * 13 / (13 + 28)
            (Overlap(1.0, 13 / 28, 0.5, score=26 / 41), None),
        ),
        (
            "a copy: each wholly in the other",
            [hit(RETRY, 0.9, document="a.md"), hit(RETRY, 0.8, document="copy.md")],
            (Overlap(contained=1.0, contains=1.0, alike=1.0, score=1.0), None),
        ),
        (
            "one document chunked two ways: the characters both spans cover",
            [
                hit(RETRY, 0.9, collection="backend", seq=3, char_start=200),
                hit(AROUND, 0.8, collection="ops", seq=2, char_start=150),
            ],
            # no word 3-gram in common, 2 of 34 distinct words, so a score of 0; RETRY's 78
            # characters all lie in AROUND's span
            (Overlap(contained=0.0, contains=0.0, alike=2 / 34, score=0.0), 1.0),
        ),
    ],
)
def test_a_place_measures_itself_against_its_parent_both_ways_and_as_a_whole(
    name: str, found: list[Hit], expected: tuple[Overlap, float | None]
) -> None:
    (kept,) = collapse.hits(found, words_scan(found), 2)

    (reference,) = kept.also_in
    words, chars = expected
    measured = msgspec.structs.astuple(reference.to_parent.words)
    assert measured == pytest.approx(msgspec.structs.astuple(words)), name
    assert reference.to_parent.chars == chars, name


@pytest.mark.parametrize(
    ("name", "mode", "expected"),
    [
        ("a hybrid search folds a near copy on its words", SearchMode.HYBRID, ["near.md"]),
        ("a full-text search does the same", SearchMode.FTS, ["near.md"]),
        ("a vector search folds by its vectors alone: apart, so both stay", SearchMode.VECTOR, []),
    ],
)
@pytest.mark.anyio
async def test_words_decide_a_fold_as_they_rank_the_search(
    name: str, mode: SearchMode, expected: list[str], bge_small: EmbeddingModel
) -> None:
    """NEAR is RETRY with one word changed, and its vector is set far from RETRY's: only the words
    can fold it. An exact copy would fold in every mode: a duplicate needs no space."""
    found = [hit(RETRY, 0.9, document="a.md"), hit(NEAR, 0.8, document="near.md")]

    kept = collapse.hits(found, _embedded(found, _unit([1.0, 0.0], [0.0, 1.0]), bge_small, mode), 2)

    assert [ref.document for ref in kept[0].also_in] == expected, name
    assert kept[0].also_in == [] or kept[0].also_in[0].to_parent.embedding is not None, (
        "measured by vectors in every mode"
    )


@pytest.mark.anyio
async def test_a_repeat_the_new_leader_does_not_place_stays_under_the_one_it_repeats(
    bge_small: EmbeddingModel,
) -> None:
    """REWORDED shares no words with RETRY, but the model embeds the two alike, so it folds under
    RETRY as equivalent. FULLER then holds RETRY word for word and takes the slot. REWORDED stays
    under RETRY, the place it repeats, and says how far it is from FULLER as well."""
    found = [
        hit(RETRY, 0.9, document="a.md"),
        hit(REWORDED, 0.85, document="reworded.md"),
        hit(FULLER, 0.8, document="book.md"),
    ]
    spaces = _embedded(found, _unit([1.0, 0.0, 0.0], [0.99, 0.14, 0.0], [0.0, 0.0, 1.0]), bge_small)

    (kept,) = collapse.hits(found, spaces, 3)

    assert kept.document == "book.md"
    assert _tree(kept.also_in) == [
        ("a.md", Relation.CONTAINED, [("reworded.md", Relation.EQUIVALENT, [])])
    ]
    (retry,) = kept.also_in
    (reworded,) = retry.also_in
    assert reworded.to_parent.embedding is not None and reworded.to_root.embedding is not None
    assert reworded.to_parent.embedding.alike == pytest.approx(0.99, abs=0.01), "to RETRY"
    assert reworded.to_root.embedding.alike == pytest.approx(0.0, abs=0.01), "to FULLER"
    assert reworded.to_root.words.contained == 0.0, "no three words of it in FULLER"
    assert retry.to_parent == retry.to_root, "RETRY sits right under the root"
    assert collapse.places(kept.also_in) == 2, "every place, at every level"


@pytest.mark.parametrize(
    ("name", "fold"),
    [
        ("hits", lambda found, spaces: collapse.hits(found, spaces, 3)),
        (
            "ranges",
            lambda found, spaces: collapse.ranges(ranges(found, how=HARMONIC), found, spaces, 3),
        ),
    ],
)
@pytest.mark.anyio
async def test_a_place_stays_under_the_chunk_it_repeats_though_another_cites_the_same_lines(
    name: str, fold: Callable, bge_small: EmbeddingModel
) -> None:
    """Chunks cut from one long line all cite that line. RETRY is found twice on line 1 of a.md,
    and REWORDED folds under the first by its vector before FULLER takes the slot: the tree keeps
    it under that chunk, where a citation of the line could not say which."""
    first_chunk = hit(RETRY, 0.9, document="a.md", seq=1)
    # the same sentence again further along the same line: other characters, the same citation
    later_chunk = msgspec.structs.replace(
        hit(RETRY, 0.88, document="a.md", seq=40, char_start=400),
        line_start=first_chunk.line_start,
        line_end=first_chunk.line_end,
        location=first_chunk.location,
    )
    found = [
        first_chunk,
        later_chunk,
        hit(REWORDED, 0.85, document="reworded.md"),
        hit(FULLER, 0.8, document="book.md"),
    ]
    vectors = _unit([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.99, 0.14, 0.0], [0.0, 0.0, 1.0])

    (kept,) = fold(found, _embedded(found, vectors, bge_small))

    (first,) = kept.also_in
    second, reworded = first.also_in
    assert first.location == second.location, f"{name}: two places, one citation"
    assert (reworded.document, reworded.relation) == ("reworded.md", Relation.EQUIVALENT), name
    assert second.to_parent.chars == 0.0, f"{name}: one document, spans apart"


def test_every_repeat_is_listed_in_the_same_document_or_another() -> None:
    """Two copies in one other book and a repeat further down the same document: three places."""
    found = [
        hit(RETRY, 0.9, document="book.md", seq=4),
        hit(RETRY, 0.8, document="notes.md", seq=2),
        hit(RETRY, 0.7, document="notes.md", seq=9, char_start=900),
        hit(RETRY, 0.6, document="book.md", seq=40, char_start=4000),
    ]

    (kept,) = collapse.hits(found, words_scan(found), 3)

    assert [(ref.document, ref.seq) for ref in kept.also_in] == [
        ("notes.md", 2),
        ("notes.md", 9),
        ("book.md", 40),
    ], "three places, best first; the UI counts the other documents among them: one"


def test_also_in_lists_every_repeat_uncapped() -> None:
    found = [hit(RETRY, 1.0 - i / 100, document=f"copy{i}.md") for i in range(12)]

    (kept,) = collapse.hits(found, words_scan(found), 3)

    assert kept.document == "copy0.md"
    assert [ref.document for ref in kept.also_in] == [f"copy{i}.md" for i in range(1, 12)]


@pytest.mark.parametrize(
    ("name", "cosine", "folds"),
    [
        ("a cosine above the model's bar folds", 0.93, True),
        ("a cosine just under the bar stays", 0.91, False),
        ("a cosine below the bar stays", 0.80, False),
    ],
)
@pytest.mark.anyio
async def test_hits_fold_near_duplicates_in_embeddings(
    name: str, cosine: float, folds: bool, bge_small: EmbeddingModel
) -> None:
    found = [hit(RETRY, 0.9, document="a.md"), hit(REWORDED, 0.8, document="b.md")]
    vectors = _unit([1.0, 0.0], [cosine, float(np.sqrt(1 - cosine**2))])

    kept = collapse.hits(found, _embedded(found, vectors, bge_small), 2)

    assert len(kept) == (1 if folds else 2), name


@pytest.mark.anyio
async def test_identical_text_under_other_headings_folds_though_its_cosine_misses(
    bge_small: EmbeddingModel,
) -> None:
    """Each chunk is embedded under its heading path, so one paragraph in two books embeds apart:
    0.927 was measured with bge-small. The words still say it is a copy."""
    found = [
        hit(RETRY, 0.9, document="book.md"),
        msgspec.structs.replace(hit(RETRY, 0.8, document="notes.md"), header="Delivery"),
    ]
    vectors = _unit([1.0, 0.0], [0.90, float(np.sqrt(1 - 0.90**2))])

    (kept,) = collapse.hits(found, _embedded(found, vectors, bge_small), 2)

    (reference,) = kept.also_in
    assert (reference.document, reference.to_parent.words.contained) == ("notes.md", 1.0)
    assert reference.to_parent.embedding is not None
    assert reference.to_parent.embedding.alike == pytest.approx(0.90), "under the model's bar"


# --- ranges ---------------------------------------------------------------------------


def _passage(texts: list[str], document: str, first_seq: int, score: float) -> list[Hit]:
    """Consecutive chunks of one document: what `passage.ranges` merges into one range."""
    return [
        hit(text, score, document=document, seq=first_seq + i, char_start=100 * i)
        for i, text in enumerate(texts)
    ]


def _ranges(
    scanned: list[Hit], vectors: np.ndarray, limit: int, model: EmbeddingModel
) -> list[HitRange]:
    return collapse.ranges(
        ranges(scanned, how=HARMONIC), scanned, _embedded(scanned, vectors, model), limit
    )


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
            [("book.md", 1, 3, [("note.md", "contained")])],
        ),
        (
            "chunk by chunk: the three-chunk passage takes the slot of the copy that outranked it",
            0.5,
            0.9,
            [REWORDED],
            [E2],
            [("book.md", 1, 3, [("note.md", "contained")])],
        ),
        (
            "passage by passage: two chunks that match none singly but sum to the same point fold",
            0.9,
            0.5,
            [REWORDED, OUTBOX],
            [[1.0, 1.0, 0.9, 0.0], [1.0, 1.0, 1.1, 0.0]],
            [("book.md", 1, 3, [("note.md", "equivalent")])],
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
@pytest.mark.anyio
async def test_ranges_fold_by_containment_or_by_their_mean_vectors(
    name: str,
    big_score: float,
    small_score: float,
    small: list[str],
    small_vectors: list[list[float]],
    expected: list[tuple[str, int, int, list[tuple[str, str]]]],
    bge_small: EmbeddingModel,
) -> None:
    """The case from the design session: a mean vector of three chunks sits far from any one of
    them, so the passage-level cosine alone would keep a copy of its middle chunk."""
    big = _passage([BACKOFF, RETRY, CLOCKS], "book.md", 1, big_score)
    scanned = big + _passage(small, "note.md", 10, small_score)
    vectors = _unit(E1, E2, E3, *small_vectors)

    kept = _ranges(scanned, vectors, 2, bge_small)

    shape = [
        (
            r.hits[0].document,
            r.seq_start,
            r.seq_end,
            [(ref.document, ref.relation) for ref in r.also_in],
        )
        for r in kept
    ]
    assert shape == expected, name


@pytest.mark.anyio
async def test_a_folded_range_points_at_its_own_lines(bge_small: EmbeddingModel) -> None:
    big = _passage([BACKOFF, RETRY, CLOCKS], "book.md", 1, 0.9)
    small = _passage([RETRY], "note.md", 10, 0.5)
    scanned = big + small

    (kept,) = _ranges(scanned, _unit(E1, E2, E3, E2), 2, bge_small)

    (reference,) = kept.also_in
    (small_range,) = ranges(small, how=HARMONIC)
    assert (reference.document, reference.seq_start, reference.seq_end) == ("note.md", 10, 10)
    assert (reference.line_start, reference.line_end) == (
        small_range.line_start,
        small_range.line_end,
    ), "the lines the UI reads it back by"
    assert reference.location == location(
        "note.md", None, None, small_range.line_start, small_range.line_end
    )
    assert reference.to_parent.embedding is not None
    assert reference.to_parent.embedding.contained == pytest.approx(1.0), (
        "its one chunk is the passage's middle"
    )
    assert reference.to_parent.embedding.contains == pytest.approx(1 / 3), "one of three chunks"
    assert len(kept.also_in) == 1


@pytest.mark.parametrize(
    ("name", "found", "expected", "excerpts"),
    [
        (
            "a passage never folds under a range too short to stand alone",
            [(RETRY, "a.md", 0.9, True), (RETRY, "copy.md", 0.8, False)],
            [("a.md", True, []), ("copy.md", False, [])],
            ["copy.md"],
        ),
        (
            "a range too short to stand alone folds under a passage it repeats",
            [(RETRY, "a.md", 0.9, False), (RETRY, "copy.md", 0.8, True)],
            [("a.md", False, ["copy.md"])],
            ["a.md"],
        ),
        (
            "a range too short to stand alone never takes the slot of a passage it holds",
            [(RETRY, "a.md", 0.9, False), (FULLER, "book.md", 0.8, True)],
            [("a.md", False, []), ("book.md", True, [])],
            ["a.md"],
        ),
        (
            "a passage takes the slot of a range too short to stand alone it holds",
            [(RETRY, "a.md", 0.9, True), (FULLER, "book.md", 0.8, False)],
            [("book.md", False, ["a.md"])],
            ["book.md"],
        ),
        (
            "nothing folds under a range too short to stand alone, not even another",
            [(RETRY, "a.md", 0.9, True), (RETRY, "copy.md", 0.8, True)],
            [("a.md", True, []), ("copy.md", True, [])],
            [],
        ),
    ],
)
def test_a_range_too_short_to_stand_alone_leads_no_fold(
    name: str,
    found: list[tuple[str, str, float, bool]],
    expected: list[tuple[str, bool, list[str]]],
    excerpts: list[str],
) -> None:
    """A section whose ranges are all too short to stand alone is dropped (`section.group`), with
    every place folded under them. So a passage must never sit under such a range: it would be
    lost with it, though it stands on its own."""
    scanned = [hit(text, score, document=document) for text, document, score, _ in found]
    marked = [
        msgspec.structs.replace(one, alone=alone)
        for one, (*_, alone) in zip(ranges(scanned, how=HARMONIC), found, strict=True)
    ]

    kept = collapse.ranges(marked, scanned, words_scan(scanned), None)

    shape = [(r.hits[0].document, r.alone, [ref.document for ref in r.also_in]) for r in kept]
    assert shape == expected, name
    # each document one section of one chunk, as `CollectionIndex.outline_rows` reads it
    rows = [
        {
            "document_id": one.document_id,
            "seq": one.seq,
            "headings": one.headings,
            "char_start": one.char_start,
            "char_end": one.char_end,
        }
        for one in scanned
    ]
    outlines = section.outlines(
        [(one.collection, row) for one, row in zip(scanned, rows, strict=True)]
    )
    grouped = section.group(kept, outlines, max_chars=10_000, limit=len(kept) + 1)
    assert [one.document_id for one in grouped] == excerpts, f"{name}: the excerpts answered with"


# --- the space ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "vectors", "model", "mode", "deciding", "measured"),
    [
        (
            "every row carries a vector under a model with thresholds",
            [E1, E2],
            lambda bge: bge,
            SearchMode.HYBRID,
            ["embedding", "words"],
            ["embedding", "words"],
        ),
        (
            "a vector search decides by vectors alone, and still measures the words",
            [E1, E2],
            lambda bge: bge,
            SearchMode.VECTOR,
            ["embedding"],
            ["embedding", "words"],
        ),
        (
            "a full-text search decides by both, vectors first",
            [E1, E2],
            lambda bge: bge,
            SearchMode.FTS,
            ["embedding", "words"],
            ["embedding", "words"],
        ),
        (
            "one row without a vector sends the whole scan to words, whatever the mode",
            [E1, None],
            lambda bge: bge,
            SearchMode.VECTOR,
            ["words"],
            ["words"],
        ),
        (
            "no embedding model at all",
            [None, None],
            lambda bge: None,
            SearchMode.HYBRID,
            ["words"],
            ["words"],
        ),
        (
            "a model without thresholds",
            [E1, E2],
            lambda bge: EmbeddingModel("test/tiny", 2),
            SearchMode.VECTOR,
            ["words"],
            ["words"],
        ),
        ("nothing scanned", [], lambda bge: bge, SearchMode.HYBRID, ["words"], ["words"]),
    ],
)
@pytest.mark.anyio
async def test_the_spaces_follow_the_rows_the_model_and_the_mode(
    name: str,
    vectors: list,
    model: Callable[[EmbeddingModel], EmbeddingModel | None],
    mode: SearchMode,
    deciding: list[str],
    measured: list[str],
    bge_small: EmbeddingModel,
) -> None:
    texts = [RETRY, BACKOFF][: len(vectors)]

    scan = collapse.spaces(texts, vectors, model(bge_small), mode)

    assert [space.kind for space in scan.deciding] == deciding, f"{name}: decided by"
    assert [space.kind for space in scan.measured] == measured, f"{name}: measured in"


def test_folding_leaves_the_hits_it_was_given_alone() -> None:
    found = [hit(RETRY, 0.9, document="a.md"), hit(RETRY, 0.8, document="copy.md")]
    before = msgspec.json.encode(found)

    collapse.hits(found, words_scan(found), 2)

    assert msgspec.json.encode(found) == before, "a pure fold: the scan is not rewritten"
