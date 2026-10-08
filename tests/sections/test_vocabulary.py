"""A collection's controlled vocabulary (`sections.vocabulary`): which variants are one concept,
and which preferred term each stands for."""

import math

import numpy as np
import pytest

from haskie.sections import vocabulary
from haskie.sections.vocabulary import JUDGE_COSINE, Term

SAME_CONCEPT = 0.15  # a describer's bar for one concept (`gguf_models.Generator.same_concept`)


def at_angles(*degrees: float) -> np.ndarray:
    """Unit vectors in a plane, one per angle: two of them meet at the cosine of the angle
    between them, so a test sets every cosine it needs by hand."""
    radians = np.radians(degrees)
    return np.stack([np.cos(radians), np.sin(radians)], axis=1).astype(np.float32)


def apart(cosine: float) -> float:
    """The angle in degrees two of `at_angles` vectors are at for this cosine."""
    return math.degrees(math.acos(cosine))


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("Event Sourcing", "event sourcing"),
        ("  Event   sourcing\t", "event sourcing"),
        ("", ""),
    ],
)
def test_a_variant_is_the_lowercase_form_with_its_blanks_collapsed(
    phrase: str, expected: str
) -> None:
    assert vocabulary.variant(phrase) == expected


@pytest.mark.parametrize(
    ("name", "a", "b", "expected"),
    [
        ("a hyphen", "time-out handling", "timeout handling", True),
        ("a blank", "life cycle", "lifecycle", True),
        ("a plural, by stem", "system call", "system calls", True),
        ("an inflection of each word", "bounded context", "bounded contexts", True),
        ("a prefix that turns the meaning", "synchronous calls", "asynchronous calls", False),
        ("another word", "block ordering", "lock ordering", False),
        ("another number of words", "data validation", "input data validation", False),
        ("the same words in another order", "application logic", "logic application", False),
        ("different language symbols", "c++ programming", "c# programming", False),
        ("a leading dot", ".net", "net", False),
        ("an internal dot", "node.js", "nodejs", False),
        ("an underscore", "user_id", "userid", False),
        ("different Unicode words", "数据库", "机器学习", False),
        ("an accented letter", "résumé", "rsum", False),
        ("identical Unicode words", "数据库", "数据库", True),
        ("symbols with an inflection", "c++ system call", "c++ system calls", True),
        ("empty phrases", "", "", False),
        ("one empty phrase", "events", "", False),
        ("only separators", " - ", "--", False),
    ],
)
def test_same_words(name: str, a: str, b: str, expected: bool) -> None:
    assert vocabulary.same_words(a, b) is expected, name


@pytest.mark.parametrize(
    "variants", [["c++ programming", "c# programming"], ["数据库", "机器学习"]]
)
def test_distinct_topics_need_a_verdict_even_with_similar_vectors(variants: list[str]) -> None:
    vectors = at_angles(0, 1)
    assert vocabulary.to_judge(variants, vectors) == [vocabulary.pair(*variants)]
    assert vocabulary.cluster(variants, [2, 1], vectors, {}, SAME_CONCEPT) == [0, 1]
    assert vocabulary.cluster(variants, [2, 1], at_angles(0, 90), {}, SAME_CONCEPT) == [0, 1]


@pytest.mark.parametrize("count", [0, 1])
def test_too_few_vectors_have_no_neighbours(count: int) -> None:
    ids, cosines = vocabulary.neighbours(at_angles(*range(count)))
    assert ids.shape == cosines.shape == (count, 0)


def test_a_vectors_neighbours_are_the_others_by_cosine() -> None:
    ids, cosines = vocabulary.neighbours(at_angles(0, 10, 90))
    nearest = {at: sorted(zip(cosines[at], ids[at], strict=True), reverse=True) for at in range(3)}
    assert [int(other) for _, other in nearest[0]] == [1, 2], "itself never"
    assert nearest[0][0][0] == pytest.approx(math.cos(math.radians(10)), abs=1e-6)


def test_the_pairs_to_judge_are_close_and_say_other_words() -> None:
    """A pair over the cosine bar is judged once, in sorted order; one under it, or one the
    stems already decide, is not."""
    variants = ["performance improvement", "performance enhancement", "system call", "system calls"]
    close = apart(JUDGE_COSINE) / 2
    vectors = at_angles(0, close, 90, 90 + close)  # the second pair the same words
    far = at_angles(0, apart(JUDGE_COSINE) + 1, 90, 90 + close)
    assert vocabulary.to_judge(variants, vectors) == [
        ("performance enhancement", "performance improvement")
    ]
    assert vocabulary.to_judge(variants, far) == [], "under the bar"
    assert vocabulary.to_judge([], at_angles()) == []


SYNONYM = (SAME_CONCEPT + 1) / 2  # a verdict over the bar
CLOSE = apart(JUDGE_COSINE) * 0.6  # an angle over the cosine bar
FAR = apart(JUDGE_COSINE) * 1.2  # an angle under it


@pytest.mark.parametrize(
    ("name", "variants", "uses", "degrees", "verdicts", "expected"),
    [
        (
            "the same words join with no verdict, at any cosine",
            ["system call", "system calls"], [3, 1], [0, 80], {}, [0, 0],
        ),
        (
            "a judged synonym joins the more used",
            ["performance enhancement", "performance improvement"], [1, 3], [0, CLOSE],
            {("performance enhancement", "performance improvement"): SYNONYM}, [1, 1],
        ),
        (
            "a verdict at the bar is one concept",
            ["performance enhancement", "performance improvement"], [1, 3], [0, CLOSE],
            {("performance enhancement", "performance improvement"): SAME_CONCEPT}, [1, 1],
        ),
        (
            "a verdict under the bar keeps both",
            ["data validation", "input validation"], [3, 1], [0, CLOSE],
            {("data validation", "input validation"): SAME_CONCEPT / 2}, [0, 1],
        ),
        (
            "a pair never asked keeps both",
            ["data validation", "input validation"], [3, 1], [0, CLOSE], {}, [0, 1],
        ),
        (
            "a pair under the cosine bar keeps both, whatever its verdict",
            ["data validation", "input validation"], [3, 1], [0, FAR],
            {("data validation", "input validation"): 1.0}, [0, 1],
        ),
        (
            # b joins a; c is close to b alone, and b is no term, so c is a term of its own
            "a variant joins a preferred term, never a chain",
            ["a", "b", "c"], [3, 2, 1], [0, CLOSE, 2 * CLOSE],
            {("a", "b"): SYNONYM, ("b", "c"): SYNONYM}, [0, 0, 2],
        ),
        (
            "of two terms, the nearer wins",
            ["a", "b", "c"], [3, 2, 1], [0, 2 * CLOSE, 1.1 * CLOSE],
            {("a", "c"): SYNONYM, ("b", "c"): SYNONYM}, [0, 1, 1],
        ),
        (
            "a tie in use goes to the shorter variant",
            ["time-out handling", "timeout handling"], [2, 2], [0, CLOSE], {}, [1, 1],
        ),
        ("no variants", [], [], [], {}, []),
    ],
)  # fmt: skip
def test_cluster(
    name: str,
    variants: list[str],
    uses: list[int],
    degrees: list[float],
    verdicts: dict[tuple[str, str], float],
    expected: list[int],
) -> None:
    found = vocabulary.cluster(variants, uses, at_angles(*degrees), verdicts, SAME_CONCEPT)
    assert found == expected, name


def test_terms_sum_their_variants_uses_most_used_first() -> None:
    variants = ["event sourcing", "event-sourcing", "domain events"]
    found = vocabulary.terms(
        variants, [3, 1, 2], ["Event sourcing", "event-sourcing", "Domain events"], [0, 0, 2]
    )
    assert found == [
        Term("event sourcing", "Event sourcing", 4, ["event sourcing", "event-sourcing"]),
        Term("domain events", "Domain events", 2, ["domain events"]),
    ]


@pytest.mark.parametrize(
    ("name", "phrases", "expected"),
    [
        ("each by its term", ["Event-sourcing", "Domain events"],
         ["Event sourcing", "Domain events"]),
        ("two variants of one term, once", ["event sourcing", "Event-Sourcing"],
         ["Event sourcing"]),
        ("a phrase the vocabulary lacks stays", ["Sagas"], ["Sagas"]),
        ("none", [], []),
    ],
)  # fmt: skip
def test_compact(name: str, phrases: list[str], expected: list[str]) -> None:
    preferred = {
        "event sourcing": "Event sourcing",
        "event-sourcing": "Event sourcing",
        "domain events": "Domain events",
    }
    assert vocabulary.compact(phrases, preferred) == expected, name
