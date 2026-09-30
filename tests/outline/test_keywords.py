"""The word statistics an outline and a map of sections share: which words are terms, how a term
weighs in one section against the others, and which candidates a section keeps."""

from collections import Counter

import numpy as np
import pytest

from haskie.outline import keywords

# A paragraph of a real book, as the PDF converter writes it: emphasis marks, a word hyphenated
# over a line end, a table row, a number and a URL.
PARAGRAPH = (
    "An **aggregate root** guards the invariants of the aggregate. Each aggregate root is\n"
    "loaded by its repository, and a con-\nsistency boundary is drawn around it.\n\n"
    "| order | 2024 |\n\nSee https://example.com/aggregate for the root of the aggregate."
)


def _keys(text: str) -> list[str]:
    return [key for key, _ in keywords.terms(text)]


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        (
            "a pair of neighbours is a term, and each of its words",
            "aggregate root",
            ["aggreg", "root", "aggreg root"],
        ),
        ("a stopword between two words breaks the pair", "root of aggregate", ["root", "aggreg"]),
        ("punctuation breaks the pair", "root. aggregate", ["root", "aggreg"]),
        ("a blank line breaks the pair", "root\n\naggregate", ["root", "aggreg"]),
        ("a single line break does not", "root\naggregate", ["root", "aggreg", "root aggreg"]),
        ("emphasis marks do not", "**aggregate** root", ["aggreg", "root", "aggreg root"]),
        ("a word hyphenated over a line end is one word", "con-\nsistency", ["consist"]),
        ("numbers and short words are no terms", "2024 of an id x9", []),
        ("a word with a digit and a letter is", "k8s", ["k8s"]),
        ("a contraction is one word, and a stopword", "don't know", ["know"]),
        ("a curly apostrophe too", "isn\u2019t known", ["known"]),
        (
            "a possessive keeps its word's stem",
            "the aggregate's root",
            ["aggreg", "root", "aggreg root"],
        ),
        (
            "forms of one word share a stem",
            "aggregates aggregate",
            ["aggreg", "aggreg", "aggreg aggreg"],
        ),
    ],
)
def test_terms(name: str, text: str, expected: list[str]) -> None:
    assert _keys(text) == expected, name


def test_counted_and_shown_keep_the_form_written_most() -> None:
    forms: Counter[keywords.Term] = Counter()
    counts = keywords.counted(keywords.terms(PARAGRAPH), forms)

    assert counts["aggreg"] == 5, "every form of the word, the URL's included"
    assert counts["aggreg root"] == 2
    assert "root aggreg" not in counts, "'root of the aggregate' is no pair"
    assert keywords.shown(["aggreg", "aggreg root", "consist"], forms) == {
        "aggreg": "aggregate",
        "aggreg root": "aggregate root",
        "consist": "consistency",
    }


def test_shown_breaks_a_tie_by_the_form_seen_first() -> None:
    forms: Counter[keywords.Term] = Counter()
    keywords.counted(keywords.terms("Saga saga"), forms)

    assert keywords.shown(["saga"], forms) == {"saga": "Saga"}


def test_key_of_a_written_form_is_the_key_terms_give_it() -> None:
    assert keywords.key_of("Aggregate Roots") == "aggreg root"


@pytest.mark.parametrize(
    ("name", "counts", "enough", "expected"),
    [
        (
            "terms used once are no candidates when enough words are used twice",
            Counter({"saga": 3, "compens": 2, "orchestr": 2, "saga compens": 1}),
            2,
            {"saga", "compens", "orchestr"},
        ),
        (
            "a short text keeps every term: too few words are used twice",
            Counter({"saga": 2, "compens": 1, "saga compens": 1}),
            2,
            {"saga", "compens", "saga compens"},
        ),
        (
            "a pair used twice is no word: it does not make the text long",
            Counter({"saga": 2, "saga compens": 2, "step": 1}),
            2,
            {"saga", "saga compens", "step"},
        ),
    ],
)
def test_frequent(name: str, counts: Counter[str], enough: int, expected: set[str]) -> None:
    assert keywords.frequent(counts, enough) == expected, name


def test_ctfidf_ranks_a_term_one_class_uses_above_one_every_class_uses() -> None:
    """Three chapters: every one says "data"; only the first says "saga"."""
    chapters = [
        Counter({"data": 4, "saga": 4}),
        Counter({"data": 4, "replica": 4}),
        Counter({"data": 4, "partit": 4}),
    ]

    first, second, _ = keywords.ctfidf(chapters)

    assert first["saga"] > first["data"], "the chapter's own word outranks the shared one"
    assert second["replica"] > second["data"]
    assert first["data"] == pytest.approx(second["data"]), "the shared word weighs the same"


@pytest.mark.parametrize(
    ("name", "classes", "expected"),
    [
        ("no class: no weights", [], []),
        ("an empty class weighs nothing", [Counter(), Counter({"saga": 2})], [{}, {"saga": None}]),
        ("one class alone is weighed against itself", [Counter({"saga": 1})], [{"saga": None}]),
    ],
)
def test_ctfidf_edges(name: str, classes: list[Counter[str]], expected: list[dict]) -> None:
    found = keywords.ctfidf(classes)

    assert [set(one) for one in found] == [set(one) for one in expected], name
    assert all(np.isfinite(weight) for one in found for weight in one.values()), name


@pytest.mark.parametrize(
    ("name", "weights", "k", "expected"),
    [
        (
            "a heavier pair keeps its words out",
            {"aggreg root": 3.0, "aggreg": 2.0, "root": 1.5, "invari": 1.0},
            3,
            ["aggreg root", "invari"],
        ),
        (
            "a heavier word keeps a pair holding it out",
            {"aggreg": 3.0, "aggreg root": 2.0, "invari": 1.0},
            2,
            ["aggreg", "invari"],
        ),
        ("cut at k, heaviest first", {"a1": 1.0, "b2": 3.0, "c3": 2.0}, 2, ["b2", "c3"]),
        ("a tie goes by the key", {"beta": 1.0, "alpha": 1.0}, 1, ["alpha"]),
    ],
)
def test_best(name: str, weights: dict[str, float], k: int, expected: list[str]) -> None:
    assert keywords.best(weights, k) == expected, name


def _unit(*rows: list[float]) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float64)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


BOTH = _unit([1.0, 1.0])[0]  # a section about two topics at once


@pytest.mark.parametrize(
    ("name", "candidates", "k", "expected"),
    [
        ("no candidate: none kept", np.empty((0, 2)), 3, []),
        (
            "the closest first, then the other topic over a near copy of the first",
            _unit([1.0, 0.2], [1.0, 0.25], [0.2, 1.0]),
            2,
            [1, 2],
        ),
        ("k past the candidates keeps them all", _unit([1.0, 0.0], [0.0, 1.0]), 5, [0, 1]),
    ],
)
def test_rerank(name: str, candidates: np.ndarray, k: int, expected: list[int]) -> None:
    assert keywords.rerank(candidates, BOTH, k) == expected, name
