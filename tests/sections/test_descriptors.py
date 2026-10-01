"""The word statistics descriptors are picked by: which words are terms, how a term weighs in
one section against the others, and which candidates a section keeps."""

import math
from collections import Counter

import numpy as np
import pytest

from haskie.sections import descriptors

# A paragraph of a real book, as the PDF converter writes it: emphasis marks, a word hyphenated
# over a line end, a table row, a number and a URL.
PARAGRAPH = (
    "An **aggregate root** guards the invariants of the aggregate. Each aggregate root is\n"
    "loaded by its repository, and a con-\nsistency boundary is drawn around it.\n\n"
    "| order | 2024 |\n\nSee https://example.com/aggregate for the root of the aggregate."
)


def _keys(text: str) -> list[str]:
    return [key for key, _ in descriptors.terms(text)]


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
    forms: Counter[descriptors.Term] = Counter()
    counts = descriptors.counted(descriptors.terms(PARAGRAPH), forms)

    assert counts["aggreg"] == 5, "every form of the word, the URL's included"
    assert counts["aggreg root"] == 2
    assert "root aggreg" not in counts, "'root of the aggregate' is no pair"
    assert descriptors.shown(["aggreg", "aggreg root", "consist"], forms) == {
        "aggreg": "aggregate",
        "aggreg root": "aggregate root",
        "consist": "consistency",
    }


def test_shown_breaks_a_tie_by_the_form_seen_first() -> None:
    forms: Counter[descriptors.Term] = Counter()
    descriptors.counted(descriptors.terms("Saga saga"), forms)

    assert descriptors.shown(["saga"], forms) == {"saga": "Saga"}


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
    assert descriptors.frequent(counts, enough) == expected, name


def test_ctfidf_ranks_a_term_one_class_uses_above_one_every_class_uses() -> None:
    """Three chapters: every one says "data"; only the first says "saga"."""
    chapters = [
        Counter({"data": 4, "saga": 4}),
        Counter({"data": 4, "replica": 4}),
        Counter({"data": 4, "partit": 4}),
    ]

    first, second, _ = descriptors.ctfidf(chapters)

    assert first["saga"] > first["data"], "the chapter's own word outranks the shared one"
    assert second["replica"] > second["data"]
    assert first["data"] == pytest.approx(second["data"]), "the shared word weighs the same"


def test_ctfidf_sinks_a_term_most_sections_use_below_one_few_use_however_often() -> None:
    """Ten sections of 100 terms. "design" is used once in eight of them, 8 uses in all; "saga"
    once in the first and 39 times in the second, 40 in all. How many sections use a term
    decides, not its uses: counting uses against the class size, as BERTopic does, would give
    "design" an inverse frequency of 2.5 and "saga" 0.9, and rank "design" first here."""
    sections = [Counter({"design": 1, "saga": 1, "filler": 98})]
    sections += [Counter({"saga": 39, "filler": 61})]
    sections += [Counter({"design": 1, f"topic{n}": 99}) for n in range(6)]
    sections += [Counter({"design": 1, "other": 99}), Counter({"other": 100})]
    first = descriptors.ctfidf(sections)[0]
    assert first["saga"] > first["design"], "a term two sections use outranks one eight use"
    assert first["design"] == pytest.approx(math.sqrt(1 / 100) * math.log(1 + 2.5 / 8.5))


@pytest.mark.parametrize(
    ("name", "classes", "expected"),
    [
        ("no class: no weights", [], []),
        ("an empty class weighs nothing", [Counter(), Counter({"saga": 2})], [{}, {"saga": None}]),
        ("one class alone is weighed against itself", [Counter({"saga": 1})], [{"saga": None}]),
    ],
)
def test_ctfidf_edges(name: str, classes: list[Counter[str]], expected: list[dict]) -> None:
    found = descriptors.ctfidf(classes)

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
    assert descriptors.best(weights, k) == expected, name


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
    assert descriptors.rerank(candidates, BOTH, k) == expected, name


@pytest.mark.parametrize(
    ("name", "classes", "expected"),
    [
        ("no class: nothing widespread", [], set()),
        ("one class alone: nothing to be set apart from", [Counter({"saga": 3})], set()),
        (
            "two classes: a term both use is widespread",
            [Counter({"saga": 1, "step": 2}), Counter({"saga": 5, "log": 1})],
            {"saga"},
        ),
        (
            "half is not more than half",
            [Counter({"saga": 1}), Counter({"saga": 1}), Counter({"log": 1}), Counter({"step": 1})],
            set(),
        ),
        (
            "three of four is",
            [Counter({"saga": 1}), Counter({"saga": 1}), Counter({"saga": 1}), Counter({"log": 1})],
            {"saga"},
        ),
    ],
)
def test_widespread(name: str, classes: list[Counter[str]], expected: set[str]) -> None:
    assert descriptors.widespread(classes) == expected, name


def test_a_term_most_sections_use_is_no_descriptor_even_where_a_short_section_repeats_it() -> None:
    """Four chapters all say "domain"; the short second one says it three times and little else
    twice. Its descriptors are its own words; the whole document, one class, still lists
    "domain"."""
    texts = [
        "domain, saga, compensation. saga, compensation.",
        "domain, ledger. domain, ledger. domain.",
        "domain, replica, quorum. replica, quorum.",
        "domain, partition, hashing. partition, hashing.",
    ]
    runs = [descriptors.Run((), 0, 3)] + [descriptors.Run((f"Ch{n}",), n, n) for n in range(4)]
    whole, _, second, *_ = descriptors.ClassTfidf().pick(texts, runs, None, None)
    assert "domain" in whole, "the document's own topic names the document"
    assert list(second) == ["ledger"], "the short chapter keeps only its own word"


@pytest.mark.parametrize(
    ("name", "headings", "expected"),
    [
        ("the whole document: no heading", (), set()),
        ("each word by its stem", ("Sagas",), {"saga"}),
        (
            "the whole path, stopwords and short words aside, pairs not words",
            ("Part I", "Rule: Design Small Aggregates"),
            {"part", "rule", "design", "small", "aggreg"},
        ),
    ],
)
def test_said(name: str, headings: tuple[str, ...], expected: set[str]) -> None:
    assert descriptors.said(headings) == expected, name


def test_a_term_the_header_already_says_is_no_descriptor() -> None:
    """Two sections under "Aggregates": the header's words, and a pair of them, are no descriptor of
    either; the word a pair adds to them still is, alone or in the pair."""
    texts = [
        "aggregates, separate aggregates. aggregates, separate aggregates. invariants, invariants.",
        "aggregates, rules. aggregates, rules. transactions, transactions.",
    ]
    runs = [descriptors.Run(("Aggregates", "Small Aggregates"), 0, 0)]
    runs += [descriptors.Run(("Aggregates", "Rules"), 1, 1)]
    small, rules = descriptors.ClassTfidf().pick(texts, runs, None, None)
    assert "aggregates" not in small and "aggregates" not in rules, "the path's words"
    assert "rules" not in rules, "the section's own heading"
    assert any("separate" in descriptor for descriptor in small), "a pair adds a word of its own"
    assert "invariants" in small and "transactions" in rules


def test_a_section_with_words_gets_one_to_five_descriptors() -> None:
    """The first section says only what its header says: it still gets its best word. The second
    repeats seven words of its own: it gets five."""
    texts = [
        "sagas, saga.",
        "ledger, quorum, replica, leader, follower, partition, hashing. "
        "ledger, quorum, replica, leader, follower, partition, hashing.",
    ]
    runs = [descriptors.Run(("Sagas",), 0, 0), descriptors.Run(("Storage",), 1, 1)]
    sagas, storage = descriptors.ClassTfidf().pick(texts, runs, None, None)
    assert list(sagas) == ["sagas"], "one descriptor, though its header says it"
    assert len(storage) == descriptors.DESCRIPTORS == 5
