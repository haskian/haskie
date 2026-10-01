"""The words of a question the answer never mentions: which they are, which passage the probe
found helps which question, where its section goes, and what the answer reports it lacks."""

import msgspec
import pytest
from conftest import hit

from haskie.search import probe
from haskie.search.passage import Excerpt, Span, ranges
from haskie.search.probe import Question
from haskie.search.section import Group, Section
from haskie.settings import ScoreFold

HARMONIC = ScoreFold.HARMONIC  # the rule these cases were written against

ORDER = Question(None, "How does an order keep inventory consistent?", label="order")
LEDGER = Question(None, "How is the ledger reconciled?", label="ledger")
ALONE = Question(None, "How is the ledger reconciled?")


def _group(document: str, text: str, path: tuple[str, ...] = ("Shop",)) -> Group:
    (found,) = ranges([hit(text, 1.0, document=document)], how=HARMONIC)
    return Group("backend", document, Section(path, 1, 1), [found])


@pytest.mark.parametrize(
    ("name", "word", "text", "expected"),
    [
        ("the word itself", "keep", "We keep it.", True),
        ("a longer form", "keep", "It keeps and keeping goes on.", True),
        ("a form with another ending", "consistent", "Consistency matters.", True),
        ("a noun of the verb, many letters longer", "deploy", "The deployment ran.", True),
        ("the verb of a noun", "configuration", "Configure it first.", True),
        ("a word that only starts the same is another word", "cat", "A category.", False),
        ("nor is a longer one", "test", "The testament.", False),
        ("an irregular form does not count", "run", "It ran.", False),
        ("a word not there", "inventory", "Orders ship.", False),
    ],
)
def test_a_text_holds_a_word_or_a_form_of_it(
    name: str, word: str, text: str, expected: bool
) -> None:

    assert (probe.stem(word) in probe.vocabulary([text])) is expected, name


@pytest.mark.parametrize(
    ("name", "texts", "stemmed", "expected"),
    [
        ("no text, nothing stemmed", [], set(), set()),
        (
            "a word repeated in one text is stemmed once",
            ["Orders ship. Orders keep. Orders wait."],
            {"orders", "ship", "keep", "wait"},
            {"order", "ship", "keep", "wait"},
        ),
        (
            "one word in several texts and cases is stemmed once",
            ["Inventory", "the inventory keeps", "INVENTORY"],
            {"inventory", "the", "keeps"},
            {"inventori", "the", "keep"},
        ),
    ],
)
def test_the_vocabulary_stems_each_distinct_word_once(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    texts: list[str],
    stemmed: set[str],
    expected: set[str],
) -> None:
    """The stemmer is pure Python and an answer runs to about 36,000 characters, most of its
    words many times over."""
    calls: list[str] = []
    real = probe.stem

    def counted(word: str) -> str:
        calls.append(word)
        return real(word)

    monkeypatch.setattr(probe, "stem", counted)

    assert probe.vocabulary(texts) == expected, name
    assert sorted(calls) == sorted(stemmed), f"{name}: each distinct word once"


def test_a_word_is_stemmed_once_across_reads() -> None:
    """Each search reads its words twice (`retrieval.probe_gaps`, then `probe.report`), so `stem`
    remembers what it stemmed, in a bounded cache: the second read stems nothing again."""
    probe.stem.cache_clear()
    texts = ["Orders keep inventory consistent.", "orders KEEP inventory"]

    first = probe.vocabulary(texts)
    cold = probe.stem.cache_info()
    second = probe.vocabulary(texts)
    warm = probe.stem.cache_info()

    assert first == second == {"order", "keep", "inventori", "consist"}
    assert (cold.misses, cold.hits) == (4, 0), "each distinct word stemmed once"
    assert (warm.misses, warm.hits) == (4, 4), "the second read stems none of them again"
    assert warm.maxsize == probe.STEM_CACHE, "bounded"


@pytest.mark.parametrize(
    ("name", "questions", "covered", "expected"),
    [
        (
            "a word no text holds is missing, with the questions that used it",
            [ORDER],
            ["An order keeps its lines consistent."],
            {"inventory": ["order"]},
        ),
        (
            "a form of the word holds it: keeps for keep, consistency for consistent",
            [ORDER],
            ["Orders keep their inventory in consistency."],
            {},
        ),
        ("a heading holds a word too", [ORDER], ["Inventory", "orders keep consistent"], {}),
        (
            "words of several questions, in the order asked",
            [ORDER, LEDGER],
            ["An order keeps its lines consistent."],
            {"inventory": ["order"], "ledger": ["ledger"], "reconciled": ["ledger"]},
        ),
        (
            "nothing kept: every word is missing",
            [LEDGER],
            [],
            {"ledger": ["ledger"], "reconciled": ["ledger"]},
        ),
        (
            "a question of stopwords asks for nothing",
            [Question(None, "How is it?")],
            [],
            {},
        ),
    ],
)
def test_the_missing_words_are_those_no_kept_text_holds(
    name: str, questions: list[Question], covered: list[str], expected: dict[str, list[str]]
) -> None:
    found = probe.missing(questions, covered)

    assert {word: [q.label for q in qs] for word, qs in found.items()} == expected, name
    assert list(found) == list(expected), f"{name}: in the order asked"


def test_the_kept_text_is_every_passage_and_heading() -> None:
    groups = [_group("a.md", "Orders ship.", ("Shop", "Orders"))]

    assert probe.covered(groups) == ["Orders ship.", "Messaging", "Retries"]


@pytest.mark.parametrize(
    ("name", "text", "questions", "expected"),
    [
        (
            "tagged with the question whose word it holds",
            "Inventory counts drop.",
            [ORDER, LEDGER],
            ["order"],
        ),
        (
            "with every question it helps, in the order asked",
            "The ledger shows inventory.",
            [ORDER, LEDGER],
            ["order", "ledger"],
        ),
        ("a single question tags nothing", "The ledger balances.", [ALONE], []),
    ],
)
def test_a_probed_passage_is_tagged_with_the_questions_it_helps(
    name: str, text: str, questions: list[Question], expected: list[str]
) -> None:
    (found,) = ranges([hit(text, 1.0)], how=HARMONIC)
    wanted = probe.missing(questions, [])

    assert probe.tags(found, wanted) == expected, name


def test_the_probed_section_joins_the_section_it_is_part_of_else_comes_after() -> None:
    kept = [_group("a.md", "Orders ship."), _group("b.md", "Ledgers balance.")]
    same = _group("b.md", "The ledger is reconciled nightly.")
    other = _group("c.md", "Inventory counts drop.")

    joined = probe.placed(kept, same)
    added = probe.placed(kept, other)

    assert [len(one.ranges) for one in joined] == [1, 2], "into its section, not a slot"
    assert [one.document_id for one in added] == ["a.md", "b.md", "c.md"], "past the others"


def _excerpt(text: str, aspects: list[str], header: str = "Shop") -> Excerpt:
    span = Span(
        header=f"{header} > Stock",
        section_id="stock",
        location="a.md L1-1",
        seq_start=1,
        seq_end=1,
        line_start=1,
        line_end=1,
        char_start=0,
        char_end=len(text),
        page_start=None,
        page_end=None,
        score=1.0,
        aspects=aspects,
    )
    return Excerpt(
        collection="backend",
        document_id="a.md",
        document="a.md",
        header=header,
        section_id=header.lower(),
        location="a.md L1-1",
        seq_start=1,
        seq_end=1,
        line_start=1,
        line_end=1,
        char_start=0,
        char_end=len(text),
        page_start=None,
        page_end=None,
        text=text,
        score=1.0,
        source_file="/a.md",
        markdown_file="/a.md.md",
        spans=[span],
        aspects=aspects,
    )


@pytest.mark.parametrize(
    ("name", "excerpts", "questions", "uncovered", "missing"),
    [
        (
            "everything answered",
            [_excerpt("Orders keep inventory consistent.", ["order"])],
            [ORDER],
            [],
            [],
        ),
        (
            "a question no excerpt names is uncovered",
            [_excerpt("Orders keep inventory consistent.", ["order"])],
            [ORDER, LEDGER],
            ["ledger"],
            ["ledger", "reconciled"],
        ),
        (
            "a span's heading holds a word",
            [_excerpt("Orders keep it consistent.", ["order"], header="Inventory")],
            [ORDER],
            [],
            [],
        ),
        ("a single question is never uncovered", [], [ALONE], [], ["ledger", "reconciled"]),
    ],
)
def test_the_answer_reports_what_it_lacks(
    name: str,
    excerpts: list[Excerpt],
    questions: list[Question],
    uncovered: list[str],
    missing: list[str],
) -> None:
    answer = probe.report(excerpts, questions)

    assert answer.excerpts == excerpts
    assert (answer.uncovered, answer.missing_terms) == (uncovered, missing), name


def test_an_answer_is_the_wire_shape_the_tool_returns() -> None:
    answer = probe.report([], [ALONE])

    assert msgspec.to_builtins(answer) == {
        "excerpts": [],
        "uncovered": [],
        "missing_terms": ["ledger", "reconciled"],
    }
