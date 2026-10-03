"""The descriptors and summaries a language model writes (`sections.generated`): what it reads,
and how its answer is read back."""

import pytest

from haskie.sections import generated
from haskie.sections.build import Section
from haskie.sections.descriptors import Run

SAGAS = (
    "A saga splits a long transaction into local steps. When a step fails, the saga runs the "
    "compensating step of every step before it."
)
QUORUMS = "A quorum write waits for a majority of replicas, and a quorum read asks a majority too."


@pytest.mark.parametrize(
    ("name", "answer", "expected"),
    [
        (
            "the list alone",
            "Saga orchestration | Compensating steps | Local transactions",
            ["Saga orchestration", "Compensating steps", "Local transactions"],
        ),
        (
            "a sentence before the list",
            "Here are the descriptors:\nQuorum reads | Majority writes",
            ["Quorum reads", "Majority writes"],
        ),
        (
            "a label on the list's own line",
            "Descriptors: Quorum reads | Majority writes",
            ["Quorum reads", "Majority writes"],
        ),
        (
            "marks and blanks around the phrases, an empty one between",
            '  *"Quorum reads"* |  | `Majority writes`. ',
            ["Quorum reads", "Majority writes"],
        ),
        (
            "more than five phrases",
            "a | b | c | d | e | f | g",
            ["a", "b", "c", "d", "e"],
        ),
        ("one phrase without a separator", "Replica staleness", ["Replica staleness"]),
        ("an empty answer", "", []),
        ("blank lines only", "\n  \n", []),
    ],
)
def test_parse_reads_the_first_list_line(name: str, answer: str, expected: list[str]) -> None:
    assert generated.parse(answer) == expected, name


@pytest.mark.parametrize(
    ("name", "phrases", "expected"),
    [
        (
            "a phrase whose every word the path holds goes",
            ["Saga orchestration", "Sagas", "Compensating steps"],
            ["Saga orchestration", "Compensating steps"],
        ),
        (
            "a stem the path holds in another form",
            ["Saga", "Compensation"],
            ["Compensation"],
        ),
        (
            "a phrase with a word of its own stays",
            ["Book sagas", "Saga failures"],
            ["Saga failures"],
        ),
        (
            "a stopword or a short word adds nothing to keep it",
            ["The sagas", "Of a Book"],
            [],
        ),
        ("a phrase of no words stays", ["--", "Saga steps"], ["--", "Saga steps"]),
        ("nothing to drop", ["Quorum reads"], ["Quorum reads"]),
    ],
)
def test_unsaid_drops_what_the_heading_path_says(
    name: str, phrases: list[str], expected: list[str]
) -> None:
    assert generated.unsaid(phrases, ["Book", "Sagas"]) == expected, name


def test_a_short_section_is_read_whole() -> None:
    assert generated.excerpt([SAGAS, QUORUMS]) == f"{SAGAS}\n\n{QUORUMS}"


@pytest.mark.parametrize(
    ("name", "texts", "starts"),
    [
        (
            "ten chunks: six, the first and the last among them",
            [f"chunk{at} " + "x" * 2000 for at in range(10)],
            ["chunk0", "chunk2", "chunk4", "chunk5", "chunk7", "chunk9"],
        ),
        (
            "two chunks: each once, not one of them twice",
            ["first " + "x" * 5000, "second " + "y" * 5000],
            ["first", "second"],
        ),
    ],
)
def test_a_long_section_is_read_as_chunks_spread_over_it(
    name: str, texts: list[str], starts: list[str]
) -> None:
    """Too long together: chunks spread from the first to the last, each cut to its share, so the
    model sees the whole span rather than the opening alone."""
    pieces = generated.excerpt(texts).split("\n\n[...]\n\n")

    assert [piece.split()[0] for piece in pieces] == starts, name
    assert {len(piece) for piece in pieces} == {generated.EXCERPT_CHARS // len(starts)}, name


def test_each_section_is_one_prompt_with_its_heading_path() -> None:
    """The whole document is named as such, a section by its heading path; a section with no
    prose (code or tables alone) is not asked about, and has no descriptors."""
    prompts: list[str] = []

    def reply(prompt: str, max_tokens: int) -> str:
        prompts.append(prompt)
        assert max_tokens == generated.REPLY_TOKENS
        return "Topic one | Topic two"

    texts = [SAGAS, QUORUMS, "\n\n"]
    runs = [Run((), 0, 2), Run(("Book", "Sagas"), 0, 0), Run(("Book", "Code"), 2, 2)]

    picked = generated.Generated(reply).pick(texts, runs, None, None)

    assert picked == [["Topic one", "Topic two"], ["Topic one", "Topic two"], []]
    assert len(prompts) == 2, "the code-only section was not asked about"
    assert "Heading path: (the whole document)" in prompts[0]
    assert SAGAS in prompts[0] and QUORUMS in prompts[0]
    assert "Heading path: Book > Sagas" in prompts[1]
    assert SAGAS in prompts[1] and QUORUMS not in prompts[1], "its own chunks only"


@pytest.mark.parametrize(
    ("name", "answer", "expected"),
    [
        ("an echo of the path goes", "Sagas | Compensating steps", ["Compensating steps"]),
        ("echoes alone are kept rather than nothing", "Sagas | Book", ["Sagas", "Book"]),
    ],
)
def test_a_section_keeps_what_its_heading_path_does_not_say(
    name: str, answer: str, expected: list[str]
) -> None:
    pick = generated.Generated(lambda prompt, max_tokens: answer).pick
    assert pick([SAGAS], [Run(("Book", "Sagas"), 0, 0)], None, None) == [expected], name


def _section(headings: list[str], descriptors: list[str]) -> Section:
    """A section of a book as the merge names it; only its headings and descriptors are read."""
    return Section(
        id="7hQ2vK9mR3xW1pL5nB8cT4",
        parent_id=None,
        headings=headings,
        seq_start=1,
        seq_end=2,
        line_start=1,
        line_end=40,
        char_start=0,
        char_end=1800,
        byte_start=0,
        byte_end=1800,
        page_start=1,
        page_end=3,
        descriptors=descriptors,
    )


BOOK = [
    _section([], ["Distributed transactions"]),
    _section(["Sagas"], ["Compensating steps", "Local transactions"]),
    _section(["Sagas", "Orchestration"], ["Central coordinator"]),
    _section(["Quorums"], []),
]


@pytest.mark.parametrize(
    ("name", "sections", "expected"),
    [
        ("no sections", [], ""),
        ("the whole document alone has no heading", BOOK[:1], ""),
        (
            "every depth, indented, with descriptors where there are some",
            BOOK,
            "- Sagas (Compensating steps, Local transactions)\n"
            "  - Orchestration (Central coordinator)\n"
            "- Quorums",
        ),
        (
            "too long: the deeper headings go first",
            [
                *BOOK[:2],
                *(_section(["Sagas", f"Step {n}"], ["x" * 40]) for n in range(100)),
                BOOK[3],
            ],
            "- Sagas (Compensating steps, Local transactions)\n- Quorums",
        ),
    ],
)
def test_the_outline_is_every_heading_with_its_descriptors(
    name: str, sections: list[Section], expected: str
) -> None:
    assert generated.outline(sections) == expected, name


def test_an_outline_too_long_at_its_first_depth_is_cut() -> None:
    chapters = [_section([f"Chapter {n}"], ["y" * 60]) for n in range(100)]
    text = generated.outline(chapters)
    assert len(text) == generated.OUTLINE_CHARS
    assert text.startswith("- Chapter 0 (y")


@pytest.mark.parametrize(
    ("name", "answer", "expected"),
    [
        ("plain sentences", "It covers sagas. And quorums.", "It covers sagas. And quorums."),
        ("a label goes", "Summary: It covers sagas.", "It covers sagas."),
        (
            "lines and markdown marks join into one paragraph",
            "**Summary:**\n\n- It covers *sagas*.\n- It covers quorums.",
            "It covers *sagas*. It covers quorums.",
        ),
        (
            "a numbered list: its numbers are no sentences",
            "1. Explains sagas.\n2) Covers quorums.\n3. Shows clocks.",
            "Explains sagas. Covers quorums. Shows clocks.",
        ),
        (
            "a sentence goes on past e.g. and i.e.",
            "Covers tools, e.g. Docker, i.e. containers. Shows more.",
            "Covers tools, e.g. Docker, i.e. containers. Shows more.",
        ),
        ("more than five sentences", "A. B. C. D. E. F. G.", "A. B. C. D. E."),
        ("a last sentence cut off goes", "It covers sagas. It also", "It covers sagas."),
        ("a lone sentence cut off stays", "It covers sagas and", "It covers sagas and"),
        ("an empty answer", "", ""),
        ("blank lines only", "\n \n", ""),
    ],
)
def test_sentences_read_the_summary_back(name: str, answer: str, expected: str) -> None:
    assert generated.sentences(answer) == expected, name


@pytest.mark.parametrize(
    ("name", "sections", "texts", "asked", "in_prompt"),
    [
        ("headings and prose", BOOK, [SAGAS, QUORUMS], True, ["- Sagas (Compensating", SAGAS]),
        ("prose without headings", BOOK[:1], [SAGAS], True, ["Outline (each heading", "(none)"]),
        ("headings without prose", BOOK, ["\n\n"], True, ["  - Orchestration"]),
        ("neither: nothing to read", BOOK[:1], ["\n\n"], False, []),
    ],
)
def test_a_document_is_summarized_from_its_outline_and_an_excerpt(
    name: str, sections: list[Section], texts: list[str], asked: bool, in_prompt: list[str]
) -> None:
    prompts: list[str] = []

    def reply(prompt: str, max_tokens: int) -> str:
        prompts.append(prompt)
        assert max_tokens == generated.SUMMARY_TOKENS
        return "Summary: It covers sagas. It covers quorums."

    summary = generated.summarize(sections, texts, reply)

    assert summary == ("It covers sagas. It covers quorums." if asked else ""), name
    assert len(prompts) == asked, name
    for part in in_prompt:
        assert part in prompts[0], (name, part)


@pytest.mark.parametrize(
    ("name", "descriptions", "asked", "in_prompt"),
    [
        (
            "each description numbered, the count named",
            ["Explains sagas.", "  ", "Covers quorums."],
            True,
            ["A person collected 2 documents.", "1. Explains sagas.", "2. Covers quorums."],
        ),
        ("no description: nothing to read", ["", " \n"], False, []),
    ],
)
def test_a_collection_is_summarized_from_its_documents_descriptions(
    name: str, descriptions: list[str], asked: bool, in_prompt: list[str]
) -> None:
    prompts: list[str] = []

    def reply(prompt: str, max_tokens: int) -> str:
        prompts.append(prompt)
        assert max_tokens == generated.COLLECTION_TOKENS
        return "Spans systems. Covers sagas. Covers quorums. A. B. C. D. E."

    summary = generated.summarize_collection(descriptions, reply)

    expected = "Spans systems. Covers sagas. Covers quorums. A. B. C. D." if asked else ""
    assert summary == expected, f"{name}: at most {generated.COLLECTION_SENTENCES} sentences"
    assert len(prompts) == asked, name
    for part in in_prompt:
        assert part in prompts[0], (name, part)


def test_a_collection_too_large_reads_each_description_cut_to_its_share() -> None:
    prompts: list[str] = []
    generated.summarize_collection(
        [f"Book {n} " + "x" * 400 for n in range(100)],
        lambda prompt, max_tokens: prompts.append(prompt) or "Spans books.",
    )
    share = generated.COLLECTION_CHARS // 100
    assert f"1. Book 0 {'x' * (share - len('Book 0 '))}\n2. Book 1" in prompts[0]
