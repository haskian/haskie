"""The descriptors a language model writes (`sections.generated`): what it reads, and how its
answer is read back."""

import pytest

from haskie.sections import generated
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
