"""The descriptors and summaries a language model writes (`sections.generated`): what it reads,
and how its answer is read back."""

import msgspec
import pytest

from haskie.sections import generated
from haskie.sections.build import Section
from haskie.sections.descriptors import Description, Run

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
            "more than six phrases",
            "Sagas | Quorums | Replication | Consensus | Transactions | Recovery | Logging | "
            "Sharding",
            ["Sagas", "Quorums", "Replication", "Consensus", "Transactions", "Recovery"],
        ),
        ("one phrase without a separator", "Replica staleness", ["Replica staleness"]),
        ("an empty answer", "", []),
        ("blank lines only", "\n  \n", []),
    ],
)
def test_parse_reads_the_first_list_line(name: str, answer: str, expected: list[str]) -> None:
    assert generated.parse(answer) == expected, name


@pytest.mark.parametrize("empty", [False, True])
def test_section_descriptions_read_only_the_sections_prose(empty: bool) -> None:
    prompts: list[str] = []

    def reply(prompt: str, max_tokens: int) -> str:
        prompts.append(prompt)
        assert max_tokens == generated.SECTION_TOKENS
        return "Description: Explains sagas. Covers compensation. Shows retries."

    text = "  " if empty else SAGAS
    result = generated.describe_section(Run(("Book", "Sagas"), 1, 1), [QUORUMS, text], reply)
    assert result == ("" if empty else "Explains sagas. Covers compensation.")
    assert len(prompts) == (not empty)
    if prompts:
        assert "Heading path: Book > Sagas" in prompts[0]
        assert SAGAS in prompts[0] and QUORUMS not in prompts[0]


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

    assert picked == [Description(["Topic one", "Topic two"])] * 2 + [Description()]
    assert len(prompts) == 2, "the code-only section was not asked about"
    assert "Heading path: (the whole document)" in prompts[0]
    assert "up to 6 descriptors" in prompts[0]
    assert "even when the heading already names them" in prompts[0]
    assert SAGAS in prompts[0] and QUORUMS in prompts[0]
    assert "Heading path: Book > Sagas" in prompts[1]
    assert SAGAS in prompts[1] and QUORUMS not in prompts[1], "its own chunks only"


# A chapter of chunks 0 to 3: two sections, the first with a subsection, and a sibling chapter.
CHAPTER = Run(("Book", "Sagas"), 0, 3)
ORCHESTRATION = Run(("Book", "Sagas", "Orchestration"), 0, 1)
TIMEOUTS = Run(("Book", "Sagas", "Orchestration", "Timeouts"), 1, 1)
RECOVERY = Run(("Book", "Sagas", "Recovery"), 2, 3)
QUORUM = Run(("Book", "Quorums"), 4, 4)


@pytest.mark.parametrize(
    ("name", "run", "known", "expected"),
    [
        (
            "every section under it, at any depth, in document order",
            CHAPTER,
            [(RECOVERY, []), (TIMEOUTS, []), (QUORUM, []), (ORCHESTRATION, [])],
            [ORCHESTRATION, TIMEOUTS, RECOVERY],
        ),
        ("not itself, nor a sibling", ORCHESTRATION, [(ORCHESTRATION, []), (RECOVERY, [])], []),
        ("a leaf has none", TIMEOUTS, [(CHAPTER, []), (ORCHESTRATION, [])], []),
        (
            "the whole document holds every section",
            Run((), 0, 4),
            [(QUORUM, []), (CHAPTER, [])],
            [CHAPTER, QUORUM],
        ),
        ("nothing known", CHAPTER, [], []),
    ],
)
def test_subsections(name: str, run: Run, known: list, expected: list[Run]) -> None:
    assert [one for one, _ in generated.subsections(run, known)] == expected, name


def test_an_outline_indents_by_depth_and_names_each_subsections_topics() -> None:
    below = [
        (ORCHESTRATION, ["Saga coordinator"]),
        (TIMEOUTS, ["Step deadlines", "Retries"]),
        (RECOVERY, []),
    ]
    assert generated.subsection_outline(CHAPTER, below) == (
        "- Orchestration (Saga coordinator)\n  - Timeouts (Step deadlines, Retries)\n- Recovery"
    )


@pytest.mark.parametrize(
    ("topics", "cut"),
    [
        pytest.param(1600, False, id="too long whole: the deeper headings go first"),
        pytest.param(4000, True, id="too long even so: cut at the limit"),
    ],
)
def test_an_outline_too_long_drops_depth_then_cuts(topics: int, cut: bool) -> None:
    long = "x" * topics
    below: list[generated.Described] = [
        (ORCHESTRATION, [long]),
        (TIMEOUTS, [long]),
        (RECOVERY, ["Undo"]),
    ]
    outline = generated.subsection_outline(CHAPTER, below)
    if cut:
        assert len(outline) == generated.OUTLINE_CHARS
        assert outline.startswith("- Orchestration (")
    else:
        assert outline.splitlines() == [f"- Orchestration ({long})", "- Recovery (Undo)"]


def test_a_long_section_with_subsections_reads_their_outline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deepest first, as the pipeline orders them: a leaf gets the plain prompt, and a section
    above it, longer than `OUTLINE_FROM_CHARS`, the outline of what was described under it, in this
    pick or before it (`known`)."""
    monkeypatch.setattr(generated, "OUTLINE_FROM_CHARS", 0)
    prompts: dict[str, str] = {}

    def reply(prompt: str, max_tokens: int) -> str:
        heading = prompt.split("Heading path: ", 1)[1].split("\n", 1)[0]
        prompts[heading] = prompt
        return f"{heading.rsplit(' > ', 1)[-1]} topic"

    texts = [SAGAS, QUORUMS, SAGAS, QUORUMS]
    known = [(RECOVERY, ["Compensation"])]  # described by an earlier batch

    picked = generated.Generated(reply, known).pick(
        texts, [TIMEOUTS, ORCHESTRATION, CHAPTER], None, None
    )

    assert picked == [Description(["Timeouts topic"]), Description(["Orchestration topic"]),
                      Description(["Sagas topic"])]  # fmt: skip
    assert "Outline of its subsections" not in prompts["Book > Sagas > Orchestration > Timeouts"]
    assert "- Timeouts (Timeouts topic)" in prompts["Book > Sagas > Orchestration"]
    chapter = prompts["Book > Sagas"]
    assert "Name what the section as a whole is about" in chapter
    assert chapter.split("Outline of its subsections:\n", 1)[1].split("\n\n", 1)[0] == (
        "- Orchestration (Orchestration topic)\n"
        "  - Timeouts (Timeouts topic)\n"
        "- Recovery (Compensation)"
    )


@pytest.mark.parametrize(
    ("name", "answer", "expected"),
    [
        ("heading terms stay", "Sagas | Compensating steps", ["Sagas", "Compensating steps"]),
        ("heading stems stay", "Saga | Compensation", ["Saga", "Compensation"]),
        ("echoes alone are kept rather than nothing", "Sagas | Book", ["Sagas", "Book"]),
    ],
)
def test_a_section_keeps_descriptors_already_in_its_heading_path(
    name: str, answer: str, expected: list[str]
) -> None:
    pick = generated.Generated(lambda prompt, max_tokens: answer).pick
    assert pick([SAGAS], [Run(("Book", "Sagas"), 0, 0)], None, None) == [Description(expected)], (
        name
    )


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


@pytest.mark.parametrize("kind", ["both", "description", "descriptors", "empty"])
def test_a_document_reads_saved_section_descriptions_and_descriptors(kind: str) -> None:
    section = _section([], ["Compensating steps"] if kind in ("both", "descriptors") else [])
    section = msgspec.structs.replace(
        section, description="Explains saga recovery." if kind in ("both", "description") else ""
    )
    prompts: list[str] = []

    def reply(prompt: str, max_tokens: int) -> str:
        prompts.append(prompt)
        assert max_tokens == generated.SUMMARY_TOKENS
        return "Summary: Explains transactions. Covers recovery."

    result = generated.summarize([section], reply)
    assert result == ("" if kind == "empty" else "Explains transactions. Covers recovery.")
    assert len(prompts) == (kind != "empty")
    if prompts:
        assert "Heading: (the whole document)" in prompts[0]
        assert f"Description: {section.description}" in prompts[0]
        assert f"Descriptors: {' | '.join(section.descriptors)}" in prompts[0]


def test_long_section_metadata_is_grouped_without_losing_descriptors() -> None:
    sections = [
        msgspec.structs.replace(_section([f"Chapter {n}"], [f"Topic {n}"]), description="x" * 1500)
        for n in range(20)
    ]
    prompts: list[str] = []
    generated.summarize(
        sections, lambda prompt, tokens: prompts.append(prompt) or "Covers transactions."
    )
    for n in range(20):
        assert any(
            f"Description: {'x' * 1500}\nDescriptors: Topic {n}\n" in prompt + "\n"
            for prompt in prompts
        )
    assert all(
        len(prompt.split("Section information:\n")[1]) <= generated.SUMMARY_CHARS
        for prompt in prompts
    )


@pytest.mark.parametrize("empty_group", [False, True])
def test_long_documents_reduce_all_sections_in_bounded_groups(empty_group: bool) -> None:
    sections = [_section([f"Chapter {n}"], [f"Topic {n}"]) for n in range(260)]
    prompts: list[str] = []

    def reply(prompt: str, max_tokens: int) -> str:
        text = prompt.split("Section information:\n", 1)[1]
        assert len(text) <= generated.SUMMARY_CHARS
        prompts.append(text)
        return "" if empty_group and len(prompts) == 2 else "Covers transactions."

    result = generated.summarize(sections, reply)
    assert result == ("" if empty_group else "Covers transactions.")
    if empty_group:
        assert len(prompts) == 2
    else:
        assert len(prompts) == 17 + 2 + 1
        for n in range(260):
            assert any(f"Descriptors: Topic {n}\n" in text + "\n" for text in prompts[:17])
        assert prompts[-1] == "Covers transactions.\n\nCovers transactions."


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


@pytest.mark.parametrize(
    ("over", "outline"),
    [
        pytest.param(0, False, id="at the bar: the excerpt alone"),
        pytest.param(1, True, id="one character over it: the outline too"),
    ],
)
def test_the_outline_starts_past_its_bar(
    monkeypatch: pytest.MonkeyPatch, over: int, outline: bool
) -> None:
    """Below `OUTLINE_FROM_CHARS` the outline gained nothing when judged: the plain prompt."""
    texts = [SAGAS, QUORUMS, SAGAS, QUORUMS]
    monkeypatch.setattr(generated, "OUTLINE_FROM_CHARS", len("".join(texts)) - over)
    prompts: list[str] = []

    def reply(prompt: str, max_tokens: int) -> str:
        prompts.append(prompt)
        return "Topic"

    generated.Generated(reply, [(RECOVERY, ["Compensation"])]).pick(texts, [CHAPTER], None, None)

    (prompt,) = prompts
    assert ("- Recovery (Compensation)" in prompt) is outline
    assert ("Outline of its subsections" in prompt) is outline


def test_the_outline_bar_is_eight_excerpts() -> None:
    assert generated.OUTLINE_FROM_CHARS == 8 * generated.EXCERPT_CHARS == 48_000
