"""`generate.py` without Claude: a fake `ask` stands in for `claude -p`, so these prove what is
asked, what is kept, what is cached, and that nothing becomes gold without `accept`."""

import hashlib
import json
from pathlib import Path

import pytest

from evals.bookqa import generate, schema, sources
from evals.bookqa.sources import Segment
from evals.bookqa.tests import books

DRAFTS = [
    {
        "query": "How many attempts may worker 65 make?",
        "query_type": "direct",
        "answerable": True,
        "expected_answer": "Worker 65 may make 455 attempts.",
        "expected_facts": ["455 attempts"],
        "passages": [{"quote": books.LINES[65], "page": 2, "section": ""}],
    },
    {
        "query": "Which worker runs the nightly compaction?",
        "query_type": "paraphrase",
        "answerable": False,
        "expected_answer": "The book does not say.",
        "expected_facts": [],
        "passages": [],
    },
]
REPLY = f"Here are the questions:\n```json\n{json.dumps(DRAFTS)}\n```"


class FakeClaude:
    def __init__(self, reply: str = REPLY) -> None:
        self.reply = reply
        self.prompts: list[tuple[str, str]] = []

    def __call__(self, prompt: str, model: str) -> tuple[str, str]:
        self.prompts.append((prompt, model))
        return self.reply, f"claude-{model}-test"


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(generate, "CANDIDATES", tmp_path / "candidates")
    return books.corpus(tmp_path / "corpus")


def _generate(
    corpus: Path,
    ask: FakeClaude,
    seed: int = 1,
    model: str = "sonnet",
    dataset: Path | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Every segment of the PDF (there is one), two questions each."""
    dataset = dataset or corpus.parent / "dataset.jsonl"
    return generate.generate(corpus / books.PDF, 0, 2, seed, model, dataset, ask, dry_run)


def test_the_generation_key_is_stable_and_changes_with_every_input() -> None:
    key = generate.generation_key("a" * 64, "p001-012", 1, "sonnet", 5)

    assert generate.generation_key("a" * 64, "p001-012", 1, "sonnet", 5) == key
    assert {
        generate.generation_key("b" * 64, "p001-012", 1, "sonnet", 5),
        generate.generation_key("a" * 64, "p013-024", 1, "sonnet", 5),
        generate.generation_key("a" * 64, "p001-012", 2, "sonnet", 5),
        generate.generation_key("a" * 64, "p001-012", 1, "opus", 5),
        generate.generation_key("a" * 64, "p001-012", 1, "sonnet", 6),
    }.isdisjoint({key}), "every input is in the key"


def test_a_new_prompt_version_is_a_new_key(monkeypatch: pytest.MonkeyPatch) -> None:
    key = generate.generation_key("a" * 64, "p001-012", 1, "sonnet", 5)
    monkeypatch.setattr(generate, "PROMPT_VERSION", "v2")

    assert generate.generation_key("a" * 64, "p001-012", 1, "sonnet", 5) != key


def test_segments_are_drawn_by_seed_and_kept_in_document_order() -> None:
    segments = [Segment(f"part{i:02d}", None, None, "text") for i in range(10)]

    draws = {tuple(s.label for s in generate.select(segments, 3, seed)) for seed in range(1, 6)}

    assert generate.select(segments, 3, 7) == generate.select(segments, 3, 7)
    assert all(list(draw) == sorted(draw) and len(draw) == 3 for draw in draws)
    assert len(draws) > 1, "the seed decides which segments are read"
    assert generate.select(segments, 0, 1) == segments


def test_a_pdf_is_cut_into_page_runs_and_a_text_into_paragraph_runs(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sources, "MIN_CHARS", 10)
    monkeypatch.setattr(sources, "MAX_PAGES", 1)
    monkeypatch.setattr(sources, "MAX_CHARS", 85)  # the two sections of `books.NOTES`, apart

    pdf = sources.segments(corpus / books.PDF)
    notes = sources.segments(corpus / books.MARKDOWN)

    assert [(s.label, s.first_page, s.last_page) for s in pdf] == [
        ("p001-001", 1, 1),
        ("p002-002", 2, 2),
    ]
    assert pdf[1].text.startswith("[page 2]\n") and books.LINES[65] in pdf[1].text
    assert [s.label for s in notes] == ["part01", "part02"]
    assert "Leases" in notes[0].text and "Fencing" in notes[1].text


def test_the_prompt_carries_the_text_the_pages_and_the_seed(corpus: Path) -> None:
    (segment,) = sources.segments(corpus / books.PDF)

    text = generate.prompt(books.PDF, segment, 4, 9)

    assert books.LINES[65] in text and "[page 2]" in text
    assert "pages 1 to 2" in text and "Variation seed: 9" in text and "write 4 questions" in text


def test_the_reply_is_read_through_a_code_fence_and_prose() -> None:
    drafts = generate.parse(REPLY)

    assert [d.query for d in drafts] == [d["query"] for d in DRAFTS]
    assert drafts[0].passages[0].page == 2


@pytest.mark.parametrize(
    "reply",
    ["I could not find any questions.", '[{"query": "What?"}]', '[{"query_type": "trivia"}]'],
)
def test_a_reply_that_is_not_a_list_of_questions_is_an_error(reply: str) -> None:
    with pytest.raises(generate.GenerationError):
        generate.parse(reply)


def test_records_get_ids_and_metadata_from_their_generation(corpus: Path) -> None:
    made = generate.records(
        generate.parse(REPLY), books.PDF, "f" * 64, "p001-002", 3, "claude-x", "abcdef99", "t0"
    )

    assert [r.id for r in made] == ["book-abcdef-01", "book-abcdef-02"]
    assert made[0].meta == schema.Generation(
        schema.SCHEMA_VERSION, "f" * 64, "claude-x", generate.PROMPT_VERSION, 3, "p001-002", "t0"
    )
    assert (made[0].relevant_documents, made[0].relevant_passages[0].document) == (
        [books.PDF],
        books.PDF,
    )
    assert (made[1].answerable, made[1].relevant_documents) == (False, [])


def test_generation_writes_candidates_and_never_the_dataset(corpus: Path) -> None:
    ask = FakeClaude()
    dataset = corpus.parent / "dataset.jsonl"

    (path,) = _generate(corpus, ask)

    records, broken = schema.load(path)
    assert path.parent == generate.CANDIDATES / "book" and broken == []
    assert [r.query for r in records] == [d["query"] for d in DRAFTS]
    assert records[0].meta.model == "claude-sonnet-test", "the model that answered, not the alias"
    assert not dataset.exists(), "a candidate is not gold until it is accepted"
    assert schema.validate(records, corpus) == []


def test_a_generation_already_on_disk_is_never_asked_again(corpus: Path) -> None:
    ask = FakeClaude()

    first = _generate(corpus, ask)
    written = first[0].read_text()
    second = _generate(corpus, ask)

    assert len(ask.prompts) == 1
    assert second == first and first[0].read_text() == written


def test_a_new_seed_or_model_is_a_new_generation(corpus: Path) -> None:
    ask = FakeClaude()

    (seed_1,) = _generate(corpus, ask)
    (seed_2,) = _generate(corpus, FakeClaude(REPLY.replace("worker 65", "worker 66")), seed=2)
    (opus,) = _generate(corpus, FakeClaude(REPLY.replace("worker 65", "worker 67")), model="opus")

    assert len({seed_1, seed_2, opus}) == 3


def test_a_question_already_in_the_dataset_or_a_candidate_is_dropped(corpus: Path) -> None:
    dataset = corpus.parent / "dataset.jsonl"
    dataset.write_text(schema.dump([books.record(corpus)]))

    (path,) = _generate(corpus, FakeClaude(), dataset=dataset)
    (again,) = _generate(corpus, FakeClaude(), seed=2, dataset=dataset)

    assert [r.query for r in schema.load(path)[0]] == [DRAFTS[1]["query"]]
    assert schema.load(again)[0] == [], "both questions are taken by now"


def test_a_dry_run_asks_nothing_and_writes_nothing(corpus: Path) -> None:
    ask = FakeClaude()

    assert _generate(corpus, ask, dry_run=True) == []
    assert ask.prompts == [] and not generate.CANDIDATES.exists()


def test_accept_moves_only_the_candidates_that_pass_review(corpus: Path) -> None:
    dataset = corpus.parent / "dataset.jsonl"
    wrong_page = {
        **DRAFTS[0],
        "query": "What retry budget does worker 65 have?",
        "passages": [{"quote": books.LINES[65], "page": 1}],
    }
    (good,) = _generate(corpus, FakeClaude())
    (bad,) = _generate(corpus, FakeClaude(json.dumps([wrong_page])), seed=2)

    accepted, issues = generate.accept([good, bad], dataset, corpus)

    assert [r.query for r in accepted] == [d["query"] for d in DRAFTS]
    assert [r.id for r in schema.load(dataset)[0]] == [r.id for r in accepted]
    assert [i.problem for i in issues] == [f"quote is on page 2 of {books.PDF}, not 1"]
    again, _ = generate.accept([good, bad], dataset, corpus)
    assert again == [] and len(schema.load(dataset)[0]) == 2, "accepting twice adds nothing"


RELATION_DRAFTS = [
    {
        "query": "How does worker 65's retry budget compare with worker 3's?",
        "query_type": "relationship",
        "relation": "correlates_positively",
        "answerable": True,
        "expected_answer": "Worker 65 may retry 455 times, far more than worker 3's 21.",
        "expected_facts": ["worker 65: 455 attempts", "worker 3: 21 attempts"],
        "passages": [
            {"quote": books.LINES[3], "page": 1, "section": ""},
            {"quote": books.LINES[65], "page": 2, "section": ""},
        ],
    }
]


def test_the_facts_key_is_the_one_it_was_before_relations() -> None:
    parts = ["a" * 64, "p001-012", "1", "sonnet", "v1", "5"]
    before = hashlib.sha256("\0".join(parts).encode()).hexdigest()

    assert generate.generation_key("a" * 64, "p001-012", 1, "sonnet", 5) == before
    relations = generate.generation_key(
        "a" * 64, "p001-012", 1, "sonnet", 5, generate.RELATIONSHIPS
    )
    assert relations != before, "the two kinds never share a key"


def test_relations_ask_the_relate_prompt_and_keep_the_relation(corpus: Path) -> None:
    ask = FakeClaude(json.dumps(RELATION_DRAFTS))
    dataset = corpus.parent / "relations.jsonl"

    (path,) = generate.generate(
        corpus / books.PDF, 0, 2, 1, "sonnet", dataset, ask, kind=generate.RELATIONSHIPS
    )

    prompt, _ = ask.prompts[0]
    assert "`trades_off`" in prompt and "`analogous`" in prompt
    assert path.name.startswith("relate-")
    (made,) = schema.load(path)[0]
    assert made.relation is schema.Relation.CORRELATES_POSITIVELY
    assert made.meta.prompt_version == f"relate-{generate.RELATE_VERSION}"
    assert schema.validate([made], corpus) == []
    assert not dataset.exists(), "candidates are never gold until accepted"
