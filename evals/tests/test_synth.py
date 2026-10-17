"""The synthetic corpus only measures what it claims to if its dials are clean: sizes that differ
in nothing but size, a level 2 that really avoids the runbook's wording, and output that is the
same every time for a seed."""

import json
import random
from pathlib import Path

import pytest

from evals import synth


@pytest.fixture
def root(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(synth, "ROOT", tmp_path)
    monkeypatch.setattr(synth, "SYNTH_DIR", tmp_path / "synth")
    monkeypatch.setattr(synth, "TASK_ROOT", tmp_path / "tasks")
    return tmp_path


def _meta(root: Path, size: int, level: int) -> dict:
    return json.loads((root / "tasks" / synth.task_name(1, size, level) / "meta.json").read_text())


def test_the_data_tables_satisfy_their_own_invariants() -> None:
    synth._assert_paraphrases_disjoint()
    synth._assert_boilerplate_neutral()


def test_generate_writes_every_task_in_the_grid(root: Path) -> None:
    names = synth.generate(1)

    assert names == synth.task_names(1)
    assert all((root / "tasks" / name / "test_task.py").exists() for name in names)


def test_sizes_nest_so_only_the_amount_of_distraction_changes(root: Path) -> None:
    synth.generate(1)
    corpora = [synth.corpus_dir(1, size) for size in synth.SIZES]

    for smaller, larger in zip(corpora, corpora[1:], strict=False):
        for doc in smaller.iterdir():
            assert (larger / doc.name).read_text() == doc.read_text()


def test_the_target_is_in_every_size(root: Path) -> None:
    synth.generate(1)
    target = _meta(root, synth.SIZES[0], 0)["evidence"][0]

    for size in synth.SIZES:
        assert (synth.corpus_dir(1, size) / target).exists()


def test_only_level_zero_names_the_target(root: Path) -> None:
    synth.generate(1)
    target_doc = synth.corpus_dir(1, synth.SIZES[0]) / _meta(root, 5, 0)["evidence"][0]
    name = target_doc.read_text().splitlines()[0].removeprefix("# ")

    prompts = [
        (root / "tasks" / synth.task_name(1, 5, level) / "task.md").read_text()
        for level in synth.LEVELS
    ]
    assert name in prompts[0]
    assert name not in prompts[1]
    assert name not in prompts[2]


def _kind(root: Path, kind: str, size: int = 50) -> dict:
    return json.loads(
        (root / "tasks" / synth.kind_task_name(1, kind, size) / "meta.json").read_text()
    )


def test_the_base_runbooks_are_unchanged_when_no_pointer_is_added() -> None:
    """Already-imported documents depend on this: the same name must keep the same content."""
    service = synth.services(1, 5)[0]
    assert synth.runbook(service, random.Random("x")) == synth.runbook(
        service, random.Random("x"), ""
    )


def test_supersede_gives_the_target_a_notice_that_changes_its_port(root: Path) -> None:
    synth.generate(1)
    runbook, notice = (
        synth.corpus_dir(1, 50, "supersede") / d for d in _kind(root, "supersede")["evidence"]
    )

    port = next(line for line in runbook.read_text().splitlines() if "listen port" in line)
    assert port.split("|")[2].strip() in notice.read_text()  # the notice names the old port
    assert "takes precedence" in notice.read_text()


def test_absent_leaves_the_target_out_but_keeps_its_decoys(root: Path) -> None:
    synth.generate(1)
    base, absent = synth.corpus_dir(1, 50), synth.corpus_dir(1, 50, "absent")
    target = _meta(root, 5, 0)["evidence"][0]

    assert (base / target).exists()
    assert not (absent / target).exists()
    assert {p.name for p in absent.iterdir()} == {p.name for p in base.iterdir()} - {target}


def test_pdf_hides_its_text_from_a_plain_byte_search(root: Path) -> None:
    synth.generate(1)
    (target,) = _kind(root, "pdf")["evidence"]
    pdf = (synth.corpus_dir(1, 50, "pdf") / target).read_bytes()

    assert pdf.startswith(b"%PDF-")
    assert b"listen port" not in pdf


def test_multihop_puts_the_answer_on_a_separate_team_page(root: Path) -> None:
    synth.generate(1)
    runbook, page = (
        synth.corpus_dir(1, 50, "multihop") / d for d in _kind(root, "multihop")["evidence"]
    )

    assert "On-call channel" not in runbook.read_text()
    assert "On-call channel" in page.read_text()


def test_a_second_seed_is_a_different_organization_under_different_names(root: Path) -> None:
    """Replication needs other targets and other values - and names that can't collide with the
    first seed's documents, collections or tasks, since both live in the same instances."""
    synth.generate(1)
    synth.generate(2)
    one, two = synth.layout(1), synth.layout(2)

    assert one[0][one[1]].name != two[0][two[1]].name
    assert not {p.name for p in synth.corpus_dir(1, 50).iterdir()} & {
        p.name for p in synth.corpus_dir(2, 50).iterdir()
    }
    assert not set(synth.task_names(1)) & set(synth.task_names(2))


def test_generation_is_deterministic_for_a_seed(root: Path) -> None:
    synth.generate(1)
    first = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    synth.generate(1)
    second = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}

    assert first == second
