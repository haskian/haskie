"""Read against a recorded run: one coding task, haskie reachable, no guidance beyond the prompt.

The transcript is the one that reproduced the reported behaviour - three searches too large for
the context, then a grep straight into the document store - so the fixture is also the evidence.
"""

from pathlib import Path

import pytest

from evals import metrics, trace

FIXTURE = Path(__file__).parent / "fixtures" / "coding_run.jsonl"
# The document store the run was recorded against; paths in the fixture are absolute.
LIBRARY = Path("/Users/arianna/.haskie/documents")


@pytest.fixture
def recorded() -> trace.Trace:
    return trace.read(FIXTURE, LIBRARY)


def test_the_run_is_read_back_whole(recorded: trace.Trace) -> None:
    assert recorded.ok
    assert recorded.num_turns == 16
    assert recorded.session_id == "8b8fcd98-cee6-4ef4-be17-4e22ab83bc12"
    assert [c.kind for c in recorded.calls[:5]] == [
        "other",
        "haskie",
        "haskie",
        "haskie",
        "haskie",
    ]



def test_the_terminal_result_is_used_when_multiple_result_events_exist(recorded: trace.Trace) -> None:
    events = [
        {"type": "result", "subtype": "error", "session_id": "old"},
        {
            "type": "result",
            "subtype": "success",
            "session_id": recorded.session_id,
            "num_turns": recorded.num_turns,
            "total_cost_usd": recorded.cost_usd,
            "duration_ms": recorded.duration_ms,
        },
    ]
    result = trace.extract(events, LIBRARY)
    assert result.ok
    assert result.session_id == recorded.session_id

def test_a_refused_call_gathered_nothing(recorded: trace.Trace) -> None:
    """Five Bash calls were refused by the allowlist. A route the run tried is not a route it
    took, so none of them may land in the substitution rate."""
    refused = [c for c in recorded.calls if c.is_error]
    assert len(refused) == 5
    assert not any(c.is_error for c in metrics.lookups(recorded))


def test_a_refusal_quoting_a_path_is_not_a_disclosure(recorded: trace.Trace) -> None:
    """The refusal of call 11 quotes the document path it refused to grep. Only a search
    discloses a path, otherwise the refusal would read as the library handing one back."""
    refused_grep = recorded.calls[11]
    assert refused_grep.is_error
    assert refused_grep.on_library
    assert refused_grep.returned == []


def test_searches_too_large_for_the_context_are_counted(recorded: trace.Trace) -> None:
    """Both `search_text` calls, at the default page size of 100 hits, and `list_documents` came
    back too large and were written to a file: the model saw a notice, not the passages."""
    spilled = metrics.spills(recorded)
    assert [c.name.removeprefix(trace.HASKIE_PREFIX) for c in spilled] == [
        "search_text",
        "search_documents",
        "search_documents",
        "search_text",
    ]


def test_the_grep_that_followed_is_a_leak(recorded: trace.Trace) -> None:
    """One Grep reached a document the searches had already named. That is the behaviour under
    test: the hit carried the path, so the run was told where to look."""
    leaked = metrics.leaks(recorded)
    assert [c.seq for c in leaked] == [12]
    assert leaked[0].name == "Grep"
    assert str(LIBRARY) in leaked[0].input["path"]


def test_substitution_counts_only_what_succeeded(recorded: trace.Trace) -> None:
    gathered = metrics.lookups(recorded)
    assert len(gathered) == 5
    assert metrics.substitution_rate(recorded) == pytest.approx(1 / 5)


def test_a_run_that_gathered_nothing_has_no_rate() -> None:
    """Not searching at all is a different failure from searching in the wrong place, and a rate
    of zero would read as the good one."""
    empty = trace.Trace("s", "m", "/tmp", True, 1, 0.0, 0, [], "", [])
    assert metrics.substitution_rate(empty) is None


def test_a_spilled_result_is_recovered_when_the_file_is_still_there(tmp_path: Path) -> None:
    """The notice names the file. Reading it back is what lets `returned` describe the search
    rather than the truncation."""
    saved = tmp_path / "spilled.txt"
    saved.write_text(f'{{"source_file":"{LIBRARY}/a9/paper.pdf"}}', encoding="utf-8")
    absent = trace.read(FIXTURE, LIBRARY).calls[1].spilled_to
    assert absent is not None
    rewritten = tmp_path / "events.jsonl"
    text = FIXTURE.read_text(encoding="utf-8").replace(absent, str(saved))
    rewritten.write_text(text, encoding="utf-8")
    recovered = trace.read(rewritten, LIBRARY)
    assert recovered.calls[1].spilled_to == str(saved)
    assert recovered.calls[1].returned == [f"{LIBRARY}/a9/paper.pdf"]


def test_haskie_metadata_is_not_counted_as_knowledge_lookup(recorded: trace.Trace) -> None:
    assert not any(c.name.endswith("list_documents") for c in metrics.lookups(recorded))


def test_filesystem_use_after_haskie_is_distinguished_from_filesystem_first(recorded: trace.Trace) -> None:
    assert not metrics.filesystem_first(recorded)
    assert metrics.first_lookup_kind(recorded) == "haskie"
    assert metrics.haskie_then_filesystem(recorded) == 1
