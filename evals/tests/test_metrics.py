"""What `metrics.py` measures, against small hand-built transcripts.

Each transcript is built to be the smallest possible example of one real situation: a run that
never searches, a run that greps a path a search just disclosed, a run that greps first. No test
here spawns an agent - that is `run.py`'s job. These tests only prove the arithmetic is right.
"""

import json
from pathlib import Path

from evals import metrics

DOC_ROOT = Path("/home/eval/.haskie-eval/documents")
OUTSIDE = Path("/home/eval/work")  # the agent's own workspace: never a lookup


def _assistant(tool_use_id: str, name: str, tool_input: dict) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": tool_use_id, "name": name, "input": tool_input}
                ]
            },
        }
    )


def _result(tool_use_id: str, text: str) -> str:
    return json.dumps(
        {
            "type": "user",
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": text}]
            },
        }
    )


def _init_listing_every_tool() -> str:
    """The one line every real transcript starts with: it names every tool the server exposes,
    `mcp__haskie__search` included, before the agent has called anything. A transcript that
    forgets to include this and still passes its test is not proving what it looks like it is."""
    return json.dumps(
        {"type": "system", "subtype": "init", "tools": [f"{metrics.HASKIE_PREFIX}search"]}
    )


def _transcript(*lines: str) -> str:
    return "\n".join((_init_listing_every_tool(), *lines))


def test_a_haskie_call_in_the_tool_listing_alone_is_not_a_call() -> None:
    """The bug this eval already shipped once: `system.init` lists every tool the MCP server
    exposes whether or not the agent ever calls it. A run that touches nothing should measure
    as having touched nothing."""
    empty = _transcript()

    result = metrics.behaviour(empty, DOC_ROOT)

    assert result.haskie_calls == 0
    assert result.lookups == 0
    assert result.search_first is None


def test_search_first_is_true_when_the_first_lookup_is_a_search() -> None:
    transcript = _transcript(
        _assistant("t1", f"{metrics.HASKIE_PREFIX}search_collection", {"q": "reusable barrier"}),
        _result("t1", "no hits"),
        _assistant("t2", "Write", {"file_path": str(OUTSIDE / "barrier.py")}),
    )

    result = metrics.behaviour(transcript, DOC_ROOT)

    assert result.search_first is True
    assert result.haskie_calls == 1


def test_search_first_is_false_when_grep_comes_before_any_search() -> None:
    transcript = _transcript(
        _assistant("t1", "Grep", {"pattern": "barrier", "path": str(DOC_ROOT / "sem.pdf.md")}),
        _result("t1", "3 matches"),
    )

    result = metrics.behaviour(transcript, DOC_ROOT)

    assert result.search_first is False
    assert result.illegitimate_doc_store_lookups == 1
    assert result.legitimate_doc_store_lookups == 0


def test_a_grep_of_a_path_a_search_just_disclosed_is_legitimate() -> None:
    hit_path = str(DOC_ROOT / "ab" / "semaphores.pdf.md")
    transcript = _transcript(
        _assistant("t1", f"{metrics.HASKIE_PREFIX}search_collection", {"q": "reusable barrier"}),
        _result("t1", json.dumps({"source_file": hit_path})),
        _assistant("t2", "Grep", {"pattern": "turnstile2", "path": hit_path}),
        _result("t2", "1 match"),
    )

    result = metrics.behaviour(transcript, DOC_ROOT)

    assert result.search_first is True
    assert result.legitimate_doc_store_lookups == 1
    assert result.illegitimate_doc_store_lookups == 0


def test_a_grep_of_a_path_no_search_ever_disclosed_is_illegitimate_even_after_a_search() -> None:
    """Having searched once does not license grepping anything: the specific path still has to
    trace back to a result, or it is the same default-to-grep behaviour with an unrelated search
    in front of it."""
    transcript = _transcript(
        _assistant("t1", f"{metrics.HASKIE_PREFIX}search_collection", {"q": "reusable barrier"}),
        _result("t1", json.dumps({"source_file": str(DOC_ROOT / "sem.pdf.md")})),
        _assistant("t2", "Read", {"file_path": str(DOC_ROOT / "other.pdf.md")}),
    )

    result = metrics.behaviour(transcript, DOC_ROOT)

    assert result.illegitimate_doc_store_lookups == 1
    assert result.legitimate_doc_store_lookups == 0


def test_substitution_rate_counts_doc_store_lookups_against_haskie_calls() -> None:
    transcript = _transcript(
        _assistant("t1", f"{metrics.HASKIE_PREFIX}search_collection", {"q": "a"}),
        _result("t1", "hits"),
        _assistant("t2", "Grep", {"path": str(DOC_ROOT / "x.pdf.md")}),
        _result("t2", "hits"),
        _assistant("t3", "Grep", {"path": str(DOC_ROOT / "y.pdf.md")}),
        _result("t3", "hits"),
    )

    result = metrics.behaviour(transcript, DOC_ROOT)

    assert result.substitution_rate == 2 / 3


def test_substitution_rate_is_none_rather_than_zero_when_nothing_was_looked_up() -> None:
    """Not searching at all is a different failure from searching in the wrong place, and a
    rate of 0.0 would read as the good outcome."""
    transcript = _transcript(_assistant("t1", "Write", {"file_path": str(OUTSIDE / "x.py")}))

    result = metrics.behaviour(transcript, DOC_ROOT)

    assert result.substitution_rate is None


def test_a_lookup_outside_the_document_root_is_not_counted() -> None:
    """Reading the file the agent itself just wrote is not a lookup against the library, however
    many times it happens."""
    transcript = _transcript(
        _assistant("t1", "Read", {"file_path": str(OUTSIDE / "barrier.py")}),
        _result("t1", "class Barrier..."),
    )

    result = metrics.behaviour(transcript, DOC_ROOT)

    assert result.lookups == 0
    assert result.doc_store_lookups == 0


def test_retrieved_credits_only_what_a_haskie_result_actually_disclosed() -> None:
    """A document name appearing in a later grep's own output does not count as retrieved - only
    haskie's own result is evidence that the search found it."""
    transcript = _transcript(
        _assistant("t1", f"{metrics.HASKIE_PREFIX}search_collection", {"q": "barrier"}),
        _result("t1", json.dumps({"source_file": str(DOC_ROOT / "semaphores.pdf.md")})),
        _assistant("t2", "Grep", {"path": str(DOC_ROOT / "unrelated.pdf.md")}),
        _result("t2", "mentions semaphores.pdf.md in a comment"),
    )

    found = metrics.retrieved(transcript, ["semaphores.pdf.md", "unrelated.pdf.md"])

    assert found == ["semaphores.pdf.md"]


def test_retrieved_credits_a_document_named_without_a_path() -> None:
    """`search_sections` lists documents by name, with no on-disk path - still a retrieval."""
    transcript = _transcript(
        _assistant("t1", f"{metrics.HASKIE_PREFIX}search_sections", {"q": "ledger"}),
        _result("t1", json.dumps({"documents": [{"name": "doc-s1-0003.md"}]})),
    )

    assert metrics.retrieved(transcript, ["doc-s1-0003.md"]) == ["doc-s1-0003.md"]
