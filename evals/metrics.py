from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

HASKIE_PREFIX = "mcp__haskie__"
# Only these tools can reach a file directly. `Bash` counts only when its command names a path
# under the document root - most of what Bash does in a coding task is not a lookup at all.
FILE_TOOLS = ("Read", "Grep", "Glob", "Bash")


@dataclass(frozen=True)
class Call:
    seq: int
    name: str
    input: dict
    result: str  # the tool's result text; empty if the transcript ends before it answers


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def calls(transcript: str) -> list[Call]:
    """Every tool call the agent made, paired with its result, in order.

    Parsed from `tool_use`/`tool_result` blocks rather than a text search over the whole
    transcript. The transcript's `system.init` event lists every tool the MCP server exposes -
    `mcp__haskie__search` included - whether or not the agent ever calls it, so a substring
    search over the raw text would read every run as having used haskie.
    """
    events = [json.loads(line) for line in transcript.splitlines() if line.strip()]
    results = {
        block["tool_use_id"]: _text(block.get("content"))
        for event in events
        if event.get("type") == "user"
        for block in event.get("message", {}).get("content", [])
        if block.get("type") == "tool_result"
    }
    found = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "tool_use":
                found.append(
                    Call(
                        len(found),
                        block.get("name", ""),
                        block.get("input") or {},
                        results.get(block["id"], ""),
                    )
                )
    return found


def is_haskie(call: Call) -> bool:
    return call.name.startswith(HASKIE_PREFIX)


def _mentions(value: object, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, dict):
        return any(_mentions(v, needle) for v in value.values())
    if isinstance(value, list):
        return any(_mentions(v, needle) for v in value)
    return False


def is_doc_store_lookup(call: Call, doc_root: Path) -> bool:
    """A call that reached a haskie-managed document directly, bypassing search."""
    return call.name in FILE_TOOLS and _mentions(call.input, str(doc_root))


def disclosed_paths(call: Call, doc_root: Path) -> set[str]:
    """Document-store paths this haskie call's result handed back.

    A hit carries `source_file`/`markdown_file` as absolute paths - this is what makes
    `grep <that path>` possible at all, and it is what "knows where the markdown lives" means in
    practice.
    """
    return set(re.findall(re.escape(str(doc_root)) + r"[\w./+-]*", call.result))


@dataclass(frozen=True)
class Behaviour:
    lookups: int  # haskie calls + doc-store lookups, combined
    haskie_calls: int
    doc_store_lookups: int
    search_first: bool | None  # None when the run never looked anything up at all
    legitimate_doc_store_lookups: int  # a prior haskie result had already named that path
    illegitimate_doc_store_lookups: int  # reached a document with no search justifying it
    substitution_rate: float | None  # share of lookups that were doc-store, not haskie


def behaviour(transcript: str, doc_root: Path) -> Behaviour:
    every = calls(transcript)
    lookups = [c for c in every if is_haskie(c) or is_doc_store_lookup(c, doc_root)]
    doc_store = [c for c in every if is_doc_store_lookup(c, doc_root)]

    search_first = is_haskie(lookups[0]) if lookups else None

    seen: set[str] = set()
    legitimate = illegitimate = 0
    for call in every:
        if is_haskie(call):
            seen.update(disclosed_paths(call, doc_root))
        elif is_doc_store_lookup(call, doc_root):
            if any(_mentions(call.input, path) for path in seen):
                legitimate += 1
            else:
                illegitimate += 1

    return Behaviour(
        lookups=len(lookups),
        haskie_calls=sum(1 for c in every if is_haskie(c)),
        doc_store_lookups=len(doc_store),
        search_first=search_first,
        legitimate_doc_store_lookups=legitimate,
        illegitimate_doc_store_lookups=illegitimate,
        substitution_rate=(len(doc_store) / len(lookups)) if lookups else None,
    )


def retrieved(transcript: str, evidence: list[str]) -> list[str]:
    """Which of a task's evidence documents a haskie call surfaced: its name appears in a haskie
    result. By name rather than by disclosed path, since `search_sources` returns document rows
    without on-disk paths. Separate from `behaviour`: this checks retrieval quality, not tool
    choice - a run can search correctly and still miss the right document, or vice versa."""
    surfaced = " ".join(call.result for call in calls(transcript) if is_haskie(call))
    return [name for name in evidence if name in surfaced]


def opened(transcript: str, evidence: list[str]) -> list[str]:
    """Arm D's counterpart to `retrieved`: which evidence files a file tool named in its input.

    Inputs only, never results - a directory listing names every file in the corpus, and counting
    that as retrieval would score every run as having found its evidence."""
    named = [call for call in calls(transcript) if call.name in FILE_TOOLS]
    return [name for name in evidence if any(_mentions(call.input, name) for call in named)]


@dataclass(frozen=True)
class Cost:
    turns: int
    tokens: int  # input, cache reads, cache writes and output together
    usd: float


def cost(transcript: str) -> Cost:
    """From the closing `result` event Claude Code writes once per run, which carries the whole
    run's totals. A run killed before that event gets zeros, not a guess."""
    for line in reversed(transcript.splitlines()):
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") != "result":
            continue
        usage = event.get("usage") or {}
        tokens = sum(
            usage.get(key, 0)
            for key in (
                "input_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "output_tokens",
            )
        )
        return Cost(event.get("num_turns", 0), tokens, event.get("total_cost_usd", 0.0))
    return Cost(0, 0, 0.0)
