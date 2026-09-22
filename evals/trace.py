"""What one headless Claude Code run did, reduced to the calls that gathered knowledge.

`claude -p --output-format stream-json` writes one JSON object per line, and that stream is what
is read here. The audit trail is no help: `audit.audited` decorates the mutations and not the
searches, and `audit.attach` refuses to store a query or a path in any case. A session does now
keep its own search history (`session.record_search`), but only when the caller passes a
`session_id`, and whether it passes one is part of what is being measured - so the transcript is
the channel that always has the answer.
"""

import json
import re
from pathlib import Path
from typing import Any

import msgspec

from haskie import home

HASKIE_PREFIX = "mcp__haskie__"

# A result too large for the context is replaced by a notice naming the file it was spilled to, so
# the transcript alone does not say what came back. `search_text` at its default page size of 100
# hits spills every time, which is worth counting rather than papering over.
SPILLED = re.compile(r"saved to (/\S+?)\.?(?:\s|$)")

# Only the tools that gather knowledge are named; anything else is `other` and never a lookup.
HASKIE_RETRIEVAL = frozenset({
    "mcp__haskie__get_document",
    "mcp__haskie__search",
    "mcp__haskie__search_text",
    "mcp__haskie__search_documents",
    "mcp__haskie__search_collection",
    "mcp__haskie__document_passages",
    "mcp__haskie__search_cited_works",
})

KINDS = {
    "Grep": "grep",
    "Glob": "glob",
    "Read": "read",
    "NotebookRead": "read",
    "WebSearch": "web",
    "WebFetch": "web",
    "Bash": "bash",
}
# The denominator of the substitution rate. `bash` is not here because most of what Bash does is
# not a lookup; `is_lookup` lets one in when it names the document store.
LOOKUPS = frozenset({"haskie", "grep", "glob", "read", "web"})


class ToolCall(msgspec.Struct):
    seq: int
    name: str
    kind: str
    input: dict[str, Any]
    result_chars: int
    is_error: bool
    on_library: bool  # the call named a path inside the document store
    returned: list[str]  # document-store paths this call's result handed back
    spilled_to: str | None = None
    subagent: bool = False


class Trace(msgspec.Struct):
    session_id: str
    model: str
    cwd: str
    ok: bool
    num_turns: int
    cost_usd: float
    duration_ms: int
    calls: list[ToolCall]
    answer: str
    denied: list[str]


def mentions(value: Any, needle: str) -> bool:
    """Whether `needle` appears anywhere in a tool input, whatever shape that input has.

    One rule instead of a path argument per tool: `Read` names `file_path`, `Grep` names `path`,
    and `Bash` buries it in a command line, and all three are answering the same question.
    """
    if isinstance(value, str):
        return needle in value
    if isinstance(value, dict):
        return any(mentions(v, needle) for v in value.values())
    if isinstance(value, list):
        return any(mentions(v, needle) for v in value)
    return False


def _text(content: Any) -> str:
    """A tool result as one string, whether it came back as text, as blocks, or as a notice."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def _result(found: tuple[dict, dict] | None, root: Path) -> tuple[str, bool, str | None, bool]:
    """`(body, spilled, saved_to, failed)` for one tool call.

    `is_error` comes off the block rather than from the text, because a refusal reads as ordinary
    prose ("This command requires approval") and a refused call gathered nothing.
    """
    if found is None:
        return "", False, None, False  # the run ended before the tool answered
    block, event = found
    failed = bool(block.get("is_error"))
    structured = event.get("tool_use_result")
    content = structured.get("content") if isinstance(structured, dict) else block.get("content")
    body = _text(content)
    spill = SPILLED.search(body)
    if spill is None:
        return body, False, None, failed
    # The spilled file holds what the model was refused, and the paths in it are still paths the
    # model never saw. Read it so `returned` describes the search, not the truncation.
    saved = Path(spill.group(1))
    recovered = saved.read_text(encoding="utf-8") if saved.is_file() else body
    return recovered, True, str(saved), failed


def _returned(body: str, root: Path) -> list[str]:
    """Document-store paths a result handed back, deduplicated, in the order they appear.

    Every hit carries `source_file` and `markdown_file` as absolute paths, which is what makes a
    later grep of that file possible at all.
    """
    found = re.findall(re.escape(str(root)) + r"[\w./+-]*", body)
    return list(dict.fromkeys(found))


def extract(events: list[dict[str, Any]], root: Path | None = None) -> Trace:
    """Reduce a stream-json transcript to its tool calls and the run's outcome."""
    root = root or home.DOCUMENT_ROOT
    init = next((e for e in events if e.get("subtype") == "init"), {})
    # Claude Code can emit more than one result-like terminal record over a stream/session.
    # The terminal result is the last `type=result` event; taking the first one can mark a
    # successfully completed run as unfinished even though the transcript contains a final
    # `subtype=success` result.
    result_events = [e for e in events if e.get("type") == "result"]
    done = result_events[-1] if result_events else {}
    results = {
        block["tool_use_id"]: (block, event)
        for event in events
        if event.get("type") == "user"
        for block in event.get("message", {}).get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    }
    calls: list[ToolCall] = []
    answer: list[str] = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "text":
                answer.append(block["text"])
            if block.get("type") != "tool_use":
                continue
            name = block["name"]
            arguments = block.get("input") or {}
            body, spilled, saved, failed = _result(results.get(block["id"]), root)
            kind = "haskie" if name.startswith(HASKIE_PREFIX) else KINDS.get(name, "other")
            calls.append(
                ToolCall(
                    seq=len(calls),
                    name=name,
                    kind=kind,
                    input=arguments,
                    result_chars=len(body),
                    is_error=failed,
                    on_library=mentions(arguments, str(root)),
                    # Only a search discloses a path. A refusal quotes the command it refused,
                    # which would otherwise read as the library handing that path back.
                    returned=_returned(body, root) if kind == "haskie" and not failed else [],
                    spilled_to=saved,
                    subagent=event.get("parent_tool_use_id") is not None,
                )
            )
    return Trace(
        session_id=str(done.get("session_id") or init.get("session_id") or ""),
        model=str(init.get("model") or ""),
        cwd=str(init.get("cwd") or ""),
        ok=done.get("subtype") == "success",
        num_turns=int(done.get("num_turns") or 0),
        cost_usd=float(done.get("total_cost_usd") or 0.0),
        duration_ms=int(done.get("duration_ms") or 0),
        calls=calls,
        answer="\n".join(answer[-1:]),
        denied=[str(d) for d in done.get("permission_denials") or []],
    )


def read(path: Path, root: Path | None = None) -> Trace:
    lines = path.read_text(encoding="utf-8").splitlines()
    return extract([json.loads(line) for line in lines if line.strip()], root)
