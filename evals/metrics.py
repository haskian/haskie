"""Trace-derived measures of retrieval strategy.

The eval is interested in *how* Claude gathers knowledge, not only whether the final code passes.
In particular, it distinguishes Haskie discovery from later filesystem inspection. Grepping a file
that Haskie already identified is not the same behaviour as grepping the document store before
using Haskie at all.
"""

import msgspec

from evals.trace import HASKIE_RETRIEVAL, ToolCall, Trace, mentions


def is_lookup(call: ToolCall) -> bool:
    """A successful knowledge-gathering call.

    Haskie metadata calls such as ``list_collections`` are deliberately excluded: they establish
    what exists, but they are not evidence retrieval. A Bash call counts only when it names the
    document store.
    """
    if call.is_error:
        return False
    if call.kind == "haskie":
        return call.name in HASKIE_RETRIEVAL
    if call.kind in {"grep", "glob", "read", "web"}:
        return True
    return call.kind == "bash" and call.on_library


def lookups(trace: Trace) -> list[ToolCall]:
    return [call for call in trace.calls if is_lookup(call)]


def haskie_calls(trace: Trace) -> list[ToolCall]:
    return [call for call in trace.calls if call.kind == "haskie"]


def haskie_lookups(trace: Trace) -> list[ToolCall]:
    return [call for call in trace.calls if call.kind == "haskie" and call.name in HASKIE_RETRIEVAL]


def queries(trace: Trace) -> list[str]:
    found = []
    for call in haskie_lookups(trace):
        for name in ("q", "query"):
            if isinstance(call.input.get(name), str):
                found.append(call.input[name])
    return found


def substitution_rate(trace: Trace) -> float | None:
    """Share of knowledge-gathering calls that did not use Haskie retrieval."""
    gathered = lookups(trace)
    if not gathered:
        return None
    return sum(1 for call in gathered if call.kind != "haskie") / len(gathered)


def leaks(trace: Trace) -> list[ToolCall]:
    """Filesystem lookups of a path previously disclosed by a Haskie retrieval call."""
    seen: set[str] = set()
    found = []
    for call in trace.calls:
        if call.kind == "haskie" and call.name in HASKIE_RETRIEVAL:
            seen.update(call.returned)
            continue
        if is_lookup(call) and call.on_library and any(mentions(call.input, path) for path in seen):
            found.append(call)
    return found


def filesystem_first(trace: Trace) -> bool:
    """Whether the first actual knowledge lookup touched the document store without Haskie."""
    first = next(iter(lookups(trace)), None)
    return first is not None and first.kind != "haskie" and first.on_library


def first_lookup_kind(trace: Trace) -> str:
    first = next(iter(lookups(trace)), None)
    return "none" if first is None else ("haskie" if first.kind == "haskie" else first.kind)


def haskie_then_filesystem(trace: Trace) -> int:
    """Count filesystem lookups after Haskie has already disclosed at least one path."""
    seen: set[str] = set()
    count = 0
    for call in trace.calls:
        if call.kind == "haskie" and call.name in HASKIE_RETRIEVAL:
            seen.update(call.returned)
        elif is_lookup(call) and call.on_library and any(
            mentions(call.input, path) for path in seen
        ):
            count += 1
    return count


def spills(trace: Trace) -> list[ToolCall]:
    return [call for call in haskie_lookups(trace) if call.spilled_to is not None]


class Summary(msgspec.Struct):
    session_id: str
    ok: bool
    num_turns: int
    cost_usd: float
    duration_ms: int
    lookups: int
    haskie: int
    substitution_rate: float | None
    filesystem_first: bool
    first_lookup: str
    haskie_then_filesystem: int
    leaks: int
    spills: int
    denied: int
    queries: list[str]


def summarize(trace: Trace) -> Summary:
    return Summary(
        session_id=trace.session_id,
        ok=trace.ok,
        num_turns=trace.num_turns,
        cost_usd=trace.cost_usd,
        duration_ms=trace.duration_ms,
        lookups=len(lookups(trace)),
        haskie=len(haskie_lookups(trace)),
        substitution_rate=substitution_rate(trace),
        filesystem_first=filesystem_first(trace),
        first_lookup=first_lookup_kind(trace),
        haskie_then_filesystem=haskie_then_filesystem(trace),
        leaks=len(leaks(trace)),
        spills=len(spills(trace)),
        denied=len(trace.denied),
        queries=queries(trace),
    )


class Trigger(msgspec.Struct):
    searched_when_needed: int
    needed: int
    searched_when_not: int
    not_needed: int

    @property
    def precision(self) -> float | None:
        searched = self.searched_when_needed + self.searched_when_not
        return self.searched_when_needed / searched if searched else None

    @property
    def recall(self) -> float | None:
        return self.searched_when_needed / self.needed if self.needed else None


def trigger(runs: list[tuple[bool, bool]]) -> Trigger:
    return Trigger(
        searched_when_needed=sum(1 for expected, searched in runs if expected and searched),
        needed=sum(1 for expected, _ in runs if expected),
        searched_when_not=sum(1 for expected, searched in runs if not expected and searched),
        not_needed=sum(1 for expected, _ in runs if not expected),
    )
