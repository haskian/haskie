"""The shape of a search: what runs, in what order, for each of the four answers.

Read this file for what a search does; read `retrieval.py` for what each step does with the IO it
needs, and `passage.py` for the pure folds under them. Nothing here does work — every step is one
line handing the state to one function and passing its answer on — so a stage can be added,
dropped or reordered by reading this file alone.

Four pipelines over one set of steps:

    chunks     retrieve -> merge -> rerank -> hits -> collapse_hits
    passages   retrieve -> merge -> rerank -> hits -> fill_thin -> collapse_ranges -> read
    sources    retrieve -> merge -> rerank -> hits -> shortlist

and the excerpts an agent reads, which run the shared ranking once per question asked:

    answers    (retrieve -> merge -> rerank -> hits -> judge_thin) per question -> fold -> group
               -> budget -> probe_gaps -> fill -> quote

The first four steps are the search every answer shares; what follows is the fold that answer is
made of, and it is a step rather than something every search pays for. `chunks` folds each
near-duplicate hit into the hit it repeats (`collapse`). `passages` and `answers` merge the chunks
of one section that sit next to each other into one readable span, grow a span too short to stand
alone by the neighbours that match the question or drop it (`thin`), fold near-duplicate spans the
same way, and read only the spans they answer with. `answers` then groups the spans by the section
they sit in, so `limit` counts sections, searches once more for the words of the question no
section holds (`probe`), adds the text around and between the passages that answers too (`fill`),
and writes each section out as one excerpt. `sources`
folds the same hits per document instead.

Two numbers steer that. `scan` is how deep the ranking goes and is what `hits` cuts to; `limit`
is how many answers the caller asked for and is what the last fold cuts to. Every pipeline scans
deeper than it answers: a folded near-duplicate frees its slot for the next result down, several
chunks go into one passage, and many into one document row.
"""

import asyncio
import functools
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast

import msgspec
from pydantic_graph import Graph, GraphBuilder, StepContext
from pydantic_graph.step import StepFunction

from haskie.collection.index import Hit
from haskie.paging import check_page_size
from haskie.search import aspects, probe, retrieval, scoring, section
from haskie.search.passage import Answer, Excerpt, HitRange, Passage, Sources
from haskie.settings import load_user_settings

# How deep any of these searches reads. A passage or a document row is folded from several chunks,
# so the scan goes deeper than the answer; this is where that stops.
MAX_SCAN = 200
CHUNK_SCAN = 2  # chunks scanned per chunk asked for: a folded near-duplicate frees its slot
PASSAGE_SCAN = 4  # chunks scanned per passage asked for: consecutive ones merge into one passage
DEFAULT_SECTIONS = 3  # hot sections per document: where in it the answer is, not an outline
MAX_SECTIONS = 20
DEFAULT_DOCUMENTS = 10  # a shortlist to choose from, not a page of passages
MAX_DOCUMENTS = 100  # a shortlist nobody reads past; `excerpts` is there for the passages
# Chunks scanned per document asked for. A document can hold many matching chunks, so the scan has
# to go deeper than the answer or the tail of the shortlist would be whichever documents happened
# to crowd the top with chunks.
DOCUMENT_SCAN = 20


class Search(msgspec.Struct):
    """One search, as every step of it sees it. The values that flow between the steps are their
    inputs and outputs; this is what they all read."""

    query: str  # the question alone: what full-text search and the word scores read
    framed: str  # the shared context, then the question: what the embedding and reranker read
    plan: retrieval.Plan
    limit: int  # answers the caller asked for; the last step cuts to it
    scan: int  # how deep the ranking goes; `hits` cuts to it
    candidates: int  # rows each collection returns, and the pool the reranker rescores
    # every question of the search as the steps after the ranking read them: each search of an
    # `answers` holds them all, and `query` is its own
    questions: list[probe.Question]
    sections: int = DEFAULT_SECTIONS  # `sources` only


# --- the trace --------------------------------------------------------------------


class StepTime(msgspec.Struct, frozen=True):
    """How long one step of a search took."""

    step: str  # the step's function name
    label: str  # what it does, as the web UI names it
    ms: float


# What each step is, for a person reading the breakdown, in the order the pipelines run them.
# `plan` is the work before the graph: settings, the query embedding, the model checks.
STEP_LABELS: dict[str, str] = {
    "plan": "Embed the query",
    "retrieve": "LanceDB retrieval",
    "merge": "Fuse rankings",
    "rerank": "Rerank",
    "hits": "Read hits",
    "collapse_hits": "Fold near-duplicates",
    "fill_thin": "Merge chunks and grow or drop short passages",
    "judge_thin": "Merge chunks and find short passages",
    "collapse_ranges": "Fold passages",
    "fold": "Take turns and fold passages",
    "group": "Group passages by section",
    "budget": "Cut to the answer's budget",
    "probe_gaps": "Search for missing words",
    "fill": "Fill in around passages",
    "quote": "Read excerpts",
    "read": "Read passages",
    "shortlist": "Fold into documents",
}


class ScoreStep(msgspec.Struct, frozen=True):
    """How one step set or changed the scores it answered with (`scoring.RULES`)."""

    step: str  # the step's function name
    label: str  # what it does, as the web UI names it
    rule: str


class Trace(msgspec.Struct):
    """What the searches of one request report beside their answer, each in the order its steps
    finished: how long each step took, and how each step that touched a score scored."""

    steps: list[StepTime] = []
    scoring: list[ScoreStep] = []


_PIPELINE_ORDER = {step: at for at, step in enumerate(STEP_LABELS)}

# One per request, started by the app (`app.bind_request_context`) and turned into its headers:
# the handlers answer what they always did, and an MCP call is traced the same way unseen.
_trace: ContextVar[Trace | None] = ContextVar("haskie_search_trace", default=None)


def start_trace() -> Trace:
    """A trace for the current request; every step timed from here on lands in it."""
    trace = Trace()
    _trace.set(trace)
    return trace


def _lineage(step: str, state: Search, read: Any, answered: Any) -> None:
    """Records how `step` scored what it answered with, when it set or changed a score. Once per
    rule: every question of an `answers` runs the same steps, mostly to the same rule."""
    trace, rule = _trace.get(), scoring.RULES.get(step)
    if trace is None or rule is None:
        return
    said = rule(state, read, answered)
    if said is not None and all((one.step, one.rule) != (step, said) for one in trace.scoring):
        trace.scoring.append(ScoreStep(step=step, label=STEP_LABELS.get(step, step), rule=said))
        # in pipeline order: the questions of an `answers` run at once, and finish interleaved
        trace.scoring.sort(key=lambda one: _PIPELINE_ORDER.get(one.step, len(_PIPELINE_ORDER)))


@contextmanager
def _timing(step: str) -> Iterator[None]:
    """Records how long the block took under `step`, failed or not."""
    started = time.perf_counter()
    try:
        yield
    finally:
        trace = _trace.get()
        if trace is not None:
            elapsed = (time.perf_counter() - started) * 1000
            trace.steps.append(StepTime(step=step, label=STEP_LABELS.get(step, step), ms=elapsed))


def server_timing(steps: list[StepTime]) -> str:
    """The `Server-Timing` header (W3C) of a trace: `retrieve;dur=41.2;desc="LanceDB retrieval"`."""
    return ", ".join(f'{one.step};dur={one.ms:.1f};desc="{one.label}"' for one in steps)


def _traced[F: StepFunction[Any, Any, Any, Any]](step: F, name: str) -> F:
    """`step`, recording under `name` how long it took and how it scored."""

    @functools.wraps(step)
    async def run(ctx: StepContext[Any, Any, Any]) -> Any:
        with _timing(name):
            answered = await step(ctx)
        _lineage(name, ctx.state, ctx.inputs, answered)
        return answered

    return cast(F, run)


# --- the steps --------------------------------------------------------------------


async def retrieve(ctx: StepContext[Search, None, None]) -> retrieval.Pool:
    """Every chosen collection, read at once, one row per chunk."""
    return await retrieval.fan_out(ctx.state.plan, ctx.state.query, ctx.state.candidates)


async def merge(ctx: StepContext[Search, None, retrieval.Pool]) -> retrieval.Pool:
    """The per-collection rankings, fused into one by rank."""
    return retrieval.merge(ctx.inputs, ctx.state.plan.settings.rrf_k, ctx.state.candidates)


async def rerank(ctx: StepContext[Search, None, retrieval.Pool]) -> retrieval.Pool:
    """The merged candidates, rescored by a cross-encoder that reads query and chunk together."""
    return await retrieval.rerank(ctx.inputs, ctx.state.framed, ctx.state.plan.settings)


async def hits(ctx: StepContext[Search, None, retrieval.Pool]) -> retrieval.Scanned:
    """The ranking, as far down as this search scans, as hits."""
    return retrieval.scan(ctx.inputs, ctx.state.scan)


async def collapse_hits(ctx: StepContext[Search, None, retrieval.Scanned]) -> list[Hit]:
    """The best hits, each near-duplicate folded into the hit it repeats."""
    plan = ctx.state.plan
    return await retrieval.collapse_hits(
        ctx.inputs, plan.embedding, plan.settings.mode, ctx.state.limit
    )


async def fill_thin(ctx: StepContext[Search, None, retrieval.Scanned]) -> retrieval.Ranged:
    """Consecutive chunks of one section merged into one range, and each range too short to stand
    alone grown by the neighbours that match the query, or dropped."""
    return await retrieval.fill_thin(ctx.inputs, ctx.state.plan, ctx.state.query, ctx.state.framed)


async def judge_thin(ctx: StepContext[Search, None, retrieval.Scanned]) -> retrieval.Ranged:
    """Consecutive chunks of one section merged into one range, and each range too short to stand
    alone judged by the neighbours that match the query, but not grown: the fill grows every
    passage of an excerpt once, short ones included."""
    return await retrieval.fill_thin(
        ctx.inputs, ctx.state.plan, ctx.state.query, ctx.state.framed, grows=False
    )


async def collapse_ranges(ctx: StepContext[Search, None, retrieval.Ranged]) -> list[HitRange]:
    """The `limit` best ranges, each near-duplicate folded into the range it repeats. A range too
    short to stand alone (`thin`) is no passage, so it takes no slot."""
    plan = ctx.state.plan
    return await retrieval.collapse_ranges(
        retrieval.standing(ctx.inputs), plan.embedding, plan.settings.mode, ctx.state.limit
    )


async def fold(ctx: StepContext[Search, None, list[retrieval.Ranged]]) -> list[HitRange]:
    """Each question's ranges folded into one list, none cut: an excerpt is a section, and the
    sections are what `limit` counts (`group`). One question's are its own, each near-duplicate
    folded into the range it repeats; several take turns (`retrieval.cover`). A range too short
    to stand alone is kept too, since the section it sits in may hold another."""
    state = ctx.state
    plan, ranged = state.plan, ctx.inputs
    if len(ranged) == 1:
        return await retrieval.collapse_ranges(ranged[0], plan.embedding, plan.settings.mode, None)
    labels = [one.asked for one in state.questions]
    depth = aspects.depth(len(labels), state.limit)
    return await retrieval.cover(
        ranged, labels, plan.embedding, plan.settings.mode, depth, state.scan
    )


async def read(ctx: StepContext[Search, None, list[HitRange]]) -> list[Passage]:
    """The kept ranges, read out of their documents."""
    return await retrieval.read(ctx.inputs)


async def group(ctx: StepContext[Search, None, list[HitRange]]) -> list[section.Group]:
    """The first `limit` sections the ranges fall in, each with every range of it."""
    state = ctx.state
    labels = [one.label for one in state.questions if one.label is not None] or None
    return await retrieval.sections(ctx.inputs, state.plan, state.limit, labels)


async def budget(ctx: StepContext[Search, None, list[section.Group]]) -> list[section.Group]:
    """The sections that fit the answer's budget, the last cut first. The probe's passage comes
    after this cut, and the fill spends what room is left."""
    return retrieval.budget(ctx.inputs, ctx.state.plan)


async def probe_gaps(
    ctx: StepContext[Search, None, list[section.Group]],
) -> list[section.Group]:
    """The sections, and the best passage a full-text search finds for the words of the
    questions none of them holds."""
    return await retrieval.probe_gaps(ctx.inputs, ctx.state.questions, ctx.state.plan)


async def fill(ctx: StepContext[Search, None, list[section.Group]]) -> list[section.Group]:
    """The sections with the text around and between their passages that answers too, while it
    fits the answer's budget."""
    return await retrieval.fill(ctx.inputs, ctx.state.questions, ctx.state.plan)


async def quote(ctx: StepContext[Search, None, list[section.Group]]) -> list[Excerpt]:
    """Each section read out of its document as one excerpt."""
    return await retrieval.read_excerpts(ctx.inputs)


async def shortlist(ctx: StepContext[Search, None, retrieval.Scanned]) -> Sources:
    """The same hits folded per document instead of per passage, with the collections to read
    them from."""
    return await retrieval.shortlist(
        ctx.inputs.hits, ctx.state.plan.names, ctx.state.limit, ctx.state.sections
    )


# --- the pipelines ----------------------------------------------------------------


def _chain[T](
    output: type[T], *steps: StepFunction[Search, None, Any, Any], input_type: Any = None
) -> Graph[Search, None, Any, T]:
    """One pipeline: the steps in order, each one's answer the next one's input.

    Linear on purpose. A step that sends a search back for more (a decision model asking for
    another pass over the collections, say) is an edge this helper does not draw, and gets its own
    builder here rather than a branch inside a step.
    """
    builder = GraphBuilder(state_type=Search, input_type=input_type, output_type=output)
    # `list[Any]`: a chain is heterogeneous — each step's output is the next one's input — and
    # the builder checks that pairing itself when it draws the edges
    # every step is a module function; the protocol they are typed by does not promise a name
    names: list[str] = [cast(Any, step).__name__ for step in steps]
    chain: list[Any] = [
        builder.step(_traced(step, name), node_id=name)
        for step, name in zip(steps, names, strict=True)
    ]
    pairs = zip(chain, chain[1:], strict=False)  # one edge short of the chain, by construction
    builder.add(
        builder.edge_from(builder.start_node).to(chain[0]),
        *(builder.edge_from(one).to(after) for one, after in pairs),
        builder.edge_from(chain[-1]).to(builder.end_node),
    )
    return builder.build()


RANKING = (retrieve, merge, rerank, hits)  # the search every answer shares

RANKED = _chain(retrieval.Ranged, *RANKING, judge_thin)  # one question's part of `answers`

CHUNKS = _chain(list[Hit], *RANKING, collapse_hits)
PASSAGES = _chain(list[Passage], *RANKING, fill_thin, collapse_ranges, read)
# each question's ranked ranges into excerpts
ANSWERED = _chain(
    list[Excerpt], fold, group, budget, probe_gaps, fill, quote, input_type=list[retrieval.Ranged]
)
SOURCES = _chain(Sources, *RANKING, shortlist)


# --- what a caller asks for -------------------------------------------------------


async def chunks(names: list[str], query: str, limit: int | None = None) -> list[Hit]:
    """The `limit` best matching chunks of `names`, best first.

    The merged `Hit.score` is an RRF score, or the cross-encoder's when a reranker is on; a single
    collection keeps its own scores, because there is nothing to compare them with.
    """
    state = await _search(names, query, limit, deeper=CHUNK_SCAN)
    return await CHUNKS.run(state=state) if state else []


async def passages(names: list[str], query: str, limit: int | None = None) -> list[Passage]:
    """The `limit` best passages of `names`, best first."""
    state = await _search(names, query, limit, deeper=PASSAGE_SCAN)
    return await PASSAGES.run(state=state) if state else []


async def answers(names: list[str], asked: aspects.Questions, limit: int | None = None) -> Answer:
    """The `limit` best sections of `names` for every question asked, as an agent quotes them,
    and what they leave out.

    Each question runs the shared ranking, all at once and each as deep as one search of `limit`
    would go (`RANKED`); then their ranges fold into one list (`fold`: several take turns at the
    slots, `aspects`) and become excerpts (`ANSWERED`). With several questions, each excerpt says
    which of them it answers.
    """
    limit = limit or (await load_user_settings()).search.limit
    aspects.depth(len(asked.questions), limit)  # a limit below the questions fails before searching
    states = await _searches(names, asked, limit, deeper=PASSAGE_SCAN)
    if states is None:
        return probe.report([], asked.asked())
    ranged = await asyncio.gather(*(RANKED.run(state=state) for state in states))
    found = await ANSWERED.run(state=states[0], inputs=list(ranged))
    return probe.report(found, states[0].questions)


async def sources(
    names: list[str], query: str, limit: int | None = None, sections: int | None = None
) -> Sources:
    """Which documents of `names` answer the query, and the smallest set of collections holding
    them."""
    documents = check_page_size(
        DEFAULT_DOCUMENTS if limit is None else limit, MAX_DOCUMENTS, "limit"
    )
    state = await _search(names, query, documents, deeper=DOCUMENT_SCAN, sections=sections)
    return await SOURCES.run(state=state) if state else Sources(documents=[], collections=[])


async def _search(
    names: list[str],
    query: str,
    limit: int | None,
    deeper: int,
    sections: int | None = None,
) -> Search | None:
    """One search, planned but not yet run, or None when nothing is left to search."""
    found = await _searches(names, aspects.Questions(questions=[query]), limit, deeper, sections)
    return found[0] if found else None


async def _searches(
    names: list[str],
    asked: aspects.Questions,
    limit: int | None,
    deeper: int,
    sections: int | None = None,
) -> list[Search] | None:
    """One search per query over the same collections, planned but not yet run, or None when
    nothing is left to search.

    The only place a `Search` is built, so every bound a caller asked for is resolved here and
    the steps read numbers rather than compute them. `deeper` is how many chunks the answer this
    pipeline builds is folded from, which is what turns the caller's limit into the scan depth.
    """
    limit = limit or (await load_user_settings()).search.limit
    with _timing("plan"):
        plans = await retrieval.plan(names, asked.framed)
    if plans is None:
        return None
    # a pipeline that folds scans deeper than it answers, and that is what `MAX_SCAN` bounds
    scan = max(limit, min(limit * deeper, MAX_SCAN))
    candidates = max(plans[0].settings.candidates, scan)
    sections = check_page_size(
        DEFAULT_SECTIONS if sections is None else sections, MAX_SECTIONS, "sections"
    )
    questions = asked.asked([where.vector for where in plans])
    return [
        Search(
            query=query,
            framed=framed,
            plan=where,
            limit=limit,
            scan=scan,
            candidates=candidates,
            sections=sections,
            questions=questions,
        )
        for query, framed, where in zip(asked.questions, asked.framed, plans, strict=True)
    ]
