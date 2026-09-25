"""The shape of a search: what runs, in what order, for each of the four answers.

Read this file for what a search does; read `retrieval.py` for what each step does with the IO it
needs, and `passage.py` for the pure folds under them. Nothing here does work — every step is one
line handing the state to one function and passing its answer on — so a stage can be added,
dropped or reordered by reading this file alone.

Four pipelines over one set of steps:

    chunks     retrieve -> merge -> rerank -> hits -> collapse_hits
    passages   retrieve -> merge -> rerank -> hits -> collapse_ranges -> widen
    excerpts   retrieve -> merge -> rerank -> hits -> collapse_ranges -> widen
    sources    retrieve -> merge -> rerank -> hits -> shortlist

The first four steps are the search every answer shares; what follows is the fold that answer is
made of, and it is a step rather than something every search pays for. `chunks` folds each
near-duplicate hit into the hit it repeats (`collapse`). `passages` and `excerpts` merge the chunks
of one document that sit next to each other into one readable span, fold near-duplicate spans the
same way, and read only the spans they answer with. `sources` folds the same hits per document
instead.

Two numbers steer that. `scan` is how deep the ranking goes and is what `hits` cuts to; `limit`
is how many answers the caller asked for and is what the last fold cuts to. Every pipeline scans
deeper than it answers: a folded near-duplicate frees its slot for the next result down, several
chunks go into one passage, and many into one document row.
"""

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
from haskie.search import retrieval
from haskie.search.passage import Excerpt, HitRange, Passage, Sources
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

    query: str
    plan: retrieval.Plan
    limit: int  # answers the caller asked for; the last step cuts to it
    scan: int  # how deep the ranking goes; `hits` cuts to it
    candidates: int  # rows each collection returns, and the pool the reranker rescores
    shape: type[Passage] = Passage  # which passage type `widen` builds
    sections: int = DEFAULT_SECTIONS  # `sources` only


# --- the trace --------------------------------------------------------------------


class StepTime(msgspec.Struct, frozen=True):
    """How long one step of a search took."""

    step: str  # the step's function name
    label: str  # what it does, as the web UI names it
    ms: float


# What each step is, for a person reading the breakdown. `plan` is the work before the graph:
# settings, the query embedding, the model checks.
STEP_LABELS: dict[str, str] = {
    "plan": "Embed the query",
    "retrieve": "LanceDB retrieval",
    "merge": "Fuse rankings",
    "rerank": "Rerank",
    "hits": "Read hits",
    "collapse_hits": "Fold near-duplicates",
    "collapse_ranges": "Merge and fold passages",
    "widen": "Read and widen passages",
    "shortlist": "Fold into documents",
}

# The steps of the searches one request runs, in the order they finished. A list per request,
# started by the app (`app.bind_request_context`) and turned into its `Server-Timing` header: the
# handlers answer what they always did, and an MCP call is timed the same way without seeing it.
_trace: ContextVar[list[StepTime] | None] = ContextVar("haskie_search_trace", default=None)


def start_trace() -> list[StepTime]:
    """A trace for the current request; every step timed from here on lands in it."""
    steps: list[StepTime] = []
    _trace.set(steps)
    return steps


@contextmanager
def _timing(step: str) -> Iterator[None]:
    """Records how long the block took under `step`, failed or not."""
    started = time.perf_counter()
    try:
        yield
    finally:
        steps = _trace.get()
        if steps is not None:
            elapsed = (time.perf_counter() - started) * 1000
            steps.append(StepTime(step=step, label=STEP_LABELS.get(step, step), ms=elapsed))


def server_timing(steps: list[StepTime]) -> str:
    """The `Server-Timing` header (W3C) of a trace: `retrieve;dur=41.2;desc="LanceDB retrieval"`."""
    return ", ".join(f'{one.step};dur={one.ms:.1f};desc="{one.label}"' for one in steps)


def _timed[F: StepFunction[Any, Any, Any, Any]](step: F, name: str) -> F:
    """`step`, recording how long it took under `name`."""

    @functools.wraps(step)
    async def run(ctx: StepContext[Any, Any, Any]) -> Any:
        with _timing(name):
            return await step(ctx)

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
    return await retrieval.rerank(ctx.inputs, ctx.state.query, ctx.state.plan.settings)


async def hits(ctx: StepContext[Search, None, retrieval.Pool]) -> retrieval.Scanned:
    """The ranking, as far down as this search scans, as hits."""
    return retrieval.scan(ctx.inputs, ctx.state.scan)


async def collapse_hits(ctx: StepContext[Search, None, retrieval.Scanned]) -> list[Hit]:
    """The best hits, each near-duplicate folded into the hit it repeats."""
    plan = ctx.state.plan
    return await retrieval.collapse_hits(
        ctx.inputs, plan.embedding, plan.settings.mode, ctx.state.limit
    )


async def collapse_ranges(
    ctx: StepContext[Search, None, retrieval.Scanned],
) -> list[HitRange]:
    """Consecutive chunks of one document merged into one range, and each near-duplicate range
    folded into the range it repeats."""
    plan = ctx.state.plan
    return await retrieval.collapse_ranges(
        ctx.inputs, plan.embedding, plan.settings.mode, ctx.state.limit
    )


async def widen(ctx: StepContext[Search, None, list[HitRange]]) -> list[Passage]:
    """The kept ranges, widened to where a reader stops.

    `Search.shape` decides the type: an excerpt is the whole passage today, and cutting the parts
    of it that do not answer the query is a later step that would go here.
    """
    return await retrieval.widen(ctx.inputs, ctx.state.shape)


async def shortlist(ctx: StepContext[Search, None, retrieval.Scanned]) -> Sources:
    """The same hits folded per document instead of per passage, with the collections to read
    them from."""
    return await retrieval.shortlist(
        ctx.inputs.hits, ctx.state.plan.names, ctx.state.limit, ctx.state.sections
    )


# --- the pipelines ----------------------------------------------------------------


def _chain[T](
    output: type[T], *steps: StepFunction[Search, None, Any, Any]
) -> Graph[Search, None, None, T]:
    """One pipeline: the steps in order, each one's answer the next one's input.

    Linear on purpose. A step that sends a search back for more (a decision model asking for
    another pass over the collections, say) is an edge this helper does not draw, and gets its own
    builder here rather than a branch inside a step.
    """
    builder = GraphBuilder(state_type=Search, output_type=output)
    # `list[Any]`: a chain is heterogeneous — each step's output is the next one's input — and
    # the builder checks that pairing itself when it draws the edges
    # every step is a module function; the protocol they are typed by does not promise a name
    names: list[str] = [cast(Any, step).__name__ for step in steps]
    chain: list[Any] = [
        builder.step(_timed(step, name), node_id=name)
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

CHUNKS = _chain(list[Hit], *RANKING, collapse_hits)
PASSAGES = _chain(list[Passage], *RANKING, collapse_ranges, widen)
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


async def excerpts(names: list[str], query: str, limit: int | None = None) -> list[Excerpt]:
    """The `limit` best passages of `names` as an agent quotes them, best first."""
    state = await _search(names, query, limit, deeper=PASSAGE_SCAN, shape=Excerpt)
    if state is None:
        return []
    # the graph builds whatever `shape` says, and this one said `Excerpt`
    return cast(list[Excerpt], await PASSAGES.run(state=state))


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
    shape: type[Passage] = Passage,
    sections: int | None = None,
) -> Search | None:
    """One search, planned but not yet run, or None when nothing is left to search.

    The only place a `Search` is built, so every bound a caller asked for is resolved here and
    the steps read numbers rather than compute them. `deeper` is how many chunks the answer this
    pipeline builds is folded from, which is what turns the caller's limit into the scan depth.
    """
    limit = limit or (await load_user_settings()).search.limit
    with _timing("plan"):
        where = await retrieval.plan(names, query)
    if where is None:
        return None
    # a pipeline that folds scans deeper than it answers, and that is what `MAX_SCAN` bounds
    scan = max(limit, min(limit * deeper, MAX_SCAN))
    return Search(
        query=query,
        plan=where,
        limit=limit,
        scan=scan,
        candidates=max(where.settings.candidates, scan),
        shape=shape,
        sections=check_page_size(
            DEFAULT_SECTIONS if sections is None else sections, MAX_SECTIONS, "sections"
        ),
    )
