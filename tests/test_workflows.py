"""Durable execution: the DBOS runtime, the three pipelines and the jobs read model.

Every test here takes the `dbos` fixture, which launches DBOS on the test home's SQLite file and
destroys it afterwards. Concurrency is driven by `threading.Event`, never by sleeping: a fake
pipeline step signals when it has been entered and blocks until the test releases it, so the
"mid-flight" moment is a fact rather than a guess.

Threading events rather than anyio ones, because two event loops are involved: the test body runs
on the loop `pytest.mark.anyio` gives it, while the steps it gates run on DBOS's background loop.
Both sides wait on them through `conftest.wait_event`, which hands the blocking wait to a worker
thread so neither loop is ever blocked.

Three workflows carry a document: `import_document` (`imp:`) converts it once and pre-warms the
embedding cache, `ensure_embedding` (`emb:`) computes one cached embedding per distinct
`embed_cache.Params`, and `index_collection_document` (`idx-col:`) writes cached rows into one
collection's table. The cache is what makes the second collection cheap, so the tests below assert
the mechanism - how often the embed work ran, which workflow ran it - and not only the end state.
"""

import os
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import anyio
import anyio.to_thread
import pytest
from conftest import (
    MD,
    WAIT,
    WAITING_STATUS,
    attach_document,
    audit_lines,
    await_terminal,
    counted_list_workflows,
    delete_collection,
    delete_document,
    document_names,
    events,
    forget_settings,
    import_document,
    import_row,
    maintenance_state,
    restart_dbos,
    text_pdf,
    until,
    wait_event,
    wait_for,
)
from dbos import DBOS, SetWorkflowID
from dbos._error import DBOSAwaitedWorkflowCancelledError
from dbos._registrations import get_dbos_func_name

from haskie import (
    audit,
    chunk,
    convert,
    cpu,
    db,
    dbos_names,
    document,
    embed_cache,
    home,
    jobs,
    maintenance,
    models,
    paging,
    pipeline,
    settings,
    workflows,
)
from haskie.collection import Collection
from haskie.document import Document
from haskie.errors import (
    Conflict,
    InvalidInput,
    NotFound,
    PermanentError,
)
from haskie.pipeline import Batch
from haskie.settings import (
    ChunkSettings,
    CollectionSettings,
    EmbeddingModel,
    PipelineSettings,
    RetentionSettings,
    SearchOverrides,
    UserSettings,
    load_user_settings,
    save_user_settings,
)

pytestmark = pytest.mark.anyio

BLOCKED_WAIT = 2.0  # how long a step that must not run is given to prove it by not running


async def _add_member(collection: Collection, doc: str) -> None:
    """A membership whose document never finished importing - what a re-import of an attached
    document leaves behind. `Collection.add` refuses to create one, so the row is written here."""
    now = time.time()
    async with db.connect() as conn:
        await conn.execute(
            "insert into collection_documents (collection, document, added_at, updated_at) "
            "values (?, ?, ?, ?)",
            (collection.name, doc, now, now),
        )


def _embed_id(workflow_id: str, doc: str) -> str:
    """The `emb:` child one `imp:` or `idx-col:` workflow asks for (see `_ensure_embedding`)."""
    return f"{workflows.EMBED_PREFIX}:{doc}:{workflows.run_id(workflow_id)}"


async def _steps(workflow_id: str) -> list[str]:
    """The step names one workflow recorded, in order. DBOS writes the row when a step finishes,
    so this is what really ran rather than what the body would have run."""
    async with db.connect() as conn:
        rows = await conn.execute_fetchall(
            "select function_name from operation_outputs where workflow_uuid = ? "
            "order by function_id",
            (workflow_id,),
        )
    return [name for (name,) in rows]


async def _workflow_ids(name: str) -> list[str]:
    found = await DBOS.list_workflows_async(name=name, load_input=False, load_output=False)
    return [s.workflow_id for s in found]


GROUPED = PipelineSettings().index_group_parts  # default: a whole document in one index write
NEVER = 3600  # an idle period no test waits out: maintenance stays DELAYED unless asked for


STAGES = 3  # convert, embed, index: the stages a budget is shared out over


async def _use(
    dbos,
    *,
    workers: int | None = None,
    batch_pages: int,
    cpu_budget: int | None = None,
    converting_weight: int = 1,
    embedding_weight: int = 1,
    indexing_weight: int = 1,
    document_parallelism: int = PipelineSettings().document_parallelism,
    index_group_parts: int = GROUPED,
    maintenance_docs: int = PipelineSettings().maintenance_docs,
    maintenance_idle_seconds: int = PipelineSettings().maintenance_idle_seconds,
) -> None:
    """Store and apply pipeline settings, so the queues really carry the given limits.

    `workers=n` is shorthand for "a cap of n on every stage queue", which is what a test that only
    wants room for `n` tasks per stage means: a budget of `3 * n` shared equally by the three
    stages. A test that pins one stage against another, or the budget against the stages, names
    `cpu_budget` and the weights instead. `index_group_parts=1` gives the per-part index layout:
    one LanceDB commit, and one index task, per micro-batch. `document_parallelism=1` puts a whole
    stage back into a single child, which is what a test that reads one step log in order needs.
    `maintenance_docs=1` makes every document ask for maintenance with no delay; the defaults
    leave it DELAYED for longer than any test runs."""
    budget = cpu_budget or (workers * STAGES if workers else PipelineSettings().cpu_budget)
    indexing = PipelineSettings(
        cpu_budget=budget,
        converting_weight=converting_weight,
        embedding_weight=embedding_weight,
        indexing_weight=indexing_weight,
        document_parallelism=document_parallelism,
        batch_pages=batch_pages,
        index_group_parts=index_group_parts,
        maintenance_docs=maintenance_docs,
        maintenance_idle_seconds=maintenance_idle_seconds,
    )
    await dbos.apply_settings(await save_user_settings(UserSettings(pipeline=indexing)))


async def _statuses(workflow_ids: list[str]) -> list[str]:
    found = {
        s.workflow_id: s.status for s in await DBOS.list_workflows_async(workflow_ids=workflow_ids)
    }
    return [found[i] for i in workflow_ids]


def _all_cancelled(workflow_ids: list[str]):
    """A condition for `until`: every one of these workflows has been marked CANCELLED. The mark
    is all a cancel writes, so this says nothing about the steps still running under it."""

    async def cancelled() -> bool:
        return await _statuses(workflow_ids) == ["CANCELLED"] * len(workflow_ids)

    return cancelled


async def _drain(timeout: float = WAIT) -> None:
    """Wait until no workflow is active any more. A step that outlives `DBOS.destroy()` keeps a
    non-daemon thread alive, so a test that released a blocked step waits for it here rather than
    leaving it to the teardown's grace period."""
    deadline = time.monotonic() + timeout
    while await DBOS.list_workflows_async(
        status=dbos_names.ACTIVE_STATUS, load_input=False, load_output=False
    ):
        assert time.monotonic() < deadline, "workflows still running at the end of the test"
        await anyio.sleep(workflows.TASK_POLL)


async def _drain_maintenance(timeout: float = WAIT) -> None:
    """Like `_drain`, but a debounced workflow still waiting out its period (DELAYED) counts as
    unfinished. Only for tests that asked for maintenance with no delay; with the default idle
    period this would wait a minute."""
    deadline = time.monotonic() + timeout
    while await DBOS.list_workflows_async(
        status=WAITING_STATUS, load_input=False, load_output=False
    ):
        assert time.monotonic() < deadline, "maintenance still pending at the end of the test"
        await anyio.sleep(workflows.TASK_POLL)


async def _cached_rows(doc: str) -> int:
    """Chunks the embedding cache holds for a document: the row count a collection's table must
    match once the index stage has read the cache file into it."""
    return sum(entry.rows for entry in await embed_cache.entries(doc))


async def _fragments(collection: Collection) -> int:
    """Data fragments of the collection's table: LanceDB writes one per commit that carries rows."""
    table = await (await collection.index())._existing()
    if table is None:
        return 0
    # lancedb annotates stats() as a dataclass but returns plain dicts
    return (await table.stats())["fragment_stats"]["num_fragments"]  # ty: ignore[not-subscriptable]


async def _rows_of(collection: Collection, doc: str) -> int:
    """Rows one document has in a collection's table right now: what a detach or a delete has to
    leave none of, whichever way the write in flight was ordered against it."""
    table = await (await collection.index())._existing()
    return 0 if table is None else await table.count_rows(f"doc = '{doc}'")


def _batch_of(args: tuple) -> Batch | None:
    """The micro-batch one pipeline call was given. The three stage steps take it in three
    different positions (`convert_batch`, `embed_batch`, `index_batch`), so it is found by type
    rather than by index: one wrapper then fits all three."""
    for value in args:
        if isinstance(value, Batch):
            return value
    return None


def _doc_of(args: tuple) -> str:
    """The document one pipeline call was given, by the same reasoning as `_batch_of`."""
    for value in args:
        if isinstance(value, Document):
            return value.name
    return ""


class Gate:
    """A pipeline step the test can hold open: `entered` fires on the first blocked call, and that
    call returns only once `release` is set."""

    def __init__(
        self, seq: int | None = None, doc: str | None = None, *, holds_cpu: bool = False
    ) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls: list[int] = []
        self.seq = seq  # block only this batch; None = every batch
        self.doc = doc  # block only this document; None = every document
        self.holds_cpu = holds_cpu  # hold one slot of the CPU budget while blocked (see `wrap`)

    def _blocks(self, doc: str, batch: Batch | None) -> bool:
        by_seq = self.seq is None or (batch is not None and batch.seq == self.seq)
        return by_seq and (self.doc is None or doc == self.doc)

    def wrap(self, real):
        async def blocking(*args):
            batch = _batch_of(args)
            self.calls.append(-1 if batch is None else batch.seq)
            if self._blocks(_doc_of(args), batch):
                if self.holds_cpu:
                    await cpu.on_cpu(self._hold)
                else:
                    self.entered.set()
                    assert await wait_event(self.release), "the test never released the step"
            return await real(*args)

        return blocking

    def _hold(self) -> None:
        """The same wait, taken inside `cpu.on_cpu`, so it occupies one slot of the CPU budget for
        as long as it lasts.

        A step takes its slot inside `cpu.on_cpu`, around its CPU work alone, so a gate at the
        step's entry holds no slot at all. A test about the budget rather than about a queue asks
        for one. Sync, and run in the worker thread `cpu.on_cpu` gave it."""
        self.entered.set()
        assert self.release.wait(timeout=WAIT), "the test never released the step"


class EmbedSpy:
    """Counts the embed work per cache id: one entry per micro-batch really chunked and embedded.

    The cache is the point of the refactor, so its tests assert how often this ran rather than how
    many rows came out: a second collection with the same effective `Params` must add nothing."""

    def __init__(self, real) -> None:
        self.real = real
        self.calls: Counter[str] = Counter()

    async def __call__(self, doc, batch, cache_id, chunking, embedding):
        self.calls[cache_id] += 1
        return await self.real(doc, batch, cache_id, chunking, embedding)


def _spy_embed(monkeypatch: pytest.MonkeyPatch) -> EmbedSpy:
    spy = EmbedSpy(pipeline.embed_batch)
    monkeypatch.setattr(pipeline, "embed_batch", spy)
    return spy


# --- import and attach -------------------------------------------------------------


async def test_import_converts_and_prewarms_the_cache(dbos, tmp_path: Path) -> None:
    """An import is collection-independent: it converts the document once and computes the
    embedding the user's default chunk settings call for, so attaching a default collection later
    costs a cache read."""
    doc = await import_document(dbos, "guide.md", MD, tmp_path)

    assert (doc.status, doc.error, doc.suffix) == ("imported", None, ".md")
    assert doc.markdown.read_text() == MD
    assert sorted(p.name for p in doc.parts_dir.glob("*.md")) == ["000000.md"], "parts stay"
    assert await Collection.names() == [], "no collection was touched"
    defaults = (await load_user_settings()).conversion.chunking
    (entry,) = await embed_cache.entries(doc.name)
    assert (entry.chunk_size, entry.chunker, entry.model) == (
        defaults.chunk_size,
        defaults.chunker,
        embed_cache.NO_MODEL,
    )
    assert (entry.parser, entry.skip_ocr_pages) == (doc.parser, doc.skip_ocr_pages)
    assert embed_cache.file_path(doc.name, entry.id).is_file() and entry.rows > 0
    assert not embed_cache.scratch_dir(doc.name, entry.id).exists(), "the scratch rows are merged"
    assert await document.collections_of(doc.name) == []


async def test_attach_indexes_the_document_into_the_collection(dbos, tmp_path: Path) -> None:
    """Attaching writes the cached rows into that collection's table and moves the membership,
    never the document's own status."""
    collection = await Collection.create("Notes & Stuff")
    assert collection.name == "Notes-Stuff"
    await collection.set_settings(CollectionSettings(chunk_size=40, chunk_overlap=0))
    assert (await (await Collection.get("Notes-Stuff")).settings()).chunk_size == 40
    doc = await import_document(dbos, "guide.md", MD, tmp_path)
    previewed, preview = await document.ensure_preview(doc.name)
    assert preview.kind == "text"
    assert previewed.status == "imported", "building a preview is not a lifecycle step"

    await attach_document(dbos, "Notes-Stuff", doc.name)

    member = await collection.member(doc.name)
    assert (member.status, member.error) == ("indexed", None)
    assert (await document.get(doc.name)).status == "imported", "the document's status is its own"
    assert await document.collections_of(doc.name) == ["Notes-Stuff"]
    hits = await collection.search("lancedb", SearchOverrides(limit=5))
    assert hits and hits[0].heading == "Alpha" and hits[0].collection == "Notes-Stuff"
    assert (hits[0].line_start, hits[0].line_end, hits[0].parents) == (5, 7, ["Title"])
    assert hits[0].markdown_path == doc.relative(doc.markdown), "home-relative"
    assert hits[0].markdown_file == str(home.HOME / hits[0].markdown_path), "absolute"
    assert (home.HOME / hits[0].markdown_path).read_text() == MD
    assert MD[hits[0].char_start : hits[0].char_end] == hits[0].text
    assert (hits[0].page_start, hits[0].page_end) == (None, None), "no pages for markdown"
    assert (hits[0].header, hits[0].location) == ("Title > Alpha", "guide.md L5-7")


async def test_pipeline_cuts_a_pdf_into_micro_batches(dbos, tmp_path: Path) -> None:
    """One import job per document, one embedding job per cache id, one index job per collection,
    each with a task per micro-batch. The index is deduplicated while it runs; a second import is
    refused instead, because the document's status has left `queued` by then."""
    await _use(dbos, workers=2, batch_pages=10, index_group_parts=1)
    collection = await Collection.create("q")
    await collection.set_settings(CollectionSettings(chunk_size=60, chunk_overlap=0))
    pdf = await import_row(
        "book.pdf", text_pdf([f"Chapter {i} word{i}" for i in range(1, 26)]), tmp_path
    )

    first = await dbos.start_import(pdf.name)
    assert await wait_for(first) == "imported"
    with pytest.raises(Conflict, match="document is imported; only a queued, failed or cancelled"):
        await dbos.start_import(pdf.name)
    indexing = await dbos.attach("q", pdf.name)
    assert await dbos.start_index_collection_document("q", pdf.name) == indexing, "deduplicated"
    assert await wait_for(indexing) == "indexed"

    converting = await jobs.list_tasks(first)
    assert [(t.stage, t.seq, t.page_start, t.page_end, t.status) for t in converting] == [
        ("convert", 0, 0, 10, "SUCCESS"), ("convert", 1, 10, 20, "SUCCESS"),
        ("convert", 2, 20, 25, "SUCCESS"),
    ]  # fmt: skip
    embedding = _embed_id(first, pdf.name)
    assert [(t.stage, t.seq, t.status) for t in await jobs.list_tasks(embedding)] == [
        ("embed", 0, "SUCCESS"), ("embed", 1, "SUCCESS"), ("embed", 2, "SUCCESS"),
    ]  # fmt: skip
    assert [(t.stage, t.seq, t.status) for t in await jobs.list_tasks(indexing)] == [
        ("index", 0, "SUCCESS"), ("index", 1, "SUCCESS"), ("index", 2, "SUCCESS"),
    ]  # fmt: skip
    hit = (await Collection("q").search("word25", SearchOverrides(limit=1)))[0]
    assert (hit.page_start, hit.part) == (25, 2)
    full = pdf.markdown.read_text()
    assert full[hit.char_start : hit.char_end] == hit.text
    assert await dbos.start_index_collection_document("q", pdf.name) != indexing, "dedup ends"
    await _drain()


async def test_start_import_and_attach_validate_before_they_enqueue(dbos, tmp_path: Path) -> None:
    await Collection.create("v")
    with pytest.raises(NotFound, match="document not found: a.md"):
        await dbos.start_import("a.md")
    with pytest.raises(NotFound, match="collection not found: ghost"):
        await dbos.start_index_collection_document("ghost", "a.md")
    queued = await import_row("a.md", into=tmp_path)
    with pytest.raises(NotFound, match="document not in collection v: a.md"):
        await dbos.start_index_collection_document("v", queued.name)
    with pytest.raises(Conflict, match="a.md"):  # `Collection.add` takes only an imported document
        await dbos.attach("v", queued.name)
    with pytest.raises(NotFound, match="collection not found: ghost"):
        await dbos.attach("ghost", queued.name)

    assert (await jobs.list_jobs()).items == [], "a rejected request leaves no job behind"


async def test_documents_of_one_collection_index_without_conflict(dbos, tmp_path: Path) -> None:
    """Index tasks share one partition per collection, so concurrent documents never collide."""
    await _use(dbos, workers=8, batch_pages=2)
    await Collection.create("serial")
    names = [
        (
            await import_document(
                dbos, f"d{i}.pdf", text_pdf([f"D{i}P{p}" for p in range(6)]), tmp_path
            )
        ).name
        for i in range(4)
    ]

    ids = [await dbos.attach("serial", name) for name in names]

    assert [await wait_for(i) for i in ids] == ["indexed"] * len(ids)
    counts = await Collection("serial").counts()
    assert (counts.total, counts.indexed) == (4, 4)
    assert {h.doc for h in await Collection("serial").search("D3P5", SearchOverrides(limit=1))} == {
        "d3.pdf"
    }


async def test_workflow_ids_name_their_kind_and_their_names(dbos, tmp_path: Path) -> None:
    """Every id starts with a prefix naming the kind and the names it belongs to, so one prefix
    query finds a whole job (see the `workflows` module docstring)."""
    await Collection.create("c")
    doc = await import_row("a.md", into=tmp_path)

    import_id = await dbos.start_import(doc.name)
    assert import_id.startswith(f"{workflows.IMPORT_PREFIX}:a.md:"), import_id
    assert await wait_for(import_id) == "imported"
    index_id = await dbos.attach("c", doc.name)
    assert index_id.startswith(f"{workflows.COLLECTION_DOCUMENT_PREFIX}:c:a.md:"), index_id
    assert await wait_for(index_id) == "indexed"

    assert sorted(await _workflow_ids(dbos_names.EMBED_WORKFLOW)) == sorted(
        {_embed_id(import_id, doc.name), _embed_id(index_id, doc.name)}
    ), "each parent derives its embedding child's id from its own run"
    delete_id = await dbos.start_delete_document(doc.name)
    assert delete_id.startswith(f"{workflows.DELETE_DOCUMENT_PREFIX}:a.md:"), delete_id
    await wait_for(delete_id)


# --- stage caps and parallelism ----------------------------------------------------


def _budget(
    cpu_budget: int,
    convert: int = 1,
    embed: int = 1,
    index: int = 1,
    document_parallelism: int = PipelineSettings().document_parallelism,
) -> PipelineSettings:
    """Pipeline settings that only say how the CPU budget is shared out."""
    return PipelineSettings(
        cpu_budget=cpu_budget,
        converting_weight=convert,
        embedding_weight=embed,
        indexing_weight=index,
        document_parallelism=document_parallelism,
    )


@pytest.mark.parametrize(
    ("name", "indexing", "expected"),
    [
        (
            "the default weights give converting and embedding three slots of seven each",
            PipelineSettings(cpu_budget=7),
            {"convert": 3, "embed": 3, "index": 1},
        ),
        (
            "equal weights split a budget that divides evenly",
            _budget(9),
            {"convert": 3, "embed": 3, "index": 3},
        ),
        (
            "twice the weight is twice the slots",
            _budget(8, convert=2),
            {"convert": 4, "embed": 2, "index": 2},
        ),
        (
            "the floor of one slot per stage outweighs a budget of two",
            _budget(2),
            {"convert": 1, "embed": 1, "index": 1},
        ),
        (
            "and a budget of one, which the semaphore then holds to one task at a time",
            PipelineSettings(cpu_budget=1),
            {"convert": 1, "embed": 1, "index": 1},
        ),
    ],
)
def test_stage_caps_share_the_cpu_budget_by_weight(
    name: str, indexing: PipelineSettings, expected: dict[workflows.Stage, int]
) -> None:
    caps = workflows.stage_caps(indexing)

    assert caps == expected, name
    assert sum(caps.values()) == max(indexing.cpu_budget, len(caps)), "the budget, or the floors"


def test_stage_caps_hand_the_remainder_to_one_stage() -> None:
    """Ten slots over three equal weights: three each, and the spare one goes to a single stage
    rather than to nobody."""
    caps = workflows.stage_caps(_budget(10))

    assert sum(caps.values()) == 10
    assert sorted(caps.values()) == [3, 3, 4]


@pytest.mark.parametrize(
    ("name", "indexing", "kind", "expected"),
    [
        (
            "0 means as many slices as the stage's share of the budget",
            _budget(10, convert=3),
            "convert",
            6,
        ),
        ("each stage resolves against its own share", _budget(10, convert=3), "embed", 2),
        ("a lower cap is kept", _budget(10, convert=3, document_parallelism=2), "convert", 2),
        (
            "a higher cap is cut to the stage's share",
            _budget(10, convert=3, document_parallelism=9),
            "embed",
            2,
        ),
        ("equal to the stage's share", _budget(12, document_parallelism=4), "convert", 4),
        ("the floor of one slot is a share too", _budget(2, document_parallelism=9), "index", 1),
    ],
)
def test_document_parallelism_resolves_against_the_stage_share(
    name: str, indexing: PipelineSettings, kind: workflows.Stage, expected: int
) -> None:
    """A slice occupies one slot of its own stage's queue, so asking for more than that stage's
    share of the CPU budget only queues them."""
    assert workflows.resolve_parallelism(indexing, kind) == expected, name


class Overlap:
    """Counts how many batches are inside one wrapped pipeline step at the same time. `wait_for`
    batches hold each other there, so an overlap is observed rather than timed.

    `wait_for=1` never blocks, which is how a stage whose queue admits one at a time is counted.

    The wrapped steps are coroutines sharing DBOS's event loop, so the wait goes through
    `_wait_event`: a step that blocked that loop would keep every other step off it, and the
    count it waits for could never be reached. The counter's lock stays a threading lock, because
    nothing awaits while it is held."""

    def __init__(self, real, wait_for: int) -> None:
        self.real = real
        self.wait_for = wait_for
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.order: list[int] = []
        self.reached = threading.Event()

    def wrap(self, real):
        """A second step counted on the same counter, so what is observed is the overlap between
        two stages rather than inside one."""

        async def counted(*args):
            return await self._count(real, *args)

        return counted

    async def __call__(self, *args):
        return await self._count(self.real, *args)

    async def _count(self, real, *args):
        batch = _batch_of(args)
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.order.append(-1 if batch is None else batch.seq)
            if self.active >= self.wait_for:
                self.reached.set()
        await wait_event(self.reached)  # already set once `wait_for` callers are inside
        try:
            return await real(*args)
        finally:
            with self.lock:
                self.active -= 1


class CpuOverlap:
    """Counts how many pieces of CPU work are inside a slot of the budget at the same time.

    Wraps the sync work `cpu.on_cpu` hands to a worker thread, which is exactly the region one
    slot is held for. A pipeline step wrapped instead would also count the steps that are waiting
    for a slot, which is what the budget is there to make them do."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0

    def wrap(self, real):
        def counted(*args, **kwargs):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
                self.calls += 1
            try:
                return real(*args, **kwargs)
            finally:
                with self.lock:
                    self.active -= 1

        return counted


async def test_documents_run_in_parallel_up_to_workers(
    dbos, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Documents overlap each other, up to the stage's workers. Pinned to one slice per document,
    so the overlap that is observed is between documents and not inside one of them."""
    workers = 3
    await _use(dbos, workers=workers, batch_pages=1, document_parallelism=1)
    await Collection.create("par")
    docs = [
        await import_row(f"p{i}.pdf", text_pdf([f"P{i}A tokq{i}", f"P{i}B"]), tmp_path)
        for i in range(3)
    ]
    overlap = Overlap(pipeline.convert_batch, wait_for=2)
    monkeypatch.setattr(pipeline, "convert_batch", overlap)

    ids = [await dbos.start_import(d.name) for d in docs]
    assert [await wait_for(job_id) for job_id in ids] == ["imported"] * len(ids)

    assert overlap.reached.is_set(), "documents never overlapped"
    assert 2 <= overlap.peak <= workers, f"peak {overlap.peak} outside 2..{workers}"
    for d in docs:
        await attach_document(dbos, "par", d.name)
    found = [(await Collection("par").search(f"tokq{i}"))[0].page_start for i in range(3)]
    assert found == [1, 1, 1]


async def test_batches_of_one_document_run_in_parallel_up_to_workers(
    dbos, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One document is cut into as many convert slices as the convert queue has workers, so
    re-importing a single large file uses every worker instead of one. The batches hold each other
    inside the step, so the overlap is observed rather than timed."""
    workers = 3
    await _use(dbos, workers=workers, batch_pages=1)
    await Collection.create("wide")
    doc = await import_row("s.pdf", text_pdf([f"S{i} toks{i}" for i in range(6)]), tmp_path)
    overlap = Overlap(pipeline.convert_batch, wait_for=workers)
    monkeypatch.setattr(pipeline, "convert_batch", overlap)

    job_id = await dbos.start_import(doc.name)
    assert await wait_for(job_id) == "imported"

    assert overlap.reached.is_set(), "the batches of one document never overlapped"
    assert overlap.peak == workers, f"peak {overlap.peak}, expected {workers}"
    listed = await DBOS.list_workflows_async(
        workflow_id_prefix=f"{job_id}:convert", load_input=False
    )
    assert {c.workflow_id for c in listed} == {f"{job_id}:convert:{i}" for i in range(workers)}
    tasks = await jobs.list_tasks(job_id)
    assert [t.seq for t in tasks if t.stage == "convert"] == [0, 1, 2, 3, 4, 5]
    await attach_document(dbos, "wide", doc.name)
    assert (await Collection("wide").search("toks5"))[0].page_start == 6, "one document"


async def test_stage_queues_cap_each_stage_separately(
    dbos, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every stage has a queue and a cap of its own: a budget of four, and twice the weight on
    converting, converts two batches at a time and embeds one, rather than letting the fastest
    stage take a share of one pool."""
    await _use(dbos, cpu_budget=4, converting_weight=2, batch_pages=1)
    caps = workflows.stage_caps((await load_user_settings()).pipeline)
    assert (caps["convert"], caps["embed"]) == (2, 1), "the split this test is about"
    docs = [
        await import_row(f"c{i}.pdf", text_pdf([f"C{i}A tokc{i}", f"C{i}B"]), tmp_path)
        for i in range(3)
    ]
    converting = Overlap(pipeline.convert_batch, wait_for=caps["convert"])
    embedding = Overlap(pipeline.embed_batch, wait_for=caps["embed"])
    monkeypatch.setattr(pipeline, "convert_batch", converting)
    monkeypatch.setattr(pipeline, "embed_batch", embedding)

    ids = [await dbos.start_import(d.name) for d in docs]
    assert [await wait_for(job_id) for job_id in ids] == ["imported"] * len(ids)

    assert converting.reached.is_set(), "the convert queue never ran two batches at once"
    assert converting.peak == caps["convert"], f"convert peak {converting.peak}, cap {caps}"
    assert embedding.peak == caps["embed"], f"embed peak {embedding.peak}, cap {caps}"
    assert len(converting.order) == len(embedding.order) == 6, "three documents, two pages each"


@pytest.mark.parametrize(
    ("name", "cpu_budget"),
    [
        ("a budget of one runs one task at a time, whatever stage it belongs to", 1),
        ("a budget of two runs two, while the stage caps would have allowed three", 2),
    ],
)
async def test_the_cpu_budget_bounds_every_stage_together(
    name: str, cpu_budget: int, dbos, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The floor of one slot per stage puts the caps over a small budget: three stages, one slot
    each, is three. The process-wide semaphore is what holds the line, so the tasks of all stages
    together never exceed the budget.

    The counter only observes; holding work open to force an overlap would deadlock against the
    very limit under test. It counts the CPU work of both stages rather than their steps, because
    the slot is taken around that work alone (see `CpuOverlap`)."""
    await _use(dbos, cpu_budget=cpu_budget, batch_pages=1)
    caps = workflows.stage_caps((await load_user_settings()).pipeline)
    assert sum(caps.values()) > cpu_budget, "the queues alone would have allowed more"
    await Collection.create("budget")
    docs = [
        await import_row(f"b{i}.pdf", text_pdf([f"B{i}A tokb{i}", f"B{i}B"]), tmp_path)
        for i in range(3)
    ]
    running = CpuOverlap()  # one counter for both stages
    monkeypatch.setattr(convert, "pdf_pages_markdown", running.wrap(convert.pdf_pages_markdown))
    monkeypatch.setattr(chunk, "split", running.wrap(chunk.split))

    ids = [await dbos.start_import(d.name) for d in docs]
    assert [await wait_for(job_id) for job_id in ids] == ["imported"] * len(ids)

    assert running.peak <= cpu_budget, f"{running.peak} tasks at once, budget {cpu_budget}: {name}"
    assert running.calls == 12, "three documents, two pages each, converted and embedded"
    for d in docs:
        await attach_document(dbos, "budget", d.name)
    found = [(await Collection("budget").search(f"tokb{i}"))[0].page_start for i in range(3)]
    assert found == [1, 1, 1]


async def test_embedding_overlaps_conversion_of_other_documents(
    dbos, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One slot per stage and a budget of two to pay for two of them: while one document's convert
    slice is held open, another document's embed slice runs. On one shared queue of that size the
    embed would have waited."""
    await _use(dbos, cpu_budget=2, batch_pages=1)
    first = await import_row("a.pdf", text_pdf(["alpha one"]), tmp_path)
    second = await import_row("b.pdf", text_pdf(["beta two"]), tmp_path)
    # `holds_cpu`: both gates take a slot of the budget, which is what pays for the overlap
    embedding = Gate(doc=first.name, holds_cpu=True)  # holds one of the two slots
    converting = Gate(doc=second.name, holds_cpu=True)  # and the convert step takes the other
    monkeypatch.setattr(pipeline, "embed_batch", embedding.wrap(pipeline.embed_batch))
    monkeypatch.setattr(pipeline, "convert_batch", converting.wrap(pipeline.convert_batch))

    first_job = await dbos.start_import(first.name)
    assert await wait_event(embedding.entered), "the first document never reached its embed step"
    second_job = await dbos.start_import(second.name)

    assert await wait_event(converting.entered), (
        "the convert step waited for the embed step of the other document"
    )
    assert embedding.calls == [0] and not embedding.release.is_set(), "the embed step is still in"

    converting.release.set()
    embedding.release.set()
    assert [await wait_for(first_job), await wait_for(second_job)] == ["imported", "imported"]


async def test_a_budget_of_one_stops_the_stages_from_overlapping(
    dbos, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same run on a budget of one: every stage still has its slot, but there is only one slot
    to take, so the convert of the second document waits for the embed of the first to let go."""
    await _use(dbos, cpu_budget=1, batch_pages=1)
    first = await import_row("a.pdf", text_pdf(["alpha one"]), tmp_path)
    second = await import_row("b.pdf", text_pdf(["beta two"]), tmp_path)
    embedding = Gate(doc=first.name, holds_cpu=True)  # holds the only CPU slot
    converting = Gate(doc=second.name, holds_cpu=True)
    monkeypatch.setattr(pipeline, "embed_batch", embedding.wrap(pipeline.embed_batch))
    monkeypatch.setattr(pipeline, "convert_batch", converting.wrap(pipeline.convert_batch))

    first_job = await dbos.start_import(first.name)
    assert await wait_event(embedding.entered), "the first document never reached its embed step"
    second_job = await dbos.start_import(second.name)

    assert not await wait_event(converting.entered, BLOCKED_WAIT), (
        "the convert step took a second slot"
    )
    assert not embedding.release.is_set(), "the embed step still holds the only slot"

    embedding.release.set()  # the slot comes free, and the convert of the second document takes it
    assert await wait_event(converting.entered), "the convert step never got the freed slot"
    converting.release.set()
    assert [await wait_for(first_job), await wait_for(second_job)] == ["imported", "imported"]


async def test_document_parallelism_caps_a_single_document(
    dbos, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`document_parallelism` bounds the slices a document is cut into, so one big file cannot
    take every worker while other documents wait. At 1 the whole stage is one child again."""
    await _use(dbos, workers=3, batch_pages=1, document_parallelism=1)
    doc = await import_row("s.pdf", text_pdf([f"S{i} toks{i}" for i in range(4)]), tmp_path)
    overlap = Overlap(pipeline.convert_batch, wait_for=1)
    monkeypatch.setattr(pipeline, "convert_batch", overlap)

    job_id = await dbos.start_import(doc.name)
    assert await wait_for(job_id) == "imported"

    assert overlap.peak == 1, f"batches overlapped, peak {overlap.peak}"
    assert overlap.order == [0, 1, 2, 3], "one batch after another, in plan order"
    listed = await DBOS.list_workflows_async(
        workflow_id_prefix=f"{job_id}:convert", load_input=False
    )
    assert [c.workflow_id for c in listed] == [f"{job_id}:convert:0"], "a single slice"


async def test_one_document_creates_a_bounded_number_of_workflows(dbos, tmp_path: Path) -> None:
    """However many batches a document has, it costs one orchestrator per pipeline, one child per
    convert and embed slice, and one index child - never one workflow per micro-batch. The
    maintenance run the index asks for is debounced, so a burst of documents shares a single one."""
    workers = 3
    await _use(dbos, workers=workers, batch_pages=1, index_group_parts=1)
    await Collection.create("few")
    doc = await import_row("p.pdf", text_pdf(["alpha", "beta", "gamma"]), tmp_path)

    import_id = await dbos.start_import(doc.name)
    assert await wait_for(import_id) == "imported"
    index_id = await dbos.attach("few", doc.name)
    assert await wait_for(index_id) == "indexed"

    async def named(prefix: str) -> Counter[str]:
        listed = await DBOS.list_workflows_async(
            workflow_id_prefix=prefix, load_input=False, load_output=False
        )
        return Counter(s.name for s in listed)

    assert await named(import_id) == {dbos_names.IMPORT_WORKFLOW: 1, dbos_names.STAGE_WORKFLOW: 3}
    embedding = _embed_id(import_id, doc.name)
    assert await named(embedding) == {dbos_names.EMBED_WORKFLOW: 1, dbos_names.STAGE_WORKFLOW: 3}
    assert await named(index_id) == {
        dbos_names.COLLECTION_DOCUMENT_WORKFLOW: 1,
        dbos_names.STAGE_WORKFLOW: 1,  # the index stage is one writer, so never sliced
        dbos_names.MAINTAIN_WORKFLOW: 1,
    }
    stages = {
        *(f"{import_id}:convert:{i}" for i in range(3)),
        *(f"{embedding}:embed:{i}" for i in range(3)),
        f"{index_id}:index",
    }
    listed = await DBOS.list_workflows_async(
        name=dbos_names.STAGE_WORKFLOW, load_input=False, load_output=False
    )
    assert {s.workflow_id for s in listed} == stages
    assert len(await jobs.list_tasks(import_id)) == 3
    assert len(await jobs.list_tasks(embedding)) == 3
    assert len(await jobs.list_tasks(index_id)) == 3


async def test_index_groups_parts_into_one_write(dbos, tmp_path: Path) -> None:
    """S2: the index stage writes a whole document in one LanceDB commit. Three micro-batches
    convert and embed separately, but they land as a single index task and a single fragment."""
    await _use(dbos, workers=2, batch_pages=1)  # one part per page, default grouping
    collection = await Collection.create("grouped")
    seed = await import_document(dbos, "a.md", MD, tmp_path)
    await attach_document(dbos, "grouped", seed.name)
    before = await _fragments(collection)
    doc = await import_document(
        dbos, "p.pdf", text_pdf(["alpha one", "beta two", "gamma three"]), tmp_path
    )

    job_id = await dbos.attach("grouped", doc.name)
    assert await wait_for(job_id) == "indexed"

    tasks = await jobs.list_tasks(job_id)
    assert [(t.stage, t.seq, t.page_start, t.page_end) for t in tasks] == [("index", 0, 0, 3)]
    assert {t.status for t in tasks} == {"SUCCESS"}
    assert await _fragments(collection) == before + 1, "one commit for the document"
    table = await (await collection.index())._existing()
    assert table is not None
    rows = [r for r in (await table.to_arrow()).to_pylist() if r["doc"] == doc.name]
    assert len(rows) == await _cached_rows(doc.name) > 0
    assert sorted({r["part"] for r in rows}) == [0, 1, 2], "every part landed in that one commit"
    assert (await Collection("grouped").search("gamma"))[0].page_start == 3


async def test_parts_stay_and_only_the_scratch_rows_are_consumed(dbos, tmp_path: Path) -> None:
    """The part files are the input of every later embedding, so they live as long as the
    document; only the `rows.json` scratch of one computation is consumed by the cache write."""
    await _use(dbos, workers=2, batch_pages=1)
    doc = await import_document(dbos, "p.pdf", text_pdf(["alpha one", "beta two"]), tmp_path)

    (entry,) = await embed_cache.entries(doc.name)
    assert sorted(p.name for p in doc.parts_dir.glob("*.md")) == ["000000.md", "000001.md"]
    assert not embed_cache.scratch_dir(doc.name, entry.id).exists(), "scratch rows are consumed"
    assert [p.name for p in doc.embeddings_dir.glob("*.parquet")] == [f"{entry.id}.parquet"]
    assert "alpha one" in doc.markdown.read_text()


# --- the embedding cache -----------------------------------------------------------


async def test_two_collections_with_the_same_params_embed_once(
    dbos, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the refactor: two collections whose effective chunk settings match compute
    one embedding between them. Asserted on the call count of the embed work, not on the equal
    row counts it would also produce."""
    spy = _spy_embed(monkeypatch)
    for name in ("left", "right"):
        await Collection.create(name)
    doc = await import_document(dbos, "a.md", MD, tmp_path)
    assert sum(spy.calls.values()) == 1, "the import pre-warmed the default params, once"

    await attach_document(dbos, "left", doc.name)
    await attach_document(dbos, "right", doc.name)

    assert len(spy.calls) == 1, f"more than one embedding was computed: {spy.calls}"
    assert sum(spy.calls.values()) == 1, "and neither attach recomputed it"
    (entry,) = await embed_cache.entries(doc.name)
    assert list(spy.calls) == [entry.id], "the one computation is the one cached row"
    assert [p.name for p in doc.embeddings_dir.glob("*.parquet")] == [f"{entry.id}.parquet"]
    assert (await Collection("left").search("lancedb"))[0].collection == "left"
    assert (await Collection("right").search("lancedb"))[0].collection == "right"
    assert sorted(await document.collections_of(doc.name)) == ["left", "right"]


async def test_two_collections_with_different_chunk_size_get_their_own_cache(
    dbos, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chunk settings are part of the cache key, so a second collection that splits differently
    computes and stores an embedding of its own instead of reading the first one's."""
    spy = _spy_embed(monkeypatch)
    await Collection.create("wide")
    narrow = await Collection.create("narrow")
    await narrow.set_settings(CollectionSettings(chunk_size=60, chunk_overlap=0))
    doc = await import_document(dbos, "a.md", MD, tmp_path)

    await attach_document(dbos, "wide", doc.name)
    await attach_document(dbos, "narrow", doc.name)

    entries = {entry.chunk_size: entry for entry in await embed_cache.entries(doc.name)}
    default = (await load_user_settings()).conversion.chunk_size
    assert sorted(entries) == sorted({default, 60}), "one cache row per distinct chunk size"
    assert len({entry.id for entry in entries.values()}) == 2, "distinct ids, so no collision"
    assert {p.name for p in doc.embeddings_dir.glob("*.parquet")} == {
        f"{entry.id}.parquet" for entry in entries.values()
    }
    assert spy.calls == Counter({entries[default].id: 1, entries[60].id: 1}), "one run each"
    assert entries[60].rows > entries[default].rows, "smaller chunks, more of them"
    assert (await Collection("wide").search("lancedb"))[0].doc == doc.name
    assert (await Collection("narrow").search("lancedb"))[0].doc == doc.name


async def test_reindexing_with_unchanged_settings_hits_the_cache(
    dbos, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second index of the same member recomputes nothing: `ensure_embedding` looks the cache
    up and returns, so its whole step log is that one lookup."""
    spy = _spy_embed(monkeypatch)
    await Collection.create("again")
    doc = await import_document(dbos, "a.md", MD, tmp_path)
    await attach_document(dbos, "again", doc.name)
    computed = sum(spy.calls.values())

    job_id = await dbos.start_index_collection_document("again", doc.name)
    assert await wait_for(job_id) == "indexed"

    assert sum(spy.calls.values()) == computed == 1, "the embed work did not run again"
    embedding = _embed_id(job_id, doc.name)
    steps = await _steps(embedding)
    assert "cache_lookup" in steps, f"the run never looked the cache up: {steps}"
    assert "plan" not in steps and "finalize_embed" not in steps, "it returned on the hit"
    assert len(await embed_cache.entries(doc.name)) == 1


async def test_concurrent_attaches_converge_on_one_embedding_run(
    dbos, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two collections asking for the same missing embedding at once share one run: the second
    enqueue is deduplicated by the cache id and returns the workflow already in flight, so both
    index jobs name the same `emb:` child."""
    await _use(dbos, workers=4, batch_pages=1)
    for name in ("one", "two"):
        # the same non-default chunk size in both, so neither can read the import's pre-warm
        await (await Collection.create(name)).set_settings(
            CollectionSettings(chunk_size=60, chunk_overlap=0)
        )
    doc = await import_document(dbos, "a.md", MD, tmp_path)
    gate = Gate()
    monkeypatch.setattr(pipeline, "embed_batch", gate.wrap(pipeline.embed_batch))

    first = await dbos.attach("one", doc.name)
    assert await wait_event(gate.entered), "the first attach never reached the embed step"
    second = await dbos.attach("two", doc.name)

    async def asked() -> bool:
        """The child enqueue is recorded under the child workflow's name, so the step log says
        when the second index really asked for the embedding - and it asked while the first run
        was still held open, which is what makes this a race rather than a sequence."""
        return dbos_names.EMBED_WORKFLOW in await _steps(second)

    await until(asked, "the second attach never asked for the embedding")
    gate.release.set()

    assert [await wait_for(first), await wait_for(second)] == ["indexed", "indexed"]
    shared = _embed_id(first, doc.name)
    ids = sorted(await _workflow_ids(dbos_names.EMBED_WORKFLOW))
    assert ids == sorted({_embed_id(await _import_id(doc.name), doc.name), shared}), (
        "the import's pre-warm and one shared run, not one run per collection"
    )
    assert _embed_id(second, doc.name) not in ids, "the second enqueue returned the first run"
    assert gate.calls == [0], "and the embed step ran exactly once"
    assert {e.chunk_size for e in await embed_cache.entries(doc.name)} == {
        (await load_user_settings()).conversion.chunk_size,
        60,
    }
    await _drain()


async def _import_id(doc: str) -> str:
    """The import job of one document: the only `imp:` workflow it has in these tests."""
    (found,) = await DBOS.list_workflows_async(
        name=dbos_names.IMPORT_WORKFLOW,
        workflow_id_prefix=f"{workflows.IMPORT_PREFIX}:{doc}:",
        load_input=False,
        load_output=False,
    )
    return found.workflow_id


async def test_reimport_reconverts_and_drops_the_stale_cache(dbos, tmp_path: Path) -> None:
    """Re-importing rewrites the markdown every cached embedding was chunked from, so the whole
    cache of the document goes first - rows and files - and only the fresh pre-warm is left."""
    collection = await Collection.create("stale")
    await collection.set_settings(CollectionSettings(chunk_size=60, chunk_overlap=0))
    doc = await import_document(dbos, "a.md", MD, tmp_path)
    await attach_document(dbos, "stale", doc.name)
    before = await embed_cache.entries(doc.name)
    assert len(before) == 2, "the import's default params and the collection's"
    assert len(list(doc.embeddings_dir.glob("*.parquet"))) == 2
    await document.set_status(doc.name, "error", "boom")

    assert await wait_for(await dbos.start_import(doc.name)) == "imported"

    after = await embed_cache.entries(doc.name)
    default = (await load_user_settings()).conversion.chunk_size
    assert [e.chunk_size for e in after] == [default], "only the fresh pre-warm is left"
    assert [p.name for p in doc.embeddings_dir.glob("*.parquet")] == [f"{after[0].id}.parquet"]
    assert (await document.get(doc.name)).status == "imported"
    assert (
        await embed_cache.lookup(
            embed_cache.params(
                await document.get(doc.name), ChunkSettings(chunk_size=60, chunk_overlap=0), None
            )
        )
        is None
    ), "the collection's entry is a miss until it is indexed again"


async def test_ensure_embedding_fails_permanently_under_another_model(dbos, tmp_path: Path) -> None:
    """The cache id names the model the rows were computed with, so a run whose params no longer
    match the installed profile cannot produce them: it fails at once instead of writing rows
    under the wrong id, and the parent asks again under the new model."""
    doc = await import_document(dbos, "a.md", MD, tmp_path)
    row = await document.get(doc.name)
    wanted = embed_cache.params(
        row,
        ChunkSettings(chunk_size=300, chunk_overlap=0),
        EmbeddingModel("BAAI/bge-small-en-v1.5", 384),
    )
    assert (await load_user_settings()).embedding_model is None, "the profile has no model"

    with SetWorkflowID(f"{workflows.EMBED_PREFIX}:{row.name}:{uuid4().hex}"):
        handle = await DBOS.enqueue_workflow_async(
            workflows.EMBEDDING_QUEUE, workflows.ensure_embedding, row.name, wanted
        )

    with pytest.raises(PermanentError, match="embedding model changed: wanted BAAI/"):
        await handle.get_result(polling_interval_sec=workflows.TASK_POLL)
    assert await embed_cache.lookup(wanted) is None, "nothing was written under the wrong model"
    assert "plan" not in await _steps(handle.workflow_id), "it never reached the embed stage"


# --- collection index maintenance --------------------------------------------------


async def _await(predicate, message: str, timeout: float = WAIT):
    """Poll until the coroutine function `predicate` returns something truthy, then return it."""
    deadline = time.monotonic() + timeout
    while True:
        found = await predicate()
        if found:
            return found
        assert time.monotonic() < deadline, message
        await anyio.sleep(workflows.TASK_POLL)


async def _maintenance_runs(name: str) -> list[str]:
    listed = await DBOS.list_workflows_async(name=name, load_input=False, load_output=False)
    return [s.status for s in listed]


async def test_indexing_a_member_requests_maintenance_and_counts_pending(
    dbos, tmp_path: Path
) -> None:
    """Every indexed member counts towards the collection's next maintenance run. The run is
    debounced, so the second document (with the threshold at one) starts the one the first
    document had already asked for, rather than a second one."""
    await _use(dbos, workers=2, batch_pages=10, maintenance_idle_seconds=NEVER)
    collection = await Collection.create("kept")
    first = await import_document(dbos, "a.md", MD, tmp_path)
    await attach_document(dbos, "kept", first.name)

    waiting = await maintenance_state("kept")
    assert (waiting.pending_docs, waiting.last_maintained_at) == (1, None), "counted, not due"
    assert waiting.last_write_at is not None
    assert await Collection.pending_names() == ["kept"]

    await _use(dbos, workers=2, batch_pages=10, maintenance_docs=1)  # the next one is due at once
    second = await import_document(dbos, "b.md", MD, tmp_path)
    await attach_document(dbos, "kept", second.name)
    await _drain_maintenance()

    settled = await maintenance_state("kept")
    assert (settled.pending_docs, await Collection.pending_names()) == (0, [])
    assert settled.last_maintained_at is not None
    assert await _maintenance_runs(dbos_names.MAINTAIN_WORKFLOW) == ["SUCCESS"], "one run"
    info = await collection.info()
    assert info.index is not None and info.maintenance.pending_docs == 0
    assert info.index.num_rows > 0 and info.index.has_fts_index, "the run built the index"
    assert info.index.unindexed_rows == 0, "and folded every row written since into it"


async def test_maintenance_runs_on_the_collection_partition(
    dbos, tmp_path: Path, monkeypatch
) -> None:
    """LanceDB takes one writer per collection, so maintenance shares the index stage's partition:
    while a document's index step is in flight, the run waits instead of compacting under it."""
    await _use(dbos, workers=4, batch_pages=1, index_group_parts=1, maintenance_idle_seconds=NEVER)
    await Collection.create("onewriter")
    doc = await import_document(dbos, "p.pdf", text_pdf(["alpha", "beta"]), tmp_path)
    gate = Gate(seq=0)
    monkeypatch.setattr(pipeline, "index_batch", gate.wrap(pipeline.index_batch))
    entered: list[str] = []
    real_run = maintenance.run

    async def noting_run(collection, *rest):
        entered.append(collection.name)
        return await real_run(collection, *rest)

    monkeypatch.setattr(maintenance, "run", noting_run)
    job_id = await dbos.attach("onewriter", doc.name)
    assert await wait_event(gate.entered), "the index step never started"

    # one pending document is already `maintenance_docs`, so the run is due with no delay
    await workflows.request_maintenance("onewriter", 1, PipelineSettings(maintenance_docs=1))

    async def enqueued() -> list[str]:
        return await _maintenance_runs(dbos_names.MAINTAIN_PARTITION_WORKFLOW)

    async def finished() -> bool:
        return await _maintenance_runs(dbos_names.MAINTAIN_PARTITION_WORKFLOW) == ["SUCCESS"]

    assert await _await(enqueued, "maintenance was never enqueued") == ["ENQUEUED"], (
        "the queue has it, but the collection's partition is taken"
    )
    assert entered == [], "so it has not touched the table"

    gate.release.set()
    assert await wait_for(job_id) == "indexed"

    assert await _await(finished, "maintenance never finished")
    assert entered == ["onewriter"], "it ran once the index stage let go of the partition"
    await _drain()  # the document asked for a second run, which stays DELAYED for `NEVER` seconds


async def test_maintenance_skips_a_collection_without_a_table(dbos) -> None:
    """A collection nobody indexed has nothing to compact. The run settles anyway, so it is not
    rescheduled at every boot."""
    await Collection.create("empty")
    assert await Collection("empty").note_indexed() == 1

    report = await wait_for(
        (await workflows.MAINTAIN.debounce_async("empty", 0.0, "empty")).workflow_id
    )

    assert report.skipped == "no-table" and report.collection == "empty"
    assert (await maintenance_state("empty")).pending_docs == 0
    assert await Collection.pending_names() == []


async def test_maintenance_skips_a_collection_deleted_while_it_waited(dbos, tmp_path: Path) -> None:
    """A run may sit in the queue while the collection is deleted, so it never asks for the
    collection before it has checked that there still is one."""
    await _use(dbos, workers=2, batch_pages=10, maintenance_idle_seconds=NEVER)
    collection = await Collection.create("vanish")
    doc = await import_document(dbos, "a.md", MD, tmp_path)
    await attach_document(dbos, "vanish", doc.name)
    await collection.delete()

    report = await wait_for(
        (await workflows.MAINTAIN.debounce_async("vanish", 0.0, "vanish")).workflow_id
    )

    assert report.skipped == "no-collection"
    assert await Collection.pending_names() == [], "the row went with the collection"
    assert (await document.get(doc.name)).status == "imported", "the document is untouched"


async def test_boot_schedules_pending_collections(dbos, tmp_path: Path) -> None:
    """A run lost to a shutdown leaves `pending_docs` standing, so the next boot asks again."""
    await _use(dbos, workers=2, batch_pages=10, maintenance_idle_seconds=NEVER)
    await Collection.create("left")
    doc = await import_document(dbos, "a.md", MD, tmp_path)
    await attach_document(dbos, "left", doc.name)
    assert await Collection.pending_names() == ["left"]
    await save_user_settings(UserSettings(pipeline=PipelineSettings(maintenance_idle_seconds=1)))
    await restart_dbos()  # the boot reschedules every collection that is still pending

    await _drain_maintenance()
    state = await maintenance_state("left")
    assert (state.pending_docs, await Collection.pending_names()) == (0, [])
    assert state.last_maintained_at is not None


# --- retries: transient versus permanent -------------------------------------------


async def test_transient_step_failure_is_retried_and_recovers(
    dbos, tmp_path: Path, monkeypatch
) -> None:
    doc = await import_row("p.pdf", text_pdf(["alpha"]), tmp_path)
    real = pipeline.convert_batch
    calls: list[int] = []

    async def flaky(doc_, batch):
        calls.append(batch.seq)
        if len(calls) < 3:
            raise RuntimeError("database is locked")
        return await real(doc_, batch)

    monkeypatch.setattr(pipeline, "convert_batch", flaky)

    assert await wait_for(await dbos.start_import(doc.name)) == "imported"

    assert calls == [0, 0, 0], "two failures, then the third attempt succeeds"
    assert (await document.get(doc.name)).status == "imported"
    assert (await jobs.list_jobs()).items[0].status == "SUCCESS"


async def test_transient_step_failure_gives_up_after_max_attempts(
    dbos, tmp_path: Path, monkeypatch
) -> None:
    """One slice, so the whole stage is one step log and "the rest never ran" is a fact about it."""
    await _use(dbos, workers=4, batch_pages=1, document_parallelism=1)
    doc = await import_row("p.pdf", text_pdf([f"P{i}" for i in range(4)]), tmp_path)
    real = pipeline.convert_batch
    calls: list[int] = []

    async def flaky(doc_, batch):
        calls.append(batch.seq)
        if batch.seq == 2:
            raise RuntimeError("boom")
        return await real(doc_, batch)

    monkeypatch.setattr(pipeline, "convert_batch", flaky)
    job_id = await dbos.start_import(doc.name)

    with pytest.raises(workflows.PipelineError, match="RuntimeError: boom"):
        await wait_for(job_id)

    assert calls.count(2) == 3, "step retried max_attempts times"
    row = await document.get(doc.name)
    assert row.status == "error" and "boom" in (row.error or "")
    (job,) = (await jobs.list_jobs()).items
    assert job.status == "ERROR" and job.error and "boom" in job.error
    statuses = {t.seq: t.status for t in await jobs.list_tasks(job_id) if t.stage == "convert"}
    assert statuses == {0: "SUCCESS", 1: "SUCCESS", 2: "ERROR", 3: "ENQUEUED"}, (
        "the stage stops at the failed batch; the rest never ran"
    )


async def test_permanent_step_failure_is_not_retried(dbos, tmp_path: Path, monkeypatch) -> None:
    """A document that needs OCR fails the same way on every attempt, so it is reported at once
    instead of burning three attempts."""
    doc = await import_row("p.pdf", text_pdf(["alpha"]), tmp_path)
    calls: list[int] = []

    async def needs_ocr(doc_, batch):
        calls.append(batch.seq)
        raise PermanentError("all 1 pages need OCR")

    monkeypatch.setattr(pipeline, "convert_batch", needs_ocr)
    job_id = await dbos.start_import(doc.name)

    with pytest.raises(workflows.PipelineError, match="PermanentError: all 1 pages need OCR"):
        await wait_for(job_id)

    assert calls == [0], "a permanent failure is raised by the workflow, not retried by the step"
    row = await document.get(doc.name)
    assert (row.status, row.error) == ("error", "PermanentError: all 1 pages need OCR")
    (job,) = (await jobs.list_jobs()).items
    assert job.status == "ERROR" and job.error == "PermanentError: all 1 pages need OCR"
    (task,) = await jobs.list_tasks(job_id)
    assert (task.stage, task.status, task.error) == (
        "convert",
        "ERROR",
        "PermanentError: all 1 pages need OCR",
    )


async def test_indexing_a_document_that_is_not_imported_fails_the_membership(
    dbos, tmp_path: Path
) -> None:
    """The collection index reads markdown the import produces, so a member whose document never
    finished importing fails permanently - and only the membership carries that failure."""
    collection = await Collection.create("early")
    doc = await import_row("a.md", into=tmp_path)
    await _add_member(collection, doc.name)

    job_id = await dbos.start_index_collection_document("early", doc.name)

    with pytest.raises(workflows.PipelineError, match="document is not imported: queued"):
        await wait_for(job_id)
    member = await collection.member(doc.name)
    assert member.status == "error" and "not imported" in (member.error or "")
    assert (await document.get(doc.name)).status == "queued", "the document's status is untouched"
    assert await embed_cache.entries(doc.name) == [], "nothing was computed"
    assert await collection.search("intro") == []


# --- crash recovery ----------------------------------------------------------------


async def test_import_resumes_after_a_crash_without_duplicating_chunks(
    dbos, tmp_path: Path, monkeypatch
) -> None:
    """F2: destroy DBOS while a convert step is in flight, then launch it again. The workflow
    resumes from its step log, so the document ends `imported` once, with no duplicate rows."""
    await _use(dbos, workers=2, batch_pages=1)
    await Collection.create("dur")
    doc = await import_row("p.pdf", text_pdf(["alpha one", "beta two"]), tmp_path)
    gate = Gate(seq=0)
    monkeypatch.setattr(pipeline, "convert_batch", gate.wrap(pipeline.convert_batch))

    job_id = await dbos.start_import(doc.name)
    assert await wait_event(gate.entered), "the convert step never started"
    DBOS.destroy(workflow_completion_timeout_sec=0)  # crash, mid-step
    gate.release.set()
    await dbos.start()  # restart: same application_version, so recovery picks the workflow up

    assert await wait_for(job_id) == "imported"

    assert (await document.get(doc.name)).status == "imported"
    assert gate.calls.count(0) == 2, "the interrupted batch ran again after recovery"
    (job,) = [j for j in (await jobs.list_jobs()).items if j.action == "import"]
    assert (job.id, job.status) == (job_id, "SUCCESS"), "recovery resumes, it does not re-enqueue"
    await attach_document(dbos, "dur", doc.name)
    table = await (await Collection("dur").index())._existing()
    assert table is not None and await table.count_rows() == await _cached_rows(doc.name) > 0


async def test_adopt_orphans_resumes_only_stale_in_flight_workflows(
    dbos, tmp_path: Path, monkeypatch
) -> None:
    gate = Gate()
    real = pipeline.convert_batch

    async def slow_pdf(doc_, batch):
        if doc_.name.endswith(".pdf"):
            return await gate.wrap(real)(doc_, batch)
        return await real(doc_, batch)

    monkeypatch.setattr(pipeline, "convert_batch", slow_pdf)
    # the finished one first: a cold model load on CI can outlast the gate's patience
    quick = await import_row("done.md", "# d\n", tmp_path)
    finished = await dbos.start_import(quick.name)  # done before the stale mark; left alone
    await wait_for(finished)
    slow = await import_row("slow.pdf", text_pdf(["x"]), tmp_path)
    running = await dbos.start_import(slow.name)
    assert await wait_event(gate.entered)

    async with db.connect() as conn:  # pretend everything so far ran under an older build
        await conn.execute("update workflow_status set application_version = 'old-build'")
    resumed: list[str] = []

    async def resume_workflows(ids, **kwargs) -> None:
        resumed.extend(ids)

    monkeypatch.setattr(DBOS, "resume_workflows_async", resume_workflows)

    assert await dbos.adopt_orphans() == len(resumed) > 0
    in_flight = {
        s.workflow_id for s in await DBOS.list_workflows_async(status=["PENDING", "ENQUEUED"])
    }
    assert set(resumed) == in_flight and running in in_flight and finished not in in_flight
    assert await dbos.adopt_orphans() == len(in_flight), "idempotent while they remain stale"

    gate.release.set()
    await wait_for(running)  # drain before teardown


async def test_adopt_orphans_runs_in_the_background_on_start(dbos, monkeypatch) -> None:
    """A deep backlog must not hold up the boot: `start` hands adoption to a task on the loop it
    runs on and returns while it is still running."""
    started = threading.Event()
    release = threading.Event()

    async def slow_adopt(batch: int = 0) -> int:
        started.set()
        assert await wait_event(release), "the test never released the adoption"
        return 3

    monkeypatch.setattr(workflows, "adopt_orphans", slow_adopt)
    await restart_dbos()  # boot again, with adoption blocked

    assert await wait_event(started), "adoption started"
    assert not release.is_set(), "start did not wait for it"
    adopting = workflows._adoption
    assert adopting is not None and adopting.get_name() == "haskie-adopt"
    assert not adopting.done(), "the boot returned while the adoption is still running"
    # the module holds the only strong reference, so a stuck adoption is still cancellable
    release.set()
    await adopting
    assert adopting.done(), "and it finished once the test released it"


# --- cancel, detach and delete while a pipeline runs -------------------------------


async def test_cancel_job_marks_the_document_cancelled(dbos, tmp_path: Path, monkeypatch) -> None:
    doc = await import_row("p.pdf", text_pdf(["one"]), tmp_path)
    gate = Gate()
    monkeypatch.setattr(pipeline, "convert_batch", gate.wrap(pipeline.convert_batch))
    job_id = await dbos.start_import(doc.name)
    assert await wait_event(gate.entered)

    await workflows.cancel_job(job_id)

    row = await document.get(doc.name)
    assert (row.status, row.error) == ("cancelled", None)
    assert (await jobs.list_jobs()).items[0].status == "CANCELLED"
    gate.release.set()
    with pytest.raises(DBOSAwaitedWorkflowCancelledError):
        await wait_for(job_id)  # let its worker thread observe the cancellation before teardown


async def test_cancel_job_marks_the_member_cancelled(dbos, tmp_path: Path, monkeypatch) -> None:
    """An `idx-col:` job belongs to one membership, so cancelling it moves that membership and
    leaves the document (and every other collection holding it) alone."""
    await _use(dbos, workers=2, batch_pages=1, index_group_parts=1)
    collection = await Collection.create("cx")
    doc = await import_document(dbos, "p.pdf", text_pdf(["one", "two"]), tmp_path)
    gate = Gate(seq=0)
    monkeypatch.setattr(pipeline, "index_batch", gate.wrap(pipeline.index_batch))
    job_id = await dbos.attach("cx", doc.name)
    assert await wait_event(gate.entered)

    await workflows.cancel_job(job_id)

    assert (await collection.member(doc.name)).status == "cancelled"
    assert (await document.get(doc.name)).status == "imported", "the document is not a member"
    assert (await jobs.list_jobs("cx")).items[0].status == "CANCELLED"
    gate.release.set()
    await await_terminal([job_id])
    await _drain()


async def test_cancel_job_rejects_unknown_ids_and_leaves_finished_jobs_alone(
    dbos, tmp_path: Path
) -> None:
    await Collection.create("done")
    doc = await import_document(dbos, "g.md", MD, tmp_path)
    job_id = await dbos.attach("done", doc.name)
    assert await wait_for(job_id) == "indexed"

    with pytest.raises(NotFound, match="job not found: ghost"):
        await workflows.cancel_job("ghost")
    with pytest.raises(NotFound, match="job not found: ghost"):
        await jobs.list_tasks("ghost")

    await workflows.cancel_job(job_id)  # no-op: the job is already terminal

    assert (await Collection("done").member(doc.name)).status == "indexed"
    assert (await jobs.list_jobs("done")).items[0].status == "SUCCESS"


async def test_delete_document_while_it_indexes_leaves_nothing_behind(
    dbos, tmp_path: Path, monkeypatch
) -> None:
    """F6: the delete cancels every pipeline of the document, waits, then removes its rows from
    every collection, its folder and its row."""
    await _use(dbos, workers=4, batch_pages=1)
    collection = await Collection.create("mid")
    doc = await import_row("p.pdf", text_pdf(["alpha", "beta"]), tmp_path)
    gate = Gate()
    monkeypatch.setattr(pipeline, "convert_batch", gate.wrap(pipeline.convert_batch))
    job_id = await dbos.start_import(doc.name)
    assert await wait_event(gate.entered)

    with anyio.fail_after(WAIT * 2):  # a delete that never returns fails the test here
        await delete_document(dbos, doc.name)

    assert await _statuses([job_id]) == ["CANCELLED"]
    assert await document_names() == [], "the DB row goes last, and it is gone"
    assert not document.root(doc.name).exists()
    assert (await collection.counts()).total == 0, "the membership went with the document"
    gate.release.set()
    await await_terminal([job_id])
    await _drain()


async def test_delete_document_clears_every_collection_it_is_in(dbos, tmp_path: Path) -> None:
    """Many-to-many, so a delete is not one collection's business: both memberships go, both
    tables lose the rows, and only then do the folder and the cache rows follow."""
    for name in ("left", "right"):
        await Collection.create(name)
    doc = await import_document(dbos, "a.md", MD, tmp_path)
    other = await import_document(dbos, "b.md", "# B\n\nbeta body\n", tmp_path)
    for name in ("left", "right"):
        await attach_document(dbos, name, doc.name)
    await attach_document(dbos, "left", other.name)
    assert await embed_cache.entries(doc.name), "the cache holds the document's embedding"

    await delete_document(dbos, doc.name)

    assert await document_names() == [other.name]
    assert not document.root(doc.name).exists()
    assert await embed_cache.entries(doc.name) == [], "the cache rows cascade with the document"
    for name in ("left", "right"):
        assert await Collection(name).member_names() == ([other.name] if name == "left" else []), (
            name
        )
        assert {
            h.doc for h in await Collection(name).search("lancedb", SearchOverrides(limit=5))
        } == set(), name
    assert {h.doc for h in await Collection("left").search("beta", SearchOverrides(limit=5))} == {
        other.name
    }


async def test_attaching_while_a_delete_runs_is_refused(dbos, tmp_path: Path, monkeypatch) -> None:
    """The delete sets `deleting` before it snapshots the memberships, so an attach that lands
    after the snapshot is refused rather than leaving rows in a table no membership points at."""
    for name in ("left", "right", "third"):
        await Collection.create(name)
    doc = await import_document(dbos, "a.md", MD, tmp_path)
    for name in ("left", "right"):
        await attach_document(dbos, name, doc.name)
    entered, release = threading.Event(), threading.Event()
    real = document.collections_of
    reads = 0

    async def gated(name: str) -> list[str]:
        # the delete reads the collections twice: first to cancel their index jobs, then for the
        # membership snapshot, which is the one this test has to land behind
        nonlocal reads
        reads += 1
        if reads > 1:
            entered.set()
            assert await wait_event(release), "the test never released the snapshot"
        return await real(name)

    monkeypatch.setattr(document, "collections_of", gated)

    job_id = await dbos.start_delete_document(doc.name)
    assert await wait_event(entered), "the delete never took its membership snapshot"
    assert (await document.get(doc.name)).status == "deleting"

    with pytest.raises(Conflict, match="a.md"):  # `Collection.add` takes only an imported document
        await dbos.attach("third", doc.name)
    with pytest.raises(Conflict, match="document is deleting; only a queued, failed or cancelled"):
        await dbos.start_import(doc.name)

    release.set()
    await wait_for(job_id)
    assert await Collection("third").member_names() == [], "no membership the snapshot missed"
    assert await Collection("third").search("lancedb") == [], "and no orphaned index row"
    assert not Collection("third").index_dir.exists(), "the refused attach wrote no table"
    assert await document_names() == []


async def test_detach_leaves_the_document_and_the_other_collection(dbos, tmp_path: Path) -> None:
    """A detach is one collection's business: its rows and its membership go, and the document,
    its cache and every other collection holding it are untouched."""
    for name in ("keep", "drop"):
        await Collection.create(name)
    doc = await import_document(dbos, "a.md", MD, tmp_path)
    for name in ("keep", "drop"):
        await attach_document(dbos, name, doc.name)
    (entry,) = await embed_cache.entries(doc.name)

    await dbos.detach("drop", doc.name)

    assert await Collection("drop").member_names() == []
    assert await Collection("drop").search("lancedb") == []
    assert await Collection("keep").member_names() == [doc.name]
    assert (await Collection("keep").search("lancedb"))[0].doc == doc.name
    assert (await document.get(doc.name)).status == "imported"
    assert await document.collections_of(doc.name) == ["keep"]
    assert [e.id for e in await embed_cache.entries(doc.name)] == [entry.id], "the cache stays"
    assert embed_cache.file_path(doc.name, entry.id).is_file()
    assert doc.markdown.exists() and doc.parts_dir.exists()


async def test_detach_waits_for_the_index_write_in_flight(
    dbos, tmp_path: Path, monkeypatch
) -> None:
    """F3: the detach cancels the member's index workflow, and that cancel rewrites the status row
    and nothing else - the LanceDB write already running writes its rows anyway. The removal takes
    the collection's write lock after that write, so the rows go with the membership instead of
    outliving it."""
    await _use(dbos, workers=4, batch_pages=1, index_group_parts=1)
    collection = await Collection.create("part")
    done = await import_document(dbos, "done.md", MD, tmp_path)
    await attach_document(dbos, "part", done.name)
    slow = await import_document(dbos, "slow.pdf", text_pdf(["alpha", "beta"]), tmp_path)
    gate = Gate()
    monkeypatch.setattr(pipeline, "index_batch", gate.wrap(pipeline.index_batch))
    in_flight = await dbos.attach("part", slow.name)
    assert await wait_event(gate.entered)
    order: list[str] = []

    async def release_once_cancelled() -> None:
        await until(_all_cancelled([in_flight]), "the detach never cancelled the index in flight")
        order.append("released")
        gate.release.set()

    async with anyio.create_task_group() as releasing:
        releasing.start_soon(release_once_cancelled)
        await dbos.detach("part", slow.name)
        order.append("detached")

    assert order == ["released", "detached"], "the detach waited for the write holding the lock"
    await await_terminal([in_flight])
    assert gate.calls == [0], "one batch wrote past the cancel; the next stopped at its step"
    assert await collection.member_names() == [done.name]
    assert await _rows_of(collection, slow.name) == 0, "its rows went with the membership"
    assert await _rows_of(collection, done.name) > 0, "and the other member kept its own"
    assert {h.doc for h in await collection.search("alpha", SearchOverrides(limit=5))} == {
        done.name
    }, "both documents carry 'alpha'; only the one still attached is found"
    with pytest.raises(NotFound, match="document not in collection part: slow.pdf"):
        await dbos.detach("part", slow.name)
    await _drain()


async def test_detach_rejects_an_unknown_membership(dbos, tmp_path: Path) -> None:
    await Collection.create("rm")
    with pytest.raises(NotFound, match="document not in collection rm: ghost.md"):
        await dbos.detach("rm", "ghost.md")
    with pytest.raises(NotFound, match="collection not found: ghost"):
        await dbos.detach("ghost", "a.md")
    with pytest.raises(NotFound, match="document not found: ghost.md"):
        await dbos.start_delete_document("ghost.md")


async def test_delete_collection_cancels_every_document_in_flight(
    dbos, tmp_path: Path, monkeypatch
) -> None:
    await _use(dbos, workers=4, batch_pages=1, index_group_parts=1)
    collection = await Collection.create("dl")
    names = [
        (await import_document(dbos, f"d{i}.pdf", text_pdf(["alpha", "beta"]), tmp_path)).name
        for i in range(2)
    ]
    gate = Gate()
    monkeypatch.setattr(pipeline, "index_batch", gate.wrap(pipeline.index_batch))

    ids = [await dbos.attach("dl", name) for name in names]
    # one writer per collection, so the first member holds the partition and the second waits on
    # it; both are active, and the delete has to reach both
    assert await wait_event(gate.entered), "no member reached its index step"

    async def release_once_cancelled() -> None:
        # the cancel leaves the gated write running, and it holds the collection's lock: the
        # removal only gets it once the test lets that write finish
        await until(_all_cancelled(ids), "the delete never cancelled the members in flight")
        gate.release.set()

    async with anyio.create_task_group() as releasing:
        releasing.start_soon(release_once_cancelled)
        await delete_collection(dbos, "dl")

    assert await _statuses(ids) == ["CANCELLED", "CANCELLED"]
    assert await Collection.names() == [] and not collection.root.exists()
    assert sorted(await document_names()) == names, "the documents outlive the collection"
    await await_terminal(ids)
    await _drain()


async def test_delete_collection_rejects_an_unknown_name(dbos) -> None:
    with pytest.raises(NotFound, match="collection not found: ghost"):
        await delete_collection(dbos, "ghost")
    with pytest.raises(NotFound, match="collection not found: ghost"):
        await dbos.start_index_collection("ghost")


# --- whole-collection jobs ---------------------------------------------------------


async def _member_workflows(collection: str) -> list[str]:
    """Every collection-index workflow of one collection, finished or not."""
    found = await DBOS.list_workflows_async(
        name=dbos_names.COLLECTION_DOCUMENT_WORKFLOW,
        workflow_id_prefix=f"{workflows.COLLECTION_DOCUMENT_PREFIX}:{collection}:",
        load_input=False,
        load_output=False,
    )
    return [s.workflow_id for s in found]


async def test_index_collection_workflow_enqueues_every_member_in_pages(
    dbos, tmp_path: Path, monkeypatch
) -> None:
    """D2: "Index all" is a background job that walks the collection one page of enqueues at a
    time, so the request costs the same whether it holds five documents or ten thousand."""
    monkeypatch.setattr(workflows, "BULK_INDEX_PAGE", 2)  # three pages for five documents
    collection = await Collection.create("b")
    names = []
    for i in range(5):
        doc = await import_document(dbos, f"d{i}.md", f"# d{i}\n\nbody {i}\n", tmp_path)
        await collection.add(doc.name)
        names.append(doc.name)

    job_id = await dbos.start_index_collection("b")

    assert await wait_for(job_id) == workflows.BulkResult(done=5, skipped=0)
    queued = await _member_workflows("b")
    assert len(queued) == 5, "one index per member, and no second one for any of them"
    assert [await wait_for(i) for i in queued] == ["indexed"] * 5
    assert (
        sorted(
            m.document.name
            for m in (await collection.members_page(paging.PageRequest())).items
            if m.status == "indexed"
        )
        == names
    )
    progress = await DBOS.get_event_async(job_id, workflows.PROGRESS_EVENT, timeout_seconds=0)
    assert (progress.done, progress.skipped, progress.total, progress.last) == (5, 0, 5, None)
    await _drain()


async def test_index_collection_workflow_is_idempotent_on_replay(
    dbos, tmp_path: Path, monkeypatch
) -> None:
    """The page listing is a step and every child id is derived from the bulk job, so a crash
    between two pages re-attaches to the documents already queued instead of queueing them twice."""
    monkeypatch.setattr(workflows, "BULK_INDEX_PAGE", 2)
    collection = await Collection.create("b")
    for i in range(5):
        doc = await import_document(dbos, f"d{i}.md", f"# d{i}\n\nbody {i}\n", tmp_path)
        await collection.add(doc.name)
    real = workflows.member_page
    pages: list[str | None] = []
    entered, release = threading.Event(), threading.Event()

    async def gated(name: str, after: str | None) -> list[str]:
        pages.append(after)
        if len(pages) == 2:  # the first page is queued; stop the job right here
            entered.set()
            assert await wait_event(release), "the test never released the listing"
        return await real(name, after)

    monkeypatch.setattr(workflows, "member_page", gated)
    job_id = await dbos.start_index_collection("b")
    assert await wait_event(entered), "the second page never started"
    DBOS.destroy(workflow_completion_timeout_sec=0)  # crash, between two pages
    release.set()
    await dbos.start()  # restart: same application_version, so recovery picks the workflow up

    assert await wait_for(job_id) == workflows.BulkResult(done=5, skipped=0)

    queued = await _member_workflows("b")
    assert len(queued) == 5, "the replay re-attached instead of queueing the first page again"
    await await_terminal(queued)
    await _drain()


async def test_delete_collection_workflow_cancels_and_removes(
    dbos, tmp_path: Path, monkeypatch
) -> None:
    """F2: the deletion is a job too. It cancels everything the collection has in flight, and the
    cancel is final for the status row alone - the LanceDB write already running keeps going - so
    it waits on the collection's write lock before it drops the rows and the folder. A write that
    outlived the cancel must not recreate either."""
    await _use(dbos, workers=4, batch_pages=1, index_group_parts=1)
    collection = await Collection.create("wipe")
    done = await import_document(dbos, "done.md", MD, tmp_path)
    await attach_document(dbos, "wipe", done.name)
    slow = await import_document(dbos, "slow.pdf", text_pdf(["alpha", "beta"]), tmp_path)
    gate = Gate()
    monkeypatch.setattr(pipeline, "index_batch", gate.wrap(pipeline.index_batch))
    in_flight = await dbos.attach("wipe", slow.name)
    assert await wait_event(gate.entered)
    order: list[str] = []

    async def release_once_cancelled() -> None:
        await until(_all_cancelled([in_flight]), "the delete never cancelled the index in flight")
        order.append("released")
        gate.release.set()

    async with anyio.create_task_group() as releasing:
        releasing.start_soon(release_once_cancelled)
        bulk_id = await dbos.start_delete_collection("wipe")
        assert await wait_for(bulk_id) is None
        order.append("removed")

    assert order == ["released", "removed"], "the removal waited for the write holding the lock"
    assert await _statuses([in_flight]) == ["CANCELLED"]
    assert await Collection.names() == [] and not collection.root.exists()
    assert sorted(await document_names()) == ["done.md", "slow.pdf"], "documents are untouched"
    job = await jobs.bulk_job(bulk_id)
    assert (job.kind, job.collection, job.status, job.progress) == (
        "delete_collection",
        "wipe",
        "SUCCESS",
        None,
    )
    await await_terminal([in_flight])
    await _drain()
    assert await Collection.names() == [] and not collection.root.exists(), (
        "and the cancelled child put back neither the row nor the table folder"
    )


async def test_bulk_job_reads_the_three_bulk_kinds(dbos, tmp_path: Path) -> None:
    """The three whole-thing jobs share one read model; only `del-doc:` names a document rather
    than a collection in its second segment, so its `collection` is None."""
    await Collection.create("kinds")
    doc = await import_document(dbos, "a.md", MD, tmp_path)
    await attach_document(dbos, "kinds", doc.name)

    index_id = await dbos.start_index_collection("kinds")
    await wait_for(index_id)
    delete_doc_id = await dbos.start_delete_document(doc.name)
    await wait_for(delete_doc_id)
    delete_id = await dbos.start_delete_collection("kinds")
    await wait_for(delete_id)

    assert [
        (j.kind, j.collection)
        for j in [await jobs.bulk_job(i) for i in (index_id, delete_doc_id, delete_id)]
    ] == [
        ("index_collection", "kinds"),
        ("delete_document", None),
        ("delete_collection", "kinds"),
    ]
    await _drain()


# --- jobs read model ---------------------------------------------------------------


async def _walk_jobs(collection: str | None = None, limit: int = 2) -> list[jobs.Job]:
    """Every job of the listing, one page at a time, exactly as the UI's "Load more" reads it."""
    walked: list[jobs.Job] = []
    cursor: str | None = None
    while True:
        page = await jobs.list_jobs(collection, limit, cursor)
        walked.extend(page.items)
        cursor = page.next_cursor
        if cursor is None:
            return walked


async def test_list_jobs_reports_the_action_collection_and_document(dbos, tmp_path: Path) -> None:
    """One document costs three jobs of three actions. Only the collection index belongs to a
    collection; an import and an embedding run belong to the document alone."""
    await Collection.create("c")
    doc = await import_document(dbos, "a.md", MD, tmp_path)
    await attach_document(dbos, "c", doc.name)

    listed = (await jobs.list_jobs()).items

    assert {(j.action, j.collection, j.doc) for j in listed} == {
        ("import", None, "a.md"),
        ("embed", None, "a.md"),
        ("index", "c", "a.md"),
    }
    assert {j.status for j in listed} == {"SUCCESS"}
    rows = (await jobs.list_kind("document")).items
    assert {row.title for row in rows} == {"import a.md", "embed a.md", "c / a.md"}
    assert {row.kind for row in rows} == {"document"}


async def test_list_jobs_filters_by_collection_before_it_cuts_the_window(
    dbos, tmp_path: Path
) -> None:
    """A busy collection must not push an older one out of the window: the filter is the
    job id's prefix, so the database applies it before it cuts the page."""
    for name in ("noisy", "quiet"):
        await Collection.create(name)
    doc = await import_document(dbos, "a.md", MD, tmp_path)
    await attach_document(dbos, "quiet", doc.name)  # oldest collection index of all
    await attach_document(dbos, "noisy", doc.name)
    for _ in range(2):
        await wait_for(await dbos.start_index_collection_document("noisy", doc.name))

    quiet = await jobs.list_jobs("quiet", page_size=2)
    (job,) = quiet.items
    assert (job.action, job.collection, job.doc) == ("index", "quiet", "a.md")
    assert job.status == "SUCCESS"
    assert quiet.next_cursor is None, "the filtered listing has one page"

    noisy = await jobs.list_jobs("noisy", page_size=2)
    assert [j.collection for j in noisy.items] == ["noisy", "noisy"], "newest first"
    assert noisy.next_cursor is not None, "one more behind this page"
    assert [j.collection for j in await _walk_jobs("noisy")] == ["noisy"] * 3
    assert {j.action for j in await _walk_jobs()} == {"import", "embed", "index"}, "unfiltered"


async def test_list_jobs_rejects_a_bad_cursor(dbos) -> None:
    """The cursor is an opaque source and offset into one fixed ordering: anything else is a bad
    request, not an empty page."""
    with pytest.raises(InvalidInput, match="invalid cursor"):
        await jobs.list_jobs(cursor="not-a-cursor")
    with pytest.raises(InvalidInput, match="cursor does not match"):
        await jobs.list_jobs(cursor=paging.encode_cursor(["a.md"], "name", "asc"))
    with pytest.raises(InvalidInput, match="cursor does not match"):  # the cursor of an older build
        await jobs.list_jobs(cursor=paging.encode_cursor([0], jobs.JOB_SORT, jobs.JOB_ORDER))
    # an offset that is no offset, another listing's cursor, and an identity that is no name
    for key in ([jobs.DOCUMENT_KIND, -1], ["collection", 0], [7, 0]):
        with pytest.raises(InvalidInput, match="invalid cursor"):
            await jobs.list_jobs(cursor=paging.encode_cursor(key, jobs.JOB_SORT, jobs.JOB_ORDER))
    for limit in (0, paging.MAX_PAGE_SIZE + 1):
        with pytest.raises(InvalidInput, match=r"page_size must be 1\.\.1000"):
            await jobs.list_jobs(page_size=limit)


async def test_list_kind_pages_on_a_cursor_of_its_own(dbos) -> None:
    """Each section of the jobs view pages through one kind of workflow, newest first, on an
    opaque offset cursor bound to that kind: one from another section would page another history,
    and an unknown kind is a bad request rather than an empty page."""
    await Collection.create("pager")
    older = await dbos.start_index_collection("pager")
    await await_terminal([older])  # a second "index all" while one runs is the same job
    newer = await dbos.start_index_collection("pager")
    await await_terminal([newer])

    first = await jobs.list_kind("collection", page_size=1)

    assert [row.id for row in first.items] == [newer], "newest first"
    assert first.items[0].title == "index collection pager"
    assert first.next_cursor is not None
    second = await jobs.list_kind("collection", page_size=1, cursor=first.next_cursor)
    assert [row.id for row in second.items] == [older]
    assert second.next_cursor is None, "the last page ends the walk"

    with pytest.raises(InvalidInput, match="invalid cursor"):
        await jobs.list_kind("download", page_size=1, cursor=first.next_cursor)
    with pytest.raises(InvalidInput, match="unknown job kind 'bogus'"):
        await jobs.list_kind("bogus")
    with pytest.raises(InvalidInput, match=r"page_size must be 1\.\.1000"):
        await jobs.list_kind("collection", page_size=0)


async def test_list_jobs_never_loads_inputs(dbos, tmp_path: Path, monkeypatch) -> None:
    """The collection and the document come out of the job id, so a page of jobs costs two
    queries and no input payload at all."""
    await Collection.create("lean")
    doc = await import_document(dbos, "a.md", MD, tmp_path)
    await attach_document(dbos, "lean", doc.name)
    calls = counted_list_workflows(monkeypatch)

    (job,) = (await jobs.list_jobs("lean")).items

    assert (job.collection, job.doc) == ("lean", "a.md"), "read from the id"
    parent, children = calls
    assert parent["load_input"] is False, "the parent listing never reads inputs"
    assert parent["workflow_id_prefix"] == "idx-col:lean:" and parent["sort_desc"] is True
    assert children["load_output"] is False, "one query for the children of the whole page"
    assert len(calls) == 2, "no query per job"


async def test_active_collection_workflows_use_the_id_prefix(
    dbos, tmp_path: Path, monkeypatch
) -> None:
    """`idx-col:{collection}:` ends in the separator, so collection `a` never matches `ab`."""
    for name in ("a", "ab"):
        await Collection.create(name)
    doc = await import_document(dbos, "p.pdf", text_pdf(["alpha"]), tmp_path)
    gate = Gate()
    monkeypatch.setattr(pipeline, "index_batch", gate.wrap(pipeline.index_batch))
    job_id = await dbos.attach("ab", doc.name)
    assert await wait_event(gate.entered)

    assert await dbos._active_collection_workflows("ab") == [job_id]
    assert await dbos._active_collection_workflows("a") == [], (
        "a collection whose name is a prefix of another"
    )
    assert await dbos._active_collection_workflows("ab", doc.name) == [job_id]
    assert await dbos._active_collection_workflows("ab", "q.pdf") == [], "another document"

    gate.release.set()
    await wait_for(job_id)
    await _drain()


async def _seed_jobs(job_id: str, collection: str, count: int) -> None:
    """`count` more collection-index rows in the DBOS history, copied from a real one: only the id
    and the timestamp differ, so every column holds what DBOS itself writes."""
    async with db.connect() as conn:
        cursor = await conn.execute(
            "select * from workflow_status where workflow_uuid = ?", (job_id,)
        )
        columns = [description[0] for description in cursor.description]
        seed = await cursor.fetchone()
        assert seed is not None, "the workflow row to copy is there"
        row = dict(zip(columns, seed, strict=True))
    row["deduplication_id"] = None  # unique per queue; a copy may not claim the original's
    copies = [
        tuple(
            dict(
                row,
                workflow_uuid=f"idx-col:{collection}:d{i}.md:{uuid4().hex}",
                created_at=row["created_at"] - i - 1,
            )[column]
            for column in columns
        )
        for i in range(count)
    ]
    async with db.connect() as conn:
        await conn.executemany(
            f"insert into workflow_status ({','.join(columns)}) "
            f"values ({','.join('?' * len(columns))})",
            copies,
        )


async def test_list_jobs_stays_fast_over_a_long_history(dbos, tmp_path: Path) -> None:
    """The point of the id prefix: one quiet collection's page costs the same with five thousand
    jobs of a busy one behind it as with none."""
    for name in ("noisy", "quiet"):
        await Collection.create(name)
    doc = await import_document(dbos, "a.md", MD, tmp_path)
    job_id = await dbos.attach("quiet", doc.name)
    assert await wait_for(job_id) == "indexed"
    await _seed_jobs(job_id, "noisy", 5000)

    started = time.perf_counter()
    page = await jobs.list_jobs("quiet", page_size=100)
    elapsed = time.perf_counter() - started

    assert [(j.collection, j.doc) for j in page.items] == [("quiet", "a.md")]
    assert page.next_cursor is None
    busy = (await jobs.list_jobs("noisy", page_size=100)).items
    assert len(busy) == 100, "the busy collection really is in the history"
    assert elapsed < 0.2, f"one page of a quiet collection took {elapsed:.3f}s over 5000 jobs"


async def test_list_tasks_reports_stage_slices_still_waiting(
    dbos, tmp_path: Path, monkeypatch
) -> None:
    """A slice is one child workflow with a step per batch, so its tasks come from the step log:
    finished ones from their output, the next one running, the rest enqueued. One slice here, so
    "the rest" is the whole remainder of the stage."""
    await _use(dbos, workers=4, batch_pages=1, document_parallelism=1, index_group_parts=1)
    doc = await import_row("p.pdf", text_pdf(["alpha", "beta", "gamma"]), tmp_path)
    gate = Gate(seq=1)
    monkeypatch.setattr(pipeline, "convert_batch", gate.wrap(pipeline.convert_batch))
    job_id = await dbos.start_import(doc.name)
    assert await wait_event(gate.entered)

    tasks = await jobs.list_tasks(job_id)

    assert [(t.stage, t.seq) for t in tasks] == [("convert", 0), ("convert", 1), ("convert", 2)]
    assert tasks[0].status == "SUCCESS" and tasks[0].result is not None
    assert [t.status for t in tasks[1:]] == ["PENDING", "ENQUEUED"]
    (job,) = [j for j in (await jobs.list_jobs()).items if j.action == "import"]
    assert (job.tasks_total, job.tasks_done, job.tasks_running) == (3, 1, 1)
    gate.release.set()
    assert await wait_for(job_id) == "imported"
    assert {t.status for t in await jobs.list_tasks(job_id)} == {"SUCCESS"}
    assert len(await jobs.list_tasks(_embed_id(job_id, doc.name))) == 3, "the embed job's own"


async def test_list_tasks_merges_the_slices_of_a_stage(dbos, tmp_path: Path, monkeypatch) -> None:
    """With the default parallelism the stage is several children, so a batch held up in one slice
    no longer holds up the others: the listing merges their step logs back into plan order."""
    await _use(dbos, workers=4, batch_pages=1, index_group_parts=1)
    doc = await import_row("p.pdf", text_pdf(["alpha", "beta", "gamma"]), tmp_path)
    gate = Gate(seq=1)
    monkeypatch.setattr(pipeline, "convert_batch", gate.wrap(pipeline.convert_batch))
    job_id = await dbos.start_import(doc.name)
    assert await wait_event(gate.entered)

    async def both_slices_done() -> list[jobs.Task] | None:
        found = await jobs.list_tasks(job_id)
        return found if sum(t.status == "SUCCESS" for t in found) == 2 else None

    tasks = await _await(both_slices_done, "the un-gated slices never finished")

    assert [(t.stage, t.seq) for t in tasks] == [("convert", 0), ("convert", 1), ("convert", 2)]
    assert [t.status for t in tasks] == ["SUCCESS", "PENDING", "SUCCESS"], (
        "seq 1 is gated in its own slice; the other two ran without it"
    )
    (job,) = [j for j in (await jobs.list_jobs()).items if j.action == "import"]
    assert (job.tasks_total, job.tasks_done, job.tasks_running) == (3, 2, 1)
    assert {t.id for t in tasks} == {f"{job_id}:convert:{i}:{i}" for i in range(3)}
    gate.release.set()
    assert await wait_for(job_id) == "imported"
    assert {t.status for t in await jobs.list_tasks(job_id)} == {"SUCCESS"}


async def test_step_outcome_reads_a_plain_step_output(dbos) -> None:
    """Steps that predate `BatchResult` (and every non-batch step) return their value directly."""
    assert jobs._step_outcome({"output": 5, "error": None}) == (5, None)
    assert jobs._step_outcome({"output": None, "error": RuntimeError("boom")}) == (None, "boom")
    assert jobs._step_outcome({"output": "text", "error": None}) == (None, None)


@pytest.mark.parametrize(
    ("name", "job_id", "expected"),
    [
        ("an import", "imp:book.pdf:cafe", ("import", None, "book.pdf")),
        ("an embedding", "emb:book.pdf:cafe", ("embed", None, "book.pdf")),
        ("a collection index", "idx-col:law:book.pdf:cafe", ("index", "law", "book.pdf")),
        ("a document delete", "del-doc:book.pdf:cafe", None),
        ("a bulk index", "bulk-index:law:cafe", None),
        ("an import without its uuid", "imp:book.pdf", None),
        ("a collection index without its uuid", "idx-col:law:book.pdf", None),
        ("a name that is no id at all", "book.pdf", None),
    ],
)
def test_a_pipeline_job_id_names_its_action_collection_and_document(
    name: str, job_id: str, expected: tuple | None
) -> None:
    assert workflows.job_names(job_id) == expected, name


@pytest.mark.parametrize(
    ("name", "job_id", "collection"),
    [
        ("a bulk index", "bulk-index:law:cafe", "law"),
        ("a bulk delete", "bulk-delete:law:cafe", "law"),
        ("a maintenance run", "maint:law:cafe", "law"),
        ("a document delete, which spans every collection", "del-doc:book.pdf:cafe", None),
        ("an id with nothing in that place", "bulk-index", None),
    ],
)
def test_only_a_job_of_one_collection_carries_its_name(
    name: str, job_id: str, collection: str | None
) -> None:
    assert jobs._collection_of(job_id) == collection, name


async def test_a_document_delete_is_listed_as_a_collection_job_with_no_collection(
    dbos, tmp_path: Path
) -> None:
    """The three whole-collection workflows share one kind and one id shape, but a document delete
    carries a document where the other two carry a collection."""
    doc = await import_document(dbos, "gone.md", MD, tmp_path)
    job_id = await dbos.start_delete_document(doc.name)
    assert await wait_for(job_id) is None

    (row,) = (await jobs.list_kind("collection", page_size=10)).items

    assert (row.id, row.kind, row.title) == (job_id, "collection", f"delete document {doc.name}")
    assert (await jobs.list_kind("collection", collection="any", page_size=10)).items == [], (
        "a collection filter keeps the jobs of one collection, and this job has none"
    )


def test_the_workflow_names_the_jobs_view_spells_out_are_the_ones_dbos_records() -> None:
    """Every name `jobs` selects by and groups by is pinned against the registration DBOS made:
    a mismatch is a silent miss in a query, not an error."""
    assert dbos_names.PIPELINE_WORKFLOWS == [
        get_dbos_func_name(workflows.import_document),
        get_dbos_func_name(workflows.ensure_embedding),
        get_dbos_func_name(workflows.index_collection_document),
    ]
    assert dbos_names.STAGE_WORKFLOW == get_dbos_func_name(workflows.stage_slice)
    assert dbos_names.STAGE_STEP == get_dbos_func_name(workflows.try_batch)
    assert workflows.stage_slice.__name__ != dbos_names.STAGE_WORKFLOW, (
        "the durable name is pinned, not derived from the function name"
    )
    assert jobs.KIND_NAMES == {
        "collection": [
            get_dbos_func_name(workflows.index_collection_workflow),
            get_dbos_func_name(workflows.delete_collection_workflow),
            get_dbos_func_name(workflows.delete_document_workflow),
        ],
        "download": [get_dbos_func_name(models.ensure_model)],
        "maintenance": [
            get_dbos_func_name(workflows.maintain_on_partition),
            get_dbos_func_name(workflows.daily_maintenance),
        ],
    }
    assert set(jobs.KIND_BY_NAME) == set(dbos_names.PIPELINE_WORKFLOWS) | {
        name for names in jobs.KIND_NAMES.values() for name in names
    }, "every kind counts the workflows it lists, and nothing else"


# --- settings applied to the runtime -----------------------------------------------


async def test_an_unreadable_settings_row_does_not_stop_the_boot(dbos, monkeypatch) -> None:
    """The loader already tolerates a row another build wrote, so `start()` applies defaults and
    `/api/status` reports the problem."""
    await save_user_settings(UserSettings(embedding="compact"))
    async with db.connect() as conn:
        await conn.execute(
            'update settings set json = \'{"pipeline": {"cpu_budget": 0}}\' where id = 1'
        )
    forget_settings()  # the row was written behind the loader's back, as another build would
    applied: list[UserSettings] = []

    async def apply_settings(value: UserSettings) -> None:
        applied.append(value)

    monkeypatch.setattr(workflows, "apply_settings", apply_settings)

    await restart_dbos()

    assert applied == [UserSettings()], "defaults, so the UI can fix the stored row"
    assert settings.settings_problem() is not None
    assert await load_user_settings() == UserSettings(), "the unreadable row is not rewritten"


async def test_settings_rejected_while_applying_fall_back_to_defaults_at_boot(
    dbos, monkeypatch, caplog
) -> None:
    """The last line of defence: whatever `apply_settings` rejects, boot continues on defaults."""
    stored = await save_user_settings(UserSettings(embedding="compact"))
    applied: list[UserSettings] = []

    async def apply_settings(value: UserSettings) -> None:
        applied.append(value)
        if len(applied) == 1:
            raise InvalidInput("cpu_budget must be >= 1, got 0")

    monkeypatch.setattr(workflows, "apply_settings", apply_settings)

    DBOS.destroy(workflow_completion_timeout_sec=0)
    with caplog.at_level("ERROR"):
        await workflows.start()

    assert applied == [stored, UserSettings()], "the stored settings first, then the defaults"
    assert events(caplog) == ["settings_invalid_at_boot"]


# --- audit trail -------------------------------------------------------------------


async def test_audit_records_carry_the_collection_and_the_document(dbos, tmp_path: Path) -> None:
    """One line per finished pipeline. An import belongs to the document alone, a collection index
    to the pair, and a failed membership records the reason it stopped."""
    collection = await Collection.create("aud")
    doc = await import_document(dbos, "a.md", MD, tmp_path)
    await attach_document(dbos, "aud", doc.name)
    early = await import_row("b.md", into=tmp_path)
    await _add_member(collection, early.name)
    with pytest.raises(workflows.PipelineError):
        await wait_for(await dbos.start_index_collection_document("aud", early.name))

    lines = audit_lines()

    assert {(r["event"], r.get("collection"), r.get("doc")) for r in lines} == {
        ("import.completed", None, "a.md"),
        ("index.completed", "aud", "a.md"),
        ("index.failed", "aud", "b.md"),
    }
    assert {r["actor"] for r in lines} == {"workflow"}
    (failed,) = [r for r in lines if r["event"] == "index.failed"]
    assert failed["outcome"] == "error" and "not imported" in failed["error"]
    assert failed["workflow_id"].startswith("idx-col:aud:b.md:")
    assert all(r["duration_ms"] >= 0 and r["app_version"] == audit.APP_VERSION for r in lines)


# --- schedules and nightly housekeeping --------------------------------------------


async def test_start_registers_the_nightly_schedule(dbos) -> None:
    """The definition lives in the system database: a second boot finds the one the first wrote
    instead of adding another."""

    async def registered() -> list:
        found = await DBOS.list_schedules_async()
        return [s for s in found if s["schedule_name"] == workflows.MAINTENANCE_SCHEDULE]

    (nightly,) = await registered()
    assert (nightly["schedule"], nightly["queue_name"]) == (
        workflows.MAINTENANCE_CRON,
        workflows.MAINTENANCE_QUEUE,
    )
    assert "daily_maintenance" in nightly["workflow_name"]

    await restart_dbos()

    assert len(await registered()) == 1


async def test_start_prunes_the_audit_trail_once(dbos) -> None:
    """A desktop session rarely lives until 03:17, so the boot prunes as well, with the retention
    the user set."""
    await save_user_settings(UserSettings(retention=RetentionSettings(audit_days=1)))
    old = home.AUDIT_DIR / "audit-2020-01-01.jsonl"
    old.write_text("{}\n")
    today = audit.path()
    today.write_text("{}\n")

    await restart_dbos()

    assert not old.exists(), "a file older than the retention is gone after one boot"
    assert today.exists(), "today's file is inside every window"


async def test_daily_maintenance_prunes_the_audit_trail(dbos) -> None:
    """The scheduled workflow does the same work on its own clock. It ignores the two arguments
    every DBOS schedule passes."""
    await save_user_settings(UserSettings(retention=RetentionSettings(audit_days=1)))
    old = home.AUDIT_DIR / "audit-2020-01-01.jsonl"
    old.write_text("{}\n")
    kept = audit.path()
    kept.write_text("{}\n")

    await workflows.daily_maintenance(datetime.now(UTC), None)

    assert not old.exists()
    assert kept.exists()


async def test_daily_maintenance_purges_the_job_history_past_the_retention(
    dbos, tmp_path: Path
) -> None:
    """DBOS keeps a finished workflow forever, so the nightly run is what bounds the Jobs view:
    a job that finished longer ago than `retention.job_days` goes, with the stage children and
    the step logs that carry its micro-batches."""
    await save_user_settings(UserSettings(retention=RetentionSettings(job_days=1)))
    doc = await import_document(dbos, "old.md", MD, tmp_path)
    listed = (await jobs.list_jobs()).items
    assert {job.doc for job in listed} == {doc.name}, "the import and the embedding it warmed"
    job_id = listed[0].id
    assert await jobs.list_tasks(job_id), "the job has micro-batches while DBOS holds it"
    two_days_ago = int((time.time() - 2 * 86400) * 1000)
    async with db.connect() as conn:  # the only way to age a job: DBOS stamps its own clock
        await conn.execute(
            "update workflow_status set completed_at = ? where completed_at is not null",
            (two_days_ago,),
        )

    await workflows.daily_maintenance(datetime.now(UTC), None)

    assert (await jobs.list_jobs()).items == [], "nothing of the document's history is left"
    with pytest.raises(NotFound, match="job not found"):
        await jobs.list_tasks(job_id)


async def test_daily_maintenance_keeps_a_job_inside_the_retention(dbos, tmp_path: Path) -> None:
    """The cutoff is the only thing that decides: a job that finished within the window stays,
    micro-batches included."""
    await save_user_settings(UserSettings(retention=RetentionSettings(job_days=28)))
    doc = await import_document(dbos, "fresh.md", MD, tmp_path)
    before = [job.id for job in (await jobs.list_jobs()).items]

    await workflows.daily_maintenance(datetime.now(UTC), None)

    assert [job.id for job in (await jobs.list_jobs()).items] == before
    assert {job.doc for job in (await jobs.list_jobs()).items} == {doc.name}


async def test_daily_maintenance_sweeps_stale_staged_uploads(dbos) -> None:
    """An upload nobody imported is not a document, so the nightly run drops it once it is older
    than the staging TTL. A fresh one is still waiting to be imported and stays."""
    stale = await document.stage("old.md", b"# old\n")
    fresh = await document.stage("new.md", b"# new\n")
    aged = time.time() - workflows.STAGING_TTL_SECONDS - 60
    os.utime(document.staging_path(stale.staging_id), (aged, aged))
    async with db.connect() as conn:  # the row carries the age; the file only does for an orphan
        await conn.execute(
            "update staging set created_at = ? where staging_id = ?",
            (aged, stale.staging_id),
        )

    await workflows.daily_maintenance(datetime.now(UTC), None)

    assert not document.staging_path(stale.staging_id).exists()
    assert document.staging_path(fresh.staging_id).is_file(), "a fresh upload is still wanted"
