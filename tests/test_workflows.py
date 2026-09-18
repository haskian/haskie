"""Durable execution: the DBOS runtime, the indexing pipeline, jobs and model lifecycle.

Every test here takes the `dbos` fixture, which launches DBOS on the test home's SQLite file and
destroys it afterwards. Concurrency is driven by `threading.Event`, never by sleeping: a fake
pipeline step signals when it has been entered and blocks until the test releases it, so the
"mid-flight" moment is a fact rather than a guess.

Threading events rather than anyio ones, because two event loops are involved: the test body runs
on the loop `pytest.mark.anyio` gives it, while the steps it gates run on DBOS's background loop.
Both sides wait on them through `_wait_event`, which hands the blocking wait to a worker thread so
neither loop is ever blocked.
"""

import functools
import shutil
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import anyio
import anyio.to_thread
import pytest
from conftest import maintenance_state, restart_dbos, text_pdf, wait_for
from dbos import DBOS
from dbos._error import DBOSAwaitedWorkflowCancelledError

from haskie import (
    audit,
    chunk,
    convert,
    cpu,
    db,
    dbos_names,
    embed,
    home,
    jobs,
    maintenance,
    models,
    paging,
    pipeline,
    session,
    settings,
    workflows,
)
from haskie.errors import (
    DocumentNotFound,
    HaskieError,
    InvalidInput,
    JobNotFound,
    LibraryNotFound,
    NeedsOcr,
    NotReady,
)
from haskie.library import Library
from haskie.paging import PageRequest
from haskie.settings import (
    LibrarySettings,
    PipelineSettings,
    RetentionSettings,
    SearchOverrides,
    SearchSettings,
    UserSettings,
    load_user_settings,
    save_user_settings,
)

pytestmark = pytest.mark.anyio

MD = "# Title\n\nintro text\n\n## Alpha\n\nalpha body about lancedb\n\n## Beta\n\nbeta body\n"
WAIT = 30.0  # generous: every wait below is released by another thread, never by a timer
BLOCKED_WAIT = 2.0  # how long a step that must not run is given to prove it by not running


async def _wait_event(event: threading.Event, timeout: float = WAIT) -> bool:
    """Wait for a `threading.Event` without blocking the loop of the caller (see the module
    docstring): the blocking wait goes to a worker thread, the caller awaits it."""
    return await anyio.to_thread.run_sync(functools.partial(event.wait, timeout))


async def _wait(job_id: str):
    return await wait_for(job_id)


async def _index(dbos, library: str, doc: str) -> None:
    assert await _wait(await dbos.start_index(library, doc)) == "indexed"


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
    """Store and apply indexing settings, so the queues really carry the given limits.

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


async def _await_terminal(workflow_ids: list[str]) -> None:
    """Drain before teardown: a step outliving DBOS.destroy() blocks interpreter exit."""
    for workflow_id in workflow_ids:
        try:
            await _wait(workflow_id)
        except Exception:  # the outcome is asserted elsewhere; this call only drains
            pass


def _events(caplog) -> list[str]:
    """Our loggers pass structlog's event dict as the record message."""
    return [r.msg["event"] for r in caplog.records if isinstance(r.msg, dict)]


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
    waiting = dbos_names.WAITING_STATUS
    while await DBOS.list_workflows_async(status=waiting, load_input=False, load_output=False):
        assert time.monotonic() < deadline, "maintenance still pending at the end of the test"
        await anyio.sleep(workflows.TASK_POLL)


async def _embedded_chunks(job_id: str) -> int:
    """Chunks the embed stage reported, from the job read model: the row count the index must
    match. The `rows.json` files it wrote are gone once the index stage finishes."""
    return sum(t.result or 0 for t in await jobs.list_tasks(job_id) if t.stage == "embed")


async def _one_part(part: int, rows) -> AsyncIterator[tuple[int, list]]:
    """`LibraryIndex.add_parts` consumes an async iterator: the pipeline decodes one `rows.json`
    at a time and awaits each read (see `pipeline._grouped_rows`)."""
    yield part, rows


async def _fragments(lib: Library) -> int:
    """Data fragments of the library's table: LanceDB writes one per commit that carries rows."""
    table = await (await lib.index())._existing()
    if table is None:
        return 0
    # lancedb annotates stats() as a dataclass but returns plain dicts
    return (await table.stats())["fragment_stats"]["num_fragments"]  # ty: ignore[not-subscriptable]


class Gate:
    """A pipeline step the test can hold open: `entered` fires on the first blocked call, and that
    call returns only once `release` is set.

    `*rest` rather than a fixed argument list, so the same gate wraps `convert_batch` and
    `embed_batch`, which takes the embedding model as well."""

    def __init__(
        self, seq: int | None = None, doc: str | None = None, *, holds_cpu: bool = False
    ) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls: list[int] = []
        self.seq = seq  # block only this batch; None = every batch
        self.doc = doc  # block only this document; None = every document
        self.holds_cpu = holds_cpu  # hold one slot of the CPU budget while blocked (see `wrap`)

    def _blocks(self, doc: str, batch) -> bool:
        return (self.seq is None or batch.seq == self.seq) and (self.doc is None or doc == self.doc)

    def wrap(self, real):
        async def blocking(lib, doc, batch, *rest):
            self.calls.append(batch.seq)
            if self._blocks(doc, batch):
                if self.holds_cpu:
                    await cpu.on_cpu("gate", self._hold)
                else:
                    self.entered.set()
                    assert await _wait_event(self.release), "the test never released the step"
            return await real(lib, doc, batch, *rest)

        return blocking

    def _hold(self) -> None:
        """The same wait, taken inside `cpu.on_cpu`, so it occupies one slot of the CPU budget for
        as long as it lasts.

        A step takes its slot inside `cpu.on_cpu`, around its CPU work alone, so a gate at the
        step's entry holds no slot at all. A test about the budget rather than about a queue asks
        for one. Sync, and run in the worker thread `cpu.on_cpu` gave it."""
        self.entered.set()
        assert self.release.wait(timeout=WAIT), "the test never released the step"


# --- indexing pipeline -------------------------------------------------------------


async def test_library_pipeline_fts(dbos) -> None:
    lib = await Library.create("Notes & Stuff")
    assert lib.name == "Notes-Stuff"
    await lib.set_settings(LibrarySettings(chunk_size=40, chunk_overlap=0))
    assert (await (await Library.get("Notes-Stuff")).settings()).chunk_size == 40

    doc = await lib.save("guide.md", MD.encode())
    assert (doc.status, doc.preview) == ("uploaded", None)
    assert not (await lib.index()).path.exists(), "upload must not index"

    previewed = await lib.ensure_preview(doc.name)
    assert previewed.preview is not None and previewed.preview.kind == "text"
    assert previewed.status == "uploaded"
    assert (lib.preview_dir(doc.name) / "preview.md").read_text() == MD

    await _index(dbos, "Notes-Stuff", doc.name)
    assert (await lib.document(doc.name)).status == "indexed"
    assert lib.markdown_path(doc.name).read_text() == MD
    hits = await lib.search("lancedb", limit=5)
    assert hits and hits[0].heading == "Alpha" and hits[0].library == "Notes-Stuff"
    assert (hits[0].line_start, hits[0].line_end, hits[0].parents) == (5, 7, ["Title"])
    assert hits[0].markdown_path == lib.relative(lib.markdown_path(doc.name)), "home-relative"
    assert hits[0].home == str(lib.home)
    assert (lib.home / hits[0].markdown_path).read_text() == MD
    assert MD[hits[0].char_start : hits[0].char_end] == hits[0].text
    assert (hits[0].page_start, hits[0].page_end) == (None, None), "no pages for markdown"
    assert (hits[0].header, hits[0].location) == ("Title > Alpha", "guide.md L5-7")

    await lib.set_status(doc.name, "error", "boom")
    assert (await lib.document(doc.name)).error == "boom"

    await lib.remove_document(doc.name)
    assert await lib.document_names() == []
    assert await lib.search("lancedb", limit=5) == []
    await lib.delete()
    assert await Library.names() == []


async def test_workflow_pipeline_indexes_documents(dbos) -> None:
    await _use(dbos, workers=2, batch_pages=10, index_group_parts=1)
    lib = await Library.create("q")
    await lib.set_settings(LibrarySettings(chunk_size=60, chunk_overlap=0))
    pdf = await lib.save("book.pdf", text_pdf([f"Chapter {i} word{i}" for i in range(1, 26)]))
    md = await lib.save("a.md", b"# a\n\nhello from a\n")

    first = await dbos.start_index("q", pdf.name)
    assert await dbos.start_index("q", pdf.name) == first, "deduplicated while active"
    second = await dbos.start_index("q", md.name)
    assert await _wait(first) == "indexed" and await _wait(second) == "indexed"

    listed = (await jobs.list_jobs("q")).items
    assert {(j.doc, j.status, j.tasks_done, j.tasks_total) for j in listed} == {
        ("book.pdf", "SUCCESS", 9, 9),  # 3 batches in each of the three stages
        ("a.md", "SUCCESS", 3, 3),
    }
    tasks = await jobs.list_tasks(first)
    assert [(t.stage, t.seq, t.page_start, t.page_end, t.status) for t in tasks] == [
        ("convert", 0, 0, 10, "SUCCESS"), ("convert", 1, 10, 20, "SUCCESS"),
        ("convert", 2, 20, 25, "SUCCESS"),
        ("embed", 0, 0, 1, "SUCCESS"), ("embed", 1, 1, 2, "SUCCESS"), ("embed", 2, 2, 3, "SUCCESS"),
        ("index", 0, 0, 1, "SUCCESS"), ("index", 1, 1, 2, "SUCCESS"), ("index", 2, 2, 3, "SUCCESS"),
    ]  # fmt: skip
    assert {d.status for d in (await lib.documents_page(PageRequest())).items} == {"indexed"}
    hit = (await lib.search("word25", limit=1))[0]
    assert (hit.page_start, hit.part) == (25, 2)
    full = lib.markdown_path(pdf.name).read_text()
    assert full[hit.char_start : hit.char_end] == hit.text

    # reindex after success starts a new job (dedup only covers active ones)
    assert await dbos.start_index("q", md.name) != second


async def test_start_index_validates_before_it_enqueues(dbos) -> None:
    await Library.create("v")
    with pytest.raises(LibraryNotFound, match="library not found: ghost"):
        await dbos.start_index("ghost", "a.md")
    with pytest.raises(DocumentNotFound, match="document not found: a.md"):
        await dbos.start_index("v", "a.md")
    assert (await jobs.list_jobs()).items == [], "a rejected request leaves no job behind"


async def test_documents_in_one_library_index_without_conflict(dbos) -> None:
    """Index tasks share one partition per library, so concurrent documents never collide."""
    await _use(dbos, workers=8, batch_pages=2)
    lib = await Library.create("serial")
    ids = [
        await dbos.start_index(
            "serial", (await lib.save(f"d{i}.pdf", text_pdf([f"D{i}P{p}" for p in range(6)]))).name
        )
        for i in range(4)
    ]
    assert [await _wait(i) for i in ids] == ["indexed"] * len(ids)
    assert {d.status for d in (await lib.documents_page(PageRequest())).items} == {"indexed"}
    assert {h.doc for h in await lib.search("D3P5", limit=1)} == {"d3.pdf"}


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

        async def counted(lib, doc, batch, *rest):
            return await self._count(real, lib, doc, batch, *rest)

        return counted

    async def __call__(self, lib, doc, batch, *rest):
        return await self._count(self.real, lib, doc, batch, *rest)

    async def _count(self, real, lib, doc, batch, *rest):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.order.append(batch.seq)
            if self.active >= self.wait_for:
                self.reached.set()
        await _wait_event(self.reached)  # already set once `wait_for` callers are inside
        try:
            return await real(lib, doc, batch, *rest)
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


def _budget(
    cpu_budget: int,
    convert: int = 1,
    embed: int = 1,
    index: int = 1,
    document_parallelism: int = PipelineSettings().document_parallelism,
) -> PipelineSettings:
    """Indexing settings that only say how the CPU budget is shared out."""
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
        (
            "each stage resolves against its own share",
            _budget(10, convert=3),
            "embed",
            2,
        ),
        (
            "a lower cap is kept",
            _budget(10, convert=3, document_parallelism=2),
            "convert",
            2,
        ),
        (
            "a higher cap is cut to the stage's share",
            _budget(10, convert=3, document_parallelism=9),
            "embed",
            2,
        ),
        (
            "equal to the stage's share",
            _budget(12, document_parallelism=4),
            "convert",
            4,
        ),
        (
            "the floor of one slot is a share too",
            _budget(2, document_parallelism=9),
            "index",
            1,
        ),
    ],
)
def test_document_parallelism_resolves_against_the_stage_share(
    name: str, indexing: PipelineSettings, kind: workflows.Stage, expected: int
) -> None:
    """A slice occupies one slot of its own stage's queue, so asking for more than that stage's
    share of the CPU budget only queues them."""
    assert workflows.resolve_parallelism(indexing, kind) == expected, name


async def test_documents_run_in_parallel_up_to_workers(
    dbos, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Documents overlap each other, up to the stage's workers. Pinned to one slice per document,
    so the overlap that is observed is between documents and not inside one of them."""
    workers = 3
    await _use(dbos, workers=workers, batch_pages=1, document_parallelism=1)
    lib = await Library.create("par")
    names = [
        (await lib.save(f"p{i}.pdf", text_pdf([f"P{i}A tokq{i}", f"P{i}B"]))).name for i in range(3)
    ]
    overlap = Overlap(pipeline.convert_batch, wait_for=2)
    monkeypatch.setattr(pipeline, "convert_batch", overlap)

    ids = [await dbos.start_index("par", name) for name in names]
    assert [await _wait(job_id) for job_id in ids] == ["indexed"] * len(ids)

    assert overlap.reached.is_set(), "documents never overlapped"
    assert 2 <= overlap.peak <= workers, f"peak {overlap.peak} outside 2..{workers}"
    assert [(await lib.search(f"tokq{i}"))[0].page_start for i in range(3)] == [1, 1, 1]


async def test_batches_of_one_document_run_in_parallel_up_to_workers(
    dbos, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One document is cut into as many convert slices as the convert queue has workers, so
    re-indexing a single large file uses every worker instead of one. The batches hold each other
    inside the step, so the overlap is observed rather than timed."""
    workers = 3
    await _use(dbos, workers=workers, batch_pages=1)
    lib = await Library.create("wide")
    doc = await lib.save("s.pdf", text_pdf([f"S{i} toks{i}" for i in range(6)]))
    overlap = Overlap(pipeline.convert_batch, wait_for=workers)
    monkeypatch.setattr(pipeline, "convert_batch", overlap)

    job_id = await dbos.start_index("wide", doc.name)
    assert await _wait(job_id) == "indexed"

    assert overlap.reached.is_set(), "the batches of one document never overlapped"
    assert overlap.peak == workers, f"peak {overlap.peak}, expected {workers}"
    listed = await DBOS.list_workflows_async(
        workflow_id_prefix=f"{job_id}:convert", load_input=False
    )
    assert {c.workflow_id for c in listed} == {f"{job_id}:convert:{i}" for i in range(workers)}
    assert [t.seq for t in await jobs.list_tasks(job_id) if t.stage == "convert"] == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]
    assert (await lib.search("toks5"))[0].page_start == 6, "every slice landed in the same document"


async def test_stage_queues_cap_each_stage_separately(
    dbos, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every stage has a queue and a cap of its own: a budget of four, and twice the weight on
    converting, converts two batches at a time and embeds one, rather than letting the fastest
    stage take a share of one pool."""
    await _use(dbos, cpu_budget=4, converting_weight=2, batch_pages=1)
    caps = workflows.stage_caps((await load_user_settings()).pipeline)
    assert (caps["convert"], caps["embed"]) == (2, 1), "the split this test is about"
    lib = await Library.create("caps")
    names = [
        (await lib.save(f"c{i}.pdf", text_pdf([f"C{i}A tokc{i}", f"C{i}B"]))).name for i in range(3)
    ]
    converting = Overlap(pipeline.convert_batch, wait_for=caps["convert"])
    embedding = Overlap(pipeline.embed_batch, wait_for=caps["embed"])
    monkeypatch.setattr(pipeline, "convert_batch", converting)
    monkeypatch.setattr(pipeline, "embed_batch", embedding)

    ids = [await dbos.start_index("caps", name) for name in names]
    assert [await _wait(job_id) for job_id in ids] == ["indexed"] * len(ids)

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
    name: str, cpu_budget: int, dbos, monkeypatch: pytest.MonkeyPatch
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
    lib = await Library.create("budget")
    names = [
        (await lib.save(f"b{i}.pdf", text_pdf([f"B{i}A tokb{i}", f"B{i}B"]))).name for i in range(3)
    ]
    running = CpuOverlap()  # one counter for both stages
    monkeypatch.setattr(convert, "pdf_pages_markdown", running.wrap(convert.pdf_pages_markdown))
    monkeypatch.setattr(chunk, "split", running.wrap(chunk.split))

    ids = [await dbos.start_index("budget", doc) for doc in names]
    assert [await _wait(job_id) for job_id in ids] == ["indexed"] * len(ids)

    assert running.peak <= cpu_budget, f"{running.peak} tasks at once, budget {cpu_budget}: {name}"
    assert running.calls == 12, "three documents, two pages each, converted and embedded"
    assert [(await lib.search(f"tokb{i}"))[0].page_start for i in range(3)] == [1, 1, 1]


async def test_embedding_overlaps_conversion_of_other_documents(
    dbos, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One slot per stage and a budget of two to pay for two of them: while one document's convert
    slice is held open, another document's embed slice runs. On one shared queue of that size the
    embed would have waited."""
    await _use(dbos, cpu_budget=2, batch_pages=1)
    lib = await Library.create("stages")
    first = await lib.save("a.pdf", text_pdf(["alpha one"]))
    second = await lib.save("b.pdf", text_pdf(["beta two"]))
    # `holds_cpu`: both gates take a slot of the budget, which is what pays for the overlap
    embedding = Gate(doc=first.name, holds_cpu=True)  # holds one of the two slots
    converting = Gate(doc=second.name, holds_cpu=True)  # and the convert step takes the other
    monkeypatch.setattr(pipeline, "embed_batch", embedding.wrap(pipeline.embed_batch))
    monkeypatch.setattr(pipeline, "convert_batch", converting.wrap(pipeline.convert_batch))

    first_job = await dbos.start_index("stages", first.name)
    assert await _wait_event(embedding.entered), "the first document never reached its embed step"
    second_job = await dbos.start_index("stages", second.name)

    assert await _wait_event(converting.entered), (
        "the convert step waited for the embed step of the other document"
    )
    assert embedding.calls == [0] and not embedding.release.is_set(), "the embed step is still in"

    converting.release.set()
    embedding.release.set()
    assert [await _wait(first_job), await _wait(second_job)] == ["indexed", "indexed"]


async def test_a_budget_of_one_stops_the_stages_from_overlapping(
    dbos, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same run on a budget of one: every stage still has its slot, but there is only one slot
    to take, so the convert of the second document waits for the embed of the first to let go."""
    await _use(dbos, cpu_budget=1, batch_pages=1)
    lib = await Library.create("alone")
    first = await lib.save("a.pdf", text_pdf(["alpha one"]))
    second = await lib.save("b.pdf", text_pdf(["beta two"]))
    embedding = Gate(doc=first.name, holds_cpu=True)  # holds the only CPU slot
    converting = Gate(doc=second.name, holds_cpu=True)
    monkeypatch.setattr(pipeline, "embed_batch", embedding.wrap(pipeline.embed_batch))
    monkeypatch.setattr(pipeline, "convert_batch", converting.wrap(pipeline.convert_batch))

    first_job = await dbos.start_index("alone", first.name)
    assert await _wait_event(embedding.entered), "the first document never reached its embed step"
    second_job = await dbos.start_index("alone", second.name)

    assert not await _wait_event(converting.entered, BLOCKED_WAIT), (
        "the convert step took a second slot"
    )
    assert not embedding.release.is_set(), "the embed step still holds the only slot"

    embedding.release.set()  # the slot comes free, and the convert of the second document takes it
    assert await _wait_event(converting.entered), "the convert step never got the freed slot"
    converting.release.set()
    assert [await _wait(first_job), await _wait(second_job)] == ["indexed", "indexed"]


async def test_document_parallelism_caps_a_single_document(
    dbos, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`document_parallelism` bounds the slices a document is cut into, so one big file cannot
    take every worker while other documents wait. At 1 the whole stage is one child again."""
    await _use(dbos, workers=3, batch_pages=1, document_parallelism=1)
    lib = await Library.create("capped")
    doc = await lib.save("s.pdf", text_pdf([f"S{i} toks{i}" for i in range(4)]))
    overlap = Overlap(pipeline.convert_batch, wait_for=1)
    monkeypatch.setattr(pipeline, "convert_batch", overlap)

    job_id = await dbos.start_index("capped", doc.name)
    assert await _wait(job_id) == "indexed"

    assert overlap.peak == 1, f"batches overlapped, peak {overlap.peak}"
    assert overlap.order == [0, 1, 2, 3], "one batch after another, in plan order"
    listed = await DBOS.list_workflows_async(
        workflow_id_prefix=f"{job_id}:convert", load_input=False
    )
    assert [c.workflow_id for c in listed] == [f"{job_id}:convert:0"], "a single slice"


async def test_one_document_creates_a_bounded_number_of_workflows(dbos) -> None:
    """However many batches a document has, it costs one orchestrator, one child per convert and
    embed slice, and one index child - never one workflow per micro-batch. The maintenance run it
    asks for is debounced, so a burst of documents shares a single one."""
    workers = 3
    await _use(dbos, workers=workers, batch_pages=1, index_group_parts=1)
    lib = await Library.create("few")
    doc = await lib.save("p.pdf", text_pdf(["alpha", "beta", "gamma"]))

    job_id = await dbos.start_index("few", doc.name)
    assert await _wait(job_id) == "indexed"

    listed = await DBOS.list_workflows_async(
        workflow_id_prefix=job_id, load_input=False, load_output=False
    )
    assert Counter(s.name for s in listed) == {
        workflows.index_document.__qualname__: 1,
        dbos_names.STAGE_WORKFLOW: 7,  # three convert slices, three embed, one index
        workflows.maintain_library.__qualname__: 1,
    }
    stages = {s.workflow_id for s in listed if s.name == dbos_names.STAGE_WORKFLOW}
    assert stages == {
        *(f"{job_id}:convert:{i}" for i in range(3)),
        *(f"{job_id}:embed:{i}" for i in range(3)),
        f"{job_id}:index",
    }
    assert len(await jobs.list_tasks(job_id)) == 9, "three batches in each of the three stages"
    (job,) = (await jobs.list_jobs("few")).items
    assert (job.tasks_total, job.tasks_done, job.tasks_running) == (9, 9, 0)


async def test_index_groups_parts_into_one_write(dbos) -> None:
    """S2: the index stage writes a whole document in one LanceDB commit. Three micro-batches
    convert and embed separately, but they land as a single index task, a single fragment, and the
    parts they used are dropped when the stage finishes."""
    await _use(dbos, workers=2, batch_pages=1)  # one part per page, default grouping
    lib = await Library.create("grouped")
    await _index(dbos, "grouped", (await lib.save("a.md", MD.encode())).name)
    before = await _fragments(lib)
    doc = await lib.save("p.pdf", text_pdf(["alpha one", "beta two", "gamma three"]))

    job_id = await dbos.start_index("grouped", doc.name)
    assert await _wait(job_id) == "indexed"

    tasks = await jobs.list_tasks(job_id)
    assert [(t.stage, t.seq, t.page_start, t.page_end) for t in tasks] == [
        ("convert", 0, 0, 1), ("convert", 1, 1, 2), ("convert", 2, 2, 3),
        ("embed", 0, 0, 1), ("embed", 1, 1, 2), ("embed", 2, 2, 3),
        ("index", 0, 0, 3),  # one task, covering the whole part range
    ]  # fmt: skip
    assert {t.status for t in tasks} == {"SUCCESS"}
    assert await _fragments(lib) == before + 1, "one commit for the document, not one per part"
    table = await (await lib.index())._existing()
    assert table is not None
    rows = [r for r in (await table.to_arrow()).to_pylist() if r["doc"] == doc.name]
    assert len(rows) == await _embedded_chunks(job_id) > 0
    assert sorted({r["part"] for r in rows}) == [0, 1, 2], "every part landed in that one commit"
    assert not lib.parts_dir(doc.name).exists(), "the parts go once the stage is finished"
    assert lib.markdown_path(doc.name).read_text(), "the assembled markdown stays"
    assert (await lib.search("gamma"))[0].page_start == 3


async def test_cleanup_parts_runs_after_finish_and_a_reindex_recreates_them(
    dbos, monkeypatch
) -> None:
    """The rows files are the input of every index step, so they may only go after the stage's
    finalizer. A second run starts at `plan_convert`, which writes them again."""
    await _use(dbos, workers=2, batch_pages=1)
    lib = await Library.create("tidy")
    doc = await lib.save("p.pdf", text_pdf(["alpha one", "beta two"]))
    real = pipeline.cleanup_parts
    rows_seen: list[int] = []

    async def watch(lib_: Library, doc_: str) -> None:
        rows_seen.append(len(list(lib_.parts_dir(doc_).glob("*.rows.json"))))
        await real(lib_, doc_)

    monkeypatch.setattr(pipeline, "cleanup_parts", watch)

    await _index(dbos, "tidy", doc.name)

    assert rows_seen == [2], "the stage still had every rows file when it finished"
    assert not lib.parts_dir(doc.name).exists()
    indexed = await (await lib.index())._existing()
    assert indexed is not None
    rows = await indexed.count_rows()

    await _index(dbos, "tidy", doc.name)  # re-index: convert recreates the parts directory

    assert rows_seen == [2, 2], "the second run rebuilt the parts it needed"
    assert not lib.parts_dir(doc.name).exists()
    again = await (await lib.index())._existing()
    assert again is not None and await again.count_rows() == rows, "no duplicate rows"
    assert (await lib.search("beta"))[0].doc == doc.name


async def test_start_index_ids_carry_library_and_doc(dbos) -> None:
    lib = await Library.create("lib")
    doc = await lib.save("a.md", MD.encode())

    job_id = await dbos.start_index("lib", doc.name)

    assert job_id.startswith("idx:lib:a.md:"), job_id
    assert await _wait(job_id) == "indexed"


# --- library index maintenance -----------------------------------------------------


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


async def test_index_document_requests_maintenance_and_counts_pending(dbos) -> None:
    """Every indexed document counts towards the library's next maintenance run. The run is
    debounced, so the second document (with the threshold at one) starts the one the first
    document had already asked for, rather than a second one."""
    await _use(dbos, workers=2, batch_pages=10, maintenance_idle_seconds=NEVER)
    lib = await Library.create("kept")
    await _index(dbos, "kept", (await lib.save("a.md", MD.encode())).name)

    waiting = await maintenance_state("kept")
    assert (waiting.pending_docs, waiting.last_maintained_at) == (1, None), "counted, not due"
    assert waiting.last_write_at is not None
    assert await Library.pending_names() == ["kept"]

    await _use(
        dbos, workers=2, batch_pages=10, maintenance_docs=1
    )  # the next document is due at once
    await _index(dbos, "kept", (await lib.save("b.md", MD.encode())).name)
    await _drain_maintenance()

    settled = await maintenance_state("kept")
    assert (settled.pending_docs, await Library.pending_names()) == (0, [])
    assert settled.last_maintained_at is not None
    assert await _maintenance_runs(workflows.maintain_library.__qualname__) == ["SUCCESS"], (
        "one run"
    )
    status = (await lib.info()).index
    assert status is not None and status.pending_docs == 0
    assert status.num_rows > 0 and status.has_fts_index, "the run built the full-text index"
    assert status.unindexed_rows == 0, "and folded every row written since into it"


async def test_maintenance_runs_on_the_library_partition(dbos, monkeypatch) -> None:
    """LanceDB takes one writer per library, so maintenance shares the index stage's partition:
    while a document's index step is in flight, the run waits instead of compacting under it."""
    await _use(dbos, workers=4, batch_pages=1, maintenance_idle_seconds=NEVER)
    lib = await Library.create("onewriter")
    doc = await lib.save("p.pdf", text_pdf(["alpha", "beta"]))
    gate = Gate(seq=0)
    monkeypatch.setattr(pipeline, "index_batch", gate.wrap(pipeline.index_batch))
    entered: list[str] = []
    real_run = maintenance.run

    async def noting_run(lib_, *rest):
        entered.append(lib_.name)
        return await real_run(lib_, *rest)

    monkeypatch.setattr(maintenance, "run", noting_run)
    job_id = await dbos.start_index("onewriter", doc.name)
    assert await _wait_event(gate.entered), "the index step never started"

    await workflows.request_maintenance("onewriter", 1, 1, 0)  # due now, with no delay

    child = workflows.maintain_on_partition.__qualname__

    async def enqueued() -> list[str]:
        return await _maintenance_runs(child)

    async def finished() -> bool:
        return await _maintenance_runs(child) == ["SUCCESS"]

    assert await _await(enqueued, "maintenance was never enqueued") == ["ENQUEUED"], (
        "the queue has it, but the library's partition is taken"
    )
    assert entered == [], "so it has not touched the table"

    gate.release.set()
    assert await _wait(job_id) == "indexed"

    assert await _await(finished, "maintenance never finished")
    assert entered == ["onewriter"], "it ran once the index stage let go of the partition"
    await _drain()  # the document asked for a second run, which stays DELAYED for `NEVER` seconds


async def test_maintenance_skips_a_library_without_a_table(dbos) -> None:
    """A library nobody indexed has nothing to compact. The run settles anyway, so it is not
    rescheduled at every boot."""
    await Library.create("empty")
    assert await Library("empty").note_indexed() == 1

    report = await _wait(
        (await workflows.MAINTAIN.debounce_async("empty", 0.0, "empty")).workflow_id
    )

    assert report.skipped == "no-table" and report.library == "empty"
    assert (await maintenance_state("empty")).pending_docs == 0
    assert await Library.pending_names() == []


async def test_maintenance_skips_a_library_deleted_while_it_waited(dbos) -> None:
    """A run may sit in the queue while the library is deleted, so it never asks for the library
    before it has checked that there still is one."""
    await _use(dbos, workers=2, batch_pages=10, maintenance_idle_seconds=NEVER)
    lib = await Library.create("vanish")
    await _index(dbos, "vanish", (await lib.save("a.md", MD.encode())).name)
    await lib.delete()

    report = await _wait(
        (await workflows.MAINTAIN.debounce_async("vanish", 0.0, "vanish")).workflow_id
    )

    assert report.skipped == "no-library"
    assert await Library.pending_names() == [], "the row went with the library"


async def test_boot_schedules_pending_libraries(dbos) -> None:
    """A run lost to a shutdown leaves `pending_docs` standing, so the next boot asks again."""
    await _use(dbos, workers=2, batch_pages=10, maintenance_idle_seconds=NEVER)
    lib = await Library.create("left")
    await _index(dbos, "left", (await lib.save("a.md", MD.encode())).name)
    assert await Library.pending_names() == ["left"]
    await save_user_settings(UserSettings(pipeline=PipelineSettings(maintenance_idle_seconds=1)))
    await restart_dbos()  # the boot reschedules every library that is still pending

    await _drain_maintenance()
    state = await maintenance_state("left")
    assert (state.pending_docs, await Library.pending_names()) == (0, [])
    assert state.last_maintained_at is not None


# --- retries: transient versus permanent -------------------------------------------


async def test_transient_step_failure_is_retried_and_recovers(dbos, monkeypatch) -> None:
    lib = await Library.create("flaky")
    doc = await lib.save("p.pdf", text_pdf(["alpha"]))
    real = pipeline.convert_batch
    calls: list[int] = []

    async def flaky(lib_, doc_, batch, settings_):
        calls.append(batch.seq)
        if len(calls) < 3:
            raise RuntimeError("database is locked")
        return await real(lib_, doc_, batch, settings_)

    monkeypatch.setattr(pipeline, "convert_batch", flaky)

    assert await _wait(await dbos.start_index("flaky", doc.name)) == "indexed"

    assert calls == [0, 0, 0], "two failures, then the third attempt succeeds"
    assert (await lib.document(doc.name)).status == "indexed"
    assert (await jobs.list_jobs("flaky")).items[0].status == "SUCCESS"


async def test_transient_step_failure_gives_up_after_max_attempts(dbos, monkeypatch) -> None:
    """One slice, so the whole stage is one step log and "the rest never ran" is a fact about it."""
    await _use(dbos, workers=4, batch_pages=1, document_parallelism=1)
    lib = await Library.create("bad")
    doc = await lib.save("p.pdf", text_pdf([f"P{i}" for i in range(4)]))
    real = pipeline.convert_batch
    calls: list[int] = []

    async def flaky(lib_, doc_, batch, settings_):
        calls.append(batch.seq)
        if batch.seq == 2:
            raise RuntimeError("boom")
        return await real(lib_, doc_, batch, settings_)

    monkeypatch.setattr(pipeline, "convert_batch", flaky)
    job_id = await dbos.start_index("bad", doc.name)

    with pytest.raises(workflows.PipelineError, match="RuntimeError: boom"):
        await _wait(job_id)

    assert calls.count(2) == 3, "step retried max_attempts times"
    assert (await lib.document(doc.name)).status == "error"
    assert "boom" in ((await lib.document(doc.name)).error or "")
    (job,) = (await jobs.list_jobs("bad")).items
    assert job.status == "ERROR" and job.error and "boom" in job.error
    statuses = {t.seq: t.status for t in await jobs.list_tasks(job_id) if t.stage == "convert"}
    assert statuses == {0: "SUCCESS", 1: "SUCCESS", 2: "ERROR", 3: "ENQUEUED"}, (
        "the stage stops at the failed batch; the rest never ran"
    )


async def test_permanent_step_failure_is_not_retried(dbos, monkeypatch) -> None:
    """A document that needs OCR fails the same way on every attempt, so it is reported at once
    instead of burning three attempts (A9)."""
    lib = await Library.create("perm")
    doc = await lib.save("p.pdf", text_pdf(["alpha"]))
    calls: list[int] = []

    async def needs_ocr(lib_, doc_, batch, settings_):
        calls.append(batch.seq)
        raise NeedsOcr("all 1 pages need OCR")

    monkeypatch.setattr(pipeline, "convert_batch", needs_ocr)
    job_id = await dbos.start_index("perm", doc.name)

    with pytest.raises(workflows.PipelineError, match="NeedsOcr: all 1 pages need OCR"):
        await _wait(job_id)

    assert calls == [0], "a permanent failure is raised by the workflow, not retried by the step"
    document = await lib.document(doc.name)
    assert (document.status, document.error) == ("error", "NeedsOcr: all 1 pages need OCR")
    (job,) = (await jobs.list_jobs("perm")).items
    assert job.status == "ERROR" and job.error == "NeedsOcr: all 1 pages need OCR"
    (task,) = await jobs.list_tasks(job_id)
    assert (task.stage, task.status, task.error) == (
        "convert",
        "ERROR",
        "NeedsOcr: all 1 pages need OCR",
    )


# --- crash recovery ----------------------------------------------------------------


async def test_index_resumes_after_a_crash_without_duplicating_chunks(dbos, monkeypatch) -> None:
    """F2: destroy DBOS while a convert step is in flight, then launch it again. The workflow
    resumes from its step log, so the document ends `indexed` once, with no duplicate rows."""
    await _use(dbos, workers=2, batch_pages=1)
    lib = await Library.create("dur")
    doc = await lib.save("p.pdf", text_pdf(["alpha one", "beta two"]))
    gate = Gate(seq=0)
    monkeypatch.setattr(pipeline, "convert_batch", gate.wrap(pipeline.convert_batch))

    job_id = await dbos.start_index("dur", doc.name)
    assert await _wait_event(gate.entered), "the convert step never started"
    DBOS.destroy(workflow_completion_timeout_sec=0)  # crash, mid-step
    gate.release.set()
    await dbos.start()  # restart: same application_version, so recovery picks the workflow up

    assert await _wait(job_id) == "indexed"

    assert (await lib.document(doc.name)).status == "indexed"
    assert gate.calls.count(0) == 2, "the interrupted batch ran again after recovery"
    (job,) = (await jobs.list_jobs("dur")).items
    assert (job.id, job.status) == (job_id, "SUCCESS"), "recovery resumes, it does not re-enqueue"
    table = await (await lib.index())._existing()
    assert table is not None and await table.count_rows() == await _embedded_chunks(job_id) > 0


async def test_adopt_orphans_resumes_only_stale_in_flight_workflows(dbos, monkeypatch) -> None:
    gate = Gate()
    real = pipeline.convert_batch

    async def slow_pdf(lib_, doc_, batch, settings_):
        if doc_.endswith(".pdf"):
            return await gate.wrap(real)(lib_, doc_, batch, settings_)
        return await real(lib_, doc_, batch, settings_)

    monkeypatch.setattr(pipeline, "convert_batch", slow_pdf)
    lib = await Library.create("orph")
    running = await dbos.start_index("orph", (await lib.save("slow.pdf", text_pdf(["x"]))).name)
    assert await _wait_event(gate.entered)
    await lib.save("done.md", b"# d\n")
    finished = await dbos.start_index("orph", "done.md")  # finishes fast; must be left alone
    await _wait(finished)

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
    await _wait(running)  # drain before teardown


# --- cancel and remove while a pipeline runs ---------------------------------------


async def test_cancel_job_marks_the_document_cancelled(dbos, monkeypatch) -> None:
    lib = await Library.create("cx")
    doc = await lib.save("p.pdf", text_pdf(["one"]))
    gate = Gate()
    monkeypatch.setattr(pipeline, "convert_batch", gate.wrap(pipeline.convert_batch))
    job_id = await dbos.start_index("cx", doc.name)
    assert await _wait_event(gate.entered)

    await jobs.cancel_job(job_id)

    document = await lib.document(doc.name)
    assert (document.status, document.error) == ("cancelled", None)
    assert (await jobs.list_jobs("cx")).items[0].status == "CANCELLED"
    gate.release.set()
    with pytest.raises(DBOSAwaitedWorkflowCancelledError):
        await _wait(job_id)  # let its worker thread observe the cancellation before teardown


async def test_cancel_job_rejects_unknown_ids_and_leaves_finished_jobs_alone(dbos) -> None:
    lib = await Library.create("done")
    doc = await lib.save("g.md", MD.encode())
    job_id = await dbos.start_index("done", doc.name)
    assert await _wait(job_id) == "indexed"

    with pytest.raises(JobNotFound, match="job not found: ghost"):
        await jobs.cancel_job("ghost")
    with pytest.raises(JobNotFound, match="job not found: ghost"):
        await jobs.list_tasks("ghost")

    await jobs.cancel_job(job_id)  # no-op: the job is already terminal

    assert (await lib.document(doc.name)).status == "indexed", (
        "a finished document keeps its status"
    )
    assert (await jobs.list_jobs("done")).items[0].status == "SUCCESS"


async def test_remove_document_while_it_indexes_leaves_nothing_behind(dbos, monkeypatch) -> None:
    """F6: the removal cancels the pipeline, waits, and then deletes index rows, files and row."""
    await _use(dbos, workers=4, batch_pages=1)
    lib = await Library.create("mid")
    doc = await lib.save("p.pdf", text_pdf(["alpha", "beta"]))
    await lib.ensure_preview(doc.name)
    gate = Gate()
    monkeypatch.setattr(pipeline, "convert_batch", gate.wrap(pipeline.convert_batch))
    job_id = await dbos.start_index("mid", doc.name)
    assert await _wait_event(gate.entered)

    with anyio.fail_after(WAIT * 2):  # a removal that never returns fails the test here
        await dbos.remove_document("mid", doc.name)

    assert await _statuses([job_id]) == ["CANCELLED"]
    assert await lib.document_names() == [], "the DB row goes last, and it is gone"
    assert not lib.parts_dir(doc.name).exists() and not lib.preview_dir(doc.name).exists()
    assert not lib.file_path(doc.name).exists() and not lib.markdown_path(doc.name).exists()
    assert await lib.search("alpha") == []
    gate.release.set()
    await _await_terminal([job_id])
    await _drain()


async def test_remove_document_rejects_an_unknown_document(dbos) -> None:
    await Library.create("rm")
    with pytest.raises(DocumentNotFound, match="document not found: ghost.md"):
        await dbos.remove_document("rm", "ghost.md")
    with pytest.raises(LibraryNotFound, match="library not found: ghost"):
        await dbos.remove_document("ghost", "a.md")


async def test_delete_library_cancels_every_document_in_flight(dbos, monkeypatch) -> None:
    await _use(dbos, workers=4, batch_pages=1)
    lib = await Library.create("dl")
    names = [(await lib.save(f"d{i}.pdf", text_pdf(["alpha", "beta"]))).name for i in range(2)]
    started = threading.Barrier(3)  # both documents plus the test
    release = threading.Event()
    arrived: set[str] = set()  # one document arrives at the barrier once, however often it retries
    real = pipeline.convert_batch

    async def blocking(lib_, doc_, batch, settings_):
        if batch.seq == 0 and doc_ not in arrived:
            arrived.add(doc_)
            await anyio.to_thread.run_sync(functools.partial(started.wait, WAIT))
        assert await _wait_event(release)
        return await real(lib_, doc_, batch, settings_)

    monkeypatch.setattr(pipeline, "convert_batch", blocking)
    ids = [await dbos.start_index("dl", name) for name in names]
    await anyio.to_thread.run_sync(functools.partial(started.wait, WAIT))

    await dbos.delete_library("dl")

    assert await _statuses(ids) == ["CANCELLED", "CANCELLED"]
    assert await Library.names() == [] and not lib.root.exists()
    release.set()
    await _await_terminal(ids)
    await _drain()


async def test_delete_library_rejects_an_unknown_name(dbos) -> None:
    with pytest.raises(LibraryNotFound, match="library not found: ghost"):
        await dbos.delete_library("ghost")


# --- whole-library jobs ------------------------------------------------------------


async def _index_workflows(library: str) -> list[str]:
    """Every document pipeline of one library, finished or not."""
    found = await DBOS.list_workflows_async(
        name=workflows.index_document.__qualname__,
        workflow_id_prefix=f"idx:{library}:",
        load_input=False,
        load_output=False,
    )
    return [s.workflow_id for s in found]


async def test_index_library_workflow_enqueues_every_document_in_pages(dbos, monkeypatch) -> None:
    """D2: "Index all" is a background job that walks the library one page of enqueues at a time,
    so the request costs the same whether the library holds five documents or ten thousand."""
    monkeypatch.setattr(workflows, "BULK_INDEX_PAGE", 2)  # three pages for five documents
    lib = await Library.create("b")
    names = [(await lib.save(f"d{i}.md", f"# d{i}\n\nbody {i}\n".encode())).name for i in range(5)]

    job_id = await dbos.start_index_library("b")

    assert await _wait(job_id) == workflows.BulkResult(done=5, skipped=0)
    queued = await _index_workflows("b")
    assert len(queued) == 5, "one pipeline per document, and no second one for any of them"
    assert [await _wait(i) for i in queued] == ["indexed"] * 5
    assert (
        sorted(
            d.name for d in (await lib.documents_page(PageRequest())).items if d.status == "indexed"
        )
        == names
    )
    progress = await DBOS.get_event_async(job_id, workflows.PROGRESS_EVENT, timeout_seconds=0)
    assert (progress.done, progress.skipped, progress.total, progress.last) == (5, 0, 5, None)
    await _drain()


async def test_index_library_workflow_is_idempotent_on_replay(dbos, monkeypatch) -> None:
    """The page listing is a step and every child id is derived from the bulk job, so a crash
    between two pages re-attaches to the documents already queued instead of queueing them twice."""
    monkeypatch.setattr(workflows, "BULK_INDEX_PAGE", 2)
    lib = await Library.create("b")
    for i in range(5):
        await lib.save(f"d{i}.md", f"# d{i}\n\nbody {i}\n".encode())
    real = workflows.document_page
    pages: list[str | None] = []
    entered, release = threading.Event(), threading.Event()

    async def gated(library: str, after: str | None) -> list[str]:
        pages.append(after)
        if len(pages) == 2:  # the first page is queued; stop the job right here
            entered.set()
            assert await _wait_event(release), "the test never released the listing"
        return await real(library, after)

    monkeypatch.setattr(workflows, "document_page", gated)
    job_id = await dbos.start_index_library("b")
    assert await _wait_event(entered), "the second page never started"
    DBOS.destroy(workflow_completion_timeout_sec=0)  # crash, between two pages
    release.set()
    await dbos.start()  # restart: same application_version, so recovery picks the workflow up

    assert await _wait(job_id) == workflows.BulkResult(done=5, skipped=0)

    queued = await _index_workflows("b")
    assert len(queued) == 5, "the replay re-attached instead of queueing the first page again"
    await _await_terminal(queued)
    await _drain()


async def test_delete_library_workflow_cancels_and_removes(dbos, monkeypatch) -> None:
    """The deletion is a job too: it cancels everything the library has in flight, waits for the
    last running step, and only then drops the rows and the folder."""
    await _use(dbos, workers=4, batch_pages=1)
    lib = await Library.create("wipe")
    done = await lib.save("done.md", MD.encode())
    await _index(dbos, "wipe", done.name)
    slow = await lib.save("slow.pdf", text_pdf(["alpha", "beta"]))
    gate = Gate()
    monkeypatch.setattr(pipeline, "convert_batch", gate.wrap(pipeline.convert_batch))
    in_flight = await dbos.start_index("wipe", slow.name)
    assert await _wait_event(gate.entered)

    bulk_id = await dbos.start_delete_library("wipe")

    assert await _wait(bulk_id) is None
    assert await _statuses([in_flight]) == ["CANCELLED"]
    assert await Library.names() == [] and not lib.root.exists()
    job = await jobs.bulk_job(bulk_id)
    assert (job.kind, job.library, job.status, job.progress) == (
        "delete_library",
        "wipe",
        "SUCCESS",
        None,
    )
    gate.release.set()
    await _await_terminal([in_flight])
    await _drain()


# --- jobs read model ---------------------------------------------------------------


async def _walk_jobs(library: str | None = None, limit: int = 2) -> list[jobs.Job]:
    """Every job of the listing, one page at a time, exactly as the UI's "Load more" reads it."""
    walked: list[jobs.Job] = []
    cursor: str | None = None
    while True:
        page = await jobs.list_jobs(library, limit, cursor)
        walked.extend(page.items)
        cursor = page.next_cursor
        if cursor is None:
            return walked


async def test_list_jobs_filters_by_library_before_it_cuts_the_window(dbos) -> None:
    """A busy library must not push an older one out of the window (A11): the filter is the job
    id's prefix, so the database applies it before it cuts the page."""
    for name in ("noisy", "quiet"):
        lib = await Library.create(name)
        await lib.save("a.md", MD.encode())
    await _index(dbos, "quiet", "a.md")  # oldest job of all
    for _ in range(3):
        await _index(dbos, "noisy", "a.md")

    newest = await jobs.list_jobs(page_size=2)
    assert [j.library for j in newest.items] == ["noisy", "noisy"], "newest first"
    assert newest.next_cursor is not None, "three more jobs behind this page"

    quiet = await jobs.list_jobs("quiet", page_size=2)
    (job,) = quiet.items
    assert (job.library, job.doc, job.status) == ("quiet", "a.md", "SUCCESS")
    assert quiet.next_cursor is None, "the filtered listing has one page"

    # the unfiltered walk reaches the same job, and stops
    assert [j.library for j in await _walk_jobs()] == ["noisy", "noisy", "noisy", "quiet"]


async def test_list_jobs_rejects_a_bad_cursor(dbos) -> None:
    """The cursor is an opaque source and offset into one fixed ordering: anything else is a bad
    request, not an empty page."""
    with pytest.raises(InvalidInput, match="invalid cursor"):
        await jobs.list_jobs(cursor="not-a-cursor")
    with pytest.raises(InvalidInput, match="cursor does not match"):
        await jobs.list_jobs(cursor=paging.encode_cursor(["a.md"], "name", "asc"))
    with pytest.raises(InvalidInput, match="cursor does not match"):  # the cursor of an older build
        await jobs.list_jobs(cursor=paging.encode_cursor([0], jobs.JOB_SORT, jobs.JOB_ORDER))
    for key in ([jobs.LIVE, -1], ["jobs_99999999", 0], [7, 0]):
        with pytest.raises(InvalidInput, match="invalid cursor"):
            await jobs.list_jobs(cursor=paging.encode_cursor(key, jobs.JOB_SORT, jobs.JOB_ORDER))
    for limit in (0, paging.MAX_PAGE_SIZE + 1):
        with pytest.raises(InvalidInput, match=r"page_size must be 1\.\.1000"):
            await jobs.list_jobs(page_size=limit)


async def test_list_kind_pages_on_a_cursor_of_its_own(dbos) -> None:
    """Each section of the jobs view pages through one kind of workflow, newest first, on an
    opaque offset cursor bound to that kind: one from another section would page another history,
    and an unknown kind is a bad request rather than an empty page."""
    await Library.create("pager")
    older = await dbos.start_index_library("pager")
    await _await_terminal([older])  # a second "index all" while one runs is the same job
    newer = await dbos.start_index_library("pager")
    await _await_terminal([newer])

    first = await jobs.list_kind("library", page_size=1)

    assert [row.id for row in first.items] == [newer], "newest first"
    assert first.items[0].title == "index library pager"
    assert first.next_cursor is not None
    second = await jobs.list_kind("library", page_size=1, cursor=first.next_cursor)
    assert [row.id for row in second.items] == [older]
    assert second.next_cursor is None, "the last page ends the walk"

    with pytest.raises(InvalidInput, match="invalid cursor"):
        await jobs.list_kind("download", page_size=1, cursor=first.next_cursor)
    with pytest.raises(InvalidInput, match="unknown job kind 'bogus'"):
        await jobs.list_kind("bogus")
    with pytest.raises(InvalidInput, match=r"page_size must be 1\.\.1000"):
        await jobs.list_kind("library", page_size=0)


def _counted_list_workflows(monkeypatch) -> list[dict]:
    """The keyword arguments of every `DBOS.list_workflows_async` call from here on."""
    calls: list[dict] = []
    real = DBOS.list_workflows_async

    async def counted(**kwargs):
        calls.append(kwargs)
        return await real(**kwargs)

    monkeypatch.setattr(DBOS, "list_workflows_async", counted)
    return calls


async def test_list_jobs_never_loads_inputs(dbos, monkeypatch) -> None:
    """The library and the document come out of the job id, so a page of jobs costs two queries
    and no input payload at all."""
    await (await Library.create("lean")).save("a.md", MD.encode())
    await _index(dbos, "lean", "a.md")
    calls = _counted_list_workflows(monkeypatch)

    (job,) = (await jobs.list_jobs("lean")).items

    assert (job.library, job.doc) == ("lean", "a.md"), "read from the id"
    parent, children = calls
    assert parent["load_input"] is False, "the parent listing never reads inputs"
    assert parent["workflow_id_prefix"] == "idx:lean:" and parent["sort_desc"] is True
    assert children["load_output"] is False, "one query for the children of the whole page"
    assert len(calls) == 2, "no query per job"


async def test_active_index_workflows_use_the_id_prefix(dbos, monkeypatch) -> None:
    """`idx:{library}:` ends in the separator, so library `a` never matches library `ab`."""
    gate = Gate()
    monkeypatch.setattr(pipeline, "convert_batch", gate.wrap(pipeline.convert_batch))
    for name in ("a", "ab"):
        await (await Library.create(name)).save("p.pdf", text_pdf(["alpha"]))
    job_id = await dbos.start_index("ab", "p.pdf")
    assert await _wait_event(gate.entered)

    assert await dbos._active_index_workflows("ab") == [job_id]
    assert await dbos._active_index_workflows("a") == [], (
        "a library whose name is a prefix of another"
    )
    assert await dbos._active_index_workflows("ab", "p.pdf") == [job_id]
    assert await dbos._active_index_workflows("ab", "q.pdf") == [], "another document, same library"

    gate.release.set()
    await _wait(job_id)


async def test_adopt_orphans_runs_in_the_background_on_start(dbos, monkeypatch) -> None:
    """A deep backlog must not hold up the boot: `start` hands adoption to a task on the loop it
    runs on and returns while it is still running."""
    started = threading.Event()
    release = threading.Event()

    async def slow_adopt(batch: int = 0) -> int:
        started.set()
        assert await _wait_event(release), "the test never released the adoption"
        return 3

    monkeypatch.setattr(workflows, "adopt_orphans", slow_adopt)
    await restart_dbos()  # boot again, with adoption blocked

    assert await _wait_event(started), "adoption started"
    assert not release.is_set(), "start did not wait for it"
    adopting = workflows._adoption
    assert adopting is not None and adopting.get_name() == "haskie-adopt"
    assert not adopting.done(), "the boot returned while the adoption is still running"
    # the module holds the only strong reference, so a stuck adoption is still cancellable
    release.set()
    await adopting
    assert adopting.done(), "and it finished once the test released it"


async def test_require_ready_answers_from_the_process_that_loaded_the_model(
    dbos, monkeypatch
) -> None:
    """Every search asks whether the model is ready, and the process that loaded it knows without
    asking the database; a failure is never remembered, because the retry must be visible."""
    attempts: list[str] = []

    async def load_model(kind: str, name: str) -> None:
        attempts.append(name)
        if len(attempts) == 1:
            raise RuntimeError("connection reset")

    monkeypatch.setattr(models, "load_model", load_model)
    user = await save_user_settings(UserSettings(embedding="compact"))
    model_name = _compact_model_name()
    await models.ensure_models(user)
    await _await_terminal([models._model_id("embedding", model_name)])
    with pytest.raises(NotReady, match="failed to load"):
        await models.require_ready("embedding", model_name)  # a failure is never cached

    await models.ensure_models(user)  # deletes the failed record and enqueues the same id again
    await _await_terminal([models._model_id("embedding", model_name)])
    calls = _counted_list_workflows(monkeypatch)

    await models.require_ready("embedding", model_name)
    await models.require_ready("embedding", model_name)
    assert calls == [], "the load ran here, so no search has to look the record up"

    await models.ensure_models(user)  # a healthy record: nothing is deleted or started again
    queries = len(calls)
    await models.require_ready("embedding", model_name)
    assert len(calls) == queries, "and reapplying the settings does not make the model cold"
    assert attempts == [model_name, model_name], "no third load"


async def _seed_jobs(job_id: str, library: str, count: int) -> None:
    """`count` more `index_document` rows in the DBOS history, copied from a real one: only the id
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
                workflow_uuid=f"idx:{library}:d{i}.md:{uuid4().hex}",
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


async def test_list_jobs_stays_fast_over_a_long_history(dbos) -> None:
    """The point of the id prefix: one quiet library's page costs the same with five thousand
    jobs of a busy one behind it as with none."""
    for name in ("noisy", "quiet"):
        await (await Library.create(name)).save("a.md", MD.encode())
    job_id = await dbos.start_index("quiet", "a.md")
    assert await _wait(job_id) == "indexed"
    await _seed_jobs(job_id, "noisy", 5000)

    started = time.perf_counter()
    page = await jobs.list_jobs("quiet", page_size=100)
    elapsed = time.perf_counter() - started

    assert [(j.library, j.doc) for j in page.items] == [("quiet", "a.md")]
    assert page.next_cursor is None
    busy = (await jobs.list_jobs(page_size=100)).items
    assert len(busy) == 100, "the busy library really is in the history"
    assert elapsed < 0.2, f"one page of a quiet library took {elapsed:.3f}s over 5000 jobs"


async def test_list_tasks_reports_stage_slices_still_waiting(dbos, monkeypatch) -> None:
    """A slice is one child workflow with a step per batch, so its tasks come from the step log:
    finished ones from their output, the next one running, the rest enqueued. A stage that has not
    started has no child yet, so it reports no tasks at all. One slice here, so "the rest" is the
    whole remainder of the stage."""
    await _use(dbos, workers=4, batch_pages=1, document_parallelism=1, index_group_parts=1)
    lib = await Library.create("parts")
    doc = await lib.save("p.pdf", text_pdf(["alpha", "beta", "gamma"]))
    gate = Gate(seq=1)
    monkeypatch.setattr(pipeline, "convert_batch", gate.wrap(pipeline.convert_batch))
    job_id = await dbos.start_index("parts", doc.name)
    assert await _wait_event(gate.entered)

    tasks = await jobs.list_tasks(job_id)

    assert [(t.stage, t.seq) for t in tasks] == [("convert", 0), ("convert", 1), ("convert", 2)]
    assert tasks[0].status == "SUCCESS" and tasks[0].result is not None
    assert [t.status for t in tasks[1:]] == ["PENDING", "ENQUEUED"]
    (job,) = (await jobs.list_jobs("parts")).items
    assert (job.tasks_total, job.tasks_done, job.tasks_running) == (3, 1, 1)
    gate.release.set()
    assert await _wait(job_id) == "indexed"
    assert {t.status for t in await jobs.list_tasks(job_id)} == {"SUCCESS"}
    assert len(await jobs.list_tasks(job_id)) == 9


async def test_list_tasks_merges_the_slices_of_a_stage(dbos, monkeypatch) -> None:
    """With the default parallelism the stage is several children, so a batch held up in one slice
    no longer holds up the others: the listing merges their step logs back into plan order."""
    await _use(dbos, workers=4, batch_pages=1, index_group_parts=1)
    lib = await Library.create("sliced")
    doc = await lib.save("p.pdf", text_pdf(["alpha", "beta", "gamma"]))
    gate = Gate(seq=1)
    monkeypatch.setattr(pipeline, "convert_batch", gate.wrap(pipeline.convert_batch))
    job_id = await dbos.start_index("sliced", doc.name)
    assert await _wait_event(gate.entered)

    async def both_slices_done() -> list[jobs.Task] | None:
        found = await jobs.list_tasks(job_id)
        return found if sum(t.status == "SUCCESS" for t in found) == 2 else None

    tasks = await _await(both_slices_done, "the un-gated slices never finished")

    assert [(t.stage, t.seq) for t in tasks] == [("convert", 0), ("convert", 1), ("convert", 2)]
    assert [t.status for t in tasks] == ["SUCCESS", "PENDING", "SUCCESS"], (
        "seq 1 is gated in its own slice; the other two ran without it"
    )
    (job,) = (await jobs.list_jobs("sliced")).items
    assert (job.tasks_total, job.tasks_done, job.tasks_running) == (3, 2, 1)
    assert {t.id for t in tasks} == {f"{job_id}:convert:{i}:{i}" for i in range(3)}
    gate.release.set()
    assert await _wait(job_id) == "indexed"
    assert {t.status for t in await jobs.list_tasks(job_id)} == {"SUCCESS"}


# --- search over a live index ------------------------------------------------------


async def test_search_limit_user_and_library_level(dbos) -> None:
    await save_user_settings(UserSettings(search=SearchSettings(limit=2)))
    lib = await Library.create("lim")
    await lib.set_settings(LibrarySettings(chunk_size=30, chunk_overlap=0))
    doc = await lib.save(
        "m.md", "".join(f"# H{i}\n\ncommon token {i}\n\n" for i in range(6)).encode()
    )
    await _index(dbos, "lim", doc.name)

    assert len(await lib.search("common")) == 2, "user default"
    await lib.set_settings(
        LibrarySettings(chunk_size=30, chunk_overlap=0, search=SearchOverrides(limit=4))
    )
    assert (await lib.info()).search.limit == 4
    assert len(await lib.search("common")) == 4, "library override"
    assert len(await lib.search("common", limit=1)) == 1, "explicit beats both"
    chunking = (await lib.effective_settings()).chunk_size
    assert chunking == 30, "search_limit does not leak into ConversionSettings"

    await session.set_libraries("s", ["lim"])
    assert len(await session.search("s", "common")) == 2, "session cut to user limit"
    assert len(await session.search("s", "common", limit=3)) == 3


async def test_session_search_merges_libraries(dbos) -> None:
    for name in ("a", "b"):
        lib = await Library.create(name)
        await _index(
            dbos, name, (await lib.save("d.md", f"# {name}\n\nshared token {name}\n".encode())).name
        )
    with pytest.raises(LibraryNotFound, match="library not found"):
        await session.set_libraries("s1", ["a", "ghost"])
    await session.set_libraries("s1", ["a", "b"])
    assert {h.library for h in await session.search("s1", "shared", limit=10)} == {"a", "b"}
    assert await session.search("unknown-session", "shared") == []


async def test_search_tolerates_and_rebuilds_outdated_index(dbos) -> None:
    import lancedb

    lib = await Library.create("old")
    doc = await lib.save("g.md", b"# Hi\n\nhello world\n")
    lib.index_dir.mkdir(parents=True)
    old = lancedb.connect(str(lib.index_dir)).create_table(
        "chunks",
        data=[{"doc": "g.md", "chunk_id": 0, "heading": "Hi", "text": "hello world"}],
    )
    old.create_fts_index("text", replace=True)

    assert (await lib.info()).index_outdated is True
    (hit,) = await lib.search("hello")  # read path degrades instead of raising
    assert (hit.heading, hit.source_path, hit.page_start, hit.location) == (
        "Hi",
        lib.relative(lib.file_path("g.md")),  # the row carries no path; the library knows it
        None,
        "g.md L0-0",
    )

    await _index(dbos, "old", doc.name)  # write path drops the old table and rebuilds
    assert (await lib.info()).index_outdated is False
    (hit,) = await lib.search("hello")
    assert hit.source_path == lib.relative(lib.file_path("g.md")) and hit.line_start == 1


async def test_delete_cascades_rows_and_files(dbos) -> None:
    lib = await Library.create("casc")
    a = await lib.save("a.md", b"# a\n")
    b = await lib.save("b.md", b"# b\n")
    for d in (a, b):
        await _index(dbos, "casc", d.name)
        await lib.ensure_preview(d.name)

    async def documents() -> int:
        async with db.connect() as conn:
            cursor = await conn.execute("select count(*) from documents where library = 'casc'")
            counted = await cursor.fetchone()
            assert counted is not None, "count(*) always returns a row"
            return counted[0]

    assert await documents() == 2
    await lib.remove_document("a.md")
    assert await documents() == 1
    assert not lib.file_path("a.md").exists()
    assert not lib.markdown_path("a.md").exists()
    assert not lib.parts_dir("a.md").exists() and not lib.preview_dir("a.md").exists()
    assert {h.doc for h in await lib.search("b", limit=5)} == {"b.md"}

    await lib.delete()
    assert await documents() == 0, "library delete cascades to documents"
    assert not lib.root.exists()


async def test_home_is_portable(dbos, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Move the whole home directory: DB, files and index keep working, paths still resolve."""
    from haskie import home

    lib = await Library.create("port")
    doc = await lib.save("p.md", b"# P\n\nportable text\n")
    await _index(dbos, "port", doc.name)

    moved = tmp_path / "elsewhere"
    shutil.copytree(home.HOME, moved)
    monkeypatch.setattr(home, "HOME", moved)
    monkeypatch.setattr(home, "LIBRARY_ROOT", moved / "library")
    monkeypatch.setattr(home, "DB_FILE", moved / "haskie.db")
    monkeypatch.setattr(db, "_migrated", set())

    again = await Library.get("port")
    (hit,) = await again.search("portable")
    assert hit.home == str(moved)
    assert (moved / hit.source_path).read_bytes().startswith(b"# P")
    assert (moved / hit.markdown_path).exists()


# --- model lifecycle ---------------------------------------------------------------


async def test_no_model_is_required_for_full_text_only(dbos) -> None:
    assert await models.ensure_models(UserSettings(embedding="none")) == []
    assert await models.model_statuses(UserSettings(embedding="none")) == []


def _compact_model_name() -> str:
    model = UserSettings(embedding="compact").embedding_model
    assert model is not None
    return model.name


@pytest.mark.parametrize(
    ("name", "outcome", "state", "match"),
    [
        ("loaded in this process", "ready", "ready", None),
        ("download failed", "error", "error", "failed to load: RuntimeError: no such model"),
        ("never required before", "missing", "pending", "is not loaded yet"),
        ("still downloading", "blocked", "loading", "is downloading .job dl:embedding:"),
        ("downloaded, caches cold", "cold", "loading", "is loading in this process"),
    ],
)
async def test_model_state_decides_whether_search_may_run(
    dbos, monkeypatch, name: str, outcome: str, state: str, match: str | None
) -> None:
    """A download record that says SUCCESS is not enough: the model lives in the caches of the
    process that loaded it, so a boot that inherits the record still reports "loading" until it
    has warmed the model itself (A8).

    `load_model` is replaced rather than `embed.warm`, so no case waits out the download retry
    schedule (five attempts, five seconds apart)."""
    blocked = threading.Event()

    async def load_model(kind: str, name_: str) -> None:
        if outcome == "error":
            raise RuntimeError("no such model")
        if outcome == "blocked":
            assert await _wait_event(blocked)

    monkeypatch.setattr(models, "load_model", load_model)
    user = await save_user_settings(UserSettings(embedding="compact"))
    model_name = _compact_model_name()

    if outcome != "missing":
        await models.ensure_models(user)
        if outcome != "blocked":
            await _await_terminal([models._model_id("embedding", model_name)])
    if outcome == "cold":
        models._ready.clear()  # the record of a boot whose caches this process does not have

    try:
        (status,) = await models.model_statuses(user)
        assert (status.kind, status.name, status.state) == ("embedding", model_name, state), name
        if match is None:
            await models.require_ready("embedding", model_name)  # no raise
        else:
            with pytest.raises(NotReady, match=match):
                await models.require_ready("embedding", model_name)
    finally:
        blocked.set()
        await _await_terminal([models._model_id("embedding", model_name)])


async def test_ensure_models_retries_a_model_that_failed(dbos, monkeypatch) -> None:
    attempts: list[str] = []

    async def load_model(kind: str, name: str) -> None:
        attempts.append(name)
        if len(attempts) == 1:
            raise RuntimeError("connection reset")

    monkeypatch.setattr(models, "load_model", load_model)
    user = await save_user_settings(UserSettings(embedding="compact"))
    model_name = _compact_model_name()

    await models.ensure_models(user)
    await _await_terminal([models._model_id("embedding", model_name)])
    assert (await models.model_statuses(user))[0].state == "error"

    await models.ensure_models(user)  # retries under the same id instead of leaving it failed
    await _await_terminal([models._model_id("embedding", model_name)])

    assert (await models.model_statuses(user))[0].state == "ready"
    assert attempts == [model_name, model_name], "the retry really called the loader again"
    assert len(await DBOS.list_workflows_async(name=models.ensure_model.__qualname__)) == 1, (
        "same workflow id"
    )


@pytest.mark.network
async def test_embed_stage_precomputes_vectors_and_hybrid_search_uses_them(dbos) -> None:
    user = await save_user_settings(
        UserSettings(embedding="compact", search=SearchSettings(reranker="cross-encoder"))
    )
    await models.ensure_models(user)
    for kind, name in await models._required(user):
        await wait_for(models._model_id(kind, name))
    assert {(m.kind, m.state) for m in await models.model_statuses(user)} == {
        ("embedding", "ready"),
        ("reranker", "ready"),
    }
    lib = await Library.create("vec")
    await lib.set_settings(LibrarySettings(chunk_size=40, chunk_overlap=0))
    doc = await lib.save(
        "v.md", b"# Cats\n\nCats purr and chase mice.\n\n# Finance\n\nBonds yield interest.\n"
    )
    await _index(dbos, "vec", doc.name)

    table = await (await lib.index())._existing()
    assert table is not None and "vector" in (await table.schema()).names
    records = sorted((await table.to_arrow()).to_pylist(), key=lambda r: (r["part"], r["chunk_id"]))
    assert [r["heading"] for r in records] == ["Cats", "Finance"]
    assert all(len(r["vector"]) == 384 for r in records)
    assert (await lib.search("kitten"))[0].heading == "Cats", "semantic hit without lexical overlap"

    # search options: every mode/fusion answers; fts alone cannot find "kitten"
    assert (await lib.search("kitten", mode="vector"))[0].heading == "Cats"
    assert await lib.search("kitten", mode="fts") == []
    assert (await lib.search("bonds", mode="hybrid", fusion="linear"))[0].heading == "Finance"
    lexical_only = await lib.search("bonds", fusion="linear", vector_weight=0.0, bm25_weight=1.0)
    assert lexical_only[0].heading == "Finance"
    assert (await lib.search("kitten", fusion="rrf", limit=1))[0].heading == "Cats"

    # cross-encoder reranker works on top of any mode, including vector-only and fts
    for mode in ("vector", "hybrid"):
        hits = await lib.search("kitten", mode=mode, reranker="cross-encoder", candidates=10)
        assert hits[0].heading == "Cats" and hits[0].score > hits[1].score, mode
    assert (await lib.search("bonds", mode="fts", reranker="cross-encoder"))[0].heading == "Finance"


@pytest.mark.network
async def test_models_are_idempotent_and_fail_fast_when_missing(dbos, monkeypatch) -> None:
    plain = UserSettings(embedding="none")
    assert await models.ensure_models(plain) == [], "nothing required for full-text only"
    with pytest.raises(NotReady, match="not loaded yet"):
        await models.require_ready("embedding", "BAAI/bge-small-en-v1.5")

    wanted = UserSettings(embedding="compact")
    (status,) = await models.ensure_models(wanted)
    assert status.state in ("loading", "ready")
    await wait_for(models._model_id("embedding", status.name))
    assert (await models.model_statuses(wanted))[0].state == "ready"
    before = len(await DBOS.list_workflows_async(name=models.ensure_model.__qualname__))
    await models.ensure_models(wanted)
    assert len(await DBOS.list_workflows_async(name=models.ensure_model.__qualname__)) == before, (
        "no second load"
    )

    # SearchSettings only accepts known reranker ids, so add the missing one to the allowed set
    monkeypatch.setattr(settings, "RERANKER_MODELS", (*settings.RERANKER_MODELS, "nope/x"))
    broken = UserSettings(
        embedding="none", search=SearchSettings(reranker="cross-encoder", reranker_model="nope/x")
    )
    await models.ensure_models(broken)
    with pytest.raises(HaskieError):  # the model does not exist
        await wait_for(models._model_id("reranker", "nope/x"))
    (status,) = await models.model_statuses(broken)
    assert (status.kind, status.state) == ("reranker", "error") and status.error
    with pytest.raises(NotReady, match="failed to load"):
        await models.require_ready("reranker", "nope/x")


@pytest.mark.network
async def test_search_rejects_a_query_while_the_embedding_model_loads(dbos, monkeypatch) -> None:
    """The index has vectors, so a hybrid query needs the model; a request must fail fast with
    503 semantics rather than block on a download."""
    user = await save_user_settings(UserSettings(embedding="compact"))
    await models.ensure_models(user)
    await _await_terminal([models._model_id("embedding", _compact_model_name())])
    lib = await Library.create("busy")
    doc = await lib.save("g.md", MD.encode())
    await _index(dbos, "busy", doc.name)
    assert await lib.search("lancedb"), "ready model answers"

    monkeypatch.setattr(models, "_ready", set())  # as after a restart: caches are cold

    with pytest.raises(NotReady, match="is loading in this process"):
        await lib.search("lancedb")


# --- settings applied to the runtime -----------------------------------------------


async def test_an_unreadable_settings_row_does_not_stop_the_boot(dbos, monkeypatch) -> None:
    """The loader already tolerates a row another build wrote, so `start()` applies defaults and
    `/api/status` reports the problem (A13)."""
    await save_user_settings(UserSettings(embedding="compact"))
    async with db.connect() as conn:
        await conn.execute(
            'update settings set json = \'{"pipeline": {"cpu_budget": 0}}\' where id = 1'
        )
    settings.invalidate()  # the row was written behind the loader's back, as another build would
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
    assert _events(caplog) == ["settings_invalid_at_boot"]


async def test_cancel_wait_gives_up_instead_of_blocking_forever(dbos, monkeypatch, caplog) -> None:
    """Cancellation is cooperative: a step that ignores it must not hold a request open."""
    lib = await Library.create("stuck")
    doc = await lib.save("p.pdf", text_pdf(["alpha"]))
    gate = Gate()
    monkeypatch.setattr(pipeline, "convert_batch", gate.wrap(pipeline.convert_batch))
    job_id = await dbos.start_index("stuck", doc.name)
    assert await _wait_event(gate.entered)

    async def cancel_workflow(*args, **kwargs) -> None:  # a cancel that never lands
        return None

    monkeypatch.setattr(DBOS, "cancel_workflow_async", cancel_workflow)

    with caplog.at_level("WARNING"):
        await workflows._cancel_and_wait([job_id], wait_seconds=0)

    assert _events(caplog) == ["cancel_wait_timeout"]
    assert await _statuses([job_id]) == ["PENDING"], "still running, and reported as such"
    gate.release.set()
    await _await_terminal([job_id])


async def test_step_outcome_reads_a_plain_step_output(dbos) -> None:
    """Steps that predate `BatchResult` (and every non-batch step) return their value directly."""
    assert jobs._step_outcome({"output": 5, "error": None}) == (5, None)
    assert jobs._step_outcome({"output": None, "error": RuntimeError("boom")}) == (None, "boom")
    assert jobs._step_outcome({"output": "text", "error": None}) == (None, None)


@pytest.mark.parametrize(
    ("name", "user", "library_rerankers", "expected"),
    [
        ("full text only", UserSettings(embedding="none"), [], []),
        (
            "an embedding profile",
            UserSettings(embedding="compact"),
            [],
            [("embedding", "BAAI/bge-small-en-v1.5")],
        ),
        (
            "an embedding profile and a cross-encoder",
            UserSettings(embedding="compact", search=SearchSettings(reranker="cross-encoder")),
            [],
            [
                ("embedding", "BAAI/bge-small-en-v1.5"),
                ("reranker", "Xenova/ms-marco-MiniLM-L-6-v2"),
            ],
        ),
        (
            "a library override nobody else asks for",
            UserSettings(embedding="none"),
            ["BAAI/bge-reranker-base"],
            [("reranker", "BAAI/bge-reranker-base")],
        ),
        (
            "the same model at both levels is wanted once",
            UserSettings(embedding="none", search=SearchSettings(reranker="cross-encoder")),
            ["Xenova/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"],
            [
                ("reranker", "Xenova/ms-marco-MiniLM-L-6-v2"),
                ("reranker", "BAAI/bge-reranker-base"),
            ],
        ),
    ],
)
async def test_required_models_follow_the_settings(
    dbos, name: str, user: UserSettings, library_rerankers: list[str], expected: list
) -> None:
    assert await models._required(user, library_rerankers) == expected, name


async def test_library_reranker_override_is_downloaded(dbos, monkeypatch) -> None:
    """A reranker chosen for one library is a model the installation needs: nothing else would
    ever fetch it, and the first search of that library would fail with "not loaded yet"."""
    loaded: list[str] = []
    monkeypatch.setattr(embed, "warm_reranker", lambda name, accelerator: loaded.append(name))
    override = "jinaai/jina-reranker-v1-turbo-en"
    lib = await Library.create("picky")
    await lib.set_settings(
        LibrarySettings(search=SearchOverrides(reranker="cross-encoder", reranker_model=override))
    )
    user = await save_user_settings(UserSettings(embedding="none"))

    assert await Library.reranker_overrides() == [override]
    (status,) = await models.ensure_models(user)

    assert (status.kind, status.name) == ("reranker", override)
    workflow_id = models._model_id("reranker", override)
    assert workflow_id.startswith("dl:reranker:")
    await _await_terminal([workflow_id])
    assert loaded == [override], "the download workflow really called the loader"
    (download,) = await jobs.list_downloads()
    assert (download.id, download.kind, download.model) == (workflow_id, "reranker", override)
    assert (download.status, download.error) == ("SUCCESS", None)
    await models.require_ready("reranker", override)  # no raise: the library can be searched


async def test_downloads_list_one_row_per_required_model(dbos, monkeypatch) -> None:
    """The list is the read model of the `ensure_model` workflows, so it has one row per model
    the settings ask for, whatever state it is in."""
    monkeypatch.setattr(embed, "warm", lambda name, accelerator: None)
    monkeypatch.setattr(embed, "warm_reranker", lambda name, accelerator: None)
    user = await save_user_settings(
        UserSettings(embedding="compact", search=SearchSettings(reranker="cross-encoder"))
    )

    await models.ensure_models(user)
    await _await_terminal(
        [models._model_id(kind, name) for kind, name in await models._required(user)]
    )

    downloads = await jobs.list_downloads()
    assert {(d.kind, d.model) for d in downloads} == {
        ("embedding", _compact_model_name()),
        ("reranker", "Xenova/ms-marco-MiniLM-L-6-v2"),
    }, "one row per required model, kind and name read out of the workflow id"
    assert {d.status for d in downloads} == {"SUCCESS"}
    assert all(d.warm for d in downloads), "loaded here, so this process can search with them"
    assert all(d.created_at > 0 and d.error is None for d in downloads)
    assert [d.created_at for d in downloads] == sorted(
        (d.created_at for d in downloads), reverse=True
    ), "newest first"


async def _until(condition, message: str, timeout: float = WAIT) -> None:
    """Wait for something a background task does; polled, because no result handle carries it.
    `condition` is a coroutine function."""
    deadline = time.monotonic() + timeout
    while not await condition():
        assert time.monotonic() < deadline, message
        await anyio.sleep(workflows.TASK_POLL)


async def test_restart_does_not_create_a_second_download_record(dbos, monkeypatch) -> None:
    """The files stay on disk and the record is durable, so a restart reuses both: one row per
    model, however often the dev server reloads. The id used to carry a boot token, which gave
    every restart a download of its own and the Downloads list the same model over and over."""
    monkeypatch.setattr(embed, "warm", lambda name, accelerator: None)
    user = await save_user_settings(UserSettings(embedding="compact"))
    workflow_id = models._model_id("embedding", _compact_model_name())
    await models.ensure_models(user)
    await _await_terminal([workflow_id])

    models._ready.clear()  # a new process has the files, not the caches
    await restart_dbos()
    await workflows.apply_settings(user)  # the boot applies them once; twice must change nothing

    (download,) = await jobs.list_downloads()
    assert download.id == workflow_id, "the same record, not one per boot"
    assert download.status == "SUCCESS", "and it is not downloaded again"

    async def warmed() -> bool:
        return (await models.model_statuses(user))[0].state == "ready"

    await _until(warmed, "the model was never warmed")
    assert (await jobs.list_downloads())[0].warm is True


async def test_a_downloaded_model_is_warmed_after_restart_before_search_uses_it(
    dbos, monkeypatch
) -> None:
    """Warming is a local read of the disk cache, but it still takes seconds, so the boot hands it
    to a background thread. A search that arrives first fails fast with 503 semantics (A8)."""
    warming, release = threading.Event(), threading.Event()

    def warm(name, accelerator) -> None:
        # sync, and run in a worker thread through `cpu.on_cpu`: blocking here blocks no loop
        warming.set()
        assert release.wait(timeout=WAIT), "the test never released the warm-up"

    async def load_model(kind, name) -> None:  # the download itself
        return None

    monkeypatch.setattr(models, "load_model", load_model)
    monkeypatch.setattr(embed, "warm", warm)
    user = await save_user_settings(UserSettings(embedding="compact"))
    model_name = _compact_model_name()
    await models.ensure_models(user)
    await _await_terminal([models._model_id("embedding", model_name)])

    models._ready.clear()  # a new process has the files, not the caches
    await restart_dbos()

    assert await _wait_event(warming), "the boot warms the model it already downloaded"
    with pytest.raises(NotReady, match="is loading in this process"):
        await models.require_ready("embedding", model_name)
    assert (await models.model_statuses(user))[0].state == "loading"

    release.set()

    async def loaded() -> bool:
        return models.is_warm(models._model_id("embedding", model_name))

    await _until(loaded, "never warmed")
    await models.require_ready("embedding", model_name)  # no raise: the search may run now


# --- S3: cross-library session search ----------------------------------------------
#
# One query embedding and one model check for the whole fan-out, the libraries read in parallel,
# and one cross-encoder pass over the merged candidates instead of one per library.


async def test_session_search_embeds_once_and_checks_the_model_once(dbos, monkeypatch) -> None:
    """Three libraries, one embedding: the query used to be embedded (and the model checked)
    once per library."""
    from haskie.chunk import split
    from haskie.index import Row
    from haskie.settings import PROFILES

    await save_user_settings(UserSettings(embedding="compact"))
    compact = PROFILES["compact"]
    assert compact is not None
    for name in ("a", "b", "c"):
        lib = await Library.create(name)
        (chunk_,) = split(f"# {name}\n\nshared token {name}\n", settings.ConversionSettings())
        index = lib.index_with(compact)
        row = Row(chunk=chunk_, vector=[0.1] * compact.dims)
        await index.add_parts("d.md", "files/d.md", "markdown/d.md.md", _one_part(0, [row]))
        await index.finish()
    await session.set_libraries("s", ["a", "b", "c"])
    embedded: list[str] = []
    checked: list[tuple[str, str]] = []

    def fake_embed(model, text: str) -> list[float]:
        embedded.append(text)
        return [0.1] * model.dims

    async def require_ready(kind: str, model: str) -> None:
        checked.append((kind, model))

    monkeypatch.setattr(session, "embed_query", fake_embed)
    monkeypatch.setattr(models, "require_ready", require_ready)

    hits = await session.search("s", "shared", limit=10)

    assert {h.library for h in hits} == {"a", "b", "c"}
    assert embedded == ["shared"], "one embedding for the whole fan-out"
    assert checked == [("embedding", compact.name)], "one check, not one per library"


async def test_session_search_reranks_once_over_the_merge(dbos, monkeypatch) -> None:
    """The cross-encoder sees the merged candidates of every library once, and its score is the
    score of the returned hits."""
    await save_user_settings(
        UserSettings(search=SearchSettings(limit=2, candidates=4, reranker="cross-encoder"))
    )
    for name, token in (("a", "alpha"), ("b", "beta")):
        lib = await Library.create(name)
        await lib.set_settings(LibrarySettings(chunk_size=30, chunk_overlap=0))
        body = "".join(f"# H{i}\n\nshared {token} {i}\n\n" for i in range(3))
        await _index(dbos, name, (await lib.save("d.md", body.encode())).name)
    await session.set_libraries("s", ["a", "b"])
    calls: list[list[str]] = []

    async def require_ready(kind: str, model: str) -> None:
        return None

    monkeypatch.setattr(models, "require_ready", require_ready)

    def fake_rerank(model: str, accelerator: str, query: str, texts: list[str]) -> list[float]:
        calls.append(texts)
        return [float(i) for i in range(len(texts))]

    monkeypatch.setattr(session, "rerank_scores", fake_rerank)

    hits = await session.search("s", "shared")

    assert len(calls) == 1, "one cross-encoder pass, not one per library"
    (texts,) = calls
    assert len(texts) == 4, "the merge is cut to `candidates` before it is rescored"
    assert any("alpha" in t for t in texts), "candidates from both libraries"
    assert any("beta" in t for t in texts)
    assert [h.text for h in hits] == [texts[-1], texts[-2]], "best cross-encoder score first"
    assert [h.score for h in hits] == [3.0, 2.0], "the hit carries the cross-encoder score"


async def test_session_search_propagates_a_broken_library(dbos, monkeypatch, caplog) -> None:
    """A library that cannot answer must not be silently dropped: an empty result reads as
    "no match", which is a different answer."""
    from haskie.index import LibraryIndex

    for name in ("a", "b"):
        lib = await Library.create(name)
        await _index(
            dbos, name, (await lib.save("d.md", f"# {name}\n\nshared token\n".encode())).name
        )
    await session.set_libraries("s", ["a", "b"])

    async def boom(self, query, vector, settings_, limit):
        raise RuntimeError("index unreadable")

    monkeypatch.setattr(LibraryIndex, "search_rows", boom)

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="index unreadable"):
            await session.search("s", "shared")

    assert "session_library_search_failed" in _events(caplog)


# --- daily maintenance -------------------------------------------------------------


async def test_start_registers_the_daily_maintenance_schedule(dbos) -> None:
    """Like the archive schedule, the definition lives in the system database: a second boot
    finds the one the first wrote instead of adding another."""

    async def registered() -> list:
        return [
            s
            for s in await DBOS.list_schedules_async()
            if s["schedule_name"] == workflows.MAINTENANCE_SCHEDULE
        ]

    (schedule,) = await registered()
    assert (schedule["schedule"], schedule["queue_name"]) == (
        workflows.MAINTENANCE_CRON,
        workflows.MAINTENANCE_QUEUE,
    )
    assert "daily_maintenance" in schedule["workflow_name"]

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
