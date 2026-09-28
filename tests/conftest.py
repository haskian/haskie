"""Fixtures shared by every test module: the temp home, and the DBOS runtime on top of it."""

import functools
import shutil
import signal
import sqlite3
import threading
import time
from collections.abc import AsyncIterator, Iterator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import anyio.to_thread
import pytest
from dbos import WorkflowStatusString

from haskie.indexing.segment import PieceType

if TYPE_CHECKING:  # every helper below imports haskie when it runs, not when pytest collects
    from haskie.catalogue.catalogue import EmbeddingModel
    from haskie.collection.index import CollectionIndex, Hit
    from haskie.indexing.chunk import Chunk
    from haskie.search.collapse import Scan
    from haskie.settings import SearchOverrides, SearchSettings

TEARDOWN_GRACE_SECONDS = 2.0  # how long a cancelled step may still be running at teardown
TEARDOWN_POLL_SECONDS = 0.05
DELAY_SWEEP_SECONDS = 0.05  # how often a debounced workflow is promoted (see `_sweep_delayed`)
# The first run of a home that needs no model: full-text search, no reranker. The suite downloads
# nothing, so a test that searches starts here unless it is about a model.
NO_MODELS = {"profile": "none", "search": {"reranker": "none"}}

# Still on its way, including a debounced run waiting out its period (DELAYED). Only the suite
# waits on that: the app counts `dbos_names.ACTIVE_STATUS`, where a debounce is not yet work.
WAITING_STATUS = [
    WorkflowStatusString.DELAYED.value,
    WorkflowStatusString.ENQUEUED.value,
    WorkflowStatusString.PENDING.value,
]


@pytest.fixture(scope="session", autouse=True)
def logging_configured() -> None:
    """One structlog/stdlib configuration for the whole run, as the app does at startup."""
    from haskie import logs

    logs.configure()


@pytest.fixture(scope="session", autouse=True)
def fast_runtime() -> None:
    """The production settings the suite cannot afford, set once before anything boots DBOS.

    The queue poll intervals ship at 1 s / 0.25 s and are paid at every queue hand-off;
    `workflows.start` reads them when it registers the queues, so assigning here is enough.
    `CONVERT_WORKERS = 0` extracts PDFs inline: a process pool per xdist worker costs more to
    start than the tests would save. `test_cpu_pool` is the one module that puts that back.
    ONNX Runtime's telemetry goes off as the app turns it off (`embed.onnx_runtime`): some tests
    import fastembed without passing through the app, and a worker exiting mid-upload crashes.

    The step retry intervals are not here: DBOS copies them into the decorator at import, so the
    two retry tests pay the real wait. `-n auto` absorbs it.
    """
    from haskie import cpu
    from haskie.indexing import embed, workflows

    embed.onnx_runtime()
    workflows.OPERATION_POLL = 0.02
    workflows.TASK_POLL = 0.02
    cpu.CONVERT_WORKERS = 0


@pytest.fixture
def server_handler() -> Iterator[list[int]]:
    """Stand in for the server: a Python handler on SIGINT and SIGTERM that records each call, in
    the order the calls came. Whatever a test installs over it is put back afterwards."""
    from haskie import shutdown

    seen: list[int] = []
    found = {number: signal.getsignal(number) for number in shutdown.SHUTDOWN_SIGNALS}
    for number in shutdown.SHUTDOWN_SIGNALS:
        signal.signal(number, lambda n, _frame: seen.append(n))
    yield seen
    for number, handler in found.items():
        signal.signal(number, handler)


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """The backend every `pytest.mark.anyio` test runs on. asyncio only: that is what Litestar and
    DBOS run on. Session-scoped so session fixtures may be async too."""
    return "asyncio"


def _use_home(path: Path) -> Path:
    """Point this process at `path`, the way `haskie --home` does, and make the directories a boot
    would. Returns the root that was in use, for the caller to restore with `home.use` afterwards.

    Restoring is paths only, never directories: the root a test started from may be the user's
    real home, which the suite must never create anything in.
    """
    from haskie import home

    previous = home.HOME
    home.use(path)
    home.ensure_home_sync()
    return previous


def _drop_caches(patch: pytest.MonkeyPatch) -> None:
    """Drop the process caches that would otherwise answer from another home."""
    from haskie import db, settings
    from haskie.catalogue import catalogue
    from haskie.indexing import models

    patch.setattr(db, "_migrated", set())
    patch.setattr(catalogue, "_embedders", {})
    patch.setattr(settings, "_state", None)
    # a loaded model is process state, and the process outlives the test that loaded it
    patch.setattr(models, "_ready", set())
    patch.setattr(models, "_warming", set())


def forget_settings() -> None:
    """Make the next load read the `settings` row again. For a test that wrote the row with SQL,
    which the process cache (`settings._state`) cannot see."""
    from haskie import settings

    settings._state = None


@pytest.fixture(scope="session")
def template_home(tmp_path_factory: pytest.TempPathFactory, fast_runtime: None) -> Path:
    """A home booted once, so every test that needs DBOS copies its database instead of building
    one: the migrations, the DBOS system schema, the registered queues and the cron schedules are
    all in the file already.

    The queues are the reason this is worth a fixture. DBOS's queue manager discovers queues by
    listing them from the system database once a second, so queues registered after
    `DBOS.launch()` — which is when `apply_settings` can register them — are only served from the
    next sweep, and the first dequeue of every test waits out that second. Rows that are in the
    file before launch are found by the first sweep instead.

    Sync, so one worker of an xdist run builds it and the others copy the file, as before. The
    boot is a coroutine now, so it gets a loop of its own through `anyio.run`; nothing is running
    when a session fixture executes, which is exactly what `anyio.run` needs.
    """
    from dbos import DBOS

    from haskie import home
    from haskie.indexing import workflows

    path = tmp_path_factory.mktemp("template-home")
    with pytest.MonkeyPatch.context() as patch:
        previous = _use_home(path)
        _drop_caches(patch)
        try:
            anyio.run(workflows.start)  # migrations, the DBOS schema, queues and schedules
            # a boot enqueues nothing on an empty home, but a test starts from an empty history.
            # Sync calls: `anyio.run` has returned, so no event loop is running in this thread and
            # DBOS's `check_async` guard is satisfied.
            for workflow in DBOS.list_workflows(load_input=False, load_output=False):
                DBOS.delete_workflow(workflow.workflow_id)
            DBOS.destroy(workflow_completion_timeout_sec=0)
        finally:
            home.use(previous)  # this ran inside the test that asked for it
    connection = sqlite3.connect(path / "haskie.db")
    try:
        connection.execute("pragma wal_checkpoint(truncate)")  # so one file carries everything
    finally:
        connection.close()
    return path


@pytest.fixture(autouse=True)
def haskie_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A fresh temp home, with nothing in it: no database file, no migrations."""
    from haskie import home

    previous = _use_home(tmp_path)
    _drop_caches(monkeypatch)
    yield tmp_path
    home.use(previous)  # the temp directory goes; leave no module pointing into it


@pytest.fixture
def seeded_home(haskie_home: Path, template_home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`haskie_home` with the booted template's database file in it (see `template_home`)."""
    shutil.copyfile(template_home / "haskie.db", haskie_home / "haskie.db")
    _drop_caches(monkeypatch)  # the template ran with a home of its own
    return haskie_home


def _sweep_delayed(stop: threading.Event) -> None:
    """Promote debounced workflows whose delay has expired, at the interval everything else here
    polls at. DBOS's queue manager does exactly this on its own sweep, once a second, which is
    longer than most of these tests take. The delay still has to have expired - only the sweep's
    granularity goes away, not the debounce.

    Runs while DBOS is up and gives up quietly otherwise: a test may be restarting it, and the
    next sweep finds it again."""
    from dbos import _dbos as dbos_module

    while not stop.wait(DELAY_SWEEP_SECONDS):
        instance = dbos_module._dbos_global_instance
        if instance is None or not instance._initialized:
            continue
        try:
            instance._sys_db.transition_delayed_workflows()
        except Exception:  # raised when DBOS is torn down between the check above and the call
            continue


@pytest.fixture
async def dbos(seeded_home: Path):
    """DBOS launched on the test home's SQLite file; torn down after the test.

    Async, so the boot, the teardown and the test body share one event loop: an async workflow
    enqueued by the test is enqueued from that loop, while DBOS runs it on its own background
    loop (see `workflows.start`)."""
    from haskie.indexing import workflows

    await workflows.start()
    stop_sweeping = threading.Event()
    sweeper = threading.Thread(target=_sweep_delayed, args=(stop_sweeping,), daemon=True)
    sweeper.start()
    yield workflows
    stop_sweeping.set()
    sweeper.join(timeout=TEARDOWN_GRACE_SECONDS)
    await stop_dbos()


async def stop_dbos() -> None:
    """Shut DBOS down without waiting out a grace period counted in whole seconds.

    The app shuts down through `workflows.stop`, which gives in-flight steps ten seconds;
    `DBOS.destroy` sleeps a whole second before it even looks, and anything the test left behind
    keeps it looking. Cancelling that first says the same thing in one round trip. A step still
    running in a worker thread is waited out here: one that outlives `destroy` blocks interpreter
    exit.
    """
    from dbos import DBOS
    from dbos import _dbos as dbos_module

    instance = dbos_module._dbos_global_instance
    if instance is None:  # the test destroyed it itself
        return
    unfinished = [
        found.workflow_id
        for found in await DBOS.list_workflows_async(
            status=WAITING_STATUS, load_input=False, load_output=False
        )
    ]
    if unfinished:
        await DBOS.cancel_workflows_async(unfinished)
    deadline = time.monotonic() + TEARDOWN_GRACE_SECONDS
    while instance._active_workflows_set.activeList() and time.monotonic() < deadline:
        await anyio.sleep(TEARDOWN_POLL_SECONDS)
    await anyio.to_thread.run_sync(partial(DBOS.destroy, workflow_completion_timeout_sec=0))


async def wait_for(workflow_id: str):
    """The result of one workflow, polled at the interval the app itself uses. `get_result()`
    defaults to a whole second, which is longer than most of these workflows take."""
    from dbos import DBOS

    from haskie.indexing import workflows

    handle = await DBOS.retrieve_workflow_async(workflow_id)
    return await handle.get_result(polling_interval_sec=workflows.TASK_POLL)


WAIT = 30.0  # generous: every wait in the suite is released by another thread, never by a timer


async def wait_event(event: threading.Event, timeout: float = WAIT) -> bool:
    """Wait for a `threading.Event` without blocking the caller's loop. Two loops are involved -
    the test's and DBOS's background one - so the blocking wait goes to a worker thread."""
    return await anyio.to_thread.run_sync(functools.partial(event.wait, timeout))


async def await_terminal(workflow_ids: list[str]) -> None:
    """Drain before teardown: a step outliving `DBOS.destroy()` blocks interpreter exit."""
    for workflow_id in workflow_ids:
        try:
            await wait_for(workflow_id)
        except Exception:  # the outcome is asserted where it matters; this only drains
            pass


async def until(condition, message: str, timeout: float = WAIT) -> None:
    """Wait for something a background task does; polled, because no result handle carries it.
    `condition` is a coroutine function."""
    from haskie.indexing import workflows

    deadline = time.monotonic() + timeout
    while not await condition():
        assert time.monotonic() < deadline, message
        await anyio.sleep(workflows.TASK_POLL)


async def one_part(part: int, rows) -> AsyncIterator[tuple[int, list]]:
    """`CollectionIndex.add_parts` consumes an async iterator: the pipeline decodes one row group
    of the cache file at a time and awaits each read (see `embed_cache.read`)."""
    yield part, rows


def counted_list_workflows(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """The keyword arguments of every `DBOS.list_workflows_async` call from here on."""
    from dbos import DBOS

    calls: list[dict] = []
    real = DBOS.list_workflows_async

    async def counted(**kwargs):
        calls.append(kwargs)
        return await real(**kwargs)

    monkeypatch.setattr(DBOS, "list_workflows_async", counted)
    return calls


async def restart_dbos() -> None:
    """Stop DBOS and boot it again, as restarting the app would. Cheap: the system database it
    reopens already carries the queues and the schedules, so the boot only reads them."""
    from dbos import DBOS

    from haskie.indexing import workflows

    await anyio.to_thread.run_sync(partial(DBOS.destroy, workflow_completion_timeout_sec=0))
    await workflows.start()


def text_pdf(pages: list[str | None]) -> bytes:
    """Minimal PDF: one Helvetica line per page; None = blank page (needs OCR)."""
    objs: list[str] = ["<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", ""]
    page_ids: list[int] = []
    for text in pages:
        stream = "" if text is None else f"BT /F1 18 Tf 40 150 Td ({text}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Contents {len(objs)} 0 R "
            "/Resources << /Font << /F1 1 0 R >> >> >>"
        )
        page_ids.append(len(objs))
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objs[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>"
    objs.append("<< /Type /Catalog /Pages 2 0 R >>")
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root {len(objs)} 0 R >>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    return out


async def document_names() -> list[str]:
    """Every document name, in name order: what a test that asserts on the whole store reads."""
    from sqlalchemy import select

    from haskie import db
    from haskie.tables import documents

    async with db.connect() as conn:
        return list(await conn.scalars(select(documents.c.name).order_by(documents.c.name)))


async def maintenance_state(collection: str):
    """The maintenance columns of a collection that is expected to exist."""
    from haskie.collection.collection import Collection

    state = await Collection(collection).maintenance_state()
    assert state is not None, f"no collection row for {collection}"
    return state


# The sample document most tests import: three headings, one body with a searchable word.
MD = "# Title\n\nintro text\n\n## Alpha\n\nalpha body about lancedb\n\n## Beta\n\nbeta body\n"


async def import_row(name: str, content: bytes | str = MD, into: Path | None = None, **options):
    """The document row and its file, with no pipeline started: the source is written outside the
    document store (under the home's `incoming/` unless `into` says where) and copied in, the way
    a real import reads a file the user already has."""
    from haskie import home
    from haskie.document import document

    source = (into or home.HOME / "incoming") / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(content.encode() if isinstance(content, str) else content)
    return await document.import_path(str(source), document.ImportOptions(**options))


def hit(
    text: str,
    score: float,
    *,
    document: str = "patterns.md",
    collection: str = "backend",
    seq: int = 1,
    char_start: int = 0,
) -> "Hit":
    """One indexed chunk as a search reads it: `text` at `char_start` of `document`, on the line
    its offset gives."""
    from haskie.collection.index import Hit, location

    line = char_start // 80 + 1
    return Hit(
        collection=collection,
        document=document,
        source_path=f"documents/{document}",
        markdown_path=f"documents/{document}.md",
        part=0,
        seq=seq,
        line_start=line,
        line_end=line,
        char_start=char_start,
        char_end=char_start + len(text),
        byte_start=char_start,
        byte_end=char_start + len(text),
        page_start=None,
        page_end=None,
        headings=["Messaging", "Retries"],
        frame=["Messaging", "Retries"],
        header="Messaging > Retries",
        location=location(document, None, None, line, line),
        text=text,
        score=score,
        source_file=f"/home/documents/{document}",
        markdown_file=f"/home/documents/{document}.md",
    )


def chunk_hit(
    chunk: "Chunk",
    seq: int,
    score: float = 1.0,
    *,
    document: str = "patterns.md",
    collection: str = "backend",
) -> "Hit":
    """One chunk the chunker cut (`chunk.split`), as a search reads it back from the index: its
    own offsets, lines, headings and cut reasons, numbered `seq`."""
    from haskie.collection.index import Hit, location

    return Hit(
        collection=collection,
        document=document,
        source_path=f"documents/{document}",
        markdown_path=f"documents/{document}.md",
        part=0,
        seq=seq,
        line_start=chunk.line_start,
        line_end=chunk.line_end,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        byte_start=chunk.byte_start,
        byte_end=chunk.byte_end,
        page_start=None,
        page_end=None,
        headings=chunk.headings,
        frame=chunk.frame,
        header=chunk.header,
        location=location(document, None, None, chunk.line_start, chunk.line_end),
        text=chunk.text,
        score=score,
        layout=chunk.layout,
        start_reason=chunk.start_reason,
        end_reason=chunk.end_reason,
        source_file=f"/home/documents/{document}",
        markdown_file=f"/home/documents/{document}.md",
    )


def words_scan(hits: "list[Hit]") -> "Scan":
    """The comparison spaces of a search without embeddings."""
    from haskie.search import collapse

    return collapse.spaces([one.text for one in hits], [None] * len(hits), None)


async def compact_model() -> "EmbeddingModel":
    """The "compact" profile's model as the catalogue holds it: bge-small, with the seed's own
    thresholds."""
    from haskie.catalogue import catalogue
    from haskie.settings import UserSettings

    model = await catalogue.embedding_model(UserSettings(embedding="compact"))
    assert model is not None
    return model


async def index_hits(
    index: "CollectionIndex", query: str, settings: "SearchSettings"
) -> list["Hit"]:
    """What one bare index retrieves for `query` by full text, as a search's `retrieve` step reads
    it (`search_rows`), cut to `settings.limit`: no embedding, no reranker, no fold."""
    rows = await index.search_rows(query, None, settings, settings.limit)
    return [index.hit(row) for row in rows]


async def collection_hits(name: str, query: str) -> list["Hit"]:
    """What a chunk search of the one collection `name` answers (`flow.chunks`), the one search
    the API runs, with the collection's own settings."""
    from haskie.search import flow

    return await flow.chunks([name], query)


async def search_with(name: str, query: str, search: "SearchOverrides") -> list["Hit"]:
    """`search` saved as the collection's search overrides, replacing any before, then its chunk
    search (`collection_hits`): no route takes search settings per call. Later searches of the
    collection keep them."""
    import msgspec

    from haskie.collection.collection import Collection

    collection = Collection(name)
    current = (await collection.info()).overrides
    await collection.set_overrides(msgspec.structs.replace(current, search=search))
    return await collection_hits(name, query)


async def seed_index(collection: str, doc: str, text: str, heading: str = "Alpha") -> None:
    """One indexed chunk of an imported document in a collection's table: the common case of
    `seed_chunks`, for a test that only needs something to match."""
    from haskie.indexing.chunk import Chunk, Piece

    await seed_chunks(
        collection,
        doc,
        [
            Chunk(
                headings=["Title", heading],
                frame=["Title", heading],
                pieces=[Piece(PieceType.TEXT, text)],
                line_start=5,
                line_end=7,
                char_start=0,
                char_end=len(text),
                byte_start=0,
                byte_end=len(text.encode()),
            )
        ],
    )


async def seed_chunks(collection: str, doc: str, chunks: "list[Chunk]") -> None:
    """Several indexed chunks of one imported document, numbered `seq` 1..N the way a real index
    numbers them (see `embed_cache._merge`).

    The real write path with no embedding model, so what a test gets is what a full-text-only
    collection holds — without paying for a pipeline run to put it there. The chunks carry real
    offsets into the document's markdown, which the caller builds itself.
    """
    from haskie.collection.collection import Collection
    from haskie.collection.index import Row
    from haskie.document import document

    row = await document.get(doc)
    rows = [Row(chunk=chunk, seq=seq) for seq, chunk in enumerate(chunks, start=1)]
    index = Collection(collection).index_with(None)
    await index.add_parts(
        doc, row.relative(row.original), row.relative(row.markdown), one_part(0, rows)
    )
    await index.finish()  # the full-text index the search reads


def legacy_index(path: Path, doc: str, text: str, heading: str = "H"):
    """A `chunks` table of a schema older than `index.Row`, returned for the test to read.

    What the read and delete paths have to degrade on rather than raise or drop.
    """
    import lancedb

    table = lancedb.connect(str(path)).create_table(
        "chunks", data=[{"doc": doc, "chunk_id": 0, "heading": heading, "text": text}]
    )
    table.create_fts_index("text", replace=True)
    return table


async def import_document(dbos, name: str, content: bytes | str, tmp_dir: Path):
    """Import one file and wait for its pipeline; returns the document row. What most workflow
    and API tests start from."""
    from haskie.document import document

    row = await import_row(name, content, tmp_dir)
    assert await wait_for(await dbos.start_import(row.name)) == "imported"
    return await document.get(row.name)


async def attach_document(dbos, collection: str, doc: str) -> None:
    """Attach one imported document to a collection and wait for its index."""
    assert await wait_for(await dbos.attach(collection, doc)) == "indexed"


async def delete_document(dbos, doc: str) -> None:
    """Delete one document and wait for the job: the app only ever starts it and polls the
    progress, so waiting for the result is a test's business, not the runtime's."""
    await wait_for(await dbos.start_delete_document(doc))


async def delete_collection(dbos, collection: str) -> None:
    """Delete one collection and wait for the job (see `delete_document`)."""
    await wait_for(await dbos.start_delete_collection(collection))


# --- the same intake over HTTP, for the API tests ----------------------------

IMPORT_TIMEOUT_SECONDS = 60
IMPORT_POLL_SECONDS = 0.02
MAX_WALK_PAGES = 100  # a cursor that never ends is the bug to catch, not a walk to hang on


# Where a client reaches haskie: a loopback host, the only one the app serves by default (see
# `app.served_hosts`), unlike the test client's own `testserver.local`.
LOOPBACK_URL = "http://127.0.0.1:8451"


def api_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The app with `WEB_DIST` pointed at a directory that does not exist, so its routing table is
    the API alone whether or not `web/dist` has been built."""
    from haskie import app as app_module

    monkeypatch.setattr(app_module, "WEB_DIST", tmp_path / "no-web-build")
    return app_module.create_app()


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A client for that app with no lifespan entered, so nothing here launches DBOS; a test that
    needs it takes the `dbos` fixture too (see `test_api`)."""
    from litestar.testing import AsyncTestClient

    return AsyncTestClient(api_app(tmp_path, monkeypatch), base_url=LOOPBACK_URL)


async def _release_default_executor() -> None:
    """Hand the portal loop a thread pool of its own, so closing it shuts that one down.

    Litestar's test transport answers every request on a blocking portal: an event loop of its
    own, in another thread. Closing a portal shuts down its loop's default executor, and DBOS
    makes *its* thread pool that executor as soon as an async DBOS call runs on the loop
    (`DBOS._configure_asyncio_thread_pool`). Without this, the first request would leave the
    running DBOS unable to schedule anything, teardown included. The pool below never starts a
    thread: a `ThreadPoolExecutor` only spawns one when something is submitted to it.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))


@pytest.fixture
async def client(api_client, dbos) -> AsyncIterator:
    """The shared client, plus DBOS and one portal for the whole test, and no lifespan (see
    `test_api`'s module docstring)."""
    from anyio.from_thread import start_blocking_portal

    with start_blocking_portal(backend="asyncio") as portal:
        api_client.blocking_portal = portal
        try:
            yield api_client
        finally:
            portal.call(_release_default_executor)


async def get_page(client, path: str, **params) -> dict:
    """One page; `cursor=None` is dropped, so the same call reads the first page too."""
    response = await client.get(path, params={k: v for k, v in params.items() if v is not None})
    assert response.status_code == 200, response.text
    return response.json()


async def walk_pages(
    client, path: str, **params
) -> tuple[list[dict], list[int], list[int | None], list[str | None]]:
    """Follow `next_cursor` to the last page; returns the items, the size of each page, the total
    each page reported and every page's cursor."""
    items: list[dict] = []
    sizes: list[int] = []
    totals: list[int | None] = []
    cursors: list[str | None] = []
    cursor: str | None = None
    for _ in range(MAX_WALK_PAGES):
        page = await get_page(client, path, cursor=cursor, **params)
        items.extend(page["items"])
        sizes.append(len(page["items"]))
        totals.append(page["total"])
        cursors.append(page["next_cursor"])
        cursor = page["next_cursor"]
        if cursor is None:
            return items, sizes, totals, cursors
    raise AssertionError(f"{path} never ran out of pages")


# What `wait_import` stops on: every state an import can end in.
FINISHED_STATUSES = frozenset({"imported", "error", "cancelled"})


async def wait_import(client, doc: str) -> dict:
    """Poll one document until its import ends. The import route answers with the row at `queued`
    and starts the pipeline, so a test that needs the markdown waits here."""
    deadline = time.monotonic() + IMPORT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = await client.get(f"/api/documents/{doc}")
        assert response.status_code == 200, response.text
        row = response.json()
        if row["status"] in FINISHED_STATUSES:
            return row
        await anyio.sleep(IMPORT_POLL_SECONDS)
    raise AssertionError(f"import of {doc} never finished")


async def stage_and_import(client, name: str, body: bytes, wait: bool = True, **fields) -> dict:
    """The whole intake, as a caller does it: stage, import, and wait for `imported` unless the
    test wants the row the import route answered with.

    The import names the document: staging keeps the bytes under an id of its own, so the caller
    passes back the `filename` the staging call returned (or a name of its choosing)."""
    staged = await client.post(
        "/api/documents/staging", files={"data": (name, body, "text/markdown")}
    )
    assert staged.status_code == 201, staged.text
    started = await client.post(
        "/api/documents/import",
        json={
            "staging_id": staged.json()["staging_id"],
            "name": staged.json()["filename"],
            **fields,
        },
    )
    assert started.status_code == 201, started.text
    if not wait:
        return started.json()
    row = await wait_import(client, started.json()["name"])
    assert row["status"] == "imported", row["error"]
    return row


async def attach_via_api(client, collection: str, doc: str) -> str:
    """Attach one imported document over the API and wait for the collection to index it."""
    response = await client.post(f"/api/collections/{collection}/documents", json={"document": doc})
    assert response.status_code == 202, response.text
    operation_id = response.json()["operation_id"]
    assert await wait_for(operation_id) == "indexed"
    return operation_id


def events(caplog) -> list[str]:
    """The events logged so far. Our loggers pass structlog's event dict as the record message."""
    return [record.msg["event"] for record in caplog.records if isinstance(record.msg, dict)]


def audit_lines() -> list[dict]:
    """Every audit record written to today's file, in the order they were appended."""
    import json

    from haskie import audit

    if not audit.path().exists():
        return []
    return [json.loads(line) for line in audit.path().read_text().splitlines()]
