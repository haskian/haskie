"""Fixtures shared by every test module: the temp home, and the DBOS runtime on top of it."""

import os

# Before `haskie` is imported anywhere: these are read from the environment at import time, and
# every one of them is a real wait. The two poll intervals ship at 1 s / 0.25 s, which the suite
# would otherwise sit through at every queue hand-off. pytest imports this file before any test
# module, and nothing else in the suite imports `haskie` earlier.
os.environ.setdefault("HASKIE_CONVERT_WORKERS", "0")  # extract inline; see test_cpu_pool
os.environ.setdefault("HASKIE_JOB_POLL_SECONDS", "0.02")
os.environ.setdefault("HASKIE_TASK_POLL_SECONDS", "0.02")
os.environ.setdefault("HASKIE_RETRY_INTERVAL_SECONDS", "0.01")
os.environ.setdefault("HASKIE_DOWNLOAD_RETRY_INTERVAL_SECONDS", "0.01")

import shutil  # noqa: E402
import sqlite3  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402

import anyio  # noqa: E402
import anyio.to_thread  # noqa: E402
import pytest  # noqa: E402

TEARDOWN_GRACE_SECONDS = 2.0  # how long a cancelled step may still be running at teardown
TEARDOWN_POLL_SECONDS = 0.05
DELAY_SWEEP_SECONDS = 0.05  # how often a debounced workflow is promoted (see `_sweep_delayed`)


@pytest.fixture(scope="session", autouse=True)
def logging_configured() -> None:
    """One structlog/stdlib configuration for the whole run, as the app does at startup."""
    from haskie import logs

    logs.configure()


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """The backend every `pytest.mark.anyio` test runs on. asyncio only: that is what Litestar and
    DBOS run on. Session-scoped so session fixtures may be async too."""
    return "asyncio"


def _use_home(path: Path) -> Path:
    """Point this process at `path`, the way `haskie --home` does, and make the directories a boot
    would. `home.use` owns which paths follow from the root, so they are not restated here.

    Returns the root that was in use, for the caller to hand to `_restore_home` afterwards: one
    call rebinds every path, so monkeypatch has no single attribute to undo.
    """
    from haskie import home

    previous = home.HOME
    home.use(path)
    # what `home.ensure_home` does, without an event loop: this runs from sync fixtures
    for directory in (home.COLLECTION_ROOT, home.DOCUMENT_ROOT, home.STAGING_ROOT, home.AUDIT_DIR):
        directory.mkdir(parents=True, exist_ok=True, mode=home.DIR_MODE)
    return previous


def _restore_home(path: Path) -> None:
    """Point the process back at `path`. Paths only, no directories: the root a test started from
    may be the user's real home, which the suite must never create anything in."""
    from haskie import home

    home.use(path)


def _drop_caches(patch: pytest.MonkeyPatch) -> None:
    """Drop the process caches that would otherwise answer from another home."""
    from haskie import collection, db, models, settings

    patch.setattr(db, "_migrated", set())
    patch.setattr(settings, "_cached", None)
    # a loaded model is process state, and the process outlives the test that loaded it
    patch.setattr(models, "_ready", set())
    patch.setattr(models, "_warming", set())
    collection.invalidate_collection_caches()


@pytest.fixture(scope="session")
def template_home(tmp_path_factory: pytest.TempPathFactory) -> Path:
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

    from haskie import workflows

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
            _restore_home(previous)  # this ran inside the test that asked for it
    connection = sqlite3.connect(path / "haskie.db")
    try:
        connection.execute("pragma wal_checkpoint(truncate)")  # so one file carries everything
    finally:
        connection.close()
    return path


@pytest.fixture(autouse=True)
def haskie_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A fresh temp home, with nothing in it: no database file, no migrations."""
    previous = _use_home(tmp_path)
    _drop_caches(monkeypatch)
    yield tmp_path
    _restore_home(previous)  # the temp directory goes; leave no module pointing into it


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
    from haskie import workflows

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

    from haskie import dbos_names

    instance = dbos_module._dbos_global_instance
    if instance is None:  # the test destroyed it itself
        return
    unfinished = [
        found.workflow_id
        for found in await DBOS.list_workflows_async(
            status=dbos_names.WAITING_STATUS, load_input=False, load_output=False
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

    from haskie import workflows

    handle = await DBOS.retrieve_workflow_async(workflow_id)
    return await handle.get_result(polling_interval_sec=workflows.TASK_POLL)


async def restart_dbos() -> None:
    """Stop DBOS and boot it again, as restarting the app would. Cheap: the system database it
    reopens already carries the queues and the schedules, so the boot only reads them."""
    from dbos import DBOS

    from haskie import workflows

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


async def maintenance_state(collection: str):
    """The maintenance columns of a collection that is expected to exist."""
    from haskie.collection import Collection

    state = await Collection(collection).maintenance_state()
    assert state is not None, f"no collection row for {collection}"
    return state


async def import_row(name: str, content: bytes | str, into: Path | None = None, **options):
    """The document row and its file, with no pipeline started: the source is written outside the
    document store (under the home's `incoming/` unless `into` says where) and copied in, the way
    a real import reads a file the user already has."""
    from haskie import document, home

    source = (into or home.HOME / "incoming") / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(content.encode() if isinstance(content, str) else content)
    return await document.import_path(str(source), **options)


async def import_document(dbos, name: str, content: bytes | str, tmp_dir: Path):
    """Import one file and wait for its pipeline; returns the document row. What most workflow
    and API tests start from."""
    from haskie import document

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
    job_id = response.json()["job_id"]
    assert await wait_for(job_id) == "indexed"
    return job_id


def audit_lines() -> list[dict]:
    """Every audit record written to today's file, in the order they were appended."""
    import json

    from haskie import audit

    if not audit.path().exists():
        return []
    return [json.loads(line) for line in audit.path().read_text().splitlines()]
