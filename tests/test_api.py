"""HTTP contract: status codes, error bodies, request ids and the audit trail.

Lifespan: `create_app()` registers `workflows.start` as a startup hook, and the `dbos`
fixture already launches DBOS on the same temp home. Both were measured against dbos 3.0.0:
`DBOS(config=...)` returns the existing singleton and `DBOS.launch()` only warns when it has
already launched, so the two are compatible. The choice here is still the simpler one: the
`client` fixture never enters the `AsyncTestClient` context manager, so no lifespan runs and DBOS
is owned by the `dbos` fixture alone, which also owns the teardown. `test_lifespan_*` below is the
one place that does run the lifespan, and it takes no `dbos` fixture. The portal every request
runs on is opened by the fixture instead, for the reason `_release_default_executor` gives.

`WEB_DIST` is pointed at a directory that does not exist, so the routing table is the API alone
whether or not `web/dist` has been built. `test_static_files_*` covers the other branch.
"""

import asyncio
import json
import logging
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anyio.to_thread
import lancedb
import pytest
import structlog
from anyio.from_thread import start_blocking_portal
from dbos import DBOS
from litestar.testing import AsyncTestClient, RequestFactory

from haskie import app as app_module
from haskie import audit, home, logs, workflows
from haskie.library import Library

from conftest import wait_for  # isort: skip

pytestmark = pytest.mark.anyio

MD = "# Title\n\nintro text\n\n## Alpha\n\nalpha body about lancedb\n\n## Beta\n\nbeta body\n"
LONG_SESSION_ID = "s" * 129
TOO_MANY_LIBRARIES = {"libraries": [f"l{i}" for i in range(101)]}
# Deliberately the flat, pre-shard paths an older build wrote: a hit recomputes both from the
# document name (Library.resolve_hit), so a row like this still opens the files that exist now.
INDEXED_ROW = {
    "doc": "guide.md",
    "source_path": "library/notes/files/guide.md",
    "markdown_path": "library/notes/markdown/guide.md.md",
    "part": 0,
    "chunk_id": 0,
    "line_start": 5,
    "line_end": 7,
    "char_start": 0,
    "char_end": 24,
    "page_start": None,
    "page_end": None,
    "parents": "Title",
    "heading": "Alpha",
    "text": "alpha body about lancedb",
}


async def _release_default_executor() -> None:
    """Hand the portal loop a thread pool of its own, so closing it shuts that one down.

    Litestar's test transport answers every request on a blocking portal: an event loop of its
    own, in another thread. Closing a portal shuts down its loop's default executor, and DBOS
    makes *its* thread pool that executor as soon as an async DBOS call runs on the loop
    (`DBOS._configure_asyncio_thread_pool`). Without this, the first request would leave the
    running DBOS unable to schedule anything, teardown included. The pool below never starts a
    thread: a `ThreadPoolExecutor` only spawns one when something is submitted to it.
    """
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))


@pytest.fixture
async def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dbos
) -> AsyncIterator[AsyncTestClient]:
    """One portal for the whole test, and no lifespan (see the module docstring)."""
    monkeypatch.setattr(app_module, "WEB_DIST", tmp_path / "no-web-build")
    client = AsyncTestClient(app_module.create_app())
    with start_blocking_portal(backend="asyncio") as portal:
        client.blocking_portal = portal
        try:
            yield client
        finally:
            portal.call(_release_default_executor)


@pytest.fixture
async def ready(client: AsyncTestClient) -> AsyncTestClient:
    """An initialized app with one library, one uploaded document, one indexed chunk and one
    session: the state every row of the error table is asserted against."""
    await client.post("/api/init", json={"profile": "none"})
    await client.post("/api/libraries", json={"name": "notes"})
    await client.post(
        "/api/libraries/notes/documents",
        files={"data": ("guide.md", MD.encode(), "text/markdown")},
    )
    library = await Library.get("notes")
    library.index_dir.mkdir(parents=True)
    table = lancedb.connect(str(library.index_dir)).create_table("chunks", data=[INDEXED_ROW])
    table.create_fts_index("text", replace=True)
    await client.put("/api/sessions/s1", json={"libraries": ["notes"]})
    return client


def _audit_lines() -> list[dict]:
    if not audit.path().exists():
        return []
    return [json.loads(line) for line in audit.path().read_text().splitlines()]


async def _finish(job_id: str):
    """Wait for one background job; the bulk routes answer 202 with its id."""
    return await wait_for(job_id)


async def _document_jobs(library: str) -> list[str]:
    """Every document pipeline a bulk index queued for one library."""
    found = await DBOS.list_workflows_async(
        name=workflows.index_document.__qualname__,
        workflow_id_prefix=f"idx:{library}:",
        load_input=False,
        load_output=False,
    )
    return [s.workflow_id for s in found]


async def _by_kind(client: AsyncTestClient, kind: str, **params) -> list[dict]:
    """One page of the jobs of one kind, as the section for it in the Jobs view asks for them."""
    response = await client.get("/api/jobs/by-kind", params={"kind": kind, **params})
    assert response.status_code == 200, response.text
    return response.json()["items"]


async def _index_library(client: AsyncTestClient, library: str) -> str:
    """Queue "index all" and wait until every document it queued is indexed."""
    started = await client.post(f"/api/libraries/{library}/index")
    assert started.status_code == 202, started.text
    job_id = started.json()["job_id"]
    await _finish(job_id)
    for document_job in await _document_jobs(library):
        assert await _finish(document_job) == "indexed"
    return job_id


# --- the route and error table ------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "method", "path", "body", "files", "status", "detail"),
    [
        (
            "second init -> conflict",
            "POST", "/api/init", {"profile": "none"}, None,
            409, "already initialized",
        ),
        (
            "unknown embedding profile -> unprocessable",
            "POST", "/api/init", {"profile": "bogus"}, None,
            422, "Invalid enum value 'bogus'",
        ),
        (
            "settings out of range -> unprocessable",
            "PUT", "/api/settings", {"pipeline": {"embedding_weight": 0}}, None,
            422, "embedding_weight must be >= 1, got 0",
        ),
        (
            "chunk overlap not smaller than chunk size -> unprocessable",
            "PUT", "/api/settings", {"conversion": {"chunk_size": 10, "chunk_overlap": 10}}, None,
            422, "chunk_overlap must be <",
        ),
        (
            "duplicate library -> conflict",
            "POST", "/api/libraries", {"name": "notes"}, None,
            409, "library already exists: notes",
        ),
        (
            "library name with nothing usable in it -> unprocessable",
            "POST", "/api/libraries", {"name": "***"}, None,
            422, "invalid name: '***'",
        ),
        (
            "unknown library -> not found",
            "GET", "/api/libraries/ghost", None, None,
            404, "library not found: ghost",
        ),
        (
            "delete unknown library -> not found",
            "DELETE", "/api/libraries/ghost", None, None,
            404, "library not found: ghost",
        ),
        (
            "index unknown library -> not found",
            "POST", "/api/libraries/ghost/index", None, None,
            404, "library not found: ghost",
        ),
        (
            "library override out of range -> unprocessable",
            "PUT", "/api/libraries/notes/settings", {"chunk_size": 10, "chunk_overlap": 10}, None,
            422, "chunk_overlap must be <",
        ),
        (
            "search limit below one -> unprocessable",
            "GET", "/api/libraries/notes/search?q=alpha&limit=-5", None, None,
            422, "Expected `int` >= 1",
        ),
        (
            "search needs a reranker that is not loaded -> service unavailable",
            "GET", "/api/libraries/notes/search?q=alpha&reranker=cross-encoder", None, None,
            503, "is not loaded yet",
        ),
        (
            "session search limit below one -> unprocessable",
            "GET", "/api/search?session_id=s1&q=alpha&limit=0", None, None,
            422, "Expected `int` >= 1",
        ),
        (
            "tasks of an unknown job -> not found",
            "GET", "/api/jobs/ghost/tasks", None, None,
            404, "job not found: ghost",
        ),
        (
            "progress of an unknown job -> not found",
            "GET", "/api/jobs/ghost/progress", None, None,
            404, "job not found: ghost",
        ),
        (
            "cancel an unknown job -> not found",
            "DELETE", "/api/jobs/ghost", None, None,
            404, "job not found: ghost",
        ),
        (
            "upload an unsupported file type -> unprocessable",
            "POST", "/api/libraries/notes/documents", None, ("virus.exe", b"MZ"),
            422, "unsupported file type: virus.exe",
        ),
        (
            "upload to an unknown library -> not found",
            "POST", "/api/libraries/ghost/documents", None, ("a.md", b"# a\n"),
            404, "library not found: ghost",
        ),
        (
            "import a relative path -> unprocessable",
            "POST", "/api/libraries/notes/documents/import", {"path": "notes/a.md"}, None,
            422, "path must be absolute: notes/a.md",
        ),
        (
            "import a path that is not there -> unprocessable",
            "POST", "/api/libraries/notes/documents/import", {"path": "/nowhere/a.md"}, None,
            422, "file not found: /nowhere/a.md",
        ),
        (
            "reindex an unknown document -> not found",
            "POST", "/api/libraries/notes/documents/ghost.md/index", None, None,
            404, "document not found: ghost.md",
        ),
        (
            "delete an unknown document -> not found",
            "DELETE", "/api/libraries/notes/documents/ghost.md", None, None,
            404, "document not found: ghost.md",
        ),
        (
            "source of an unknown document -> not found",
            "GET", "/api/libraries/notes/documents/ghost.md/source", None, None,
            404, "document file missing: ghost.md",
        ),
        (
            "preview of an unknown document -> not found",
            "GET", "/api/libraries/notes/documents/ghost.md/preview", None, None,
            404, "document not found: ghost.md",
        ),
        (
            "full markdown before the document is indexed -> not found",
            "GET", "/api/libraries/notes/documents/guide.md/markdown?full=true", None, None,
            404, "document not indexed yet: guide.md",
        ),
        (
            "session pointing at an unknown library -> not found",
            "PUT", "/api/sessions/s2", {"libraries": ["ghost"]}, None,
            404, "library not found: ghost",
        ),
        (
            "session id longer than the cap -> unprocessable",
            "PUT", f"/api/sessions/{LONG_SESSION_ID}", {"libraries": []}, None,
            422, "session id must be 1..128 characters",
        ),
        (
            "more libraries than a session may hold -> unprocessable",
            "PUT", "/api/sessions/s2", TOO_MANY_LIBRARIES, None,
            422, "at most 100 libraries per session, got 101",
        ),
        (
            "unknown route -> litestar's own not found",
            "GET", "/api/nope", None, None,
            404, "Not Found",
        ),
        (
            "wrong method on a known route -> method not allowed",
            "POST", "/api/status", None, None,
            405, "Method Not Allowed",
        ),
    ],
)  # fmt: skip
async def test_route_errors(
    ready: AsyncTestClient,
    name: str,
    method: str,
    path: str,
    body: dict | None,
    files: tuple[str, bytes] | None,
    status: int,
    detail: str,
) -> None:
    upload = None if files is None else {"data": (files[0], files[1], "application/octet-stream")}

    response = await ready.request(method, path, json=body, files=upload)

    assert response.status_code == status, f"{name}: {response.text}"
    assert detail in response.text, name


@pytest.mark.parametrize(
    ("name", "path", "expected"),
    [
        ("a matched route carries the id it logged under", "/api/status", True),
        ("an unmatched route never reaches the request hook", "/api/nope", False),
    ],
)
async def test_request_id_header(
    client: AsyncTestClient, name: str, path: str, expected: bool
) -> None:
    response = await client.get(path)
    assert (app_module.REQUEST_ID_HEADER.lower() in response.headers) is expected, name


async def test_model_not_ready_asks_the_caller_to_come_back(ready: AsyncTestClient) -> None:
    response = await ready.get("/api/libraries/notes/search?q=alpha&reranker=cross-encoder")
    assert response.status_code == 503
    assert response.headers["Retry-After"] == app_module.RETRY_AFTER_SECONDS


async def test_rejected_settings_are_never_stored(ready: AsyncTestClient) -> None:
    before = (await ready.get("/api/settings")).json()

    rejected = await ready.put("/api/settings", json={"pipeline": {"cpu_budget": 0}})

    assert rejected.status_code == 422
    assert (await ready.get("/api/settings")).json() == before, (
        "the body was rejected while decoding"
    )


async def test_accepted_settings_are_stored_and_applied(ready: AsyncTestClient) -> None:
    body = {"pipeline": {"cpu_budget": 3, "converting_weight": 2, "batch_pages": 4}}

    response = await ready.put("/api/settings", json=body)

    assert response.status_code == 200
    pipeline = response.json()["pipeline"]
    assert (pipeline["cpu_budget"], pipeline["converting_weight"]) == (3, 2)
    assert pipeline["indexing_weight"] == 1, "the weight nobody named keeps its default"
    assert (await ready.get("/api/settings")).json()["pipeline"]["batch_pages"] == 4


async def test_library_reranker_override_starts_its_download(
    ready: AsyncTestClient, monkeypatch
) -> None:
    """Q1: saving a reranker for one library used to change nothing but the row, so the first
    search of that library answered "not loaded yet" for a model nothing ever fetched."""
    from haskie import embed

    loaded: list[str] = []
    monkeypatch.setattr(embed, "warm_reranker", lambda name, accelerator: loaded.append(name))
    override = "jinaai/jina-reranker-v1-turbo-en"

    saved = await ready.put(
        "/api/libraries/notes/settings",
        json={"search": {"reranker": "cross-encoder", "reranker_model": override}},
    )

    assert saved.status_code == 200
    assert saved.json()["search"]["reranker_model"] == override
    await _finish(f"dl:reranker:{override}")
    assert loaded == [override], "the PUT started the download"
    downloads = await _by_kind(ready, "download")
    assert [(d["title"], d["status"], d["detail"]) for d in downloads] == [
        (f"download reranker {override}", "SUCCESS", {"warm": True})
    ]
    listed = (await ready.get("/api/status")).json()["models"]
    assert [(m["kind"], m["name"], m["state"]) for m in listed] == [
        ("reranker", override, "ready")
    ], "and /api/status reports it like any other required model"


async def test_downloads_are_empty_without_a_required_model(ready: AsyncTestClient) -> None:
    """`by-kind` is a literal segment under /api/jobs, so it must not be read as a job id (which
    would 404), and a kind with nothing in it is an empty page, not an error."""
    response = await ready.get("/api/jobs/by-kind", params={"kind": "download"})

    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None, "total": None}


async def test_status_reports_an_unreadable_settings_row(ready: AsyncTestClient) -> None:
    from haskie import db, settings

    async with db.connect() as conn:
        await conn.execute("update settings set json = '{not json' where id = 1")
    settings.invalidate()  # a direct write bypasses the process cache (see settings.invalidate)

    status = (await ready.get("/api/status")).json()

    assert status["settings_error"] is not None and "unreadable" in status["settings_error"]
    assert status["initialized"] is True, "defaults are in use, the app still runs"


async def test_options_and_status_before_init(client: AsyncTestClient) -> None:
    status = (await client.get("/api/status")).json()
    assert (status["initialized"], status["embedding"], status["models"]) == (False, None, [])
    assert status["home"] == str(home.HOME) and status["device"]

    options = (await client.get("/api/options")).json()
    assert "anydoc" in options["parsers"] and "hybrid" in options["search_modes"]
    assert options["docs"]["conversion.chunk_size"]["title"] == "Chunk size (characters)"
    assert options["embedding_profiles"]["compact"]["dims"] == 384


# --- the happy path ------------------------------------------------------------------


async def test_index_search_read_and_delete_a_document(client: AsyncTestClient) -> None:
    assert (await client.post("/api/init", json={"profile": "none"})).status_code == 201
    assert (await client.post("/api/libraries", json={"name": "notes"})).status_code == 201
    settings = await client.put(
        "/api/libraries/notes/settings", json={"chunk_size": 40, "chunk_overlap": 0}
    )
    assert settings.status_code == 200 and settings.json()["effective"]["chunk_size"] == 40

    upload = await client.post(
        "/api/libraries/notes/documents", files={"data": ("guide.md", MD.encode(), "text/markdown")}
    )
    assert upload.status_code == 201
    document = upload.json()
    assert {key: document[key] for key in ("name", "size", "status", "error", "preview")} == {
        "name": "guide.md",
        "size": len(MD),
        "status": "uploaded",
        "error": None,
        "preview": None,
    }
    assert document["created_at"] > 0 and document["updated_at"] > 0, "stamped on upload"

    await _index_library(client, "notes")
    (job_id,) = await _document_jobs("notes")

    (job,) = (await client.get("/api/jobs", params={"library": "notes"})).json()["items"]
    assert (job["id"], job["status"], job["doc"]) == (job_id, "SUCCESS", "guide.md")
    tasks = (await client.get(f"/api/jobs/{job_id}/tasks")).json()
    assert [t["stage"] for t in tasks] == ["convert", "embed", "index"]
    assert {t["status"] for t in tasks} == {"SUCCESS"}

    (hit,) = (await client.get("/api/libraries/notes/search", params={"q": "lancedb"})).json()
    assert (hit["doc"], hit["heading"], hit["header"]) == ("guide.md", "Alpha", "Title > Alpha")
    assert hit["location"] == "guide.md L5-7"

    # the viewer is streamed as NDJSON: one `head`, then one `page` of rendered HTML per page
    streamed = await client.get("/api/libraries/notes/documents/guide.md/markdown?full=true")
    assert streamed.headers["content-type"].startswith("application/x-ndjson")
    frames = [json.loads(line) for line in streamed.text.splitlines() if line]
    head, *pages = frames
    assert head["kind"] == "head"
    assert [h["text"] for h in head["toc"]] == ["Title", "Alpha", "Beta"]
    assert head["preview"]["kind"] == "text"
    assert head["pages"] == len(pages)
    body = "".join(page["html"] for page in pages)
    assert '<h1 id="h-0">Title</h1>' in body, "rendered server-side, anchored for the contents"
    assert "<script" not in body

    preview = await client.get("/api/libraries/notes/documents/guide.md/preview")
    assert preview.status_code == 200 and preview.text == MD
    source = await client.get("/api/libraries/notes/documents/guide.md/source")
    assert source.status_code == 200 and source.text == MD

    assert (await client.delete(f"/api/jobs/{job_id}")).status_code == 204, (
        "cancelling a finished job"
    )
    (indexed,) = (await client.get("/api/libraries/notes/documents")).json()["items"]
    assert indexed["status"] == "indexed", "a finished document keeps its status"
    assert (await client.get("/api/libraries/notes")).json()["counts"]["indexed"] == 1

    assert (await client.delete("/api/libraries/notes/documents/guide.md")).status_code == 204
    assert (await client.get("/api/libraries/notes/documents")).json()["items"] == []
    assert (await client.get("/api/libraries/notes")).json()["counts"]["total"] == 0
    deleted = await client.delete("/api/libraries/notes")
    assert deleted.status_code == 202, "the deletion is queued, not done in the request"
    await _finish(deleted.json()["job_id"])
    assert (await client.get("/api/libraries")).json()["items"] == []


async def test_session_search_survives_the_deletion_of_its_library(client: AsyncTestClient) -> None:
    await client.post("/api/init", json={"profile": "none"})
    for name in ("kept", "dropped"):
        await client.post("/api/libraries", json={"name": name})
        await client.post(
            f"/api/libraries/{name}/documents",
            files={"data": ("d.md", f"# {name}\n\nshared token\n".encode(), "text/markdown")},
        )
        await _index_library(client, name)
    assert (
        await client.put("/api/sessions/s1", json={"libraries": ["kept", "dropped"]})
    ).json() == [
        "kept",
        "dropped",
    ]
    assert (
        len((await client.get("/api/search", params={"session_id": "s1", "q": "shared"})).json())
        == 2
    )

    deleted = await client.delete("/api/libraries/dropped")

    assert deleted.status_code == 202
    await _finish(deleted.json()["job_id"])

    search = await client.get("/api/search", params={"session_id": "s1", "q": "shared"})
    assert search.status_code == 200, "one deleted library must not break every later search"
    assert [hit["library"] for hit in search.json()] == ["kept"]
    assert (await client.get("/api/sessions")).json() == {"s1": ["kept"]}


async def test_index_library_skips_a_document_that_vanished(
    client: AsyncTestClient, monkeypatch
) -> None:
    from haskie.errors import DocumentNotFound

    await client.post("/api/init", json={"profile": "none"})
    await client.post("/api/libraries", json={"name": "race"})
    await client.post(
        "/api/libraries/race/documents", files={"data": ("gone.md", b"# g\n", "text/markdown")}
    )

    async def vanished(library: str, doc: str, workflow_id: str | None = None) -> str:
        raise DocumentNotFound(f"document not found: {doc}")

    monkeypatch.setattr(workflows, "start_index", vanished)
    response = await client.post("/api/libraries/race/index")

    assert response.status_code == 202
    assert set(response.json()) == {"job_id"}
    assert await _finish(response.json()["job_id"]) == workflows.BulkResult(done=0, skipped=1)


async def test_bulk_progress_reports_a_whole_library_job(client: AsyncTestClient) -> None:
    await client.post("/api/init", json={"profile": "none"})
    await client.post("/api/libraries", json={"name": "notes"})
    for i in range(2):
        await client.post(
            "/api/libraries/notes/documents",
            files={"data": (f"d{i}.md", f"# d{i}\n\nbody\n".encode(), "text/markdown")},
        )

    job_id = await _index_library(client, "notes")

    progress = await client.get(f"/api/jobs/{job_id}/progress")
    assert progress.status_code == 200
    body = progress.json()
    assert (body["id"], body["kind"], body["library"], body["status"]) == (
        job_id,
        "index_library",
        "notes",
        "SUCCESS",
    )
    assert body["progress"] == {"done": 2, "skipped": 0, "total": 2, "last": None}
    assert body["error"] is None

    deleted = await client.delete("/api/libraries/notes")
    await _finish(deleted.json()["job_id"])

    removal = (await client.get(f"/api/jobs/{deleted.json()['job_id']}/progress")).json()
    assert (removal["kind"], removal["status"], removal["progress"]) == (
        "delete_library",
        "SUCCESS",
        None,
    ), "a deletion has no pages to report"
    assert (
        await client.get(f"/api/jobs/{(await _document_jobs('notes'))[0]}/progress")
    ).status_code == 404


async def test_jobs_by_kind_lists_each_kind(client: AsyncTestClient, monkeypatch) -> None:
    """Every kind of workflow the app runs is one listing with one row shape, so the Jobs view can
    show a section per kind; the title of a row is what that job is doing, in words."""
    from haskie import embed

    monkeypatch.setattr(embed, "warm_reranker", lambda name, accelerator: None)
    override = "jinaai/jina-reranker-v1-turbo-en"
    await client.post("/api/init", json={"profile": "none"})
    await client.post("/api/libraries", json={"name": "notes"})
    await client.post(
        "/api/libraries/notes/documents", files={"data": ("guide.md", MD.encode(), "text/markdown")}
    )
    await _index_library(client, "notes")
    await client.put(
        "/api/libraries/notes/settings",
        json={"search": {"reranker": "cross-encoder", "reranker_model": override}},
    )
    await _finish(f"dl:reranker:{override}")
    maintenance = await workflows.MAINTAIN.debounce_async("notes", 0.0, "notes")
    await _finish(maintenance.workflow_id)

    rows = {
        kind: await _by_kind(client, kind)
        for kind in ("document", "library", "download", "maintenance")
    }

    assert [r["title"] for r in rows["document"]] == ["notes / guide.md"]
    assert [r["title"] for r in rows["library"]] == ["index library notes"]
    assert [r["title"] for r in rows["download"]] == [f"download reranker {override}"]
    assert [r["title"] for r in rows["maintenance"]] == ["maintain notes"]
    assert {kind: rows[kind][0]["kind"] for kind in rows} == {k: k for k in rows}
    assert {r["status"] for group in rows.values() for r in group} == {"SUCCESS"}
    assert rows["document"][0]["detail"]["tasks_total"] >= 1, "a document counts its batches"
    assert rows["library"][0]["detail"] == {"done": 1, "skipped": 0, "total": 1}
    assert rows["download"][0]["detail"] == {"warm": True}
    assert await _by_kind(client, "document", library="other") == [], "the library filter applies"
    assert await _by_kind(client, "maintenance", library="other") == []
    assert (await client.get("/api/jobs/by-kind", params={"kind": "bogus"})).status_code == 422


async def test_jobs_kinds_reports_active_counts(client: AsyncTestClient, monkeypatch) -> None:
    """The sections come from the backend, in display order, and each carries how much of it is
    running right now: that is what the view polls on."""
    started, release = threading.Event(), threading.Event()

    async def blocked_count(library: str) -> int:
        started.set()
        # waited out in a worker thread: blocking DBOS's event loop would stop every other job
        assert await anyio.to_thread.run_sync(release.wait, 30.0), (
            "the test never released the bulk index"
        )
        return 0

    monkeypatch.setattr(workflows, "count_documents", blocked_count)
    await client.post("/api/init", json={"profile": "none"})
    await client.post("/api/libraries", json={"name": "notes"})

    idle = (await client.get("/api/jobs/kinds")).json()
    assert [(k["kind"], k["label"]) for k in idle] == [
        ("document", "Documents"),
        ("library", "Libraries"),
        ("download", "Model downloads"),
        ("maintenance", "Maintenance"),
        ("archive", "Archive"),
    ]
    assert {k["kind"]: k["active"] for k in idle}["library"] == 0
    nothing = {"queued": 0, "running": 0}
    assert (await client.get("/api/jobs/activity")).json() == {"jobs": nothing, "tasks": nothing}

    job_id = (await client.post("/api/libraries/notes/index")).json()["job_id"]
    assert started.wait(timeout=30), "the bulk index reached its first step"

    running = {k["kind"]: k["active"] for k in (await client.get("/api/jobs/kinds")).json()}
    assert running["library"] == 1, "the bulk index is counted under its own kind"
    assert (running["document"], running["download"]) == (0, 0), "and under no other"
    activity = (await client.get("/api/jobs/activity")).json()
    assert activity == {"jobs": {"queued": 0, "running": 1}, "tasks": nothing}, (
        "a running bulk index is one running job on a job.* queue and no task"
    )

    release.set()
    await _finish(job_id)

    finished = {k["kind"]: k["active"] for k in (await client.get("/api/jobs/kinds")).json()}
    assert finished["library"] == 0, "a finished job is not active any more"
    assert (await client.get("/api/jobs/activity")).json() == {"jobs": nothing, "tasks": nothing}


# --- audit trail ---------------------------------------------------------------------


async def test_every_audited_route_appends_one_record(
    client: AsyncTestClient, tmp_path: Path
) -> None:
    source = tmp_path / "imported.md"
    source.write_text(MD)

    await client.post("/api/init", json={"profile": "none"})
    await client.put("/api/settings", json={"search": {"limit": 7}})
    await client.post("/api/libraries", json={"name": "notes"})
    await client.put("/api/libraries/notes/settings", json={"chunk_size": 400})
    await client.post(
        "/api/libraries/notes/documents", files={"data": ("guide.md", MD.encode(), "text/markdown")}
    )
    await client.post("/api/libraries/notes/documents/import", json={"path": str(source)})
    job_id = (await client.post("/api/libraries/notes/documents/guide.md/index")).json()["job_id"]
    assert await wait_for(job_id) == "indexed"
    # waited out rather than left in flight: every document the bulk index re-runs writes a record
    # of its own, and which of them lands before the trail is read is not what this test is about
    reindex_job = await _index_library(client, "notes")
    await client.delete(f"/api/jobs/{job_id}")
    await client.put("/api/sessions/s1", json={"libraries": ["notes"]})
    await client.delete("/api/libraries/notes/documents/guide.md")
    delete_job = (await client.delete("/api/libraries/notes")).json()["job_id"]
    await _finish(delete_job)  # the trail is read once the queued work is cancelled and gone

    lines = _audit_lines()
    assert [line["event"] for line in lines] == [
        "settings.init",
        "settings.update",
        "library.create",
        "library.settings.update",
        "document.add",
        "document.add",
        "document.reindex",
        "index.completed",
        "library.reindex",
        "index.completed",  # the bulk index re-runs both documents
        "index.completed",
        "job.cancel",
        "session.libraries.set",
        "document.delete",
        "library.delete",
    ]
    by_event = {line["event"]: line for line in lines}
    assert all(line["outcome"] == "ok" for line in lines)
    assert all(line["app_version"] == audit.APP_VERSION for line in lines)
    assert all("request_id" in line for line in lines if line["actor"] == "web")
    assert by_event["settings.init"]["detail"] == {"profile": "none"}
    assert by_event["settings.update"]["detail"] == {"changed": "search.limit"}
    assert by_event["library.create"]["library"] == "notes"
    assert by_event["document.add"]["detail"] == {
        "name": "imported.md",
        "suffix": ".md",
        "size": len(MD),
    }
    assert by_event["document.reindex"]["doc"] == "guide.md"
    assert by_event["document.reindex"]["detail"] == {"job_id": job_id}
    indexed = next(line for line in lines if line["event"] == "index.completed")
    assert indexed["actor"] == "workflow"
    assert indexed["workflow_id"] == job_id
    assert "request_id" not in indexed, "a worker has no request context"
    assert by_event["job.cancel"]["workflow_id"] == job_id, "a known field, not free-form detail"
    assert by_event["library.reindex"]["detail"] == {"job_id": reindex_job}
    assert by_event["session.libraries.set"]["session_id"] == "s1"
    assert by_event["library.delete"]["detail"] == {"job_id": delete_job}


async def test_a_failed_request_is_audited_with_its_scrubbed_error(client: AsyncTestClient) -> None:
    await client.post("/api/libraries", json={"name": "notes"})

    assert (await client.post("/api/libraries", json={"name": "notes"})).status_code == 409

    ok, failed = _audit_lines()
    assert (ok["event"], ok["outcome"], ok["library"]) == ("library.create", "ok", "notes")
    assert (failed["event"], failed["outcome"]) == ("library.create", "error")
    assert failed["error"] == "Conflict: library already exists: notes"
    assert "library" not in failed, "the name is only attached once the library exists"
    assert ok["request_id"] != failed["request_id"], "one id per request"


async def test_an_import_records_the_file_name_but_never_the_path(
    client: AsyncTestClient, tmp_path: Path
) -> None:
    source = tmp_path / "private" / "salary.md"
    source.parent.mkdir()
    source.write_text(MD)
    await client.post("/api/libraries", json={"name": "notes"})

    assert (
        await client.post("/api/libraries/notes/documents/import", json={"path": str(source)})
    ).status_code == 201

    (_, record) = _audit_lines()
    assert record["detail"]["name"] == "salary.md"
    assert str(source.parent) not in json.dumps(record), "the source directory stays out of it"


async def test_the_audit_trail_is_written_even_when_logging_is_silenced(
    client: AsyncTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`HASKIE_LOG_LEVEL=CRITICAL` lands on the root logger, so this sets the same thing the
    env var configures. The file append is the durable sink: no level may drop it."""
    monkeypatch.setattr(logging.getLogger(), "level", logging.CRITICAL)
    assert not logging.getLogger("haskie.audit").isEnabledFor(logs.AUDIT)

    assert (await client.post("/api/libraries", json={"name": "notes"})).status_code == 201

    (record,) = _audit_lines()
    assert (record["event"], record["outcome"], record["library"]) == (
        "library.create",
        "ok",
        "notes",
    )


# --- unexpected failures --------------------------------------------------------------


async def test_an_unexpected_failure_answers_500_with_a_scrubbed_message(
    ready: AsyncTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preview row without a preview cannot happen through the API, so it is the one bug shape
    the generic handler is for: the body names it, with no absolute path in it."""
    from haskie.library import Document

    async def without_a_preview(self, doc: str) -> Document:
        return Document(name=doc, size=1, status="uploaded", preview=None)

    monkeypatch.setattr(Library, "ensure_preview", without_a_preview)

    response = await ready.get("/api/libraries/notes/documents/guide.md/preview")

    assert response.status_code == 500
    assert response.json()["detail"] == "RuntimeError: preview not stored for guide.md"
    assert str(home.HOME) not in response.text
    assert app_module.REQUEST_ID_HEADER in response.headers


async def test_a_domain_error_without_its_own_status_answers_400(
    ready: AsyncTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from haskie.errors import HaskieError

    async def refuse(request) -> list[str]:
        raise HaskieError("the library index is busy")

    monkeypatch.setattr(Library, "page", staticmethod(refuse))

    response = await ready.get("/api/libraries")

    assert response.status_code == 400
    assert response.json()["detail"] == "the library index is busy"


async def test_re_uploading_a_document_cancels_the_pipeline_that_still_runs(
    ready: AsyncTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B7: writing over a running document would race its steps, so the upload cancels first."""
    from haskie import workflows

    cancelled: list[tuple[str, str]] = []

    async def record(library: str, doc: str) -> None:
        cancelled.append((library, doc))

    monkeypatch.setattr(workflows, "cancel_document", record)

    replaced = await ready.post(
        "/api/libraries/notes/documents",
        files={"data": ("guide.md", b"# new\n", "text/markdown")},
    )

    assert replaced.status_code == 201 and replaced.json()["size"] == 6
    assert cancelled == [("notes", "guide.md")], "the first upload had no document to cancel"


# --- request context ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "path", "actor"),
    [
        ("the web UI", "/api/status", "web"),
        ("an MCP client", f"{app_module.MCP_PATH}/tools", "mcp"),
    ],
)
async def test_bind_request_context_names_the_actor(name: str, path: str, actor: str) -> None:
    request = RequestFactory().get(path=path)

    try:
        # read inside the same task: a context var set in one task is not visible in its caller
        await app_module.bind_request_context(request)
        context = dict(structlog.contextvars.get_contextvars())
        assert (context["actor"], context["method"], context["path"]) == (actor, "GET", path), name
        assert context["request_id"] == request.scope["state"][app_module.REQUEST_ID_KEY]
    finally:
        logs.clear()


# --- body size, static files and lifespan ---------------------------------------------


async def test_a_body_over_the_upload_cap_is_rejected_before_the_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dbos
) -> None:
    """Litestar enforces `request_max_body_size`, so a 512 MiB upload never has to be sent here:
    the cap is lowered and the same code path answers 413."""
    monkeypatch.setattr(app_module, "WEB_DIST", tmp_path / "no-web-build")
    monkeypatch.setattr(app_module, "UPLOAD_MAX_BYTES", 64)
    client = AsyncTestClient(app_module.create_app())
    await client.post("/api/libraries", json={"name": "notes"})

    response = await client.post(
        "/api/libraries/notes/documents", files={"data": ("big.md", b"x" * 500, "text/markdown")}
    )

    assert response.status_code == 413
    assert "Request Entity Too Large" in response.text
    assert app_module.REQUEST_ID_HEADER in response.headers, "traceable like any other rejection"
    assert await (await Library.get("notes")).document_names() == [], "nothing was stored"


async def test_static_files_are_served_when_the_web_build_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dbos
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>haskie</title>")
    monkeypatch.setattr(app_module, "WEB_DIST", dist)
    client = AsyncTestClient(app_module.create_app())

    assert "haskie" in (await client.get("/index.html")).text
    assert (await client.get("/api/status")).status_code == 200, (
        "the API still wins over the catch-all"
    )


async def test_lifespan_starts_and_destroys_dbos_on_every_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, seeded_home: Path
) -> None:
    """A1: the shutdown hook takes no parameter, so Litestar does not hand it the app. Running
    the lifespan twice proves start and destroy are both repeatable."""
    from dbos import _dbos as dbos_module

    monkeypatch.setattr(app_module, "WEB_DIST", tmp_path / "no-web-build")
    before = set(threading.enumerate())

    for _ in range(2):
        async with AsyncTestClient(app_module.create_app()) as client:
            assert (await client.get("/api/status")).status_code == 200
            assert dbos_module._dbos_global_instance is not None
        assert dbos_module._dbos_global_instance is None, "destroyed by the shutdown hook"

    leaked = [
        thread
        for thread in threading.enumerate()
        if thread not in before and not thread.daemon and thread.name.startswith("dbos-")
    ]
    assert leaked == [], f"threads outliving DBOS block interpreter exit: {leaked}"


# --- S4: full-text search across libraries --------------------------------------------

TEXT_DOCS = 3  # documents per library, one chunk each: six rows to merge and page over


async def _text_libraries(client: AsyncTestClient, *names: str) -> None:
    """Libraries with `TEXT_DOCS` indexed one-chunk documents each, all matching "haskell".

    The term is repeated once more per document, so no two chunks of one library score the same
    and the ranking a page cuts is the same ranking every time it is recomputed.
    """
    await client.post("/api/init", json={"profile": "none"})
    for name in names:
        await client.post("/api/libraries", json={"name": name})
        for i in range(TEXT_DOCS):
            body = f"# {name} {i}\n\n{'haskell ' * (i + 1)}chapter {i} of {name}\n"
            await client.post(
                f"/api/libraries/{name}/documents",
                files={"data": (f"d{i}.md", body.encode(), "text/markdown")},
            )
        await _index_library(client, name)


def _text_identity(page: dict) -> list[tuple]:
    """What identifies every chunk of a page, in the order the page listed them."""
    return [(h["library"], h["doc"], h["part"], h["chunk_id"]) for h in page["items"]]


def _text_cursor(
    q: str = "haskell",
    libraries: list[str] | None = None,
    page_size: int = 100,
    offset: int = 1,
) -> str:
    """A real cursor of some query, built at collection time for the table below."""
    from haskie import textsearch

    return textsearch.make_cursor(q, libraries or ["alpha", "beta"], page_size, offset)


def _listing_cursor() -> str:
    """A cursor of the library listing: another sort, so this route must not read it."""
    from haskie.paging import encode_cursor

    return encode_cursor(["alpha"], "name", "asc")


async def test_text_search_spans_all_libraries_by_default(client: AsyncTestClient) -> None:
    await _text_libraries(client, "alpha", "beta")
    assert (await client.post("/api/libraries", json={"name": "blank"})).status_code == 201

    response = await client.get("/api/search/text", params={"q": "haskell"})

    assert response.status_code == 200, response.text
    page = response.json()
    assert page["next_cursor"] is None, "six hits fit in one default page"
    assert page["total"] is None, "counting the ranking costs as much as producing it"
    assert len(page["items"]) == 2 * TEXT_DOCS
    assert {h["library"] for h in page["items"]} == {"alpha", "beta"}, "the blank library is absent"
    scores = [h["score"] for h in page["items"]]
    assert scores == sorted(scores, reverse=True), "raw BM25, best first, across both libraries"

    alpha = await Library.get("alpha")
    hit = next(h for h in page["items"] if h["library"] == "alpha" and h["doc"] == "d0.md")
    assert hit["markdown_path"] == alpha.relative(alpha.markdown_path("d0.md")), "sharded today"
    assert hit["source_path"] == alpha.relative(alpha.file_path("d0.md"))
    assert hit["home"] == str(home.HOME) and "haskell" in hit["text"]
    assert not [line for line in _audit_lines() if "search" in line["event"]], "never audited"


async def test_text_search_filters_libraries_and_rejects_unknown(client: AsyncTestClient) -> None:
    await _text_libraries(client, "alpha", "beta")

    one = await client.get("/api/search/text", params={"q": "haskell", "libraries": "alpha"})
    assert one.status_code == 200, one.text
    assert {h["library"] for h in one.json()["items"]} == {"alpha"}
    assert len(one.json()["items"]) == TEXT_DOCS

    deduped = await client.get(
        "/api/search/text", params={"q": "haskell", "libraries": " alpha ,alpha,"}
    )
    assert deduped.json() == one.json(), "duplicates and blanks drop out, the page is the same"

    everything = await client.get("/api/search/text", params={"q": "haskell", "libraries": ""})
    assert len(everything.json()["items"]) == 2 * TEXT_DOCS, "an empty filter is not a filter"

    unknown = await client.get("/api/search/text", params={"q": "haskell", "libraries": "ghost"})
    assert unknown.status_code == 404, "a name nobody owns is a mistake, not an empty page"
    assert "library not found: ghost" in unknown.text


async def test_text_search_pages_without_overlap(client: AsyncTestClient) -> None:
    await _text_libraries(client, "alpha", "beta")
    whole = (await client.get("/api/search/text", params={"q": "haskell", "page_size": 100})).json()

    walked: list[dict] = []
    cursor: str | None = None
    for _ in range(2 * TEXT_DOCS + 1):  # bounded: a cursor that never ends is the bug to catch
        params: dict = {"q": "haskell", "page_size": 1}
        if cursor is not None:
            params["cursor"] = cursor
        response = await client.get("/api/search/text", params=params)
        assert response.status_code == 200, response.text
        page = response.json()
        assert len(page["items"]) == 1, "a full page while there is a next cursor"
        walked.extend(page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break

    assert cursor is None, "the walk ended on its own"
    identity = _text_identity({"items": walked})
    assert len(set(identity)) == len(identity), "no chunk is shown twice"
    assert identity == _text_identity(whole), "the walk is the one-page ranking, cut up"


async def test_text_search_answers_an_empty_page_past_the_last_hit(client: AsyncTestClient) -> None:
    """A cursor is an offset into a ranking that may have shrunk since it was issued."""
    await _text_libraries(client, "alpha")
    deep = _text_cursor(libraries=["alpha"], page_size=100, offset=100)

    page = await client.get(
        "/api/search/text", params={"q": "haskell", "libraries": "alpha", "cursor": deep}
    )

    assert page.status_code == 200, page.text
    assert page.json() == {"items": [], "next_cursor": None, "total": None}


@pytest.mark.parametrize(
    ("name", "params", "status", "detail"),
    [
        ("page size below one", {"page_size": 0}, 422, "page_size must be 1..200, got 0"),
        ("page size over the cap", {"page_size": 201}, 422, "page_size must be 1..200, got 201"),
        (
            "a cursor of another query",
            {"cursor": _text_cursor(q="scala")},
            422,
            "cursor was issued for another query",
        ),
        (
            "a cursor of another page size",
            {"cursor": _text_cursor(page_size=50)},
            422,
            "cursor was issued for another query",
        ),
        (
            "a cursor of another set of libraries",
            {"cursor": _text_cursor(libraries=["alpha"])},
            422,
            "cursor was issued for another query",
        ),
        (
            "a cursor of the library listing",
            {"cursor": _listing_cursor()},
            422,
            "cursor does not match sort/order",
        ),
        ("a cursor that is not base64", {"cursor": "not a cursor!!"}, 422, "invalid cursor"),
        (
            "a cursor past the depth cap",
            {"cursor": _text_cursor(offset=1000)},
            422,
            "cannot read past 1000 results",
        ),
    ],
)
async def test_text_search_validates_page_size_and_cursor(
    client: AsyncTestClient, name: str, params: dict, status: int, detail: str
) -> None:
    await client.post("/api/init", json={"profile": "none"})
    for library in ("alpha", "beta"):  # the libraries the cursors above were issued for
        await client.post("/api/libraries", json={"name": library})

    response = await client.get("/api/search/text", params={"q": "haskell", **params})

    assert response.status_code == status, f"{name}: {response.text}"
    assert detail in response.json()["detail"], name
