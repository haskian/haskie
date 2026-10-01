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

import json
import logging
import math
import threading
import time
from pathlib import Path
from urllib.parse import unquote

import msgspec
import pytest
import structlog
from conftest import id_of
from litestar.testing import AsyncTestClient, RequestFactory
from sqlalchemy import update

from haskie import app as app_module
from haskie import audit, claude, db, errors, home, ids, logs
from haskie.catalogue import catalogue
from haskie.collection.collection import Collection, MemberStatus
from haskie.collection.index import CollectionIndex
from haskie.document import document
from haskie.document.document import DocumentStatus
from haskie.indexing import embed_cache, gguf_models, mlx_models
from haskie.indexing.chunk import Chunk, Piece, split
from haskie.indexing.segment import PieceType
from haskie.paging import Order
from haskie.search import aspects, flow, gaps, log
from haskie.settings import (
    DEFAULT_RERANKER,
    Accelerator,
    ChunkSettings,
    CollectionOverrides,
    PipelineSettings,
    Reranker,
    SearchOverrides,
    UserSettings,
    save_user_settings,
)
from haskie.tables import searches

from conftest import (  # isort: skip
    LOOPBACK_URL,
    Gate,
    NO_MODELS,
    api_app,
    attach_via_api,
    audit_lines,
    claude_installed,
    refresh_settled,
    document_names,
    forget_settings,
    get_page,
    seed_chunks,
    seed_index,
    stage_and_import,
    text_pdf,
    until,
    wait_event,
    wait_for,
    wait_import,
    walk_pages,
)

pytestmark = pytest.mark.anyio

MD = "# Title\n\nintro text\n\n## Alpha\n\nalpha body about lancedb\n\n## Beta\n\nbeta body\n"
LONG_SESSION_ID = "s" * 129
TOO_MANY_COLLECTIONS = {"collections": [f"c{i}" for i in range(101)]}


@pytest.fixture
async def ready(client: AsyncTestClient, tmp_path: Path) -> AsyncTestClient:
    """An initialized app with one collection, one imported member, one document still queued,
    one indexed chunk and one session: the state every row of the error table is asserted against.

    The rows are written directly and the index is seeded through `seed_index`, so the fixture
    costs no pipeline run; what the error table needs is the shape of the data, not how it got
    there.
    """
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "notes"})
    source = tmp_path / "guide.md"
    source.write_text(MD)
    imported = await document.import_path(str(source))
    await document.set_status(imported.id, DocumentStatus.IMPORTED)
    (tmp_path / "pending.md").write_text("# pending\n")
    await document.import_path(str(tmp_path / "pending.md"))  # stays `queued`: nothing started it

    notes = await Collection.get("notes")
    await notes.add(imported.id)
    await notes.set_member_status(imported.id, MemberStatus.INDEXED)
    await seed_index("notes", imported.name, "alpha body about lancedb")

    await client.put("/api/sessions/s1", json={"collections": ["notes"]})
    return client


async def test_a_rename_moves_only_the_name(ready: AsyncTestClient) -> None:
    """Every table and index holds the id, so the rename writes one row and a search cites the
    new name at once. The name is spelled as at import, suffix the original's, so another
    spelling of the same name changes nothing."""
    before = (await ready.get("/api/documents/guide.md")).json()

    renamed = await ready.put("/api/documents/guide.md/name", json={"name": "Retry Handbook"})

    assert renamed.status_code == 200, renamed.text
    assert (renamed.json()["name"], renamed.json()["id"]) == ("retry-handbook.md", before["id"])
    assert (await ready.get("/api/documents/guide.md")).status_code == 404
    assert (await ready.get("/api/documents/retry-handbook.md/collections")).json() == ["notes"]
    members = (await ready.get("/api/collections/notes/documents")).json()["items"]
    assert [member["document"]["name"] for member in members] == ["retry-handbook.md"]
    mapped = (await ready.get("/api/search/sections", params={"q": "lancedb"})).json()
    assert [one["document"] for one in mapped["documents"]] == ["retry-handbook.md"]
    assert {one["location"].split()[0] for one in mapped["sections"]} == {"retry-handbook.md"}, (
        "the index was not rewritten, yet it cites the new name"
    )
    same = await ready.put("/api/documents/retry-handbook.md/name", json={"name": "Retry_HANDBOOK"})
    assert same.json() == renamed.json(), "another spelling of the same name is a no-op"


def _requested(lines: list[dict]) -> list[str]:
    """The events a request produced, in order; an operation writes its own (see `workflows`)."""
    return [line["event"] for line in lines if line["actor"] != "operation"]


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
            422, "unknown embedding profile: bogus",
        ),
        (
            "unknown reranker model at init -> unprocessable",
            "POST", "/api/init",
            {"profile": "none", "search": {"reranker_model": "no/such-model"}}, None,
            422, "unknown reranker model: no/such-model",
        ),
        (
            "settings out of range -> unprocessable",
            "PUT", "/api/settings", {"pipeline": {"embedding_weight": 0}}, None,
            422, "embedding_weight must be >= 1, got 0",
        ),
        (
            "settings naming a profile the catalogue does not hold -> unprocessable",
            "PUT", "/api/settings", {"embedding": "bogus"}, None,
            422, "unknown embedding profile: bogus",
        ),
        (
            "settings naming a reranker the catalogue does not hold -> unprocessable",
            "PUT", "/api/settings", {"search": {"reranker_model": "no/such-model"}}, None,
            422, "unknown reranker model: no/such-model",
        ),
        (
            "an embedder is no reranker -> unprocessable",
            "PUT", "/api/settings", {"search": {"reranker_model": "intfloat/e5-base-v2"}}, None,
            422, "unknown reranker model: intfloat/e5-base-v2",
        ),
        (
            "merge share past 100% -> unprocessable",
            "PUT", "/api/settings", {"conversion": {"chunk_merge_below": 101}}, None,
            422, "chunk_merge_below must be 0 to 100, got 101",
        ),
        (
            "duplicate collection -> conflict",
            "POST", "/api/collections", {"name": "notes"}, None,
            409, "collection already exists: notes",
        ),
        (
            "collection name with nothing usable in it -> unprocessable",
            "POST", "/api/collections", {"name": "***"}, None,
            422, "invalid name: '***'",
        ),
        (
            "unknown collection -> not found",
            "GET", "/api/collections/ghost", None, None,
            404, "collection not found: ghost",
        ),
        (
            "delete unknown collection -> not found",
            "DELETE", "/api/collections/ghost", None, None,
            404, "collection not found: ghost",
        ),
        (
            "index unknown collection -> not found",
            "POST", "/api/collections/ghost/index", None, None,
            404, "collection not found: ghost",
        ),
        (
            "collection override out of range -> unprocessable",
            "PUT", "/api/collections/notes/overrides", {"chunk_size": 0}, None,
            422, "chunk_size must be >= 1, got 0",
        ),
        (
            "collection search override out of range -> unprocessable",
            "PUT", "/api/collections/notes/overrides", {"search": {"limit": 0}}, None,
            422, "limit must be >= 1, got 0",
        ),
        (
            "a section cap of none -> unprocessable",
            "PUT", "/api/settings", {"search": {"max_section_chars": 0}}, None,
            422, "max_section_chars must be >= 1, got 0",
        ),
        (
            "an answer budget of none -> unprocessable",
            "PUT", "/api/collections/notes/overrides", {"search": {"max_answer_chars": 0}}, None,
            422, "max_answer_chars must be >= 1, got 0",
        ),
        (
            "a negative shortest passage -> unprocessable",
            "PUT", "/api/settings", {"search": {"min_passage_chars": -1}}, None,
            422, "min_passage_chars must be >= 0, got -1",
        ),
        (
            "a negative growth -> unprocessable",
            "PUT", "/api/collections/notes/overrides", {"search": {"max_passage_grow": -1}}, None,
            422, "max_passage_grow must be >= 0, got -1",
        ),
        (
            "explore has no excerpt granularity: excerpts have their own route",
            "GET", "/api/search/explore?q=alpha&granularity=excerpt", None, None,
            422, "Invalid enum value 'excerpt'",
        ),
        (
            "collection override naming an unknown reranker -> unprocessable",
            "PUT", "/api/collections/notes/overrides",
            {"search": {"reranker_model": "no/such-model"}}, None,
            422, "unknown reranker model: no/such-model",
        ),
        (
            "explore limit below one -> unprocessable",
            "GET", "/api/search/explore?session_id=s1&q=alpha&limit=0", None, None,
            422, "Expected `int` >= 1",
        ),
        (
            "explore limit past the scan depth -> unprocessable",
            "GET", "/api/search/explore?session_id=s1&q=alpha&limit=201", None, None,
            422, "Expected `int` <= 200",
        ),
        (
            "excerpts limit past the scan depth -> unprocessable",
            "GET", "/api/search/excerpts?session_id=s1&q=alpha&limit=100000", None, None,
            422, "Expected `int` <= 200",
        ),
        (
            "tasks of an unknown job -> not found",
            "GET", "/api/jobs/ghost/tasks", None, None,
            404, "job not found: ghost",
        ),
        (
            "progress of an unknown operation -> not found",
            "GET", "/api/operations/ghost/progress", None, None,
            404, "operation not found: ghost",
        ),
        (
            "cancel an unknown operation -> not found",
            "DELETE", "/api/operations/ghost", None, None,
            404, "operation not found: ghost",
        ),
        (
            "an unknown operation kind -> unprocessable",
            "GET", "/api/operations?kind=bogus", None, None,
            422, "unknown operation kind 'bogus'",
        ),
        (
            "stage an unsupported file type -> unprocessable",
            "POST", "/api/documents/staging", None, ("virus.exe", b"MZ"),
            422, "unsupported file type: virus.exe",
        ),
        (
            "import a staging id nobody issued -> not found",
            "POST", "/api/documents/import", {"staging_id": "0" * 32 + ".md"}, None,
            404, "staged upload not found",
        ),
        (
            "import a malformed staging id -> unprocessable",
            "POST", "/api/documents/import", {"staging_id": "../escape.md"}, None,
            422, "invalid staging id",
        ),
        (
            "import with neither source -> unprocessable",
            "POST", "/api/documents/import", {}, None,
            422, "give either staging_id or path",
        ),
        (
            "import with both sources -> unprocessable",
            "POST", "/api/documents/import", {"staging_id": f"{'a' * 32}.md", "path": "/a.md"},
            None,
            422, "give either staging_id or path",
        ),
        (
            "import a relative path -> unprocessable",
            "POST", "/api/documents/import", {"path": "notes/a.md"}, None,
            422, "path must be absolute: a.md",
        ),
        (
            "import a path that is not there -> unprocessable",
            "POST", "/api/documents/import", {"path": "/nowhere/a.md"}, None,
            422, "file not found: a.md",
        ),
        (
            "re-import an unknown document -> not found",
            "POST", "/api/documents/ghost.md/import", None, None,
            404, "document not found: ghost.md",
        ),
        (
            # the route defers the rule to `start_import`, so this is that message
            "re-import a document that did not fail -> conflict",
            "POST", "/api/documents/guide.md/import", None, None,
            409, "only a queued, failed or cancelled import runs: guide.md",
        ),
        (
            "delete an unknown document -> not found",
            "DELETE", "/api/documents/ghost.md", None, None,
            404, "document not found: ghost.md",
        ),
        (
            "rename an unknown document -> not found",
            "PUT", "/api/documents/ghost.md/name", {"name": "x.md"}, None,
            404, "document not found: ghost.md",
        ),
        (
            "rename to a name taken, in any spelling -> conflict",
            "PUT", "/api/documents/guide.md/name", {"name": "Pending.md"}, None,
            409, "document already exists: pending.md",
        ),
        (
            "rename to a name that folds to nothing -> unprocessable, never `md.md`",
            "PUT", "/api/documents/guide.md/name", {"name": "Отчёт.md"}, None,
            422, "invalid name",
        ),
        (
            "rename to a blank name -> unprocessable",
            "PUT", "/api/documents/guide.md/name", {"name": "  "}, None,
            422, "a document needs a name",
        ),
        (
            "source of an unknown document -> not found",
            "GET", "/api/documents/ghost.md/source", None, None,
            404, "document not found: ghost.md",
        ),
        (
            "preview of an unknown document -> not found",
            "GET", "/api/documents/ghost.md/preview", None, None,
            404, "document not found: ghost.md",
        ),
        (
            "collections of an unknown document -> not found",
            "GET", "/api/documents/ghost.md/collections", None, None,
            404, "document not found: ghost.md",
        ),
        (
            "embeddings of an unknown document -> not found",
            "GET", "/api/documents/ghost.md/embeddings", None, None,
            404, "document not found: ghost.md",
        ),
        (
            "full markdown before the document is converted -> not found",
            "GET", "/api/documents/guide.md/markdown?full=true", None, None,
            404, "document not imported yet: guide.md",
        ),
        (
            "attach an unknown document -> not found",
            "POST", "/api/collections/notes/documents", {"document": "ghost.md"}, None,
            404, "document not found: ghost.md",
        ),
        (
            "attach a document that is still importing -> conflict",
            "POST", "/api/collections/notes/documents", {"document": "pending.md"}, None,
            409, "document is queued; only an imported document joins a collection: pending.md",
        ),
        (
            "attach to an unknown collection -> not found",
            "POST", "/api/collections/ghost/documents", {"document": "guide.md"}, None,
            404, "collection not found: ghost",
        ),
        (
            "detach a document the collection never held -> not found",
            "DELETE", "/api/collections/notes/documents/pending.md", None, None,
            404, "document not in collection notes: pending.md",
        ),
        (
            "re-index a document the collection never held -> not found",
            "POST", "/api/collections/notes/documents/pending.md/index", None, None,
            404, "document not in collection notes: pending.md",
        ),
        (
            "session pointing at an unknown collection -> not found",
            "PUT", "/api/sessions/s2", {"collections": ["ghost"]}, None,
            404, "collection not found: ghost",
        ),
        (
            "session id longer than the cap -> unprocessable",
            "PUT", f"/api/sessions/{LONG_SESSION_ID}", {"collections": []}, None,
            422, "Expected `str` of length <= 128",
        ),
        (
            "empty session id on a search -> unprocessable",
            "GET", "/api/search/excerpts?q=alpha&session_id=", None, None,
            422, "session_id=: Expected `str` of length >= 1",
        ),
        (
            "session id longer than the cap on a gap report -> unprocessable",
            "POST", f"/api/gaps/report?session_id={LONG_SESSION_ID}",
            {"question": "alpha", "verdict": "partial"}, None,
            422, "Expected `str` of length <= 128",
        ),
        (
            "more collections than a session may hold -> unprocessable",
            "PUT", "/api/sessions/s2", TOO_MANY_COLLECTIONS, None,
            422, "at most 100 collections per session, got 101",
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
    ("name", "method", "path", "rejected_by", "documented"),
    [
        (
            "a query that does not decode answers the 422 the document declares",
            "get", "/api/search/explore", "?q=alpha&limit=0", {"200", "422"},
        ),
        (
            "a body that does not decode answers the 422 the document declares",
            "put", "/api/settings", {"search": "not an object"}, {"200", "422"},
        ),
        (
            "a route with nothing to validate declares no rejection",
            "get", "/api/status", None, {"200"},
        ),
    ],
)  # fmt: skip
async def test_the_openapi_document_declares_the_rejections_answered(
    ready: AsyncTestClient,
    name: str,
    method: str,
    path: str,
    rejected_by: str | dict | None,
    documented: set[str],
) -> None:
    """Litestar documents its own 400 `{status_code, detail, extra}`; a client built from the
    document must read the 422 `{detail}` that `validation_error` actually answers."""
    responses = (await ready.get("/schema/openapi.json")).json()["paths"][path][method]["responses"]

    assert set(responses) == documented, name
    if rejected_by is None:
        return
    declared = responses["422"]["content"]["application/json"]["schema"]
    if isinstance(rejected_by, str):
        rejected = await ready.request(method, path + rejected_by)
    else:
        rejected = await ready.request(method, path, json=rejected_by)
    assert rejected.status_code == 422, f"{name}: {rejected.text}"
    assert set(rejected.json()) == set(declared["properties"]) == set(declared["required"]), name


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
    """A collection whose search needs a reranker that is not loaded: written straight to the row,
    since the route that saves it would start the download."""
    notes = await Collection.get("notes")
    await notes.set_overrides(
        CollectionOverrides(search=SearchOverrides(reranker=Reranker.CROSS_ENCODER))
    )

    response = await ready.get("/api/search/explore", params={"q": "alpha", "collections": "notes"})

    assert response.status_code == 503
    assert response.headers["Retry-After"] == errors.NotReady.headers["Retry-After"]


@pytest.mark.parametrize(
    ("name", "reranker", "floor", "logit", "expected", "uncovered"),
    [
        (
            "no reranker: the fused retrieval scores stand, and no model is read",
            None,
            None,
            None,
            None,
            [],
        ),
        (
            "a reranker on: the search's model scores every chunk",
            Reranker.CROSS_ENCODER,
            None,
            2.0,
            DEFAULT_RERANKER,
            [],
        ),
        (
            "a chunk it scores far under its floor still counts, and the map says it is weak",
            Reranker.CROSS_ENCODER,
            None,
            -9.0,
            DEFAULT_RERANKER,
            ["alpha"],
        ),
        (
            "the user's floor neither cuts a map nor judges it: 0.88 sits under 0.9",
            Reranker.CROSS_ENCODER,
            0.9,
            2.0,
            DEFAULT_RERANKER,
            [],
        ),
    ],
)
async def test_a_map_weighs_its_chunks_with_the_reranker(
    ready: AsyncTestClient,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    reranker: Reranker | None,
    floor: float | None,
    logit: float | None,
    expected: str | None,
    uncovered: list[str],
) -> None:
    """`search_sections` reranks with the excerpts' model, and keeps every chunk: its scores weigh
    the map, they do not cut it. Its log row names the model and no floor of the user's, so the
    gaps judge the map by that model's own floor, and so does the answer's `uncovered`."""
    from haskie.indexing import embed, models

    if reranker is not None:
        notes = await Collection.get("notes")
        await notes.set_overrides(
            CollectionOverrides(search=SearchOverrides(reranker=reranker, min_rerank_score=floor))
        )
        monkeypatch.setattr(
            models, "_ready", {models._model_id(models.ModelKind.RERANKER, DEFAULT_RERANKER)}
        )
    read: list[str] = []

    def scores(model: str, accelerator: str, query: str, texts: list[str]) -> list[float]:
        read.append(model)
        return [logit or 0.0] * len(texts)

    monkeypatch.setattr(embed, "rerank_scores", scores)

    response = await ready.get(
        "/api/search/sections", params={"q": "alpha", "collections": "notes", "session_id": "r"}
    )

    assert response.status_code == 200, f"{name}: {response.text}"
    (one,) = response.json()["sections"]
    assert one["document"] == "guide.md", name
    assert read == ([] if expected is None else [expected]), name
    assert response.json()["uncovered"] == uncovered, name
    if logit is not None:
        assert one["score"] == pytest.approx(1 / (1 + math.exp(-logit))), f"{name}: the sigmoid"
    (logged,) = (await ready.get("/api/searches", params={"session_id": "r"})).json()
    assert (logged["tool"], logged["reranker"]) == ("sections", expected), name
    assert logged["min_rerank_score"] is None, f"{name}: no floor of the user's is logged"


@pytest.mark.parametrize(
    ("name", "logit", "excerpts", "uncovered"),
    [
        ("judged an answer: excerpts, and nothing uncovered", 2.0, 1, []),
        (
            "under the reranker's floor: no excerpt, and the one question is uncovered",
            -9.0,
            0,
            ["alpha"],
        ),
    ],
)
async def test_one_question_hears_when_the_sources_match_it_only_weakly(
    ready: AsyncTestClient,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    logit: float,
    excerpts: int,
    uncovered: list[str],
) -> None:
    """`uncovered` used to list only the parts of several questions. A single question now hears
    the verdict the Gaps page gives as weak: its best match under the bar its models were
    measured at. The log keeps the excerpts' own `uncovered`, so the Gaps page still says weak."""
    from haskie.indexing import embed, models

    await _converted("guide.md", MD)
    notes = await Collection.get("notes")
    await notes.set_overrides(
        CollectionOverrides(search=SearchOverrides(reranker=Reranker.CROSS_ENCODER))
    )
    monkeypatch.setattr(
        models, "_ready", {models._model_id(models.ModelKind.RERANKER, DEFAULT_RERANKER)}
    )
    monkeypatch.setattr(embed, "rerank_scores", lambda m, a, q, texts: [logit] * len(texts))

    response = await ready.get(
        "/api/search/excerpts", params={"q": "alpha", "collections": "notes", "session_id": "w"}
    )

    assert response.status_code == 200, f"{name}: {response.text}"
    answer = response.json()
    assert (len(answer["excerpts"]), answer["uncovered"]) == (excerpts, uncovered), name
    (logged,) = (await ready.get("/api/searches", params={"session_id": "w"})).json()
    assert [one["uncovered"] for one in logged["questions"]] == [False], f"{name}: the log's own"


async def test_a_limit_at_the_scan_depth_is_searched(ready: AsyncTestClient) -> None:
    """The top of the shared `limit` bound (`settings.MAX_SCAN`) is a search, not a rejection; one
    past it is in the error table."""
    response = await ready.get(
        "/api/search/explore", params={"q": "alpha", "session_id": "s1", "limit": 200}
    )

    assert response.status_code == 200, response.text
    assert [hit["text"] for hit in response.json()] == ["alpha body about lancedb"]


async def test_rejected_settings_are_never_stored(ready: AsyncTestClient) -> None:
    before = (await ready.get("/api/settings")).json()

    rejected = await ready.put("/api/settings", json={"pipeline": {"cpu_budget": 0}})

    assert rejected.status_code == 422
    assert (await ready.get("/api/settings")).json() == before, (
        "the body was rejected while decoding"
    )


@pytest.mark.parametrize(
    ("session_id", "refusal"),
    [(LONG_SESSION_ID, "Expected `str` of length <= 128"), ("", "Expected `str` of length >= 1")],
    ids=["too long", "empty"],
)
@pytest.mark.parametrize(
    ("name", "method", "path", "body", "probe"),
    [
        (
            "an import", "POST", "/api/documents/import", {"path": "{source}"},
            "/api/documents/new.md",
        ),
        (
            "a description", "PUT", "/api/documents/guide.md/description",
            {"description": "changed"}, "/api/documents/guide.md",
        ),
        (
            "an attach", "POST", "/api/collections/other/documents", {"document": "guide.md"},
            "/api/collections/other/documents",
        ),
        (
            "a detach", "DELETE", "/api/collections/notes/documents/guide.md", None,
            "/api/collections/notes/documents",
        ),
    ],
)  # fmt: skip
async def test_a_bad_session_id_is_refused_before_the_change_it_would_record(
    ready: AsyncTestClient,
    tmp_path: Path,
    name: str,
    method: str,
    path: str,
    body: dict | None,
    probe: str,
    session_id: str,
    refusal: str,
) -> None:
    """Refused after the change, the caller would read a 422 for work done, and its retry a 409."""
    await ready.post("/api/collections", json={"name": "other"})
    source = tmp_path / "new.md"
    source.write_text(MD)
    if body is not None:
        body = {key: value.format(source=source) for key, value in body.items()}
    before = await ready.get(probe)

    response = await ready.request(method, path, json=body, params={"session_id": session_id})

    assert response.status_code == 422, f"{name}: {response.text}"
    assert f"session_id={session_id}: {refusal}" in response.text, name
    after = await ready.get(probe)
    assert (after.status_code, after.json()) == (before.status_code, before.json()), name


async def test_accepted_settings_are_stored_and_applied(ready: AsyncTestClient) -> None:
    body = {"pipeline": {"cpu_budget": 3, "converting_weight": 2, "batch_pages": 4}}

    response = await ready.put("/api/settings", json=body)

    assert response.status_code == 200
    pipeline = response.json()["pipeline"]
    assert (pipeline["cpu_budget"], pipeline["converting_weight"]) == (3, 2)
    assert pipeline["indexing_weight"] == 1, "the weight nobody named keeps its default"
    assert (await ready.get("/api/settings")).json()["pipeline"]["batch_pages"] == 4


async def test_collection_reranker_override_starts_its_download(
    ready: AsyncTestClient, monkeypatch
) -> None:
    """Q1: saving a reranker for one collection used to change nothing but the row, so the first
    search of that collection answered "not loaded yet" for a model nothing ever fetched."""
    from haskie.indexing import embed, hardware

    loaded: list[str] = []
    monkeypatch.setattr(embed, "warm_reranker", lambda name, accelerator: loaded.append(name))
    override = "cross-encoder/ettin-reranker-150m-v1"

    saved = await ready.put(
        "/api/collections/notes/overrides",
        json={"search": {"reranker": "cross-encoder", "reranker_model": override}},
    )

    assert saved.status_code == 200
    assert saved.json()["search"]["reranker_model"] == override
    await wait_for(f"dl:reranker:{override}")
    assert loaded == [override], "the PUT started the download"
    listed = (await ready.get("/api/status")).json()["models"]
    here = hardware.device(override, Accelerator.AUTO)  # the CPU, or Apple Silicon by WebGPU
    assert here is not None
    assert [(m["kind"], m["name"], m["state"], m["device"]) for m in listed] == [
        ("reranker", override, "ready", here.value)
    ], "and /api/status reports it like any other required model, with where it runs"


async def test_status_reports_an_unreadable_settings_row(ready: AsyncTestClient) -> None:
    from sqlalchemy import update

    from haskie import db
    from haskie.tables import settings as settings_table

    async with db.connect() as conn:
        await conn.execute(update(settings_table).values(json="{not json"))
    forget_settings()  # a direct write bypasses the process cache

    status = (await ready.get("/api/status")).json()

    assert status["settings_error"] is not None and "unreadable" in status["settings_error"]
    assert status["initialized"] is True, "defaults are in use, the app still runs"


async def test_rows_written_with_the_maps_own_reranker_still_read(ready: AsyncTestClient) -> None:
    """v0.23.0 stored `map_reranker_model` in the settings row and in a collection's overrides.
    The setting is gone; both rows still read, keeping every value they hold besides it."""
    from sqlalchemy import update

    from haskie import db
    from haskie.tables import collections
    from haskie.tables import settings as settings_table

    chosen = "cross-encoder/ettin-reranker-17m-v1"
    old_key = {"map_reranker_model": "cross-encoder/ms-marco-MiniLM-L2-v2"}
    stored = (await ready.get("/api/settings")).json()
    stored["search"] |= {"reranker_model": chosen, **old_key}
    overrides = {"search": {"reranker_model": chosen, **old_key}}
    async with db.connect() as conn:
        await conn.execute(update(settings_table).values(json=json.dumps(stored)))
        await conn.execute(
            update(collections)
            .where(collections.c.name == "notes")
            .values(overrides=json.dumps(overrides))
        )
    forget_settings()  # a direct write bypasses the process cache

    assert (await ready.get("/api/status")).json()["settings_error"] is None
    assert (await ready.get("/api/settings")).json()["search"]["reranker_model"] == chosen
    notes = (await ready.get("/api/collections/notes")).json()
    assert notes["overrides"]["search"]["reranker_model"] == chosen
    assert "map_reranker_model" not in notes["overrides"]["search"]


async def test_options_and_status_before_init(client: AsyncTestClient) -> None:
    status = (await client.get("/api/status")).json()
    assert (status["initialized"], status["embedding"], status["models"]) == (False, None, [])
    assert status["home"] == str(home.HOME)

    options = (await client.get("/api/options")).json()
    assert "anydoc" in options["parsers"] and "hybrid" in options["search_modes"]
    assert options["docs"]["conversion.chunk_size"]["title"] == "Chunk size (characters)"
    assert options["docs"]["search.grow_bias"]["title"] == "Growth bias"
    assert options["embedding_profiles"]["granite-97m-multilingual"]["dims"] == 384
    # the catalogue, read from the database: full-text only first, then the models by size
    profiles = options["embedding_profiles"]
    assert next(iter(profiles)) == "none" and profiles["none"] is None
    sizes = [model["dims"] for model in profiles.values() if model]
    assert "granite-97m-multilingual" in profiles and sizes == sorted(sizes), (
        "the smaller vectors first"
    )
    metadata = options["embedding_metadata"]
    assert set(metadata) == set(await catalogue.embedders()), "every profile's, offered or not"
    assert set(profiles) - {"none"} <= set(metadata), "metadata for every offered model"
    assert metadata["granite-97m-multilingual"] == {
        "description": (
            "The best all-round small multilingual embedder: #1 on multilingual and reasoning "
            "retrieval; a good default (~390 MB)."
        ),
        "parameters": 97441152,
        "context_tokens": 32768,
        "languages": "multilingual (200+, 52 enhanced)",
        "license": "Apache-2.0",
        "released": "2026-04-20",
        "model_card_url": "https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2",
        "runtime": "onnx",
        "devices": ["cpu", "apple_silicon", "gpu"],
        "dimensions": 384,
    }
    assert metadata["bekko-a25m-256"]["description"] != metadata["bekko-a25m"]["description"]
    assert metadata["bekko-a25m-256"]["parameters"] == metadata["bekko-a25m"]["parameters"]
    reranker = options["reranker_metadata"][DEFAULT_RERANKER]
    assert (reranker["parameters"], reranker["context_tokens"]) == (31883136, 8192)
    assert (reranker["runtime"], reranker["devices"]) == ("onnx", ["cpu", "apple_silicon", "gpu"])
    assert "dimensions" not in reranker, "a reranker has no vectors"
    assert options["reranker_models"][:3] == [
        "cross-encoder/ms-marco-MiniLM-L2-v2",
        "cross-encoder/ettin-reranker-17m-v1",
        DEFAULT_RERANKER,
    ], "smallest first: MiniLM-L2, then ettin-17m, then the default"
    # the vocabularies the UI renders rows with, so it never spells a status out for itself
    assert (
        options["document_statuses"][:3]
        == options["active_document_statuses"]
        == [
            "queued",
            "converting",
            "embedding",
        ]
    )
    assert options["active_run_statuses"] == ["ENQUEUED", "PENDING"]


@pytest.mark.parametrize(("name", "installed"), [("MLX installed", True), ("no MLX", False)])
async def test_the_options_offer_mlx_models_only_where_mlx_is_installed(
    client: AsyncTestClient, monkeypatch: pytest.MonkeyPatch, name: str, installed: bool
) -> None:
    monkeypatch.setattr(mlx_models, "available", lambda: installed)
    mlx_rerankers = set(mlx_models.RERANKERS)

    options = (await client.get("/api/options")).json()

    profiles = options["embedding_profiles"].values()
    embedders = {model["name"] for model in profiles if model} & set(mlx_models.POOLED)
    assert embedders == (set(mlx_models.POOLED) if installed else set()), name
    rerankers = set(options["reranker_models"]) & mlx_rerankers
    assert rerankers == (mlx_rerankers if installed else set()), name
    assert mlx_rerankers <= set(options["reranker_metadata"]), "metadata, offered or not"


@pytest.mark.parametrize(
    ("name", "installed", "accelerator", "offer"),
    [
        ("llama.cpp installed", True, "auto", True),
        ("no llama.cpp", False, "auto", False),
        ("the settings ask for the CPU, which llama.cpp is not run on", True, "cpu", False),
    ],
)
async def test_the_options_offer_gguf_models_only_where_they_run(
    client: AsyncTestClient,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    installed: bool,
    accelerator: Accelerator,
    offer: bool,
) -> None:
    monkeypatch.setattr(gguf_models, "available", lambda: installed)
    await save_user_settings(UserSettings(pipeline=PipelineSettings(accelerator=accelerator)))
    embedders = await catalogue.embedders()
    gguf = {profile for profile, model in embedders.items() if model.name in gguf_models.PINS}

    options = (await client.get("/api/options")).json()

    assert {embedders[profile].name for profile in gguf} == set(gguf_models.PINS)
    offered = set(options["embedding_profiles"]) & gguf
    assert offered == (gguf if offer else set()), name
    assert gguf <= set(options["embedding_metadata"]), "metadata, offered or not"
    # the llm descriptors' describer is a GGUF model too
    assert options["descriptors"] == (["c-tf-idf", "llm"] if offer else ["c-tf-idf"]), name


@pytest.mark.parametrize(
    ("name", "path", "body", "read"),
    [
        ("an unknown profile", "/api/settings", {"embedding": "bogus"}, "/api/settings"),
        (
            "an unknown reranker",
            "/api/settings",
            {"search": {"reranker_model": "no/such-model"}},
            "/api/settings",
        ),
        (
            "an unknown reranker for one collection",
            "/api/collections/notes/overrides",
            {"search": {"reranker_model": "no/such-model"}},
            "/api/collections/notes",
        ),
    ],
)
async def test_a_write_naming_a_model_the_catalogue_lacks_stores_nothing(
    ready: AsyncTestClient, name: str, path: str, body: dict, read: str
) -> None:
    before = (await ready.get(read)).json()

    response = await ready.put(path, json=body)

    assert response.status_code == 422, name
    assert (await ready.get(read)).json() == before, name


@pytest.mark.parametrize(
    ("name", "body", "llama_cpp", "expected"),
    [
        (
            "the profile alone: hybrid search, reranked by ettin-32m, c-TF-IDF descriptors",
            {"profile": "none"},
            False,
            ("hybrid", "cross-encoder", "cross-encoder/ettin-reranker-32m-v1", "c-tf-idf"),
        ),
        (
            "the search picked with it is stored as given, what it leaves out as no reranker",
            {
                "profile": "none",
                "search": {
                    "mode": "fts",
                    "reranker_model": "Alibaba-NLP/gte-reranker-modernbert-base",
                },
            },
            False,
            ("fts", "none", "Alibaba-NLP/gte-reranker-modernbert-base", "c-tf-idf"),
        ),
        (
            "llm descriptors where llama.cpp runs: stored, and their describer downloads",
            {**NO_MODELS, "descriptors": "llm"},
            True,
            ("hybrid", "none", "cross-encoder/ettin-reranker-32m-v1", "llm"),
        ),
    ],
)
async def test_init_stores_what_was_picked(
    client: AsyncTestClient,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    body: dict,
    llama_cpp: bool,
    expected: tuple[str, str, str, str],
) -> None:
    from haskie.indexing import embed

    monkeypatch.setattr(gguf_models, "available", lambda: llama_cpp)
    loaded: list[str] = []
    monkeypatch.setattr(embed, "warm_generator", lambda name, accelerator: loaded.append(name))

    response = await client.post("/api/init", json=body)

    assert response.status_code == 201, f"{name}: {response.text}"
    settings = (await client.get("/api/settings")).json()
    search = settings["search"]
    picked = (search["mode"], search["reranker"], search["reranker_model"])
    assert (*picked, settings["pipeline"]["descriptors"]) == expected, name
    if expected[-1] == "llm":
        await wait_for(f"dl:describer:{gguf_models.DESCRIBER}")
        assert loaded == [gguf_models.DESCRIBER], f"{name}: the first run starts its download"


async def test_init_refuses_llm_descriptors_where_their_model_cannot_run(
    client: AsyncTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without llama.cpp the describer runs nowhere, so the first run is refused whole: nothing
    stored, and the page can be submitted again with c-TF-IDF."""
    monkeypatch.setattr(gguf_models, "available", lambda: False)

    response = await client.post("/api/init", json={**NO_MODELS, "descriptors": "llm"})

    assert response.status_code == 422, response.text
    assert gguf_models.DESCRIBER in response.json()["detail"]
    assert (await client.get("/api/status")).json()["initialized"] is False
    retried = await client.post("/api/init", json={**NO_MODELS, "descriptors": "c-tf-idf"})
    assert retried.status_code == 201, retried.text


async def test_the_first_run_starts_from_a_cross_encoder_and_then_reads_what_was_picked(
    client: AsyncTestClient,
) -> None:
    """Before the first run, the settings are the ones it starts from, which the first-run page
    shows: a cross-encoder reranks. After it, they are what was picked."""
    before = (await client.get("/api/settings")).json()["search"]
    await client.post("/api/init", json=NO_MODELS)
    after = (await client.get("/api/settings")).json()["search"]

    assert (before["reranker"], before["reranker_model"]) == (
        "cross-encoder",
        "cross-encoder/ettin-reranker-32m-v1",
    ), "the smallest cross-encoder, by default"
    assert after["reranker"] == "none", "the pick, once there is one"


# --- the two-call intake --------------------------------------------------------------


async def test_staging_commits_nothing_and_import_commits_the_name(
    client: AsyncTestClient,
) -> None:
    """Upload and import are two calls on purpose: the bytes land first, the name is fixed only
    when the caller says so, and a name already taken is refused with the upload still staged."""
    await client.post("/api/init", json=NO_MODELS)

    staged = await client.post(
        "/api/documents/staging", files={"data": ("guide.md", MD.encode(), "text/markdown")}
    )

    assert staged.status_code == 201, staged.text
    upload = staged.json()
    assert (upload["filename"], upload["size"]) == ("guide.md", len(MD))
    assert (await client.get("/api/documents")).json()["items"] == [], "no row yet"

    imported = await client.post(
        "/api/documents/import",
        json={
            "staging_id": upload["staging_id"],
            "name": upload["filename"],
            "description": "the guide",
        },
    )

    assert imported.status_code == 201, imported.text
    row = imported.json()
    assert {key: row[key] for key in ("name", "size", "status", "error", "description")} == {
        "name": "guide.md",
        "size": len(MD),
        "status": "queued",
        "error": None,
        "description": "the guide",
    }
    assert row["created_at"] > 0 and row["updated_at"] > 0, "stamped on import"
    assert (await wait_import(client, "guide.md"))["status"] == "imported"

    again = await client.post(
        "/api/documents/staging", files={"data": ("guide.md", b"# other\n", "text/markdown")}
    )
    collision = await client.post(
        "/api/documents/import",
        json={"staging_id": again.json()["staging_id"], "name": "guide.md"},
    )

    assert collision.status_code == 409, collision.text
    assert "document already exists: guide.md" in collision.text
    renamed = await client.post(
        "/api/documents/import",
        json={"staging_id": again.json()["staging_id"], "name": "other"},
    )
    assert renamed.status_code == 201, "the same upload imports under a free name"
    assert renamed.json()["name"] == "other.md", "a rename keeps the original suffix"


async def test_import_by_path_copies_the_file(client: AsyncTestClient, tmp_path: Path) -> None:
    """The other source: a file already on this machine, imported without an upload."""
    await client.post("/api/init", json=NO_MODELS)
    source = tmp_path / "paper.md"
    source.write_text(MD)

    imported = await client.post("/api/documents/import", json={"path": str(source)})

    assert imported.status_code == 201, imported.text
    assert imported.json()["name"] == "paper.md"
    assert source.is_file(), "the original is copied, not moved"
    assert (await wait_import(client, "paper.md"))["status"] == "imported"


async def test_a_failed_import_can_be_re_run(client: AsyncTestClient, tmp_path: Path) -> None:
    await client.post("/api/init", json=NO_MODELS)
    row = await stage_and_import(client, "guide.md", MD.encode())
    await document.set_status(await id_of(row["name"]), DocumentStatus.ERROR, "converter fell over")

    again = await client.post(f"/api/documents/{row['name']}/import")

    assert again.status_code == 202, again.text
    assert await wait_for(again.json()["operation_id"]) == "imported"
    assert (await client.get(f"/api/documents/{row['name']}")).json()["error"] is None


# --- membership -----------------------------------------------------------------------


async def test_an_upload_names_the_document_it_repeats(
    client: AsyncTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staging names the document that already is these bytes, before anything is imported, and
    the import is refused: the bytes are what a document is. `similar` names the nearest by
    document vector under the embedding model; full-text only has no vectors to compare."""
    await client.post("/api/init", json=NO_MODELS)
    guide = await stage_and_import(client, "guide.md", MD.encode())  # the first: MD, unchanged
    await stage_and_import(client, "other.md", b"# Other\n\nsomething else entirely\n")

    staged = await client.post(
        "/api/documents/staging", files={"data": ("copy.md", MD.encode(), "text/markdown")}
    )
    assert staged.status_code == 201, staged.text
    assert staged.json()["duplicate"] == "guide.md", "the same bytes, under another name"
    refused = await client.post(
        "/api/documents/import",
        json={"staging_id": staged.json()["staging_id"], "name": "copy.md"},
    )
    assert refused.status_code == 409, refused.text
    assert "this file is already imported as guide.md" in refused.text

    similar = await client.get("/api/documents/guide.md/similar")

    assert similar.status_code == 200, similar.text
    assert similar.json() == {"nearest": []}, "no model, no vectors"

    asked: list[tuple[str, str, int]] = []

    async def nearest(doc: str, model: str, limit: int) -> list[embed_cache.Neighbour]:
        asked.append((doc, model, limit))
        return [embed_cache.Neighbour(document="other.md", similarity=0.42)]

    async def tiny(_settings: object) -> catalogue.EmbeddingModel:
        return catalogue.EmbeddingModel("test/tiny", 4)

    monkeypatch.setattr(catalogue, "embedding_model", tiny)
    monkeypatch.setattr(embed_cache, "nearest", nearest)
    under_model = await client.get("/api/documents/guide.md/similar")

    assert under_model.json() == {"nearest": [{"document": "other.md", "similarity": 0.42}]}
    tiny_model = catalogue.EmbeddingModel("test/tiny", 4).cache_name
    assert asked == [(guide["id"], tiny_model, 3)], "asked by id, answered by name"
    assert (await client.get("/api/documents/ghost.md/similar")).status_code == 404


async def test_attach_list_and_detach_a_member(
    client: AsyncTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The detach answers once its removal is queued: the removal waits its turn on the
    collection's single writer, and the member reads `removing` until it ran."""
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "notes"})
    await stage_and_import(client, "guide.md", MD.encode())

    await attach_via_api(client, "notes", "guide.md")

    (member,) = (await client.get("/api/collections/notes/documents")).json()["items"]
    assert (member["document"]["name"], member["status"]) == ("guide.md", "indexed")
    assert member["added_at"] > 0
    assert (await client.get("/api/collections/notes")).json()["counts"]["indexed"] == 1
    assert (await client.get("/api/documents/guide.md/collections")).json() == ["notes"]
    cached = (await client.get("/api/documents/guide.md/embeddings")).json()
    guide = (await client.get("/api/documents/guide.md")).json()
    assert [entry["document_id"] for entry in cached] == [guide["id"]] * len(cached)
    assert cached, "indexing the member filled the document's embedding cache"

    in_notes = {"q": "lancedb", "collections": "notes"}
    (hit,) = (await client.get("/api/search/explore", params=in_notes)).json()
    assert (hit["collection"], hit["document"]) == ("notes", "guide.md")

    removal = Gate()
    monkeypatch.setattr(
        CollectionIndex, "delete_document", removal.wrap(CollectionIndex.delete_document)
    )

    detached = await client.delete("/api/collections/notes/documents/guide.md")

    assert detached.status_code == 204, detached.text
    assert await wait_event(removal.entered), "the removal never ran"
    removing = (
        await client.get("/api/collections/notes/documents", params={"status": "removing"})
    ).json()["items"]
    assert [(m["document"]["name"], m["status"]) for m in removing] == [("guide.md", "removing")]
    assert (await client.get("/api/collections/notes")).json()["counts"]["active"] == 1
    removal.release.set()

    async def gone() -> bool:
        return (await client.get("/api/collections/notes/documents")).json()["items"] == []

    await until(gone, "the removal never took the membership")
    assert (await client.get("/api/collections/notes")).json()["counts"]["active"] == 0
    assert (await client.get("/api/documents/guide.md/collections")).json() == []
    assert (await client.get("/api/search/explore", params=in_notes)).json() == []
    assert (await client.get("/api/documents/guide.md")).json()["status"] == "imported", (
        "a detach takes nothing from the document"
    )


async def test_one_document_serves_two_collections(client: AsyncTestClient) -> None:
    """A document is imported once and held by many collections."""
    await client.post("/api/init", json=NO_MODELS)
    for name in ("alpha", "beta"):
        await client.post("/api/collections", json={"name": name})
    await stage_and_import(client, "guide.md", MD.encode())

    for name in ("alpha", "beta"):
        await attach_via_api(client, name, "guide.md")

    assert (await client.get("/api/documents/guide.md/collections")).json() == ["alpha", "beta"]
    for name in ("alpha", "beta"):
        found = (
            await client.get("/api/search/explore", params={"q": "lancedb", "collections": name})
        ).json()
        assert [hit["collection"] for hit in found] == [name]
    assert len((await client.get("/api/documents/guide.md/embeddings")).json()) == 1, (
        "identical chunk settings, so both collections read one cache entry"
    )


async def test_the_operations_listing_names_its_sections_and_its_activity(
    client: AsyncTestClient,
) -> None:
    """The three collection-free routes answer beside `/api/operations/{id}`: `kinds` and
    `activity` are their own path segments, so neither is read as an operation id."""
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "notes"})
    await stage_and_import(client, "guide.md", MD.encode())
    await attach_via_api(client, "notes", "guide.md")

    kinds = await client.get("/api/operations/kinds")
    activity = await client.get("/api/operations/activity")

    assert kinds.status_code == 200, kinds.text
    assert [k["kind"] for k in kinds.json()] == [
        "document",
        "collection",
        "download",
        "maintenance",
    ]
    assert all(k["label"] and k["active"] >= 0 for k in kinds.json())
    assert activity.status_code == 200, activity.text
    assert set(activity.json()) == {"operations", "tasks"}
    assert all(set(counts) == {"queued", "running"} for counts in activity.json().values())
    listed = (await client.get("/api/operations", params={"kind": "document"})).json()["items"]
    (indexed,) = [row for row in listed if row["title"] == "notes / guide.md"]
    assert [job["stage"] for job in indexed["jobs"]] == ["embed", "index"]
    assert (await client.get(f"/api/jobs/{indexed['jobs'][-1]['id']}/tasks")).status_code == 200


async def test_deleting_a_collection_keeps_its_documents(client: AsyncTestClient) -> None:
    await client.post("/api/init", json=NO_MODELS)
    for name in ("kept", "dropped"):
        await client.post("/api/collections", json={"name": name})
    await stage_and_import(client, "guide.md", MD.encode())
    for name in ("kept", "dropped"):
        await attach_via_api(client, name, "guide.md")

    deleted = await client.delete("/api/collections/dropped")

    assert deleted.status_code == 202, "the deletion is queued, not done in the request"
    operation_id = deleted.json()["operation_id"]
    await wait_for(operation_id)
    progress = (await client.get(f"/api/operations/{operation_id}/progress")).json()
    assert (progress["id"], progress["kind"], progress["status"]) == (
        operation_id,
        "delete_collection",
        "SUCCESS",
    )
    assert progress["collection"] == "dropped"
    assert [c["name"] for c in (await client.get("/api/collections")).json()["items"]] == ["kept"]
    assert (await client.get("/api/documents/guide.md")).json()["status"] == "imported"
    assert (await client.get("/api/documents/guide.md/collections")).json() == ["kept"]
    kept = {"q": "lancedb", "collections": "kept"}
    assert (await client.get("/api/search/explore", params=kept)).json()


async def test_renaming_a_collection_moves_everything_that_names_it(
    client: AsyncTestClient,
) -> None:
    """Members, settings, description, the index and every session that chose it follow the new
    name; the old name is gone."""
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "notes", "description": "what I read"})
    await client.put("/api/collections/notes/overrides", json={"search": {"limit": 7}})
    await stage_and_import(client, "guide.md", MD.encode())
    await attach_via_api(client, "notes", "guide.md")
    await client.put("/api/sessions/s1", json={"collections": ["notes"]})

    renamed = await client.put("/api/collections/notes/name", json={"name": "read later"})

    assert renamed.status_code == 200, renamed.text
    info = renamed.json()
    assert (info["name"], info["description"]) == ("read-later", "what I read"), "a safe name"
    assert (info["overrides"]["search"]["limit"], info["counts"]["indexed"]) == (7, 1)
    listed = (await client.get("/api/collections")).json()["items"]
    assert [one["name"] for one in listed] == ["read-later"]
    assert (await client.get("/api/collections/notes")).status_code == 404
    assert (await client.get("/api/documents/guide.md/collections")).json() == ["read-later"]
    assert (await client.get("/api/sessions")).json()[0]["collections"] == ["read-later"]
    found = (await client.get("/api/search/explore", params=_explore("s1", "lancedb"))).json()
    assert [(hit["collection"], hit["document"]) for hit in found] == [("read-later", "guide.md")]


@pytest.mark.parametrize(
    ("name", "path", "body", "status", "detail"),
    [
        ("the same name is a no-op", "notes", "notes", 200, None),
        ("a name taken is refused", "notes", "other", 409, "collection already exists: other"),
        ("a missing collection", "ghost", "spirit", 404, "collection not found: ghost"),
        ("a name with nothing safe in it", "notes", "***", 422, "invalid name"),
    ],
)
async def test_a_rename_refused_changes_nothing(
    client: AsyncTestClient, name: str, path: str, body: str, status: int, detail: str | None
) -> None:
    await client.post("/api/init", json=NO_MODELS)
    for existing in ("notes", "other"):
        await client.post("/api/collections", json={"name": existing})

    response = await client.put(f"/api/collections/{path}/name", json={"name": body})

    assert response.status_code == status, f"{name}: {response.text}"
    if detail is not None:
        assert detail in response.json()["detail"], name
    listed = (await client.get("/api/collections")).json()["items"]
    assert [one["name"] for one in listed] == ["notes", "other"], name


@pytest.mark.parametrize(
    ("claim", "method", "path", "body", "left"),
    [
        ("create", "POST", "/api/collections", {"name": "gone"}, ["gone", "other"]),
        ("rename", "PUT", "/api/collections/other/name", {"name": "gone"}, ["gone"]),
    ],
)
async def test_a_name_whose_delete_still_runs_can_be_taken(
    client: AsyncTestClient,
    monkeypatch,
    claim: str,
    method: str,
    path: str,
    body: dict,
    left: list[str],
) -> None:
    """The delete moves the folder aside before it frees the name, so a create or a rename onto
    the name while the delete still runs lands in a folder of its own, which the delete's last
    step leaves alone."""
    await client.post("/api/init", json=NO_MODELS)
    for existing in ("gone", "other"):
        await client.post("/api/collections", json={"name": existing})
    gate = Gate()
    monkeypatch.setattr(Collection, "remove_aside", gate.wrap(Collection.remove_aside))
    deleting = (await client.delete("/api/collections/gone")).json()["operation_id"]
    assert await wait_event(gate.entered), "the delete reached its last step"

    taken = await client.request(method, path, json=body)

    assert taken.status_code in (200, 201), f"{claim}: {taken.text}"
    gate.release.set()
    await wait_for(deleting)
    listed = (await client.get("/api/collections")).json()["items"]
    assert [one["name"] for one in listed] == left, claim
    assert Collection("gone").root.is_dir(), f"{claim}: the folder the claim made stays"
    assert [
        p.name for p in Collection("gone").root.parent.iterdir() if p.name.startswith(".")
    ] == [], f"{claim}: the folder the delete moved aside is gone"


@pytest.mark.parametrize(
    ("change", "method", "path", "body", "expect", "not_expect"),
    [
        (
            "create",
            "POST",
            "/api/collections",
            {"name": "adr", "description": "ADRs."},
            "adr: ADRs",
            None,
        ),
        (
            "describe",
            "PUT",
            "/api/collections/notes/description",
            {"description": "Field notes."},
            "notes: Field notes",
            None,
        ),
        ("rename", "PUT", "/api/collections/notes/name", {"name": "journal"}, "journal", "notes"),
        ("delete", "DELETE", "/api/collections/notes", None, "currently other", "notes"),
    ],
)
async def test_a_collection_change_refreshes_every_installation(
    client: AsyncTestClient,
    tmp_path: Path,
    change: str,
    method: str,
    path: str,
    body: dict | None,
    expect: str,
    not_expect: str | None,
) -> None:
    """The skill and rule name the collections, so each change rewrites them where `install
    claude` put them, in the background: the request never waits on it."""
    await client.post("/api/init", json=NO_MODELS)
    directory = claude_installed(tmp_path / "project" / ".claude")
    await claude.record_installation(directory)
    for name in ("notes", "other"):
        await client.post("/api/collections", json={"name": name})
    skill, rule = claude.skill_path(directory), claude.rule_path(directory)

    async def names(text: str) -> bool:
        return all(path.is_file() and text in path.read_text() for path in (skill, rule))

    await until(lambda: names("notes; other"), "the creates reached the skill and rule")

    response = await client.request(method, path, json=body)
    assert response.status_code in (200, 201, 202), f"{change}: {response.text}"
    if method == "DELETE":
        await wait_for(response.json()["operation_id"])

    await until(lambda: names(expect), f"{change}: the change reached the skill and rule")
    if not_expect is not None:
        assert f"{not_expect};" not in skill.read_text(), f"{change}: the old name is gone"
        assert f"currently {not_expect}" not in rule.read_text(), f"{change}: the old name is gone"


async def test_deleting_a_document_removes_it_from_every_collection(
    client: AsyncTestClient,
) -> None:
    await client.post("/api/init", json=NO_MODELS)
    for name in ("alpha", "beta"):
        await client.post("/api/collections", json={"name": name})
    await stage_and_import(client, "guide.md", MD.encode())
    for name in ("alpha", "beta"):
        await attach_via_api(client, name, "guide.md")

    deleted = await client.delete("/api/documents/guide.md")

    assert deleted.status_code == 202, deleted.text
    await wait_for(deleted.json()["operation_id"])
    assert (await client.get("/api/documents")).json()["items"] == []
    assert (await client.get("/api/documents/guide.md")).status_code == 404
    for name in ("alpha", "beta"):
        assert (await client.get(f"/api/collections/{name}/documents")).json()["items"] == []
        assert (await client.get(f"/api/collections/{name}")).json()["counts"]["total"] == 0
        scoped = {"q": "lancedb", "collections": name}
        found = await client.get("/api/search/explore", params=scoped)
        assert found.json() == [], "the rows go from every collection's index too"


async def test_reading_one_document(client: AsyncTestClient) -> None:
    """The viewer panes, which are document-scoped now: no collection appears in any of them."""
    await client.post("/api/init", json=NO_MODELS)
    await stage_and_import(client, "guide.md", MD.encode())

    streamed = await client.get("/api/documents/guide.md/markdown?full=true")
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

    preview = await client.get("/api/documents/guide.md/preview")
    assert preview.status_code == 200 and preview.text == MD
    source = await client.get("/api/documents/guide.md/source")
    assert source.status_code == 200 and source.text == MD

    described = await client.put(
        "/api/documents/guide.md/description", json={"description": "the guide"}
    )
    assert described.json()["description"] == "the guide"


ATTACK_HTML = b"<html><body><script>fetch('/api/documents')</script>page</body></html>"
ATTACK_SVG = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'


@pytest.mark.parametrize(
    ("name", "body", "sandboxed"),
    [
        pytest.param("page.html", ATTACK_HTML, True, id="html-runs-no-script"),
        pytest.param("logo.svg", ATTACK_SVG, True, id="svg-runs-no-script"),
        pytest.param("guide.md", MD.encode(), True, id="text-is-sandboxed-too"),
        pytest.param("paper.pdf", text_pdf(["one"]), False, id="pdf-keeps-its-viewer"),
    ],
)
async def test_a_documents_own_bytes_are_served_sandboxed(
    client: AsyncTestClient, name: str, body: bytes, sandboxed: bool
) -> None:
    """The source and the preview carry the file's own bytes on haskie's origin, where a script
    in them would reach an API that authenticates no one: `sandbox` takes the origin away."""
    await client.post("/api/init", json=NO_MODELS)
    await stage_and_import(client, name, body)

    for route in ("source", "preview"):
        response = await client.get(f"/api/documents/{name}/{route}")

        assert response.status_code == 200, (route, response.text)
        csp = response.headers.get("content-security-policy")
        assert csp == ("sandbox" if sandboxed else None), route
        nosniff = response.headers.get("x-content-type-options")
        assert nosniff == ("nosniff" if sandboxed else None), route


@pytest.mark.parametrize(
    ("name", "params", "status", "text"),
    [
        (
            "a heading and its paragraph",
            {"line_start": 5, "line_end": 7},
            200,
            "## Alpha\n\nalpha body about lancedb",
        ),
        ("one line", {"line_start": 3, "line_end": 3}, 200, "intro text"),
        ("past the end reads what is there", {"line_start": 11, "line_end": 30}, 200, "beta body"),
        ("an end before the start", {"line_start": 7, "line_end": 5}, 422, None),
        ("line zero: lines count from 1", {"line_start": 0, "line_end": 3}, 422, None),
        ("more than a pointer's worth", {"line_start": 1, "line_end": 401}, 422, None),
    ],
)
async def test_reading_lines_of_one_document(
    client: AsyncTestClient, name: str, params: dict, status: int, text: str | None
) -> None:
    """What the web UI reads when an `also_in` place is opened: its lines, and no more."""
    await client.post("/api/init", json=NO_MODELS)
    await stage_and_import(client, "guide.md", MD.encode())

    response = await client.get("/api/documents/guide.md/lines", params=params)

    assert response.status_code == status, f"{name}: {response.text}"
    if text is not None:
        assert response.json() == {"text": text}, name


async def test_reading_lines_of_a_document_nobody_imported(client: AsyncTestClient) -> None:
    await client.post("/api/init", json=NO_MODELS)

    response = await client.get(
        "/api/documents/ghost.md/lines", params={"line_start": 1, "line_end": 2}
    )

    assert response.status_code == 404


# --- search over several collections ---------------------------------------------------


def _explore(session_id: str, q: str, granularity: str = "chunk") -> dict[str, str]:
    """The query arguments of one exploration, at the granularity the web UI asks for."""
    return {"session_id": session_id, "q": q, "granularity": granularity}


async def test_session_search_returns_each_passage_once(client: AsyncTestClient) -> None:
    """A document in two collections of one session is two copies of the same chunk. A caller
    searching passages wants it once, credited to the first collection it chose."""
    await client.post("/api/init", json=NO_MODELS)
    for name in ("alpha", "beta"):
        await client.post("/api/collections", json={"name": name})
    await stage_and_import(client, "shared.md", b"# Shared\n\nshared token about lancedb\n")
    await stage_and_import(client, "only-beta.md", b"# Beta only\n\nanother shared token here\n")
    for name in ("alpha", "beta"):
        await attach_via_api(client, name, "shared.md")
    await attach_via_api(client, "beta", "only-beta.md")

    assert (
        await client.put("/api/sessions/s1", json={"collections": ["alpha", "beta"]})
    ).json() == [
        "alpha",
        "beta",
    ]
    hits = (await client.get("/api/search/explore", params=_explore("s1", "shared"))).json()

    identity = [(h["document"], h["seq"]) for h in hits]
    assert len(set(identity)) == len(identity), "no passage is returned twice"
    assert {h["document"] for h in hits} == {"shared.md", "only-beta.md"}
    shared = next(h for h in hits if h["document"] == "shared.md")
    assert shared["collection"] == "alpha", "the first collection of the session is credited"


SEARCH_PATHS: dict[str, tuple[str, dict, str]] = {
    # name -> (route, extra arguments, key of the list of results in the answer; "" for the root)
    "explore chunks": ("/api/search/explore", {"granularity": "chunk"}, ""),
    "explore passages": ("/api/search/explore", {"granularity": "passage"}, ""),
    "excerpts": ("/api/search/excerpts", {}, "excerpts"),
    "sections": ("/api/search/sections", {}, "sections"),
    "text": ("/api/search/text", {}, "items"),
}


@pytest.mark.parametrize("path", list(SEARCH_PATHS))
@pytest.mark.parametrize(
    ("leaving", "found", "sources"),
    [
        (
            "nothing",
            {("notes", "guide.md"), ("notes", "keep.md")},
            {("notes", "guide.md"), ("other", "guide.md"), ("notes", "keep.md")},
        ),
        (
            "guide.md removing from notes",
            {("other", "guide.md"), ("notes", "keep.md")},
            {("other", "guide.md"), ("notes", "keep.md")},
        ),
        ("guide.md deleting", {("notes", "keep.md")}, {("notes", "keep.md")}),
    ],
)
async def test_a_document_on_its_way_out_answers_no_search(
    client: AsyncTestClient,
    tmp_path: Path,
    path: str,
    leaving: str,
    found: set[tuple[str, str]],
    sources: set[tuple[str, str]],
) -> None:
    """A detach or a document delete answers once its removal is queued, and the rows stay in the
    table until it ran: a membership `removing` does not answer from that collection, and a
    document `deleting` from none. `guide.md` sits in both collections, so it still answers from
    the one it is not leaving; the map's `documents` list every collection holding a document,
    and a collection it is leaving does not hold it any more."""
    await client.post("/api/init", json=NO_MODELS)
    for name in ("notes", "other"):
        await client.post("/api/collections", json={"name": name})
    await stage_and_import(client, "guide.md", b"# Guide\n\nA guide to lancedb tables.\n")
    await stage_and_import(client, "keep.md", b"# Keep\n\nWhy lancedb keeps its old versions.\n")
    for collection, name in (("notes", "guide.md"), ("other", "guide.md"), ("notes", "keep.md")):
        await attach_via_api(client, collection, name)
    if leaving == "guide.md removing from notes":
        await Collection("notes").start_removal(await id_of("guide.md"))
    elif leaving == "guide.md deleting":
        await document.set_status(await id_of("guide.md"), DocumentStatus.DELETING)

    route, extra, key = SEARCH_PATHS[path]
    response = await client.get(
        route, params={"q": "lancedb", "collections": "notes,other", **extra}
    )
    assert response.status_code == 200, response.text
    results = response.json()[key] if key else response.json()

    assert {(row["collection"], row["document"]) for row in results} == found, f"{path}, {leaving}"
    if path == "sections":
        books = response.json()["documents"]
        held = {(one, row["document"]) for row in books for one in row["collections"]}
        assert held == sources, f"{path}, {leaving}"


async def test_documents_are_listed_with_their_collections(ready: AsyncTestClient) -> None:
    """The gallery names the collections holding each document, by name; a member of none has
    none."""
    await ready.post("/api/collections", json={"name": "archive"})
    await ready.post("/api/collections/archive/documents", json={"document": "guide.md"})
    held = ["archive", "notes"]
    items = (await ready.get("/api/documents")).json()["items"]
    listed = {row["name"]: row["collections"] for row in items}
    assert listed == {"guide.md": held, "pending.md": []}
    assert (await ready.get("/api/documents/guide.md")).json()["collections"] == held


async def test_session_history_holds_every_action_newest_first(
    ready: AsyncTestClient, tmp_path: Path
) -> None:
    """Every tool call that names a `session_id` leaves one event: what it did, to what, and the
    job it started. The first action under an id creates the session, so an import made before
    any selection belongs to the conversation too; a call without a session leaves nothing."""
    (tmp_path / "later.md").write_text("# Later\n\nalpha again\n")
    imported = await ready.post(
        "/api/documents/import",
        params={"session_id": "s2"},
        json={"path": str(tmp_path / "later.md")},
    )
    assert imported.status_code == 201, imported.text
    # the route answers at `queued` and converts in the background; only an imported document
    # joins a collection, and forcing the status here would race the pipeline writing its own
    assert (await wait_import(ready, "later.md"))["status"] == "imported"
    attached = await ready.post(
        "/api/collections/notes/documents",
        params={"session_id": "s2"},
        json={"document": "later.md"},
    )
    assert attached.status_code == 202, attached.text
    await ready.put("/api/sessions/s2", json={"collections": ["notes"]})
    hits = (await ready.get("/api/search/explore", params=_explore("s2", "alpha"))).json()
    await ready.get("/api/search/text", params={"session_id": "s2", "q": "alpha"})
    await ready.get("/api/search/text", params={"q": "alpha"})  # no session: no event
    await ready.put(
        "/api/documents/later.md/description",
        params={"session_id": "s2"},
        json={"description": "the later one"},
    )
    assert (
        await ready.delete("/api/collections/notes/documents/later.md", params={"session_id": "s2"})
    ).status_code == 204

    history = (await ready.get("/api/sessions/s2/history")).json()
    assert [(row["action"], row["subject"]) for row in history] == [
        ("detach", "later.md"),
        ("describe", "later.md"),
        ("search", "alpha"),
        ("search", "alpha"),
        ("collections", "notes"),
        ("attach", "later.md"),
        ("import", "later.md"),
    ], "newest first, and the unsessioned search is absent"
    by_action = {(row["action"], row["detail"].get("scope")): row for row in history}
    session_search = by_action[("search", "explore")]
    assert session_search["detail"]["hits"] == len(hits) > 0
    assert session_search["detail"]["documents"] == list(
        dict.fromkeys(hit["document"] for hit in hits)
    )
    assert by_action[("search", "text")]["detail"]["hits"] > 0
    assert by_action[("import", None)]["operation_id"] == history[-1]["operation_id"] is not None
    assert by_action[("attach", None)]["operation_id"] == attached.json()["operation_id"]
    assert by_action[("attach", None)]["detail"] == {"collection": "notes"}
    assert by_action[("collections", None)]["detail"] == {"collections": ["notes"]}
    assert all(row["duration_ms"] >= 0 and row["ts"] > 0 for row in history)
    listed = {one["id"]: one for one in (await ready.get("/api/sessions")).json()}
    assert listed["s2"]["collections"] == ["notes"], "the import created it, the put filled it"
    assert listed["s2"]["last_at"] == history[0]["ts"], "last seen at its newest event"
    assert listed["s1"]["last_at"] is not None, "the fixture put its selection: an event too"
    assert (await ready.get("/api/sessions/never-set/history")).json() == []

    (tmp_path / "plain.md").write_text("# Plain\n")
    await ready.post("/api/documents/import", json={"path": str(tmp_path / "plain.md")})
    listed = (await ready.get("/api/operations", params={"kind": "document"})).json()["items"]
    origins = {row["id"]: row["origin"] for row in listed}
    assert origins[attached.json()["operation_id"]] == "s2"
    assert origins[history[-1]["operation_id"]] == "s2"
    assert [row["origin"] for row in listed if "plain.md" in row["title"]] == [None], (
        "an import from the web UI came from nobody's session"
    )


async def _converted(name: str, markdown: str) -> None:
    """The markdown a conversion would have written: what an excerpt is widened against."""
    (await document.named(name)).markdown.write_text(markdown)


async def test_every_search_is_logged_with_what_it_returned(ready: AsyncTestClient) -> None:
    """Every search endpoint writes one row to the search log: with or without a session, failed
    or not, each question on its own. A later page of a full-text search is the same search, and
    writes nothing."""
    await _converted("guide.md", MD)
    await ready.get("/api/search/excerpts", params={"q": "alpha", "session_id": "s3"})
    await ready.get(
        "/api/search/excerpts",
        params={"q": ["alpha body", "zebra stripes"], "context": "notes", "session_id": "s3"},
    )
    await ready.get("/api/search/sections", params={"q": "alpha"})
    first = (await ready.get("/api/search/text", params={"q": "alpha", "page_size": 1})).json()
    assert first["next_cursor"] is not None, "one chunk, a page of one: the walk could go on"
    await ready.get(
        "/api/search/text", params={"q": "alpha", "page_size": 1, "cursor": first["next_cursor"]}
    )
    await ready.get("/api/search/explore", params={"q": "alpha"})
    failed = await ready.get(
        "/api/search/excerpts", params={"q": "alpha", "collections": "ghost", "session_id": "s3"}
    )
    assert failed.status_code == 404

    logged = await log.load()

    assert [(one.tool, one.session_id, one.result_count) for one in logged] == [
        (log.Tool.EXCERPTS, "s3", 0),
        (log.Tool.EXPLORE, None, 1),
        (log.Tool.TEXT, None, 1),
        (log.Tool.SECTIONS, None, 1),
        (log.Tool.EXCERPTS, "s3", 1),
        (log.Tool.EXCERPTS, "s3", 1),
    ], "newest first, the second page left out"
    several = logged[-2]
    assert [(one.question, one.uncovered) for one in several.questions] == [
        ("alpha body", False),
        ("zebra stripes", True),
    ], "each question, and the one no excerpt answers"
    assert several.context == "notes"
    assert logged[0].error == "NotFound: collection not found: ghost"
    assert all(one.error is None for one in logged[1:])
    assert all(one.collections == ["notes"] and one.mode == "fts" for one in logged[1:])
    assert {one.actor for one in logged} == {"web"}
    assert logged[-1].result_limit == flow.DEFAULT_EXCERPTS, "the excerpts' default, resolved"
    (result,) = (await log.top_results([logged[-1].id], 5))[logged[-1].id]
    assert (result.document, result.parent) == ("guide.md", None)
    history = (await ready.get("/api/sessions/s3/history")).json()
    assert [(row["subject"], row["detail"]) for row in history] == [
        (
            "alpha",
            {
                "scope": "excerpts",
                "hits": 0,
                "documents": [],
                "error": "NotFound: collection not found: ghost",
            },
        ),
        (
            "alpha body | zebra stripes",
            {
                "scope": "excerpts",
                "hits": 1,
                "documents": ["guide.md"],
                "questions": ["alpha body", "zebra stripes"],
                "context": "notes",
            },
        ),
        ("alpha", {"scope": "excerpts", "hits": 1, "documents": ["guide.md"]}),
    ], "a session's searches are its history, failed ones too"


async def test_gaps_group_review_and_replay(ready: AsyncTestClient, tmp_path: Path) -> None:
    """Questions that found nothing come back as one topic per question, each on its own when a
    search asked several; the curator dismisses one, and a replay shows the gap closing once a
    document answers it."""
    await _converted("guide.md", MD)
    await ready.get("/api/search/excerpts", params={"q": "zebra stripes", "session_id": "g1"})
    await ready.get(
        "/api/search/excerpts", params={"q": ["alpha body", "Zebra stripes?"], "session_id": "g2"}
    )
    await ready.get("/api/search/excerpts", params={"q": "alpha", "session_id": "g1"})
    await ready.get("/api/search/excerpts", params={"q": "zebra", "collections": "ghost"})

    (topic,) = (await ready.get("/api/gaps")).json()

    assert topic["question"] == "Zebra stripes?", "named by its newest question"
    assert [(one["question"], one["signal"]) for one in topic["questions"]] == [
        ("Zebra stripes?", "uncovered"),
        ("zebra stripes", "empty"),
    ], "one topic by their words; the answered questions and the failed search are no gaps"
    assert (topic["sessions"], topic["collections"]) == (2, ["notes"])
    newest, oldest = (one["id"] for one in topic["questions"])

    reviewed = await ready.put("/api/gaps/review", json={"ids": [oldest], "review": "dismissed"})
    assert reviewed.json() == 1
    (still,) = (await ready.get("/api/gaps")).json()
    assert [one["id"] for one in still["questions"]] == [newest]
    (dismissed,) = (await ready.get("/api/gaps", params={"review": "dismissed"})).json()
    assert [one["id"] for one in dismissed["questions"]] == [oldest]
    assert (await ready.get("/api/gaps", params={"review": "resolved"})).json() == []

    replayed = (await ready.post("/api/gaps/replay", json={"ids": [newest]})).json()
    assert [(one["id"], one["signal"], one["result_count"]) for one in replayed] == [
        (newest, "empty", 0)
    ]
    source = tmp_path / "zebra.md"
    source.write_text("# Zebra\n\nzebra stripes run across the flank\n")
    imported = await document.import_path(str(source))
    await document.set_status(imported.id, DocumentStatus.IMPORTED)
    await Collection("notes").add(imported.id)
    await Collection("notes").set_member_status(imported.id, MemberStatus.INDEXED)
    await seed_index("notes", imported.name, "zebra stripes run across the flank")
    await _converted(imported.name, "zebra stripes run across the flank\n")
    (closed,) = (await ready.post("/api/gaps/replay", json={"ids": [newest]})).json()
    assert (closed["signal"], closed["results"][0]["document"]) == (None, "zebra.md")
    assert len(await log.load()) == 4, "a replay is not a search anyone made"

    await ready.put("/api/gaps/review", json={"ids": [newest], "review": "resolved"})
    assert (await ready.get("/api/gaps")).json() == []
    await ready.put("/api/gaps/review", json={"ids": [newest, oldest], "review": "open"})
    assert len((await ready.get("/api/gaps")).json()[0]["questions"]) == 2, "reopened"

    too_many = await ready.post("/api/gaps/replay", json={"ids": list(range(51))})
    assert too_many.status_code == 422 and "at most 50" in too_many.text
    no_window = await ready.get("/api/gaps", params={"days": 0})
    assert no_window.status_code == 422 and "Expected `int` >= 1" in no_window.text


async def test_an_agent_reports_a_gap_on_a_question_it_asked(ready: AsyncTestClient) -> None:
    """A verdict lands on the newest question of that session with those words, and makes it a
    gap whatever its scores say; a question it did not ask, or whose search failed, is refused."""
    await _converted("guide.md", MD)
    await ready.get("/api/search/excerpts", params={"q": "alpha body", "session_id": "r1"})
    await ready.get("/api/search/excerpts", params={"q": "alpha body", "session_id": "r2"})
    assert (await ready.get("/api/gaps")).json() == [], "answered: no gap yet"

    reported = await ready.post(
        "/api/gaps/report",
        params={"session_id": "r1"},
        json={"question": " alpha body ", "verdict": "partial", "missing": "the retry limit"},
    )

    assert reported.status_code == 200, reported.text
    (topic,) = (await ready.get("/api/gaps")).json()
    (gap,) = topic["questions"]
    assert (gap["id"], gap["session_id"], gap["signal"]) == (
        reported.json()["id"],
        "r1",
        "reported",
    )
    assert (gap["agent_verdict"], gap["agent_note"]) == ("partial", "the retry limit")
    assert gap["result_count"] == 1, "the search did answer, by its scores"
    only = await ready.get("/api/gaps", params={"signals": ["empty", "weak"]})
    assert only.json() == [], "listed by the reasons asked for"
    again = await ready.get("/api/gaps", params={"signals": ["reported", "borderline"]})
    assert [one["questions"][0]["id"] for one in again.json()] == [gap["id"]]

    await ready.get(
        "/api/search/excerpts", params={"q": "zebra", "collections": "ghost", "session_id": "r1"}
    )
    for name, params, body, status, detail in [
        ("a question not asked", {"session_id": "r1"}, {"question": "beta"}, 404, "no search"),
        ("another session's", {"session_id": "r3"}, {"question": "alpha body"}, 404, "r3"),
        ("a failed search", {"session_id": "r1"}, {"question": "zebra"}, 409, "failed"),
        ("an empty question", {"session_id": "r1"}, {"question": "  "}, 422, "empty"),
        (
            "a long note",
            {"session_id": "r1"},
            {"question": "alpha body", "missing": "x" * 301},
            422,
            "300",
        ),
        ("no session", {}, {"question": "alpha body"}, 422, "session_id"),
    ]:
        response = await ready.post(
            "/api/gaps/report", params=params, json={"verdict": "insufficient", **body}
        )
        assert response.status_code == status and detail in response.text, (name, response.text)

    await ready.get("/api/search/sections", params={"q": " alpha sections ", "session_id": "r1"})
    mapped = await ready.post(
        "/api/gaps/report",
        params={"session_id": "r1"},
        json={"question": " alpha sections ", "verdict": "insufficient"},
    )
    assert mapped.status_code == 200, "a map's question, passed back word for word"

    async with db.connect() as conn:  # the search, asked just over an hour ago
        await conn.execute(update(searches).values(ts=time.time() - gaps.REPORT_WINDOW - 1))
    stale = await ready.post(
        "/api/gaps/report",
        params={"session_id": "r1"},
        json={"question": "alpha body", "verdict": "insufficient"},
    )
    assert stale.status_code == 404, "only a search of the last hour is reported on"


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/api/search/explore", {}),
        ("/api/search/sections", {}),
        ("/api/search/text", {}),
    ],
)
async def test_a_blank_question_is_refused_and_not_logged(
    ready: AsyncTestClient, path: str, params: dict
) -> None:
    """A blank search finds nothing, so it would show on the Gaps page as an unnamed gap."""
    refused = await ready.get(path, params={"q": "   ", **params})

    assert refused.status_code == 422 and "q is empty" in refused.text, refused.text
    assert await log.load() == []


async def test_session_search_survives_the_deletion_of_a_collection(
    client: AsyncTestClient,
) -> None:
    await client.post("/api/init", json=NO_MODELS)
    for name in ("kept", "dropped"):
        await client.post("/api/collections", json={"name": name})
        await stage_and_import(client, f"{name}.md", f"# {name}\n\nshared {name}\n".encode())
        await attach_via_api(client, name, f"{name}.md")
    await client.put("/api/sessions/s1", json={"collections": ["kept", "dropped"]})
    assert (
        len((await client.get("/api/search/explore", params=_explore("s1", "shared"))).json()) == 2
    )

    deleted = await client.delete("/api/collections/dropped")

    assert deleted.status_code == 202
    await wait_for(deleted.json()["operation_id"])

    search = await client.get("/api/search/explore", params=_explore("s1", "shared"))
    assert search.status_code == 200, "one deleted collection must not break every later search"
    assert [hit["collection"] for hit in search.json()] == ["kept"]
    assert [(s["id"], s["collections"]) for s in (await client.get("/api/sessions")).json()] == [
        ("s1", ["kept"])
    ]


# --- passages, excerpts and sources -----------------------------------------------------

# One paragraph, one sentence per line, so a passage over two chunks spans three lines. "lancedb"
# is in both of the chunks seeded below and in neither section around them, so the query reaches
# exactly the two chunks that are meant to merge.
PASSAGE_MD = (
    "# Guide\n"
    "\n"
    "## Retrieval\n"
    "\n"
    "One table holds every chunk of a lancedb document.\n"
    "A chunk overlaps the chunk before it, so one sentence can sit in two of them at once.\n"
    "Merging the consecutive chunks of a lancedb answer back into a passage is what a reader "
    "wants.\n"
    "The passage then begins and ends where a sentence does.\n"
    "\n"
    "## Elsewhere\n"
    "\n"
    "This section says nothing about tables at all.\n"
)


async def _markdown_of(doc: str) -> str:
    """The converted markdown of an imported document, as it is on disk: what a chunk's
    `char_start` and `char_end` are offsets into, and what a passage is read from."""
    row = await document.named(doc)
    return (home.HOME / row.relative(row.markdown)).read_text(encoding="utf-8")


def _chunk(markdown: str, start: str, end: str, heading: str = "Retrieval") -> Chunk:
    """One chunk cut out of `markdown` between two of its own substrings, with the offsets and
    line numbers that cut really has. `start` and `end` fall mid-sentence on purpose: a passage
    quotes its chunks as they were cut, and adds nothing around them."""
    char_start = markdown.index(start)
    char_end = markdown.index(end) + len(end)
    return Chunk(
        headings=["Guide", heading],
        frame=["Guide", heading],
        pieces=[Piece(PieceType.TEXT, markdown[char_start:char_end])],
        line_start=markdown.count("\n", 0, char_start) + 1,
        line_end=markdown.count("\n", 0, char_end) + 1,
        char_start=char_start,
        char_end=char_end,
        byte_start=len(markdown[:char_start].encode()),
        byte_end=len(markdown[:char_end].encode()),
    )


async def _member(collection: str, doc: str) -> None:
    """Make an imported document a member of a collection without running its index; the chunks
    are seeded by hand right after (as the `ready` fixture does with `seed_index`)."""
    found = await Collection.get(collection)
    await found.add(await id_of(doc))
    await found.set_member_status(await id_of(doc), MemberStatus.INDEXED)


async def _guide_with_two_chunks(client: AsyncTestClient) -> None:
    """One collection holding `guide.md` as two consecutive chunks that overlap each other, plus
    a third chunk of the section below that the query never matches."""
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "notes"})
    await stage_and_import(client, "guide.md", PASSAGE_MD.encode())
    markdown = await _markdown_of("guide.md")
    await _member("notes", "guide.md")
    await seed_chunks(
        "notes",
        "guide.md",
        [
            _chunk(markdown, "every chunk of a lancedb", "so one sentence can sit"),
            _chunk(markdown, "sit in two of them", "answer back into a"),
            _chunk(markdown, "This section says", "about tables at all.", heading="Elsewhere"),
        ],
    )


RANKING_STEPS = ["retrieve", "merge", "rerank", "hits"]
EXCERPT_STEPS = ["fold", "group", "budget", "probe_gaps", "fill", "quote", "rerank_excerpts"]


@pytest.mark.parametrize(
    ("name", "path", "params", "steps", "branches"),
    [
        (
            "chunks: the ranking, then the fold",
            "/api/search/explore",
            {"q": "lancedb", "granularity": "chunk"},
            ["plan", *RANKING_STEPS, "collapse_hits"],
            {},
        ),
        (
            "excerpts: the ranking, then passages merged, folded and read",
            "/api/search/excerpts",
            {"q": "lancedb"},
            ["plan", *RANKING_STEPS, "judge_thin", *EXCERPT_STEPS],
            {},
        ),
        (
            "excerpts of several questions: each ranked in its own branch, side by side",
            "/api/search/excerpts",
            {"q": ["lancedb", "beta body"]},
            ["plan", *EXCERPT_STEPS],
            {"Q1": [*RANKING_STEPS, "judge_thin"], "Q2": [*RANKING_STEPS, "judge_thin"]},
        ),
        (
            "sections: the ranking, then the map",
            "/api/search/sections",
            {"q": "lancedb"},
            ["plan", *RANKING_STEPS, "map_sections"],
            {},
        ),
    ],
)
async def test_a_search_answers_with_the_time_each_step_took(
    client: AsyncTestClient,
    name: str,
    path: str,
    params: dict,
    steps: list[str],
    branches: dict[str, list[str]],
) -> None:
    """`Server-Timing`, the W3C header browsers show beside a request: one entry per step, in the
    order the steps ran, each with how long it took and what it is. A step of one of several runs
    side by side names its branch; the branches finish interleaved, so each is in order alone."""
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "notes"})
    await stage_and_import(client, "guide.md", MD.encode())
    await attach_via_api(client, "notes", "guide.md")

    response = await client.get(path, params=params)

    assert response.status_code == 200, f"{name}: {response.text}"
    entries = [entry.strip().split(";") for entry in response.headers["server-timing"].split(",")]
    ran: dict[str | None, list[str]] = {}
    for entry in entries:
        branch = next((p.removeprefix("branch=") for p in entry if p.startswith("branch=")), None)
        ran.setdefault(branch, []).append(entry[0])
    assert ran == {None: steps, **branches}, name
    assert all(entry[1].startswith("dur=") and float(entry[1][4:]) >= 0 for entry in entries), name
    retrieved = next(entry for entry in entries if entry[0] == "retrieve")
    assert retrieved[2] == 'desc="LanceDB retrieval"', name


@pytest.mark.parametrize(
    ("name", "path", "params", "lineage"),
    [
        ("chunks keep their own score", "/api/search/explore", {"q": "lancedb"}, []),
        (
            "passages fold theirs",
            "/api/search/explore",
            {"q": "lancedb", "granularity": "passage"},
            ["fill_thin"],
        ),
        (
            "excerpts, and several questions say each rule once",
            "/api/search/excerpts",
            {"q": ["lancedb", "how are rows retrieved"]},
            ["judge_thin", "fold", "group"],
        ),
        ("sections", "/api/search/sections", {"q": "lancedb"}, ["map_sections"]),
    ],
)
async def test_a_search_answers_with_its_score_lineage(
    client: AsyncTestClient, name: str, path: str, params: dict, lineage: list[str]
) -> None:
    """`X-Score-Lineage`: each step that set or changed a score, in the order it ran, and how.
    JSON, percent-encoded, because the formulas are not Latin-1."""
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "notes"})
    await stage_and_import(client, "guide.md", MD.encode())
    await attach_via_api(client, "notes", "guide.md")

    response = await client.get(path, params=params)

    assert response.status_code == 200, f"{name}: {response.text}"
    steps = json.loads(unquote(response.headers["x-score-lineage"]))
    assert [one["step"] for one in steps] == ["retrieve", "merge", *lineage], name
    assert steps[0]["label"] == "LanceDB retrieval", name
    assert steps[0]["rule"].endswith("No embedding model, so every mode is BM25."), name
    assert steps[1]["rule"] == "One collection: its scores are kept.", name


async def test_a_request_that_searches_nothing_carries_no_timing(client: AsyncTestClient) -> None:
    response = await client.get("/api/status")

    assert "server-timing" not in response.headers
    assert "x-score-lineage" not in response.headers


async def test_explore_merges_consecutive_chunks_into_one_passage(
    client: AsyncTestClient,
) -> None:
    """Two chunks that sit next to each other in one document are one passage: stitched by their
    offsets, so the text they share is in it once, and nothing around them is added."""
    await _guide_with_two_chunks(client)

    chunks = (await client.get("/api/search/explore", params={"q": "lancedb"})).json()
    response = await client.get(
        "/api/search/explore", params={"q": "lancedb", "granularity": "passage"}
    )

    assert response.status_code == 200, response.text
    assert {hit["seq"] for hit in chunks} == {1, 2}, "only the two that mention lancedb match"
    (passage,) = response.json()
    assert (passage["seq_start"], passage["seq_end"]) == (1, 2), "the range the chunks cover"
    assert passage["text"].startswith("every chunk of a lancedb"), "starts where chunk 1 does"
    assert passage["text"].endswith("answer back into a"), "and ends where chunk 2 does"
    assert passage["text"].count("sit in two of them") == 1, "the shared text is not repeated"
    assert (passage["line_start"], passage["line_end"]) == (5, 7), "lines recounted for the text"
    assert passage["header"] == "Guide > Retrieval"
    assert passage["location"] == "guide.md L5-7", "written to be cited"
    assert passage["document"] == "guide.md" and passage["collection"] == "notes"
    assert passage["score"] > 0


async def test_an_excerpt_is_the_section_its_passages_share(client: AsyncTestClient) -> None:
    """The passages of one section come back as one excerpt: the section's heading path, each
    passage a span. Excerpts have one route, the one the MCP tool calls."""
    await _guide_with_two_chunks(client)

    passages = (
        await client.get("/api/search/explore", params={"q": "lancedb", "granularity": "passage"})
    ).json()
    route = await client.get("/api/search/excerpts", params={"q": "lancedb"})
    explored = await client.get(
        "/api/search/explore", params={"q": "lancedb", "granularity": "excerpt"}
    )
    answer = route.json()
    excerpts = answer["excerpts"]

    (passage,) = passages
    (excerpt,) = excerpts
    assert excerpt["header"] == "Guide > Retrieval", "the section under the title"
    assert excerpt["text"] == passage["text"], "one passage that is its section's only one"
    assert [(span["seq_start"], span["seq_end"]) for span in excerpt["spans"]] == [(1, 2)]
    assert excerpt["spans"][0]["location"] == passage["location"]
    assert (excerpt["line_start"], excerpt["line_end"]) == (
        passage["line_start"],
        passage["line_end"],
    )
    assert route.status_code == 200, route.text
    assert (answer["uncovered"], answer["missing_terms"]) == ([], [])
    assert explored.status_code == 422, "excerpts have a route of their own"
    nothing = await client.get("/api/search/excerpts", params={"q": "nothingmatchesthis"})
    assert nothing.json() == {
        "excerpts": [],
        "uncovered": [],
        "missing_terms": ["nothingmatchesthis"],
        "searched": ["notes"],
    }, "no hits is an answer, not an error, and it says which words the sources lack"


# One section with two paragraphs on "lancedb" and one between them that is not, then a second
# section on it: the chunker cuts every paragraph into a chunk of its own at this size.
SECTIONS_MD = (
    "# Guide\n\n"
    "## Storage\n\n"
    "Lancedb holds every chunk of a document in one table, with its vector and its offsets.\n\n"
    "The offsets are bytes as well as characters, so a reader can seek straight to a passage.\n\n"
    "Lancedb compacts the small fragments once the table has grown past a few thousand rows.\n\n"
    "## Search\n\n"
    "Lancedb answers a hybrid query by fusing the vector and the full-text lists by their rank.\n"
)


async def test_the_passages_of_one_section_come_back_as_one_excerpt(
    client: AsyncTestClient,
) -> None:
    """Two passages of one section are two results as passages, and one excerpt: the section,
    with `[…]` where the paragraph between them did not match. The other section is the other."""
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "notes"})
    await stage_and_import(client, "sections.md", SECTIONS_MD.encode())
    await _member("notes", "sections.md")
    chunks = split(
        await _markdown_of("sections.md"), ChunkSettings(chunk_size=200, chunk_merge_below=0)
    )
    await seed_chunks("notes", "sections.md", chunks)
    params = {"q": "lancedb", "limit": 2}

    passages = (
        await client.get("/api/search/explore", params={**params, "granularity": "passage"})
    ).json()
    response = await client.get("/api/search/excerpts", params=params)

    assert response.status_code == 200, response.text
    assert len(passages) == 2, "a passage each for the two best"
    by_header = {one["header"]: one for one in response.json()["excerpts"]}
    assert set(by_header) == {"Guide > Storage", "Guide > Search"}, "the limit counts sections"
    storage = by_header["Guide > Storage"]
    assert [(span["seq_start"], span["seq_end"]) for span in storage["spans"]] == [(1, 1), (3, 3)]
    assert storage["text"] == "\n\n".join([chunks[0].text, "[…]", chunks[2].text])
    assert (storage["seq_start"], storage["seq_end"]) == (1, 3)


# Six paragraphs of one section, every one on retries, longer and longer: a search that scans
# fewer than six chunks keeps only some of them, and the rest answer just as well.
FILL_MD = "# Guide\n\n## Retries\n\n" + "\n\n".join(
    "A retry " + "waits a little longer each time " * (count + 1) + "before it runs again."
    for count in range(6)
)


async def test_the_text_between_and_around_kept_passages_is_filled_when_it_answers(
    client: AsyncTestClient,
) -> None:
    """One excerpt asked for scans four chunks, so the ranking keeps four of the six paragraphs.
    The two it left out match the question as well as those it kept, so the excerpt reads the
    whole section, one passage with no `[…]`, while the passage granularity stays the ranking's."""
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "notes"})
    await stage_and_import(client, "fill.md", FILL_MD.encode())
    await _member("notes", "fill.md")
    chunks = split(
        await _markdown_of("fill.md"), ChunkSettings(chunk_size=300, chunk_merge_below=0)
    )
    await seed_chunks("notes", "fill.md", chunks)
    # short passages growing on their own would hide what the fill does
    overrides = {"search": {"min_passage_chars": 0}}
    assert (await client.put("/api/collections/notes/overrides", json=overrides)).is_success
    params = {"q": "retry", "limit": 1}

    scanned = (
        await client.get("/api/search/explore", params={**params, "granularity": "chunk"})
    ).json()
    response = await client.get("/api/search/excerpts", params=params)

    assert len(chunks) == 6, "a chunk per paragraph"
    assert len(scanned) == 1 and response.status_code == 200, response.text
    (excerpt,) = response.json()["excerpts"]
    assert [(span["seq_start"], span["seq_end"]) for span in excerpt["spans"]] == [(1, 6)]
    assert "[…]" not in excerpt["text"]
    assert excerpt["text"] == "\n\n".join(chunk.text for chunk in chunks)


# Orders, section after section, and one note on stock: a question about both ranks the order
# sections first, and "inventory" is in none of them.
ORDERS_MD = "# Orders\n\n" + "\n\n".join(
    f"## Step {step}\n\nAn order keeps its lines consistent at step {step}, and the order total "
    "follows the lines." + (" Reconciliation against the ledger runs here." if step == 5 else "")
    for step in range(1, 9)
)
STOCK_MD = "# Stock\n\n## Counts\n\nInventory counts drop when stock ships to a customer.\n"


async def test_a_word_no_excerpt_holds_is_searched_for_once_more(
    client: AsyncTestClient, caplog
) -> None:
    """The ranking fills the one slot with the order section on reconciliation, and "inventory"
    is not in it.
    The probe searches that word alone, and its best passage joins as one excerpt past the limit;
    the answer then lacks no word of the question."""
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "shop"})
    for name, body in (("orders.md", ORDERS_MD), ("stock.md", STOCK_MD)):
        await stage_and_import(client, name, body.encode())
        await attach_via_api(client, "shop", name)
    question = "How does order reconciliation against the ledger keep inventory consistent?"
    params = {"q": question, "limit": 1}

    with caplog.at_level(logging.INFO):
        response = await client.get("/api/search/excerpts", params=params)

    assert response.status_code == 200, response.text
    answer = response.json()
    probed = [r.msg for r in caplog.records if isinstance(r.msg, dict)]
    (logged,) = [msg for msg in probed if msg["event"] == "search_probe"]
    assert logged["terms"] == ["inventory"], "the only word the order sections lack"
    assert (logged["found"], logged["joined"]) == (True, False), "past the limit, apart"
    assert [one["document"] for one in answer["excerpts"]] == ["orders.md", "stock.md"]
    assert answer["excerpts"][0]["header"] == "Orders > Step 5"
    assert answer["missing_terms"] == []


async def test_the_probed_excerpt_comes_past_the_budget_the_sections_were_cut_to(
    client: AsyncTestClient, caplog
) -> None:
    """Two order sections rank, the budget keeps one of them, and "inventory" is in neither: the
    probe's passage still joins, past the budget, since the cut came before it."""
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "shop"})
    for name, body in (("orders.md", ORDERS_MD), ("stock.md", STOCK_MD)):
        await stage_and_import(client, name, body.encode())
        await attach_via_api(client, "shop", name)
    overrides = {"search": {"max_answer_chars": 60}}
    assert (await client.put("/api/collections/shop/overrides", json=overrides)).is_success
    question = "How does order reconciliation against the ledger keep inventory consistent?"

    with caplog.at_level(logging.INFO):
        response = await client.get("/api/search/excerpts", params={"q": question, "limit": 2})

    assert response.status_code == 200, response.text
    answer = response.json()
    logged = {r.msg["event"]: r.msg for r in caplog.records if isinstance(r.msg, dict)}
    assert logged["search_budget"]["cut"] == 1, "one order section cut to the budget"
    assert (logged["search_probe"]["found"], logged["search_probe"]["joined"]) == (True, False)
    assert [one["document"] for one in answer["excerpts"]] == ["orders.md", "stock.md"]
    probed = answer["excerpts"][-1]
    assert probed["score"] == 0.0 and [one["score"] for one in probed["spans"]] == [0.0], (
        "its BM25 score is on another scale than the ranked excerpt's: it scores 0, last"
    )
    assert answer["missing_terms"] == []


# A section that answers, and a lead-in of another whose table below it says nothing on the query:
# the lead-in matches the word "lancedb" alone, and the chunk it would grow into matches nothing.
THIN_MD = (
    "# Guide\n\n"
    "## Storage\n\n"
    "Lancedb stores every chunk of a document in one table, with its vector, its offsets and the "
    "heading path it sits under, and a full-text index over the framed text, so the words of a "
    "heading find every chunk under it and a search reads a chunk back by its own byte offsets, "
    "never the whole markdown document.\n\n"
    "## Rules\n\n"
    "Lancedb rules:\n\n"
    "| step | what happens |\n"
    "|---|---|\n"
    "| open | the table is opened once per search and shared by the parts |\n"
    "| read | rows come back in no order, so the caller sorts them |\n"
    "| write | a part is replaced whole, so a reader never sees half of one |\n"
    "| compact | small fragments merge once the table has grown enough |\n"
    "| drop | an outdated table is dropped and rebuilt from the cache |\n"
)


async def _guide_with_a_lead_in(client: AsyncTestClient) -> list[Chunk]:
    """`thin.md` in `notes`, cut by the real chunker: the storage section, the lead-in right after
    its heading, and the table under the lead-in."""
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "notes"})
    await stage_and_import(client, "thin.md", THIN_MD.encode())
    await _member("notes", "thin.md")
    chunks = split(await _markdown_of("thin.md"), ChunkSettings(chunk_size=400))
    await seed_chunks("notes", "thin.md", chunks)
    return chunks


async def test_a_collection_reads_the_rows_of_named_chunks(client: AsyncTestClient) -> None:
    """What a search reads to look at the neighbours of a chunk it matched: the stored rows by
    (document, seq), and nothing for a chunk or a name it does not hold."""
    chunks = await _guide_with_a_lead_in(client)
    index = Collection("notes").index_with(None)

    thin = await id_of("thin.md")
    rows = await index.rows_at([(thin, 3), (thin, 2), (thin, 99)], vectors=True)
    quoted = await index.rows_at([("o'brien", 1)], vectors=False)

    assert sorted((row["document_id"], row["seq"]) for row in rows) == [(thin, 2), (thin, 3)]
    assert {row["seq"]: row["text"] for row in rows}[3] == chunks[2].text
    assert quoted == [], "a quote in a name is a literal, not the end of the filter"
    assert all("vector" not in row for row in rows), "a table without vectors has none to read"


async def test_an_excerpt_too_short_to_stand_alone_goes_when_nothing_around_it_matches(
    client: AsyncTestClient,
) -> None:
    """The lead-in matched one word of the query and its table none. It sits right after the
    section that answers, but a heading is between them, so it does not join that passage. On its
    own it is one line, so it is dropped, and the section that answers is all that is left."""
    chunks = await _guide_with_a_lead_in(client)
    lead_in = next(seq for seq, one in enumerate(chunks, start=1) if one.text == "Lancedb rules:")

    chunk_hits = (
        await client.get(
            "/api/search/explore",
            params={"q": "lancedb stores every chunk", "granularity": "chunk"},
        )
    ).json()
    response = await client.get("/api/search/excerpts", params={"q": "lancedb stores every chunk"})

    assert response.status_code == 200, response.text
    assert lead_in in {hit["seq"] for hit in chunk_hits}, "the lead-in did match"
    assert [(one["seq_start"], one["seq_end"]) for one in response.json()["excerpts"]] == [(1, 1)]


# Three short notes on aggregates: one on references, one on events, and a copy of the first under
# another name, as a note pasted into a second file is. The two notes share no word, and each
# question uses only its own note's words, so full-text search finds each part in one note alone.
REFERENCES = (
    "# Aggregates\n\n"
    "Reference other aggregates by identity, never through a direct object pointer.\n"
)
EVENTS = "# Events\n\nDomain events carry each change eventually, keeping consistency loose.\n"
BY_IDENTITY = "Reference other aggregates by identity or by direct object pointer?"
BY_EVENT = "Which domain events carry each change?"


async def _notes_on_aggregates(client: AsyncTestClient) -> None:
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "ddd"})
    for name, body in (
        ("references.md", REFERENCES),
        ("events.md", EVENTS),
        ("pasted.md", REFERENCES),
    ):
        await stage_and_import(client, name, body.encode())
        await attach_via_api(client, "ddd", name)


async def test_several_questions_take_turns_and_say_which_they_answer(
    client: AsyncTestClient,
) -> None:
    """Each part of the question is searched on its own, the copy of one note folds into it across
    the parts, and each excerpt names the parts it answers. The session keeps what was asked."""
    await _notes_on_aggregates(client)

    response = await client.get(
        "/api/search/excerpts",
        params={"q": [BY_IDENTITY, BY_EVENT], "session_id": "s1", "limit": 4},
    )

    assert response.status_code == 200, response.text
    answer = response.json()
    found = answer["excerpts"]
    assert (answer["uncovered"], answer["missing_terms"]) == ([], [])
    tagged = {one["document"]: one["aspects"] for one in found}
    kept_note = "references.md" if "references.md" in tagged else "pasted.md"
    assert tagged == {kept_note: [BY_IDENTITY], "events.md": [BY_EVENT]}
    ((copy,),) = [
        span["also_in"] for one in found if one["document"] == kept_note for span in one["spans"]
    ]
    assert copy["document"] == ({"references.md", "pasted.md"} - {kept_note}).pop()
    assert copy["relation"] == "duplicate", "the pasted note is the same text"
    steps = [entry.split(";")[0].strip() for entry in response.headers["server-timing"].split(",")]
    assert sorted(steps) == sorted(
        ["plan"]
        + ["retrieve", "merge", "rerank", "hits", "judge_thin"] * 2
        + ["fold", "group", "budget", "probe_gaps", "fill", "quote", "rerank_excerpts"]
    ), "one plan for every question, the ranking once per question, then the turns, the sections"
    (event,) = (await client.get("/api/sessions/s1/history")).json()
    assert event["subject"] == f"{BY_IDENTITY} | {BY_EVENT}"
    assert event["detail"]["questions"] == [BY_IDENTITY, BY_EVENT]
    assert event["detail"]["hits"] == 2


async def test_every_question_plans_over_one_open_index(client: AsyncTestClient) -> None:
    """The parts of one question run at once over the same indexes. Planned together, they share
    each index with its table already open, so they read one version of it rather than each
    opening its own."""
    from haskie.search import retrieval

    await _notes_on_aggregates(client)

    plans = await retrieval.plan(["ddd"], [BY_IDENTITY, BY_EVENT])

    assert plans is not None and len(plans) == 2
    (first, _), (second, _) = plans[0].indexes[0], plans[1].indexes[0]
    assert first is second, "one index for every question"
    assert first._cached is not None, "its table opened before any part searches"


async def test_a_question_nothing_answers_is_named_with_the_words_it_used(
    client: AsyncTestClient,
) -> None:
    """Two questions, and the notes answer one: the other is `uncovered`, and its words no
    excerpt holds are `missing_terms`, even after the search looked for them once more."""
    await _notes_on_aggregates(client)
    unanswered = "Which warehouse ledger reconciles stock?"

    response = await client.get(
        "/api/search/excerpts", params={"q": [BY_EVENT, unanswered], "limit": 2}
    )

    assert response.status_code == 200, response.text
    answer = response.json()
    assert [one["document"] for one in answer["excerpts"]] == ["events.md"]
    assert answer["uncovered"] == [unanswered]
    assert answer["missing_terms"] == ["warehouse", "ledger", "reconciles", "stock"]


async def test_the_answer_budget_cuts_the_last_sections(client: AsyncTestClient, caplog) -> None:
    """A collection whose answers may hold 100 characters: the best section stays, however long,
    and the next one is cut, which the search logs."""
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "notes"})
    await stage_and_import(client, "sections.md", SECTIONS_MD.encode())
    await _member("notes", "sections.md")
    markdown = await _markdown_of("sections.md")
    await seed_chunks(
        "notes", "sections.md", split(markdown, ChunkSettings(chunk_size=200, chunk_merge_below=0))
    )
    overrides = {"search": {"max_answer_chars": 100}}
    assert (await client.put("/api/collections/notes/overrides", json=overrides)).is_success

    with caplog.at_level(logging.INFO):
        response = await client.get("/api/search/excerpts", params={"q": "lancedb", "limit": 2})

    assert response.status_code == 200, response.text
    (only,) = response.json()["excerpts"]
    assert only["header"] == "Guide > Storage", "the best section, over the budget, stays"
    logged = [r.msg for r in caplog.records if isinstance(r.msg, dict)]
    (cut,) = [msg for msg in logged if msg["event"] == "search_budget"]
    assert cut["cut"] == 1


async def test_one_question_with_a_context_is_the_single_search(client: AsyncTestClient) -> None:
    """One question, however it is sent, is today's search: no turns and no tags. Its context is
    for the models to read; the full-text search reads the question's own words."""
    await _notes_on_aggregates(client)

    plain = await client.get("/api/search/excerpts", params={"q": BY_EVENT})
    repeated = await client.get("/api/search/excerpts", params={"q": [BY_EVENT, f" {BY_EVENT} "]})
    framed = await client.get(
        "/api/search/excerpts",
        params={
            "q": "Which change reaches the warehouse?",
            "context": "Domain events, keeping consistency loose.",
        },
    )

    assert plain.status_code == repeated.status_code == framed.status_code == 200
    assert [one["document"] for one in plain.json()["excerpts"]] == ["events.md"]
    assert all(one["aspects"] == [] for one in plain.json()["excerpts"])
    assert repeated.json() == plain.json(), "a repeated question is asked once"
    assert "cover" not in plain.headers["server-timing"]
    found = framed.json()["excerpts"]
    assert "events.md" in {one["document"] for one in found}, "found by its own word, change"
    missing = framed.json()["missing_terms"]
    assert missing == ["reaches", "warehouse"], (
        "the question's words no note has, not the context's"
    )


async def test_a_context_never_makes_a_question_match_by_its_own_words(
    client: AsyncTestClient,
) -> None:
    """The shared context names the notes' topic, and one part asks about something they never
    mention. Its full-text search reads the part alone, so the context's words find it nothing:
    it is uncovered, and no excerpt is tagged with it."""
    await _notes_on_aggregates(client)
    offtopic = "Which warehouse ledger reconciles stock?"

    response = await client.get(
        "/api/search/excerpts",
        params={
            "q": [BY_EVENT, offtopic],
            "context": "Domain events carry each change between aggregates.",
            "limit": 4,
        },
    )

    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["uncovered"] == [offtopic]
    assert all(offtopic not in one["aspects"] for one in answer["excerpts"])
    assert "events.md" in {one["document"] for one in answer["excerpts"]}


async def test_several_questions_over_collections_since_deleted_find_nothing(
    client: AsyncTestClient,
) -> None:
    """A session whose only collection is gone has nothing left to search: an empty answer, as
    one question gets, not an error."""
    await _notes_on_aggregates(client)
    await client.put("/api/sessions/s1", json={"collections": ["ddd"]})
    deleted = await client.delete("/api/collections/ddd")
    await wait_for(deleted.json()["operation_id"])

    response = await client.get(
        "/api/search/excerpts", params={"q": [BY_IDENTITY, BY_EVENT], "session_id": "s1"}
    )

    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["excerpts"] == []
    assert answer["uncovered"] == [BY_IDENTITY, BY_EVENT], "every question, unanswered"
    too_few = {"q": [BY_IDENTITY, BY_EVENT], "session_id": "s1", "limit": 1}
    refused = await client.get("/api/search/excerpts", params=too_few)
    assert refused.status_code == 422, "a caller's own bad limit is refused all the same"
    one = await client.get("/api/search/excerpts", params={"q": BY_EVENT, "session_id": "s1"})
    assert one.json() == {
        "excerpts": [],
        "uncovered": [],
        "missing_terms": ["domain", "events", "carry", "change"],
        "searched": [],
    }, "one question: no excerpts, and every word of it missing"


@pytest.mark.parametrize(
    ("name", "params", "message"),
    [
        ("a blank question", {"q": "  "}, "q must hold 1..5 questions, got 0"),
        ("six parts", {"q": [f"{BY_EVENT} {n}" for n in range(6)]}, "got 6"),
        ("fewer slots than parts", {"q": [BY_IDENTITY, BY_EVENT], "limit": 1}, "got 1"),
        ("a context past 200 characters", {"q": BY_EVENT, "context": "x" * 201}, "got 201"),
    ],
)
async def test_questions_a_search_cannot_run_are_refused_before_it_runs(
    client: AsyncTestClient, name: str, params: dict, message: str
) -> None:
    await _notes_on_aggregates(client)

    response = await client.get("/api/search/excerpts", params=params)

    assert response.status_code == 422, f"{name}: {response.text}"
    assert message in response.text, name


async def test_excerpts_default_to_ten_whatever_the_collections_limit(
    client: AsyncTestClient,
) -> None:
    """A caller who set no limit, as Explore and an agent do, gets `DEFAULT_EXCERPTS`: the
    collection's own `limit`, here below the questions asked, is for chunks and passages. Ten is
    above the most questions one call may ask, so each still gets a slot."""
    assert flow.DEFAULT_EXCERPTS >= aspects.MAX_QUESTIONS
    await _notes_on_aggregates(client)
    await client.put("/api/collections/ddd/overrides", json={"search": {"limit": 1}})

    response = await client.get(
        "/api/search/excerpts",
        params={"q": [BY_IDENTITY, BY_EVENT], "collections": "ddd", "session_id": "ten"},
    )

    assert response.status_code == 200, response.text
    assert len(response.json()["excerpts"]) >= 2, "one slot per question at least"
    (logged,) = (await client.get("/api/searches", params={"session_id": "ten"})).json()
    assert logged["result_limit"] == flow.DEFAULT_EXCERPTS


async def test_the_mcp_tool_takes_one_question_or_several(api_client: AsyncTestClient) -> None:
    """The tool schema is what an agent reads: `q` is a list, one question or up to five parts."""
    from litestar_mcp import LitestarMCP
    from litestar_mcp.schema_builder import generate_schema_for_handler

    tool = api_client.app.plugins.get(LitestarMCP).discovered_tools["search_excerpts"]
    schema = generate_schema_for_handler(tool)

    assert schema["properties"]["q"] == {"type": "array", "items": {"type": "string"}}
    assert schema["properties"]["context"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert schema["required"] == ["q"]


async def _two_collections_sharing_a_document(client: AsyncTestClient) -> str:
    """`shared.md` in both collections and `beta-only.md` in one: two documents that only `beta`
    covers on its own. Returns the markdown of the shared document."""
    await client.post("/api/init", json=NO_MODELS)
    for name in ("alpha", "beta"):
        await client.post("/api/collections", json={"name": name})
    await stage_and_import(client, "shared.md", PASSAGE_MD.encode())
    await stage_and_import(client, "beta-only.md", PASSAGE_MD.encode())
    await client.put("/api/documents/shared.md/description", json={"description": "the guide"})
    markdown = await _markdown_of("shared.md")
    chunks = [
        _chunk(markdown, "every chunk of a lancedb", "so one sentence can sit"),
        _chunk(markdown, "sit in two of them", "answer back into a"),
        _chunk(markdown, "This section says", "about tables at all.", heading="Elsewhere"),
    ]
    for name in ("alpha", "beta"):
        await _member(name, "shared.md")
        await seed_chunks(name, "shared.md", chunks)
    await _member("beta", "beta-only.md")
    await seed_chunks("beta", "beta-only.md", chunks)
    return markdown


async def test_a_map_lists_the_documents_it_reached_and_covers_them_with_collections(
    client: AsyncTestClient,
) -> None:
    """One row per document, whichever collections hold it, and the fewest collections a
    follow-up search has to select to reach every section and document listed."""
    await _two_collections_sharing_a_document(client)

    response = await client.get("/api/search/sections", params={"q": "lancedb"})

    assert response.status_code == 200, response.text
    found = response.json()
    rows = {row["document"]: row for row in found["documents"]}
    assert set(rows) == {"shared.md", "beta-only.md"}
    assert found["collections"] == ["beta"], "one collection holds both: the cover is one name"
    assert found["searched"] == ["alpha", "beta"], "every collection, none chosen"
    shared = rows["shared.md"]
    assert shared["collections"] == ["alpha", "beta"], "every searched collection holding it"
    assert rows["beta-only.md"]["collections"] == ["beta"]
    assert shared["description"] == "the guide", "the document's own description, not a chunk's"
    assert shared["chunks"] == 2, "the chunks that matched, each counted once across collections"
    row = await document.named("shared.md")
    assert shared["markdown_file"] == str(home.HOME / row.relative(row.markdown)), "on disk"
    picked = [one["document_id"] for one in found["sections"]]
    assert {row["document_id"]: row["sections"] for row in found["documents"]} == {
        row["document_id"]: picked.count(row["document_id"]) for row in found["documents"]
    }, "each document counts the map's sections in it"
    scores = [row["score"] for row in found["documents"]]
    assert scores == sorted(scores, reverse=True), "best document first"


SAGAS_BOOK = """# Sagas

## Choreography

In a choreographed saga every service listens for events and emits compensating events.
Choreography keeps services loosely coupled: each compensating event undoes one step.

## Orchestration

An orchestrator tells each service which saga step to run and which compensation to call.
The orchestrator holds the saga state, so a failed saga step triggers its compensation.

# Replication

## Leaders

A single leader accepts every write and ships its replication log to the followers.
Followers apply the leader log in order, so replication lag shows as stale reads.
"""
SAGAS_NOTE = """# Saga notes

A saga is a sequence of local transactions; a failed saga step runs compensating steps.
The compensating steps of a saga undo what the earlier steps did, one by one.
"""


async def _saga_shelf(client: AsyncTestClient) -> None:
    """Two documents about sagas in one collection, indexed by the real pipeline, no model."""
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "notes"})
    for name, body in (("book.md", SAGAS_BOOK), ("note.md", SAGAS_NOTE)):
        await stage_and_import(client, name, body.encode())
        await attach_via_api(client, "notes", name)


async def test_sections_map_a_topic_with_what_each_section_is_about(
    client: AsyncTestClient,
) -> None:
    """Full text only: sections by relevance, at most two of one document while another has
    some left, each with its descriptors, cited by header and location, and the
    collections to select."""
    await _saga_shelf(client)

    response = await client.get(
        "/api/search/sections", params={"q": "saga compensation", "session_id": "m1"}
    )

    assert response.status_code == 200, response.text
    assert "map_sections" in response.headers["server-timing"]
    found = response.json()
    assert found["collections"] == ["notes"]
    headers = [(one["document"], one["header"]) for one in found["sections"]]
    assert set(headers) == {("book.md", "Sagas"), ("note.md", "Saga notes")}, (
        "the section an excerpt would quote: the whole chapter fits; replication says nothing"
    )
    scores = [one["score"] for one in found["sections"]]
    assert scores == sorted(scores, reverse=True), "without vectors: by relevance"
    sagas = next(one for one in found["sections"] if one["document"] == "book.md")
    assert sagas["depth"] == 1 and sagas["location"].startswith("book.md L")
    assert sagas["chars"] > 0 and sagas["chunks"] >= 1
    assert "compensating" in sagas["descriptors"], "its own, against the other chapter"
    assert "saga" not in sagas["descriptors"], "not what its header says"
    assert "distinct" not in sagas

    searches = (await client.get("/api/searches", params={"session_id": "m1"})).json()
    (logged,) = searches
    assert logged["tool"] == "sections"
    assert {one["header"] for one in logged["results"]} == {header for _, header in headers}


async def _sections_of(document: str, collection: str) -> list[dict]:
    """The sections of `document` as `collection` indexed it, each with its header: what a
    search of that collection names by id. None where it holds no such document."""
    doc = await id_of(document)
    entry = (await Collection.indexed_entries([(collection, doc)])).get((collection, doc))
    found = [] if entry is None else await embed_cache.read_sections(doc, entry)
    return [msgspec.to_builtins(one) | {"header": one.header} for one in found]


async def test_each_chunking_names_and_describes_its_own_sections(client: AsyncTestClient) -> None:
    """One book in two collections chunked two ways: each collection's sections, ids and
    descriptors come from its own chunking, in the map and in the document's sections alike."""
    await _saga_shelf(client)
    await client.post("/api/collections", json={"name": "raw"})
    await client.put("/api/collections/raw/overrides", json={"chunker": "text", "chunk_size": 300})
    await attach_via_api(client, "raw", "book.md")

    for name in ("notes", "raw"):
        found = await client.get(
            "/api/search/sections", params={"q": "saga compensation", "collections": name}
        )
        book = next(one for one in found.json()["sections"] if one["document"] == "book.md")
        by_id = {one["id"]: one for one in await _sections_of("book.md", name)}
        assert by_id[book["id"]]["header"] == book["header"], f"{name}: the id names its section"
        assert book["descriptors"] == by_id[book["id"]]["descriptors"], name
        assert book["descriptors"], f"{name}: described"
    notes, raw = [
        {one["id"]: one["header"] for one in await _sections_of("book.md", name)}
        for name in ("notes", "raw")
    ]
    assert any(raw.get(id) not in (None, header) for id, header in notes.items()), (
        "the text chunking puts another section at one of the same places"
    )

    # new chunk settings, not indexed yet: the rows, and so the sections, are still the old ones
    await client.put("/api/collections/raw/overrides", json={"chunker": "text", "chunk_size": 500})
    kept = {one["id"]: one["header"] for one in await _sections_of("book.md", "raw")}
    assert kept == raw, "the entry the rows were indexed from, not today's settings"
    found = await client.get(
        "/api/search/sections", params={"q": "saga compensation", "collections": "raw"}
    )
    book = next(one for one in found.json()["sections"] if one["document"] == "book.md")
    assert book["descriptors"], "still described from the entry the collection indexed"
    assert await _sections_of("note.md", "raw") == [], "a document raw does not hold"


async def test_a_search_keeps_to_the_documents_and_sections_it_is_given(
    client: AsyncTestClient,
) -> None:
    """Every search that takes a scope answers from inside it alone: `document_ids` by document,
    `section_ids` by a section and the sections under it, and the two together by what both
    allow. The ids are the ones the document's sections and the answers carry."""
    await _saga_shelf(client)
    book, note = await id_of("book.md"), await id_of("note.md")
    listed = await _sections_of("book.md", "notes")
    by_header = {one["header"]: one["id"] for one in listed}
    sagas, orchestration = by_header["Sagas"], by_header["Sagas > Orchestration"]
    question = {"q": "saga compensation step", "limit": 10}

    async def excerpts(**scope) -> list[dict]:
        found = await client.get("/api/search/excerpts", params={**question, **scope})
        assert found.status_code == 200, found.text
        return found.json()["excerpts"]

    async def explore(granularity: str, **scope) -> list[dict]:
        found = await client.get(
            "/api/search/explore", params={**question, "granularity": granularity, **scope}
        )
        assert found.status_code == 200, found.text
        return found.json()

    everything = await excerpts()
    assert {one["document"] for one in everything} == {"book.md", "note.md"}
    assert all(ids.ID.fullmatch(one["section_id"]) for one in everything), "each names its section"
    assert {one["document"] for one in await excerpts(document_ids=[note])} == {"note.md"}
    chapter = await excerpts(section_ids=[sagas])
    assert [(one["document"], one["header"]) for one in chapter] == [("book.md", "Sagas")], (
        "the chapter kept to is one excerpt, as it is unscoped"
    )
    assert all(span["header"].startswith("Sagas") for one in chapter for span in one["spans"]), (
        "a chapter holds its subsections"
    )
    within = await excerpts(section_ids=[orchestration])
    assert [span["section_id"] for one in within for span in one["spans"]] == [orchestration], (
        "the text an excerpt adds around a passage stays inside the section too"
    )
    assert await excerpts(document_ids=[note], section_ids=[sagas]) == [], "what both allow"

    chunks = await explore("chunk", section_ids=[sagas])
    assert chunks and all(sagas in hit["section_ids"] for hit in chunks)
    passages = await explore("passage", document_ids=[book])
    assert passages and {one["document"] for one in passages} == {"book.md"}
    assert all(one["section_id"] in by_header.values() for one in passages)

    mapped = await client.get(
        "/api/search/sections", params={"q": "saga compensation", "document_ids": [note]}
    )
    assert {one["document"] for one in mapped.json()["sections"]} == {"note.md"}
    (saga_notes,) = mapped.json()["sections"]
    note_sections = await _sections_of("note.md", "notes")
    assert saga_notes["id"] in {one["id"] for one in note_sections}

    logged = (await client.get("/api/searches")).json()
    assert [one["scoped"] for one in logged if one["tool"] == "excerpts"] == [True] * 4 + [False]
    assert not [
        asked
        for topic in (await client.get("/api/gaps")).json()
        for asked in topic["questions"]
        if asked["signal"] == "empty"
    ], "the search both scopes left empty is no gap: other documents answer it"


@pytest.mark.parametrize(
    ("name", "path", "params"),
    [
        ("a document name is no id", "/api/search/excerpts", {"document_ids": ["book.md"]}),
        ("a hex MD5 is no base58 id", "/api/search/explore", {"section_ids": ["0" * 32]}),
        ("too short", "/api/search/sections", {"document_ids": ["abc"]}),
        ("0 is not a base58 digit", "/api/search/sections", {"document_ids": ["0" * 22]}),
        (
            "more than a search may keep to",
            "/api/search/excerpts",
            {"section_ids": ["a" * 22] * 101},
        ),
    ],
)
async def test_a_scope_of_what_is_no_id_is_refused(
    client: AsyncTestClient, name: str, path: str, params: dict
) -> None:
    await _saga_shelf(client)

    found = await client.get(path, params={"q": "saga", **params})

    assert found.status_code == 422, f"{name}: {found.text}"


async def test_sections_bound_their_limit(client: AsyncTestClient) -> None:
    await _saga_shelf(client)

    one = await client.get("/api/search/sections", params={"q": "saga", "limit": 1})
    assert one.status_code == 200 and len(one.json()["sections"]) == 1
    too_many = await client.get("/api/search/sections", params={"q": "saga", "limit": 41})
    assert too_many.status_code == 422 and "limit must be 1..40, got 41" in too_many.text
    nothing = (await client.get("/api/search/sections", params={"q": "zeppelin"})).json()
    assert (nothing["sections"], nothing["documents"], nothing["collections"]) == ([], [], []), (
        "no section is an answer"
    )
    await client.get("/api/search/sections", params={"q": "saga", "session_id": "map"})
    (logged,) = (await client.get("/api/searches", params={"session_id": "map"})).json()
    assert logged["result_limit"] == flow.DEFAULT_MAP == 15, "no limit: a map of fifteen"


async def _scoped_collections(client: AsyncTestClient) -> None:
    """One document per collection and a session that selected only `alpha`."""
    await client.post("/api/init", json=NO_MODELS)
    for name in ("alpha", "beta"):
        await client.post("/api/collections", json={"name": name})
        await stage_and_import(client, f"{name}.md", f"# {name}\n\nshared {name}\n".encode())
        await attach_via_api(client, name, f"{name}.md")
    await client.put("/api/sessions/s1", json={"collections": ["alpha"]})


@pytest.mark.parametrize(
    ("name", "params", "expected"),
    [
        (
            "named collections win over the session",
            {"session_id": "s1", "collections": "beta"},
            ["beta.md"],
        ),
        ("the session wins over everything", {"session_id": "s1"}, ["alpha.md"]),
        ("neither: every collection", {}, ["alpha.md", "beta.md"]),
        ("an empty filter is not a filter", {"collections": ""}, ["alpha.md", "beta.md"]),
    ],
)
async def test_the_search_scope_is_the_names_then_the_session_then_everything(
    client: AsyncTestClient, name: str, params: dict, expected: list[str]
) -> None:
    await _scoped_collections(client)

    response = await client.get("/api/search/explore", params={"q": "shared", **params})

    assert response.status_code == 200, response.text
    assert sorted({hit["document"] for hit in response.json()}) == expected, name


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ("/api/search/explore", None),
        ("/api/search/excerpts", "excerpts"),
        ("/api/search/sections", "sections"),
    ],
)
async def test_every_search_rejects_a_collection_nobody_owns(
    client: AsyncTestClient, path: str, key: str | None
) -> None:
    """A name the caller chose just now is a mistake in the request, not an empty result."""
    await _scoped_collections(client)

    unknown = await client.get(path, params={"q": "shared", "collections": "ghost"})

    assert unknown.status_code == 404, unknown.text
    assert "collection not found: ghost" in unknown.text
    known = await client.get(path, params={"q": "shared", "collections": "alpha"})
    assert known.status_code == 200, known.text
    assert (known.json()[key] if key else known.json()) != []


async def test_the_mcp_surface_offers_one_search_per_question(api_client: AsyncTestClient) -> None:
    """Two tools for the two questions an agent has: what do the sources say, and where in which
    sources does a topic live. The searches the web UI drives stay REST-only, or an agent would
    have to choose between several that answer with overlapping chunks."""
    from litestar_mcp import LitestarMCP

    served = set(api_client.app.plugins.get(LitestarMCP).discovered_tools)

    assert {"search_excerpts", "search_sections"} <= served
    assert served.isdisjoint(
        {"search", "search_text", "explore", "search_collection", "search_sources"}
    )


@pytest.mark.parametrize("bias", [-1.0, 1.0])
async def test_a_growth_bias_at_either_end_is_saved_and_searched_with(
    client: AsyncTestClient, bias: float
) -> None:
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "notes"})

    body = {"search": {"grow_bias": bias}}
    saved = await client.put("/api/collections/notes/overrides", json=body)

    assert saved.status_code == 200, saved.text
    after = (await client.get("/api/collections/notes")).json()
    assert (after["overrides"]["search"]["grow_bias"], after["search"]["grow_bias"]) == (bias, bias)
    scoped = {"q": "alpha", "collections": "notes"}
    searched = await client.get("/api/search/explore", params=scoped)
    assert searched.status_code == 200, searched.text


@pytest.mark.parametrize(
    ("name", "search"),
    [
        ("a limit below one", {"limit": 0}),
        ("a negative weight", {"vector_weight": -1}),
        ("no candidates", {"candidates": 0}),
        ("a reranker floor above 1", {"min_rerank_score": 1.5}),
        ("a reranker floor below 0", {"min_rerank_score": -0.1}),
        ("a growth bias above 1", {"grow_bias": 1.5}),
        ("a growth bias below -1", {"grow_bias": -1.01}),
    ],
)
async def test_a_refused_search_override_is_never_saved(
    client: AsyncTestClient, name: str, search: dict
) -> None:
    """Refused as it is read, before anything is written: a value no search can run with would
    otherwise be stored, and every later read of the collection would fail on it."""
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "notes"})
    await client.put("/api/collections/notes/overrides", json={"search": {"limit": 3}})

    refused = await client.put("/api/collections/notes/overrides", json={"search": search})

    assert refused.status_code == 422, f"{name}: {refused.text}"
    after = await client.get("/api/collections/notes")
    assert after.status_code == 200, f"{name}: the collection still reads: {after.text}"
    assert after.json()["overrides"]["search"]["limit"] == 3, f"{name}: the old value stands"
    assert after.json()["search"]["limit"] == 3, name
    scoped = {"q": "alpha", "collections": "notes"}
    searched = await client.get("/api/search/explore", params=scoped)
    assert searched.status_code == 200, f"{name}: and it still searches: {searched.text}"


async def test_the_original_opens_under_its_own_name_and_media_type(
    client: AsyncTestClient,
) -> None:
    """The UI's "Open original" opens this URL in a new tab: a PDF has to arrive as a PDF,
    named for the document, or the browser downloads a nameless file instead of showing it."""
    await client.post("/api/init", json=NO_MODELS)
    await stage_and_import(client, "paper.pdf", text_pdf(["Facility location covers the pool"]))

    source = await client.get("/api/documents/paper.pdf/source")

    assert source.status_code == 200, source.text
    assert source.headers["content-type"] == "application/pdf"
    assert source.headers["content-disposition"] == 'inline; filename="paper.pdf"'
    assert source.content.startswith(b"%PDF-")


# --- audit trail ---------------------------------------------------------------------


async def test_every_audited_route_appends_one_record(client: AsyncTestClient) -> None:
    await client.post("/api/init", json=NO_MODELS)
    await client.put("/api/settings", json={"search": {"limit": 7}})
    await client.post("/api/collections", json={"name": "notes"})
    await client.put("/api/collections/notes/overrides", json={"chunk_size": 400})
    await client.put("/api/collections/notes/description", json={"description": "what I read"})
    await stage_and_import(client, "guide.md", MD.encode())
    await client.put("/api/documents/guide.md/description", json={"description": "the guide"})
    attach_job = await attach_via_api(client, "notes", "guide.md")
    reindex = (await client.post("/api/collections/notes/index")).json()["operation_id"]
    await wait_for(reindex)
    await client.delete(f"/api/operations/{attach_job}")
    await client.put("/api/sessions/s1", json={"collections": ["notes"]})
    await client.delete("/api/collections/notes/documents/guide.md")
    delete_doc = (await client.delete("/api/documents/guide.md")).json()["operation_id"]
    await wait_for(delete_doc)
    delete_collection = (await client.delete("/api/collections/notes")).json()["operation_id"]
    await wait_for(delete_collection)  # read once the queued work is cancelled and gone

    lines = audit_lines()
    assert _requested(lines) == [
        "settings.init",
        "settings.update",
        "collection.create",
        "collection.overrides.update",
        "collection.describe",
        "document.stage",
        "document.import",
        "document.describe",
        "collection.attach",
        "collection.reindex",
        "operation.cancel",
        "session.collections.set",
        "collection.detach",
        "document.delete",
        "collection.delete",
    ]
    by_event = {line["event"]: line for line in lines}
    assert all(line["outcome"] == "ok" for line in lines)
    assert all(line["app_version"] == audit.APP_VERSION for line in lines)
    assert all("request_id" in line for line in lines if line["actor"] == "web")
    assert by_event["settings.init"]["detail"] == {"profile": "none"}
    assert by_event["settings.update"]["detail"] == {"changed": "search.limit"}
    assert by_event["collection.create"]["collection"] == "notes"
    assert "document" not in by_event["collection.create"]
    assert by_event["document.stage"]["detail"] == {
        "name": "guide.md",
        "suffix": ".md",
        "size": len(MD),
    }
    assert by_event["document.import"]["document"] == "guide.md"
    assert "collection" not in by_event["document.import"], "a document belongs to no collection"
    assert (
        by_event["collection.attach"]["collection"],
        by_event["collection.attach"]["document"],
    ) == (
        "notes",
        "guide.md",
    )
    assert by_event["collection.attach"]["operation_id"] == attach_job
    assert (
        by_event["collection.detach"]["collection"],
        by_event["collection.detach"]["document"],
    ) == (
        "notes",
        "guide.md",
    )
    assert by_event["document.delete"]["document"] == "guide.md"
    assert by_event["collection.reindex"]["operation_id"] == reindex
    assert by_event["operation.cancel"]["operation_id"] == attach_job, "a known field, not free"
    assert by_event["session.collections.set"]["session_id"] == "s1"
    assert by_event["collection.delete"]["operation_id"] == delete_collection

    indexed = next(line for line in lines if line["event"] == "index.completed")
    assert (indexed["actor"], indexed["collection"], indexed["document"]) == (
        "operation",
        "notes",
        "guide.md",
    )
    assert "request_id" not in indexed, "a worker has no request context"
    completed = next(line for line in lines if line["event"] == "import.completed")
    assert completed["document"] == "guide.md"
    assert "collection" not in completed, "an import is not a collection's business"


async def test_a_failed_request_is_audited_with_its_scrubbed_error(client: AsyncTestClient) -> None:
    await client.post("/api/collections", json={"name": "notes"})

    assert (await client.post("/api/collections", json={"name": "notes"})).status_code == 409

    ok, failed = audit_lines()
    assert (ok["event"], ok["outcome"], ok["collection"]) == ("collection.create", "ok", "notes")
    assert (failed["event"], failed["outcome"]) == ("collection.create", "error")
    assert failed["error"] == "Conflict: collection already exists: notes"
    assert "collection" not in failed, "the name is only attached once the collection exists"
    assert ok["request_id"] != failed["request_id"], "one id per request"


async def test_an_import_records_the_file_name_but_never_the_path(
    client: AsyncTestClient, tmp_path: Path
) -> None:
    source = tmp_path / "private" / "salary.md"
    source.parent.mkdir()
    source.write_text(MD)
    await client.post("/api/init", json=NO_MODELS)

    assert (
        await client.post("/api/documents/import", json={"path": str(source)})
    ).status_code == 201

    record = next(line for line in audit_lines() if line["event"] == "document.import")
    assert record["document"] == "salary.md"
    assert record["detail"]["source"] == "salary.md"
    assert str(source.parent) not in json.dumps(record), "the source directory stays out of it"


async def test_the_audit_trail_is_written_even_when_logging_is_silenced(
    client: AsyncTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`HASKIE_LOG_LEVEL=CRITICAL` lands on the root logger, so this sets the same thing the
    env var configures. The file append is the durable sink: no level may drop it.

    Through `setLevel`, both ways: it clears every logger's cached level check, which assigning
    `level` does not, so a later test would find its loggers still silenced."""
    root = logging.getLogger()
    before = root.level
    root.setLevel(logging.CRITICAL)
    try:
        assert not logging.getLogger("haskie.audit").isEnabledFor(logs.AUDIT)
        created = await client.post("/api/collections", json={"name": "notes"})
    finally:
        root.setLevel(before)

    assert created.status_code == 201

    (record,) = audit_lines()
    assert (record["event"], record["outcome"], record["collection"]) == (
        "collection.create",
        "ok",
        "notes",
    )


# --- unexpected failures --------------------------------------------------------------


async def test_an_unexpected_failure_answers_500_with_a_scrubbed_message(
    ready: AsyncTestClient, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """A read that fails for no reason the domain has a name for is the bug shape the generic
    handler is for: the body names it, with the home path scrubbed out of it."""

    async def unreadable(doc: str) -> document.Document:
        raise RuntimeError(f"row unreadable: {home.HOME / 'documents' / doc}")

    monkeypatch.setattr(document, "get", unreadable)

    with caplog.at_level(logging.DEBUG):
        response = await ready.get("/api/documents/guide.md/preview")

    assert response.status_code == 500
    assert response.json()["detail"] == (
        f"RuntimeError: row unreadable: $HASKIE_HOME/documents/{await id_of('guide.md')}"
    )
    assert str(home.HOME) not in response.text
    assert app_module.REQUEST_ID_HEADER in response.headers
    (logged,) = [r for r in caplog.records if isinstance(r.msg, dict)]
    assert logged.msg["event"] == "unhandled_error"
    assert logged.msg["exception"].startswith("Traceback"), "a bug keeps its traceback"


async def test_a_rejected_request_is_logged_once(client: AsyncTestClient, caplog) -> None:
    """Litestar's exception middleware would log a 404 as an ERROR with a traceback of its own;
    `logging_config=None` (see `logs`) leaves the one warning `_client_error` writes."""
    with caplog.at_level(logging.DEBUG):
        assert (await client.get("/api/collections/absent")).status_code == 404

    (record,) = [r for r in caplog.records if isinstance(r.msg, dict)]
    assert record.msg["event"] == "request_rejected"
    assert record.levelno == logging.WARNING


async def test_a_domain_error_without_its_own_status_answers_400(
    ready: AsyncTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from haskie.errors import HaskieError

    async def refuse(request) -> list[str]:
        raise HaskieError("the collection index is busy")

    monkeypatch.setattr(Collection, "page", staticmethod(refuse))

    response = await ready.get("/api/collections")

    assert response.status_code == 400
    assert response.json()["detail"] == "the collection index is busy"


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


@pytest.mark.parametrize(
    ("name", "path", "routed", "expected"),
    [
        (
            "a collection route binds both names",
            "/api/collections/notes/documents/guide.md",
            {"collection": "notes", "document": "guide.md"},
            {"collection": "notes", "document": "guide.md"},
        ),
        (
            "a document route has no collection at all",
            "/api/documents/guide.md/source",
            {"document": "guide.md"},
            {"document": "guide.md"},
        ),
        ("a route with neither binds neither", "/api/status", {}, {}),
    ],
)
async def test_bind_request_context_scopes_the_names(
    name: str, path: str, routed: dict, expected: dict
) -> None:
    request = RequestFactory().get(path=path)
    request.scope["path_params"] = routed

    try:
        await app_module.bind_request_context(request)
        context = dict(structlog.contextvars.get_contextvars())
        assert {k: context[k] for k in ("collection", "document") if k in context} == expected, name
    finally:
        logs.clear()


# --- body size, static files and lifespan ---------------------------------------------


FOREIGN = "https://evil.example"


@pytest.mark.parametrize(
    ("bound", "trusted", "base_url", "method", "origin", "status"),
    [
        pytest.param(None, "", LOOPBACK_URL, "POST", None, 422, id="agent-without-origin-passes"),
        pytest.param(None, "", LOOPBACK_URL, "POST", LOOPBACK_URL, 422, id="own-ui-passes"),
        pytest.param(None, "", LOOPBACK_URL, "POST", FOREIGN, 403, id="foreign-page-post-refused"),
        pytest.param(None, "", LOOPBACK_URL, "POST", "null", 403, id="opaque-origin-refused"),
        pytest.param(None, "", LOOPBACK_URL, "GET", FOREIGN, 422, id="foreign-read-passes"),
        pytest.param(None, f" {FOREIGN}/ ,", LOOPBACK_URL, "POST", FOREIGN, 422, id="trusted"),
        pytest.param(None, "", "http://localhost:8451", "POST", None, 422, id="localhost-name"),
        pytest.param(None, "", "http://rebind.example", "GET", None, 403, id="rebinding-host"),
        pytest.param(
            "http://box.lan:8451", "", "http://box.lan:8451", "GET", None, 422, id="bound"
        ),
        pytest.param("http://0.0.0.0:8451", "", "http://box.lan", "GET", None, 422, id="wildcard"),
    ],
)
async def test_only_served_hosts_and_trusted_browser_origins_reach_a_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bound: str | None,
    trusted: str,
    base_url: str,
    method: str,
    origin: str | None,
    status: int,
) -> None:
    """A browser ignores the loopback bind, so a page the user opens could post to the API, and a
    DNS-rebinding page could read it. Agents send no `Origin` and pass. A request that passes is
    answered by the route itself: a 422 for its invalid input, which the guard never looks at."""
    if bound is not None:
        monkeypatch.setenv(home.ADDRESS_ENV, bound)
    monkeypatch.setenv(app_module.ALLOWED_ORIGINS_ENV, trusted)
    client = AsyncTestClient(api_app(tmp_path, monkeypatch), base_url=base_url)
    headers = {} if origin is None else {"Origin": origin}

    if method == "GET":
        response = await client.get("/api/search/excerpts?q=x&limit=0", headers=headers)
    else:
        response = await client.post("/api/documents/render", json={"x": 1}, headers=headers)

    assert response.status_code == status, response.text
    if status == 403:
        assert app_module.ALLOWED_ORIGINS_ENV in response.text or "not one haskie" in response.text


async def test_a_body_over_the_upload_cap_is_rejected_before_the_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dbos
) -> None:
    """Litestar enforces `request_max_body_size`, so a 512 MiB upload never has to be sent here:
    the cap is lowered and the same code path answers 413."""
    monkeypatch.setattr(app_module, "UPLOAD_MAX_BYTES", 64)  # read by `create_app`, so first
    client = AsyncTestClient(api_app(tmp_path, monkeypatch), base_url=LOOPBACK_URL)

    response = await client.post(
        "/api/documents/staging", files={"data": ("big.md", b"x" * 500, "text/markdown")}
    )

    assert response.status_code == 413
    assert "Request Entity Too Large" in response.text
    assert app_module.REQUEST_ID_HEADER in response.headers, "traceable like any other rejection"
    assert await document_names() == [], "nothing was stored"


async def test_static_files_are_served_when_the_web_build_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dbos
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>haskie</title>")
    monkeypatch.setattr(app_module, "WEB_DIST", dist)
    client = AsyncTestClient(app_module.create_app(), base_url=LOOPBACK_URL)

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

    before = set(threading.enumerate())

    for _ in range(2):
        async with AsyncTestClient(api_app(tmp_path, monkeypatch), base_url=LOOPBACK_URL) as client:
            assert (await client.get("/api/status")).status_code == 200
            assert dbos_module._dbos_global_instance is not None
        assert dbos_module._dbos_global_instance is None, "destroyed by the shutdown hook"

    leaked = [
        thread
        for thread in threading.enumerate()
        if thread not in before and not thread.daemon and thread.name.startswith("dbos-")
    ]
    assert leaked == [], f"threads outliving DBOS block interpreter exit: {leaked}"


async def test_startup_refreshes_every_installation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, seeded_home: Path
) -> None:
    """A change a crash lost before its refresh, or a template an upgrade changed, reaches the
    installations at the next start."""
    await Collection.create("roasting", "Coffee.")

    await until(
        refresh_settled, "the create's own refresh ended"
    )  # before the install it would write
    directory = claude_installed(tmp_path / "project" / ".claude")
    await claude.record_installation(directory)

    async with AsyncTestClient(api_app(tmp_path, monkeypatch), base_url=LOOPBACK_URL) as client:
        assert (await client.get("/api/status")).status_code == 200

        async def rewritten() -> bool:
            rule = claude.rule_path(directory)
            return rule.is_file() and "roasting: Coffee" in rule.read_text()

        await until(rewritten, "the startup refresh reached the rule")


# --- S4: full-text search across collections ------------------------------------------

TEXT_DOCS = 3  # documents per collection, one chunk each: six rows to merge and page over


async def _text_collections(client: AsyncTestClient, *names: str) -> None:
    """Collections with `TEXT_DOCS` indexed one-chunk documents each, all matching "haskell".

    The term is repeated once more per document, so no two chunks of one collection score the same
    and the ranking a page cuts is the same ranking every time it is recomputed.
    """
    await client.post("/api/init", json=NO_MODELS)
    for name in names:
        await client.post("/api/collections", json={"name": name})
        for i in range(TEXT_DOCS):
            body = f"# {name} {i}\n\n{'haskell ' * (i + 1)}chapter {i} of {name}\n"
            doc = f"{name}-{i}.md"
            await stage_and_import(client, doc, body.encode())
            await attach_via_api(client, name, doc)


def _text_identity(page: dict) -> list[tuple]:
    """What identifies every passage of a page, in the order the page listed them."""
    return [(h["document"], h["seq"]) for h in page["items"]]


def _text_cursor(
    q: str = "haskell",
    collections: list[str] | None = None,
    page_size: int = 100,
    offset: int = 1,
) -> str:
    """A real cursor of some query, built at collection time for the table below."""
    from haskie.search import text

    return text.make_cursor(q, collections or ["alpha", "beta"], page_size, offset)


def _listing_cursor() -> str:
    """A cursor of the collection listing: another sort, so this route must not read it."""
    from haskie.paging import encode_cursor

    return encode_cursor(["alpha"], "name", Order.ASC)


async def test_text_search_spans_all_collections_by_default(client: AsyncTestClient) -> None:
    await _text_collections(client, "alpha", "beta")
    assert (await client.post("/api/collections", json={"name": "blank"})).status_code == 201

    response = await client.get("/api/search/text", params={"q": "haskell"})

    assert response.status_code == 200, response.text
    page = response.json()
    assert page["next_cursor"] is None, "six hits fit in one default page"
    assert page["total"] is None, "counting the ranking costs as much as producing it"
    assert len(page["items"]) == 2 * TEXT_DOCS
    assert {h["collection"] for h in page["items"]} == {"alpha", "beta"}, "the blank one is absent"
    scores = [h["score"] for h in page["items"]]
    assert scores == sorted(scores, reverse=True), "raw BM25, best first, across both collections"

    row = await document.named("alpha-0.md")
    hit = next(h for h in page["items"] if h["document"] == "alpha-0.md")
    assert hit["markdown_path"] == row.relative(row.markdown), "the document's own file"
    assert hit["source_path"] == row.relative(row.original)
    assert hit["source_file"] == str(home.HOME / row.relative(row.original)), "absolute"
    assert "haskell" in hit["text"]
    assert not [line for line in audit_lines() if "search" in line["event"]], "never audited"


async def test_text_search_returns_a_shared_document_once(client: AsyncTestClient) -> None:
    """One document in two collections holds the same chunk in both tables; the merge keeps it
    once, so a page is a page of passages rather than of memberships."""
    await client.post("/api/init", json=NO_MODELS)
    for name in ("alpha", "beta"):
        await client.post("/api/collections", json={"name": name})
    await stage_and_import(client, "shared.md", b"# Shared\n\nhaskell everywhere\n")
    for name in ("alpha", "beta"):
        await attach_via_api(client, name, "shared.md")

    page = (await client.get("/api/search/text", params={"q": "haskell"})).json()

    assert _text_identity(page) == [("shared.md", 1)]


async def test_text_search_filters_collections_and_rejects_unknown(
    client: AsyncTestClient,
) -> None:
    await _text_collections(client, "alpha", "beta")

    one = await client.get("/api/search/text", params={"q": "haskell", "collections": "alpha"})
    assert one.status_code == 200, one.text
    assert {h["collection"] for h in one.json()["items"]} == {"alpha"}
    assert len(one.json()["items"]) == TEXT_DOCS

    deduped = await client.get(
        "/api/search/text", params={"q": "haskell", "collections": " alpha ,alpha,"}
    )
    assert deduped.json() == one.json(), "duplicates and blanks drop out, the page is the same"

    everything = await client.get("/api/search/text", params={"q": "haskell", "collections": ""})
    assert len(everything.json()["items"]) == 2 * TEXT_DOCS, "an empty filter is not a filter"

    unknown = await client.get("/api/search/text", params={"q": "haskell", "collections": "ghost"})
    assert unknown.status_code == 404, "a name nobody owns is a mistake, not an empty page"
    assert "collection not found: ghost" in unknown.text


async def test_text_search_pages_without_overlap(client: AsyncTestClient) -> None:
    await _text_collections(client, "alpha", "beta")
    whole = await get_page(client, "/api/search/text", q="haskell", page_size=100)

    walked, sizes, _, cursors = await walk_pages(
        client, "/api/search/text", q="haskell", page_size=1
    )

    assert sizes == [1] * (2 * TEXT_DOCS), "a full page while there is a next cursor"
    assert cursors[-1] is None, "the walk ended on its own"
    identity = _text_identity({"items": walked})
    assert len(set(identity)) == len(identity), "no passage is shown twice"
    assert identity == _text_identity(whole), "the walk is the one-page ranking, cut up"


async def test_text_search_answers_an_empty_page_past_the_last_hit(
    client: AsyncTestClient,
) -> None:
    """A cursor is an offset into a ranking that may have shrunk since it was issued."""
    await _text_collections(client, "alpha")
    deep = _text_cursor(collections=["alpha"], page_size=100, offset=100)

    page = await client.get(
        "/api/search/text", params={"q": "haskell", "collections": "alpha", "cursor": deep}
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
            "a cursor of another set of collections",
            {"cursor": _text_cursor(collections=["alpha"])},
            422,
            "cursor was issued for another query",
        ),
        (
            "a cursor of the collection listing",
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
    await client.post("/api/init", json=NO_MODELS)
    for collection in ("alpha", "beta"):  # the collections the cursors above were issued for
        await client.post("/api/collections", json={"name": collection})

    response = await client.get("/api/search/text", params={"q": "haskell", **params})

    assert response.status_code == status, f"{name}: {response.text}"
    assert detail in response.json()["detail"], name


async def test_search_trend_lists_every_recent_search_oldest_first(ready: AsyncTestClient) -> None:
    """The points the Insights chart buckets: one per search, with its session, and nothing else
    a session did. A search without a session is a point too, with no session."""
    for session_id, q in [("t1", "alpha"), ("t2", "alpha"), ("t1", "beta")]:
        await ready.get("/api/search/text", params={"q": q, "session_id": session_id})
    await ready.get("/api/search/text", params={"q": "alpha"})
    await ready.put("/api/sessions/t1", json={"collections": ["notes"]})

    points = (await ready.get("/api/insights/searches", params={"days": 1})).json()

    assert [p["session_id"] for p in points] == ["t1", "t2", "t1", None], (
        "one point per search, in order, and no point for setting a selection"
    )
    assert [p["ts"] for p in points] == sorted(p["ts"] for p in points)
    bad = await ready.get("/api/insights/searches", params={"days": 0})
    assert bad.status_code == 422 and "Expected `int` >= 1" in bad.text


async def test_chunk_trend_lists_imports_and_indexes_and_bounds_its_window(
    client: AsyncTestClient,
) -> None:
    await client.post("/api/init", json=NO_MODELS)
    await client.post("/api/collections", json={"name": "notes"})
    await stage_and_import(client, "guide.md", MD.encode())
    await attach_via_api(client, "notes", "guide.md")

    points = (await client.get("/api/insights/chunks", params={"days": 1})).json()

    imported, indexed = points
    assert (imported["document"], imported["collection"]) == ("guide.md", None), "the import"
    assert (indexed["document"], indexed["collection"]) == ("guide.md", "notes"), "then the index"
    assert imported["chunks"] == indexed["chunks"] > 0, "the chunks embedded are the ones written"
    assert all(abs(point["ts"] - time.time()) < 60 for point in points)
    bad = await client.get("/api/insights/chunks", params={"days": 367})
    assert bad.status_code == 422 and "Expected `int` <= 366" in bad.text


@pytest.mark.parametrize(
    ("name", "markdown", "status", "expected"),
    [
        (
            "an excerpt's headings, lists and emphasis as HTML, headings without ids",
            "### Rules\n\n- one *rule*\n- two\n\n[…]",
            200,
            "<h3>Rules</h3>\n<ul>\n<li>one <em>rule</em></li>\n<li>two</li>\n</ul>\n<p>[…]</p>\n",
        ),
        (
            "raw HTML a document holds is stripped, never passed on",
            "text <script>alert(1)</script> after",
            200,
            "<p>text alert(1) after</p>\n",
        ),
        ("at the limit is rendered", "x" * 100_000, 200, f"<p>{'x' * 100_000}</p>\n"),
        ("past the limit is refused", "x" * 100_001, 422, "Expected `str` of length <= 100000"),
    ],
)
async def test_a_search_results_text_renders_as_markdown(
    client: AsyncTestClient, name: str, markdown: str, status: int, expected: str
) -> None:
    response = await client.post("/api/documents/render", json={"markdown": markdown})

    assert response.status_code == status, f"{name}: {response.text}"
    body = response.json()
    if status == 200:
        assert body["html"] == expected, name
    else:
        assert expected in body["detail"], name
