"""Backup and restore, through the API: what an archive holds, what a restore puts back and keeps,
and every refusal, which leaves the contents as they were."""

import io
import os
import sqlite3
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest
from dbos import DBOS
from sqlalchemy import update

from haskie import backup, db, home, sysdb
from haskie.collection.collection import Collection
from haskie.document import document
from haskie.document.document import DocumentStatus
from haskie.indexing import workflows
from haskie.indexing.workflows import BulkProgress
from haskie.tables import documents

from conftest import MD, attach_via_api, stage_and_import, text_pdf, until, wait_for  # isort: skip

pytestmark = pytest.mark.anyio


async def _backup(client) -> tuple[str, bytes]:
    started = await client.post("/api/backup")
    assert started.status_code == 202, started.text
    operation_id = started.json()["operation_id"]
    await wait_for(operation_id)
    archive = await client.get(f"/api/backup/{operation_id}/file")
    assert archive.status_code == 200, archive.text
    assert archive.headers["content-disposition"].startswith('attachment; filename="haskie-backup-')
    return operation_id, archive.content


async def _restore(client, body: bytes):
    return await client.post(
        "/api/restore", content=body, headers={"content-type": "application/zip"}
    )


async def _member_status(client, collection: str, doc: str) -> str | None:
    found = await client.get(f"/api/collections/{collection}/documents")
    items = found.json()["items"] if found.status_code == 200 else []
    return next((row["status"] for row in items if row["document"]["name"] == doc), None)


async def _seed(client) -> str:
    """One document in one collection, indexed, and settings off their defaults."""
    row = await stage_and_import(client, "notes.md", MD.encode(), description="my notes")
    created = await client.post("/api/collections", json={"name": "kept", "description": "a set"})
    assert created.status_code == 201, created.text
    await attach_via_api(client, "kept", row["name"])
    current = (await client.get("/api/settings")).json()
    current["retention"]["audit_days"] = 7
    current["search"]["reranker"] = "none"  # no model download for the searches here
    assert (await client.put("/api/settings", json=current)).status_code == 200
    return row["name"]


def _rows(archive: bytes) -> dict[str, int]:
    """Row counts of every table in an archive's database."""
    with zipfile.ZipFile(io.BytesIO(archive)) as opened:
        data = opened.read(backup.DATABASE)
    conn = sqlite3.connect(":memory:")
    conn.deserialize(data)
    try:
        names = [
            name for (name,) in conn.execute("select name from sqlite_master where type='table'")
        ]
        return {name: conn.execute(f"select count(*) from {name}").fetchone()[0] for name in names}
    finally:
        conn.close()


async def test_a_restore_puts_back_the_contents_and_keeps_the_history(client) -> None:
    doc = await _seed(client)
    operation_id, archive = await _backup(client)

    # what the archive holds: the content tables alone, and the document's files
    assert _rows(archive) == {
        "settings": 1,
        "collections": 1,
        "documents": 1,
        "embeddings": 1,
        "collection_documents": 1,
    }
    names = zipfile.ZipFile(io.BytesIO(archive)).namelist()
    assert {Path(name).name.split(".", 1)[-1] for name in names} >= {
        "json",
        "db",
        "md",
        "md.md",
        "chunks.parquet",
        "sections.parquet",
    }

    # a search after the backup: the history it logs must survive the restore
    searched = await client.get("/api/search/excerpts", params={"q": "alpha body"})
    assert searched.status_code == 200, searched.text
    # everything changed after the backup
    deleted = await client.delete(f"/api/documents/{doc}")
    await wait_for(deleted.json()["operation_id"])
    dropped = await client.delete("/api/collections/kept")
    await wait_for(dropped.json()["operation_id"])
    current = (await client.get("/api/settings")).json()
    current["retention"]["audit_days"] = 30
    await client.put("/api/settings", json=current)
    # a session that chose a collection the archive holds, and one it does not: after the
    # restore it keeps the first only
    for name in ("kept", "later"):
        await client.post("/api/collections", json={"name": name})
    await client.put("/api/sessions/s1", json={"collections": ["kept", "later"]})

    restoring = await _restore(client, archive)
    assert restoring.status_code == 202, restoring.text
    restored = await wait_for(restoring.json()["operation_id"])
    assert len(restored) == 1, "one index of the one collection, no import"
    for started in restored:
        await wait_for(started)

    async def indexed() -> bool:
        return await _member_status(client, "kept", doc) == "indexed"

    await until(indexed, "the restored member was never indexed")
    row = (await client.get(f"/api/documents/{doc}")).json()
    assert (row["status"], row["description"]) == ("imported", "my notes")
    assert (await client.get("/api/collections/kept")).json()["description"] == "a set"
    assert (await client.get("/api/settings")).json()["retention"]["audit_days"] == 7
    found = await client.get("/api/search/excerpts", params={"q": "alpha body"})
    assert found.json()["excerpts"], "the rebuilt index answers"
    async with db.read() as conn:
        logged = (await conn.exec_driver_sql("select count(*) from searches")).scalar()
    assert logged == 2, "the search history outlives the restore"
    sessions = {one["id"]: one["collections"] for one in (await client.get("/api/sessions")).json()}
    assert sessions["s1"] == ["kept"], "a session keeps the collections the archive brought back"

    listed = (await client.get("/api/operations", params={"kind": "backup"})).json()["items"]
    assert {(item["detail"]["bulk"], item["status"]) for item in listed} == {
        ("create_backup", "SUCCESS"),
        ("restore_backup", "SUCCESS"),
    }
    made = next(item for item in listed if item["id"] == operation_id)
    assert made["detail"] == {
        "bulk": "create_backup",
        "done": 4,
        "total": 4,
        "size": len(archive),
        "available": True,
    }, "a finished backup counts from its output, with no event read"
    assert list(home.RESTORE_ROOT.iterdir()) == []


async def test_a_backup_leaves_out_what_is_being_deleted(client) -> None:
    kept = await stage_and_import(client, "kept.md", MD.encode())
    going = await stage_and_import(client, "going.md", MD.encode())
    async with db.connect() as conn:
        await conn.execute(
            update(documents)
            .where(documents.c.name == going["name"])
            .values(status=DocumentStatus.DELETING)
        )
    _, archive = await _backup(client)
    assert _rows(archive)["documents"] == 1
    names = zipfile.ZipFile(io.BytesIO(archive)).namelist()
    assert any(kept["id"] in name for name in names)
    assert not any(going["id"] in name for name in names)


async def test_a_file_gone_after_the_snapshot_fails_the_backup(client, monkeypatch) -> None:
    row = await stage_and_import(client, "notes.md", MD.encode())
    taken = backup._snapshot_sync

    def then_forget(target: Path) -> backup.Manifest:
        found = taken(target)
        for path in document.embeddings_dir(row["id"]).glob("*.chunks.parquet"):
            path.unlink()  # as a re-import does, between the snapshot and the archive
        return found

    monkeypatch.setattr(backup, "_snapshot_sync", then_forget)
    started = (await client.post("/api/backup")).json()["operation_id"]
    with pytest.raises(Exception, match="documents changed during the backup"):
        await wait_for(started)
    assert not (home.BACKUP_ROOT / f"{started.split(':')[1]}.zip").exists(), "no half archive"


async def test_a_newer_backup_replaces_the_older_archive(client) -> None:
    first, _ = await _backup(client)
    second, _ = await _backup(client)
    assert [path.name for path in home.BACKUP_ROOT.iterdir()] == [f"{second.split(':')[1]}.zip"]
    assert (await client.get(f"/api/backup/{first}/file")).status_code == 404
    assert (await client.get("/api/backup/backup:../../x/file")).status_code == 404


def _archive(manifest: dict | None, names: tuple[str, ...] = (backup.DATABASE,)) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        if manifest is not None:
            archive.writestr(backup.MANIFEST, msgspec.json.encode(manifest))
        for name in names:
            archive.writestr(name, b"")
    return buffer.getvalue()


MANIFEST = {
    "format": backup.FORMAT,
    "schema_version": db.SCHEMA_VERSION,
    "app_version": "0.25.0",
    "created_at": 1_790_000_000.0,
    "documents": 1,
    "collections": 1,
}


@pytest.mark.parametrize(
    ("body", "status", "detail"),
    [
        pytest.param(b"not a zip", 422, "not a haskie backup", id="not-a-zip"),
        pytest.param(_archive(None), 422, "no manifest", id="no-manifest"),
        pytest.param(_archive({**MANIFEST, "format": 2}), 422, "format 2", id="other-format"),
        pytest.param(
            _archive({**MANIFEST, "schema_version": min(db.UPGRADES) - 1}),
            409,
            "cannot restore that schema",
            id="other-schema",
        ),
        pytest.param(
            _archive(MANIFEST, (backup.DATABASE, "documents/../../x")),
            422,
            "holds 'documents/../../x'",
            id="path-outside",
        ),
        pytest.param(_archive({"format": 1}), 422, "not a haskie backup", id="bad-manifest"),
    ],
)
async def test_a_restore_refuses_what_is_no_backup_of_this_build(
    client, body: bytes, status: int, detail: str
) -> None:
    doc = (await stage_and_import(client, "notes.md", MD.encode()))["name"]
    refused = await _restore(client, body)
    assert refused.status_code == status, refused.text
    assert detail in refused.json()["detail"]
    assert (await client.get(f"/api/documents/{doc}")).json()["status"] == "imported"
    assert not home.RESTORE_ROOT.exists() or not any(home.RESTORE_ROOT.iterdir()), "nor kept"


async def test_a_restore_waits_for_other_work_to_finish(client, monkeypatch) -> None:
    _, archive = await _backup(client)

    async def busy() -> dict[str, int]:
        return {"import_document": 1, "ensure_model": 1}

    monkeypatch.setattr(sysdb, "active_counts_by_name", busy)
    refused = await _restore(client, archive)
    assert refused.status_code == 409
    assert "1 operations are running" in refused.json()["detail"], "downloads do not count"


async def test_writes_wait_while_a_restore_runs(client, monkeypatch) -> None:
    monkeypatch.setattr(backup, "restoring", True)
    refused = await client.post("/api/collections", json={"name": "late"})
    assert refused.status_code == 503
    assert (await client.get("/api/collections")).status_code == 200


async def test_the_swap_runs_twice_and_undoes_itself_on_failure(client, monkeypatch) -> None:
    doc = await _seed(client)
    _, archive = await _backup(client)
    key = "f" * 32
    backup.upload_path(key).parent.mkdir(parents=True)
    backup.upload_path(key).write_bytes(archive)
    backup._extract_sync(key)

    # a failure in the rows leaves the folders where they were
    before = sorted(path.name for path in home.DOCUMENT_ROOT.rglob("*"))

    def failing(_restored: Path) -> list[str]:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(backup, "_replace_rows", failing)
    with pytest.raises(sqlite3.OperationalError):
        backup._swap_sync(key)
    assert sorted(path.name for path in home.DOCUMENT_ROOT.rglob("*")) == before
    assert (backup._work(key) / "archive" / "documents").is_dir(), "the restored ones move back"
    assert Collection("kept").root.is_dir()

    # a crash after the current documents moved aside: the next boot makes an empty `documents/`,
    # and the replay still moves the restored ones in rather than skipping onto the empty folder
    monkeypatch.undo()
    replaced = backup._work(key) / "replaced"
    replaced.mkdir(parents=True, exist_ok=True)
    home.DOCUMENT_ROOT.rename(replaced / "documents")
    home.ensure_home_sync()
    backup._swap_sync(key)
    backup._swap_sync(key)  # and a replay of a swap that finished changes nothing
    restored_files = sorted(path.name for path in home.DOCUMENT_ROOT.rglob("*"))
    assert "original.md" in restored_files and "original.md.md" in restored_files
    row = await document.named(doc)
    assert row.status == DocumentStatus.IMPORTED
    assert (await Collection("kept").member(row.id)).status == "pending", "indexed again later"
    assert Collection("kept").root.is_dir() and not Collection("kept").index_dir.exists()


async def test_a_swap_replayed_onto_an_empty_home_keeps_the_restored_files(client) -> None:
    """A fresh home has an empty `documents/`, which the swap moves aside as it is. A replay of the
    whole swap, after a crash before DBOS recorded it, must not move the restored files aside
    too, where the cleanup would delete them."""
    await _seed(client)
    _, archive = await _backup(client)
    key = "e" * 32
    backup.upload_path(key).parent.mkdir(parents=True)
    backup.upload_path(key).write_bytes(archive)
    backup._extract_sync(key)
    await home.remove_tree(home.DOCUMENT_ROOT)  # the empty home a restore usually lands on
    home.ensure_home_sync()
    backup._swap_sync(key)
    backup._swap_sync(key)  # the replay
    restored_files = sorted(path.name for path in home.DOCUMENT_ROOT.rglob("*"))
    assert "original.md" in restored_files and "original.md.md" in restored_files


async def test_a_failed_restore_leaves_the_contents_and_no_copy(client, monkeypatch) -> None:
    doc = (await stage_and_import(client, "notes.md", MD.encode()))["name"]
    _, archive = await _backup(client)

    def failing(_restored: Path) -> list[str]:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(backup, "_replace_rows", failing)
    started = (await _restore(client, archive)).json()["operation_id"]
    with pytest.raises(Exception, match="disk I/O error"):
        await wait_for(started)
    assert (await client.get(f"/api/documents/{doc}")).json()["status"] == "imported"
    assert list(home.RESTORE_ROOT.iterdir()) == [], "the unpacked archive and the upload are gone"
    assert (await client.post("/api/collections", json={"name": "after"})).status_code == 201


async def test_a_second_restore_while_one_waits_is_refused(client, monkeypatch) -> None:
    _, archive = await _backup(client)

    async def queued(*_args, **_kwargs) -> str:
        return "restore:" + "0" * 32  # the one already waiting, returned by the deduplication

    monkeypatch.setattr(workflows, "enqueue_operation", queued)
    refused = await _restore(client, archive)
    assert refused.status_code == 409
    assert "a restore is already queued" in refused.json()["detail"]
    assert list(home.RESTORE_ROOT.iterdir()) == [], "the second upload is not kept"


async def test_a_slow_backup_reports_its_progress_as_it_goes(client, monkeypatch) -> None:
    await stage_and_import(client, "notes.md", MD.encode())
    write = backup._write_archive_sync

    def slowly(key: str, manifest: backup.Manifest, members: list, progress) -> int:
        time.sleep(0.3)  # long enough for a few progress events, so one is published mid-step
        return write(key, manifest, members, progress)

    monkeypatch.setattr(backup, "PROGRESS_SECONDS", 0.05)
    monkeypatch.setattr(backup, "_write_archive_sync", slowly)
    started = (await client.post("/api/backup")).json()["operation_id"]

    async def reported() -> bool:
        progress = (await client.get(f"/api/operations/{started}/progress")).json()["progress"]
        return progress is not None

    await until(reported, "no progress while the archive was written")
    await wait_for(started)
    final = (await client.get(f"/api/operations/{started}/progress")).json()["progress"]
    assert final["done"] == final["total"] == 4, "two files of the document, two of its cache"


def _tampered(archive: bytes, statement: str, schema_version: int | None = None) -> bytes:
    """A real archive with one statement run on its rows, as a crafted upload would be, or as an
    older build would have written it when `schema_version` names its schema."""
    source = zipfile.ZipFile(io.BytesIO(archive))
    conn = sqlite3.connect(":memory:")
    conn.deserialize(source.read(backup.DATABASE))
    conn.execute(statement)
    conn.commit()
    rows = conn.serialize()
    conn.close()
    replaced = {backup.DATABASE: rows}
    if schema_version is not None:
        manifest = msgspec.json.decode(source.read(backup.MANIFEST))
        manifest["schema_version"] = schema_version
        replaced[backup.MANIFEST] = msgspec.json.encode(manifest)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as tampered:
        for name in source.namelist():
            tampered.writestr(name, replaced[name] if name in replaced else source.read(name))
    return buffer.getvalue()


async def test_a_backup_of_the_schema_before_page_counts_restores_and_counts_them(client) -> None:
    """An archive an older build wrote restores: the column it lacks takes its default, and its
    PDFs then get the page count that build never stored."""
    row = await stage_and_import(client, "paper.pdf", text_pdf(["alpha", "beta", "gamma"]))
    _, archive = await _backup(client)
    older = _tampered(archive, "alter table documents drop column pages", 34)

    restoring = await _restore(client, older)

    assert restoring.status_code == 202, restoring.text
    await wait_for(restoring.json()["operation_id"])
    restored = (await client.get(f"/api/documents/{row['name']}")).json()
    assert (restored["status"], restored["pages"]) == ("imported", 3)


@pytest.mark.parametrize(
    ("statement", "detail"),
    [
        pytest.param(
            "update collections set name = '../../escape'", "the collection", id="collection-path"
        ),
        pytest.param("update documents set id = '../../escape'", "the document", id="document-id"),
        pytest.param("update documents set suffix = '/../../x'", "the document", id="suffix"),
        pytest.param("update embeddings set id = '../x'", "the embedding", id="cache-id"),
        pytest.param("drop table embeddings", "no such table", id="missing-table"),
    ],
)
async def test_a_restore_refuses_rows_that_would_reach_outside_the_home(
    client, statement: str, detail: str
) -> None:
    await _seed(client)
    _, archive = await _backup(client)
    refused = await _restore(client, _tampered(archive, statement))
    assert refused.status_code == 422, refused.text
    assert detail in refused.json()["detail"]
    assert not (home.HOME.parent / "escape").exists()
    assert not (home.COLLECTION_ROOT.parent / "escape").exists()
    assert (await client.get("/api/collections/kept")).status_code == 200, "nothing replaced"


async def test_the_nightly_round_sweeps_what_restores_that_never_ran_left(
    client, monkeypatch
) -> None:
    """A restore cancelled while it waited never reaches its own cleanup: the nightly round drops
    its upload once it is a day old. Not a fresh one (its restore may not be queued yet), not one
    whose restore still waits, and never a `replaced/` copy a failed rollback left behind."""

    def restore_folder(key: str, age: float, replaced: bool = False) -> Path:
        work = backup._work(key)
        backup.upload_path(key).parent.mkdir(parents=True)
        backup.upload_path(key).write_bytes(b"zip")
        (work / "archive" / "documents").mkdir(parents=True)
        if replaced:
            (work / "replaced" / "documents").mkdir(parents=True)
            (work / "replaced" / "documents" / "kept").write_bytes(b"the only copy")
        aged = time.time() - age
        os.utime(work, (aged, aged))
        return work

    stale = restore_folder("a" * 32, workflows.STAGING_TTL_SECONDS + 60)
    rolled_back = restore_folder("b" * 32, workflows.STAGING_TTL_SECONDS + 60, replaced=True)
    restore_folder("c" * 32, 60)
    restore_folder("d" * 32, workflows.STAGING_TTL_SECONDS + 60)
    listed = DBOS.list_workflows_async

    async def still_waiting(**query):
        found = await listed(**query)
        if query.get("name") == backup.RESTORE_BACKUP_WORKFLOW:
            return [*found, SimpleNamespace(workflow_id=f"restore:{'d' * 32}")]
        return found

    monkeypatch.setattr(DBOS, "list_workflows_async", still_waiting)
    await workflows.daily_maintenance(datetime.now(UTC), None)

    assert not stale.exists(), "the upload and the archive go, and the empty folder with them"
    assert (rolled_back / "replaced" / "documents" / "kept").is_file(), "the only copy stays"
    assert not backup.upload_path("b" * 32).exists() and not (rolled_back / "archive").exists()
    assert backup.upload_path("c" * 32).is_file(), "a fresh upload may not be queued yet"
    assert backup.upload_path("d" * 32).is_file(), "a waiting restore still reads its upload"


async def test_the_archive_step_replayed_after_a_crash_writes_it_again(client) -> None:
    """A crash after the archive is written, before DBOS records the step, replays it: the rows it
    reads are still there, and only `prune` removes them."""
    await stage_and_import(client, "notes.md", MD.encode())
    key = "f" * 32
    manifest = await backup.snapshot(key)  # outside a workflow, a step is a plain call
    rows = home.BACKUP_ROOT / f"{key}.db"
    sizes = []
    for _ in range(2):  # the run, then its replay: what `write_archive` does in its thread
        members = backup._members(rows)
        progress = BulkProgress(done=0, total=len(members))
        sizes.append(backup._write_archive_sync(key, manifest, members, progress))
    assert sizes[0] == sizes[1] and progress.done == 4
    await backup.prune(key)
    assert [path.name for path in home.BACKUP_ROOT.iterdir()] == [f"{key}.zip"]
