"""Backup and restore: every document, collection and setting in one zip, and back.

A backup holds the contents and nothing a machine makes for itself. That is the rows of
`CONTENT_TABLES`, in a database of their own, and each document's original, its markdown and its
embedding cache. Left out:
- the LanceDB indexes, which the index stage rebuilds from the cache without embedding anything
- previews, covers and convert parts, which are built again on demand
- DBOS's run history, sessions, the search log and the audit trail, which record what this
  machine did
- the model catalogue, which every build seeds for itself

A restore replaces the contents and keeps the rest, so the Operations view still lists the restore
that ran.

Both run as durable workflows on a queue of their own (`workflows.BACKUP_QUEUE`), one at a time, so
they show in the Operations view and a crash resumes them. Every step can run twice:
- the archive is written through `home.atomic_replace`, so a crash leaves no half of one
- the swap skips a rename already made, and replaces the rows in one transaction

The rows go through `sqlite3`, not `db.connect`: `ATTACH` is refused inside a transaction, and
`db` opens one on every connection.

An archive is a snapshot of one schema: a restore refuses one whose manifest names another
`db.SCHEMA_VERSION` (no migrations before 1.0, see docs/storage.md).
"""

import asyncio
import contextlib
import re
import shutil
import sqlite3
import time
import zipfile
from pathlib import Path
from uuid import uuid4

import anyio
import anyio.to_thread
import msgspec
from dbos import DBOS
from sqlalchemy import Table, select

from haskie import APP_VERSION, claude, db, home, ids, logs, sysdb
from haskie.collection.collection import Collection, MemberStatus
from haskie.collection.index import forget_every_schema
from haskie.document import convert, document
from haskie.document.document import ACTIVE_DOCUMENT_STATUSES, Document, DocumentStatus
from haskie.errors import Conflict, InvalidInput, NotFound
from haskie.indexing import embed_cache, workflows
from haskie.indexing.dbos_names import (
    ACTIVE_STATUS,
    CREATE_BACKUP_WORKFLOW,
    DAILY_MAINTENANCE_WORKFLOW,
    DOWNLOAD_WORKFLOW,
    RESTORE_BACKUP_WORKFLOW,
    root_cause,
)
from haskie.indexing.workflows import PROGRESS_EVENT, BulkProgress, PipelineError, retried_step
from haskie.settings import forget_user_settings, load_user_settings
from haskie.tables import collection_documents, collections, documents, embeddings, settings

_log = logs.get_logger(__name__)

FORMAT = 1  # of the archive's layout; a restore reads this one only
MANIFEST = "manifest.json"
DATABASE = "haskie.db"
# What a backup holds of the database, parents before children: the order the rows go back in.
CONTENT_TABLES: tuple[Table, ...] = (
    settings,
    collections,
    documents,
    embeddings,
    collection_documents,
)
BACKUP_PREFIX = "backup"  # `backup:{key}`, and the archive is `backups/{key}.zip`
RESTORE_PREFIX = "restore"  # `restore:{key}`, unpacked into `restoring/{key}/`
KEY = re.compile(r"[0-9a-f]{32}")  # the uuid that ends an id: a path is built from it
CACHE_ID = re.compile(r"[0-9a-f]{64}")  # an embedding cache id: the sha256 `embed_cache.key` makes
PROGRESS_SECONDS = 1.0  # between two progress events of a backup, as often as Operations polls
# Work that may run through a restore: model downloads touch no content, the nightly round only
# prunes history, and a backup queued beside it runs before or after it on the same queue.
UNAFFECTED = frozenset(
    {CREATE_BACKUP_WORKFLOW, RESTORE_BACKUP_WORKFLOW, DOWNLOAD_WORKFLOW, DAILY_MAINTENANCE_WORKFLOW}
)
DEFLATED = frozenset({".md", ".db", ".json"})  # the rest (PDF, EPUB, parquet) is compressed already

# Every name an archive may hold. A restore refuses any other, so no name can point outside the
# folder it is unpacked into (`..`, an absolute path).
_DOCUMENT = rf"documents/[0-9a-f]{{2}}/{ids.ID.pattern}"
MEMBER = re.compile(
    rf"{re.escape(MANIFEST)}|{re.escape(DATABASE)}"
    rf"|{_DOCUMENT}/original\.[a-z0-9]+(?:\.md)?"
    rf"|{_DOCUMENT}/embeddings/{CACHE_ID.pattern}\.(?:chunks|sections)\.parquet"
)

# Set while a restore runs its steps. `app.guard_callers` turns away every request but a GET
# meanwhile, MCP's included (each is a POST); GETs stay open, so the Operations view can follow it.
restoring = False


class Manifest(msgspec.Struct):
    """What an archive says about itself, at its root."""

    format: int
    schema_version: int
    app_version: str
    created_at: float  # unix seconds
    documents: int
    collections: int


class Member(msgspec.Struct, frozen=True):
    """One file a backup copies: its path under the home, which is also its name in the archive."""

    path: str
    required: bool  # a document's markdown is missing until its import converts it


class Backup(msgspec.Struct):
    """What a backup made: the archive behind `GET /api/backup/{id}/file`."""

    size: int  # bytes
    files: int  # the documents' files it holds, beside the manifest and the rows


def _key(operation_id: str) -> str:
    """The uuid at the end of a backup's id. A trust boundary: only an id this module made becomes
    a path."""
    head, _, key = operation_id.partition(":")
    if head != BACKUP_PREFIX or not KEY.fullmatch(key):
        raise NotFound(f"operation not found: {operation_id}")
    return key


async def archive(operation_id: str) -> Path | None:
    """The archive a backup made, or None once a newer one has replaced it."""
    path = home.BACKUP_ROOT / f"{_key(operation_id)}.zip"
    return path if await anyio.Path(path).is_file() else None


def download_name(created_at: float) -> str:
    """What a browser saves an archive as: the day it was made."""
    return f"haskie-backup-{time.strftime('%Y-%m-%d', time.localtime(created_at))}.zip"


# --- the snapshot ---------------------------------------------------------------------------


def _columns(table: Table, derived: bool = True) -> str:
    """The table's columns for a copy; without `derived`, those this machine works out
    (`tables.DERIVED`), so the copy takes their default and the restored home works them out
    again."""
    return ", ".join(
        f'"{column.name}"' for column in table.columns if derived or not column.info.get("derived")
    )


def _copy_filter(table: Table) -> str:
    """What a backup leaves out of a table: work on its way out, and what hangs off it."""
    if table is documents:
        return f"where status != '{DocumentStatus.DELETING}'"
    if table is embeddings:
        return "where document_id in (select id from main.documents)"
    if table is collection_documents:
        return (
            f"where status != '{MemberStatus.REMOVING}'"
            " and document_id in (select id from main.documents)"
        )
    return ""


def _snapshot_sync(target: Path) -> Manifest:
    """`CONTENT_TABLES` copied into `target`, read in one transaction so the copy is one moment."""
    target.unlink(missing_ok=True)
    conn = sqlite3.connect(target, timeout=db.BUSY_TIMEOUT_SECONDS, isolation_level=None)
    try:
        conn.executescript(db.schema_ddl(CONTENT_TABLES))
        conn.execute("attach database ? as live", (str(home.DB_FILE),))
        conn.execute("begin")
        for table in CONTENT_TABLES:
            columns = _columns(table, derived=False)
            conn.execute(
                f"insert into main.{table.name} ({columns})"
                f" select {columns} from live.{table.name} {_copy_filter(table)}"
            )
        conn.execute("commit")
        conn.execute("detach database live")
        count = conn.execute(
            "select (select count(*) from documents), (select count(*) from collections)"
        )
        held_documents, held_collections = count.fetchone()
    finally:
        conn.close()
    return Manifest(
        format=FORMAT,
        schema_version=db.SCHEMA_VERSION,
        app_version=APP_VERSION,
        created_at=time.time(),
        documents=held_documents,
        collections=held_collections,
    )


def _members(rows: Path) -> list[Member]:
    """The files the copied rows name, under the home."""
    conn = sqlite3.connect(f"{rows.as_uri()}?mode=ro", uri=True)
    try:
        held = conn.execute("select id, suffix from documents order by id").fetchall()
        entries = conn.execute("select id, document_id from embeddings order by id").fetchall()
    finally:
        conn.close()
    files = [
        *[Member(Document.relative(document.original(doc, suffix)), True) for doc, suffix in held],
        *[Member(Document.relative(document.markdown(doc, suffix)), False) for doc, suffix in held],
        *[
            Member(Document.relative(path(doc, entry)), True)
            for entry, doc in entries
            for path in (embed_cache.file_path, embed_cache.sections_path)
        ],
    ]
    return sorted(files, key=lambda member: member.path)


@retried_step
async def snapshot(key: str) -> Manifest:
    """The rows into `backups/{key}.db`."""
    await anyio.Path(home.BACKUP_ROOT).mkdir(parents=True, exist_ok=True, mode=home.DIR_MODE)
    return await anyio.to_thread.run_sync(_snapshot_sync, home.BACKUP_ROOT / f"{key}.db")


def _write_archive_sync(
    key: str, manifest: Manifest, members: list[Member], progress: BulkProgress
) -> int:
    """The manifest, the rows and every file they name into `backups/{key}.zip`; returns its size.
    Written whole or not at all (`home.atomic_replace`). `progress.done` counts the files as they
    go, for the progress the step reports meanwhile."""
    rows = home.BACKUP_ROOT / f"{key}.db"
    target = home.BACKUP_ROOT / f"{key}.zip"
    with home.atomic_replace(target) as partial:
        with zipfile.ZipFile(partial, "w", allowZip64=True) as archive:
            archive.writestr(MANIFEST, msgspec.json.encode(manifest), zipfile.ZIP_DEFLATED)
            archive.write(rows, DATABASE, compress_type=zipfile.ZIP_DEFLATED)
            for member in members:
                path = home.HOME / member.path
                method = zipfile.ZIP_DEFLATED if path.suffix in DEFLATED else zipfile.ZIP_STORED
                try:
                    archive.write(path, member.path, compress_type=method)
                except FileNotFoundError:
                    if member.required:  # a delete or an import ran after the snapshot
                        raise Conflict(
                            "documents changed during the backup; run it again"
                        ) from None
                progress.done += 1
    return target.stat().st_size  # the rows stay for a replay of this step; `prune` removes them


@retried_step
async def write_archive(key: str, manifest: Manifest) -> Backup:
    """The archive, in one worker thread: a retry writes it again from the start. The step itself
    reports how far it got, once a second."""
    members = await anyio.to_thread.run_sync(_members, home.BACKUP_ROOT / f"{key}.db")
    progress = BulkProgress(done=0, total=len(members))

    async def report() -> None:
        while True:
            await anyio.sleep(PROGRESS_SECONDS)
            await DBOS.set_event_async(PROGRESS_EVENT, progress)

    # a task, not a task group: a group would wrap the step's own failure in an ExceptionGroup
    reporting = asyncio.create_task(report())
    try:
        size = await anyio.to_thread.run_sync(_write_archive_sync, key, manifest, members, progress)
    finally:
        reporting.cancel()
    await DBOS.set_event_async(PROGRESS_EVENT, progress)  # the last count, for the progress route
    return Backup(size=size, files=len(members))


@retried_step
async def prune(key: str) -> None:
    """Keep the newest archive only: a backup is a copy of everything, and one is enough."""
    async for path in anyio.Path(home.BACKUP_ROOT).iterdir():
        if path.name != f"{key}.zip":
            await path.unlink(missing_ok=True)


@DBOS.workflow(name=CREATE_BACKUP_WORKFLOW)
async def create_backup() -> Backup:
    with logs.bound(workflow_id=DBOS.workflow_id):
        key = workflows.run_id(DBOS.workflow_id or "")
        try:
            manifest = await snapshot(key)
            made = await write_archive(key, manifest)
            await prune(key)
        except Exception as exc:
            raise PipelineError(root_cause(exc)) from exc
        _log.info("backup_created", size=made.size, documents=manifest.documents)
        return made


async def start_backup() -> str:
    """Queue a backup; a second ask while one runs returns that one."""
    return await workflows.enqueue_operation(
        workflows.BACKUP_QUEUE,
        create_backup,
        workflow_id=f"{BACKUP_PREFIX}:{uuid4().hex}",
        dedup_id="backup",
    )


# --- the restore ----------------------------------------------------------------------------


def upload_path(key: str) -> Path:
    """Where the route writes a restore's upload, inside the folder the restore works in."""
    return _work(key) / "upload.zip"


def read_manifest(path: Path) -> Manifest:
    """The manifest of the archive at `path`, once it passes `_check`."""
    with _open(path) as archive:
        return _check(archive)


def _open(path: Path) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise InvalidInput(f"not a haskie backup: {exc}") from None


def _check(archive: zipfile.ZipFile) -> Manifest:
    """The archive's manifest, once every name in it is one a backup writes, its format and schema
    are this build's, and its rows hold only ids and names this build writes (`_check_rows`). A
    trust boundary: nothing is unpacked before this passes."""
    names = archive.namelist()
    strange = next((name for name in names if not MEMBER.fullmatch(name)), None)
    if strange is not None:
        raise InvalidInput(f"not a haskie backup: it holds {strange!r}")
    if MANIFEST not in names or DATABASE not in names:
        raise InvalidInput("not a haskie backup: no manifest or no database")
    try:
        manifest = msgspec.json.decode(archive.read(MANIFEST), type=Manifest)
    except msgspec.DecodeError as exc:
        raise InvalidInput(f"not a haskie backup: {exc}") from None
    if manifest.format != FORMAT:
        raise InvalidInput(
            f"backup format {manifest.format} is not supported; this build reads {FORMAT}"
        )
    if manifest.schema_version != db.SCHEMA_VERSION:
        raise Conflict(
            f"backup made by haskie {manifest.app_version} (schema {manifest.schema_version}); "
            f"this build restores schema {db.SCHEMA_VERSION} only"
        )
    _check_rows(archive.read(DATABASE))
    return manifest


def _check_rows(database: bytes) -> None:
    """Refuse rows this build would never write where a row becomes a path: a document's id and
    suffix (`document.root`), an embedding's id, a collection's name (`Collection.root`). The
    archive's names are checked, but its rows are what the restored home builds every path from,
    and an id of `../..` would reach outside it."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.deserialize(database)
        for doc, suffix in conn.execute("select id, suffix from documents"):
            if not ids.ID.fullmatch(str(doc)) or suffix not in convert.SUPPORTED_SUFFIXES:
                raise InvalidInput(f"not a haskie backup: it holds the document {doc!r}")
        for entry, doc in conn.execute("select id, document_id from embeddings"):
            if not CACHE_ID.fullmatch(str(entry)) or not ids.ID.fullmatch(str(doc)):
                raise InvalidInput(f"not a haskie backup: it holds the embedding {entry!r}")
        for (name,) in conn.execute("select name from collections"):
            if not isinstance(name, str) or document.safe_name(name) != name:
                raise InvalidInput(f"not a haskie backup: it holds the collection {name!r}")
    except sqlite3.DatabaseError as exc:
        raise InvalidInput(f"not a haskie backup: {exc}") from None
    finally:
        conn.close()


@retried_step  # a plain call outside a workflow, as the route makes it
async def check_idle() -> None:
    """Refuse a restore while other work runs: it would write rows and files the restore is
    about to replace."""
    counts = await sysdb.active_counts_by_name()
    running = sum(count for name, count in counts.items() if name not in UNAFFECTED)
    if running:
        raise Conflict(f"{running} operations are running; restore once they finish")


def _work(key: str) -> Path:
    return home.RESTORE_ROOT / key


def _extract_sync(key: str) -> None:
    """The upload into `restoring/{key}/archive/`, checked again first: the route checked it, and
    a replay meets whatever is on disk now."""
    unpacked = _work(key) / "archive"
    if unpacked.exists():  # a retry after a crash mid-way unpacks it again
        shutil.rmtree(unpacked)
    unpacked.mkdir(parents=True, mode=home.DIR_MODE)
    with _open(upload_path(key)) as archive:
        _check(archive)
        archive.extractall(unpacked)  # every name passed `MEMBER`: none leaves the folder
    # even for an archive of no documents: the swap reads its presence as "not moved in yet"
    (unpacked / "documents").mkdir(exist_ok=True)


@retried_step
async def extract(key: str) -> None:
    await anyio.to_thread.run_sync(_extract_sync, key)


def _replace_rows(restored: Path) -> list[str]:
    """`CONTENT_TABLES` replaced by the restored ones in one transaction; returns the collections.

    Every membership is pending again: the indexes are rebuilt from the cache. A document the
    backup caught mid-import is queued to import again."""
    conn = sqlite3.connect(home.DB_FILE, timeout=db.BUSY_TIMEOUT_SECONDS, isolation_level=None)
    try:
        # Off, so the delete below does not cascade every session's selection away: a session
        # keeps the collections the archive brings back, and loses the rest (see below).
        conn.execute("pragma foreign_keys = off")
        conn.execute("attach database ? as restored", (str(restored),))
        conn.execute("begin immediate")
        try:
            for table in reversed(CONTENT_TABLES):
                conn.execute(f"delete from main.{table.name}")
            for table in CONTENT_TABLES:
                columns = _columns(table)
                conn.execute(
                    f"insert into main.{table.name} ({columns})"
                    f" select {columns} from restored.{table.name}"
                )
            conn.execute(
                "update main.collection_documents set status = ?, error = null, updated_at = ?",
                (MemberStatus.PENDING, time.time()),
            )
            active = ", ".join(f"'{status}'" for status in ACTIVE_DOCUMENT_STATUSES)
            conn.execute(
                f"update main.documents set status = ?, error = null where status in ({active})",
                (DocumentStatus.QUEUED,),
            )
            conn.execute(
                "delete from main.session_collections"
                " where collection not in (select name from main.collections)"
            )
            names = [name for (name,) in conn.execute("select name from main.collections")]
            conn.execute("commit")
        except BaseException:
            conn.execute("rollback")
            raise
        conn.execute("detach database restored")
    finally:
        conn.close()
    return names


def _move(source: Path, target: Path) -> None:
    """One rename of a swap, skipped once it is made: a replay finds the source gone, or finds
    the target holding what was moved. An empty target is no such thing: a boot makes the home's
    folders afresh (`home.ensure_home_sync`), so a crash between two renames meets an empty
    `documents/` where the restored one belongs."""
    if not source.exists():
        return
    if target.is_dir() and not any(target.iterdir()):
        target.rmdir()
    if not target.exists():
        source.rename(target)


def _swap_sync(key: str) -> list[str]:
    """The restored documents in, the current ones and the indexes aside, then the rows. The
    folders move back when the rows fail, so a failed swap leaves what was there."""
    unpacked, replaced = _work(key) / "archive", _work(key) / "replaced"
    replaced.mkdir(parents=True, exist_ok=True, mode=home.DIR_MODE)
    # Until the restored documents move in, what sits at `documents/` is the home's own. After,
    # it is the restored one: a replay must not move it aside, where the cleanup deletes it.
    if (unpacked / "documents").exists():
        _move(home.DOCUMENT_ROOT, replaced / "documents")
        _move(home.COLLECTION_ROOT, replaced / "collections")
        _move(unpacked / "documents", home.DOCUMENT_ROOT)
    home.ensure_home_sync()
    try:
        names = _replace_rows(unpacked / DATABASE)
    except BaseException:
        _move(home.DOCUMENT_ROOT, unpacked / "documents")
        _move(replaced / "documents", home.DOCUMENT_ROOT)
        home.COLLECTION_ROOT.rmdir()
        _move(replaced / "collections", home.COLLECTION_ROOT)
        raise
    for name in names:
        Collection(name).root.mkdir(parents=True, exist_ok=True)
    return names


@retried_step
async def swap(key: str) -> list[str]:
    """The contents swapped in; returns the collections they hold."""
    return await anyio.to_thread.run_sync(_swap_sync, key)


@retried_step
async def reload() -> None:
    """What this process kept of the contents it replaced: the settings and the index schemas.
    Then the restored settings apply, which loads the models they name."""
    forget_user_settings()
    forget_every_schema()
    claude.refresh_in_background()  # the skill lists the collections
    await workflows.apply_settings_from_workflow(await load_user_settings())


@retried_step
async def queued_imports() -> list[str]:
    """The documents whose import the backup caught before it ended: the swap queued them again."""
    queued = select(documents.c.id).where(documents.c.status == DocumentStatus.QUEUED)
    async with db.read() as conn:
        return list(await conn.scalars(queued))


@retried_step
async def cleanup(key: str, swapped: bool) -> None:
    """The upload and the unpacked archive, and the contents the swap replaced once it committed."""
    if swapped:
        await home.remove_tree(_work(key))
    else:
        await _discard(key)


async def _discard(key: str) -> None:
    """The upload and the unpacked archive of a restore whose swap never committed. `replaced/`
    stays when anything is in it: a swap whose rollback failed left the contents there, and they
    are the only copy."""
    work = _work(key)
    await anyio.Path(upload_path(key)).unlink(missing_ok=True)
    await home.remove_tree(work / "archive")
    for folder in (work / "replaced", work):
        with contextlib.suppress(OSError):
            await anyio.Path(folder).rmdir()  # only when empty


async def sweep_restores(max_age_seconds: float) -> int:
    """`_discard` what restores that never ran left behind, older than `max_age_seconds`; returns
    how many it swept. A restore cancelled while it waited for its queue never reaches its own
    cleanup. One that still waits or runs keeps its folder, and so does one younger than the age:
    the route writes the upload before it queues the restore."""
    root = anyio.Path(home.RESTORE_ROOT)
    if not await root.is_dir():
        return 0
    waiting = await DBOS.list_workflows_async(
        name=RESTORE_BACKUP_WORKFLOW, status=ACTIVE_STATUS, load_input=False, load_output=False
    )
    active = {workflows.run_id(run.workflow_id) for run in waiting}
    cutoff = time.time() - max_age_seconds
    swept = 0
    async for folder in root.iterdir():
        if folder.name in active or (await folder.stat()).st_mtime >= cutoff:
            continue
        await _discard(folder.name)
        swept += 1
    return swept


@DBOS.workflow(name=RESTORE_BACKUP_WORKFLOW)
async def restore_backup() -> list[str]:
    """Replace the contents with the archive the route put in `restoring/{key}/`, then rebuild
    every index from its cache; returns the operations it started to rebuild what it put back:
    the indexes, and the imports the backup caught before they ended."""
    global restoring
    with logs.bound(workflow_id=DBOS.workflow_id):
        key = workflows.run_id(DBOS.workflow_id or "")
        restoring = True  # first: nothing may start between the check and the swap
        swapped = False
        try:
            await check_idle()
            await extract(key)
            names = await swap(key)
            swapped = True
            await reload()
            imports = await queued_imports()
        except Exception as exc:
            # the swap rolls itself back when it fails, so before it nothing was replaced
            await cleanup(key, swapped)
            raise PipelineError(root_cause(exc)) from exc
        finally:
            restoring = False
        # Not steps: DBOS starts no workflow inside one. The ids end with this restore's key, so a
        # replay re-attaches to the runs it already started, with no admission check to fail on
        # a run that has moved on since.
        started = [
            *[await workflows.enqueue_index_collection(name, key) for name in names],
            *[await workflows.enqueue_import(doc, key) for doc in imports],
        ]
        await cleanup(key, swapped)
        _log.info("backup_restored", collections=len(names))
        return started


async def start_restore(key: str) -> str:
    """Queue the restore of the archive at `upload_path(key)`. A second one while the first waits
    is refused, rather than answered with the first: it names another archive."""
    asked = f"{RESTORE_PREFIX}:{key}"
    started = await workflows.enqueue_operation(
        workflows.BACKUP_QUEUE, restore_backup, workflow_id=asked, dedup_id="restore"
    )
    if started != asked:
        raise Conflict(f"a restore is already queued ({started}); wait for it to finish")
    return started
