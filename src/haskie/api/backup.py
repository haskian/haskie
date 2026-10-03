"""Backup and restore: everything in one archive, made and put back as operations of their own."""

from uuid import uuid4

import anyio
import anyio.to_thread
from litestar import Request, get, post
from litestar.response import File

from haskie import audit, backup, home
from haskie.api.common import BulkStarted
from haskie.errors import NotFound

WRITE_BYTES = 4 * 1024 * 1024  # an upload arrives in chunks of tens of KB


@post("/api/backup", status_code=202)
@audit.audited("backup.create")
async def create_backup() -> BulkStarted:
    """Start a backup of every document, collection and setting. Follow it at
    /api/operations/{operation_id}/progress; its archive is at /api/backup/{operation_id}/file."""
    operation_id = await backup.start_backup()
    audit.attach(operation_id=operation_id)
    return BulkStarted(operation_id=operation_id)


@get("/api/backup/{operation_id:str}/file")
async def get_backup_file(operation_id: str) -> File:
    """The archive a backup made, until a newer one replaces it."""
    path = await backup.archive(operation_id)
    if path is None:
        raise NotFound(f"backup not found: {operation_id}")
    made = (await anyio.Path(path).stat()).st_mtime
    return File(path=path, filename=backup.download_name(made), media_type="application/zip")


@post("/api/restore", status_code=202, request_max_body_size=None)
@audit.audited("backup.restore")
async def restore_backup(request: Request) -> BulkStarted:
    """Replace every document, collection and setting with an archive's, sent as the raw request
    body (`application/zip`). The search history and the operations stay, and so do sessions,
    each keeping the collections the archive holds. Refused before anything is replaced when the
    archive is not a backup of this schema, or while other work runs. Only GET requests are served
    while it runs. Every index is rebuilt afterwards, and every document the backup caught
    mid-import is imported again, as operations of their own.

    The body is written to disk as it arrives: an archive holds every document, far over the
    upload cap."""
    key = uuid4().hex
    path = backup.upload_path(key)
    await anyio.Path(path.parent).mkdir(parents=True, mode=home.DIR_MODE)
    try:
        size, pending = 0, bytearray()
        async with await anyio.open_file(path, "wb") as file:
            async for chunk in request.stream():
                size += len(chunk)
                pending += chunk
                if len(pending) >= WRITE_BYTES:  # one write, so one thread hop, per block
                    await file.write(pending)  # done before it returns, so the clear is safe
                    pending.clear()
            await file.write(pending)
        audit.attach(size=size)
        manifest = await anyio.to_thread.run_sync(backup.read_manifest, path)
        await backup.check_idle()
        operation_id = await backup.start_restore(key)
    except BaseException:
        await home.remove_tree(path.parent)
        raise
    audit.attach(operation_id=operation_id, documents=manifest.documents)
    return BulkStarted(operation_id=operation_id)
