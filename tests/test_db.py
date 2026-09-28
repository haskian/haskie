"""The unit of work `db.connect` opens: one transaction from its first statement to its commit.

A unit that reads, decides, then writes must not act on a read another unit's commit has made
stale in between. The race is played on real connections through the real entry points: a
collection rename checks that no member is being deleted, and a document delete marks its
document `deleting` and snapshots its memberships while the rename sits between its check and its
first write.
"""

from dataclasses import dataclass

import anyio
import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.util import await_only

from haskie import db
from haskie.collection.collection import Collection
from haskie.document import document
from haskie.document.document import DocumentStatus

from conftest import import_row  # isort: skip

PAUSE_SECONDS = 0.5  # how long the rename sits between its check and its first write


@dataclass(frozen=True)
class RaceCase:
    name: str
    busy_timeout: float  # how long the delete's unit waits for the rename's lock
    delete_error: str | None  # the busy error the delete surfaces, when it gives up waiting


@dataclass
class Race:
    checked: anyio.Event
    deleted: anyio.Event
    deleted_during_pause: bool | None = None
    snapshot: list[str] | None = None
    delete_error: str | None = None


@pytest.mark.parametrize(
    "case",
    [
        RaceCase("the delete waits for the rename to commit", db.BUSY_TIMEOUT_SECONDS, None),
        RaceCase("the delete gives up waiting with a busy error", 0.1, "database is locked"),
    ],
    ids=lambda case: case.name,
)
@pytest.mark.anyio
async def test_a_write_waits_for_a_unit_between_its_read_and_its_write(
    case: RaceCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rename's check and its write are one transaction, so the delete's `deleting` cannot
    land between them: the delete waits for the rename, then snapshots the memberships under the
    new name. Before, the check ran outside any transaction: the delete committed during the
    pause, snapshotted the old name, and its removal would have found nothing there."""
    monkeypatch.setattr(db, "BUSY_TIMEOUT_SECONDS", case.busy_timeout)
    collection = await Collection.create("health")
    doc = (await import_row("a.md")).name
    await document.set_status(doc, DocumentStatus.IMPORTED)
    await collection.add(doc)
    race = Race(checked=anyio.Event(), deleted=anyio.Event())

    async def pause() -> None:
        race.checked.set()
        with anyio.move_on_after(PAUSE_SECONDS):
            await race.deleted.wait()
        race.deleted_during_pause = race.deleted.is_set()

    def pause_after_member_check(_conn, _cursor, statement: str, parameters, *_args) -> None:
        # the rename's check is the one SELECT that asks for a `deleting` member
        is_check = statement.lstrip().upper().startswith("SELECT") and "deleting" in parameters
        if is_check and not race.checked.is_set():
            await_only(pause())  # the listener runs in SQLAlchemy's greenlet, on the loop

    async def delete_start() -> None:
        """The first two steps of `delete_document_workflow`: mark, then snapshot."""
        await race.checked.wait()
        try:
            await document.set_status(doc, DocumentStatus.DELETING)
        except OperationalError as error:
            race.delete_error = str(error.orig)
            return
        race.deleted.set()
        race.snapshot = await document.collections_of(doc)

    event.listen(db.engine().sync_engine, "after_cursor_execute", pause_after_member_check)
    try:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(delete_start)
            tasks.start_soon(collection.rename, "new")
    finally:
        event.remove(db.engine().sync_engine, "after_cursor_execute", pause_after_member_check)

    assert race.deleted_during_pause is False, f"{case.name}: no commit between check and write"
    assert await document.collections_of(doc) == ["new"], f"{case.name}: the rename committed"
    assert race.delete_error == case.delete_error, case.name
    status = (await document.get(doc)).status
    if case.delete_error is None:
        assert race.snapshot == ["new"], f"{case.name}: the delete sees the renamed membership"
        assert status == DocumentStatus.DELETING, case.name
    else:
        assert race.snapshot is None, f"{case.name}: nothing snapshotted after the busy error"
        assert status == DocumentStatus.IMPORTED, f"{case.name}: the failed mark wrote nothing"
