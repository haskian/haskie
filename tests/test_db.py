"""The two units of work `db` opens: `connect` for a unit that may write, `read` for one that only
reads.

A unit that reads, decides, then writes must not act on a read another unit's commit has made
stale in between. The race is played on real connections through the real entry points: a
collection rename checks that no member is being deleted, and a document delete marks its
document `deleting` and snapshots its memberships while the rename sits between its check and its
first write. An attach checks that its document is imported, and the same delete runs while the
attach sits after its check.

A unit that only reads takes no lock, still reads one snapshot, and refuses a write.
"""

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import StrEnum

import anyio
import pytest
from conftest import id_of
from sqlalchemy import event, insert, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.util import await_only

from haskie import db
from haskie.collection.collection import Collection
from haskie.document import document
from haskie.document.document import DocumentStatus
from haskie.tables import collections

from conftest import import_row  # isort: skip

PAUSE_SECONDS = 0.5  # how long a unit sits paused while the delete cannot finish before it
# The ceiling on a pause that ends when the delete does: long enough for a slow runner's delete
# to open its connection and reach the lock, and ended far sooner by the delete itself.
SETTLE_SECONDS = 30.0


@dataclass(frozen=True)
class RaceCase:
    name: str
    busy_timeout: float  # how long the delete's unit waits for the rename's lock
    delete_error: str | None  # the busy error the delete surfaces, when it gives up waiting
    # The rename's pause, at most. A delete that must give up ends it; one that must wait for
    # the rename cannot, so that pause runs its time.
    pause_seconds: float = PAUSE_SECONDS


@dataclass
class Race:
    """A unit paused between its read and its write, and a document delete run meanwhile."""

    doc: str
    pause_seconds: float = SETTLE_SECONDS
    checked: anyio.Event = field(default_factory=anyio.Event)
    deleted: anyio.Event = field(default_factory=anyio.Event)
    settled: anyio.Event = field(default_factory=anyio.Event)  # the delete finished or gave up
    deleted_during_pause: bool | None = None
    snapshot: list[str] | None = None
    delete_error: str | None = None

    async def pause(self) -> None:
        """Where the paused unit waits: until the delete is done, or `pause_seconds`. Not a fixed
        window: a slow runner's delete may not reach the lock inside one."""
        self.checked.set()
        with anyio.move_on_after(self.pause_seconds):
            await self.settled.wait()
        self.deleted_during_pause = self.deleted.is_set()

    async def delete(self) -> None:
        """The first two steps of `delete_document_workflow`, once the unit paused: mark, then
        snapshot. A busy error ends it with nothing written."""
        await self.checked.wait()
        try:
            await document.set_status(await id_of(self.doc), DocumentStatus.DELETING)
        except OperationalError as error:
            self.delete_error = str(error.orig)
            self.settled.set()
            return
        self.snapshot = await document.collections_of(await id_of(self.doc))
        self.deleted.set()
        self.settled.set()


@pytest.mark.parametrize(
    "case",
    [
        RaceCase("the delete waits for the rename to commit", db.BUSY_TIMEOUT_SECONDS, None),
        RaceCase(
            "the delete gives up waiting with a busy error",
            0.1,
            "database is locked",
            SETTLE_SECONDS,
        ),
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
    await document.set_status(await id_of(doc), DocumentStatus.IMPORTED)
    await collection.add(await id_of(doc))
    race = Race(doc, case.pause_seconds)

    def pause_after_member_check(_conn, _cursor, statement: str, parameters, *_args) -> None:
        # the rename's check is the one SELECT that asks for a `deleting` member
        is_check = statement.lstrip().upper().startswith("SELECT") and "deleting" in parameters
        if is_check and not race.checked.is_set():
            await_only(race.pause())  # the listener runs in SQLAlchemy's greenlet, on the loop

    event.listen(db.engine().sync_engine, "after_cursor_execute", pause_after_member_check)
    try:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(race.delete)
            tasks.start_soon(collection.rename, "new")
    finally:
        event.remove(db.engine().sync_engine, "after_cursor_execute", pause_after_member_check)

    assert race.deleted_during_pause is False, f"{case.name}: no commit between check and write"
    assert await document.collections_of(await id_of(doc)) == ["new"], (
        f"{case.name}: the rename committed"
    )
    assert race.delete_error == case.delete_error, case.name
    status = (await document.named(doc)).status
    if case.delete_error is None:
        assert race.snapshot == ["new"], f"{case.name}: the delete sees the renamed membership"
        assert status == DocumentStatus.DELETING, case.name
    else:
        assert race.snapshot is None, f"{case.name}: nothing snapshotted after the busy error"
        assert status == DocumentStatus.IMPORTED, f"{case.name}: the failed mark wrote nothing"


@pytest.mark.anyio
async def test_an_attach_checks_and_inserts_in_one_unit() -> None:
    """The attach reads the document's status and writes the membership in one transaction, so a
    delete's `deleting` and its membership snapshot land either before the check, which refuses
    the attach, or after the insert, which the snapshot then holds. Before, they were two units:
    the delete committed between them, snapshotted no membership, and the attach wrote one the
    delete would never remove.

    The attach pauses when the connection that read the status goes back to the pool, which is
    between the two units before and after the one unit now: no lock is held there either way."""
    collection = await Collection.create("health")
    doc = await import_row("a.md")
    await document.set_status(doc.id, DocumentStatus.IMPORTED)
    race = Race(doc.name)
    reader: list[object] = []  # the DBAPI connection that read the status

    def note_status_read(conn, _cursor, statement: str, parameters, *_args) -> None:
        # the attach's check is the first SELECT of the document's row
        is_read = statement.lstrip().upper().startswith("SELECT") and doc.id in parameters
        if is_read and "FROM documents" in statement and not reader:
            reader.append(conn.connection.dbapi_connection)

    def pause_on_checkin(dbapi_connection, _record) -> None:
        if reader and dbapi_connection is reader[0] and not race.checked.is_set():
            await_only(race.pause())  # the pool returns the connection in SQLAlchemy's greenlet

    sync_engine = db.engine().sync_engine
    event.listen(sync_engine, "after_cursor_execute", note_status_read)
    event.listen(sync_engine.pool, "checkin", pause_on_checkin)
    try:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(race.delete)
            tasks.start_soon(collection.add, doc.id)
    finally:
        event.remove(sync_engine, "after_cursor_execute", note_status_read)
        event.remove(sync_engine.pool, "checkin", pause_on_checkin)

    assert race.deleted_during_pause is True, "the delete ran while the attach paused"
    assert race.snapshot == ["health"], "the delete's snapshot holds the membership"
    assert await document.collections_of(doc.id) == ["health"]


class Meanwhile(StrEnum):
    """What another unit does while the unit under test runs."""

    NOTHING = "nothing"
    COMMITS = "commits"  # between the unit's two reads
    HOLDS_THE_LOCK = "holds the lock"  # from before the unit starts until after it ends


@dataclass(frozen=True)
class EntryCase:
    name: str
    entry: str  # the `db` function that opens the unit
    meanwhile: Meanwhile
    writes: bool  # the unit inserts the collection "mine" after its two reads
    error: str | None  # what the unit raises
    after: list[str]  # the collections committed once every unit ended


async def _add(conn: AsyncConnection, name: str) -> None:
    await conn.execute(insert(collections).values(name=name, created_at=1.0))


@pytest.mark.parametrize(
    "case",
    [
        EntryCase(
            "a read unit reads one snapshot while another unit commits",
            "read",
            Meanwhile.COMMITS,
            writes=False,
            error=None,
            after=["other"],
        ),
        EntryCase(
            "a read unit waits for no unit holding the write lock",
            "read",
            Meanwhile.HOLDS_THE_LOCK,
            writes=False,
            error=None,
            after=["other"],
        ),
        EntryCase(
            "a write through a read unit raises and writes nothing",
            "read",
            Meanwhile.NOTHING,
            writes=True,
            error="attempt to write a readonly database",
            after=[],
        ),
        EntryCase(
            "a write unit commits what it writes",
            "connect",
            Meanwhile.NOTHING,
            writes=True,
            error=None,
            after=["mine"],
        ),
    ],
    ids=lambda case: case.name,
)
@pytest.mark.anyio
async def test_each_entry_point_opens_its_own_kind_of_unit(
    case: EntryCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read unit's two reads agree whatever commits between them, and neither waits for a
    writer: the busy timeout is cut short, so a read that waited for the lock would fail."""
    monkeypatch.setattr(db, "BUSY_TIMEOUT_SECONDS", 0.1)
    names = select(collections.c.name).order_by(collections.c.name)
    reads: list[list[str]] = []
    error: str | None = None
    async with AsyncExitStack() as held:
        if case.meanwhile is Meanwhile.HOLDS_THE_LOCK:
            await _add(await held.enter_async_context(db.connect()), "other")
        try:
            async with getattr(db, case.entry)() as conn:
                reads.append(list(await conn.scalars(names)))
                if case.meanwhile is Meanwhile.COMMITS:
                    async with db.connect() as other:
                        await _add(other, "other")
                reads.append(list(await conn.scalars(names)))
                if case.writes:
                    await _add(conn, "mine")
        except OperationalError as exc:
            error = str(exc.orig)

    assert reads == [[], []], f"{case.name}: both reads see the state the unit started from"
    assert error == case.error, case.name
    async with db.read() as conn:
        assert list(await conn.scalars(names)) == case.after, case.name
