"""Grouped reads of the DBOS system tables, over a document that really went through the pipeline.

The rows come from DBOS, never from this test: the only way to be sure the SQL matches the schema
DBOS writes is to import a document, attach it to a collection and read what that left behind.
"""

import time
from pathlib import Path

import pytest
from dbos import DBOS
from sqlalchemy import text

from haskie import sysdb
from haskie.collection.collection import Collection
from haskie.indexing import dbos_names, operations, workflows
from haskie.settings import PipelineSettings, UserSettings, save_user_settings

from conftest import attach_document, import_document, text_pdf, wait_for  # isort: skip

pytestmark = pytest.mark.anyio


async def _imported(dbos, tmp_path: Path, pages: int) -> str:
    """One document imported with one page per batch, so every stage has `pages` batches. The
    budget gives convert and embed two slices each. Returns the document name."""
    # An hour of debounce, because these tests read what the pipeline left behind and the default
    # minute expires under them on a loaded machine: the maintenance run the index asked for wakes
    # up and enqueues its own child on `task.indexing`, which is then a row nobody asked for.
    indexing = PipelineSettings(
        cpu_budget=6, batch_pages=1, index_group_parts=1, maintenance_idle_seconds=3600
    )
    await dbos.apply_settings(await save_user_settings(UserSettings(pipeline=indexing)))
    row = await import_document(
        dbos, "p.pdf", text_pdf([f"alpha{i}" for i in range(pages)]), tmp_path
    )
    return row.name


async def _run_id(action: str, collection: str | None = None) -> str:
    """The id of the one pipeline run with this action (and collection): the ids carry a uuid, so
    a test that names a child workflow has to read the parent's id first."""
    found = [
        run
        for run in (await operations._pipeline_page(page_size=50)).items
        if run.action == action and run.collection == collection
    ]
    assert len(found) == 1, f"expected one {action} run, got {[run.id for run in found]}"
    return found[0].id


async def test_step_counts_group_by_workflow(dbos, tmp_path) -> None:
    """The import is the parent of its convert slices and the embed run of its own. One grouped
    query counts the batches every one of them recorded."""
    doc = await _imported(dbos, tmp_path, pages=2)
    import_id = await _run_id("import")
    embed_id = await _run_id("embed")
    converts = [f"{import_id}:convert:{i}" for i in range(2)]
    embeds = [f"{embed_id}:embed:{i}" for i in range(2)]

    assert await sysdb.step_counts([*converts, *embeds], dbos_names.STAGE_STEP) == dict.fromkeys(
        [*converts, *embeds], 1
    ), "one of the two batches per slice"

    await Collection.create("grp")
    await attach_document(dbos, "grp", doc)
    index_id = await _run_id("index", "grp")

    assert await sysdb.step_counts([f"{index_id}:index"], dbos_names.STAGE_STEP) == {
        f"{index_id}:index": 2
    }, "the index stage is never sliced"
    assert await sysdb.step_counts(converts, "no-such-step") == {}, "a name nothing recorded"
    assert await sysdb.step_counts([], dbos_names.STAGE_STEP) == {}, "no ids, no query"


async def test_operation_activity_counts_the_operation_queues_by_status(dbos, tmp_path) -> None:
    """Once a document is imported and indexed, its operation and task workflows are all SUCCESS
    and drop out.

    What remains is the maintenance run the collection index asked for, DELAYED on
    `operation.maintenance` until its debounce expires. That is not queued work: nobody is waiting
    on it, and it sits there for a whole `maintenance_idle_seconds`. Counting it made the indicator
    read "1 queued" with an idle machine, while the Operations view - which counts `ACTIVE_STATUS`
    - showed nothing.
    """
    from haskie import db

    doc = await _imported(dbos, tmp_path, pages=1)
    await Collection.create("act")
    await attach_document(dbos, "act", doc)
    # Set the state rather than waiting for it: `conftest._sweep_delayed` promotes an expired
    # debounce every 50 ms, so "is the maintenance run still DELAYED" is a race, not a fact. The
    # wake time goes with it, an hour out, because the sweep promotes on that column alone.
    async with db.connect() as conn:
        await conn.exec_driver_sql(
            "update workflow_status set status = 'DELAYED', delay_until_epoch_ms = ? "
            "where queue_name = ?",
            (int((time.time() + 3600) * 1000), workflows.MAINTENANCE_QUEUE),
        )
        delayed = await conn.scalar(
            text("select count(*) from workflow_status where status = 'DELAYED'")
        )
    assert delayed is not None and delayed >= 1, "there is a debounced run to ignore"
    assert await sysdb.operation_activity() == {}, (
        "a debounce waiting out its period is not activity"
    )

    async with db.connect() as conn:  # a slice still waiting for a slot, and one running
        # The queue this slice waits on is one the app never registered, because DBOS is up: a row
        # left ENQUEUED on a real queue with a free slot is dequeued within a poll, and the status
        # this line is asserting on is gone before the read. The family still reads off the prefix.
        await conn.exec_driver_sql(
            "update workflow_status set status = 'ENQUEUED', queue_name = 'task.parked' "
            "where name = ? and workflow_uuid like '%:convert:%'",
            (dbos_names.STAGE_WORKFLOW,),
        )
        await conn.exec_driver_sql(
            "update workflow_status set status = 'PENDING' where workflow_uuid like '%:index'"
        )
        # the embedding of the import and the one the collection index asked for
        await conn.exec_driver_sql(
            "update workflow_status set status = 'PENDING' where queue_name = ?",
            (workflows.EMBEDDING_QUEUE,),
        )
    assert await sysdb.operation_activity() == {"PENDING": 2}, (
        "the operations by status; the task rows waiting and running beside them are not counted"
    )
    assert await sysdb.operation_activity(skip=[workflows.EMBEDDING_QUEUE]) == {}, (
        "a skipped queue drops out, and no status is left"
    )


async def test_adopted_workflows_of_an_older_build_run_to_the_end(dbos, tmp_path) -> None:
    """`adopt_orphans` reads in-flight workflows of an older build a page at a time, oldest first,
    moves them onto this build and resumes each on its own queue. The real resume runs here: DBOS
    dequeues only its own application version, so a resume alone left them enqueued forever."""
    from haskie import db

    await _imported(dbos, tmp_path, pages=1)
    import_id = await _run_id("import")
    async with db.connect() as conn:  # pretend the work is still waiting under an older build
        await conn.exec_driver_sql(
            "update workflow_status set status = 'ENQUEUED', application_version = 'old-build'"
        )

    stale = await sysdb.stale_active(workflows.APP_VERSION, limit=10)

    assert import_id in dict(stale) and len(stale) == 4, (
        "the import and its one convert slice, the embedding it warmed and that one's embed slice"
    )
    assert await sysdb.stale_active(workflows.APP_VERSION, limit=2) == stale[:2], "limit"
    assert await sysdb.stale_active("old-build", limit=10) == [], "this build's own workflows"
    assert all(queue is not None for _, queue in stale), "each waits on a queue of its own"

    assert await workflows.adopt_orphans(batch=3) == 4, "one page, then the remainder"

    assert await sysdb.stale_active(workflows.APP_VERSION, limit=10) == [], "all moved over"
    assert await wait_for(import_id) == "imported", "the adopted import runs to the end"
    for workflow_id, queue in stale:
        status = await DBOS.get_workflow_status_async(workflow_id)
        assert status is not None and status.status == "SUCCESS", workflow_id
        assert status.queue_name == queue, "resumed on its own queue, under its limits"
    assert await workflows.adopt_orphans() == 0, "adopted once"


def test_the_sysdb_page_stays_under_sqlites_parameter_limit() -> None:
    """Every grouped read binds one parameter per id of a page, plus a handful of its own."""
    assert sysdb.SYSDB_PAGE <= 990
