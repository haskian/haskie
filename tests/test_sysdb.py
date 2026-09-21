"""Grouped reads of the DBOS system tables, over a document that really went through the pipeline.

The rows come from DBOS, never from this test: the only way to be sure the SQL matches the schema
DBOS writes is to import a document, attach it to a collection and read what that left behind.
"""

from pathlib import Path

import pytest
from dbos import DBOS

from haskie import dbos_names, jobs, sysdb, workflows
from haskie.collection import Collection
from haskie.settings import PipelineSettings, UserSettings, save_user_settings

from conftest import attach_document, import_document, text_pdf  # isort: skip

pytestmark = pytest.mark.anyio


async def _imported(dbos, tmp_path: Path, pages: int) -> str:
    """One document imported with one page per batch, so every stage has `pages` batches. The
    budget gives convert and embed two slices each. Returns the document name."""
    indexing = PipelineSettings(cpu_budget=6, batch_pages=1, index_group_parts=1)
    await dbos.apply_settings(await save_user_settings(UserSettings(pipeline=indexing)))
    row = await import_document(
        dbos, "p.pdf", text_pdf([f"alpha{i}" for i in range(pages)]), tmp_path
    )
    return row.name


async def _job_id(action: str, collection: str | None = None) -> str:
    """The id of the one job with this action (and collection): the ids carry a uuid, so a test
    that names a child workflow has to read the parent's id first."""
    found = [
        job
        for job in (await jobs.list_jobs(page_size=50)).items
        if job.action == action and job.collection == collection
    ]
    assert len(found) == 1, f"expected one {action} job, got {[job.id for job in found]}"
    return found[0].id


async def test_step_counts_and_child_status_counts_group_by_parent(dbos, tmp_path) -> None:
    """The import is the parent of its convert slices and of the embedding it warms; the embed
    job is the parent of its own slices. One grouped query answers for all of them."""
    doc = await _imported(dbos, tmp_path, pages=2)
    import_id = await _job_id("import")
    embed_id = await _job_id("embed")
    converts = [f"{import_id}:convert:{i}" for i in range(2)]
    embeds = [f"{embed_id}:embed:{i}" for i in range(2)]

    assert await sysdb.child_status_counts([import_id, embed_id]) == {
        import_id: {"SUCCESS": 3},  # two convert slices and the `ensure_embedding` it asked for
        embed_id: {"SUCCESS": 2},  # two embed slices
    }
    assert await sysdb.step_counts([*converts, *embeds], dbos_names.STAGE_STEP) == dict.fromkeys(
        [*converts, *embeds], 1
    ), "one of the two batches per slice"

    await Collection.create("grp")
    await attach_document(dbos, "grp", doc)
    index_id = await _job_id("index", "grp")

    counts = await sysdb.child_status_counts([index_id])
    assert counts[index_id]["SUCCESS"] >= 2, (
        "the index child and the embedding the collection asked for (a cache hit)"
    )
    assert await sysdb.step_counts([f"{index_id}:index"], dbos_names.STAGE_STEP) == {
        f"{index_id}:index": 2
    }, "the index stage is never sliced"
    assert await sysdb.step_counts(converts, "no-such-step") == {}, "a name nothing recorded"
    assert await sysdb.step_counts([], dbos_names.STAGE_STEP) == {}, "no ids, no query"
    assert await sysdb.child_status_counts([]) == {}
    assert await sysdb.child_status_counts(["ghost"]) == {}, "a parent with no children"


async def test_queue_activity_groups_by_queue_family_and_status(dbos, tmp_path) -> None:
    """Once a document is imported and indexed, its job and task workflows are all SUCCESS and
    drop out.

    What remains is the maintenance run the collection index asked for, DELAYED on
    `job.maintenance` until its debounce expires. That is not queued work: nobody is waiting on
    it, and it sits there for a whole `maintenance_idle_seconds`. Counting it made the indicator
    read "1 queued" with an idle machine, while the Jobs view - which counts `ACTIVE_STATUS` -
    showed nothing.
    """
    from haskie import db

    doc = await _imported(dbos, tmp_path, pages=1)
    await Collection.create("act")
    await attach_document(dbos, "act", doc)
    # set the state rather than waiting for it: `conftest._sweep_delayed` promotes an expired
    # debounce every 50 ms, so "is the maintenance run still DELAYED" is a race, not a fact
    async with db.connect() as conn:
        await conn.execute(
            "update workflow_status set status = 'DELAYED' where queue_name = ?",
            (workflows.MAINTENANCE_QUEUE,),
        )
        delayed = list(
            await conn.execute_fetchall(
                "select count(*) from workflow_status where status = 'DELAYED'"
            )
        )
    assert delayed[0][0] >= 1, "there is a debounced run to ignore"
    assert await sysdb.queue_activity() == {}, "a debounce waiting out its period is not activity"

    async with db.connect() as conn:  # a slice still waiting for a slot, and one running
        await conn.execute(
            "update workflow_status set status = 'ENQUEUED' "
            "where name = ? and workflow_uuid like '%:convert:%'",
            (dbos_names.STAGE_WORKFLOW,),
        )
        await conn.execute(
            "update workflow_status set status = 'PENDING' where workflow_uuid like '%:index'"
        )
        # the embedding of the import and the one the collection index asked for
        await conn.execute(
            "update workflow_status set status = 'PENDING' where queue_name = ?",
            (workflows.EMBEDDING_QUEUE,),
        )
    assert await sysdb.queue_activity() == {
        "task": {"ENQUEUED": 1, "PENDING": 1},
        "job": {"PENDING": 2},
    }, "the prefix of the queue name is the family; the status is kept for the caller to fold"
    assert await sysdb.queue_activity(skip=[workflows.EMBEDDING_QUEUE]) == {
        "task": {"ENQUEUED": 1, "PENDING": 1},
    }, "a skipped queue drops out of its family, and an empty family is absent"


async def test_stale_active_ids_pages_over_another_versions_workflows(
    dbos, monkeypatch, tmp_path
) -> None:
    """`adopt_orphans` reads ids of in-flight workflows of an older build, oldest first."""
    from haskie import db

    await _imported(dbos, tmp_path, pages=1)
    import_id = await _job_id("import")
    async with db.connect() as conn:  # pretend the jobs are still running under an older build
        await conn.execute(
            "update workflow_status set status = 'ENQUEUED', application_version = 'old-build'"
        )

    active = await sysdb.stale_active_ids(workflows.APP_VERSION, limit=10)

    assert import_id in active and len(active) == 4, (
        "the import and its one convert slice, the embedding it warmed and that one's embed slice"
    )
    assert await sysdb.stale_active_ids(workflows.APP_VERSION, limit=2) == active[:2], "limit"
    assert await sysdb.stale_active_ids(workflows.APP_VERSION, limit=10, offset=2) == active[2:], (
        "offset"
    )
    assert await sysdb.stale_active_ids("old-build", limit=10) == [], "this build's own workflows"

    resumed: list[str] = []

    async def resume(ids: list[str], **kwargs) -> None:  # the bulk call `adopt_orphans` makes
        resumed.extend(ids)

    monkeypatch.setattr(DBOS, "resume_workflows_async", resume)
    assert await workflows.adopt_orphans(batch=3) == 4, "one page, then the remainder"
    assert sorted(resumed) == sorted(active), "every page adopted, none twice"


def test_the_sysdb_page_stays_under_sqlites_parameter_limit() -> None:
    """Every grouped read binds one parameter per id of a page, plus a handful of its own."""
    assert sysdb.SYSDB_PAGE <= 990
