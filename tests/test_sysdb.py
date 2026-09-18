"""Grouped reads of the DBOS system tables, over a real indexed document.

The rows come from DBOS, never from this test: the only way to be sure the SQL matches the schema
DBOS writes is to run a workflow first and read what it left behind.
"""

import pytest
from dbos import DBOS

from haskie import dbos_names, sysdb, workflows
from haskie.library import Library
from haskie.settings import PipelineSettings, UserSettings, save_user_settings

from conftest import text_pdf, wait_for  # isort: skip

pytestmark = pytest.mark.anyio


async def _indexed_job(dbos, library: str, pages: int) -> str:
    """One document indexed with one page per batch and one part per index write, so every stage
    has `pages` batches. Two workers, so convert and embed are cut into two slices each."""
    indexing = PipelineSettings(cpu_budget=6, batch_pages=1, index_group_parts=1)
    await dbos.apply_settings(await save_user_settings(UserSettings(pipeline=indexing)))
    lib = await Library.create(library)
    doc = await lib.save("p.pdf", text_pdf([f"alpha{i}" for i in range(pages)]))
    job_id = await dbos.start_index(library, doc.name)
    assert await wait_for(job_id) == "indexed"
    return job_id


async def test_step_counts_and_child_status_counts_group_by_parent(dbos) -> None:
    job_id = await _indexed_job(dbos, "grp", pages=2)
    sliced = [f"{job_id}:{kind}:{i}" for kind in ("convert", "embed") for i in range(2)]
    children = [*sliced, f"{job_id}:index"]

    assert await sysdb.child_status_counts([job_id]) == {job_id: {"SUCCESS": 5, "DELAYED": 1}}, (
        "two convert slices, two embed slices and the index child, plus the debounced "
        "maintenance run the document asked for"
    )
    assert await sysdb.step_counts(children, dbos_names.STAGE_STEP) == {
        **dict.fromkeys(sliced, 1),  # one of the two batches per slice
        f"{job_id}:index": 2,  # the index stage is never sliced
    }

    step = workflows.try_batch.__qualname__
    assert dbos_names.STAGE_STEP == step, "the step DBOS actually records"
    assert await sysdb.step_counts(children, "no-such-step") == {}, "a name nothing recorded"
    assert await sysdb.step_counts([], dbos_names.STAGE_STEP) == {}, "no ids, no query"
    assert await sysdb.child_status_counts([]) == {}
    assert await sysdb.child_status_counts(["ghost"]) == {}, "a parent with no children"


async def test_queue_activity_groups_by_queue_family_and_status(dbos) -> None:
    """After a document is indexed, its job and task workflows are all SUCCESS and drop out.

    What remains is the maintenance run the document asked for, DELAYED on `job.maintenance` until
    its debounce expires. That is not queued work: nobody is waiting on it, and it sits there for a
    whole `maintenance_idle_seconds`. Counting it made the indicator read "1 queued" with an idle
    machine, while the Jobs view - which counts `ACTIVE_STATUS` - showed nothing.
    """
    from haskie import db

    await _indexed_job(dbos, "act", pages=1)
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
    assert await sysdb.queue_activity() == {
        "task": {"ENQUEUED": 1, "PENDING": 1},
    }, "the prefix of the queue name is the family; the status is kept for the caller to fold"


async def test_stale_active_ids_pages_over_another_versions_workflows(dbos, monkeypatch) -> None:
    """`adopt_orphans` reads ids of in-flight workflows of an older build, oldest first."""
    from haskie import db

    job_id = await _indexed_job(dbos, "stale", pages=1)
    async with db.connect() as conn:  # pretend the job is still running under an older build
        await conn.execute(
            "update workflow_status set status = 'ENQUEUED', application_version = 'old-build'"
        )

    active = await sysdb.stale_active_ids(workflows.APP_VERSION, limit=10)

    assert job_id in active and len(active) == 5, (
        "the job, its three stage children (one page, so one slice each) and the maintenance run "
        "it requested"
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
    assert await workflows.adopt_orphans(batch=3) == 5, "one page, then the remainder"
    assert sorted(resumed) == sorted(active), "every page adopted, none twice"


def test_the_sysdb_page_stays_under_sqlites_parameter_limit() -> None:
    """Every grouped read binds one parameter per id of a page, plus a handful of its own."""
    assert sysdb.SYSDB_PAGE <= 990
