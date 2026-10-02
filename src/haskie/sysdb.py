"""Read-only raw SQL over the DBOS system tables, which live in our own SQLite file.

DBOS's Python API answers one workflow (or one page of workflows) at a time. A read model over a
whole page of operations needs aggregates: how many steps each stage slice recorded, how many
workflows of each name or queue are active. One grouped query answers that for every row on the
page, where the API would need a call per workflow.

DBOS owns every row in these tables, and their schema too, so they are declared here as lightweight
`table()` clauses with the columns read, outside `tables.metadata`, which creates only haskie's own.
`db.read()` opens the same file DBOS was configured with, so the reads see its committed state
through WAL, and wait for none of its writers. One function writes, `move_to_version`, because
DBOS offers no call that does it.
"""

from collections.abc import Sequence
from itertools import batched

from sqlalchemy import column, func, select, table, update

from haskie import db
from haskie.indexing.dbos_names import ACTIVE_STATUS

workflow_status = table(
    "workflow_status",
    column("workflow_uuid"),
    column("name"),
    column("status"),
    column("queue_name"),
    column("application_version"),
    column("created_at"),
)
operation_outputs = table("operation_outputs", column("workflow_uuid"), column("function_name"))

# SQLite builds before 3.32 allow only 999 bound parameters, and a custom build may set the limit
# as low; one query per page keeps every list under that floor on any build.
SYSDB_PAGE = 500


async def step_counts(workflow_ids: list[str], function_name: str) -> dict[str, int]:
    """How many steps named `function_name` each workflow has recorded.

    DBOS writes the row when a step finishes, whether it returned a value or raised, so a recorded
    step is a finished step. Workflows without one are absent from the result."""
    if not workflow_ids:
        return {}
    counts: dict[str, int] = {}
    async with db.read() as conn:
        step = operation_outputs.c
        for chunk in batched(workflow_ids, SYSDB_PAGE, strict=False):
            rows = await conn.execute(
                select(step.workflow_uuid, func.count())
                .where(step.workflow_uuid.in_(chunk), step.function_name == function_name)
                .group_by(step.workflow_uuid)
            )
            counts.update(rows.tuples().all())
    return counts


async def active_counts_by_name() -> dict[str, int]:
    """How many workflows of each name are enqueued or running right now.

    One query for the whole app: the Operations view shows an active count per kind, and a kind is
    a set of workflow names, so counting through the API would cost a listing per name."""
    async with db.read() as conn:
        workflow = workflow_status.c
        rows = await conn.execute(
            select(workflow.name, func.count())
            .where(workflow.status.in_(ACTIVE_STATUS), workflow.name.is_not(None))
            .group_by(workflow.name)
        )
        return dict(rows.tuples().all())


async def operation_activity(skip: Sequence[str] = ()) -> dict[str, int]:
    """Enqueued/running workflows on the `operation.*` queues, by status. The tasks under them are
    counted from the slice listing (`operations.batch_activity`), batch by batch, so they are not
    counted here. A workflow started outside a queue has no queue name and is not counted. `skip`
    names queues left out: a child that its parent waits for is the same work as the parent, not
    a second one.

    `ACTIVE_STATUS` only: a DELAYED workflow is a debounce waiting out its period, not work
    waiting for a slot. Counting it made the indicator read "1 queued" for a whole
    `maintenance_idle_seconds` after the last document, with nothing queued and the Operations
    view (which counts the same `ACTIVE_STATUS`) showing nothing."""
    async with db.read() as conn:
        workflow = workflow_status.c
        rows = await conn.execute(
            select(workflow.status, func.count())
            .where(
                workflow.status.in_(ACTIVE_STATUS),
                workflow.queue_name.like("operation.%"),
                workflow.queue_name.not_in(skip),
            )
            .group_by(workflow.status)
        )
        return dict(rows.tuples().all())


async def stale_active(app_version: str, limit: int) -> list[tuple[str, str | None]]:
    """One page of workflows still enqueued or running under another application version, oldest
    first, as (id, queue name). A page, never the whole backlog: a boot after a long outage must
    not load it. No offset: `move_to_version` takes each page out of the result."""
    async with db.read() as conn:
        workflow = workflow_status.c
        rows = await conn.execute(
            select(workflow.workflow_uuid, workflow.queue_name)
            .where(workflow.status.in_(ACTIVE_STATUS), workflow.application_version != app_version)
            .order_by(workflow.created_at)
            .limit(limit)
        )
        return list(rows.tuples())


async def move_to_version(workflow_ids: list[str], app_version: str) -> None:
    """Hand workflows of an older build to this one. DBOS dequeues only rows of its own
    application version, and a resume keeps the old one, so without this an adopted workflow
    waits in its queue forever."""
    async with db.connect() as conn:
        workflow = workflow_status.c
        for chunk in batched(workflow_ids, SYSDB_PAGE, strict=False):
            await conn.execute(
                update(workflow_status)
                .where(workflow.workflow_uuid.in_(chunk))
                .values(application_version=app_version)
            )
