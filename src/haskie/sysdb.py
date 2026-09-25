"""Read-only raw SQL over the DBOS system tables, which live in our own SQLite file.

DBOS's Python API answers one workflow (or one page of workflows) at a time. A read model over a
whole page of operations needs aggregates: how many steps each stage slice recorded, how many
workflows of each name or queue are active. One grouped query answers that for every row on the
page, where the API would need a call per workflow.

Nothing here writes: DBOS owns every row in these tables. `db.connect()` opens the same file DBOS
was configured with, so the reads see its committed state through WAL.
"""

from collections.abc import Sequence
from itertools import batched

from haskie import db
from haskie.indexing.dbos_names import ACTIVE_STATUS

# SQLite allows 999 bound parameters by default; one query per page keeps every list under it.
SYSDB_PAGE = 500


async def step_counts(workflow_ids: list[str], function_name: str) -> dict[str, int]:
    """How many steps named `function_name` each workflow has recorded.

    DBOS writes the row when a step finishes, whether it returned a value or raised, so a recorded
    step is a finished step. Workflows without one are absent from the result."""
    if not workflow_ids:
        return {}
    counts: dict[str, int] = {}
    async with db.connect() as conn:
        for chunk in batched(workflow_ids, SYSDB_PAGE, strict=False):
            rows = await conn.execute_fetchall(
                "select workflow_uuid, count(*) from operation_outputs "
                f"where workflow_uuid in ({db.placeholders(len(chunk))}) and function_name = ? "
                "group by workflow_uuid",
                (*chunk, function_name),
            )
            counts.update({workflow_id: count for workflow_id, count in rows})
    return counts


async def active_counts_by_name() -> dict[str, int]:
    """How many workflows of each name are enqueued or running right now.

    One query for the whole app: the Operations view shows an active count per kind, and a kind is
    a set of workflow names, so counting through the API would cost a listing per name."""
    async with db.connect() as conn:
        rows = await conn.execute_fetchall(
            "select name, count(*) from workflow_status "
            f"where status in ({db.placeholders(len(ACTIVE_STATUS))}) and name is not null "
            "group by name",
            ACTIVE_STATUS,
        )
    return {name: count for name, count in rows}


async def operation_activity(skip: Sequence[str] = ()) -> dict[str, int]:
    """Enqueued/running workflows on the `operation.*` queues, by status. The tasks under them are
    counted from the slice listing (`operations.batch_activity`), batch by batch, so they are not
    counted here. A workflow started outside a queue has no queue name and is not counted. `skip`
    names queues left out: a child that its parent waits for is the same work as the parent, not
    a second one.

    `ACTIVE_STATUS` only: a DELAYED workflow is a debounce waiting out its period, not work
    waiting for a slot. Counting it made the indicator read "1 queued" for a whole
    `maintenance_idle_seconds` after the last document, with nothing queued and the Operations
    view - which counts the same `ACTIVE_STATUS` - showing nothing."""
    async with db.connect() as conn:
        rows = await conn.execute_fetchall(
            "select status, count(*) from workflow_status "
            f"where status in ({db.placeholders(len(ACTIVE_STATUS))}) "
            "and queue_name like 'operation.%' "
            + (f"and queue_name not in ({db.placeholders(len(skip))}) " if skip else "")
            + "group by status",
            [*ACTIVE_STATUS, *skip],
        )
    return {status: count for status, count in rows}


async def stale_active_ids(app_version: str, limit: int, offset: int = 0) -> list[str]:
    """One page of ids of workflows that are still enqueued or running under another application
    version, oldest first. Ids only: a boot after a long outage must not load the whole backlog."""
    async with db.connect() as conn:
        rows = await conn.execute_fetchall(
            "select workflow_uuid from workflow_status "
            f"where status in ({db.placeholders(len(ACTIVE_STATUS))}) "
            "and application_version != ? "
            "order by created_at limit ? offset ?",
            (*ACTIVE_STATUS, app_version, limit, offset),
        )
    return [workflow_id for (workflow_id,) in rows]
