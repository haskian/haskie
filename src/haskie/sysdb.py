"""Read-only raw SQL over the DBOS system tables, which live in our own SQLite file.

DBOS's Python API answers one workflow (or one page of workflows) at a time. A read model over a
whole page of jobs needs aggregates: how many steps each stage slice recorded, how many children
each job has in which status. One grouped query answers that for every job on the page, where the
API would need a call per workflow.

Nothing here writes: DBOS owns every row in these tables. `db.connect()` opens the same file DBOS
was configured with, so the reads see its committed state through WAL.
"""

from itertools import batched

from haskie import db
from haskie.dbos_names import ACTIVE_STATUS

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


async def child_status_counts(parent_ids: list[str]) -> dict[str, dict[str, int]]:
    """Per parent workflow, how many children sit in each DBOS status."""
    if not parent_ids:
        return {}
    counts: dict[str, dict[str, int]] = {}
    async with db.connect() as conn:
        for chunk in batched(parent_ids, SYSDB_PAGE, strict=False):
            rows = await conn.execute_fetchall(
                "select parent_workflow_id, status, count(*) from workflow_status "
                f"where parent_workflow_id in ({db.placeholders(len(chunk))}) "
                "group by parent_workflow_id, status",
                chunk,
            )
            for parent_id, status, count in rows:
                counts.setdefault(parent_id, {})[status] = count
    return counts


async def active_counts_by_name(active: list[str] = ACTIVE_STATUS) -> dict[str, int]:
    """How many workflows of each name are enqueued or running right now.

    One query for the whole app: the jobs view shows an active count per kind, and a kind is a set
    of workflow names, so counting through the API would cost a listing per name."""
    async with db.connect() as conn:
        rows = await conn.execute_fetchall(
            "select name, count(*) from workflow_status "
            f"where status in ({db.placeholders(len(active))}) and name is not null "
            "group by name",
            active,
        )
    return {name: count for name, count in rows}


async def queue_activity() -> dict[str, dict[str, int]]:
    """Enqueued/running workflows per queue family ("job" or "task"), by status.

    The queue name carries the family as its prefix (`job.indexing`, `task.embedding`), so one
    grouped query over the prefix answers the whole indicator; a workflow started outside a queue
    has no name and is not counted.

    `ACTIVE_STATUS`, not `WAITING_STATUS`: a DELAYED workflow is a debounce waiting out its period,
    not work waiting for a slot. Counting it made the indicator read "1 queued" for a whole
    `maintenance_idle_seconds` after the last document, with nothing queued and the Jobs view -
    which counts the same `ACTIVE_STATUS` - showing nothing."""
    async with db.connect() as conn:
        rows = await conn.execute_fetchall(
            "select substr(queue_name, 1, instr(queue_name, '.') - 1), status, count(*) "
            "from workflow_status "
            f"where status in ({db.placeholders(len(ACTIVE_STATUS))}) "
            "and queue_name like '%.%' group by 1, 2",
            ACTIVE_STATUS,
        )
    activity: dict[str, dict[str, int]] = {}
    for family, status, count in rows:
        activity.setdefault(family, {})[status] = count
    return activity


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
