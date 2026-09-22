"""Operation routes: the history the Operations view pages, a job's tasks, and cancellation."""

from litestar import delete, get

from haskie import audit
from haskie.indexing import operations, workflows
from haskie.paging import DEFAULT_PAGE_SIZE, Page


@get("/api/operations")
async def list_operations(
    kind: str,
    collection: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> Page[operations.Operation]:
    """One page of the operations of one `kind`, newest first: document, collection, download or
    maintenance. `collection` keeps one collection's operations only, where the kind has a
    collection at all.

    Pass the `next_cursor` of a response back as `cursor` to continue; a cursor belongs to the
    kind that issued it.
    """
    return await operations.list_operations(kind, collection, page_size, cursor)


@get("/api/operations/kinds")
async def list_operation_kinds() -> list[operations.OperationKindSummary]:
    """Every kind of operation, in display order, with how many of each are running right now."""
    return await operations.list_kinds()


@get("/api/operations/activity")
async def get_activity() -> operations.Activity:
    """How many operations and tasks are queued and running right now, for the nav indicator."""
    return await operations.activity()


@get("/api/jobs/{job_id:str}/tasks")
async def list_job_tasks(job_id: str) -> list[operations.Task]:
    """The micro-batches of one job of an operation, in plan order."""
    return await operations.list_tasks(job_id)


@get("/api/operations/{operation_id:str}/progress")
async def get_operation_progress(operation_id: str) -> operations.OperationProgress:
    """How far a whole-collection index or delete, or a document delete, got; 404 for any other
    operation id."""
    return await operations.progress(operation_id)


@delete("/api/operations/{operation_id:str}")
@audit.audited("operation.cancel")
async def cancel_operation(operation_id: str) -> None:
    """Cancels the operation, its jobs and their tasks."""
    audit.attach(operation_id=operation_id)
    await workflows.cancel_operation(operation_id)
