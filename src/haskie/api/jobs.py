"""Job routes: the history the Jobs view pages, and cancellation."""

from litestar import delete, get

from haskie import audit, jobs
from haskie.paging import DEFAULT_PAGE_SIZE, Page


@get("/api/jobs")
async def list_jobs(
    library: str | None = None, page_size: int = DEFAULT_PAGE_SIZE, cursor: str | None = None
) -> Page[jobs.Job]:
    """Index jobs, newest first, one page at a time; `library` keeps one library's jobs only.

    Pass the `next_cursor` of a response back as `cursor` to continue; it is null on the last page.
    """
    return await jobs.list_jobs(library, page_size, cursor)


@get("/api/jobs/by-kind")
async def list_jobs_by_kind(
    kind: str,
    library: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> Page[jobs.JobRow]:
    """One page of the jobs of one `kind`, newest first: document, library, download, maintenance
    or archive. `library` keeps one library's jobs only, where the kind has a library at all.

    Pass the `next_cursor` of a response back as `cursor` to continue; a cursor belongs to the
    kind that issued it.
    """
    return await jobs.list_kind(kind, library, page_size, cursor)


@get("/api/jobs/kinds")
async def list_job_kinds() -> list[jobs.JobKindSummary]:
    """Every kind of job, in display order, with how many of each are running right now."""
    return await jobs.list_kinds()


@get("/api/jobs/activity")
async def get_activity() -> jobs.Activity:
    """How many jobs and tasks are queued and running right now, for the indicator in the nav."""
    return await jobs.activity()


@get("/api/jobs/{job_id:str}/tasks")
async def list_job_tasks(job_id: str) -> list[jobs.Task]:
    return await jobs.list_tasks(job_id)


@get("/api/jobs/{job_id:str}/progress")
async def get_job_progress(job_id: str) -> jobs.BulkJob:
    """How far a whole-library index or delete got; 404 for any other job id."""
    return await jobs.bulk_job(job_id)


@delete("/api/jobs/{job_id:str}")
@audit.audited("job.cancel")
async def delete_job(job_id: str) -> None:
    """Cancels the document workflow and its tasks."""
    audit.attach(workflow_id=job_id)
    await jobs.cancel_job(job_id)
