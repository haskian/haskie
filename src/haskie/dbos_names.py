"""The DBOS vocabulary shared across the app: workflow statuses, and the names DBOS stores rows
under.

A leaf on purpose. `workflows` defines the workflows, so every module that reads DBOS's tables
(`archive`, `sysdb`, `jobs`) or its statuses (`models`) would have to import `workflows` to name
them — and `workflows` imports all of those. The names lived as bare strings in each module
instead, with a comment in every one explaining the duplication. They live here once; `test_sysdb`
and `test_archive` pin them against the real registrations.

Every workflow named here is registered under that name explicitly (`@DBOS.workflow(name=...)`),
never under its bare qualname: DBOS records a workflow under its name and looks it up by that
name on recovery, so a function renamed without this would orphan every workflow already recorded
— and `archive` selects rows by these strings, where a mismatch is a silent miss, not an error.
"""

from dbos import WorkflowStatusString

# Enqueued or running: the workflow is still on its way.
PENDING_STATUS = WorkflowStatusString.PENDING.value  # dequeued and running
ACTIVE_STATUS = [WorkflowStatusString.ENQUEUED.value, PENDING_STATUS]

# A debounced workflow waits out its period as DELAYED: still on its way, but not yet enqueued.
WAITING_STATUS = [WorkflowStatusString.DELAYED.value, *ACTIVE_STATUS]

# Finished, one way or another: nothing about the workflow can change any more.
TERMINAL_STATUS = frozenset(
    {
        WorkflowStatusString.SUCCESS.value,
        WorkflowStatusString.ERROR.value,
        WorkflowStatusString.CANCELLED.value,
        WorkflowStatusString.MAX_RECOVERY_ATTEMPTS_EXCEEDED.value,
    }
)

# The three pipeline-shaped workflows: each cuts a stage into `stage_slice` children with one
# `try_batch` step per micro-batch, which is what the Jobs view reads as a job with tasks.
IMPORT_WORKFLOW = "import_document"  # convert, then pre-warm the embedding cache
EMBED_WORKFLOW = "ensure_embedding"  # one cached embedding of one document, deduplicated
COLLECTION_DOCUMENT_WORKFLOW = "index_collection_document"  # cached rows into one collection
PIPELINE_WORKFLOWS = [IMPORT_WORKFLOW, EMBED_WORKFLOW, COLLECTION_DOCUMENT_WORKFLOW]
STAGE_WORKFLOW = "stage_parts"  # workflows.stage_slice
STAGE_STEP = "try_batch"  # workflows.try_batch.__qualname__
