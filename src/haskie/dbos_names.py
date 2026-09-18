"""The DBOS vocabulary shared across the app: workflow statuses, and the names DBOS stores rows
under.

A leaf on purpose. `workflows` defines the workflows, so every module that reads DBOS's tables
(`archive`, `sysdb`, `jobs`) or its statuses (`models`) would have to import `workflows` to name
them — and `workflows` imports all of those. The names lived as bare strings in each module
instead, with a comment in every one explaining the duplication. They live here once; `test_sysdb`
and `test_archive` pin them against the real qualnames.
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

# DBOS stores a workflow under its qualname, and a step under its own.
DOCUMENT_WORKFLOW = "index_document"  # workflows.index_document.__qualname__
# Durable, not derived: `workflows.stage_slice` is registered under this name explicitly, so the
# function can be renamed without orphaning the workflows DBOS already recorded.
STAGE_WORKFLOW = "stage_parts"
STAGE_STEP = "try_batch"  # workflows.try_batch.__qualname__
