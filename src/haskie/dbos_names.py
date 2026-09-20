"""The DBOS vocabulary shared across the app: workflow statuses, the names DBOS stores rows
under, and the one helper that reads a DBOS error.

A leaf on purpose: `sysdb`, `jobs` and `models` need these names, and `workflows`, which defines
the workflows, imports all three. Every workflow is registered under its name here explicitly
(`@DBOS.workflow(name=...)`), never under its qualname, because DBOS looks a workflow up by that
name on recovery and `jobs` selects rows by these strings.
"""

from typing import Literal, get_args

from dbos import WorkflowStatusString
from dbos._error import DBOSMaxStepRetriesExceeded

# What DBOS records a workflow's progress as: ENQUEUED | PENDING | SUCCESS | ERROR | CANCELLED |
# MAX_RECOVERY_ATTEMPTS_EXCEEDED | DELAYED. A read model carries it through as the string it is.
WorkflowStatus = str

# Enqueued or running: the workflow is still on its way.
PENDING_STATUS = WorkflowStatusString.PENDING.value  # dequeued and running
ACTIVE_STATUS = [WorkflowStatusString.ENQUEUED.value, PENDING_STATUS]

# The three pipeline-shaped workflows: each cuts a stage into `stage_slice` children with one
# `try_batch` step per micro-batch, which is what the Jobs view reads as a job with tasks.
IMPORT_WORKFLOW = "import_document"  # convert, then pre-warm the embedding cache
EMBED_WORKFLOW = "ensure_embedding"  # one cached embedding of one document, deduplicated
COLLECTION_DOCUMENT_WORKFLOW = "index_collection_document"  # cached rows into one collection
PIPELINE_WORKFLOWS = [IMPORT_WORKFLOW, EMBED_WORKFLOW, COLLECTION_DOCUMENT_WORKFLOW]
STAGE_WORKFLOW = "stage_parts"  # workflows.stage_slice
STAGE_STEP = "try_batch"  # workflows.try_batch

# Whole-collection and whole-document jobs. Also the kind the API reports for them, so the three
# names are a type: `jobs.BulkKind` is this one.
BulkWorkflow = Literal["index_collection", "delete_collection", "delete_document"]
BULK_WORKFLOWS: tuple[BulkWorkflow, ...] = get_args(BulkWorkflow)
INDEX_COLLECTION_WORKFLOW, DELETE_COLLECTION_WORKFLOW, DELETE_DOCUMENT_WORKFLOW = BULK_WORKFLOWS

MAINTAIN_WORKFLOW = "maintain_collection"  # the debounced handle that only waits
MAINTAIN_PARTITION_WORKFLOW = "maintain_on_partition"  # the run itself, on the index partition
REMOVE_FROM_INDEX_WORKFLOW = "remove_from_collection_index"
DAILY_MAINTENANCE_WORKFLOW = "daily_maintenance"  # the nightly housekeeping round
DOWNLOAD_WORKFLOW = "ensure_model"  # models.ensure_model; one record per model


def root_cause(exc: BaseException) -> str:
    """Flat "Type: message" of the failure that actually matters: DBOS wraps exhausted step
    retries in DBOSMaxStepRetriesExceeded, whose own message names only the step."""
    if isinstance(exc, DBOSMaxStepRetriesExceeded) and exc.errors:
        exc = exc.errors[-1]
    return f"{type(exc).__name__}: {exc}"
