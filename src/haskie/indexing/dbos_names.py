"""The DBOS vocabulary shared across the app: run statuses, the names DBOS stores rows
under, and the one helper that reads a DBOS error.

A leaf on purpose: `sysdb`, `operations` and `models` need these names, and `workflows`, which
defines the workflows, imports all three. Every workflow is registered under its name here
explicitly (`@DBOS.workflow(name=...)`), never under its qualname, because DBOS looks a workflow up
by that name on recovery and `operations` selects rows by these strings.
"""

from enum import StrEnum

from dbos._error import DBOSMaxStepRetriesExceeded


# What DBOS records a run's progress as. Operations and jobs are runs of a DBOS workflow; a task is
# one step, and the read model reports it in the same words. Spelled out rather than carried
# through as a bare string, so a read model states the vocabulary it can hold, and the generated
# OpenAPI document gives the web client the same closed set (`test_workflows` pins it against DBOS).
class RunStatus(StrEnum):
    ENQUEUED = "ENQUEUED"
    PENDING = "PENDING"  # dequeued and running
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"
    MAX_RECOVERY_ATTEMPTS_EXCEEDED = "MAX_RECOVERY_ATTEMPTS_EXCEEDED"
    DELAYED = "DELAYED"


RUN_STATUSES: tuple[RunStatus, ...] = tuple(RunStatus)

# Enqueued or running: the workflow is still on its way. `list[str]`, not `list[RunStatus]`: DBOS
# takes it as a query argument, and a list is invariant in its item type.
ACTIVE_STATUS: list[str] = [RunStatus.ENQUEUED, RunStatus.PENDING]

# The three pipeline-shaped workflows: each cuts a stage into `stage_slice` children with one
# `try_batch` step per micro-batch, which is what the Operations view reads as a job with tasks.
IMPORT_WORKFLOW = "import_document"  # convert, then pre-warm the embedding cache
EMBED_WORKFLOW = "ensure_embedding"  # one cached embedding of one document, deduplicated
COLLECTION_DOCUMENT_WORKFLOW = "index_collection_document"  # cached rows into one collection
PIPELINE_WORKFLOWS = [IMPORT_WORKFLOW, EMBED_WORKFLOW, COLLECTION_DOCUMENT_WORKFLOW]
# The two that are document operations of their own; an embed run is a job of the one that asked.
DOCUMENT_OPERATION_WORKFLOWS = [IMPORT_WORKFLOW, COLLECTION_DOCUMENT_WORKFLOW]
STAGE_WORKFLOW = "stage_slice"  # workflows.stage_slice
STAGE_STEP = "try_batch"  # workflows.try_batch
# The steps of `summarize_collection_workflow` its tasks are read from: the members it describes,
# then one step a member, then the collection
MEMBERS_STEP = "undescribed_members"
MEMBER_STEP = "try_describe_member"
COLLECTION_STEP = "try_summarize_collection"


# Whole-thing operations a request only starts: whole-collection and whole-document work, and the
# backup and restore of everything. Also the kind the API reports for them, so the names are a
# type: `operations.BulkKind` is this one.
class BulkWorkflow(StrEnum):
    INDEX_COLLECTION = "index_collection"
    DELETE_COLLECTION = "delete_collection"
    DELETE_DOCUMENT = "delete_document"
    SUMMARIZE_DOCUMENT = "summarize_document"  # a document's description the describer writes
    SUMMARIZE_COLLECTION = "summarize_collection"  # a collection's, from its documents'
    CREATE_BACKUP = "create_backup"
    RESTORE_BACKUP = "restore_backup"


BULK_WORKFLOWS: tuple[BulkWorkflow, ...] = tuple(BulkWorkflow)
INDEX_COLLECTION_WORKFLOW = BulkWorkflow.INDEX_COLLECTION
DELETE_COLLECTION_WORKFLOW = BulkWorkflow.DELETE_COLLECTION
DELETE_DOCUMENT_WORKFLOW = BulkWorkflow.DELETE_DOCUMENT
SUMMARIZE_DOCUMENT_WORKFLOW = BulkWorkflow.SUMMARIZE_DOCUMENT
SUMMARIZE_COLLECTION_WORKFLOW = BulkWorkflow.SUMMARIZE_COLLECTION
CREATE_BACKUP_WORKFLOW = BulkWorkflow.CREATE_BACKUP
RESTORE_BACKUP_WORKFLOW = BulkWorkflow.RESTORE_BACKUP

MAINTAIN_WORKFLOW = "maintain_collection"  # the debounced handle that only waits
MAINTAIN_PARTITION_WORKFLOW = "maintain_on_partition"  # the run itself, on the index partition
VOCABULARY_WORKFLOW = "build_vocabulary"  # a collection's preferred terms, debounced
REMOVE_FROM_INDEX_WORKFLOW = "remove_from_collection_index"
DAILY_MAINTENANCE_WORKFLOW = "daily_maintenance"  # the nightly housekeeping round
DOWNLOAD_WORKFLOW = "ensure_model"  # models.ensure_model; one record per model


def root_cause(exc: BaseException) -> str:
    """Flat "Type: message" of the failure that matters: DBOS wraps exhausted step
    retries in DBOSMaxStepRetriesExceeded, whose own message names only the step."""
    if isinstance(exc, DBOSMaxStepRetriesExceeded) and exc.errors:
        exc = exc.errors[-1]
    return f"{type(exc).__name__}: {exc}"
