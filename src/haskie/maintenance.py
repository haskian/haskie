"""Library index maintenance: compaction, version cleanup and index (re)build, off the write path.

Indexing one document must stay O(document). Everything that is O(library) — compacting the
fragments each commit leaves behind, folding new rows into the full-text index, training the
approximate vector index — happens here instead, once per batch of documents rather than once per
document. `workflows.maintain_library` debounces the runs and puts each one on the library's index
partition, so maintenance never writes a table while a document's index stage does.

This module owns the maintenance columns of the `libraries` row: `pending_docs` (documents indexed
since the last finished run), `last_write_at`, `last_maintained_at` and `vector_index_rows` (rows
the vector index was last trained on).
"""

from datetime import timedelta

import msgspec

from haskie.index import IndexStats
from haskie.library import Library
from haskie.logs import get_logger
from haskie.settings import EmbeddingModel, PipelineSettings

_log = get_logger(__name__)

# Retrain the approximate vector index once a library holds this many times the rows it was
# trained on: IVF centroids stay usable while a table grows, but not once it has doubled.
RETRAIN_GROWTH = 2.0

# How long a pruned table version stays readable. A search that opened the table before a
# compaction keeps reading the version it opened, so cleanup must lag behind the longest read,
# not the last write.
KEEP_VERSIONS = timedelta(minutes=10)

SkipReason = str  # "no-library" | "no-table" | "outdated"


class Report(msgspec.Struct):
    """What one maintenance run did. Returned by the workflow, so it is the audit of the run."""

    library: str
    num_rows: int
    fragments_before: int
    fragments_after: int
    ann_trained: bool
    skipped: SkipReason | None = None


def ann_due(stats: IndexStats, settings: PipelineSettings, trained_rows: int) -> bool:
    """Whether the approximate vector index should be (re)trained.

    Below `ann_min_rows` an exact scan is both faster and exact, so no index is built at all.
    Above it, one is trained once and retrained after the library grew by `RETRAIN_GROWTH`;
    `trained_rows` is the row count of the last training, not the rows the index currently
    covers, because `optimize` folds later rows into the partitions it was trained with.
    """
    if stats.num_rows < settings.ann_min_rows:
        return False
    if not stats.has_vector_index:
        return True
    return stats.num_rows >= RETRAIN_GROWTH * trained_rows


async def run(lib: Library, embedding: EmbeddingModel | None, settings: PipelineSettings) -> Report:
    """One maintenance pass over one library's table. Runs on the library's index partition.

    Every skip is a value, not an exception: the library may have been deleted, may never have
    been indexed, or may hold a table an older build wrote, and none of those is a failure of the
    run. An outdated table is left untouched on purpose: "Index all" rewrites it (B2).

    Compaction and the index builds await LanceDB, which runs them on its own Rust runtime. The
    IVF-PQ training inside `build_vector_index` is CPU work, but it happens in that runtime rather
    than in a worker thread of ours, so it is the one piece of CPU work the budget in `cpu.py`
    cannot hold a slot for. `task.indexing` bounds it instead: one writer per library partition.
    """
    state_ = await lib.maintenance_state()
    if state_ is None:
        return _skipped(lib.name, "no-library")
    index = lib.index_with(embedding)
    before = await index.stats()
    if before is None:
        return _skipped(lib.name, "no-table")
    if not await index.schema_current():
        return _skipped(lib.name, "outdated")

    await index.finish()  # the full-text index of a library whose first document predates it
    await index.optimize(KEEP_VERSIONS)
    after = await index.stats() or before

    trained = False
    if embedding is not None and await index.has_vector_column():
        if ann_due(after, settings, state_.vector_index_rows):
            await index.build_vector_index(after.num_rows)
            trained = True
            after = await index.stats() or after

    report = Report(
        library=lib.name,
        num_rows=after.num_rows,
        fragments_before=before.num_fragments,
        fragments_after=after.num_fragments,
        ann_trained=trained,
    )
    _log.info(
        "library_maintained",
        library=lib.name,
        rows=report.num_rows,
        fragments_before=report.fragments_before,
        fragments_after=report.fragments_after,
        ann_trained=trained,
    )
    return report


def _skipped(library: str, reason: SkipReason) -> Report:
    _log.info("library_maintenance_skipped", library=library, reason=reason)
    return Report(
        library=library,
        num_rows=0,
        fragments_before=0,
        fragments_after=0,
        ann_trained=False,
        skipped=reason,
    )
