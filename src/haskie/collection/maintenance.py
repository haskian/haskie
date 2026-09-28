"""Collection index maintenance: compaction, version cleanup and index (re)build, off the write
path.

Indexing one document must stay O(document). Everything that is O(collection) — compacting the
fragments each commit leaves behind, folding new rows into the full-text index, training the
approximate vector index — happens here instead, once per burst of documents rather than once per
document. `workflows.maintain_collection` debounces the runs and puts each one on the collection's
index partition, so maintenance never writes a table while a document's index stage does.

A run's state lives in the maintenance columns of the `collections` row: `pending_documents`
(documents indexed since the last finished run), `last_write_at`, `last_maintained_at` and
`vector_index_rows` (rows the vector index was last trained on). This module reads the last one;
`Collection` reads and writes them all.
"""

from datetime import timedelta
from enum import StrEnum

import msgspec

from haskie.catalogue.catalogue import EmbeddingModel
from haskie.collection.collection import Collection
from haskie.collection.index import PQ_MIN_ROWS, IndexStats
from haskie.logs import get_logger
from haskie.settings import PipelineSettings

_log = get_logger(__name__)

# Retrain the approximate vector index once a collection holds this many times the rows it was
# trained on: IVF centroids stay usable while a table grows, but not once it has doubled.
RETRAIN_GROWTH = 2.0

# How long a pruned table version stays readable. A search that opened the table before a
# compaction keeps reading the version it opened, so cleanup must lag behind the longest read,
# not the last write.
KEEP_VERSIONS = timedelta(minutes=10)


class SkipReason(StrEnum):
    NO_COLLECTION = "no-collection"
    NO_TABLE = "no-table"
    OUTDATED = "outdated"


class Report(msgspec.Struct):
    """What one maintenance run did. `workflows.maintain_on_partition` returns it, so DBOS keeps
    it as the record of the run."""

    collection: str
    num_rows: int = 0
    fragments_before: int = 0
    fragments_after: int = 0
    ann_trained: bool = False
    skipped: SkipReason | None = None


def ann_due(stats: IndexStats, settings: PipelineSettings, trained_rows: int) -> bool:
    """Whether the approximate vector index should be (re)trained.

    Below `ann_min_rows` an exact scan is both faster and exact, so no index is built at all. Nor
    is one below `PQ_MIN_ROWS`, whatever the setting says: LanceDB refuses to train on fewer.
    Above both, one is trained once and retrained after the collection grew by `RETRAIN_GROWTH`;
    `trained_rows` is the row count of the last training, not the rows the index currently
    covers, because `optimize` folds later rows into the partitions it was trained with.
    """
    if stats.num_rows < max(settings.ann_min_rows, PQ_MIN_ROWS):
        return False
    if not stats.has_vector_index:
        return True
    return stats.num_rows >= RETRAIN_GROWTH * trained_rows


async def run(
    collection: Collection, embedding: EmbeddingModel | None, settings: PipelineSettings
) -> Report:
    """One maintenance pass over one collection's table.

    Every skip is a value, not an exception: the collection may have been deleted, may never have
    been indexed, or may hold a table an older build wrote, and none of those is a failure of the
    run. An outdated table is left untouched on purpose: "Index all" rewrites it.

    Compaction and the index builds await LanceDB, which runs them on its own Rust runtime. The
    IVF-PQ training inside `build_vector_index` is CPU work, but it happens in that runtime rather
    than in a worker thread of ours, so it is the one piece of CPU work the budget in `cpu.py`
    cannot hold a slot for. `task.indexing` bounds it instead: one writer per collection partition.
    """
    state_ = await collection.maintenance_state()
    if state_ is None:
        return _skipped(collection.name, SkipReason.NO_COLLECTION)
    index = collection.index_with(embedding)
    before = await index.stats()
    if before is None:
        return _skipped(collection.name, SkipReason.NO_TABLE)
    if not await index.schema_current():
        return _skipped(collection.name, SkipReason.OUTDATED)

    await index.finish()  # builds the full-text index when an index run left the table without one
    await index.optimize(KEEP_VERSIONS)
    after = await index.stats() or before

    # with an embedding, the current schema above already holds its vector column
    trained = embedding is not None and ann_due(after, settings, state_.vector_index_rows)
    if trained:
        await index.build_vector_index(after.num_rows)
        after = await index.stats() or after

    report = Report(
        collection=collection.name,
        num_rows=after.num_rows,
        fragments_before=before.num_fragments,
        fragments_after=after.num_fragments,
        ann_trained=trained,
    )
    _log.info("collection_maintained", **msgspec.structs.asdict(report))
    return report


def _skipped(collection: str, reason: SkipReason) -> Report:
    _log.info("collection_maintenance_skipped", collection=collection, reason=reason)
    return Report(collection=collection, skipped=reason)
