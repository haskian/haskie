"""User settings (global, chosen at init) and per-collection overrides.

Every setting carries a `Meta(title, description)`: the single definition shown in the UI
(`/api/options` -> `docs`) and in the JSON schema. Collection overrides reuse the same Meta
objects.

What is settable where follows where the work happens. Conversion (`parser`, `skip_ocr_pages`)
runs once per document, at import, so its values are chosen then and stored on the document;
`UserSettings.conversion` only supplies the defaults an import falls back on. Chunking
(`chunker`, `chunk_size`, `chunk_overlap`) splits the shared markdown per collection, so a
collection may override it (`CollectionSettings`), and the embedding cache is keyed by it.
"""

import os
import threading
from typing import Annotated, Any, Literal

import msgspec
from msgspec import Meta

from haskie import db
from haskie.errors import InvalidInput
from haskie.logs import get_logger

_log = get_logger(__name__)

Parser = Literal["anydoc", "plain"]  # PDFs always go page-wise through pdf-inspector
Chunker = Literal["markdown", "text"]  # semantic-text-splitter MarkdownSplitter / TextSplitter
EmbeddingProfile = Literal["none", "compact", "quality", "multilingual"]
Accelerator = Literal["auto", "cpu"]  # auto = best ONNX Runtime provider (CUDA, CoreML, ...)
SearchMode = Literal["hybrid", "vector", "fts"]
Fusion = Literal["rrf", "linear"]
Reranker = Literal["none", "cross-encoder"]

PARSERS: tuple[str, ...] = Parser.__args__
CHUNKERS: tuple[str, ...] = Chunker.__args__
ACCELERATORS: tuple[str, ...] = Accelerator.__args__
SEARCH_MODES: tuple[str, ...] = SearchMode.__args__
FUSIONS: tuple[str, ...] = Fusion.__args__
RERANKERS: tuple[str, ...] = Reranker.__args__

# fastembed cross-encoder ids (ONNX; downloaded on first use)
RERANKER_MODELS: tuple[str, ...] = (
    "Xenova/ms-marco-MiniLM-L-6-v2",
    "Xenova/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-base",
    "jinaai/jina-reranker-v1-turbo-en",
    "jinaai/jina-reranker-v2-base-multilingual",
)


class EmbeddingModel(msgspec.Struct):
    name: str
    dims: int
    accelerator: Accelerator = "auto"


# fastembed model ids. "none" = full-text search only.
PROFILES: dict[str, EmbeddingModel | None] = {
    "none": None,
    "compact": EmbeddingModel("BAAI/bge-small-en-v1.5", 384),
    "quality": EmbeddingModel("BAAI/bge-large-en-v1.5", 1024),
    "multilingual": EmbeddingModel("intfloat/multilingual-e5-large", 1024),
}


# --- definitions ------------------------------------------------------------------

EMBEDDING = Meta(
    title="Embedding profile",
    description=(
        "Text-embedding model that turns chunks into vectors for semantic search. Chosen at "
        'first run; changing it later requires "Index all" in every collection. none = full-text '
        "(BM25) search only; compact = bge-small (384 dims, English); quality = bge-large "
        "(1024 dims, English); multilingual = multilingual-e5-large (1024 dims)."
    ),
)
PARSER = Meta(
    title="Parser",
    description=(
        "Converter for non-PDF files, chosen when a document is imported. anydoc: Word, "
        "PowerPoint, Excel, OpenDocument, RTF, EPUB and CSV to Markdown. plain: read the file as "
        "UTF-8 text. PDFs always use pdf-inspector page by page, regardless of this setting."
    ),
)
CHUNKER = Meta(
    title="Chunker",
    description=(
        "How converted Markdown is split into indexed chunks. markdown: split on Markdown "
        "structure (headings, then paragraphs, sentences, words), filling each chunk up to Chunk "
        "size. text: ignore Markdown structure and split on paragraphs, sentences, words."
    ),
)
CHUNK_SIZE = Meta(
    title="Chunk size (characters)",
    description=(
        "Maximum length of one chunk, counted in Unicode characters, not words or tokens "
        "(about 4 characters per English token). A chunk is also the unit that gets one "
        "embedding vector and one search result."
    ),
)
CHUNK_OVERLAP = Meta(
    title="Chunk overlap (characters)",
    description=(
        "Characters repeated at the start of a chunk from the end of the previous one, so a "
        "sentence cut by a boundary stays searchable. Must be smaller than Chunk size."
    ),
)
SKIP_OCR_PAGES = Meta(
    title="Skip pages that need OCR",
    description=(
        "Chosen when a document is imported. PDF pages with no extractable text (scans, images) "
        "are dropped and replaced by a marker comment in the Markdown instead of failing the "
        "document. A document where every page needs OCR still fails. Off: any such page fails "
        "the document."
    ),
)
CPU_BUDGET = Meta(
    title="CPU budget",
    description=(
        "Maximum number of tasks haskie runs at the same time across every queue: converting, "
        "embedding, indexing and maintenance. Defaults to half the machine's cores so other work "
        "keeps the rest. Never exceeded, whatever the weights below say."
    ),
)


def _weight(stage: str, note: str) -> Meta:
    """One definition for the three stage weights: they differ only in the stage they name."""
    return Meta(
        title=f"{stage} weight",
        description=(
            "Share of the CPU budget this stage may use when every stage has work. Each stage "
            f"always gets at least one slot; the budget still caps the total. {note}"
        ),
    )


CONVERTING_WEIGHT = _weight("Converting", "Converting costs CPU and IO per page.")
EMBEDDING_WEIGHT = _weight("Embedding", "Every embedding task loads the embedding model.")
INDEXING_WEIGHT = _weight("Indexing", "LanceDB takes one writer per collection.")
DOCUMENT_PARALLELISM = Meta(
    title="Parallel tasks per document",
    description=(
        "Upper bound on micro-batches of one document converted or embedded at the same time. "
        "0 = as many as the stage's own share of the CPU budget. Lower it to keep a single large "
        "document from occupying every slot while other documents wait."
    ),
)
BATCH_PAGES = Meta(
    title="Pages per micro-batch",
    description=(
        "Number of PDF pages one task converts, or one task chunks and embeds. Bounds memory: at "
        "most the CPU budget x Pages per micro-batch pages are in flight. Non-PDF files are "
        "one batch."
    ),
)
INDEX_GROUP_PARTS = Meta(
    title="Parts per index write",
    description=(
        "Micro-batches written to LanceDB in one commit. Fewer commits mean fewer fragments and "
        "faster maintenance; a document longer than this many parts is written in several "
        "commits, each resumable."
    ),
)
MAINTENANCE_DOCS = Meta(
    title="Maintenance after documents",
    description=(
        "Run collection maintenance (compaction, index update) once this many documents were "
        "indexed since the last run."
    ),
)
MAINTENANCE_IDLE = Meta(
    title="Maintenance when idle (seconds)",
    description=(
        "Also run maintenance once a collection has had no document indexed for this long."
    ),
)
ANN_MIN_ROWS = Meta(
    title="Vector index from (rows)",
    description=(
        "Build the approximate vector index (IVF-PQ) once a collection holds at least this many "
        "chunks; smaller collections are scanned exactly. Rebuilt when the collection doubled."
    ),
)
TASK_TIMEOUT = Meta(
    title="Task timeout (seconds)",
    description=(
        "Time budget per task (micro-batch): a stage gets this much for each batch it has, then "
        "it is cancelled. A conversion that hangs fails its document instead of holding a worker."
    ),
)
ACCELERATOR = Meta(
    title="Embedding hardware",
    description=(
        "Device for the embedding and reranker models. auto: best available ONNX Runtime "
        "provider (CUDA, CoreML on Apple Silicon, else CPU). cpu: force CPU."
    ),
)
LIMIT = Meta(title="Results", description="Number of results a search returns.")
CANDIDATES = Meta(
    title="Candidates",
    description=(
        "Results fetched before fusion and reranking: per retriever in hybrid mode, in total "
        "otherwise. Then cut down to Results. Higher = better recall, slower. Ignored when "
        "neither fusion nor a reranker applies."
    ),
)
MODE = Meta(
    title="Search mode",
    description=(
        "hybrid: vector similarity and BM25 full-text, merged by Fusion. vector: semantic "
        "similarity only. fts: BM25 full-text only. Without an embedding profile every mode "
        "behaves as fts."
    ),
)
FUSION = Meta(
    title="Fusion",
    description=(
        "Hybrid mode only: how the vector and BM25 rankings are merged. rrf: reciprocal rank "
        "fusion (rank based, robust, uses RRF k). linear: weighted sum of normalized scores "
        "using Vector weight and BM25 weight."
    ),
)
RRF_K = Meta(
    title="RRF k",
    description=(
        "Reciprocal rank fusion constant: score = sum of 1 / (k + rank). Higher k flattens the "
        "difference between top ranks. 60 is the usual value."
    ),
)
VECTOR_WEIGHT = Meta(
    title="Vector weight",
    description=(
        "Linear fusion only: weight of the semantic (vector) score. Normalized together with "
        "BM25 weight, so 1 / 1 means 50 / 50."
    ),
)
BM25_WEIGHT = Meta(
    title="BM25 weight",
    description=(
        "Linear fusion only: weight of the lexical (BM25) score. Normalized together with "
        "Vector weight."
    ),
)
NPROBES = Meta(
    title="Vector probes",
    description=(
        "Partitions of the approximate vector index visited per query. Higher = better recall, "
        "slower. Ignored without a vector index."
    ),
)
REFINE_FACTOR = Meta(
    title="Refine factor",
    description=(
        "Approximate candidates re-scored exactly per result (k x factor). Higher = more "
        "precise, slower. Ignored without a vector index."
    ),
)
RERANKER = Meta(
    title="Reranker",
    description=(
        "Second-stage scoring applied to the Candidates of any mode (vector, fts or hybrid). "
        "cross-encoder: a model reads query and chunk together and rescores each pair; slower "
        "but more precise than embeddings. none: keep the retrieval order."
    ),
)
RERANKER_MODEL = Meta(
    title="Reranker model",
    description=(
        "Cross-encoder used when Reranker is cross-encoder. ms-marco-MiniLM-L-6 is fast and "
        "English; bge-reranker-base is stronger; jina-reranker-v2 is multilingual. Downloaded "
        "on first use."
    ),
)
PREVIEW_WORKERS = Meta(
    title="Preview builds",
    description=(
        "Maximum number of document previews built at the same time when they are first opened; "
        "further requests wait, so a burst of opens does not start dozens of PDF parses. Outside "
        "the CPU budget: a preview is built for a reader who is waiting for it."
    ),
)
AUDIT_RETENTION = Meta(
    title="Audit retention (days)",
    description=(
        "Days of audit files kept under HASKIE_HOME/audit; older daily files are deleted by the "
        "daily maintenance run. 0 keeps everything."
    ),
)
RETENTION_DAYS = Meta(
    title="Job history (days)",
    description=(
        "How many days of finished indexing jobs and their micro-batch results stay visible "
        "under Jobs. Each UTC day is one table; older days are dropped whole."
    ),
)
RETENTION_LIVE_HOURS = Meta(
    title="Live job window (hours)",
    description=(
        "How long finished jobs stay in the durable-execution tables. Halfway through the window "
        "they are copied to the archive; at the end they are purged. Smaller keeps job queries "
        "fast; must cover the longest job."
    ),
)


def without_none(struct: msgspec.Struct) -> dict[str, Any]:
    """Set fields only. None means "inherit", so it must never override its fallback value."""
    return {k: v for k, v in msgspec.structs.asdict(struct).items() if v is not None}


# --- validation -------------------------------------------------------------------


def _at_least(minimum: int | float, **values: int | float) -> None:
    for name, value in values.items():
        if value < minimum:
            raise InvalidInput(f"{name} must be >= {minimum}, got {value}")


def _check_chunking(chunk_size: int | None, chunk_overlap: int | None) -> None:
    """Shared by the user-level settings and the per-collection overrides, where either half may
    be unset and inherit the user value."""
    if chunk_size is not None:
        _at_least(1, chunk_size=chunk_size)
    if chunk_overlap is not None:
        _at_least(0, chunk_overlap=chunk_overlap)
    if chunk_size is not None and chunk_overlap is not None and chunk_overlap >= chunk_size:
        raise InvalidInput(
            f"chunk_overlap must be < chunk_size, got {chunk_overlap} >= {chunk_size}"
        )


# --- structs ----------------------------------------------------------------------


class ChunkSettings(msgspec.Struct, frozen=True):
    """How one collection splits a document's markdown into chunks: the three values the
    embedding cache is keyed by (see `embed_cache.Params`), and nothing else."""

    chunker: Annotated[Chunker, CHUNKER] = "markdown"
    chunk_size: Annotated[int, CHUNK_SIZE] = 1200
    chunk_overlap: Annotated[int, CHUNK_OVERLAP] = 150

    def __post_init__(self) -> None:
        _check_chunking(self.chunk_size, self.chunk_overlap)


class ConversionSettings(msgspec.Struct):
    """The user-level defaults: how a document is converted when nothing else is said at import
    (`parser`, `skip_ocr_pages`), and how a collection chunks it when it overrides nothing."""

    parser: Annotated[Parser, PARSER] = "anydoc"
    chunker: Annotated[Chunker, CHUNKER] = "markdown"
    chunk_size: Annotated[int, CHUNK_SIZE] = 1200
    chunk_overlap: Annotated[int, CHUNK_OVERLAP] = 150
    skip_ocr_pages: Annotated[bool, SKIP_OCR_PAGES] = True

    def __post_init__(self) -> None:
        _check_chunking(self.chunk_size, self.chunk_overlap)

    @property
    def chunking(self) -> ChunkSettings:
        return ChunkSettings(self.chunker, self.chunk_size, self.chunk_overlap)


class SearchSettings(msgspec.Struct):
    limit: Annotated[int, LIMIT] = 10
    candidates: Annotated[int, CANDIDATES] = 50
    mode: Annotated[SearchMode, MODE] = "hybrid"
    fusion: Annotated[Fusion, FUSION] = "rrf"
    rrf_k: Annotated[int, RRF_K] = 60
    vector_weight: Annotated[float, VECTOR_WEIGHT] = 0.7
    bm25_weight: Annotated[float, BM25_WEIGHT] = 0.3
    nprobes: Annotated[int, NPROBES] = 20
    refine_factor: Annotated[int, REFINE_FACTOR] = 10
    reranker: Annotated[Reranker, RERANKER] = "none"
    reranker_model: Annotated[str, RERANKER_MODEL] = RERANKER_MODELS[0]

    def __post_init__(self) -> None:
        _at_least(
            1,
            limit=self.limit,
            candidates=self.candidates,
            rrf_k=self.rrf_k,
            nprobes=self.nprobes,
            refine_factor=self.refine_factor,
        )
        _at_least(0, vector_weight=self.vector_weight, bm25_weight=self.bm25_weight)
        if self.reranker_model not in RERANKER_MODELS:
            raise InvalidInput(f"unknown reranker model: {self.reranker_model}")


class SearchOverrides(msgspec.Struct):
    """Per-collection search overrides. None means "use user default"."""

    limit: Annotated[int | None, LIMIT] = None
    candidates: Annotated[int | None, CANDIDATES] = None
    mode: Annotated[SearchMode | None, MODE] = None
    fusion: Annotated[Fusion | None, FUSION] = None
    rrf_k: Annotated[int | None, RRF_K] = None
    vector_weight: Annotated[float | None, VECTOR_WEIGHT] = None
    bm25_weight: Annotated[float | None, BM25_WEIGHT] = None
    nprobes: Annotated[int | None, NPROBES] = None
    refine_factor: Annotated[int | None, REFINE_FACTOR] = None
    reranker: Annotated[Reranker | None, RERANKER] = None
    reranker_model: Annotated[str | None, RERANKER_MODEL] = None

    def resolve(self, user: SearchSettings) -> SearchSettings:
        return msgspec.structs.replace(user, **without_none(self))


def _half_the_cores() -> int:
    """Default CPU budget: haskie takes half the machine, the rest stays for everything else."""
    return max(1, (os.cpu_count() or 2) // 2)


class PipelineSettings(msgspec.Struct):
    """One CPU budget for the whole app, shared out over the stages by weight.

    `cpu_budget` is how many tasks run at the same time, everywhere: it is the number of slots
    converting, embedding, indexing and maintenance draw from, and it is never exceeded. The
    weights only decide who gets which share of it when every stage has work, because the stages
    cost different things: converting is CPU and IO per page, every embedding task loads the
    model, and indexing writes to LanceDB, which takes one writer per collection. Every stage keeps
    at least one slot, so a budget smaller than three stages is spent by whoever asks first.
    """

    cpu_budget: Annotated[int, CPU_BUDGET] = msgspec.field(default_factory=_half_the_cores)
    converting_weight: Annotated[int, CONVERTING_WEIGHT] = 2
    embedding_weight: Annotated[int, EMBEDDING_WEIGHT] = 2
    indexing_weight: Annotated[int, INDEXING_WEIGHT] = 1
    # 0 = as many slices as the stage's share of the budget
    document_parallelism: Annotated[int, DOCUMENT_PARALLELISM] = 0
    batch_pages: Annotated[int, BATCH_PAGES] = 10
    index_group_parts: Annotated[int, INDEX_GROUP_PARTS] = 50
    task_timeout_seconds: Annotated[int, TASK_TIMEOUT] = 600
    maintenance_docs: Annotated[int, MAINTENANCE_DOCS] = 25
    maintenance_idle_seconds: Annotated[int, MAINTENANCE_IDLE] = 60
    ann_min_rows: Annotated[int, ANN_MIN_ROWS] = 50_000
    preview_workers: Annotated[int, PREVIEW_WORKERS] = 2
    accelerator: Annotated[Accelerator, ACCELERATOR] = "auto"

    def __post_init__(self) -> None:
        _at_least(
            1,
            cpu_budget=self.cpu_budget,
            converting_weight=self.converting_weight,
            embedding_weight=self.embedding_weight,
            indexing_weight=self.indexing_weight,
            batch_pages=self.batch_pages,
            index_group_parts=self.index_group_parts,
            task_timeout_seconds=self.task_timeout_seconds,
            maintenance_docs=self.maintenance_docs,
            maintenance_idle_seconds=self.maintenance_idle_seconds,
            ann_min_rows=self.ann_min_rows,
            preview_workers=self.preview_workers,
        )
        # 0 is a value of its own: "as many slices per document as the stage has slots"
        _at_least(0, document_parallelism=self.document_parallelism)


class RetentionSettings(msgspec.Struct):
    """How long history is kept. Job history twice: `job_live_hours` in the durable-execution
    tables, `job_days` in the day partitions the archiver copies it into (see `archive.py`). The
    audit trail is pruned by the nightly run rather than the hourly archiver, but it is the same
    question, so it is answered in the same place."""

    job_days: Annotated[int, RETENTION_DAYS] = 28
    job_live_hours: Annotated[int, RETENTION_LIVE_HOURS] = 48
    audit_days: Annotated[int, AUDIT_RETENTION] = 90

    def __post_init__(self) -> None:
        _at_least(1, job_days=self.job_days)
        # halved into an archive cutoff and a purge cutoff, so an hour each is the floor
        _at_least(2, job_live_hours=self.job_live_hours)
        _at_least(0, audit_days=self.audit_days)  # 0 = keep everything


class UserSettings(msgspec.Struct):
    embedding: Annotated[EmbeddingProfile, EMBEDDING] = "none"
    conversion: ConversionSettings = msgspec.field(default_factory=ConversionSettings)
    pipeline: PipelineSettings = msgspec.field(default_factory=PipelineSettings)
    search: SearchSettings = msgspec.field(default_factory=SearchSettings)
    retention: RetentionSettings = msgspec.field(default_factory=RetentionSettings)

    @property
    def embedding_model(self) -> EmbeddingModel | None:
        model = PROFILES[self.embedding]
        if model is None:
            return None
        return msgspec.structs.replace(model, accelerator=self.pipeline.accelerator)


class CollectionSettings(msgspec.Struct):
    """Per-collection overrides. None means "use user default".

    Only chunking and search: conversion happens once per document at import, so `parser` and
    `skip_ocr_pages` live on the document (see the module docstring)."""

    chunker: Annotated[Chunker | None, CHUNKER] = None
    chunk_size: Annotated[int | None, CHUNK_SIZE] = None
    chunk_overlap: Annotated[int | None, CHUNK_OVERLAP] = None
    search: SearchOverrides = msgspec.field(default_factory=SearchOverrides)

    def __post_init__(self) -> None:
        _check_chunking(self.chunk_size, self.chunk_overlap)

    def resolve(self, user: UserSettings) -> ChunkSettings:
        overrides = {k: v for k, v in without_none(self).items() if k != "search"}
        return msgspec.structs.replace(user.conversion.chunking, **overrides)

    def resolve_search(self, user: UserSettings) -> SearchSettings:
        return self.search.resolve(user.search)


# --- docs -------------------------------------------------------------------------


class FieldDoc(msgspec.Struct):
    title: str
    description: str


def docs(struct: type[msgspec.Struct] = UserSettings, prefix: str = "") -> dict[str, FieldDoc]:
    """Flat `{"search.limit": FieldDoc, ...}` from the Meta annotations, recursing into nested
    structs. Collection overrides use the same keys (`conversion.*`, `search.*`)."""
    import msgspec.inspect as inspect

    out: dict[str, FieldDoc] = {}
    info = inspect.type_info(struct)
    if not isinstance(info, inspect.StructType):
        raise TypeError(f"docs() needs a msgspec Struct, got {struct!r}")
    for field in info.fields:
        key = f"{prefix}{field.name}"
        kind = field.type
        if isinstance(kind, inspect.Metadata):
            extra = kind.extra_json_schema or {}
            out[key] = FieldDoc(
                title=extra.get("title", key), description=extra.get("description", "")
            )
        elif isinstance(kind, inspect.StructType):
            out.update(docs(kind.cls, f"{key}."))
    return out


# --- persistence ------------------------------------------------------------------
#
# The row changes rarely and is read on nearly every request, search and pipeline step, so the
# decoded struct is cached for the process. This process is the only writer, and the writers here
# refresh the cache; anything that writes the `settings` row another way must call `invalidate()`.


_problem: str | None = None
_cached: UserSettings | None = None
# guards both globals. A threading lock rather than an asyncio one because both event loops of
# this process (Litestar's and DBOS's) load settings; it is never held across an `await`, so no
# loop ever waits on it.
_cache_lock = threading.Lock()


def settings_problem() -> str | None:
    """Why the stored settings could not be read, or None. Set by `load_user_settings_or_none`."""
    return _problem


def invalidate() -> None:
    """Forget the cached settings, so the next load reads the row again.

    Contract: every write of the `settings` row that does not go through `save_user_settings` or
    `init_user_settings` (direct SQL, another process) must call this, or readers keep the value
    this process cached.
    """
    global _cached
    with _cache_lock:
        _cached = None


def _store(settings: UserSettings) -> None:
    """Cache a struct this process just wrote: it decodes, so there is no problem to report."""
    global _cached, _problem
    with _cache_lock:
        _cached = settings
        _problem = None


# Field names stored by earlier builds. msgspec ignores a key it does not know, so without
# this a home written before the rename would come back silently reset to defaults.
_RENAMED_SECTIONS = (("defaults", "conversion"), ("indexing", "pipeline"))
_RENAMED_RETENTION = (("days", "job_days"), ("live_hours", "job_live_hours"))


def _renamed(stored: dict[str, Any]) -> dict[str, Any]:
    """A stored settings blob under the current field names. Idempotent: a blob already written
    by this build has none of the old keys and comes back unchanged."""
    for old, new in _RENAMED_SECTIONS:
        if old in stored:
            stored.setdefault(new, stored.pop(old))
    retention = stored.setdefault("retention", {})
    if isinstance(retention, dict):
        for old, new in _RENAMED_RETENTION:
            if old in retention:
                retention.setdefault(new, retention.pop(old))
        # audit retention used to live under its own "maintenance" section
        audit_days = (stored.pop("maintenance", None) or {}).get("audit_retention_days")
        if audit_days is not None:
            retention.setdefault("audit_days", audit_days)
    return stored


def _decode(raw: str) -> UserSettings:
    """Decode a stored settings row, renaming the fields earlier builds wrote first."""
    return msgspec.convert(_renamed(msgspec.json.decode(raw, type=dict)), UserSettings)


async def load_user_settings_or_none() -> UserSettings | None:
    """None until the first run picked an embedding profile. A stored row that no longer decodes
    must not break boot, so it falls back to defaults and is reported by `settings_problem()`.

    Only a decoded row is cached: the pre-init state and an unreadable row stay live, so the run
    that fixes either one is seen without an `invalidate()`.
    """
    global _cached, _problem
    with _cache_lock:
        if _cached is not None:
            return _cached
    # The read is awaited outside the lock, because a threading lock held across an await would
    # block every other loader, event loop included. Two loads that miss at the same time
    # therefore both read, and both store the same row, which is harmless. A save that landed
    # while we read has already cached its newer row, so the second check below keeps that row
    # instead of replacing it with the one we read before the save.
    async with db.connect() as conn:
        cursor = await conn.execute("select json from settings where id = 1")
        row = await cursor.fetchone()
    problem: str | None = None
    settings: UserSettings | None = None
    if row is not None:
        try:
            settings = _decode(row[0])
        except (msgspec.ValidationError, msgspec.DecodeError) as exc:
            problem = f"stored settings unreadable, using defaults: {exc}"
            _log.error("settings_unreadable", error=str(exc))
    with _cache_lock:
        if _cached is not None:  # a save committed while we were reading; its row is the newer one
            return _cached
        _problem = problem
        if problem is not None:
            return UserSettings()
        _cached = settings  # None when there is no row yet: the pre-init state stays live
        return settings


async def initialized() -> bool:
    return (await load_user_settings_or_none()) is not None


async def load_user_settings() -> UserSettings:
    return (await load_user_settings_or_none()) or UserSettings()


async def save_user_settings(settings: UserSettings) -> UserSettings:
    async with db.connect() as conn:
        await conn.execute(
            "insert into settings (id, json) values (1, ?) "
            "on conflict (id) do update set json = excluded.json",
            (db.dumps(settings),),
        )
    _store(settings)  # after the commit, so a concurrent load cannot cache the previous row
    return settings


async def init_user_settings(settings: UserSettings) -> bool:
    """First run only: store `settings` when no row exists yet, in one statement.

    Returns True when this call created the row. Two concurrent /api/init requests would both
    pass an `initialized()` check, so the conflict clause decides instead.
    """
    async with db.connect() as conn:
        cursor = await conn.execute(
            "insert into settings (id, json) values (1, ?) on conflict (id) do nothing",
            (db.dumps(settings),),
        )
        created = cursor.rowcount == 1  # read on the open connection, before it is closed
    if created:  # the loser wrote nothing, so it must not cache what it tried to write
        _store(settings)
    return created
