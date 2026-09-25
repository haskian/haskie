"""User settings (global, chosen at init) and per-collection overrides.

Every setting carries a `Meta(title, description)`: the single definition shown in the UI
(`/api/options` -> `docs`) and in the JSON schema. Collection overrides reuse the same Meta
objects.

What is settable where follows where the work happens. Conversion (`parser`, `skip_ocr_pages`)
runs once per document, at import, so its values are chosen then and stored on the document;
`UserSettings.conversion` only supplies the defaults an import falls back on. Chunking
(`chunker`, `chunk_size`, `chunk_merge_below`, `chunk_frame`) splits the shared markdown per
collection, so a collection may override it (`CollectionOverrides`), and the embedding cache is
keyed by it. Only `chunk_size` is in characters; `chunk_merge_below` is a percentage of it, so it
still means the same thing when the size changes.
"""

import os
import threading
from enum import StrEnum
from typing import Annotated, Any

import msgspec
from msgspec import Meta
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert

from haskie import db
from haskie.errors import InvalidInput
from haskie.logs import get_logger
from haskie.tables import settings as settings_table

_log = get_logger(__name__)


class Parser(StrEnum):  # PDFs always go page-wise through pdf-inspector
    ANYDOC = "anydoc"
    PLAIN = "plain"


class Chunker(StrEnum):  # the two pipelines of `indexing.chunk`
    MARKDOWN = "markdown"
    TEXT = "text"


class Accelerator(StrEnum):
    AUTO = "auto"  # the best ONNX Runtime provider: CUDA where installed, else the CPU
    CPU = "cpu"
    COREML = "coreml"  # ONNX Runtime's CoreML on Apple Silicon, only when asked for (`embed`)


class SearchMode(StrEnum):
    HYBRID = "hybrid"
    VECTOR = "vector"
    FTS = "fts"


class Fusion(StrEnum):
    RRF = "rrf"
    LINEAR = "linear"


class Reranker(StrEnum):
    NONE = "none"
    CROSS_ENCODER = "cross-encoder"


NO_EMBEDDING = "none"  # the embedding profile of full-text search only: no model at all
# the smallest reranker in the catalogue (`catalogue.rerankers`), so the default costs least
DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"


# --- definitions ------------------------------------------------------------------

EMBEDDING = Meta(
    title="Embedding profile",
    description=(
        "Text-embedding model that turns chunks into vectors for semantic search. Chosen at "
        'first run; changing it later requires "Index all" in every collection. none = full-text '
        "(BM25) search only. Each profile lists its model, size, languages, license and the "
        "hardware it runs well on."
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
        "How converted Markdown is split into chunks. Every paragraph (text between blank lines) "
        "is a chunk, and short ones are merged (see Merge short paragraphs). A paragraph longer "
        "than Chunk size is cut between list items or blocks, else between sentences. A "
        "sentence, table or code block longer than a chunk is cut at a line, then a word. "
        "markdown: every heading starts a new chunk and stays out of its text; code blocks and "
        "tables stay whole. text: ignore the Markdown structure and split on paragraphs and "
        "sentences only."
    ),
)
CHUNK_SIZE = Meta(
    title="Chunk size (characters)",
    description=(
        "Maximum length of one chunk, counted in Unicode characters, not words or tokens "
        "(about 4 characters per English token). A chunk is also the unit that gets one "
        "embedding vector and one search result. With Prepend heading path on, the size counts "
        "that path too: the path and the text together never exceed it."
    ),
)
CHUNK_MERGE_BELOW = Meta(
    title="Merge short paragraphs (% of chunk size)",
    description=(
        "A paragraph - text between blank lines, or a whole list - shorter than "
        "this share of Chunk size is merged with the paragraphs around it: into the one below "
        "when both fit one chunk, else with the short ones next to it. Longer paragraphs are "
        "chunks of their own. 0 never merges; 100 merges every paragraph that fits."
    ),
)
CHUNK_FRAME = Meta(
    title="Prepend heading path",
    description=(
        "The embedding model and the reranker read every chunk with its heading path in front "
        "(Part I > Replication > Leaders). Chunk size counts the path. A path longer than half "
        "of it loses its outermost headings first. Off: the models read the chunk's text alone. "
        "Only the markdown chunker has a heading path to prepend."
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
MAINTENANCE_DOCUMENTS = Meta(
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
    title="Model hardware",
    description=(
        "Device for the embedding and reranker models. auto: CUDA with the gpu extra, else "
        "CPU. On Apple Silicon, the MLX and GGUF models run on the GPU, and auto runs the rest "
        "on the CPU. cpu: force CPU; the MLX and GGUF models need the GPU, so none is offered. "
        "coreml: run ONNX models through CoreML on Apple Silicon; today that is slower than the "
        "CPU for them."
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
        "The model the cross-encoder reranker scores with; what each one is, its size, languages, "
        "license and hardware are listed with it. Downloaded on first use."
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
    title="Operation history (days)",
    description=(
        "How many days of finished operations and their tasks stay visible under Operations. "
        "The nightly maintenance run deletes everything older."
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


def _check_chunking(chunk_size: int | None, chunk_merge_below: int | None) -> None:
    """Shared by the user-level settings and the per-collection overrides, where any of them may
    be unset and inherit the user value."""
    if chunk_merge_below is not None and not 0 <= chunk_merge_below <= 100:
        raise InvalidInput(f"chunk_merge_below must be 0 to 100, got {chunk_merge_below}")
    if chunk_size is not None:
        _at_least(1, chunk_size=chunk_size)


# --- structs ----------------------------------------------------------------------


class ChunkSettings(msgspec.Struct, frozen=True):
    """How one collection splits a document's markdown into chunks: the values the embedding
    cache is keyed by (see `embed_cache.Params`), and nothing else."""

    chunker: Annotated[Chunker, CHUNKER] = Chunker.MARKDOWN
    chunk_size: Annotated[int, CHUNK_SIZE] = 1200
    chunk_merge_below: Annotated[int, CHUNK_MERGE_BELOW] = 66  # percent of chunk_size
    chunk_frame: Annotated[bool, CHUNK_FRAME] = True

    def __post_init__(self) -> None:
        _check_chunking(self.chunk_size, self.chunk_merge_below)

    @classmethod
    def of(cls, source: object) -> "ChunkSettings":
        """The chunk settings `source` carries under the same field names: the user defaults or an
        embedding cache key. One copy, so a new chunk setting reaches every caller."""
        return cls(**{name: getattr(source, name) for name in cls.__struct_fields__})


class ConversionSettings(ChunkSettings, frozen=True):
    """The user-level defaults: how a document is converted when nothing else is said at import
    (`parser`, `skip_ocr_pages`), and how a collection chunks it when it overrides nothing.

    The chunk fields are inherited rather than restated, so their defaults, their `Meta` and
    their check have one home. msgspec puts inherited fields first, which is the order they were
    already written in, so the stored JSON is unchanged."""

    parser: Annotated[Parser, PARSER] = Parser.ANYDOC
    skip_ocr_pages: Annotated[bool, SKIP_OCR_PAGES] = True

    @property
    def chunking(self) -> ChunkSettings:
        return ChunkSettings.of(self)


class SearchSettings(msgspec.Struct):
    limit: Annotated[int, LIMIT] = 25
    candidates: Annotated[int, CANDIDATES] = 50
    mode: Annotated[SearchMode, MODE] = SearchMode.HYBRID
    fusion: Annotated[Fusion, FUSION] = Fusion.RRF
    rrf_k: Annotated[int, RRF_K] = 60
    vector_weight: Annotated[float, VECTOR_WEIGHT] = 0.7
    bm25_weight: Annotated[float, BM25_WEIGHT] = 0.3
    nprobes: Annotated[int, NPROBES] = 20
    refine_factor: Annotated[int, REFINE_FACTOR] = 10
    reranker: Annotated[Reranker, RERANKER] = Reranker.NONE
    reranker_model: Annotated[str, RERANKER_MODEL] = DEFAULT_RERANKER

    def __post_init__(self) -> None:
        _at_least(
            1,
            limit=self.limit,
            candidates=self.candidates,
            rrf_k=self.rrf_k,
            nprobes=self.nprobes,
            refine_factor=self.refine_factor,
        )
        # `reranker_model` is checked against the catalogue where settings are written
        # (`catalogue.check`): the catalogue is in the database, and decoding reads none
        _at_least(0, vector_weight=self.vector_weight, bm25_weight=self.bm25_weight)


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
    maintenance_documents: Annotated[int, MAINTENANCE_DOCUMENTS] = 25
    maintenance_idle_seconds: Annotated[int, MAINTENANCE_IDLE] = 60
    ann_min_rows: Annotated[int, ANN_MIN_ROWS] = 50_000
    preview_workers: Annotated[int, PREVIEW_WORKERS] = 2
    accelerator: Annotated[Accelerator, ACCELERATOR] = Accelerator.AUTO

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
            maintenance_documents=self.maintenance_documents,
            maintenance_idle_seconds=self.maintenance_idle_seconds,
            ann_min_rows=self.ann_min_rows,
            preview_workers=self.preview_workers,
        )
        # 0 is a value of its own: "as many slices per document as the stage has slots"
        _at_least(0, document_parallelism=self.document_parallelism)


class RetentionSettings(msgspec.Struct):
    """How long history is kept: the operation history in DBOS's own tables, and the audit trail
    on disk. Both are swept by the nightly maintenance run (see `workflows.daily_maintenance`), so
    they are answered in the same place."""

    operation_days: Annotated[int, RETENTION_DAYS] = 28
    audit_days: Annotated[int, AUDIT_RETENTION] = 90

    def __post_init__(self) -> None:
        _at_least(1, operation_days=self.operation_days)
        _at_least(0, audit_days=self.audit_days)  # 0 = keep everything


class UserSettings(msgspec.Struct):
    # a profile key of the catalogue, checked where settings are written (`catalogue.check`)
    embedding: Annotated[str, EMBEDDING] = NO_EMBEDDING
    conversion: ConversionSettings = msgspec.field(default_factory=ConversionSettings)
    pipeline: PipelineSettings = msgspec.field(default_factory=PipelineSettings)
    search: SearchSettings = msgspec.field(default_factory=SearchSettings)
    retention: RetentionSettings = msgspec.field(default_factory=RetentionSettings)


class CollectionOverrides(msgspec.Struct):
    """Per-collection overrides. None means "use user default".

    Only chunking and search: conversion happens once per document at import, so `parser` and
    `skip_ocr_pages` live on the document (see the module docstring)."""

    chunker: Annotated[Chunker | None, CHUNKER] = None
    chunk_size: Annotated[int | None, CHUNK_SIZE] = None
    chunk_merge_below: Annotated[int | None, CHUNK_MERGE_BELOW] = None
    chunk_frame: Annotated[bool | None, CHUNK_FRAME] = None
    search: SearchOverrides = msgspec.field(default_factory=SearchOverrides)

    def __post_init__(self) -> None:
        _check_chunking(self.chunk_size, self.chunk_merge_below)

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
# refresh the cache.


class _Loaded(msgspec.Struct, frozen=True):
    """What the last reading of the `settings` row found: the struct it decoded to, or why it did
    not. `settings` is None before the first run has stored one, and when the stored row is
    unreadable.

    Immutable and swapped whole, so a reader always sees the pair together: it is rebound, never
    mutated."""

    settings: UserSettings | None
    problem: str | None = None


_state: _Loaded | None = None  # None until the first read; only a decoded row ends the reading
# Both event loops of this process (Litestar's and DBOS's) load and store settings, so the guard
# is a threading one. Held for the re-check plus the rebind, and never across an `await`.
_cache_lock = threading.Lock()


def settings_problem() -> str | None:
    """Why the stored settings could not be read, or None. Set by `load_user_settings_or_none`."""
    return _state.problem if _state else None


def _store(settings: UserSettings) -> None:
    """Cache a struct this process just wrote: it decodes, so there is no problem to report."""
    global _state
    with _cache_lock:
        _state = _Loaded(settings)


def _decode(raw: str) -> UserSettings:
    """Decode a stored settings row. No renames of old keys: a home written under other field
    names is at another `db.SCHEMA_VERSION` and is refused before its settings are read."""
    return msgspec.json.decode(raw, type=UserSettings)


async def load_user_settings_or_none() -> UserSettings | None:
    """None until the first run picked an embedding profile. A stored row that no longer decodes,
    or names a model the catalogue does not hold, must not break boot, so it falls back to
    defaults and is reported by `settings_problem()`.

    Only a decoded row ends the reading: the pre-init state and an unreadable row are read again
    on the next call, so the run that fixes either one is seen at once.
    """
    global _state
    found = _state
    if found is not None and found.settings is not None:
        return found.settings
    # imported here: the catalogue reads the settings types, so it cannot be imported first
    from haskie.catalogue import catalogue

    loaded = _Loaded(None)
    async with db.connect() as conn:
        raw = await conn.scalar(select(settings_table.c.json).where(settings_table.c.id == 1))
        if raw is not None:
            try:
                decoded = _decode(raw)
                # the catalogue is in the database, so a decoded row may still name a lost model
                problem = await catalogue.unknown(conn, decoded)
            except (msgspec.ValidationError, msgspec.DecodeError) as exc:
                decoded, problem = None, str(exc)
            if problem:
                _log.error("settings_unreadable", error=problem)
                loaded = _Loaded(None, f"stored settings unreadable, using defaults: {problem}")
            else:
                loaded = _Loaded(decoded)
    with _cache_lock:  # re-check and rebind together, so a save mid-read is not overwritten
        found = _state
        if found is not None and found.settings is not None:
            return found.settings  # a save committed while we read; its row is the newer one
        _state = loaded
    return UserSettings() if loaded.problem else loaded.settings


async def load_user_settings() -> UserSettings:
    return (await load_user_settings_or_none()) or UserSettings()


async def save_user_settings(settings: UserSettings) -> UserSettings:
    async with db.connect() as conn:
        written = insert(settings_table).values(id=1, json=db.dumps(settings))
        await conn.execute(
            written.on_conflict_do_update(
                index_elements=[settings_table.c.id], set_={"json": written.excluded.json}
            )
        )
    _store(settings)  # after the commit, so a concurrent load cannot cache the previous row
    return settings


async def init_user_settings(settings: UserSettings) -> bool:
    """First run only: store `settings` when no row exists yet, in one statement.

    Returns True when this call created the row. Two concurrent /api/init requests would both
    pass an `initialized()` check, so the conflict clause decides instead.
    """
    async with db.connect() as conn:
        result = await conn.execute(
            insert(settings_table).values(id=1, json=db.dumps(settings)).on_conflict_do_nothing()
        )
        created = result.rowcount == 1  # read on the open connection, before it is closed
    if created:  # the loser wrote nothing, so it must not cache what it tried to write
        _store(settings)
    return created
