"""Every table of ~/.haskie/haskie.db as SQLAlchemy Core: the one source of the schema.

`db.migrate` generates the DDL from `metadata`, and every query is a Core statement over these
tables, so a column is named once. Changing a table means bumping `db.SCHEMA_VERSION`.

SQLite has no date type: a `Float` timestamp holds the unix seconds `time.time()` returns. Flags
stay `Integer` (0 or 1), the way sqlite stores them anyway.
"""

from sqlalchemy import (
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    Table,
    Text,
    text,
)

metadata = MetaData()

ZERO = text("0")

settings = Table(
    "settings",
    metadata,
    Column("id", Integer, CheckConstraint("id = 1"), primary_key=True),
    Column("json", Text, nullable=False),
)

sessions = Table(
    "sessions",
    metadata,
    Column("id", Text, primary_key=True),
)

collections = Table(
    "collections",
    metadata,
    Column("name", Text, primary_key=True),
    Column("overrides", Text, nullable=False, server_default="{}"),
    Column("description", Text, nullable=False, server_default=""),
    Column("created_at", Float, nullable=False, server_default=ZERO),
    Column("pending_documents", Integer, nullable=False, server_default=ZERO),
    Column("last_write_at", Float),
    Column("last_maintained_at", Float),
    Column("vector_index_rows", Integer, nullable=False, server_default=ZERO),
)

# a document belongs to no collection: `collection_documents` is the many-to-many, and each
# membership carries the status of writing that document into that collection's table
documents = Table(
    "documents",
    metadata,
    Column("name", Text, primary_key=True),
    Column("suffix", Text, nullable=False),
    Column("size", Integer, nullable=False),
    Column("status", Text, nullable=False, server_default="queued"),
    Column("error", Text),
    Column("preview", Text),
    Column("parser", Text, nullable=False, server_default="anydoc"),
    Column("skip_ocr_pages", Integer, nullable=False, server_default=text("1")),
    Column("created_at", Float, nullable=False, server_default=ZERO),
    Column("updated_at", Float, nullable=False, server_default=ZERO),
    Column("description", Text, nullable=False, server_default=""),
    # MD5 of the original file's bytes: an upload with the same hash is the same file again
    Column("md5", Text, nullable=False),
    Index("idx_documents_status", "status", "name"),
    Index("idx_documents_updated", "updated_at", "name"),
    Index("idx_documents_size", "size", "name"),
    Index("idx_documents_md5", "md5"),
)

collection_documents = Table(
    "collection_documents",
    metadata,
    Column(
        "collection",
        Text,
        ForeignKey("collections.name", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("document", Text, ForeignKey("documents.name", ondelete="CASCADE"), primary_key=True),
    Column("status", Text, nullable=False, server_default="pending"),
    Column("error", Text),
    Column("added_at", Float, nullable=False, server_default=ZERO),
    Column("updated_at", Float, nullable=False, server_default=ZERO),
    Index("idx_collection_documents_document", "document"),
    Index("idx_collection_documents_status", "collection", "status", "document"),
)

# the durable, content-addressed embedding cache (see indexing/embed_cache.py)
embeddings = Table(
    "embeddings",
    metadata,
    Column("id", Text, primary_key=True),
    Column("document", Text, ForeignKey("documents.name", ondelete="CASCADE"), nullable=False),
    Column("urn", Text, nullable=False),
    Column("model", Text, nullable=False),
    Column("chunk_size", Integer, nullable=False),
    Column("chunk_merge_below", Integer, nullable=False),
    Column("chunk_frame", Integer, nullable=False),
    Column("chunker", Text, nullable=False),
    Column("chunk_version", Integer, nullable=False),
    Column("parser", Text, nullable=False),
    Column("skip_ocr_pages", Integer, nullable=False),
    Column("rows", Integer, nullable=False, server_default=ZERO),
    Column("bytes", Integer, nullable=False, server_default=ZERO),
    Column("created_at", Float, nullable=False, server_default=ZERO),
    # the document as one vector: the mean of its unit chunk vectors, normalized, as float32
    # bytes; what `embed_cache.nearest` compares documents by. Null without an embedding model
    Column("vector", LargeBinary),
    Index("idx_embeddings_document", "document"),
)

session_collections = Table(
    "session_collections",
    metadata,
    Column("session_id", Text, ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True),
    Column(
        "collection",
        Text,
        ForeignKey("collections.name", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("position", Integer, nullable=False),
    Index("idx_session_collections_collection", "collection"),
)

# what a session did, so its history can be shown and an operation can name the session that
# started it; the rows go with the session
session_events = Table(
    "session_events",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("session_id", Text, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("ts", Float, nullable=False),
    Column("action", Text, nullable=False),
    Column("subject", Text, nullable=False),
    Column("detail", Text, nullable=False, server_default="{}"),
    Column("operation_id", Text),
    Column("duration_ms", Integer, nullable=False, server_default=ZERO),
    Index("idx_session_events_session", "session_id", "ts"),
    Index("idx_session_events_operation", "operation_id"),
)

# the model catalogue (see catalogue/catalogue.py): every model the runtimes can load, with its
# metadata, and every embedding profile. Seeded once from `catalogue/seed.sql`
models = Table(
    "models",
    metadata,
    Column("name", Text, primary_key=True),
    Column("kind", Text, CheckConstraint("kind in ('embedder', 'reranker')"), nullable=False),
    Column("description", Text, nullable=False),
    Column("parameters", Integer, CheckConstraint("parameters > 0"), nullable=False),
    Column("context_tokens", Integer, CheckConstraint("context_tokens > 0"), nullable=False),
    Column("languages", Text, nullable=False),
    Column("license", Text, nullable=False),
    Column("released", Text, CheckConstraint("released is date(released)"), nullable=False),
    Column(
        "model_card_url",
        Text,
        CheckConstraint("model_card_url like 'https://huggingface.co/%'"),
        nullable=False,
    ),
)

embedding_profiles = Table(
    "embedding_profiles",
    metadata,
    Column("profile", Text, CheckConstraint("profile != 'none'"), primary_key=True),
    Column("model", Text, ForeignKey("models.name"), nullable=False),
    Column("dims", Integer, CheckConstraint("dims > 0"), nullable=False),
    Column("description", Text),
    Column("query_prefix", Text, nullable=False, server_default=""),
    Column("document_prefix", Text, nullable=False, server_default=""),
    Column(
        "matryoshka_layer_norm",
        Integer,
        CheckConstraint("matryoshka_layer_norm in (0, 1)"),
    ),
    Column("duplicate_chunk", Float),
    Column("duplicate_passage", Float),
    CheckConstraint("(duplicate_chunk is null) = (duplicate_passage is null)"),
)

# how one reranker's scores read (`catalogue.calibration`): its floor, under which it judged a
# chunk no answer (`min_rerank_score`), and the beta curve that spreads its scores evenly over 0 to
# 1 (`fill_values = absolute`). Measured on borderline pairs (`catalogue.calibrate`), else the
# uncalibrated defaults the seed gives
reranker_calibration = Table(
    "reranker_calibration",
    metadata,
    Column("model", Text, ForeignKey("models.name"), primary_key=True),
    Column("floor", Float, CheckConstraint("floor between 0 and 1"), nullable=False),
    Column("beta_a", Float, CheckConstraint("beta_a > 0"), nullable=False),
    Column("beta_b", Float, CheckConstraint("beta_b > 0"), nullable=False),
    Column("source", Text, nullable=False),  # what measured it, or "uncalibrated"
)

# an upload waiting in `staging/`, before any name is taken
staging = Table(
    "staging",
    metadata,
    Column("staging_id", Text, primary_key=True),
    Column("filename", Text, nullable=False),
    Column("size", Integer, nullable=False),
    Column("md5", Text, nullable=False),  # of the bytes, carried to the import
    Column("created_at", Float, nullable=False, server_default=ZERO),
)
