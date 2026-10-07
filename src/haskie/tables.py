"""Every table haskie owns in ~/.haskie/haskie.db as SQLAlchemy Core: the one source of the schema.

`db.migrate` generates the DDL from `metadata`, and every query is a Core statement over these
tables, so a column is named once. Changing a table means bumping `db.SCHEMA_VERSION`, and adding
its upgrade to `db.UPGRADES` when the change only adds. DBOS keeps its workflow tables in the same
file and owns their schema; `sysdb.py` declares the ones it reads.

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
# What this machine works out from the rest, rather than what anyone put there: a backup copies the
# column at its default, and the restored home works it out again (`backup`).
DERIVED = {"derived": True}

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
    Column("pending_documents", Integer, nullable=False, server_default=ZERO, info=DERIVED),
    Column("last_write_at", Float, info=DERIVED),
    Column("last_maintained_at", Float, info=DERIVED),
    Column("vector_index_rows", Integer, nullable=False, server_default=ZERO, info=DERIVED),
    # the sum of the unit chunk vectors of its indexed documents under `vector_model`, and how
    # many chunks it sums (`embed_cache.corpus_sum`), set by maintenance: what a search centres
    # cosines on before it weighs them (`search.section_map`)
    Column("vector_sum", LargeBinary, info=DERIVED),
    Column("vector_rows", Integer, nullable=False, server_default=ZERO, info=DERIVED),
    Column("vector_model", Text, info=DERIVED),
)

# a document belongs to no collection: `collection_documents` is the many-to-many, and each
# membership carries the status of writing that document into that collection's table
documents = Table(
    "documents",
    metadata,
    # the MD5 of the original file's bytes: the same file is the same document, and every table
    # and index refers to a document by it
    Column("id", Text, primary_key=True),
    # what people and agents call it, unique and in lowercase-kebab-case (`stored_name`): the API,
    # the tools and a citation address a document by it
    Column("name", Text, nullable=False),
    Column("suffix", Text, nullable=False),
    Column("size", Integer, nullable=False),
    Column("status", Text, nullable=False, server_default="queued"),
    Column("error", Text),
    Column("preview", Text, info=DERIVED),  # built on first open
    Column("parser", Text, nullable=False, server_default="anydoc"),
    Column("skip_ocr_pages", Integer, nullable=False, server_default=text("1")),
    Column("created_at", Float, nullable=False, server_default=ZERO),
    Column("updated_at", Float, nullable=False, server_default=ZERO),
    Column("description", Text, nullable=False, server_default=""),
    Column("pages", Integer),  # a PDF's page count, set by its conversion; None for other formats
    # `id` last, so a search finds the documents being deleted without reading their rows
    Index("idx_documents_status", "status", "name", "id"),
    Index("idx_documents_updated", "updated_at", "name"),
    Index("idx_documents_size", "size", "name"),
    Index("idx_documents_name", "name", unique=True),
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
    Column("document_id", Text, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
    Column("status", Text, nullable=False, server_default="pending"),
    Column("error", Text),
    Column("added_at", Float, nullable=False, server_default=ZERO),
    Column("updated_at", Float, nullable=False, server_default=ZERO),
    # the embedding cache entry its rows are indexed from, None until indexed or once the entry is
    # forgotten: its section ids are that entry's, whatever the chunk settings say by now
    Column("cache_id", Text, ForeignKey("embeddings.id", ondelete="SET NULL")),
    Index("idx_collection_documents_document_id", "document_id"),
    Index("idx_collection_documents_status", "collection", "status", "document_id"),
)

# the durable, content-addressed embedding cache (see indexing/embed_cache.py)
embeddings = Table(
    "embeddings",
    metadata,
    Column("id", Text, primary_key=True),
    Column("document_id", Text, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
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
    # the document as one vector: the mean of its unit chunk vectors, not normalized, float32
    # bytes; what `embed_cache.nearest` compares documents by. Null without an embedding model
    Column("vector", LargeBinary),
    Index("idx_embeddings_document_id", "document_id"),
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
    Index("idx_session_events_session", "session_id", "ts"),
    Index("idx_session_events_operation", "operation_id"),
)

# every search as it ran (see search/log.py), with or without a session: the session history and
# the Insights trend read the searches here, and the Gaps page judges their questions. Retention
# prunes them by age (`retention.search_days`)
searches = Table(
    "searches",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("ts", Float, nullable=False),
    Column("session_id", Text),
    Column("actor", Text, nullable=False),
    Column("tool", Text, nullable=False),
    Column("context", Text),  # the background an excerpts search's questions shared
    Column("collections", Text, nullable=False, server_default="[]"),
    Column("mode", Text),
    Column("embedding", Text),  # the profile key: query vectors compare within one profile only
    Column("reranker", Text),  # the cross-encoder model, None when the search did not rerank
    Column("min_rerank_score", Float),  # the settings' floor in place of the reranker's own
    Column("result_limit", Integer),
    Column("result_count", Integer, nullable=False, server_default=ZERO),
    # kept to some documents or sections (`document_ids`, `section_ids`): a miss is no gap
    Column("scoped", Integer, nullable=False, server_default=ZERO),
    Column("duration_ms", Integer, nullable=False, server_default=ZERO),
    Column("error", Text),
    # the words of its questions no excerpt held (`Answer.missing_terms`), JSON; excerpts only
    Column("missing_terms", Text, nullable=False, server_default="[]"),
    Index("idx_searches_ts", "ts"),
    Index("idx_searches_session", "session_id", "ts"),
)

# each question one search asked, in the order asked, and what its own ranking measured: an
# excerpts search runs one ranking per question, and the Gaps page judges and reviews each alone
search_questions = Table(
    "search_questions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("search_id", Integer, ForeignKey("searches.id", ondelete="CASCADE"), nullable=False),
    Column("position", Integer, nullable=False),
    Column("question", Text, nullable=False),
    # the score profile of its ranking: the best cosines of the query to the rows read and the
    # reranker's best scores before its floor, each float32 and best first (`log.PROFILE`)
    Column("similarities", LargeBinary),
    Column("rerank_scores", LargeBinary),
    # several were asked, and no excerpt answers this one
    Column("uncovered", Integer, nullable=False, server_default=ZERO),
    Column("review", Text, CheckConstraint("review in ('dismissed', 'resolved')")),
    # the agent's own verdict on what the search gave it (`gaps.report`), and what it lacked
    Column("agent_verdict", Text, CheckConstraint("agent_verdict in ('insufficient', 'partial')")),
    Column("agent_note", Text),
    # float32, the search's `embedding` dimensions; last, since a row's columns past a blob this
    # big are read from overflow pages, and most reads skip it
    Column("query_vector", LargeBinary),
    Index("idx_search_questions_search", "search_id", "position", unique=True),
)

# what one search returned: every result and every place folded into one (`also_in`), in
# preorder, each under its parent's position
search_results = Table(
    "search_results",
    metadata,
    Column("search_id", Integer, ForeignKey("searches.id", ondelete="CASCADE"), primary_key=True),
    Column("position", Integer, primary_key=True),
    Column("parent", Integer),
    Column("relation", Text),
    Column("collection", Text, nullable=False),
    Column("document", Text, nullable=False),
    Column("seq_start", Integer),
    Column("seq_end", Integer),
    Column("line_start", Integer, nullable=False),
    Column("line_end", Integer, nullable=False),
    Column("header", Text, nullable=False),
    Column("location", Text, nullable=False),
    Column("score", Float, nullable=False),
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
    # 1: the vectors are cut to `dims` (Matryoshka Representation Learning)
    Column(
        "matryoshka",
        Integer,
        CheckConstraint("matryoshka in (0, 1)"),
        nullable=False,
        server_default="0",
    ),
    Column("duplicate_chunk", Float),
    Column("duplicate_passage", Float),
    CheckConstraint("(duplicate_chunk is null) = (duplicate_passage is null)"),
    # the cosines search/gaps.py judges by: a best match under `weak_match` is no answer, and two
    # queries over `same_topic` ask about one thing
    Column("weak_match", Float),
    Column("answered_match", Float),  # at and over it an answer; between the two, borderline
    Column("same_topic", Float),
    CheckConstraint("answered_match is null or answered_match >= weak_match"),
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

# where `haskie install <agent>` wrote the skill and rule, so a collection change can rewrite them
# (`claude.refresh_installations`); `directory` is the agent's configuration directory
installations = Table(
    "installations",
    metadata,
    Column("agent", Text, CheckConstraint("agent in ('claude', 'codex')"), primary_key=True),
    Column("directory", Text, primary_key=True),
)
