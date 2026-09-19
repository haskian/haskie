# haskie

**Your taste, as a knowledge base your AI agents can read.**

The web is an average of everyone. Your bookshelf is not. haskie turns the papers, books, manuals
and notes you own and actually trust — or simply the documents that matter for your AI flows —
into private, searchable collections, and hands them to your AI agents over MCP, so they answer
from your sources instead of whatever ranked first today.

```sh
uv tool install haskie
haskie run          # http://127.0.0.1:8000 — web UI, REST API, MCP server
```

## The problem

You already know which sources are right. Your agent doesn't.

- **It answers from the internet's median.** Ask about a niche topic and you get the blog post that
  won SEO in 2019, not the one book that settles the question.
- **You cannot vouch for what it cited.** Web search pulls in pages you have never read, from
  authors you have never heard of, and the answer arrives with the same confidence either way.
- **Your best sources are unreadable to it.** The PDFs, the standards, the internal decision
  records — all sitting in a folder no tool can search by meaning.
- **Pasting doesn't scale.** Dropping a 400-page PDF into a chat burns the context window and still
  loses the other nine documents you needed.
- **"Just use RAG" is a weekend project.** Parser, chunker, embeddings, a vector store, a reranker,
  a queue for when it all takes ten minutes — and now you maintain it.
- **Cloud knowledge bases want your documents.** Upload, accept the terms, hope the retention
  policy holds.

## The fix

Import your documents once — PDFs, markdown, office files — and curate a collection per topic:
*the three coffee-roasting books worth reading*, *our internal architecture decisions*, *the
standards that actually apply to this hardware*. The same document can sit in as many collections
as it belongs to. haskie converts, chunks and indexes on your machine, and computes each embedding
once however many collections share it. Then any MCP client — Claude Code, Claude Desktop, your
own agent — searches a collection by meaning, by keyword, or both.

- **Trusted sources, not search results.** You decide what goes in. Nothing else answers.
- **A collection per topic.** Point an agent at the collections that matter for the question at
  hand; a document is imported once and reused everywhere it belongs.
- **Local and private.** Your files, your machine, loopback by default. No account, no upload,
  nothing redistributed.
- **Agent-native.** MCP server and REST API from the same app, plus a web UI to curate with.
- **Nothing to assemble.** Conversion, chunking, embeddings, hybrid search, reranking and a durable
  job pipeline ship in one command.

Taste does not scale by explaining it. It scales by indexing it.

## Five minutes to a working collection

```sh
uv tool install haskie      # or `uv tool install .` from a checkout
haskie init                 # create ~/.haskie and its database
haskie run                  # web UI, REST API and MCP server on http://127.0.0.1:8000
haskie destroy              # delete ~/.haskie and everything in it
```

Open the UI, drop files into Documents (they are converted and embedded on import), make a
collection, add the documents that belong to it. Then point your agent at
`POST http://127.0.0.1:8000/mcp` and ask it something only your documents know.

`init` is idempotent, so it is also how a home is brought up to date after an upgrade; `run` does
the same work at startup, so `init` is only needed to do it first. Both take `--home` (or
`HASKIE_HOME`) to keep the data somewhere else, and `run` takes `--host`, `--port` and `--reload`.
It binds loopback by default: the home directory is one user's documents, and nothing in the app
authenticates a caller.

A home written by a build older than the collections model (one with a `library/` folder) is not
migrated: the app refuses to start against it and says so. Run `haskie destroy` and import the
documents again.

`destroy` asks before it deletes and prints what would be lost first. It refuses any directory
that is not a haskie home, so a mistyped `--home` cannot take the wrong tree with it; `--yes`
skips the prompt. Nothing is backed up, and a running `haskie run` should be stopped first.

The built web UI ships inside the wheel, so an installed `haskie` serves the interface and not
just the API. From a checkout, `mise run build` puts it there; without it the app still runs, API
and MCP only.

## MCP: what your agent gets

Endpoint `POST http://127.0.0.1:8000/mcp`. Tools: `list_documents`, `get_document`,
`add_document`, `describe_document`, `list_collections`, `get_collection`,
`list_collection_documents`, `add_document_to_collection`, `remove_document_from_collection`,
`set_session_collections`, `search`, `search_collection`, `search_text`, `search_documents`.
Set the session's collections first, then search with the same id — that is how an agent scopes
itself to *this topic's* sources for the rest of the conversation. `search_text` needs no session
and no model: one keyword query across everything, from a cold start.

## Search that earns the trust you put in the collection

A curated collection is only as good as the retrieval over it, so the retrieval is not an
afterthought.

User-level defaults with per-collection overrides (`limit`, `candidates`, `mode`, `fusion`,
`rrf_k`, `vector_weight`, `bm25_weight`); `search_collection` also accepts them per call. `mode`:
`hybrid` (vector + BM25, fused), `vector`, `fts`; without an embedding profile everything is `fts`.
`fusion`: `rrf` (reciprocal rank fusion, `rrf_k`) or `linear` (weighted sum of normalized scores;
the two weights are normalized together). `reranker = cross-encoder` rescores the `candidates` of
any mode (vector, fts or hybrid) with a fastembed cross-encoder (`reranker_model`).

`search_text` (`GET /api/search/text`) is the session-free alternative: one BM25 query over every
collection at once, or over the comma-separated `collections`, with no embedding model and nothing
to set up first. Scores are raw BM25 rather than fused ranks, because one lexical scorer with the
same tokenizer answers everywhere, so two collections are on one scale; normalizing per collection
would put every collection's rank-1 chunk on page one. A document that sits in several of the
collections searched is reported once per passage, not once per collection. The result is paged
(`page_size` up to 200, `next_cursor` back in as `cursor`, at most 1000 results deep): the cursor
is an opaque offset bound to the query, since a full-text query cannot be filtered by score. Every
page recomputes the ranking, so a document indexed between two pages can move a hit across a page
boundary, and a collection created or deleted meanwhile invalidates the cursor (422). A collection
with no full-text index yet — one in the middle of its first index — contributes nothing rather
than making the whole query wait for it.

A session search (`search`, `GET /api/search`) asks every collection of the session at once: the
query is embedded once and each model is checked once for the whole fan-out, the collections are
read in parallel (up to 8 at a time), and the per-collection rankings are fused by rank (reciprocal
rank fusion, `rrf_k`), because scores from two indexes are not comparable. A passage that two of
the collections both hold counts once. A session `Hit.score` is therefore an RRF score, or the
cross-encoder's when `reranker = cross-encoder` — one rerank pass over the merged candidates. A
session with a single collection keeps that collection's own scores, and `search_collection` is
unaffected either way. A collection deleted since the session chose it is skipped; a collection
that fails to answer fails the search, rather than leaving a hole that reads as "no match".

`search_documents` (`GET /api/search/documents`) answers "which documents should I read" rather
than "which passages say so": the same BM25 scan, folded to one row per document with its best
chunk, how many scanned chunks it matched, and its description.

Every setting has a title and definition (`msgspec.Meta` on the field; served as `/api/options` →
`docs`, shown in the UI), so tuning is done in the UI rather than by reading this file. Chunk sizes
are in characters.

## How it works

Documents and collections are two different things, and the storage follows that.

A **document** is imported once, under a name that never changes, and belongs to no collection.
Intake is two-phase: an upload lands in `staging/` and commits nothing; the import fixes the name
(the original suffix is kept, because it decides the parser), creates the row, moves the file into
the document's own folder and starts the pipeline: `queued` → `converting` → `embedding` →
`imported` (or `error` / `cancelled`). Conversion happens once here (pdf-inspector page-wise for
PDF, anydoc for office formats), so `parser` and `skip_ocr_pages` are chosen at import and stored
on the document; the import also pre-warms the embedding cache under the user's default chunk
settings. The preview (first 10 PDF pages) is built on first open.

A **collection** is a set of documents with one LanceDB index and its own chunk and search
settings. Adding an imported document to a collection is an indexing operation with a status of
its own (`pending` → `indexing` → `indexed` / `error`), independent per collection: it makes sure
the embedding the collection's chunk settings call for exists — computing it only when no
collection asked for that exact combination before — and writes the rows out of that cache into
the collection's table. Removing a document from a collection deletes its rows there and nothing
else; deleting a collection touches no document; deleting a document takes it out of every
collection, then drops its folder.

The **embedding cache** is what makes a document cheap to share. Every computed embedding is one
parquet file under the document (`embeddings/<id>.parquet`, one row group per convert part) plus
one `embeddings` row, keyed by a canonical URN of everything the rows depend on —
`document:<doc>;model:<model>;chunk_size:<n>;chunk_overlap:<n>;chunker:<c>;chunk_version:<v>;parser:<p>;skip_ocr_pages:<b>`
— hashed with sha256 to the id. Same inputs, same id, computed once; two collections asking for the
same missing entry at the same moment share one run (DBOS deduplication on the id). The
accelerator is not in the key: it selects where a model runs, not what it computes.

### Where everything lives

Plain files in a directory you can back up, inspect or delete. No opaque store.

```
~/.haskie/
  cache/models/            compiled CoreML models (ONNX Runtime writes it; safe to delete)
  haskie.db               SQLite (aiosqlite, WAL): settings, documents, collections, memberships,
                           embeddings metadata, sessions, plus DBOS workflow tables. Schema
                           versioned via PRAGMA user_version.
  staging/<uuid>.<ext>    uploads not yet imported; swept after a day by the nightly run
  documents/<sh>/<doc>/
    original.<ext>            the file as imported
    original.<ext>.md         full conversion (assembled from the parts at import)
    parts/NNNNNN.md           per micro-batch of `pipeline.batch_pages` pages; kept, because every
                              collection re-chunks from the same part boundaries
    preview/                  source (pdf cut to 10 pages / image / text / html) + preview.md
    embeddings/<id>.parquet   one cached embedding per chunk settings x model (see above)
  collections/<sh>/<name>/
    index/                    LanceDB table "chunks"
```

`<sh>` is the shard directory of the entry: the first byte of the SHA-1 of its name, in hex
(`layout.shard`). Ten thousand documents therefore spread over 256 directories instead of filling
one.

### Indexing that survives a crash

Indexing a shelf of books takes minutes to hours. Closing the laptop mid-run should cost the
current step, not the run — so the pipeline is durable, resumable and cancellable by design.

Durable execution is [DBOS](https://docs.dbos.dev) on the same SQLite file. Three pipeline-shaped
workflows share one machinery: `import_document` (convert, then pre-warm the cache),
`ensure_embedding` (chunk and embed one document under one set of parameters, deduplicated by the
cache id so it runs once however many callers wait on it) and `index_collection_document` (ensure
the embedding, then write it into one collection's table). Each plans its stage into micro-batches
of `pipeline.batch_pages` pages and cuts convert and embed into at most
`pipeline.document_parallelism` contiguous slices (0 = as many as that stage's share of the CPU
budget). Every slice is one child `stage_parts` workflow with a durable step per micro-batch — so
one large document spreads over the available slots, while a document still costs a handful of
workflows rather than one per micro-batch. The index stage is one `stage_parts` child on the
collection's partition, so LanceDB has one writer per collection and the full-text index is
rebuilt once, as that child's last step.

Queues are named for what they carry: a `job.*` queue holds coarse jobs, which are made of tasks
and mostly wait on them, and a `task.*` queue holds the work itself.

| queue | concurrency | runs |
| --- | --- | --- |
| `job.indexing` | twice `pipeline.cpu_budget`, capped at 64 | one import or collection-index orchestrator per document |
| `job.embedding` | same | `ensure_embedding`, one per cache id (its own queue: the orchestrators wait on it) |
| `job.collection` | 2 | "index all", collection delete, document delete |
| `job.downloads` | 2 | model downloads (`ensure_model`) |
| `job.maintenance` | 4 | debounced maintenance, hourly archive, nightly housekeeping |
| `task.converting` | `converting_weight` share of `cpu_budget` | convert slices |
| `task.embedding` | `embedding_weight` share of `cpu_budget` | embed slices |
| `task.indexing` | `indexing_weight` share of `cpu_budget`, 1 per collection | index children, maintenance, removals |

`pipeline.cpu_budget` (default: half the machine's cores) is how many tasks run at the same time,
everywhere — so indexing a collection does not take the machine you are working on. It is shared
out over the three stage weights, largest remainder first and never below one slot per stage,
because the stages cost different things: converting is CPU and IO per page, every embedding task
loads the embedding model, and indexing writes to LanceDB, which takes one writer per collection.
A slow stage therefore backs up on its own queue instead of taking every slot from the others. The
weights only decide the mix; a process-wide semaphore of `cpu_budget` slots, taken around the CPU
work of every task and every maintenance run, is what keeps the total within the budget when the
per-stage floors or a maintenance run would push it over. Each child runs under
`pipeline.task_timeout_seconds` (default 600) per micro-batch it was given.

DBOS provides crash recovery (a restarted workflow resumes at its first unfinished step or child),
deduplication (one active import per document, one active index per membership, one embedding run
per cache id), cancellation, and the job history shown in the Jobs view. Retries follow the cause:
a transient failure (busy database, slow file) is retried 3 times with backoff, a permanent one
(unsupported file type, pages that need OCR, a document the parser cannot read) fails the document
immediately, and a model download gets 5 attempts. Removing a document from a collection cancels
its index workflow there and waits, then deletes its rows and membership from that collection's
own partition; deleting a document does that in every collection it is in (one child per
collection, each on its partition) before dropping its folder and row, and sets the document
`deleting` first so nothing attaches it meanwhile. The DBOS application version is pinned to the
package version, so restarts (including `--reload` after code edits) recover in-flight work;
anything recorded under another version is re-enqueued at startup.

Model downloads (embedding profile, user and per-collection reranker) are `ensure_model` workflows
with one durable id per model (`dl:{kind}:{model}`): idempotent, retried, and their DBOS status is
the model status shown in the nav bar. The record outlives the process, because the files do — a
restart reuses both instead of downloading the model again. Being on disk is not the same as being
usable, though: a model lives in the caches of one process, so a boot that finds a finished
download warms it in a background task (a local read, no network) and only then reports it ready.
Searches that need a model still downloading or still warming fail fast with a clear message
naming the job.

### Jobs you can watch

Long work you cannot see is work you do not trust, so every background job is visible and
cancellable.

Every kind of background work is one listing with one row shape, so the Jobs view is a section per
kind, each paged on its own: `document` (an import, an embedding run or a collection index, with
its micro-batches and its cancel), `collection` ("index all", collection delete and document
delete, with the progress the bulk index publishes), `download` (`ensure_model`, plus whether the
model is loaded in this process), `maintenance` (`maintain_on_partition` runs and the nightly
housekeeping) and `archive` (the hourly retention round). `GET
/api/jobs/by-kind?kind=&collection=&limit=&cursor=` serves one page of one kind and
`GET /api/jobs/kinds` the sections themselves, with how many jobs of each kind are running right
now (one grouped query). `GET /api/jobs/activity` is the indicator in the top-right of every view:
jobs (`job.*` queues) and tasks (`task.*` queues) queued and running, one grouped query over the
queue name prefix; a debounced maintenance run counts as queued until its delay expires. The
collection filter is an id prefix the database applies, for every kind whose id carries a
collection (`idx-col:`, `bulk-index:`, `bulk-delete:`, `maint:`); imports and embedding runs
belong to no collection. `GET /api/jobs` remains the document listing on its own, and it is the
only kind that also reads the day partitions `archive` copied finished jobs into.

### Runtime

The UI stays responsive while the machine indexes, which takes some care: every IO is awaited and
every piece of CPU work is held to a budget.

Every touch of the database, the index and the filesystem is awaited, and every Litestar handler is
`async def`. SQLite goes through `aiosqlite`, one connection per unit of work and no pool: a
connection is a thread, a unit of work is one transaction, and a connection that never outlives its
unit is never shared. LanceDB goes through its async API (`lancedb.connect_async`), which is the
Rust runtime the sync API wraps anyway. Files go through `anyio.Path` and `anyio.open_file`, with
`os.replace` and `shutil.rmtree` in a worker thread because they have no async form. CPU work is not
IO and stays sync: pdf parsing, chunking, ONNX embedding and reranking, the preview page cut and
every model load run in a worker thread through `cpu.on_cpu`, which holds one slot of the
`pipeline.cpu_budget` semaphore for the length of that work. A pipeline step therefore holds a slot
for its CPU part only, never for the file IO or the LanceDB commit around it.

One piece of CPU work still reaches the loops: ONNX Runtime holds the GIL for the whole build of a
session, and the CoreML provider compiles the model inside that build, so every request freezes for
as long as the build takes (half a second for a small reranker, several seconds for a large one).
Two things bound it: CoreML keeps its compiled models under `cache/models`, so a model is compiled
once per machine rather than once per boot, and cross-encoders always build on the CPU provider,
whatever the accelerator setting says, because they score at most `candidates` texts per search,
which CPU does in milliseconds, and their CoreML build cost seconds of frozen UI.

Two event loops run in the process: Litestar's, which serves requests, and DBOS's background loop,
which runs the queued async workflows and their steps. No loop-bound primitive is shared between
them — that is why the CPU budget is a `threading` semaphore rather than an `anyio.CapacityLimiter`,
why each loop gets a thread limiter of its own, and why the search fan-out builds its semaphore per
call. Blocking file IO survives in four places, each documented as running in a worker thread and
nowhere else: `convert.py` (the parsers take a path and read it themselves), `home.atomic_write_sync`
and the inner function of `home.remove_tree`, the parquet reads and writes of `embed_cache.py`
(pyarrow is sync), and `db._migrate_sync` (the migration scripts and the one-time WAL switch, on a
stdlib connection, before anything else holds the file open).
