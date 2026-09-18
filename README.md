# haskie

**Your taste, as a knowledge base your AI agents can read.**

The web is an average of everyone. Your bookshelf is not. haskie turns the papers, books, manuals
and notes you own and actually trust — or simply the documents that matter for your AI flows —
into a private, searchable library, and hands it to your AI agents over MCP, so they answer from
your sources instead of whatever ranked first today.

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

Curate a library per topic: *the three coffee-roasting books worth reading*, *our internal
architecture decisions*, *the standards that actually apply to this hardware*. Drop in PDFs,
markdown and office files. haskie converts, chunks and indexes them on your machine. Then any MCP
client — Claude Code, Claude Desktop, your own agent — searches them by meaning, by keyword, or
both.

- **Trusted sources, not search results.** You decide what goes in. Nothing else answers.
- **A library per topic.** Point an agent at the libraries that matter for the question at hand.
- **Local and private.** Your files, your machine, loopback by default. No account, no upload,
  nothing redistributed.
- **Agent-native.** MCP server and REST API from the same app, plus a web UI to curate with.
- **Nothing to assemble.** Conversion, chunking, embeddings, hybrid search, reranking and a durable
  job pipeline ship in one command.

Taste does not scale by explaining it. It scales by indexing it.

## Five minutes to a working library

```sh
uv tool install haskie      # or `uv tool install .` from a checkout
haskie init                 # create ~/.haskie, migrate its database and layout
haskie run                  # web UI, REST API and MCP server on http://127.0.0.1:8000
haskie destroy              # delete ~/.haskie and everything in it
```

Open the UI, make a library, drop files in, press index. Then point your agent at
`POST http://127.0.0.1:8000/mcp` and ask it something only your documents know.

`init` is idempotent, so it is also how an existing home is brought up to date after an upgrade;
`run` does the same work at startup, so `init` is only needed to do it first. Both take `--home`
(or `HASKIE_HOME`) to keep the data somewhere else, and `run` takes `--host`, `--port` and
`--reload`. It binds loopback by default: the home directory is one user's documents, and nothing
in the app authenticates a caller.

`destroy` asks before it deletes and prints what would be lost first. It refuses any directory
that is not a haskie home, so a mistyped `--home` cannot take the wrong tree with it; `--yes`
skips the prompt. Nothing is backed up, and a running `haskie run` should be stopped first.

The built web UI ships inside the wheel, so an installed `haskie` serves the interface and not
just the API. From a checkout, `mise run build` puts it there; without it the app still runs, API
and MCP only.

## MCP: what your agent gets

Endpoint `POST http://127.0.0.1:8000/mcp`. Tools: `list_libraries`, `list_documents`,
`add_document`, `index_document`, `set_session_libraries`, `search`, `search_library`,
`search_text`. Set the session's libraries first, then search with the same id — that is how an
agent scopes itself to *this topic's* sources for the rest of the conversation. `search_text`
needs no session and no model: one keyword query across everything, from a cold start.

## Search that earns the trust you put in the library

A curated library is only as good as the retrieval over it, so the retrieval is not an afterthought.

User-level defaults with per-library overrides (`limit`, `candidates`, `mode`, `fusion`, `rrf_k`,
`vector_weight`, `bm25_weight`); `search_library` also accepts them per call. `mode`: `hybrid`
(vector + BM25, fused), `vector`, `fts`; without an embedding profile everything is `fts`. `fusion`:
`rrf` (reciprocal rank fusion, `rrf_k`) or `linear` (weighted sum of normalized scores; the two
weights are normalized together). `reranker = cross-encoder` rescores the `candidates` of any mode
(vector, fts or hybrid) with a fastembed cross-encoder (`reranker_model`).

`search_text` (`GET /api/search/text`) is the session-free alternative: one BM25 query over every
library at once, or over the comma-separated `libraries`, with no embedding model and nothing to set
up first. Scores are raw BM25 rather than fused ranks, because one lexical scorer with the same
tokenizer and chunk size answers everywhere, so two libraries are on one scale; normalizing per
library would put every library's rank-1 chunk on page one. The result is paged (`page_size` up to
200, `next_cursor` back in as `cursor`, at most 1000 results deep): the cursor is an opaque offset
bound to the query, since a full-text query cannot be filtered by score. Every page recomputes the
ranking, so a document indexed between two pages can move a hit across a page boundary, and a
library created or deleted meanwhile invalidates the cursor (422). A library with no full-text index
yet — one in the middle of its first index — contributes nothing rather than making the whole query
wait for it.

A session search (`search`, `GET /api/search`) asks every library of the session at once: the query
is embedded once and each model is checked once for the whole fan-out, the libraries are read in
parallel (up to 8 at a time), and the per-library rankings are fused by rank (reciprocal rank fusion,
`rrf_k`), because scores from two indexes are not comparable. A session `Hit.score` is therefore an
RRF score, or the cross-encoder's when `reranker = cross-encoder` — one rerank pass over the merged
candidates. A session with a single library keeps that library's own scores, and `search_library` is
unaffected either way. A library deleted since the session chose it is skipped; a library that fails
to answer fails the search, rather than leaving a hole that reads as "no match".

Every setting has a title and definition (`msgspec.Meta` on the field; served as `/api/options` →
`docs`, shown in the UI), so tuning is done in the UI rather than by reading this file. Chunk sizes
are in characters.

## How it works

Personal document library in `~/.haskie`. Add PDFs, markdown and office files to libraries; each
file is converted to markdown (pdf-inspector page-wise for PDF, anydoc for office formats), chunked
(semantic-text-splitter) and indexed in LanceDB (one index per library). A React web UI and an MCP
server (litestar-mcp) share one Litestar app.

Document lifecycle: `uploaded` (preview only; first 10 PDF pages parsed on first open) → `queued` →
`converting` → `embedding` → `indexing` → `indexed` | `error` | `cancelled`. Indexing is explicit:
per document, per library, or via MCP `index_document`. Upload is instant and readable; the
expensive work happens when you ask for it.

### Where everything lives

Plain files in a directory you can back up, inspect or delete. No opaque store.

```
~/.haskie/
  cache/models/            compiled CoreML models (ONNX Runtime writes it; safe to delete)
  haskie.db               SQLite (aiosqlite, WAL): settings, libraries, documents, sessions, plus
                           DBOS workflow tables. Schema versioned via PRAGMA user_version.
  library/<name>/
    files/<sh>/<doc>            original upload
    markdown/<sh>/<doc>.md      full conversion (assembled from parts when indexed)
    markdown/<sh>/<doc>.parts/  NNNNNN.md per micro-batch of `pipeline.batch_pages` pages, plus
                                NNNNNN.rows.json (chunks + vectors) from the embed stage
    preview/<sh>/<doc>/         source (pdf cut to 10 pages / image / text / html) + preview.md
    index/                      LanceDB table "chunks"
```

`<sh>` is the shard directory of the document: the first byte of the SHA-1 of its name, in hex
(`layout.shard`). A library therefore spreads over 256 directories per base instead of putting
ten thousand entries in one. `layout.migrate_layout` moves a home written by an older build into
this shape once at startup, and records `meta.layout_version` when it is done.

### Indexing that survives a crash

Indexing a shelf of books takes minutes to hours. Closing the laptop mid-run should cost the
current step, not the run — so the pipeline is durable, resumable and cancellable by design.

Durable execution is [DBOS](https://docs.dbos.dev) on the same SQLite file. One `index_document`
workflow per document plans each stage into micro-batches of `pipeline.batch_pages` pages and cuts
convert and embed into at most `pipeline.document_parallelism` contiguous slices (0 = as many as
that stage's share of the CPU budget). Every slice is one child `stage_parts` workflow with a
durable step per micro-batch - so one large document spreads over the available slots, while a
document still costs a handful of workflows rather than one per micro-batch. Lower
`document_parallelism` to keep a single large document from occupying every slot while other
documents wait. The whole index stage is one `stage_parts` child on the library's partition, so
LanceDB has one writer per library and the full-text index is rebuilt once, as that child's last
step.

Queues are named for what they carry: a `job.*` queue holds coarse jobs, which are made of tasks
and mostly wait on them, and a `task.*` queue holds the work itself.

| queue | concurrency | runs |
| --- | --- | --- |
| `job.indexing` | twice `pipeline.cpu_budget`, capped at 64 | one orchestrator per document |
| `job.library` | 2 | "index all" and library delete |
| `job.downloads` | 2 | model downloads (`ensure_model`) |
| `job.maintenance` | 4 | debounced maintenance, hourly archive, nightly housekeeping |
| `task.converting` | `converting_weight` share of `cpu_budget` | convert slices |
| `task.embedding` | `embedding_weight` share of `cpu_budget` | embed slices |
| `task.indexing` | `indexing_weight` share of `cpu_budget`, 1 per library | index children, maintenance, removals |

`pipeline.cpu_budget` (default: half the machine's cores) is how many tasks run at the same time,
everywhere — so indexing a library does not take the machine you are working on. It is shared out
over the three stage weights, largest remainder first and never below one slot per stage, because
the stages cost different things: converting is CPU and IO per page, every embedding task loads the
embedding model, and indexing writes to LanceDB, which takes one writer per library. A slow stage
therefore backs up on its own queue instead of taking every slot from the others. The weights only
decide the mix; a process-wide semaphore of `cpu_budget` slots, taken around the CPU work of every
task and every maintenance run, is what keeps the total within the budget when the per-stage floors
or a maintenance run would push it over. Each child runs under `pipeline.task_timeout_seconds`
(default 600) per micro-batch it was given.

DBOS provides crash recovery (a restarted workflow resumes at its first unfinished step or child),
deduplication (one active job per document), cancellation, and the job history shown in the Jobs
view. Retries follow the cause: a transient failure (busy database, slow file) is retried 3 times
with backoff, a permanent one (unsupported file type, pages that need OCR, a document the parser
cannot read) fails the document immediately, and a model download gets 5 attempts. Deleting a
document or a library cancels its workflows and waits for them to end, then removes index rows,
files and the database row (in that order) from the library's own partition, so nothing writes a
document while it is removed. The DBOS application version is pinned to the package version, so
restarts (including `--reload` after code edits) recover in-flight work; anything recorded under
another version is re-enqueued at startup.

Model downloads (embedding profile, user and per-library reranker) are `ensure_model` workflows with
one durable id per model (`dl:{kind}:{model}`): idempotent, retried, and their DBOS status is the
model status shown in the nav bar. The record outlives the process, because the files do - a restart
reuses both instead of downloading the model again. Being on disk is not the same as being usable,
though: a model lives in the caches of one process, so a boot that finds a finished download warms
it in a background task (a local read, no network) and only then reports it ready. Searches that
need a model still downloading or still warming fail fast with a clear message naming the job.

### Jobs you can watch

Long work you cannot see is work you do not trust, so every background job is visible and
cancellable.

Every kind of background work is one listing with one row shape, so the Jobs view is a section per
kind, each paged on its own: `document` (one `index_document` pipeline, with its micro-batches and
its cancel), `library` ("index all" and library delete, with the progress the bulk index publishes),
`download` (`ensure_model`, plus whether the model is loaded in this process), `maintenance`
(`maintain_on_partition` runs and the nightly housekeeping) and `archive` (the hourly retention
round). `GET /api/jobs/by-kind?kind=&library=&limit=&cursor=` serves one page of one kind and
`GET /api/jobs/kinds` the sections themselves, with how many jobs of each kind are running right now
(one grouped query). `GET /api/jobs/activity` is the indicator in the top-right of every view: jobs
(`job.*` queues) and tasks (`task.*` queues) queued and running, one grouped query over the queue
name prefix; a debounced maintenance run counts as queued until its delay expires. The library
filter is an id prefix the database applies, for every kind whose id carries a library (`idx:`,
`bulk-index:`, `bulk-delete:`, `maint:`). `GET /api/jobs` remains the document listing on its own,
and it is the only kind that also reads the day partitions `archive` copied finished jobs into.

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
`indexing.cpu_budget` semaphore for the length of that work. A pipeline step therefore holds a slot
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
them - that is why the CPU budget is a `threading` semaphore rather than an `anyio.CapacityLimiter`,
why each loop gets a thread limiter of its own, and why the search fan-out builds its semaphore per
call. Blocking file IO survives in four places, each documented as running in a worker thread and
nowhere else: `convert.py` (the parsers take a path and read it themselves), `home.atomic_write_sync`
and the inner function of `home.remove_tree`, `layout._migrate_home` and its helpers (one burst of
`iterdir` and `os.replace`, once per home at boot), and `db._migrate_sync` (the migration scripts and
the one-time WAL switch, on a stdlib connection, before anything else holds the file open).
