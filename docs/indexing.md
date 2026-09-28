# Indexing

Indexing a shelf of books takes minutes to hours. Closing the laptop mid-run should cost the
current step, not the run. So imports, indexing, deletes, maintenance and model downloads run as
[DBOS](https://docs.dbos.dev) workflows, recorded step by step in the same SQLite file as the rest
of the app. Model warm-ups after boot are plain background tasks.

## Operations, jobs and tasks

haskie counts work in three words:

- An **operation** is what someone asked for: import a document, index it into a collection,
  delete a collection, download a model.
- A **job** is one stage of an operation: convert, embed or index.
- A **task** is one batch of a job, and one durable step.

"Workflow" is DBOS's word. The UI does not use it, though a few API names and DBOS error
messages still carry it.

```mermaid
flowchart LR
    op["<b>operation</b><br/>import report.pdf"] --> convert["<b>job</b><br/>convert"]
    op --> embed["<b>job</b><br/>embed"]
    convert --> c1["task<br/>pages 1-10"]
    convert --> c2["task<br/>pages 11-20"]
    convert --> c3["task<br/>..."]
    embed --> e1["task<br/>part 1"]
    embed --> e2["task<br/>..."]
```

## The three workflows

```mermaid
flowchart TB
    subgraph import_document
        conv["convert slices<br/>(task.converting)"] --> warm["ensure_embedding<br/>for default settings"]
    end
    subgraph ensure_embedding
        chunk["cache hit, or chunk + embed<br/>slices (task.embedding)"] --> parquet[("parquet in cache")]
    end
    subgraph index_collection_document
        need["ensure_embedding<br/>for this collection's settings"] --> rows["write rows; build the FTS index<br/>if the table has none (task.indexing)"]
    end
    warm -.waits on.-> chunk
    need -.waits on.-> chunk
```

`ensure_embedding` always runs, and looks up the cache inside. On a hit it returns at once.

**Batches.** A PDF converts in batches of `batch_pages` pages (default 10). Any other file
converts as one batch. Embedding runs one batch per converted part. Indexing writes
`index_group_parts` parts per batch (default 50).

**Slices.** Convert and embed cut their batches into contiguous slices, at most
`document_parallelism` of them and never more than the stage's share of the CPU budget (0 means
that share). Each slice is one child workflow with one durable step per batch, so a large PDF
spreads over the free slots. The index stage is one child and is never sliced.

**One writer per table.** Every index child runs on `task.indexing`, partitioned by collection
with one slot per partition, and under a per-collection lock. So LanceDB sees one writer per
table. A detach's removal waits its turn there too. The request only queues it, and the membership
reads `removing` until it ran (see [a membership's life](documents-and-collections.md#a-memberships-life)).

## Queues

| queue | concurrency | runs |
| --- | --- | --- |
| `operation.indexing` | twice `cpu_budget`, at most 64 | one import or index orchestrator per document |
| `operation.embedding` | same | `ensure_embedding`, one per cache id |
| `operation.collection` | 2 | index all, delete a collection, delete a document |
| `operation.downloads` | 2 | model downloads |
| `operation.maintenance` | 4 | maintenance orchestrators and nightly housekeeping |
| `task.converting` | its weight's share of `cpu_budget` | convert slices |
| `task.embedding` | its weight's share of `cpu_budget` | embed slices |
| `task.indexing` | its weight's share, one per collection | index writes, compaction and index builds, removals |

Most `operation.*` workflows orchestrate and wait on `task.*` children. Some do their own work:
model downloads, deletes and the nightly housekeeping.

## The CPU budget

```mermaid
flowchart LR
    budget["<b>cpu_budget</b><br/>default: half the cores"] --> split{"split by weight,<br/>at least 1 slot each"}
    split --> cw["task.converting<br/>weight 2"]
    split --> ew["task.embedding<br/>weight 2"]
    split --> iw["task.indexing<br/>weight 1"]
    sem{{"process-wide semaphore<br/>of cpu_budget slots"}}
    cw --> sem
    ew --> sem
    sem --> cpu["parse, chunk, embed,<br/>previews, query embedding,<br/>reranking, model loads"]
```

The weights set the mix of queue slots, because the stages cost different things. Converting is
CPU and IO per page, embedding is model inference, and indexing writes to LanceDB. A slow stage
backs up on its own queue instead of taking every slot. The semaphore caps the CPU work itself.
Every piece of it holds one slot while it runs, including previews and the CPU part of a
search. Index writes and maintenance are IO and hold no slot. Lower the budget in Settings when
you need the machine back.

## Failure and recovery

- A **transient** failure (a busy database, for example) gets 3 attempts in total, with backoff.
- A **permanent** failure fails at once: a file the parser cannot read, or a PDF whose pages need
  OCR. With `skip_ocr_pages` on (the default), only a PDF where every page needs OCR fails.
  Unsupported file types are refused at import, before any operation starts.
- An embedding model that is still downloading or warming up is not a failure. The embedding
  run sleeps durably until the model is ready, before it cuts any slice. A batch that still finds
  it warming, after a restart for example, sleeps the same way. Only a model that failed to load
  ends the import in `error`.
- A slice that runs past `task_timeout_seconds` per batch is cancelled, not retried.
- A **model download** gets 5 attempts. Its durable id is `dl:{kind}:{model}`, so a restart
  reuses the files already on disk.
- On **restart**, DBOS resumes each workflow at its first unfinished step. It reads the
  recorded inputs back through `indexing/serializer.py`, which keeps each `msgspec.Struct` by
  field name. So a field added or removed since does not garble an old record. The DBOS application
  version is the package version, and DBOS runs only its own version's work. So at boot,
  `adopt_orphans` moves work recorded under another version onto this one and enqueues it again,
  each workflow on its own queue.
- **Deduplication:** one active import per document, one active index per membership, one
  embedding run per cache id. Two collections with the same chunk settings share one run. The
  run belongs to the operation that asked first. When a cancel of that operation also cancels
  the run, the other collection starts a run of its own.
- **Cancellation** works on running operations, from the Operations view or
  `DELETE /api/operations/{id}`. A cancelled import or index is marked `cancelled`.

The nightly run deletes operation history older than `retention.operation_days` (default 28).

Code: `indexing/workflows.py`, `indexing/pipeline.py`, `indexing/operations.py`,
`indexing/models.py`, `cpu.py`.
