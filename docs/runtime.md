# Runtime

The UI stays responsive while the machine indexes. Two rules make that work: every IO is awaited,
and every piece of CPU work holds a slot of one budget.

```mermaid
flowchart TB
    subgraph process["one process"]
        subgraph litestar["Litestar event loop"]
            req["requests"]
        end
        subgraph dbos["DBOS event loop"]
            wf["queued workflows<br/>and their steps"]
        end
        pool["worker threads"]
        sem{{"cpu_budget semaphore<br/>(threading)"}}
    end
    req -- "await" --> io["SQLite (aiosqlite),<br/>LanceDB (async API),<br/>files (anyio)"]
    wf -- "await" --> io
    req -- "cpu.on_cpu" --> sem
    wf -- "cpu.on_cpu" --> sem
    sem --> pool
    pool --> cpu["chunk, embed, rerank,<br/>previews, load a model"]
    pool -- "off_interpreter" --> procs["process pool:<br/>PDF page conversion, PDF previews"]
```

- **IO is async.** Every handler is `async def`. SQLite goes through SQLAlchemy Core's async engine
  on `aiosqlite`, one connection per unit of work. LanceDB goes through its async API. Files go
  through `anyio`, with `os.replace` and `shutil.rmtree` in a worker thread because they have no
  async form.
- **CPU work is sync, in a thread.** `cpu.on_cpu` runs it in a worker thread and holds one slot of
  the `pipeline.cpu_budget` semaphore for as long as it runs. PDF page conversion and a PDF's
  preview go one step further, to a process pool (`cpu.off_interpreter`), while the thread holds the
  slot. A pipeline step holds a slot for its CPU part only, never for the IO around it. The pool is
  pebble's: a parser that crashes its worker fails only its own call, and a new worker takes its
  place. One exception runs on the event loop: stemming a search's answer (`probe.vocabulary`). It
  remembers the last 32,768 words it stemmed (`probe.stem`), so it costs well under a millisecond
  once a server has seen the words, and waiting for a slot that indexing holds would cost more.
  Another runs outside our threads: maintenance has LanceDB train a vector index on its own
  runtime, where no slot can be held (`maintenance.run`).
- **A budget change applies to running work.** A resize counts the slots already held, so a
  raise from 2 to 3 admits one more piece of work, not three. The process pool has one worker per
  slot. A pool of the old size takes no new work, finishes what it took, and exits. The next
  extraction builds a pool of the new size.
- **Two loops, nothing shared.** Litestar and DBOS each run an event loop. They share no
  loop-bound primitive, so the budget is a `threading` semaphore, each loop has its own thread
  limiter, and the search fan-out builds its semaphore per call.

## Where blocking IO remains

Sync IO runs in worker threads wherever a library has no async form. The main cases:

- `document/convert.py`: the parsers take a path and read it themselves.
- `home.atomic_write_sync`, `home.remove_tree`, and the move or copy of an imported original.
- `indexing/embed_cache.py`: pyarrow reads and writes parquet synchronously.
- Reading line windows of a document's markdown, for search and the line view.
- DBOS launch, destroy and garbage collection, through sync SQLAlchemy.
- `db._migrate_sync`: the schema script and the one-time WAL switch, before anything else opens
  the file.

## Model loads

ONNX Runtime holds the global interpreter lock (GIL) while it builds a session. Every request
waits for as long as the build takes, which can be seconds. haskie runs its ONNX models itself
(`indexing/onnx_models.py`) on the ONNX Runtime build each platform needs: CUDA on Linux, which
runs on an NVIDIA GPU or else the CPU, and on Apple Silicon the standard build with the WebGPU
plugin, which reaches its GPU through Metal at about twice the CPU's speed, with the same vectors. Some models also have
an MLX profile (`indexing/mlx_models.py`, the `-mlx` profiles) or a GGUF one on llama.cpp
(`indexing/gguf_models.py`, the `-gguf` profiles), faster again on that GPU. Both load a model in
under a second and release the GIL while they compute. llama.cpp's first load on a machine also
compiles its Metal shaders, once, in about 8 s. MLX keeps its streams per thread, so every MLX
load and forward pass runs on one thread of its own. CoreML runs an ONNX model only when the
hardware setting says `coreml` and the model is in `hardware.COREML_RUNS`, which no catalogue
model is today: each failed its first batch or the process. It keeps compiled models under
`cache/models`, so a model compiles once per home. Embedders and
rerankers follow the same hardware setting. `indexing/hardware.py` decides the device of each
model. A model with no device here, such as an MLX model without Apple Silicon, is refused by its
loader and left out of `/api/options`. Settings that would strand a model are rejected. Each model
in `/api/status` reports its device.

A downloaded model still has to load into the process. A boot that finds a finished download warms
it in a background task and reports it ready only after that. A search that needs a model still
downloading, warming or failed fails fast with a 503. While it downloads, the message names the
download operation.

Models come in three groups, as the status bar shows them:

- **Search**: the embedding model and a reranker. Warmed at boot, and loaded for the life of the
  server, since every search needs them at once.
- **Knowledge**: the describer (Qwen3.5-4B or Gemma-4-E2B, a setting) and the vocabulary's embedder
  (Qwen3-Embedding-0.6B), required under llm descriptors. Downloaded at boot but loaded only when
  indexing first asks for one, and freed once nobody used it for 5 minutes (`models.IDLE_SECONDS`,
  checked every 30 s). A model in use is never freed: a describe batch keeps its describer to the
  end. The next run that asks loads it again, in a second or two. The describer alone holds 2.8 to
  3.0 GB.
- **OCR**: PP-OCRv6 small, while the `ocr` setting is on (the default). Downloaded at boot
  with the others, about 31 MB, into `haskie-ocr` in the Hugging Face cache. Usable once
  downloaded: each conversion loads it in its own worker, and reads offline. A PDF or image
  conversion waits for the model while it is still on its way. A home nobody has set up yet
  downloads no model, OCR's included.

Each model shows as an icon in its group: a turning green gear while it is in memory, a white
check when it is downloaded but not loaded, a grey spinner while it downloads or loads, and a
cross when it failed. The hint lists each one's name, kind and state.

## Shutdown

Every stage of a shutdown has a bound, and each stage past the first is harder than the one
before:

1. **Requests drain.** SIGTERM or Ctrl-C stops new connections. Requests in flight get 10 seconds.
2. **The pipeline stops.** DBOS waits up to 10 seconds for running workflows, then stops waiting.
   It does not cancel them. It stops its event loop, which cancels a workflow at its next await,
   but a step in a worker thread runs on until it returns. A second SIGINT or SIGTERM during
   those seconds ends the wait at once and logs `shutdown_hurried`.
   The extraction pool closes and kills its workers at once, even in the middle of a page. An
   extraction lost this way raises `ShuttingDown`, a `BaseException`, so DBOS records no error
   and the workflow stays pending for the next boot. A hurried stop keeps the home lock until
   the process exits, because DBOS is still running workflows.
3. **The interpreter exits.** Python waits for every non-daemon thread and ignores Ctrl-C while
   it waits. CPU work in a thread cannot be interrupted, so the app gives the exit 5 seconds.
   After that it exits anyway and logs `exit_forced` with the names of the threads it left.

One Ctrl-C can reach the server more than once. The terminal signals the whole process group,
and `mise run` and `uv run` each send the signal on again. So a signal within half a second of
the last one that counted is dropped as a copy. Only a real second press hurries a shutdown.

A second Ctrl-C while requests drain skips stage 2, and stage 3 then bounds the exit. `haskie stop`
sends SIGTERM, then SIGINT, then SIGKILL, each after a wait. Operations are durable, so a forced
stop costs a recovery at the next boot, not the work.

Code: `cpu.py`, `db.py`, `home.py`, `indexing/embed.py`, `indexing/models.py`, `shutdown.py`,
`cli.py`, `indexing/workflows.py`.
