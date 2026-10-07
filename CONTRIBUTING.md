# Contributing to haskie

How to build haskie, and why it is built this way. The details of each part live in
[docs/](docs/README.md), one topic per page with diagrams.

Every change serves the [mission](README.md#mission): a fast, transparent, easy-to-manage library
of trusted sources that gives agents relevant, non-redundant evidence and leaves the reasoning to
them. When two designs both work, pick the one that serves it better.

[AGENTS.md](AGENTS.md) holds the project map, the working rules and the commit subject format.
Read it first. Its rules apply to people too.

## Get started

Every command runs through [mise](https://mise.jdx.dev). Each task is one file in `mise/tasks/`.

```sh
mise run setup   # install Python and web dependencies
mise run dev     # API and MCP on :8452 with reload, Vite on :8453
```

| task | what it does |
| --- | --- |
| `setup` | install Python and web dependencies |
| `dev` | run `api` and `web` together |
| `api` | Litestar API and MCP server on :8452, with reload |
| `web` | Vite dev server on :8453, proxying `/api` to :8452 |
| `check` | lint, format and type-check both sides (ruff, ty, oxlint, tsc), and fail if `web/src/schema.d.ts` is stale. `--fix` applies ruff's fixes and formatting |
| `test` | Python tests in parallel, with coverage |
| `test-web` | web UI tests (`bun test`); `test` runs them first |
| `schema` | regenerate `web/src/schema.d.ts` from the OpenAPI document |
| `build` | build the web UI into `src/haskie/web`, where the wheel ships it |
| `dist` | `build`, then the wheel and sdist into `dist/` |
| `smoke` | install the built wheel in a fresh venv, check the CLI, and that one `haskie run` serves the web UI, the REST API and MCP |
| `clean-run` | destroy `~/haskie-dev` (asks first), reinstall the fresh build, run it on a clean home |
| `reload` | `build`, then restart the haskie serving `~/haskie-dev` on :8452, so it serves this checkout's UI and backend |
| `install-dev` | build, then install this checkout as the `haskie-dev` command, which always uses `~/haskie-dev` and port 8452 |
| `calibrate-rerankers` | `sample` writes this home's searched questions and their chunks ranked 10 to 30 to `eval/candidates.jsonl`; after you copy one borderline chunk a question into `eval/borderline.jsonl`, `measure --model NAME [--write]` sets each reranker's floor and score curve (`haskie.catalogue.calibrate`) |
| `calibrate-gaps` | `sample` writes this home's logged questions with their best cosines and near misses to `eval/gap-questions.jsonl`; after you mark each `answered` true or false, `measure --profile NAME [--write]` sets its `weak_match` and `answered_match` (`haskie.catalogue.calibrate_gaps`) |
| `evaluate-gaps` | scores every gap signal by AUROC on labelled questions over the Rust book (fetched at a pinned commit) and haskie's docs, and checks the current bars (`tests/gapeval/`); `--profile`, `--reranker`, `--out` |
| `bump` | version bump from the commit subjects (CI only) |

A change is done when `mise run check` and `mise run test` pass. Tests that download models carry
the `network` mark and are skipped by default.

Environment variables. Each has a working default:

| variable | default | purpose |
| --- | --- | --- |
| `HASKIE_HOME` | `~/.haskie` | the home directory, same as `--home`. `mise.toml` sets `~/haskie-dev`, so development never touches an installed haskie's data |
| `HASKIE_PORT` | `8451` | the default port of `run`, `install claude` and `install codex`. `mise.toml` sets `8452`, so development serves beside the installed haskie |
| `HASKIE_LOG_LEVEL` | `INFO` | level for every logger, DBOS included |
| `HASKIE_LOG_FORMAT` | `json` | `console` for readable logs |
| `HASKIE_ADDRESS` | unset | set by `run` itself, so a second start can name the server that holds the home, and the app can serve the host it was bound to |
| `HASKIE_ALLOWED_ORIGINS` | unset | comma-separated browser origins trusted beside haskie's own UI, for an agent UI that runs in a browser. Clients that are not browsers send no `Origin` and need nothing here |

Timing knobs are module constants, not variables (`workflows.OPERATION_POLL`,
`workflows.TASK_POLL`, `workflows.RETRY_INTERVAL_SECONDS`). `tests/conftest.py` shortens the polls.

## Technical decisions

| decision | why | details |
| --- | --- | --- |
| One handler serves REST and MCP (`litestar-mcp`) | One contract, one test surface. A handler marked `mcp_tool=` becomes a tool. The search tools are twins that answer with fewer fields (`api/agent.py`): what an agent reads costs tokens | [REST API](docs/rest-api.md) |
| MCP over HTTP, not stdio | One server serves the UI, the API and every client at once. litestar-mcp speaks MCP `2026-07-28`, which replaced `initialize` with `server/discover` | [MCP](docs/mcp.md) |
| `msgspec` for every model | Fast, strict decoding at the trust boundary. The same types generate the OpenAPI document | `api/`, `settings.py` |
| Frontend types generated from OpenAPI | One source of truth for the contract. `check` fails on drift | [REST API](docs/rest-api.md) |
| SQLite in WAL mode, through SQLAlchemy Core on `aiosqlite` | A single-user app needs no database server. One connection per unit of work, so none is ever shared. Core tables are the one source of the schema: the DDL is generated from them, and queries name columns through them | [Storage](docs/storage.md) |
| LanceDB, one table per collection | Embedded, on local disk, vector and full-text search in one table. One writer per collection keeps writes simple | [Storage](docs/storage.md) |
| DBOS on the same SQLite file | Durable, resumable, cancellable work with no broker or extra server | [Indexing](docs/indexing.md) |
| Model catalogue in SQLite, loaders in code | A model card edit is a row, not a release. A row cannot add reviewed code, so loaders and their pinned revisions stay in `indexing/`. A test keeps the two in step. The descriptor generator (`gguf_models.GENERATORS`) has no row: nobody picks it, one setting names one model, and its facts sit in that setting's text | [Storage](docs/storage.md) |
| Embedding cache keyed by everything the vectors depend on | A document is chunked and embedded once per distinct setting, however many collections share it | [Documents and collections](docs/documents-and-collections.md) |
| Structure-Aware Chunking, no overlap | Chunks follow the author's sections and paragraphs. The heading path gives the context an overlap would | [Chunking](docs/chunking.md) |
| Several collections ranked as one table | Each retriever (vector, BM25) is ranked over all of them, then fused. Fusing one ranking per collection gave each an equal share of the scan | [Search](docs/search.md) |
| Near-duplicates folded after ranking | A pointwise reranker cannot see repeats. Folding keeps the citation and frees the slot | [Search](docs/search.md) |
| One CPU budget for all work | Indexing never takes the whole machine. The UI stays responsive | [Runtime](docs/runtime.md) |
| Exclusive lock on the home | One home is one SQLite file and one set of queues. Two servers would take each other's work | [Architecture](docs/architecture.md) |
| Additive upgrades only before 1.0 | A storage change bumps `SCHEMA_VERSION`. An additive one ships its upgrade in `db.UPGRADES`. Any other leaves an older home refused, then destroyed and imported again | [Storage](docs/storage.md) |
| Loopback by default, no auth | A home is one user's documents | `claude.py` (`DEFAULT_HOST`) |
| Host and Origin checked on every request | A browser ignores the loopback bind: any page could post to the API, and a DNS-rebinding page could read it. Agents send no `Origin`, so they pass | `app.py` (`guard_callers`) |

## Pull requests

Merges to `main` are squashed, and the pull request title becomes the commit subject. That subject
decides the next version. [AGENTS.md](AGENTS.md#commit-subjects-and-the-version) has the format.
