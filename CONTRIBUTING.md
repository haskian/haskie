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
mise run dev     # API and MCP on :8451 with reload, Vite on :5173
```

| task | what it does |
| --- | --- |
| `setup` | install Python and web dependencies |
| `dev` | run `api` and `web` together |
| `api` | Litestar API and MCP server on :8451, with reload |
| `web` | Vite dev server on :5173, proxying `/api` to :8451 |
| `check` | lint, format and type-check both sides (ruff, ty, oxlint, tsc), and fail if `web/src/schema.d.ts` is stale. `--fix` applies ruff's fixes and formatting |
| `test` | Python tests in parallel, with coverage |
| `schema` | regenerate `web/src/schema.d.ts` from the OpenAPI document |
| `build` | build the web UI into `src/haskie/web`, where the wheel ships it |
| `dist` | `build`, then the wheel and sdist into `dist/` |
| `smoke` | install the built wheel in a fresh venv, check the CLI and the bundled UI |
| `clean-run` | destroy `~/haskie-dev` (asks first), reinstall the fresh build, run it on a clean home |
| `install-dev` | build, then install this checkout as the `haskie-dev` command, which always uses `~/haskie-dev` |
| `bump` | version bump from the commit subjects (CI only) |

A change is done when `mise run check` and `mise run test` pass. Tests that download models carry
the `network` mark and are skipped by default.

Environment variables. Each has a working default:

| variable | default | purpose |
| --- | --- | --- |
| `HASKIE_HOME` | `~/.haskie` | the home directory, same as `--home`. `mise.toml` sets `~/haskie-dev`, so development never touches an installed haskie's data |
| `HASKIE_LOG_LEVEL` | `INFO` | level for every logger, DBOS included |
| `HASKIE_LOG_FORMAT` | `json` | `console` for readable logs |
| `HASKIE_ADDRESS` | unset | set by `run` itself, so a second start can name the server that holds the home |

Timing knobs are module constants, not variables (`workflows.OPERATION_POLL`,
`workflows.TASK_POLL`, `workflows.RETRY_INTERVAL_SECONDS`). `tests/conftest.py` shortens the polls.

## Technical decisions

| decision | why | details |
| --- | --- | --- |
| One handler serves REST and MCP (`litestar-mcp`) | One contract, one test surface. A handler marked `mcp_tool=` becomes a tool | [REST API](docs/rest-api.md) |
| MCP over HTTP, not stdio | One server serves the UI, the API and every client at once. litestar-mcp speaks MCP `2026-07-28`, which replaced `initialize` with `server/discover` | [MCP](docs/mcp.md) |
| `msgspec` for every model | Fast, strict decoding at the trust boundary. The same types generate the OpenAPI document | `api/`, `settings.py` |
| Frontend types generated from OpenAPI | One source of truth for the contract. `check` fails on drift | [REST API](docs/rest-api.md) |
| SQLite in WAL mode, through SQLAlchemy Core on `aiosqlite` | A single-user app needs no database server. One connection per unit of work, so none is ever shared. Core tables are the one source of the schema: the DDL is generated from them, and queries name columns through them | [Storage](docs/storage.md) |
| LanceDB, one table per collection | Embedded, on local disk, vector and full-text search in one table. One writer per collection keeps writes simple | [Storage](docs/storage.md) |
| DBOS on the same SQLite file | Durable, resumable, cancellable work with no broker or extra server | [Indexing](docs/indexing.md) |
| Embedding cache keyed by everything the vectors depend on | A document is chunked and embedded once per distinct setting, however many collections share it | [Documents and collections](docs/documents-and-collections.md) |
| Structure-Aware Chunking, no overlap | Chunks follow the author's sections and paragraphs. The heading path gives the context an overlap would | [Chunking](docs/chunking.md) |
| Rank fusion across collections | Scores from two indexes are not comparable. Ranks are | [Search](docs/search.md) |
| Near-duplicates folded after ranking | A pointwise reranker cannot see repeats. Folding keeps the citation and frees the slot | [Search](docs/search.md) |
| One CPU budget for all work | Indexing never takes the whole machine. The UI stays responsive | [Runtime](docs/runtime.md) |
| Exclusive lock on the home | One home is one SQLite file and one set of queues. Two servers would take each other's work | [Architecture](docs/architecture.md) |
| No migrations before 1.0 | A storage change bumps `SCHEMA_VERSION`. An older home is refused, then destroyed and imported again | [Storage](docs/storage.md) |
| Loopback by default, no auth | A home is one user's documents | `claude.py` (`DEFAULT_HOST`) |

## Pull requests

Merges to `main` are squashed, and the pull request title becomes the commit subject. That subject
decides the next version. [AGENTS.md](AGENTS.md#commit-subjects-and-the-version) has the format.
