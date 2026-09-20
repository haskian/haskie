# AGENTS.md

haskie converts and indexes personal documents into searchable collections, and serves them
over a Litestar API, a web UI, and MCP. Python backend in `src/haskie`, tests in `tests`,
frontend in `web` (React + TypeScript, Bun).

## Commands

Run everything through `mise run <task>`, never call the underlying tool directly:

```
mise run setup   # install Python and web dependencies
mise run check   # lint, format and type-check both sides; --fix applies fixes
mise run test    # Python suite, parallel, with coverage
mise run schema  # regenerate web/src/schema.d.ts from the API; check fails on drift
mise run dev     # API on :8000 and the Vite dev server on :5173, together
mise run build   # build the web UI into src/haskie/web
mise run dist    # build, then the wheel and sdist into dist/
mise run smoke   # install the built wheel in a fresh venv, check the CLI and bundled UI
mise run bump    # conventional-commit version bump
```

Run `mise run check` and `mise run test` before treating a change as done.

## mise tasks

Every task is its own executable file under `mise/tasks/`, never an inline `[tasks.x]` entry in
`mise.toml`. `mise.toml` holds only tool versions and config. A new task means a new file in
`mise/tasks/`, with a `#MISE description="..."` comment, `set -euo pipefail`, and `#MISE
depends=[...]` instead of hand-rolled process control (background jobs, traps, wait).

## Code style

- Python 3.13, `ruff` (line length 100) for lint and format, `ty` for types. Backend: async
  Litestar handlers, `msgspec.Struct` for models, `structlog` for logging.
- Frontend: `oxlint` and `tsc` for lint/types. `web/src/schema.d.ts` is generated from the API's
  OpenAPI document — never hand-edit it, and never hand-write a type it already covers.
- Match the existing comment style: only the non-obvious why, never the what.

## Tests

`pytest`, `tests/` mirrors `src/haskie/`. Fixtures use real payloads, not placeholder dicts. New
behavior needs a test in the same file as its neighbors, not a new top-level test module unless
the area is genuinely new.
