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
mise run clean-run  # build, destroy ~/.haskie (asks first), reinstall, run on the clean home
mise run dist    # build, then the wheel and sdist into dist/
mise run smoke   # install the built wheel in a fresh venv, check the CLI and bundled UI
mise run bump    # version bump from the commits, patch when none implies one
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

## Commit subjects and the version

Merges to `main` are squashed, so the pull request title becomes the one commit subject on `main`,
and that subject decides the next version. `pr-title.yml` checks the title; `mise run bump` reads
the subjects since the last tag and picks the increment.

Write `<type>[(<scope>)][!]: <subject>`. The type is one of `majorfeat`, `feat`, `fix`, `perf`,
`refactor`, `docs`, `test`, `build`, `ci`, `chore`, `style`, `revert`.

Three rules decide the increment, over every commit subject since the last tag. The highest rule
any one subject matches wins:

| subject matches | increment | 1.4.2 becomes |
| --- | --- | --- |
| `majorfeat!:` or `majorfeat(<scope>)!:` | major | 2.0.0 |
| `feat:`, `feat(<scope>):`, `feat!:` or `feat(<scope>)!:` | minor | 1.5.0 |
| anything else, including `fix:`, `docs:` and `chore:` | patch | 1.4.3 |

Two consequences, both deliberate:

- **`majorfeat!` is the only way to a major version.** A `!` on any other type marks a breaking
  change for the changelog and for whoever reads the log; it no longer takes the major digit up on
  its own. Write `majorfeat!` when the release should be a major one, and nothing else.
- **`majorfeat` without the `!` fails the bump.** The subject asks for a major and the `!` is what
  grants it, so releasing that push as a patch would ship the wrong version under the right words.
  `mise run bump` stops instead, and the fix is to correct the subject on `main`.

No version is skipped: every push to `main` is released, and one whose subjects imply no increment
is a patch.
