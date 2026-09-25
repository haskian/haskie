# Changelog
All notable changes to this project will be documented in this file. See [conventional commits](https://www.conventionalcommits.org/) for commit guidelines.

- - -
## v0.10.1 - 2026-09-25
#### Build system
- (**dev**) ~/haskie-dev home for mise tasks and a haskie-dev command (#19) - (1c87745) - Črtomir Majer

- - -

## v0.10.0 - 2026-09-25
#### Features
- <span style="background-color: #d73a49; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 0.85em;">BREAKING</span>GGUF embedders on the Apple GPU and one hardware rule per model (#18) - (8a14a11) - Črtomir Majer

- - -

## v0.9.2 - 2026-09-25
#### Bug Fixes
- bound every stage of shutdown so Ctrl-C and stop always end the server (#17) - (4551ca0) - Črtomir Majer

- - -

## v0.9.1 - 2026-09-25
#### Refactoring
- (**db**) SQLAlchemy Core tables as the one schema source (#16) - (6efeb62) - Črtomir Majer

- - -

## v0.9.0 - 2026-09-25
#### Features
- <span style="background-color: #d73a49; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 0.85em;">BREAKING</span>model catalogue in SQLite and also_in as a tree of places (#15) - (c8ed29c) - Črtomir Majer

- - -

## v0.8.0 - 2026-09-25
#### Features
- <span style="background-color: #d73a49; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 0.85em;">BREAKING</span>serve on port 8451, rewrite README and add docs/ (#14) - (36f87b9) - Črtomir Majer

- - -

## v0.7.0 - 2026-09-24
#### Features
- <span style="background-color: #d73a49; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 0.85em;">BREAKING</span>fold near-duplicate results, model catalogue and search timing (#13) - (684d634) - Črtomir Majer

- - -

## v0.6.0 - 2026-09-24
#### Features
- <span style="background-color: #d73a49; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 0.85em;">BREAKING</span>structure-aware chunking with typed pieces and heading frames (#12) - (7e761c5) - Črtomir Majer

- - -

## v0.5.0 - 2026-09-23
#### Features
- <span style="background-color: #d73a49; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 0.85em;">BREAKING</span>search as a pipeline, passages read by seeking (#11) - (5f9ff46) - Črtomir Majer

- - -

## v0.4.0 - 2026-09-23
#### Features
- an always-loaded rule for the sources, and a clearer match modal (#10) - (53d1709) - Črtomir Majer

- - -

## v0.3.0 - 2026-09-22
#### Features
- <span style="background-color: #d73a49; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 0.85em;">BREAKING</span>search excerpts and sources over MCP, packages by feature (#9) - (3e138cc) - Črtomir Majer

- - -

## v0.2.0 - 2026-09-21
#### Features
- (**cli**) announce the conversation id, and stop a running server (#7) - (d830c93) - Črtomir Majer

- - -

## v0.1.0 - 2026-09-21
#### Features
- sessions that remember, operations with stages, and a web UI on a design system (#6) - (5f7d292) - Črtomir Majer
- install haskie into Claude Code, with a skill and a home lock (#2) - (fefa0cb) - Črtomir Majer
- collections with first-class documents and a shared embedding cache (#3) - (c94d73b) - Črtomir Majer
- CI/CD with wheel+web release and conventional-commit versioning (#1) - (5f03a9b) - Črtomir Majer
- haskie — document library with search, web UI and MCP server - (31d9760) - crtomir.majer
#### Refactoring
- (**cli**) move Claude Code's config into claude.py, and cut session-start cost (#4) - (ee476f0) - Črtomir Majer
- simplify codebase across CLI, API, pipeline, storage, jobs, and frontend (#5) - (72995a5) - Črtomir Majer
#### Miscellaneous Chores
- (**version**) 0.1.0 [skip ci] - (66cc3a1) - github-actions[bot]
- (**version**) 0.1.0 [skip ci] - (4aabe88) - github-actions[bot]
- (**version**) 0.1.0 [skip ci] - (204408c) - github-actions[bot]
- (**version**) 0.1.0 [skip ci] - (632bb38) - github-actions[bot]

- - -

## 0.1.0 - 2026-09-19
#### Features
- install haskie into Claude Code, with a skill and a home lock (#2) - (fefa0cb) - Črtomir Majer
- collections with first-class documents and a shared embedding cache (#3) - (c94d73b) - Črtomir Majer
- CI/CD with wheel+web release and conventional-commit versioning (#1) - (5f03a9b) - Črtomir Majer
- haskie — document library with search, web UI and MCP server - (31d9760) - crtomir.majer
#### Refactoring
- (**cli**) move Claude Code's config into claude.py, and cut session-start cost (#4) - (ee476f0) - Črtomir Majer
#### Miscellaneous Chores
- (**version**) 0.1.0 [skip ci] - (4aabe88) - github-actions[bot]
- (**version**) 0.1.0 [skip ci] - (204408c) - github-actions[bot]
- (**version**) 0.1.0 [skip ci] - (632bb38) - github-actions[bot]

- - -

## 0.1.0 - 2026-09-19
#### Features
- install haskie into Claude Code, with a skill and a home lock (#2) - (fefa0cb) - Črtomir Majer
- collections with first-class documents and a shared embedding cache (#3) - (c94d73b) - Črtomir Majer
- CI/CD with wheel+web release and conventional-commit versioning (#1) - (5f03a9b) - Črtomir Majer
- haskie — document library with search, web UI and MCP server - (31d9760) - crtomir.majer
#### Miscellaneous Chores
- (**version**) 0.1.0 [skip ci] - (204408c) - github-actions[bot]
- (**version**) 0.1.0 [skip ci] - (632bb38) - github-actions[bot]

- - -

## 0.1.0 - 2026-09-19
#### Features
- collections with first-class documents and a shared embedding cache (#3) - (c94d73b) - Črtomir Majer
- CI/CD with wheel+web release and conventional-commit versioning (#1) - (5f03a9b) - Črtomir Majer
- haskie — document library with search, web UI and MCP server - (31d9760) - crtomir.majer
#### Miscellaneous Chores
- (**version**) 0.1.0 [skip ci] - (632bb38) - github-actions[bot]

- - -

## 0.1.0 - 2026-09-19
#### Features
- CI/CD with wheel+web release and conventional-commit versioning (#1) - (5f03a9b) - Črtomir Majer
- haskie — document library with search, web UI and MCP server - (31d9760) - crtomir.majer

- - -

Changelog generated by [cocogitto](https://github.com/cocogitto/cocogitto).