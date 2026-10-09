# Setup and operation

Install haskie, choose its local models, then connect Codex or Claude Code. The
[user guide](user-guide.md) covers importing and curating your first collection.

## Install

Install [uv](https://docs.astral.sh/uv/). It fetches Python 3.13 if needed. On Apple Silicon,
install Apple's command-line tools first with `xcode-select --install`: llama.cpp compiles during
installation.

```sh
uv tool install haskie
haskie run
```

`haskie run` starts a background process for the web UI, REST API and Model Context Protocol
(MCP). On first run it opens the setup page at `http://127.0.0.1:8451`.

| Platform | Model execution |
| --- | --- |
| macOS on Apple Silicon | ONNX models use the Apple GPU through WebGPU. Profiles ending in `-mlx` or `-gguf` use MLX or llama.cpp. |
| Linux | ONNX models use an NVIDIA GPU with CUDA 13 and cuDNN 9 when available, otherwise the CPU. |

Settings lists profiles compatible with the selected hardware. The
[runtime guide](runtime.md#model-loads) explains model loading and memory use.

## Choose models and search settings

- **Embedding profile.** The initial selection is `granite-97m-multilingual`, about 390 MB, with
  200+ language coverage listed in the catalogue. Choose `none` for full-text search without
  embeddings. Changing the profile later requires **Index all** in each collection.
- **Search.** Hybrid search combines meaning and exact terms. Current first-run settings enable
  the `ettin-reranker-32m-v1` cross-encoder. Reranking is optional; select `none` to keep the
  retrieval order. The catalogue includes English Ettin models from roughly 17M to 150M
  parameters. A multilingual embedder does not make an English reranker multilingual.
- **Section descriptors.** `c-tf-idf` extracts distinctive terms and runs on all supported
  platforms. `llm` uses a local language model on Apple Silicon for section descriptions,
  descriptors and document summaries. Choose Gemma-4-E2B (the default) or Qwen3.5-4B; the describer is roughly
  2.8–3.0 GB, and vocabulary processing needs a separate embedding model. See
  [indexing](indexing.md#the-three-workflows) for generation and downloads.
- **Chunking.** The default uses document structure and 1,200-character chunks. Collections can
  override chunking and search settings. The [chunking guide](chunking.md#settings) explains
  the controls; the [embedding cache guide](documents-and-collections.md#the-embedding-cache)
  explains reuse.
- **CPU budget.** Background work uses half the available cores by default. Lower the budget in
  Settings to leave more capacity for other work. It applies to running work too.

Models download into a local cache and are reused. A changed model or revision may need another
download. Wait for the search models to be ready before testing a question in Explore.

## Connect agents

Run either installer or both:

```sh
haskie install codex
haskie install claude
```

Both install the MCP connection, a collection-aware skill, a search-first rule and a session
hook. In Codex, review and trust the new hook, then start a new session. Project-scoped Codex
configuration also requires trusting the project.

The default scope is the user. Add `--scope project` to install for the current project.
`CODEX_HOME` and `CLAUDE_CONFIG_DIR` select custom user configuration directories. The
[MCP guide](mcp.md) documents exact paths, instruction refresh and removal.

Run `haskie run` before the first agent session after a reboot. Session hooks start the server
without waiting, so tools may be unavailable to the session that starts it.

Other clients can use `http://127.0.0.1:8451/mcp` if they support haskie's HTTP MCP protocol.
See [protocol requirements](mcp.md#why-http-not-stdio).

## Commands and locations

| Command or option | Purpose |
| --- | --- |
| `haskie run` | Start the server in the background; reuse it when already serving this home. |
| `haskie run --foreground` | Serve in the current process for a supervisor. |
| `haskie stop` | Stop the server for the selected home. |
| `haskie version` | Show the version and home directory. |
| `haskie uninstall codex` / `haskie uninstall claude` | Remove that integration; documents and collections remain. Use the same scope as installation. |
| `haskie destroy` | Permanently delete the selected home and its contents after confirmation. Stop it first. |
| `--home` / `HASKIE_HOME` | Select the data directory; the default is `~/.haskie`. |
| `--port` / `HASKIE_PORT` | Select the server port; the default is `8451`. |
| `install codex --url` / `install claude --url` | Point an integration at the chosen MCP endpoint. |

For a custom port, set the installer's `--url` to the same endpoint. A custom home needs the
same `--home` on commands that manage it. One home can have only one server. Starting a second
process against it is refused with the lock holder's details. Repeating `haskie run` for the
already-serving home is safe.

The server has no login and binds `127.0.0.1` by default. Keep it on your own machine.
Local search does not control what the connected agent sends to its model provider.

## Recovery and upgrades

Operations shows progress, errors and cancellation. Imports and indexing resume after a crash
from their durable steps. A failed document can be re-imported after its input problem is fixed.
See [failure and recovery](indexing.md#failure-and-recovery).

Before 1.0, compatible storage upgrades run at startup. An incompatible home is refused with an
explanation. Read [storage changes](storage.md#schema-changes) and retain your original files
before choosing to reset. `haskie destroy` deletes documents, collections, indexes and settings.
After a reset, run haskie, import the files again, rebuild the collections and reinstall each
agent integration you use. [Backup and restore](storage.md#backup-and-restore) describes backups;
a backup does not bypass schema compatibility checks.
