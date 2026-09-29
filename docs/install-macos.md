# Install on macOS

From nothing to a first document your agent can search, in seven steps. Each step ends with a check.
The install puts in the Apple Silicon extras by default: `mlx` for the MLX embedding models and
rerankers, and `gguf` for the GGUF embedding models on llama.cpp. Both run on the Apple GPU.

You need an Apple Silicon Mac (M1 or later) with macOS 14 or newer.

## 1. Install the tools haskie needs

The `gguf` extra compiles llama.cpp, so install the Xcode command-line tools first. Then install
[uv](https://docs.astral.sh/uv/). uv fetches Python 3.13 on its own if you have none.

```sh
xcode-select --install                              # skip if the next line prints a path
xcode-select -p
curl -LsSf https://astral.sh/uv/install.sh | sh     # or: brew install uv
```

Open a new terminal, so the shell finds `uv`.

**Check:** `uv --version` prints a version.

## 2. Install haskie

```sh
uv tool install "haskie[mlx,gguf]"
```

The first install takes about a minute, because it compiles llama.cpp with Metal. No cmake is
needed: the build fetches its own.

**Check:** both lines answer.

```sh
haskie --version
"$(uv tool dir)/haskie/bin/python" -c 'import mlx.core as mx, llama_cpp; print(mx.default_device(), llama_cpp.llama_supports_gpu_offload())'
```

The first prints `haskie` and its version. The second ends with `Device(gpu, 0) True`: MLX and
llama.cpp both reach the GPU. llama.cpp prints a few `ggml_metal` lines before it.

If the shell says `haskie: command not found`, run `uv tool update-shell` and open a new terminal.

## 3. Start haskie and finish the first run

```sh
haskie run
```

```
starting haskie on http://127.0.0.1:8451 (log: /Users/you/.haskie/server.log)
haskie is serving http://127.0.0.1:8451
pick the embedding model and the search at http://127.0.0.1:8451/
```

One background process now serves the web UI, the REST API and the MCP server. It keeps running
after the command returns. `haskie run` again is safe: it finds the server and says so. `haskie
stop` stops it.

The browser opens on *Welcome to haskie*. Pick the embedding model. The default, bge-small, is
English and about 130 MB. Pick a multilingual one for other languages, or an MLX or GGUF one to
embed on the GPU. Keep the default search, then press **Initialize ~/.haskie**. The models download
in the background, and the status bar shows them.

**Check:** `curl -s http://127.0.0.1:8451/api/status` shows `"initialized":true`. Its `models`
list says where each model runs: `"device":"apple_silicon"` for an MLX or GGUF model, `"cpu"` for
an ONNX one such as the default.
The `coreml` hardware setting runs the ONNX ones through CoreML instead, which today is slower.

## 4. Upload your first documents

1. Open **Documents** and press **Add documents**.
2. Press **Choose files** and pick one or more files, or drop them anywhere on the page.
3. Each file gets a row. Edit the name a document will get, or remove a file with its **×**. A
   file that is already imported says so.
4. Press **Import** (**Import 3 documents** for three). Each file then shows its status until it
   is converted and embedded.

**Check:** each row drops its status (`queued`, `converting`, `embedding`) once the document is
ready, then shows its nearest documents. The Documents page lists them under *Imported*.

## 5. Put them in a collection

Your agent searches collections, not loose documents.

1. Open **Collections** and press **New collection**.
2. Give it a name and a one-line description, such as *coffee* and *Roasting and brewing notes*.
   The agent reads the description to choose where to look.
3. In the collection, press **+** beside each document under *Available*.

## 6. Smoke test

Ask something one of your documents answers. Put your own question and collection name in.

```sh
curl -sG http://127.0.0.1:8451/api/search/excerpts \
  --data-urlencode "q=when does first crack start" \
  --data-urlencode "collections=coffee"
```

The answer holds `excerpts`. The first names its `document`, its `header` (the heading path) and
its `location` (file and lines), such as `"header":"Roasting > First crack"`. That is the same
answer your agent gets.

## 7. Connect Claude Code

```sh
haskie install claude
claude mcp list
```

`install claude` registers the MCP server and writes a skill, a rule and a SessionStart hook
([what each one does](mcp.md)). `claude mcp list` lists `haskie` as connected. Start a new Claude
Code session and ask the question from step 6. Claude searches your collection first and cites
the header and location. Run `haskie install claude` again after adding a collection, so the
skill and the rule name it.

## If something goes wrong

- **The install fails while building llama-cpp-python.** Run `xcode-select --install`, then
  `uv tool install --reinstall "haskie[mlx,gguf]"`. To skip GGUF, install `"haskie[mlx]"`.
- **`haskie run` says the server exited while starting.** It prints the end of
  `~/.haskie/server.log`, which says why.
- **Port 8451 is taken.** Run `haskie run --port 8461` and
  `haskie install claude --url http://127.0.0.1:8461/mcp`, or set `HASKIE_PORT=8461`.
- **A new haskie refuses the home.** A release that changes the storage format says so. Run
  `haskie destroy`, then start again at step 3.
