# Install on Linux

From nothing to a first document your agent can search, in seven steps. Each step ends with a check.
On Linux, haskie installs ONNX Runtime's CUDA build by default. With an NVIDIA GPU and the CUDA
libraries, the embedding models and rerankers run on the GPU. Without them, the same install runs
them on the CPU.

You need Linux with glibc 2.28 or newer on x86-64 (Ubuntu 20.04, Debian 10 or later), or 2.34 or
newer on ARM64 (Ubuntu 22.04, Debian 12 or later).

## 1. Install the tools haskie needs

Install [uv](https://docs.astral.sh/uv/). uv fetches Python 3.13 or newer on its own if you have
none.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal, so the shell finds `uv`.

**Check:** `uv --version` prints a version.

**For the GPU** (skip it to run on the CPU): ONNX Runtime 1.30 needs the NVIDIA driver, CUDA 13 and
cuDNN 9 ([its requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)).
Install CUDA 13 and cuDNN 9 from NVIDIA's packages for your distribution, so the libraries land
where the loader finds them.

**Check:** `nvidia-smi` lists the GPU, and this lists both libraries:

```sh
ldconfig -p | grep -E 'libcublasLt\.so\.13|libcudnn\.so\.9'
```

## 2. Install haskie

```sh
uv tool install haskie
```

This downloads ONNX Runtime's CUDA build, about 200 MB.

**Check:** both lines answer.

```sh
haskie --version
"$(uv tool dir)/haskie/bin/python" -c 'from haskie.indexing import embed; print(embed.providers())'
```

The first prints `haskie` and its version. The second prints what haskie runs its ONNX models on:
`['CUDAExecutionProvider', 'CPUExecutionProvider']` with a working GPU, `['CPUExecutionProvider']`
without one. With TensorRT installed as well, `TensorrtExecutionProvider` comes first.

If the shell says `haskie: command not found`, run `uv tool update-shell` and open a new terminal.

## 3. Start haskie and finish the first run

```sh
haskie run
```

```
starting haskie on http://127.0.0.1:8451 (log: /home/you/.haskie/server.log)
haskie is serving http://127.0.0.1:8451
pick the embedding model and the search at http://127.0.0.1:8451/
```

One background process now serves the web UI, the REST API and the MCP server. It keeps running
after the command returns. `haskie run` again is safe: it finds the server and says so. `haskie
stop` stops it. For a supervisor such as systemd, `haskie run --foreground` serves in the
supervisor's own process instead.

On a desktop the browser opens on *Welcome to haskie*. On a machine you reach over SSH, forward the
port from your own computer, then open http://127.0.0.1:8451 there:

```sh
ssh -L 8451:127.0.0.1:8451 you@server
```

haskie listens on 127.0.0.1 only. Nothing in it checks who is calling, so keep it off the network.

Pick the embedding model. The default, bge-small, is English and about 130 MB. Pick a multilingual
one for other languages. Keep the default search, then press **Initialize ~/.haskie**. The models
download in the background, and the status bar shows them.

**Check:** `curl -s http://127.0.0.1:8451/api/status` shows `"initialized":true`. Its `models`
list says where each model runs: `"device":"gpu"` on CUDA, `"cpu"` otherwise.

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

- **The models run on the CPU on a machine with an NVIDIA GPU.** Run the checks of step 1 again.
  After installing the libraries, restart the server: `haskie stop`, then `haskie run`. It looks
  for CUDA once, when it starts.
- **`haskie run` says the server exited while starting.** It prints the end of
  `~/.haskie/server.log`, which says why.
- **Port 8451 is taken.** Run `haskie run --port 8461` and
  `haskie install claude --url http://127.0.0.1:8461/mcp`, or set `HASKIE_PORT=8461`.
- **A new haskie refuses the home.** A release that changes the storage format says so. Run
  `haskie destroy`, then start again at step 3.
