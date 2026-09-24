# Architecture

haskie is one Python process. It serves a web UI, a REST API and an MCP server from the same
Litestar app, and keeps everything under one home directory.

```mermaid
flowchart TB
    ui["Web UI<br/>(React)"] -- REST --> app
    agent["Claude Code or<br/>any MCP client"] -- "MCP over HTTP" --> app
    scripts["Scripts"] -- REST --> app

    subgraph process["haskie process"]
        app["<b>app.py</b><br/>request context, error mapping, home lock"]
        api["<b>api/</b><br/>one module per feature;<br/>handlers marked mcp_tool=<br/>are also MCP tools"]
        app --> api
        api --> document["<b>document/</b><br/>convert, store, render"]
        api --> collection["<b>collection/</b><br/>membership, LanceDB index"]
        api --> search["<b>search/</b><br/>retrieval, passages,<br/>fold repeats, sessions"]
        api --> indexing["<b>indexing/</b><br/>DBOS pipeline: chunk,<br/>embed, cache, write"]
    end

    subgraph home["~/.haskie"]
        sqlite[("haskie.db<br/>SQLite: metadata + DBOS")]
        lance[("LanceDB table<br/>per collection")]
        files[("documents/<br/>plain files")]
    end

    document --> files
    document --> sqlite
    collection --> lance
    collection --> sqlite
    indexing --> sqlite
    indexing --> lance
    indexing --> files
    search --> lance
    search --> sqlite
    search --> files
```

The code is packaged by feature. The HTTP layer is thin. Handlers parse the request, call the domain
and return `msgspec.Struct`s, or a file or stream for the document views. The domain raises typed
errors from `errors.py`, and `app.py` maps each one to its status code.

## Startup

```mermaid
sequenceDiagram
    participant CLI as haskie run
    participant App as Litestar app
    participant Home as ~/.haskie
    participant DBOS
    CLI->>Home: home_holder (is another server running?)
    Note over CLI,Home: haskie run refuses here, naming the holder
    CLI->>App: start on 127.0.0.1:8451 (default)
    App->>Home: claim_home (create folders, exclusive lock)
    Note over App,Home: any other ASGI server, or a race, stops here
    App->>Home: ensure_home
    App->>DBOS: workflows.start
    DBOS->>Home: migrate (check schema version, WAL on a new file)
    Note over DBOS,Home: a home at another schema version is refused here
    DBOS->>DBOS: start queues, recover unfinished workflows
```

`haskie run` checks the lock before it starts, and the app takes it as its first startup step, so
any ASGI server that runs the app is held to it too. A home is one SQLite file and one set of
queues, and two servers would take each other's work.

## Where to read next

- [Documents and collections](documents-and-collections.md): the two core entities
- [Indexing](indexing.md): the durable pipeline
- [Search](search.md): from query to cited excerpts
- [Runtime](runtime.md): async IO, the CPU budget, two event loops
