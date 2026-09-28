# MCP and Claude Code

haskie serves the Model Context Protocol (MCP) at `http://127.0.0.1:8451/mcp`, over HTTP. The tools
are the REST handlers marked `mcp_tool=`, so both surfaces share one contract. The skill that
`haskie install claude` writes is the full tool reference for agents:
[`SKILL.md`](../src/haskie/claude_code/skills/haskie/SKILL.md).

## Tools

| group | tools |
| --- | --- |
| search | `search_excerpts`, `search_sources`, `set_session_collections` |
| catalogue | `list_collections`, `get_collection`, `list_collection_documents`, `list_documents`, `get_document` |
| write | `add_document`, `add_document_to_collection`, `remove_document_from_collection`, `describe_document` |
| log and gaps | `list_searches`, `list_gaps`, `replay_gaps`, `review_gaps` ([Gaps](gaps.md)) |

Everything else stays in the web UI and the REST API: managing collections, re-indexing,
deleting or re-importing documents, operations and settings.

## What `haskie install claude` adds

```mermaid
flowchart LR
    install["haskie install claude"] --> mcp["MCP entry<br/>claude mcp add --transport http"]
    install --> skill["skill<br/>.claude/skills/haskie/SKILL.md"]
    install --> rule["rule<br/>.claude/rules/haskie.md"]
    install --> hook["SessionStart hook<br/>in .claude/settings.json:<br/>haskie ensure --no-wait"]
    mcp --> tools["Claude has the tools"]
    skill --> how["Claude knows how:<br/>which tool, which fields to cite"]
    rule --> when["Claude knows when:<br/>search first, in every session"]
    hook --> up["server starting or up,<br/>session id is known"]
```

The files go under `~/.claude` with `--scope user` (the default), or `./.claude` of the current
directory with `--scope project`. `--url` points them at another endpoint. The hook's full command
is the absolute path of `haskie` with `ensure --home <home> --url <url> --no-wait`. The skill's
trigger and the rule both name the home's collections, so they fire on the topics you collected.
Run `haskie install claude` again after adding a collection to refresh them.

## A session, end to end

```mermaid
sequenceDiagram
    participant CC as Claude Code
    participant Hook as haskie ensure
    participant H as haskie server
    CC->>Hook: SessionStart (JSON with session_id on stdin)
    Hook-->>CC: prints the session id announcement
    Hook->>H: probe 127.0.0.1:8451
    alt nothing serving
        Hook->>H: start detached haskie run
        Hook-->>CC: prints "starting haskie on ..."
    else already serving
        Hook-->>CC: prints "haskie is already serving ..."
    end
    Note over CC: the rule says: search the collections first
    CC->>H: search_excerpts(q, session_id)
    H-->>CC: excerpts with header, location, spans (each with also_in), uncovered, missing_terms
    opt nothing relevant, or "which documents?"
        CC->>H: search_sources(q, session_id)
        H-->>CC: documents, sections, collection cover
        CC->>H: set_session_collections(session_id, cover)
        CC->>H: search_excerpts(q, session_id)
    end
    CC->>CC: answer, citing header and location
```

The hook's output is the only way the conversation's id reaches the tools, because no MCP call
carries it. With the id, the Sessions view shows the conversation's last 100 events: searches,
imports, attaches, detaches, descriptions and collection choices. Each operation it started
names the conversation as its origin. Every search is also written to the search log, with or
without an id, and the Gaps page reads it ([Gaps](gaps.md)).

The hook does not wait for the server, and Claude Code connects to MCP while the hook still runs.
So a session that starts while nothing is serving, such as the first one after a reboot, has no
haskie tools. Sessions that start once the server is up do. `haskie install claude` itself waits
for the server, so the session right after installing has them. Keep `haskie run` open if the
first session matters.

## Why HTTP, not stdio

One server serves the web UI, the REST API and every MCP client at once. litestar-mcp speaks MCP
`2026-07-28`, which replaced `initialize` with `server/discover`. A stdio client cannot connect,
and an HTTP client that opens with an older `initialize` request is refused.

Code: `claude.py`, `cli.py` (`ensure`, `install claude`), `src/haskie/claude_code/`.
