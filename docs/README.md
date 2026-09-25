# haskie docs

How haskie works, one topic per page. Each page has diagrams and names the code it describes.
Start with the architecture, then follow a document from import to a cited answer.

| page | what it covers |
| --- | --- |
| [Architecture](architecture.md) | the process, the packages, the home lock, startup |
| [Documents and collections](documents-and-collections.md) | the two core entities, their lifecycles, the embedding cache |
| [Indexing](indexing.md) | the durable DBOS pipeline, queues, the CPU budget, recovery |
| [Chunking](chunking.md) | Structure-Aware Chunking: settings, steps, and every cut rule |
| [Search](search.md) | from query to excerpts: ranking, the four answers, folding repeats |
| [Storage](storage.md) | the home directory, the metadata database, schema changes |
| [REST API](rest-api.md) | routes, intake, errors, paging |
| [MCP and Claude Code](mcp.md) | the tools, what the installer adds, a session end to end |
| [Runtime](runtime.md) | async IO, CPU work in threads, two event loops, model loads |

For setup and the development commands, see [CONTRIBUTING.md](../CONTRIBUTING.md).
