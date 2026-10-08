# haskie docs

Guides for using haskie and understanding its design. Start with setup for your first library,
or read the rationale for the product choices and their research sources.

## Using haskie

| page | what it covers |
| --- | --- |
| [Why haskie exists](why-haskie.md) | trusted sources, personal taste, focused context, human research models and the evidence behind the design |
| [Setup and operation](setup.md) | installation, hardware, models, agent connections, commands and upgrades |
| [User guide](user-guide.md) | importing, collections, previews, Explore, activity, gap replay and supported formats |
| [MCP, Codex and Claude Code](mcp.md) | installation details, tool capabilities, scopes, citations and a session end to end |

## How it works

Follow a document from import to a cited answer. These pages describe the implementation and
link its design choices to research where applicable.

| page | what it covers |
| --- | --- |
| [Architecture](architecture.md) | the process, the packages, the home lock, startup |
| [Documents and collections](documents-and-collections.md) | the two core entities, their lifecycles, the embedding cache |
| [Indexing](indexing.md) | the durable DBOS pipeline, queues, the CPU budget, recovery |
| [Chunking](chunking.md) | Structure-Aware Chunking: settings, steps, and every cut rule |
| [Search](search.md) | from query to excerpts: ranking, the four answers, the map of sections, folding repeats |
| [Gaps](gaps.md) | the search log, which searches found no answer, and how the bars were measured |
| [Storage](storage.md) | the home directory, the metadata database, schema changes |
| [REST API](rest-api.md) | routes, intake, errors, paging |
| [Runtime](runtime.md) | async IO, CPU work in threads, two event loops, model loads |

For development setup and commands, see [CONTRIBUTING.md](../CONTRIBUTING.md).
