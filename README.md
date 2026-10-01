<p align="center">
  <img src="design/haskie-logo.jpg" alt="haskie" width="96" height="96">
</p>

<h1 align="center">haskie</h1>

<p align="center">
  <em><strong>Haskie "has a key" to your private bookshelf, giving your AI agents your exact taste.</strong> 
  Your trusted sources, searchable by your agents, cited to the page, kept on your machine.</em>
</p>

Your AI agent knows what everyone wrote. It does not know what you trust.

You picked the one book that settles the question, the standard that applies to your hardware,
the paper your team builds on. Your agent still answers from the average of the internet. haskie
hands it the key to your own shelf. Import your documents once, group them into collections, and
any Model Context Protocol (MCP) client can search them. Claude Code gets a rule that makes it
search them first. Every answer comes back as a short passage to quote, with the heading it sits
under and, for PDFs, the page.

It runs on your laptop. A few commands set it up, and a web UI handles the curating.

- [Why this exists](#why-this-exists)
- [Mission](#mission)
- [What that means in practice](#what-that-means-in-practice)
- [Status: early, and already useful](#status-early-and-already-useful)
- [Install](#install)
- [From files to answers](#from-files-to-answers)
- [How it works with Claude Code](#how-it-works-with-claude-code)
- [MCP tools](#mcp-tools)
- [Under the hood](#under-the-hood)
- [Supported formats](#supported-formats)
- [Good to know](#good-to-know)
- [References](#references)
- [License](#license)

## Why this exists

**Your taste is what AI averages away.** Writers who use AI for ideas write stories rated more
creative, and "more similar to each other than stories by humans alone" [1]. Models trained on
model output lose "the tails of the original content distribution" [2]. The tails are where taste
lives: the niche book, the unpopular opinion that turned out right.

**Half the new web is written by AI.** About 50% of new English articles online are now mostly
AI-generated, a share that has held since early 2025 [3]. 74% of new web pages contain some
AI-written text [4]. NewsGuard tracks 3,749 AI content-farm news sites across 16 languages [5].
Your agent reads this text today, and tomorrow's models train on it [2].

**Web search hands your agent unverified claims.** It reads whatever ranks, sourced or not.

- Leading chatbots repeated false news claims 35% of the time in 2025, up from 18% a year before,
  after they switched to live web search [6].
- AI search tools got more than 60% of source-citation queries wrong, and rarely signalled doubt
  [7].
- A web page can hide instructions that take over the agent reading it, the top risk for large
  language model (LLM) applications [8]. Five planted texts among millions steer a retrieval
  system's answer 90% of the time [9].

**Developers do not trust the answers.** 46% distrust the accuracy of AI tools, and 66% name
answers that are "almost right, but not quite" as their top complaint [10].

**Pasting whole books does not work.** Models lose what sits in the middle of a long context [11],
and accuracy "consistently degrades with increasing input length" [12]. Good agent context is "the
smallest set of high-signal tokens" [13].

**Building it yourself is a project.** You need a parser, a chunker, embeddings, a vector store,
keyword search, a reranker, and a job queue that survives a closed laptop. Then you maintain all
of it.

haskie does not fact-check your documents. It makes sure your agent reads the ones you chose, and
shows where every answer came from.

## Mission

haskie is a fast, transparent, easy-to-manage library of the sources you trust. It steers your AI
agents with your taste instead of the internet's average. Over MCP it aims to give the agent
relevant evidence from several sources, with no repeats, and every piece says where it came from
so the agent can dig deeper. Its tools follow how people learn. The agent first goes wide, with a
diverse map of which sections of which documents touch a topic. Then it goes deep, with focused
excerpts from the sections worth reading. The agent keeps the reasoning. haskie makes the small
retrieval decisions, so the agent needs fewer round trips and fewer tokens.

## What that means in practice

- **Your taste, your sources.** Only what you import can answer. Group documents into collections
  per topic: *coffee roasting*, *our architecture decisions*, *the standards for this board*. A
  document imported once can sit in any number of them.
- **Transparent.** Every result carries its document, heading path and lines, and pages for PDFs.
  Explore shows exactly what the agent receives, Operations every job, Sessions every search.
- **Relevant, without repeats.** Hybrid search matches meaning and exact terms, and an optional
  reranker sharpens the order. A point several sources make comes back once, with the others
  under `also_in`.
- **The agent decides, haskie does the legwork.** One `search_excerpts` call searches every
  collection in scope, merges neighbouring hits and folds repeats. A question with several parts
  goes in one call: each part gets its share of the slots, and each excerpt names the parts it
  answers. `search_sources` names the documents and collections that cover a topic. Each excerpt
  links to its full markdown file.
- **Local and polite to your machine.** Your documents never leave it. Only the models download,
  once, from Hugging Face. Indexing runs in parallel within a CPU budget you set, and after a crash
  the run resumes at the step it was on.
- **Sensible defaults, open to tuning.** The defaults are a small English embedding model, hybrid
  search and 1,200-character chunks. Each collection can override the chunk and search settings.

## Status: early, and already useful

haskie covers the whole path from import to cited answers in Claude Code. Not there yet:

- **OCR.** Scanned pages and images are stored but not searchable.
- **Other MCP clients.** Any MCP client can use the tools over HTTP. Only Claude Code has a
  one-command setup.

## Install

Needs [uv](https://docs.astral.sh/uv/), which fetches Python 3.13 if you have none.

```sh
uv tool install haskie
haskie run                  # web UI, REST API and MCP on http://127.0.0.1:8451; opens the first-run page
haskie install claude       # MCP server, skill, rule and SessionStart hook for Claude Code
haskie uninstall claude     # removes all four again; documents and collections stay
```

- **macOS (Apple Silicon):** also installs MLX and llama.cpp for the Apple GPU. llama.cpp
  compiles during the install, so run `xcode-select --install` first.
- **Linux:** ONNX Runtime runs on an NVIDIA GPU with CUDA 13 and cuDNN 9, else on the CPU.
- **First run:** pick an embedding model. The default, bge-small, is English and about 130 MB.
  Changing it later means running *Index all* in each collection.
- **Smoke test:** import a file on *Documents*, add it to a collection, then ask about it on
  *Explore*.
- **Other commands:** `haskie stop`, `haskie run --foreground` (for a supervisor),
  `haskie destroy`. `--port` or `HASKIE_PORT` moves the port, `--home` or `HASKIE_HOME` the data.

## From files to answers

1. **Documents.** Drop files onto the page, rename each one if you like, and import them with
   one button. haskie converts each to markdown (text formats are read as they are) and embeds it
   in the background, with a side-by-side preview. It warns when the same file is already
   imported, and shows the nearest documents once done.
2. **Collections.** Create one per topic and add its documents. Give it a one-line description.
   The agent reads it to choose where to look.
3. **Explore.** Search and see what your agent finds: *Excerpts*, *Sources* and *Sections*, the
   map of the sections a topic touches. Open a section to see what the map said about it and
   the sections it covers, and its document at its heading. Switch to *Chunks* or *Passages* to
   see how haskie cut the documents and built each answer.

**Operations** shows background jobs with their progress, and cancels running ones. **Sessions**
replays each agent conversation. **Gaps** lists the questions your sources did not answer, grouped
by topic, and replays them once you add a document. **Insights** charts searches and indexed
chunks over time. **Settings** describes every default.

## How it works with Claude Code

`haskie install claude` adds four things:

| what | where | why |
| --- | --- | --- |
| MCP server entry | `claude mcp add --transport http` | gives Claude the tools |
| skill | `~/.claude/skills/haskie/SKILL.md` | how to use the tools and what to cite. Its trigger names your collections, so it fires on *coffee roasting*, not on the word "documents" |
| rule | `~/.claude/rules/haskie.md` | loads into every session, so Claude searches your collections first, even for a plain "what is X?" that never triggers a skill |
| SessionStart hook | `~/.claude/settings.json` | runs `haskie run --hook`: starts the server if it is down, and passes the session id so Sessions can record it |

haskie records where it installed and rewrites the skill and rule in the background whenever
a collection is created, described, renamed or deleted. `--scope project` installs into
`./.claude` of the directory you run it from. With `CLAUDE_CONFIG_DIR` set, the user scope
installs there instead of `~/.claude`, as Claude Code reads it. The hook does not wait for the
server, so a session that starts while nothing is serving, such as the first after a reboot, has
no haskie tools. Run `haskie run` first if that session matters.

A typical exchange: you ask *"How should a background job retry a failed HTTP call without
charging twice?"* The rule sends Claude to `search_excerpts` before the web. haskie returns
passages from your books, each with a `header` and `location`, such as
`Stream Processing > Idempotence` at `ddia.pdf p.478 L21904-21931` (lines count through the
whole markdown file). Claude answers and cites them. If nothing matches, it says so and goes to
the web.

## MCP tools

The endpoint is `http://127.0.0.1:8451/mcp`, over HTTP. Tools that search, report a gap, or
change a document or collection take a `session_id`, so Sessions can replay the conversation.

| tool | what it does |
| --- | --- |
| `search_excerpts` | **The main search.** Passages ready to quote, best first (in turns for several parts), each with `header` and `location`. Repeats fold into `also_in`. Takes up to 5 parts of one question, and tags each excerpt with the parts it answers. `document_ids` and `section_ids` keep it to those documents and sections |
| `search_sources` | Which documents and collections cover a topic. One row per document, with its best sections |
| `search_sections` | A map of a topic: which sections of which documents touch it, near topics included, each with its descriptors and no text. Fast, for orientation before `search_excerpts`. Each section has an `id` to pass on as `section_ids` |
| `set_session_collections` | Limits the rest of the conversation to the collections `search_sources` suggested |
| `list_collections`, `get_collection`, `list_collection_documents` | Browse collections and their descriptions |
| `list_documents`, `get_document` | Browse documents |
| `add_document` | Import a local file by path |
| `add_document_to_collection`, `remove_document_from_collection` | Attach or detach a document |
| `describe_document` | Set what a document is about. `search_sources` shows it |
| `report_gap` | Say a search just run did not answer a question. The Gaps page shows it |
| `list_searches`, `list_gaps`, `replay_gaps`, `review_gaps` | Read the search log and the questions it did not answer, ask them again, resolve or dismiss them ([Gaps](docs/gaps.md)) |

Every search looks in the `collections` argument, else the session's collections, else all of
them. Creating and deleting collections, and deleting documents, stay in the web UI.
[docs/mcp.md](docs/mcp.md) walks through a session, and the
[skill](src/haskie/claude_code/skills/haskie/SKILL.md) lists every argument and field.

## Under the hood

[docs/](docs/README.md) has a page with diagrams for each stage.

```
indexing:  file ──► markdown ──► chunks ──────────► embeddings ──► LanceDB table
                    converted    structure-aware    cached once    one per collection

search:    query ──► hybrid search ──► rerank ────► passages ─────────► fold repeats ──► excerpts
                     vector + BM25     optional     neighbours merged   near-duplicates  cited by heading,
                     fused by rank                                      become also_in   page and line
```

**Structure-Aware Chunking.** Chunks follow the author's structure. A chunk never spans two
sections. It cuts at a blank line before it cuts inside a paragraph, and between sentences before
it cuts inside one. A table or code block stays whole unless it is longer than a chunk. By default
each chunk is embedded and indexed with its heading path in front, such as
`Part II > Replication > Leaders`. Context added to chunks cuts failed retrievals by 35%, and by
67% with BM25 and a reranker on top [14]. In that study an LLM writes the context. haskie takes it
from the headings, with no model call, and has not measured its own gain yet. Chunking by document
structure "largely improve[s]" retrieval-augmented generation (RAG) results [15]. Chunking by
embedding similarity does not justify its compute cost [16].

**LanceDB.** Each collection is one table on local disk. LanceDB is an embedded library with
vector and full-text (BM25) search in one table, on a columnar format built for fast random reads
[17]. So hybrid search needs no server.

**Hybrid search and reranking.** Vectors find meaning. BM25 finds exact terms, such as an error
code. haskie fuses both by rank (reciprocal rank fusion, RRF). An optional cross-encoder reads the
query and passage together and rescores the top candidates. Adding one takes the cut in failed
retrievals from 49% to 67% [14]. It is off by default. Settings offers models from 23 million
parameters up to multilingual ones.

**Repeats folded, passages whole.** Five books that make the same point would fill five of your
agent's slots. Most rerankers score one passage at a time, so they cannot see repeats [18]. haskie
merges hits on neighbouring chunks, then folds repeats with leader clustering. It walks the
results best first and compares each one only with the results already kept, by wording and, for
models with duplicate thresholds, by vector. The best result of each group keeps its place, so the
ranking stays intact. Diversity rerankers such as maximal marginal relevance (MMR) reorder it
instead. Comparing only with kept results stops chains, so A close to B and B close to C never
merges A with C. The same input always gives the same output. A repeat stays citable as an
`also_in` entry (`duplicate`, `contained` or `equivalent`), and its slot goes to the next distinct
result. Repeated passages do not significantly improve answer correctness, while different
documents improve it by 17–47% [19].

**Async-first, with durable jobs.** Every IO is awaited, and CPU work runs in worker threads, so
search and the UI stay responsive while the machine indexes. Imports, indexing, deletes,
maintenance and model downloads run as [DBOS](https://docs.dbos.dev) workflows. DBOS records every
step in the same SQLite file, so after a crash a workflow will "resume from the last completed
step" [20]. It also deduplicates runs, cancels them and bounds the queues. Two collections with
the same chunk settings share one embedding run.

**Parallel within a budget.** The CPU budget (default: half your cores) caps concurrent tasks
across converting, embedding and indexing. Large PDFs split into batches of pages, so one big book
does not block the rest.

## Supported formats

| kind | extensions |
| --- | --- |
| PDF | `.pdf` (page by page, with page numbers kept for citations) |
| Office | `.doc` `.docx` `.docm` `.ppt` `.pptx` `.pptm` `.pps` `.ppsx` `.ppsm` `.pot` `.xls` `.xlsx` `.xlsm` `.xlsb` |
| OpenDocument | `.odt` `.ods` `.odp` |
| Other documents | `.epub` `.rtf` |
| Text | `.md` `.markdown` `.txt` `.csv` `.json` `.html` `.htm` |
| Images | `.png` `.jpg` `.jpeg` `.gif` `.webp` `.svg` (stored and previewed, but not searchable) |

haskie does no optical character recognition (OCR). It skips scanned PDF pages and indexes the
rest. A PDF with only scanned pages fails, with a clear message.

## Good to know

- **One user, one machine.** haskie has no login. It listens on `127.0.0.1` by default. Do not
  expose it on a network.
- **Pre-1.0 storage.** A release that changes the storage format refuses to start on an older
  home and says so. Run `haskie destroy`, import your documents again, and run
  `haskie install claude` again.
- **One server per home.** A second `haskie run` on the same home refuses to start and names the
  process that holds it.

## References

1. Doshi, A. R. and Hauser, O. P. "Generative AI enhances individual creativity but reduces the
   collective diversity of novel content." *Science Advances*, 2024.
   https://doi.org/10.1126/sciadv.adn5290
2. Shumailov, I. et al. "AI models collapse when trained on recursively generated data."
   *Nature*, 2024. https://doi.org/10.1038/s41586-024-07566-y
3. Paredes, J. L. et al. "AI Now Writes as Many Online Articles as Humans." Graphite, May 2026.
   https://graphite.io/five-percent/ai-now-writes-as-many-online-articles-as-humans-do
4. Law, R. "74% of New Webpages Include AI Content (Study of 900k Pages)." Ahrefs, May 2025.
   https://ahrefs.com/blog/what-percentage-of-new-content-is-ai-generated/
5. NewsGuard. "Tracking AI-enabled Misinformation." Updated June 2026.
   https://www.newsguardtech.com/special-reports/ai-tracking-center/
6. NewsGuard. "AI False Information Rate Nearly Doubles in One Year." September 2025.
   https://www.newsguardtech.com/ai-monitor/august-2025-ai-false-claim-monitor/
7. Jaźwińska, K. and Chandrasekar, A. "AI Search Has a Citation Problem." *Columbia Journalism
   Review*, Tow Center, March 2025.
   https://www.cjr.org/tow_center/we-compared-eight-ai-search-engines-theyre-all-bad-at-citing-news.php
8. OWASP. "LLM01:2025 Prompt Injection." *OWASP Top 10 for LLM Applications*, 2025.
   https://genai.owasp.org/llmrisk/llm01-prompt-injection/
9. Zou, W. et al. "PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of
   Large Language Models." *USENIX Security*, 2025. https://arxiv.org/abs/2402.07867
10. Stack Overflow. "2025 Developer Survey: AI." https://survey.stackoverflow.co/2025/ai
11. Liu, N. F. et al. "Lost in the Middle: How Language Models Use Long Contexts." *TACL*, 2024.
    https://doi.org/10.1162/tacl_a_00638
12. Hong, K., Troynikov, A. and Huber, J. "Context Rot: How Increasing Input Tokens Impacts LLM
    Performance." Chroma, July 2025. https://www.trychroma.com/research/context-rot
13. Anthropic. "Effective context engineering for AI agents." September 2025.
    https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
14. Anthropic. "Introducing Contextual Retrieval." September 2024.
    https://www.anthropic.com/news/contextual-retrieval
15. Jimeno Yepes, A. et al. "Financial Report Chunking for Effective Retrieval Augmented
    Generation." 2024. https://arxiv.org/abs/2402.05131
16. Qu, R., Tu, R. and Bao, F. "Is Semantic Chunking Worth the Computational Cost?" 2024.
    https://arxiv.org/abs/2410.13070
17. Pace, W. et al. "Lance: Efficient Random Access in Columnar Storage through Adaptive
    Structural Encodings." 2025. https://arxiv.org/abs/2504.15247
18. Schlatt, F. et al. "Set-Encoder: Permutation-Invariant Inter-Passage Attention for Listwise
    Passage Re-Ranking with Cross-Encoders." *ECIR*, 2025. https://arxiv.org/abs/2404.06912
19. Ross, J. J. et al. "How retriever redundancy and diversity impact RAG effectiveness." 2026,
    preprint. https://arxiv.org/abs/2608.13956
20. DBOS. "dbos-transact-py." https://github.com/dbos-inc/dbos-transact-py

## License

[MIT](LICENSE)
