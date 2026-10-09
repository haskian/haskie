<p align="center">
  <img src="design/haskie-logo.jpg" alt="haskie" width="96" height="96">
</p>

<h1 align="center">haskie</h1>

<p align="center">
  <strong>A knowledge base for AI coding agents.</strong><br>
  Curate and organize your sources through an intuitive web UI.<br>
  Give your agents a library they can explore efficiently.<br>
  Built around a discovery–synthesis loop inspired by how humans research.
</p>

Give your agent trusted sources that span technical knowledge, domain expertise and your personal
taste. Import books, research, standards, product requirements, business rules and design references.
Your choices guide how it designs, codes and reviews. It finds relevant passages and cites them
without reading whole documents into its context.

haskie runs on your machine and connects to Codex and Claude Code. You choose the sources.
Your agent uses them to reason about your work.

Read [why haskie exists](docs/why-haskie.md) for the reasoning and research behind it.

## Mission

Make knowledge easy for people to curate and for AI agents to explore. Help agents see the wider
picture with a broad map of topics and sources. Guide them to focused passages for deeper reading
once they know what to look for. Return highly relevant, diverse evidence without duplicates.

Build a knowledge graph that connects related terms and topics even when their embeddings are
far apart. Give agents paths to related knowledge that vector similarity alone may miss.
Preserve citations and relations between sources so agents can follow the evidence. Leave the
reasoning to the agent.

## Why use haskie

- **Your sources shape the work.** Build collections around your domain, product or technical stack.
  The Codex and Claude Code integrations tell agents to search them first when they cover the topic.
- **Relevant, diverse results fit in context.** Agents receive focused passages from several
  sources. Duplicate folding keeps repeated evidence in one result and preserves its other
  sources through `also_in` links, leaving room for distinct evidence.
- **You can check the answer.** Each passage names its document, heading and lines, plus pages for
  PDFs. Open the source or use Explore to inspect what your agent receives.
- **Missing knowledge becomes visible.** Review agent searches in Sessions. Gaps groups unanswered
  questions by topic so you know which sources to add next.
- **You manage the library in one place.** Import, preview and organize documents in the web UI.
  Documents and search models stay on your machine.

## The Discovery–Synthesis Loop

haskie supports a research loop that moves from a broad question to focused understanding.
Agents map a topic, choose what to read, synthesize the evidence and refine their questions.
haskie handles the retrieval part of retrieval-augmented generation (RAG). The agent does the
reasoning and synthesis.

```text
Import: documents → sections and passages → searchable collections
Ask:    question → map the topic → read focused passages → synthesize → refine the question ↺
```

Semantic search finds passages by meaning. Hybrid search adds exact terms. Optional reranking
puts the closest answers first. Folding groups duplicate and overlapping passages while
`also_in` preserves their source relations and citations. Agents can follow those links and
related sections to explore further without filling their context with repeats.

Short descriptions and topic descriptors help the agent decide where to read:

- **Section descriptors** name the topics inside each section. By default, haskie extracts
  distinctive terms. With AI descriptions enabled, a local language model reads headings and
  prose to write short summaries and topic descriptors.
- **Document summaries** combine those section summaries and descriptors in AI mode. Automatic
  summaries preserve descriptions you wrote yourself.
- **Collection descriptions** explain what the library covers. Write your own or use
  **Describe with AI** to summarize the member documents.

For a broad question, the agent asks for a map of relevant sections. Descriptions help it judge
which sources fit. Descriptors suggest search terms and related sections worth reading. The agent
then selects passages for evidence. A specific question can go straight to passages.
The agent connects what it learns and cites its sources. Gaps and new questions guide the next
search.

You can inspect the map and passages in **Explore**.

## Install and setup

With [uv](https://docs.astral.sh/uv/) installed, two commands install haskie and open its setup page.
uv fetches Python 3.13 if needed. On Apple Silicon, first install Apple's command-line tools with
`xcode-select --install`.

```sh
uv tool install haskie
haskie run              # opens the setup page at http://127.0.0.1:8451
```

Connect your agent with one more command. Use either or both:

```sh
haskie install codex    # connects Codex and adds the search-first rule
haskie install claude   # connects Claude Code and adds the search-first rule
```

For Codex, review and trust the installed session hook in Codex before starting a new session.

1. Choose an embedding model on the setup page. The default multilingual model is about 390 MB.
   Changing it later requires **Index all** in each collection.
2. Import files on **Documents**. Check the preview and add them to a **Collection**.
3. Try a question in **Explore**. See the passages your agent can use and open their sources.
4. Start a new Codex or Claude Code session and ask a question your sources cover.

For example: *“Which orders qualify for a refund under our returns policy?”*
The search-first rule tells your agent to consult your collections and cite the passages it uses.
If they do not answer the question, it should say so before looking elsewhere.

See [setup and operation](docs/setup.md) for hardware and model choices. The
[user guide](docs/user-guide.md) covers curation, search views and supported file extensions.

## Connect your agent

haskie serves the Model Context Protocol (MCP) over HTTP at `http://127.0.0.1:8451/mcp`.
Codex and Claude Code have one-command setup. Other compatible MCP clients can use that endpoint.

Both integrations teach the agent when to search, how to explore and what to cite.
They follow changes to your collections. Run `haskie run` before the first session after a reboot
so the tools are ready when your agent connects. See the [MCP guide](docs/mcp.md) for setup options
and the tool reference.

## Supported files and limits

Import PDFs, Office and OpenDocument files, EPUB, RTF, Markdown, plain text, CSV, JSON, HTML and
images.
Optical character recognition (OCR) runs on your device. It reads scanned PDF pages and
images (SVG aside). Its model, about 31 MB, downloads with the other models. Documents imported
before OCR keep their old text until you delete them and import them again.

haskie is early software for one user on one machine. It has no login and listens on `127.0.0.1`
by default. Keep it off public networks. It retrieves your chosen sources; it does not fact-check
them.

Before 1.0, some storage changes require a fresh import. Compatible upgrades run at startup.
For incompatible changes, haskie refuses to start and explains why. See
[storage upgrades](docs/storage.md) before resetting your data.

## Manage and develop

Use `haskie stop` to stop the server. `haskie uninstall codex` or `haskie uninstall claude` removes
that integration and keeps your documents. `--port` changes the port; `--home` changes the data
directory.

Read the [technical docs](docs/README.md) for search, indexing and storage.
See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and checks.

Licensed under [MIT](LICENSE).
