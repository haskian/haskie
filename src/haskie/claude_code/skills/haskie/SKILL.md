---
name: haskie
description: >-
  Search the user's own curated document collections instead of answering from the web or from memory. Use whenever a question touches a topic they have collected sources on{topics}; when they say "my documents", "my collection", "what do my sources say"; or when they want an answer cited to something they own.
---

# haskie: the user's own sources

These collections hold documents the user chose and trusts. Where they cover a topic, they outrank
the web and training data. Answer from them, and name the document each answer came from.

## Flow

The always-loaded rule says when to search and how to shape a query. This is the tool reference.

1. **Session id.** The SessionStart hook prints "{announcement}" into your context. Pass that id,
   unchanged, as `session_id` on every call that takes one. A call without it belongs to no
   session, so the user's Sessions view never shows it. If the line is missing, use one short
   stable string for the whole conversation.
2. **Search.** Answer from the excerpts `search_excerpts` returns. Use `search_sources` when you
   need to know which documents or collections cover a topic, or when `search_excerpts` found
   nothing relevant. Then call `set_session_collections` with the `collections` it returns, and
   `search_excerpts` again.

## Search tools

Every search scopes to the comma-separated `collections` argument. Without it, the session's
selection. Without that, every collection. A hit held by several collections in scope is returned
once.

**`search_excerpts(q, session_id?, collections?, limit?)`** → excerpts, best first. `limit` is at
least 1 and defaults to the user's setting. One excerpt is what one document says in one place:
the matching chunks merged where they sit next to each other, widened to the whole lines or
sentences around them within 300 characters. So it usually starts and ends where the author did.

- cite: `document`, `header` (heading breadcrumb, "parent > … > heading"), `location`
  ("doc p.3-4 L10-20")
- read: `text`, `score`, `collection` (whose index matched)
- open: `markdown_file` and `source_file` (absolute paths), `line_start`/`line_end` (1-based, in
  `markdown_file`), `char_start`/`char_end` (0-based), `page_start`/`page_end` (PDF only, else
  null), `seq_start`/`seq_end` (chunk positions in the document)
- repeats: `also_in` lists every other place that says the same thing, folded into this excerpt.
  Each has `document`, `header`, `location`, `score`, `similarity` and `relation`. `duplicate`
  says the same as a whole. `contained` sits inside this excerpt, which says more. `same_span` is
  the same lines, chunked another way. A `via` names the place in the same list the relation was
  measured against, when that is not this excerpt. A place may be elsewhere in the same document,
  so cite it as a second source only when its `document` differs.

**`search_sources(q, session_id?, collections?, limit?, sections?)`** → `documents`, best first
(`limit` 1-100, default 10), and `collections`: a small set of collections that together hold every
document listed, ready for `set_session_collections`. Per document: `document`, `description`,
`score` (its best chunk folded with every chunk it matched, so a document that answers throughout
beats one that answers once), `chunks` (how many matched), `collections` (the searched ones holding
it), its best chunk as `text`, `header`, `location`, `line_start`, `line_end`, and `sections`: the
hottest headings inside it, each with `header`, `score`, `chunks`, `location`, `line_start`,
`line_end`. Read the sections to know where in a long document to look. `sections` caps how many
come back per document (default 3, at most 20).

**`set_session_collections(session_id, collections)`** → the list now selected. Replaces the
previous selection; it does not add to it.

## Catalogue tools

Paged tools take `page_size` (default 100, at most 1000), `cursor`, `sort` and `order`
(`asc`/`desc`). They return `items`, `next_cursor` (null on the last page) and `total`.

- **`list_collections(sort: name|created_at)`** → `name`, `description`, `created_at`, `counts`
  (`total`, `indexed`, `active`, `error`, `by_status`). This list is authoritative; the trigger
  above is a snapshot from install time.
- **`get_collection(collection)`** → `name`, `description`, `counts`, its chunk and search
  `overrides`, the `effective` chunk settings, the `search` settings, `index_outdated`,
  `maintenance` and `index` statistics. Members are not here.
- **`list_collection_documents(collection, status?, sort: name|size|status|updated_at)`** →
  memberships: the `document` row, `status` (`pending`, `indexing`, `indexed`, `error`,
  `cancelled`: how far this collection got indexing it, not the import status), `error`,
  `added_at`, `updated_at`.
- **`list_documents(status?, sort: name|size|status|updated_at)`** → document rows plus
  `collections`, how many hold it. `status` is one of `queued`, `converting`, `embedding`,
  `imported`, `error`, `cancelled`, `deleting`.
- **`get_document(document)`** → one such row.
- A document row: `name`, `suffix`, `size`, `status`, `error`, `preview`, `parser`,
  `skip_ocr_pages`, `description`, `created_at`, `updated_at`.

## Write tools

Every write takes `session_id`, so the change shows in the conversation's history.

- **`add_document(path, name?, description?, parser?, skip_ocr_pages?)`** → the document row at
  `queued`. `path` is an absolute local file. Convert and embed run in the background: poll
  `get_document` until `imported`, `error` or `cancelled`. A name already taken is a 409.
- **`add_document_to_collection(collection, document)`** → `operation_id`. The document must be
  `imported` first, else 409. It is not searchable there until its membership reads `indexed`:
  poll `list_collection_documents`.
- **`remove_document_from_collection(collection, document)`** → nothing. Detaches only; the document
  stays imported.
- **`describe_document(document, description)`** → the updated row. An empty description clears it.
  `search_sources` shows the description beside each document, so write one for anything an agent
  must choose between.
- Not over MCP: creating, describing, tuning, re-indexing or deleting a collection; deleting or
  re-importing a document; operations and settings. Point the user at the web UI.

## Errors

- **503**: an embedding or reranker model is still downloading or warming. Wait and retry. If the
  message says the model failed to load, retrying will not help: tell the user.
- **404**: no such collection or document, including a name in `collections`. Check the name with
  `list_collections` or `list_documents`.
- **409**: the name is taken, or the document is not `imported` yet.
- **422**: a bad argument. `limit` out of range, a page size over 1000, or an unknown `status`.
