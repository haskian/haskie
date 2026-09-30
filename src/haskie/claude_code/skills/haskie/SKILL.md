---
name: haskie
description: >-
  Search the user's own curated document collections instead of answering from the web or from memory. Use when they say "my documents", "my collection", "what do my sources say"; when they want an answer cited to something they own; or whenever a question touches a topic they have collected sources on{topics}.
---

# haskie: the user's own sources

The user chose and trusts these documents. Where they cover a topic, they outrank the web and
your training data. Answer from them, and name the document each answer came from.

The always-loaded rule says when to search and how to write the query. This file is the tool
reference.

## Session

The SessionStart hook prints "{announcement}" into your context. Pass that id, unchanged, as
`session_id` on every call that takes one. A call without it shows in no session of the user's
Sessions view. If the line is missing, pick one short string (at most 128 characters) and reuse it
for the whole conversation.

## Search tools

Each search tool's description says where it looks, how to split and word `q`, what `limit`
counts, and how `also_in` folds repeats. This section adds what those descriptions leave out: the
fields to read, and the defaults.

**`search_excerpts(q, context?, session_id?, collections?, limit?)`** → `excerpts`, `uncovered`,
`missing_terms`. Start here.

- `limit` defaults to the collection's setting when one collection is searched, else the user's.
  It never defaults to fewer than the parts.
- A passage is matching chunks that sit next to each other. A result held by several collections
  in scope comes back once.
- cite: `document`, `header` ("parent > … > heading") and `location` ("doc p.3-4 L10-20", first
  passage to last).
- read: `score` is the best passage's. `collection` is whose index matched.
- `spans`: one per passage, in document order, each with `header`, `location`, `score`, `aspects`,
  `aspect_scores`, `also_in`, `line_start`/`line_end`, `char_start`/`char_end`,
  `page_start`/`page_end` and `seq_start`/`seq_end`. `aspects` is empty for one question.
- open: `markdown_file` and `source_file` (absolute paths), `line_start`/`line_end` (1-based, in
  `markdown_file`), `char_start`/`char_end` (0-based), `page_start`/`page_end` (PDF only, else
  null), `seq_start`/`seq_end` (chunk positions in the document).
- `missing_terms`: with a reranker on, the full-text find joins only when the reranker judges it
  an answer, and scores what the reranker gave it. Without one, its span scores 0. A missing word
  is one the sources do not use: search again with a synonym, or say the sources lack it.
- Each place in `also_in` has `collection`, `document`, `header`, `location`,
  `line_start`/`line_end`, `seq_start`/`seq_end`, `score` (its own match to the query), `relation`
  to its parent and `similarity` (how strongly that relation holds).

**`search_sources(q, session_id?, collections?, limit?, sections?)`** → `documents` (best first)
and `collections`.

- `limit`: documents, 1 to 100, default 10. `sections`: headings per document, 1 to 20, default 3.
- Per document: `document`, `description`, `score`, `chunks`, `collections`, its best chunk as
  `text`, `header`, `location` and `line_start`/`line_end`, and `markdown_file`/`source_file`.
- Each of `sections` has `header`, `score`, `chunks`, `location` and `line_start`/`line_end`.
  Read them to know where to look in a long document.

**`set_session_collections(session_id, collections)`** → the selection now, at most 100 names. It
replaces the previous selection; it does not add to it. It scopes every later search of the
session, `search_sources` included. Pass `[]` to clear it before a new topic.

## Catalogue tools

Paged tools take `page_size` (default 100, at most 1000), `cursor`, `sort` and `order`
(`asc`/`desc`). They return `items`, `next_cursor` (null on the last page) and `total`.

- **`list_collections(sort: name|created_at)`** → `name`, `description`, `created_at`, `counts`
  (`total`, `indexed`, `active`, `error`, `by_status`). This list is current; the trigger above is
  only as fresh as the session that loaded it.
- **`get_collection(collection)`** → `name`, `description`, `counts`, its `overrides`, the
  `effective` chunk settings, the `search` settings, `index_outdated`, `maintenance` and `index`
  statistics. It does not list the documents.
- **`list_collection_documents(collection, status?, sort: name|size|status|updated_at)`** → one
  membership per document: the `document` row, `status`, `error`, `added_at`, `updated_at`.
  `status` is how far this collection got indexing it (`pending`, `indexing`, `indexed`, `error`,
  `cancelled`), or `removing` while a detach clears it out. It is not the import status.
- **`list_documents(status?, sort: name|size|status|updated_at)`** → document rows, each with
  `collections`: how many hold it. `status` is one of `queued`, `converting`, `embedding`,
  `imported`, `error`, `cancelled`, `deleting`.
- **`get_document(document)`** → one such row.
- A document row: `name`, `suffix`, `size`, `status`, `error`, `preview`, `parser`,
  `skip_ocr_pages`, `description`, `created_at`, `updated_at`.

## Write tools

Every write takes `session_id`, so the change shows in the conversation's history.

- **`add_document(path, name?, description?, parser?, skip_ocr_pages?)`** → the document row at
  `queued`. `path` is an absolute local file. The name is stored in lowercase-kebab-case with the
  original suffix, so use the returned row's `name` from then on. Converting and embedding run in
  the background: poll `get_document` until `imported`, `error` or `cancelled`.
- **`add_document_to_collection(collection, document)`** → `operation_id`. The document must be
  `imported` first. It is searchable there once its membership reads `indexed`: poll
  `list_collection_documents`.
- **`remove_document_from_collection(collection, document)`** → nothing. It detaches only; the
  document stays imported. It answers once the removal is queued: the membership reads `removing`
  until it is gone.
- **`describe_document(document, description)`** → the updated row. An empty description clears it.
  `search_sources` shows the description beside each document, so write one for anything an agent
  must choose between.
- Not over MCP: creating, describing, tuning, re-indexing or deleting a collection; deleting or
  re-importing a document; operations and settings. Point the user to the web UI.

## Log and gap tools

haskie keeps every search, so the collections can grow where they fall short.

- **`list_searches(session_id?, days?, limit?)`**: to recall what this conversation already
  searched.
- **`list_gaps(review?, days?, signals?)`**: when the user asks what the collections lack. Tell
  them which topics keep coming back; they are what to add next.
- **`replay_gaps(ids)`**: to check whether the collections answer a gap now.
- **`report_gap(session_id, question, verdict, missing?)`**: right after a search whose excerpts
  do not answer the question, or answer only part of it.
- **`review_gaps(ids, review)`**: to resolve a gap once `replay_gaps` shows it answered, or to
  dismiss one the collections are not meant to answer.

## Errors

A failed call is a tool error with a status code and a message that names the problem.

- **503**: an embedding or reranker model is still downloading or warming. Wait and retry. If the
  message says the model failed to load, retrying will not help: tell the user.
- **404**: no such collection or document, including a name in `collections`, or a document that
  is not in the collection. Check the name with `list_collections` or `list_documents`. From
  `report_gap`: no search in this session asked that question, word for word, in the last hour.
- **409**: the document name is taken, the same file is already imported (the message names that
  document: use it), or the document is not `imported` yet. From `report_gap`:
  that search failed, so there is nothing to judge.
- **422**: a bad argument. For example: `limit` out of range or below the number of parts, no
  parts or more than 5, a question over 500 characters, a `context` over 200 characters, a page
  size over 1000, an unknown `status` or `sort`, a `path` that is relative or missing, or a file
  type haskie cannot read.
