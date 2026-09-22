---
name: haskie
description: >-
  Search the user's own curated document collections instead of answering from the web or from memory. Use whenever a question touches a topic they have collected sources on{topics}; when they say "my documents", "my collection", "what do my sources say"; or when they want an answer cited to something they own.
---

# haskie — the user's own sources

These collections are documents the user chose and trusts. Where they cover a topic they outrank
the web and training data. Answer from them, and name the document the answer came from.

## Flow

1. **Session id.** The SessionStart hook prints "{announcement}" into your context. Pass that id,
   unchanged, as `session_id` on every call that takes one. A call without it belongs to no
   session, so the user's Sessions view never shows it. If the line is not in your context, use
   one short stable string for the whole conversation.
2. **Query.** Write the full question a colleague would ask, with its context: what is being
   built, the constraint weighed, the decision at hand, the exact terms of art. Retrieval is
   semantic (hybrid vector and BM25, fused, optionally reranked), so "how should a background job
   retry a failed HTTP call without duplicating the side effect" finds the chapter and "retry"
   finds every page with the word.
3. `search_excerpts` — the search. Same session id on every call, for this and every follow-up
   question. Quote the excerpt as it comes; cite its `location` and `header`.
4. `search_sources` — when the question is *which* documents or collections cover a topic, not
   what they say (a reading list, a shortlist to open), or as the fallback when `search_excerpts`
   returned nothing or nothing relevant. `set_session_collections` with the `collections` it
   returns, then `search_excerpts` again. Pass `collections` on a search to override the session
   for that one call.
5. **No hits is an answer.** Say the collections do not cover it, then go to the web. Never pass a
   web result off as one of their sources.

## Search tools

Scope of every search: the comma-separated `collections` argument if given, else the session's
selection, else every collection. `limit` is at least 1; its default is the user's setting. A hit
held by several collections in scope is returned once.

**`search_excerpts(q, session_id?, collections?, limit?)`** → list of excerpts, best first. One
excerpt is what one document says in one place: the matching chunks merged where they sit next to
each other and widened to whole sentences, so it starts and ends where the author did.

- cite: `doc`, `header` (heading breadcrumb, "parent > … > heading"), `location`
  ("doc p.3-4 L10-20")
- read: `text`, `score`, `collection` (whose index matched)
- open: `markdown_file` and `source_file` (absolute paths), `line_start`/`line_end` (1-based, in
  `markdown_file`), `char_start`/`char_end` (0-based), `page_start`/`page_end` (PDF only, else
  null), `seq_start`/`seq_end` (chunk positions in the document)

**`search_sources(q, session_id?, collections?, limit?, sections?)`** → `documents`, best first,
and `collections`: the fewest collections that together hold every document listed, ready for
`set_session_collections`. Per document: `doc`, `description`, `score` (its best chunk folded
with every chunk it matched, so a document that answers throughout beats one that answers once),
`chunks` (how many matched), `collections` (the searched ones holding it), its best chunk as
`text`, `heading`, `location`, `line_start`, `line_end`, and `sections`: the hottest headings
inside it, each with `header`, `score`, `chunks`, `location`, `line_start`, `line_end`. Read the
sections to know where in a long document to look. `sections` caps how many come back.

**`set_session_collections(session_id, collections)`** → the list now selected. Replaces the
previous selection rather than adding to it.

## Catalogue tools

Paged tools take `page_size` (default 100, at most 1000), `cursor`, `sort` and `order`
(`asc`/`desc`), and return `items`, `next_cursor` (null on the last page) and `total`.

- **`list_collections(sort: name|created_at)`** → `name`, `description`, `created_at`, `counts`
  (`total`, `indexed`, `active`, `error`, `by_status`). This list is authoritative; the trigger
  above is a snapshot from install time.
- **`get_collection(collection)`** → the same plus its chunk `settings` and `effective` values,
  `search` settings, `index_outdated`, `maintenance` and `index` statistics. Members are not
  here.
- **`list_collection_documents(collection, status?, sort: name|size|status|updated_at)`** →
  memberships: the `document` row, `status` (`pending`, `indexing`, `indexed`, `error`,
  `cancelled`: how far this collection got indexing it, not the import status), `error`,
  `added_at`, `updated_at`.
- **`list_documents(status?, sort: name|size|status|updated_at)`** → document rows plus
  `collections`, how many hold it. `status` is one of `queued`, `converting`, `embedding`,
  `imported`, `error`, `cancelled`, `deleting`.
- **`get_document(doc)`** → one such row.
- A document row: `name`, `suffix`, `size`, `status`, `error`, `parser`, `skip_ocr_pages`,
  `description`, `created_at`, `updated_at`.

## Write tools

Every one takes `session_id`, so the change shows in the conversation's history.

- **`add_document(path, name?, description?, parser?, skip_ocr_pages?)`** → the document row at
  `queued`. `path` is an absolute local file. Convert and embed run in the background: poll
  `get_document` until `imported` or `error`.
- **`add_document_to_collection(collection, document)`** → `operation_id`. The document is not
  searchable there until its membership reads `indexed`: poll `list_collection_documents`.
- **`remove_document_from_collection(collection, doc)`** → nothing. Detaches only; the document
  stays imported.
- **`describe_document(doc, description)`** → the updated row. Empty clears it. `search_sources`
  shows the description beside each match, so write one for anything an agent must choose
  between.
- Not over MCP: creating or deleting a collection, deleting a document. Point the user at the web
  UI.

## Errors

- **503**: an embedding or reranker model is still downloading or warming. Wait and retry, or
  tell the user what is holding the search up.
- **404**: no such collection or document. Check the name with `list_collections` or
  `list_documents`.
- **422**: a bad argument: a `collections` name nobody owns, `limit` under 1, a page size over
  1000, an unknown `status`.
