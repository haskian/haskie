---
name: haskie
description: >-
  Search the user's own curated document collections instead of answering from the web or from memory. Use whenever a question touches a topic they have collected sources on{topics}; when they say "my documents", "my collection", "what do my sources say"; or when they want an answer cited to something they own.
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

## Scope

Every search takes `collections`, a comma-separated list of names. Without it, the search covers
the session's selection. Without a selection, it covers every collection. A result held by
several collections in scope comes back once.

## Search tools

**`search_excerpts(q, context?, session_id?, collections?, limit?)`** → `excerpts`, `uncovered`,
`missing_terms`. Start here.

- `q` is a list: one question, or 2 to 5 parts of one, each at most 500 characters. Split only
  when the parts may be answered in different places. Keep in one part the conditions one passage
  must meet together. Write each part as a full question. An ambiguous question you cannot ask
  about gets one part per reading.
- `context` is the background the parts share, at most 200 characters. It steers what each part
  means, never the words a part matches (unless the `rerank_with_context` setting is on).
- `limit` counts excerpts. It defaults to the collection's setting when one collection is
  searched, else the user's, and never to fewer than the parts. A `limit` you pass must be at
  least the number of parts. The parts take turns at the slots, so one part cannot crowd out the
  others.

An excerpt is one section of one document: the largest heading that fits a few pages. It holds
every passage of that section the search matched, in document order, plus the text around and
between them that matches as well. A passage is matching chunks that sit next to each other.
Chunks are cut at headings, blank lines, blocks and sentences, so a passage starts and ends where
the author did. Excerpts come best first; for several parts, in the order the parts took turns,
so the first is not always the strongest.

- cite: `document`, `header` (the section's heading path, "parent > … > heading") and `location`
  ("doc p.3-4 L10-20", first passage to last). Cite one passage by its span's `header` and
  `location`.
- read: `text` joins the passages. Each opens with its headings below `header`, and `[…]` marks
  text skipped because it did not match. `score` is the best passage's. `collection` is whose
  index matched.
- `spans`: one per passage, in document order, each with `header`, `location`, `score`, `aspects`,
  `also_in`, `line_start`/`line_end`, `char_start`/`char_end` and `seq_start`/`seq_end`.
- `aspects`: the parts of `q` a passage answers, on each span and joined on the excerpt (empty for
  one question), with `aspect_scores` saying how well it matched each (its best chunk for that
  part). With a reranker on, a tag is its judgement; without one it is rank, and a vector or hybrid
  search finds a nearest passage for any part, so read `text` before you cite it as that part's
  answer.
- open: `markdown_file` and `source_file` (absolute paths), `line_start`/`line_end` (1-based, in
  `markdown_file`), `char_start`/`char_end` (0-based), `page_start`/`page_end` (PDF only, else
  null), `seq_start`/`seq_end` (chunk positions in the document).

What the answer lacks:

- `uncovered`: the parts of `q` no excerpt answers. Search them again in other words, or say the
  sources lack them.
- `missing_terms`: the words of `q` (stopwords aside) that no excerpt's text or headings hold in any
  form ("keeps" counts as "keep"). The search already looked for them once more by full text, and
  its best find joins the answer, past `limit` when it is in a section of its own. With a reranker
  on, it joins only when the reranker judges it an answer, and scores what the reranker gave it;
  without one, its span scores 0. A missing word is one the sources do not use: search again with a
  synonym, or say the sources lack it.

Repeats: a span's `also_in` lists every other place that says what the passage says, folded into
it rather than returned on its own. It is a tree: each place sits under what it repeats and has its
own `also_in`. Each place has `collection`, `document`, `header`, `location`,
`line_start`/`line_end`, `seq_start`/`seq_end`, `score` (its own match to the query), `relation` to
its parent and `similarity` (how strongly that relation holds).

- `duplicate`: the same text, whitespace aside.
- `contained`: it sits inside its parent, which says more.
- `equivalent`: the same meaning in other words.
- `to_parent` and `to_root` measure it against its parent and against the passage. `words` and
  `embedding` (null without vectors) each give `contained` (how much of it is in the other),
  `contains` (how much of the other is in it), `alike` and `score`. `chars` is the share of the
  shorter span both cover, in one document only.
- A place may sit in the same document. Cite it as a second source only when its `document`
  differs.

**`search_sources(q, session_id?, collections?, limit?, sections?)`** → `documents` (best first)
and `collections`. Use it to learn which documents or collections cover a topic, or when
`search_excerpts` found nothing relevant. Then pass its `collections` to `set_session_collections`
and run `search_excerpts` again.

- `limit`: documents, 1 to 100, default 10. `sections`: headings per document, 1 to 20, default 3.
- Per document: `document`, `description`, `score` (every chunk it matched folded into one, by
  default their sum, so a document that answers throughout beats one that answers once), `chunks`
  (how many matched), `collections` (the searched ones holding it), its best chunk as `text`,
  `header`, `location` and `line_start`/`line_end`, and `markdown_file`/`source_file`.
- `sections`: the hottest headings inside it, each with `header`, `score`, `chunks`, `location`
  and `line_start`/`line_end`. Read them to know where to look in a long document.
- `collections` at the top level: a small set of collections that together hold every document
  listed.

**`set_session_collections(session_id, collections)`** → the selection now, at most 100 names. It
replaces the previous selection; it does not add to it.

## Catalogue tools

Paged tools take `page_size` (default 100, at most 1000), `cursor`, `sort` and `order`
(`asc`/`desc`). They return `items`, `next_cursor` (null on the last page) and `total`.

- **`list_collections(sort: name|created_at)`** → `name`, `description`, `created_at`, `counts`
  (`total`, `indexed`, `active`, `error`, `by_status`). This list is current; the trigger above is
  a snapshot from install time.
- **`get_collection(collection)`** → `name`, `description`, `counts`, its `overrides`, the
  `effective` chunk settings, the `search` settings, `index_outdated`, `maintenance` and `index`
  statistics. It does not list the documents.
- **`list_collection_documents(collection, status?, sort: name|size|status|updated_at)`** → one
  membership per document: the `document` row, `status`, `error`, `added_at`, `updated_at`.
  `status` is how far this collection got indexing it (`pending`, `indexing`, `indexed`, `error`,
  `cancelled`), not the import status.
- **`list_documents(status?, sort: name|size|status|updated_at)`** → document rows, each with
  `collections`: how many hold it. `status` is one of `queued`, `converting`, `embedding`,
  `imported`, `error`, `cancelled`, `deleting`.
- **`get_document(document)`** → one such row.
- A document row: `name`, `suffix`, `size`, `status`, `error`, `preview`, `parser`,
  `skip_ocr_pages`, `description`, `created_at`, `updated_at`.

## Write tools

Every write takes `session_id`, so the change shows in the conversation's history.

- **`add_document(path, name?, description?, parser?, skip_ocr_pages?)`** → the document row at
  `queued`. `path` is an absolute local file. Converting and embedding run in the background: poll
  `get_document` until `imported`, `error` or `cancelled`.
- **`add_document_to_collection(collection, document)`** → `operation_id`. The document must be
  `imported` first. It is searchable there once its membership reads `indexed`: poll
  `list_collection_documents`.
- **`remove_document_from_collection(collection, document)`** → nothing. It detaches only; the
  document stays imported.
- **`describe_document(document, description)`** → the updated row. An empty description clears it.
  `search_sources` shows the description beside each document, so write one for anything an agent
  must choose between.
- Not over MCP: creating, describing, tuning, re-indexing or deleting a collection; deleting or
  re-importing a document; operations and settings. Point the user to the web UI.

## Log and gap tools

haskie keeps every search, so the collections can grow where they fall short.

- **`list_searches(session_id?, days?, limit?)`** → the searches of the last `days` days (7 by
  default), newest first, at most `limit` (50, at most 200). Each has its `questions`, each with
  `best_similarity`, `best_rerank` and `uncovered`, and its `results` cited by `header` and
  `location`. A failed search has an `error`. Use it to recall what this conversation already
  searched.
- **`list_gaps(review?, days?, signals?)`** → the questions no collection answers, grouped by
  topic, the most asked first. Each question has a `signal` (`reported`, `empty`, `uncovered` or
  `weak`), its `id`, and `near_misses`: what came closest. `borderline` (maybe answered: the best
  match sits between the bars) is left out unless `signals` names it. Tell the user which topics
  keep coming back; they are what to add next.
- **`replay_gaps(ids)`** → each gap question asked again over every collection, at most 50:
  `signal` null means it is answered now, and `results` cites where. Nothing is logged.
- **`report_gap(session_id, question, verdict, missing?)`** → the question's `id` and `verdict`.
  Call it when the excerpts of a search you just ran do not let a careful reader answer the
  question from them alone (`insufficient`), or answer only part (`partial`). `question` is the
  question as you passed it, in this session, in the last hour. `missing` is what they lacked, at
  most 300 characters. The gap then shows as `reported`, above every score-based one. Not for a
  question answered in other words.
- **`review_gaps(ids, review)`** → how many questions it reached. `resolved` after a document now
  answers them, `dismissed` when the collections are not meant to, `open` to take it back. Resolve
  a gap only after `replay_gaps` shows it answered.

## Errors

A failed call is a tool error with a status code and a message that names the problem.

- **503**: an embedding or reranker model is still downloading or warming. Wait and retry. If the
  message says the model failed to load, retrying will not help: tell the user.
- **404**: no such collection or document, including a name in `collections`, or a document that
  is not in the collection. Check the name with `list_collections` or `list_documents`.
- **409**: the document name is taken, or the document is not `imported` yet.
- **422**: a bad argument. For example: `limit` out of range or below the number of parts, no
  parts or more than 5, a question over 500 characters, a `context` over 200 characters, a page
  size over 1000, an unknown `status` or `sort`, a `path` that is relative or missing, or a file
  type haskie cannot read.
