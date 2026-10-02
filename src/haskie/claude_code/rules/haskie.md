Search the user's own haskie document collections before answering from memory or from the web
whenever a question touches a topic they have collected sources on{topics}. That includes plain
knowledge questions ("what is X", "why Y"), planning (their sources set the conventions), and any
moment of doubt. The user chose these documents, so search even when you think you know the
answer. Prefer them when the excerpts actually answer the question. When they do not, or answer
only part of it, say so, and mark what you add from memory or the web as your own.

Do not use for: the mechanics of the current codebase, refactoring, debugging business logic, or
topics no collection covers.

## Steps

Pass this conversation's haskie session id, announced at session start, to every call that
takes one.

1. Write the query as a full question that carries its context: what is being built, the constraint
   being weighed, the decision at hand, the exact terms of art. The search is hybrid by default
   (vector and BM25), so a sentence with context beats a bare keyword. Ask "how should a background
   job retry a failed HTTP call without duplicating the side effect", not "retry".
2. Pick the search. Call `search_excerpts` when you know the words the sources would use and want
   the answer. Call `search_sections` first when the question is broad, when you lack the sources'
   words for it, when it asks which documents cover a topic, or when it needs several authors. It
   maps the sections that touch the topic and the documents behind them, with no text.
3. Read the map before any text. Its descriptors and headers are the sources' own words: ask again
   in them. Read the `related` headers under each pick, because the best section can sit there.
   Skip back matter (an index, contents, a bare "Summary") and any section that only shares a word
   with the topic. Then read the sections worth reading with `search_excerpts`, their `id`s as
   `section_ids`. For several views, pick sections of several documents.
4. In `search_excerpts`, pass each part of a question that may be answered in a different place as
   its own `q` (2 to 5), and the background they share once as `context`. Keep in one `q` the
   conditions one passage must meet together. Resolve an ambiguous question first; when you cannot
   ask, pass one part per reading. Ten excerpts is already a long read: narrow with `section_ids`
   or `document_ids` rather than raise `limit`.
5. Judge the text, not the score. A search always finds the nearest passages, even on a topic the
   sources never cover. Answer from the excerpts that answer the question, and cite each by its
   `document`, `header` and `location`. An excerpt on a nearby topic is not an answer: do not
   stretch it to fit. Excerpts often lean on one document: say so when the question needs more.
6. Read what the answer says it lacks. `searched` names the collections it covered: a selection set
   earlier may hide the answer. A word in `missing_terms` is one no excerpt holds. `uncovered`
   lists the questions no excerpt answers, and any question, one alone included, the sources
   match only weakly: a map is always full, and excerpts can be too, so treat a question there as
   unanswered. No excerpts, searched without `document_ids` or `section_ids`, means the
   collections do not cover the topic. Say so, then fall back to the web. Never pass a web result
   off as one of their sources.
7. To keep a conversation in the collections a map named, pass its `collections` to
   `set_session_collections`; clear it with `[]` before a new topic. Call `list_collections` when
   unsure what exists.
8. When the excerpts do not let a careful reader answer from them alone, call `report_gap` right
   after that search: the question exactly as you asked it, `insufficient` or `partial`, and in
   `missing` what they lacked. The user sees it as a gap to close.
