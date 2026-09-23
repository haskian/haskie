Search the user's own haskie document collections before answering from memory or from the web
whenever a question touches a topic they have collected sources on{topics}. Use them for plain
knowledge questions ("what is X", "why Y"), not only for tasks. Use them when planning, so their
sources set the conventions. Use them whenever you are in doubt. These are documents the user chose
and trusts; they outrank training data, so search even when you think you know the answer. When the
collections do not answer, say so, then go to the web.

Do not use for: the mechanics of the current codebase, refactoring, debugging business logic, or
topics no collection covers.

## Steps

1. Write the query as a full question that carries its context: what is being built, the constraint
   being weighed, the decision at hand, the exact terms of art. The search is semantic (hybrid
   vector and BM25, reranked), so a sentence with context outranks a bare keyword. Ask "how should
   a background job retry a failed HTTP call without duplicating the side effect", not "retry".
2. `search_excerpts` with this conversation's haskie session id, announced at session start. Repeat
   it for each follow-up. That is the search: it returns the passages to answer from.
3. `search_sources` when the question is which documents or collections cover a topic, not what
   they say, or when `search_excerpts` returned nothing relevant. It names the documents and
   collections. Call `set_session_collections` with the `collections` it returns, then
   `search_excerpts` again. Run `list_collections` when unsure what exists.
4. Cite the document by name. Quote its `header` and `location`.
5. No hits is an answer. Say the collections do not cover it, then fall back to the web. Never pass
   a web result off as one of their sources.
