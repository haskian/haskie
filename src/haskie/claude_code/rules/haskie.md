Search the user's own haskie document collections before answering from memory or from the web
whenever a question touches a topic they have collected sources on{topics}. That includes plain
knowledge questions ("what is X", "why Y"), planning (their sources set the conventions), and any
moment of doubt. The user chose and trusts these documents. They outrank training data, so search
even when you think you know the answer.

Do not use for: the mechanics of the current codebase, refactoring, debugging business logic, or
topics no collection covers.

## Steps

1. Write the query as a full question that carries its context: what is being built, the constraint
   being weighed, the decision at hand, the exact terms of art. The search is hybrid by default
   (vector and BM25), so a sentence with context beats a bare keyword. Ask "how should a background
   job retry a failed HTTP call without duplicating the side effect", not "retry".
2. Call `search_excerpts` with this conversation's haskie session id, announced at session start.
   Call it again for each follow-up. When parts of the question may be answered in different
   places, pass each part as its own `q` in one call, and the background they share as `context`.
3. Call `search_sources` when the question is which documents or collections cover a topic, or when
   `search_excerpts` found nothing relevant. Pass the `collections` it returns to
   `set_session_collections`, then call `search_excerpts` again. Call `list_collections` when unsure
   what exists.
4. Answer from the excerpts. Cite each by its `document`, `header` and `location`.
5. Report the gaps. A part in `uncovered` or a word in `missing_terms` is something the sources do
   not say. No excerpts at all means the collections do not cover the topic. Say so, then fall back
   to the web. Never pass a web result off as one of their sources.
6. When the excerpts came back but do not let a careful reader answer from them alone, call
   `report_gap` with the question as you asked it, `insufficient` or `partial`, and in `missing` what
   they lacked. The user sees it as a gap to close.
