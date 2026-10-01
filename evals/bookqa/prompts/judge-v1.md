You are grading search results for a question a reader asked about the document `{source}`.
Each numbered passage below is one result a search returned, from `{source}` or another document
in the same library. Grade every passage on its own, from its text alone - not from what you know
about the topic, and not from what another passage says:

- 2: it states the answer to the question, or one of the facts the answer needs. A question with
  several parts may need several passages; a passage stating one of those parts is a 2.
- 1: it is on the question's topic, but states none of the facts the answer needs.
- 0: it is not relevant to the question.

A passage that only mentions the terms of the question, a table of contents or an index entry
pointing at the answer, or a heading without the text under it, is a 1 at most.

The question: {query}

Reply with a JSON array and nothing else - no prose, no code fence - one element per passage, in
order: [{{"n": 1, "grade": 2}}, {{"n": 2, "grade": 0}}, ...]

The passages:

{passages}
