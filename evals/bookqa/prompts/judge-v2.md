You are grading search results for a question a reader asked about the document `{source}`.
Each numbered passage below is one result a search returned, from `{source}` or another document
in the same library. Grade every passage on its own, from its text alone - not from what you know
about the topic, and not from what another passage says:

- 2: it states the specific fact the question asks for - the value, the command, the name, the
  reason, the steps - or, for a question with several parts, one of those parts in full. A reader
  could answer that part from this passage without guessing.
- 1: it is on the question's topic but does not state that fact: it says the topic exists, gives
  context or a related fact, or answers vaguely where the question is specific ("can be
  expensive" for "how much more expensive").
- 0: it is not relevant to the question.

A table of contents or an index entry pointing at the answer, a figure caption, or a heading
without the text under it, is a 1 at most.

The question: {query}

Reply with a JSON array and nothing else - no prose, no code fence - one element per passage, in
order: [{{"n": 1, "grade": 2}}, {{"n": 2, "grade": 0}}, ...]

The passages:

{passages}
