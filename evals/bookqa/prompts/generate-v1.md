You are building an evaluation dataset for a document search engine. Below is part of the source
document `{source}` ({segment_note}). Read it, then write {count} questions a real reader of this
document might type into a search box, each with the one answer the document gives.

Mix the kinds of question, as the text allows:

- `direct`: asks what one passage states, in words close to the document's.
- `paraphrase`: asks the same kind of thing in the reader's own words, not the document's.
- `multi_fact`: needs two or more facts, which may sit in different passages.
- `nearby_sections`: needs a passage together with one from a neighbouring section.

Make about one question in five unanswerable: a question a reader of this part would plausibly
ask, on its topic, that the document does not answer - a detail it never gives, a system or
version it does not cover. Mark it `"answerable": false`, give no passages, and set
`expected_answer` to what the document does and does not say about it.

For every answerable question:

- `expected_answer`: one canonical answer, one or two sentences, stated as the document states
  it. Not a list of alternatives.
- `expected_facts`: the atomic facts the answer needs, each a short phrase.
- `passages`: every passage the answer rests on, each with:
  - `quote`: a contiguous excerpt copied character for character from the text below, one to
    three sentences, at least eight words. Copy it exactly, including its wording and
    punctuation; do not fix typos, join hyphenated words or skip text in the middle.
  - `page`: {page_rule}
  - `section`: the heading the quote sits under, as the document writes it, or "" if none.

Questions must be answerable from the document alone, without general knowledge, and must not
mention "the document", "this section" or "the text": a reader searching does not know where the
answer is. Do not ask about the table of contents, page numbers, figures you cannot see, or the
document's bibliography. Vary the questions; no two may ask the same thing.

Variation seed: {seed}. Use it to vary which passages you pick when several would serve.

Reply with a JSON array and nothing else - no prose, no code fence. Each element:

{{"query": "...", "query_type": "direct|paraphrase|multi_fact|nearby_sections",
 "answerable": true, "expected_answer": "...", "expected_facts": ["..."],
 "passages": [{{"quote": "...", "page": 12, "section": "..."}}]}}

The text of `{source}` ({segment_note}):

{text}
