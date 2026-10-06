You are building an evaluation dataset for a document search engine. Below is part of the source
document `{source}` ({segment_note}). Read it, then write {count} questions a real reader of this
document might type into a search box about how two things the document discusses relate, each
with the one answer the document gives.

A relationship has two sides, X and Y, and one of these relations between them:

- `solves`: X is a solution to the problem Y.
- `mitigates`: X reduces the impact of Y but does not eliminate it.
- `causes`: X introduces or leads to the problem Y.
- `trades_off`: choosing X sacrifices Y; X and Y cannot both be maximized.
- `alternative_to`: X and Y are competing approaches to the same goal.
- `requires`: X only works correctly if Y is in place.
- `implemented_via`: X is typically implemented using the technique Y.
- `complements`: X is commonly used together with Y.
- `fails_when`: X breaks down or becomes a bad choice under condition Y.
- `challenges`: X contradicts or undermines the assumptions of Y.
- `generalizes`: X is a more general form of Y.
- `correlates_positively`: as X increases, Y tends to increase.
- `correlates_negatively`: as X increases, Y tends to decrease.
- `analogous`: X plays the same role in one domain as Y does in another.

Pick only relationships the document itself states or plainly shows, not ones general knowledge
supplies. Prefer pairs whose two sides are explained in different places - different sections,
or paragraphs far apart - so that finding the answer means finding both. Use different relations
across the questions where the text allows; do not force a relation the text does not support.

Ask the way a reader would: "What problem does X solve?", "When does X stop being a good
choice?", "What does X give up to get Y?", "What has to be in place for X to work?". Do not name
the relation in the question, and do not put Y's own words in a question that asks for Y.

For every question:

- `relation`: one of the relations above.
- `expected_answer`: one canonical answer, one or two sentences, naming X, Y and how they relate,
  as the document states it. Not a list of alternatives.
- `expected_facts`: the atomic facts the answer needs, each a short phrase: at least one about X
  and one about Y.
- `passages`: at least two passages, at least one for each side, each with:
  - `quote`: a contiguous excerpt copied character for character from the text below, one to
    three sentences, at least eight words. Copy it exactly, including its wording and
    punctuation; do not fix typos, join hyphenated words or skip text in the middle.
  - `page`: {page_rule}
  - `section`: the heading the quote sits under, as the document writes it, or "" if none.

Questions must be answerable from the document alone and must not mention "the document", "this
section" or "the text": a reader searching does not know where the answer is. Do not ask about
the table of contents, page numbers, figures you cannot see, or the bibliography. No two
questions may ask about the same pair.

Variation seed: {seed}. Use it to vary which pairs you pick when several would serve.

Reply with a JSON array and nothing else - no prose, no code fence. Each element:

{{"query": "...", "query_type": "relationship", "relation": "solves", "answerable": true,
 "expected_answer": "...", "expected_facts": ["..."],
 "passages": [{{"quote": "...", "page": 12, "section": "..."}}]}}

The text of `{source}` ({segment_note}):

{text}
