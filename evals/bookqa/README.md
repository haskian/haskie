# Book query retrieval eval

How well haskie's search ranks the passages that answer realistic questions about the source
books. Retrieval only: a question goes to haskie's HTTP search, and its top 10 passages are scored
against the gold quotes that answer it. No agent, no language model, no judge.

This suite is separate from the agent eval in `evals/` (`mise run eval:run`). It has its own
dataset, code, reports and mise tasks, and neither suite runs the other. It shares only the source
books (`evals/setup.py`'s `SOURCES`, downloaded into `evals/corpus/`) and the eval instance's HTTP
helpers.

## Two phases

**Phase 1: generate the dataset, once, by hand.** `mise run eval:bookqa:generate` cuts each book
into segments: a PDF by its chapters (its outline) in runs of at most 12 pages, any other format
in runs of paragraphs. It then asks Claude (`claude -p`, no tools) for questions about a few
segments. Claude sees the segment's own text, page-marked, and replies with a question, one
canonical expected answer, the atomic facts the answer needs, and the verbatim quotes that support
it, each with its page and section. About one question in five is unanswerable.

What Claude writes is a **candidate**, never gold. Candidates land in
`evals/bookqa/candidates/<book>/<segment>-<key>.jsonl`, to be reviewed. Only
`--accept` moves the candidates that pass every check into `evals/bookqa/dataset.jsonl`.

**Phase 2: evaluate retrieval, as often as you like.** `mise run eval:bookqa:run` reads the
reviewed `dataset.jsonl` and nothing else. It never generates, regenerates or judges. It indexes
the cited books into this suite's own collection (`bookqa-books`) and searches every question
once per mode. It then scores the ranking and writes `evals/bookqa/reports/<timestamp>/`
(`report.md`, and `outcomes.jsonl` with every ranking).

## Commands

```sh
mise run eval:bookqa:generate -- --source raft.pdf --segments 2 --dry-run  # what would be asked
mise run eval:bookqa:generate -- --source raft.pdf --segments 2            # write candidates
mise run eval:bookqa:review -- --candidates                                # check the candidates
mise run eval:bookqa:generate -- --source raft.pdf --segments 2 --accept   # accept what passes
mise run eval:bookqa:review                                                # check the dataset
mise run eval:bookqa:run                                                   # every mode
mise run eval:bookqa:run -- --modes fts hybrid+rerank
mise run eval:bookqa:check                                                 # lint, types, tests
```

`generate` options: `--source` (repeatable; the default is every book), `--segments N` (per
book, drawn by the seed; `0` for all), `--per-segment N` (questions per segment), `--seed`,
`--model` (default `sonnet`, or `BOOKQA_MODEL`), `--dry-run` and `--accept`. `run` needs a haskie
instance with an embedding profile for its vector modes. It starts the agent eval's embedding
instance (`HASKIE_EVAL_EMBED_URL`, default `http://127.0.0.1:8124`) and finishes its first run
with `compact` if it has none (`--profile`).

## Reviewing candidates

`review` checks what a machine can check:

- every line is a valid record;
- no duplicate ids or questions (compared case- and punctuation-free);
- an answerable record cites a passage, a fact and its own source;
- an unanswerable record cites nothing;
- the source's hash is unchanged since generation;
- every quote is in the source, on the page it names.

It cannot check whether an answer is right. Nor can it check whether an "unanswerable" question is
really unanswerable. Claude saw one segment, not the whole book. The first raft generation marked
"what log size triggers a snapshot?" unanswerable from pages 1-12, and page 13 answers it. Read
every candidate, the unanswerable ones against the whole book, and edit or delete a candidate
line before `--accept`.

Generation is deterministic by key. Each generation is keyed by the source's hash, the segment,
the seed, the model and the prompt version (`generation_key`). A key already on disk is never
asked again, so the same inputs give the same records, ids included. A new seed, model or prompt
version (`prompts/generate-vN.md`, `PROMPT_VERSION`) is a new generation. A question already in
the dataset or another candidate is dropped.

## Record schema

One JSON object per line (`schema.Record`, `SCHEMA_VERSION` 1):

```json
{
  "id": "raft-ddee11-03",
  "source": "raft.pdf",
  "query": "How many servers does a typical Raft cluster have, and how many failures can it tolerate?",
  "query_type": "multi_fact",
  "answerable": true,
  "expected_answer": "A typical Raft cluster has five servers, which can tolerate the failure of any two.",
  "expected_facts": ["five servers is typical", "tolerates two failures"],
  "relevant_documents": ["raft.pdf"],
  "relevant_passages": [
    {"document": "raft.pdf", "page": 5, "section": "5.1 Raft basics",
     "quote": "A Raft cluster contains several servers; ﬁve is a typical number, ..."}
  ],
  "meta": {"schema_version": 1, "source_sha256": "5a5c...", "model": "claude-sonnet-5",
           "prompt_version": "v1", "seed": 1, "segment": "p001-012",
           "generated_at": "2026-09-29T17:01:55+00:00"}
}
```

- `query_type` is `direct`, `paraphrase`, `multi_fact` or `nearby_sections`.
- `page` is the 1-based physical PDF page the quote starts on, the page haskie's `page_start`
  counts. It is `null` for a source without pages.
- An unanswerable record has `answerable: false`, no documents and no passages. Its
  `expected_answer` says what the book does and does not say.

## What the report measures

A result counts for a gold passage when it is from the same document and holds at least half of
the quote's four-word runs, compared letters and digits only. That makes it immune to hyphenation,
ligatures and haskie's markup. It also covers a quote that haskie cut at a chunk boundary.

- **Recall@1/5/10**: the share of a question's gold passages matched in the top k.
- **MRR**: the reciprocal rank of the first match.
- **nDCG@10**: binary gains, each gold passage credited once.
- **doc@10**: a relevant document anywhere in the top 10.
- **empty**: answerable questions that got no result at all.
- **kB**: the response body's size.
- **ms p50**: the median latency.

Unanswerable questions are scored apart and never count toward recall. **abstained** counts those
that got no result, which is the right answer, and **top score** is the best result's score, on
that mode's own scale. Only a mode with a reranker floor can return nothing, so `fts`, `vector`
and `hybrid` never abstain.

Tables are grouped by mode, by mode and source, and by mode and query type. The modes are set as
search overrides on `bookqa-books`:

- `fts`: full text, no reranker;
- `vector`;
- `hybrid`;
- `hybrid+rerank`: hybrid with the cross-encoder.

## Layout

| file | what |
| --- | --- |
| `generate.py` | phase 1: segments, prompt, `claude -p`, candidates, `--accept` |
| `prompts/generate-v1.md` | the generation prompt, versioned by file name |
| `sources.py` | the books as text: hash, pages, segments, where a quote sits |
| `schema.py` | the record, JSONL load/dump, every validation |
| `review.py` | phase 1's checks, standalone |
| `run.py` | phase 2: modes, HTTP search, outcomes |
| `metrics.py` | passage matching, Recall@k, MRR, nDCG@10, abstention |
| `report.py` | the grouped markdown tables |
| `dataset.jsonl` | the reviewed dataset: gold, versioned |
| `candidates/` | unreviewed generations, local only: what is accepted is versioned in `dataset.jsonl` |
| `reports/` | run output, not versioned |
