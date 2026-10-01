# Search

`search_excerpts`, `search_sections` and `GET /api/search/explore` share one
ranking, then run the fold their answer needs. `search/flow.py` builds these pipelines in one
screen. A search of one collection is the same search with `collections` set to it: the collection
page asks `explore` for passages. `/api/search/text` is the one search on its own path, a separate
BM25 search that sorts every collection's chunks by their raw score and folds no repeats.

```mermaid
flowchart LR
    q(["query"]) --> scope["<b>scope</b><br/>collections argument,<br/>else session, else all"]
    scope --> retrieve["<b>retrieve</b><br/>per collection:<br/>hybrid, vector or fts"]
    retrieve --> merge["<b>merge</b><br/>several collections:<br/>each half ranked over all,<br/>then fused"]
    merge --> rerank["<b>rerank</b><br/>optional<br/>cross-encoder"]
    rerank --> hits["<b>hits</b><br/>cut to scan depth"]
    hits --> fchunks["collapse hits"] --> chunks(["chunks"])
    hits --> franges["merge neighbours,<br/>grow or drop short ones,<br/>collapse ranges"] --> read["read the spans"] --> passages(["passages"])
    hits --> jranges["merge neighbours,<br/>find short ones,<br/>fold repeats"] --> group["group by section"] --> budget["cut to<br/>the budget"] --> probe["search again for<br/>missing words"] --> fill["fill around<br/>and between"] --> rerankx["rerank whole excerpts<br/>(experiment)"] --> excerpts(["excerpts"])
    merge -. "a small reranker weighs,<br/>drops none" .-> mapsec["group by section,<br/>pick to cover the scan;<br/>rank documents"] --> sections(["sections"])
```

## The shared ranking

1. **Scope.** The `collections` argument, else the session's collections, else all of them.
   Narrower, when given: `document_ids` and `section_ids` (see "Keeping to documents and
   sections" below).
2. **Retrieve.** Each collection runs `hybrid` (vector and BM25), `vector` or `fts`. A search of one
   collection lets LanceDB fuse the two halves; a search of several reads them apart, for the merge.
   Without an embedding model everything is `fts`. The query is embedded once, and up to 8
   collections are read in parallel. A collection still building its first full-text index has no
   BM25 half yet: `hybrid` answers with the vector half alone, and `fts` finds nothing there. A
   document on its way out of a collection answers from none of its rows there: a membership
   `removing`, or a document `deleting`. One query per search reads those names, and each read
   filters them out before its limit.
3. **Merge.** A search of several collections ranks each half over all of them at once: the
   vector half by distance, which one query embedding makes comparable, and the BM25 half by
   score. It then fuses the two as LanceDB fuses one table's, by the `fusion` setting: `rrf`
   (reciprocal rank fusion of the two ranks) or `linear` (a weighted sum, each half min-max
   scaled). A chunk that two collections share counts once: the same span of one document,
   whatever `seq` each collection's chunk settings give it. A search over one collection keeps
   that collection's own scores.

   It used to fuse one ranking per collection instead. Rank fusion reads only ranks, so every
   collection's first chunk scored the same however far it was from the query, and each
   collection took an equal share of the scan. On a shelf of 10 books in 3 collections, a
   collection of 3 Python books took exactly 100 of the 200 chunks scanned for "How does leader
   election work in a replicated log?", which none of them answers. Without a reranker, the
   documents a search reached hardest then had the top document a cross-encoder would rank first
   in 6 of 11 questions. Ranked as one table, they have it in 11 of 11.
4. **Rerank** (optional). A cross-encoder rescores the merged `candidates`.

The last of these steps to run sets a chunk's score. With a reranker on, it is the sigmoid of the
cross-encoder's logit, 0 to 1, so switching the mode only changes which `candidates` it reads: a
chunk found in both modes scores the same. The sigmoid is there because most logits are negative,
and a passage folds its chunks' scores (below): a negative one would subtract under `sum` and zero
the whole under `harmonic`. Without a reranker, a chunk keeps its mode's own score: BM25, `1 / (1 +
squared L2 distance)`, or the fused score of `rrf` or `linear`. A hybrid search of several
collections fuses over all of them at once; one half alone keeps its own score. Passages, excerpts
and documents then fold chunk scores their own way.

Each step that sets or changes a score says how as it runs (`search/scoring.py`), the way each
step's time goes into `Server-Timing`. A search answers with that lineage, in pipeline order, in
its `X-Score-Lineage` header: JSON, percent-encoded, one `{step, label, rule}` per step. A step
that left the scores alone says nothing. The web UI shows it beside a result's score.

An excerpts search of several questions ranks each question at once, side by side. Each step of
those runs carries `branch=Q1`, `branch=Q2` and so on in `Server-Timing`, numbered in the order
the questions were asked; a shared step carries none. The web UI draws the branches as one block,
and its server total adds the slowest branch only, since they ran at the same time.

The ranking scans deeper than the answer. A folded repeat frees its slot for the next result,
several chunks go into one passage, several passages into one excerpt, and many chunks into one
section of a map.

## The four answers

| shape | what it is | who asks for it |
| --- | --- | --- |
| chunk (`Hit`) | one indexed chunk and its score | `explore?granularity=chunk`; also returned, without folding, by `/api/search/text` |
| passage | neighbouring matched chunks of one section, merged | `explore?granularity=passage` |
| excerpt | one section of a document, with every passage of it the search kept | `search_excerpts` (the Explore page too) |
| section | one section of a document: its id, where it is, its descriptors, the sections it covers; no text. Beside the sections, the documents the search reached hardest | `search_sections` (the Explore page too, which opens the section with the sections it covers, and its document at its heading) |

A passage is the text its chunks cover, read by their offsets, with nothing added around it.
Chunks are cut at headings, blank lines, blocks and sentences (see [chunking](chunking.md)), so a
passage starts and ends where the author did. Each one carries its `header` (heading path) and
its `location` (`doc p.3-4 L10-20`) to cite.

`search_sections` scores a document by folding the scores of the chunks it matched (see "How
chunk scores fold" below). It also returns a small set of collections that holds every section
and document listed, ready for `set_session_collections`. The set comes from the standard greedy
approximation of set cover, so it is small but not guaranteed smallest. `search_excerpts` and
`search_sections` name the collections they searched (`searched`), so a session's scope, set turns
earlier, shows in each answer. Both list in `uncovered` a question the sources match only weakly
(see "Words the answer never mentions" below).

## Excerpts: passages grouped by section

A search keeps passages one at a time, so three passages under one heading would come back as
three results competing for the slots. `search/section.py` groups them instead. An excerpt is one
section of one document, holding every passage the search kept in it, in document order. `limit`
counts excerpts, so the passages below the cut join the sections they belong to, and a section
that no kept passage opens takes no slot.

The section is the largest one that still reads as a quote. A passage's heading path is tried from
the top. A level whose section holds the whole document (a title over everything) says nothing, and
a level whose section is longer than `max_section_chars` (12,000) is split one heading down. So a
book groups by chapter or by section, and a short note by its title. A passage under the deepest
heading it has takes that section, however long. The sections come from where each chunk of the
document sits: its `seq`, heading path, char span and the ids of its sections, read from the
collection's table in one LanceDB query per collection. A chunk never spans two sections, so the
grouping is exact. An excerpt names its section by id (`section_id`), and each of its spans the
deepest section its chunks sit in.

The excerpt's `text` joins its passages. Each passage opens with the headings it sits under that
the one before it did not, below the section's own `header`, as markdown headings of their depth.
`[…]` marks text between two passages that the search did not keep. `spans` lists the passages,
each with its own `header`, `location`, lines, offsets, score, `aspects` and `also_in`. The
excerpt's own offsets and lines run from the first passage to the last, and its `score` is the best
passage's.

## Words the answer never mentions

A search ranks by the whole question, so the part most of it is about can fill every slot while a
word the rest hangs on goes missing: "keep inventory consistent when an order is placed", answered
by five passages on orders and none on inventory. `search/probe.py` looks for each word of the
question (stopwords and words under three letters aside) in the kept sections' text and headings. A
text holds a word when one of its words has the same stem, by the Snowball English stemmer that
LanceDB's full-text index uses. So "keeps" holds "keep", "deployment" holds "deploy" and
"consistency" holds "consistent", while "category" does not hold "cat". The words none of them
holds are searched for once more by BM25 alone, and the best passage that search finds that no
section holds yet joins the answer. It joins the kept section it belongs to, or comes after the
others as one excerpt past `limit` and past the budget, which the sections were cut to before it.
Taking the last ranked section's slot instead would trade one gap for another, and a search of one
excerpt would lose its whole answer. With a reranker on, what the search finds is judged as the
ranked chunks are: scored against each question whose words are missing, dropped under the
reranker's floor (see "Several questions at once"), and tagged with the questions it clears.
Without one, it tags no question, since holding a missing word is not an answer. It scores 0, as
the fill's chunks do: its BM25 score is on another scale than the ranked passages', and would sort
it above them.

`search_excerpts` answers with `excerpts`, `uncovered` and `missing_terms` (the words still missing
after the probe). `uncovered` lists the questions no excerpt names, when several were asked, and
any question, one alone included, whose best match is under the bar its models were measured at
(`gaps.weak_questions`, the verdict the Gaps page gives as weak). A search finds the nearest
passages even on a topic the sources never cover, so without it a single question had no signal.
The search log keeps only the first kind, so the Gaps page still reads a weak question as weak. The
probe's passage tags no question: holding a missing word is not an answer. A synonym defeats the
probe: it asks only for the words the question used. Each search that probes logs
`search_probe` with the words, whether it found a passage, and whether that joined a kept section.
The Explore page shows the missing words and questions under the results.

## Filling around and between passages

The text between two kept passages, or just past them, often finishes the answer: the list under
"three rules:", the paragraph that explains a term. It did not rank, so `search/fill.py` weighs it.
Every chunk of the section within `max_passage_grow` (3) chunks of a kept passage is scored against
the question, by the query vector when every row has one, else by the question's words. Its value
is its score around the kept chunks' own: 0 for one as good as the median kept chunk, 1 for one as
good as the best, below 0 for a weaker one, clipped to [-1, 1]. With several questions a chunk
takes its best question's value. A filled chunk tags no question: its value is relative to this
search's own kept chunks, so it would mark a question answered that nothing answers. Only the
reranker's judgement, or the ranking a part ran, tags (`aspects.tagged`).

- A gap between two passages is filled when its values sum above 0, and the two become one. So a
  gap of up to twice `max_passage_grow` chunks can be filled. The joined passage scores by
  `score_fold` over the chunks of both.
- Every passage grows outward by the run of chunks next to it whose values sum highest, when that
  is above 0: into a gap as far as its half, so the passages on either side never reach for the
  same chunk.

The same rule grows a short passage of the `passages` answer (next section), so a search has one
way of growing a passage, one scale of value and one setting for how far. A passage grows once:
an excerpts search only judges its short passages before the slots are counted, and the fill
grows every passage, short ones included.

This is the arithmetic of Relevant Segment Extraction [3]: a weak chunk comes in only when stronger
ones around it pay for it. Where a gap is not filled, `[…]` stays.

With a reranker on, `fill_values = absolute` (an experiment) values a chunk as dsRAG does instead:
the reranker scores it against every question, one pass a question. Its value is its best score,
spread by the reranker's calibrated beta curve so the scores fall about evenly over 0 to 1, less
0.18 (dsRAG's balanced preset). It needs no kept chunks to compare with, and it trusts the
calibration: an uncalibrated reranker's curve is the identity. dsRAG also decays a chunk by its
rank; here every chunk was scored, so none is. It costs one reranker pass a question over the
chunks near every section, and short passages grow the same way.

`grow_bias` (0, from -1 to 1) says how eagerly passages grow. It is added to every value, however
the value was reached, before the values are summed (`fill.biased`). Above 0 a weaker chunk is
worth taking, so passages grow further and more gaps fill. Below 0 only a stronger chunk is. At -1
even a chunk as good as the best is worth 0, so nothing grows. Under `fill_values = absolute` the
bias moves dsRAG's penalty: 0.1 makes it 0.08. The same bias applies to short passages.

`max_answer_chars` (36,000) bounds what one excerpts search returns. Right after grouping, a
`budget` step cuts the sections to it, the last first, though the first section always stays
(`search_budget` logs how many went). The fills then go in, worth most per character first, while
they fit the room left. By default the fill does not ask the reranker even when one is on: scoring
every chunk near every section against every question would take seconds. Each search logs
`search_fill` with the signal and what it added.

## Short passages

A passage under `min_passage_chars` (300), or under 7 words, is thin: a section's lead-in ("Three
rules:", with the list in the next chunk), or a section's last line. For the `passages` answer,
`search/thin.py` grows it by the rule every excerpt's passages grow by: on each side, the run of up
to `max_passage_grow` (3) chunks whose values sum highest, when that is above 0. It never grows
past a heading. A neighbour's value is its score around the scanned hits' own: 0 for one as good as
the median, 1 for one as good as the best, scored by the reranker when one is on, else by the
cosine to the query vector, else by the share of the question's words it holds, then moved by
`grow_bias`. The floor is this search's own, so it needs no calibration per model. An excerpts
search only judges it here: is a run worth taking next to it? It leaves the growing to the fill,
and marks the passage `owed` its growth. The fill values the same chunks against the kept
passages rather than the scanned hits, and may find none worth taking. An owed passage the fill
found no run for is then alone after all (`thin.settle`), and counts as a thin passage that took
nothing (below). One the fill found a run for keeps standing, even when the budget ran out. When
a section is dropped this way, its slot is not handed on: the sections were counted before the
fill.

A thin passage that took nothing is too short to stand alone. As a passage it is dropped, and its
slot goes to the next result. As part of an excerpt it stays when another passage of its section is
kept, and a section with no other passage is no excerpt. Two thin passages stay even when nothing
around them matches:

- the best result of the search, because a short exact answer is still an answer;
- a whole section, with a heading or the document's edge on both sides, such as a short note.
  Nothing of it is missing. Where one part of a PDF ends and the next begins (`part`, see
  [chunking](chunking.md)) is no such edge: the section goes on in the next part.

The neighbours are read only when a passage is thin, in one LanceDB query per collection. Each
search logs `search_thin` with the signal used, how many passages grew, how many stayed alone, and
how many chunks they took. With several questions, each question's passages are grown or dropped
against that question.

## Sections: a map of the shelf

`search_sections` answers "what do my sources hold on this, and nearby?" before any text is
read. It returns sections, each with where it is and what it is about, and no text, and the
documents the search reached hardest: an agent reads the map, then asks `search_excerpts` about the
sections worth reading. It runs `retrieve -> merge -> rerank -> hits -> map_sections`
(`search/section_map.py`):

1. **A small reranker weighs every chunk.** With a reranker on, the map's own model
   (`map_reranker_model`, MiniLM-L2 by default) scores every chunk of the scan, 20 chunks per
   section asked for, up to 200. It drops none: the scores are the demand weights below, the
   section scores and the document scores. A floor would narrow the map; a weight only makes a
   section that shares a word with the topic count for little. Without a reranker the fused
   retrieval scores stand, which weigh the top chunk only about four times the 200th.
2. **Group by section.** Each scanned chunk joins the section an excerpt would quote it in
   (`section.section_of`), so a section on the map is the one `search_excerpts` returns. One span
   of one document counts once, whichever collections hold it.
3. **Pick to cover the scan.** Relevance-weighted facility location: every scanned chunk is a
   demand point weighed by its share of the scan's relevance, and a section covers it as closely
   as its nearest chunk is, `max(cos, 0)`. The first pick is the most relevant section; each next
   is the section that covers the most demand the picks leave uncovered, so a near copy of a pick
   adds nothing and is not picked. It stops at `limit` (15 by default, at most 40), or when no
   section covers anything new. Without an aspect list, this is the best-supported coverage
   method in the literature we follow: in GeoRAG's ablation [9] it beat maximal marginal
   relevance (MMR) and determinantal point processes (DPP) by 3 to 5 points of exact match, and
   roughly matched a cross-encoder. Greedy takes under a millisecond at 200 chunks.
4. **Centred cosines.** Facility location reads a cosine as an amount, and embedding cosines carry
   an offset that depends on the model (bge's random pairs sit near 0.3, e5's near 0.7). So the
   vectors are centred on the mean chunk vector of the collections searched, which maintenance
   stores per collection (`collections.vector_sum`), then clipped at 0. Not on the scan's own
   mean, which would take out the topic the scan shares. A collection not yet maintained has no
   mean, and the raw cosine stands (`search_map` logs `centred`).
5. **At most two sections of one document** while another document has a section left that
   covers at least half the best gain left, so one long book does not fill the map (Google's site
   cap, engineering rather than measured). A section barely on the topic does not take a slot for
   the sake of variety: without that condition the cap spent slots on sections scored 0.007 beside
   ones scored 0.15. Otherwise the cap does not apply.
6. **Without vectors** (full text only) nothing measures how close two chunks are, so no
   embedding-based selector runs: sections go by relevance with the same cap, and a section whose
   matched words repeat a pick's (word Jaccard 0.5 or more, as the collapse uses) is related to it.

Each pick lists up to five `related` sections: those it covers best, by the relevance-weighted
mean of their chunks' nearness to it. Every section the scan reached and the map did not pick goes
under the one pick closest to it, when that nearness is above 0, and each pick keeps its five
closest.
There is no threshold, so `related` means nearby, not repeated. A near copy of the pick lands
there. So does a section the pick only sits close to, and that one may be the best section on the
topic: the pick covered its chunks by their vectors, not by what it says.

`related` and `also_in` look alike and answer different questions:

| | `related` (sections) | `also_in` (excerpts) |
| --- | --- | --- |
| question | what else is near this pick? | where else is this same point? |
| entry rule | not picked, and closest to this pick | passes a repeat test with a threshold |
| label | `similarity` only | `relation`: `duplicate`, `contained` or `equivalent` |
| typical content | full: up to five per pick | often empty |
| can the agent skip it? | no: read the headers | yes: it adds nothing new |

Without vectors the rule is stricter: a section joins `related` only when its matched words repeat
the pick's (step 6), so there it does mean repeated.

`documents` lists, best first, the ten documents the search reached hardest
(`section_map.DOCUMENTS`), and any other document a listed section is in, so every section's
`document_id` has its row. A document's score folds every scanned chunk of it, by `score_fold`. The
default is a sum, so `chunks` says how much a document says. Each row also carries the document's
description, how many of the map's sections are in it, the collections that hold it, and its files.
The agent's view of a section names its document by `document_id` alone: the document's name,
collections and `markdown_file` are on its row. A document the map picked no section from can still
lead the list: the cover spreads the picks, the documents are not spread. The answer's `collections`
holds every section listed, related ones included, and every document, by greedy set cover, ready
for `set_session_collections`.

`descriptors` say what each section is about. They are fixed at indexing
([Indexing](indexing.md#the-three-workflows)) and read by the section's id from the cache entry
the collection indexed the document from: by default one to five of the section's terms (a word,
or two words side by side) weighed against the other sections of its depth by c-TF-IDF [10], less
the terms more than half of them use or its header holds (unless nothing else is left), and
reranked by meaning [11]; under the llm setting, up to five topics a small language model names
after reading the section. They never decide what is picked: in the studies we follow, clusters
of the pool used as aspects gained nothing, and terms mined from it only re-weighted the aspects
already on top.

Measured on three books (1.9 MB of markdown, bge-small), eight questions: the whole search took
35 to 50 ms warm, `map_sections` 12 to 17 ms of it. Against the top sections by relevance on the
same scans (0.627 of the relevance-weighted demand covered, 2.0 documents, 7.6 chapters):

| selection | demand covered | documents | chapters | relevance of the top sections |
| --- | --- | --- | --- | --- |
| facility location, no cap | 0.638 | 2.1 | 7.6 | 96% |
| with a hard cap of two | 0.627 | 2.6 | 7.6 | 92% |
| with the cap held at half the best gain (shipped) | 0.637 | 2.2 | 7.8 | 96% |

That is a sanity check, not an evaluation: the gains are small, as the literature says they are
without aspects, and the half is a judgement fitted on these eight questions.

The merge across collections and the map's reranker were measured on 10 books in 3 collections
(arctic-m), on 11 questions of the learning-cycle assessment. Reference for document order: the
book a cross-encoder (mxbai-rerank-xsmall) ranks first.

| | off-domain picks | on-domain books per map | back-matter picks | top document as the cross-encoder's | rerank time |
| --- | --- | --- | --- | --- | --- |
| one ranking per collection, no reranker | 42 of 132 | | 9 | 6 of 11 | 0 |
| one table, no reranker | 22 of 132 | 3.9 | 7 | 11 of 11 | 0 |
| one table, MiniLM-L2 weighs (shipped) | 5 of 132 | 2.9 | 6 | 10 of 11 | 1.5 s |
| one table, MiniLM-L6 weighs | 4 of 132 | 2.9 | 5 | 10 of 11 | 3.8 s |

An off-domain pick is a Python book's section on an architecture question, or the reverse. The
times were taken on a machine under heavy load (load average 55): what holds is L2 against L6,
about 2.5 times faster. The map is narrower with a reranker because the questions one book owns
(leader election, package stability) now map to that book, which the assessment asked for.
Two things tried and dropped:
- A scan capped at a fifth per document, over a pool twice as deep, spread the books (4.8 to 6.5
  with 5 chunks or more) before the map reranked. With the reranker it added 0.2 books and mixed
  two score scales, so it went.
- `score_fold = max` instead of `sum` changed 1 pick of 132, and put the cross-encoder's top
  document first in 8 of 11, against 10 of 11 for `sum`.

MiniLM-L2's best score on the 11 answered questions is 0.113 at the lowest; on three questions no
book covers it is 0.0 (espresso), 0.003 (Java garbage collection) and 0.861 (the Linux scheduler,
against asyncio's scheduling). Its uncalibrated floor of 0.05 flags none of the answered and two
of the three, which is the verdict a map's `uncovered` carries.

Each search logs `search_map` with the chunks, the sections reached and picked, the share of
demand covered after each pick, whether it centred, and the documents whose sections were missing.

## Keeping to documents and sections

`search_excerpts`, `search_sections` and `GET /api/search/explore`, which searches chunks and
passages, take a scope besides the collections: `document_ids` keeps to these documents, and
`section_ids`, except in `search_sections`, to these sections and every section under them. Both
together keep to what both allow. The ids are the ones the answers carry: a hit's `document_id`
and `section_ids`, an excerpt's and a span's `section_id`, and a mapped section's `id`
([Storage](storage.md#sections-and-their-ids)). An id that is not 22 base58 characters is refused
(422), and so are more than 100 of either.

The scope is a filter on every read of a collection's chunk table (`index.Scope`), applied before
the limit, so a row outside it takes no slot. A chunk carries the ids of every section that holds
it, so a chapter's id finds the chunks of its subsections (`array_has_any`). The reads a search
makes around its matches keep to it too: the neighbours a short passage grows into, the text the
fill adds around and between passages, and the chunks the excerpts are grouped by. An excerpt
kept to one section quotes that section alone.

## How chunk scores fold

A passage, a document and a document's section each hold several matched chunks, and the
`score_fold` setting (`passage.fold`) turns their scores into one:

| rule | score | who scores this way |
|---|---|---|
| `sum` (default) | every matched chunk adds, so more matching text ranks higher | Vespa's chunk example: `sum(chunk_scores())` [4] |
| `max` | the best chunk alone | Elasticsearch `semantic_text`: "the most relevant passage will be used to compute a score" [5] |
| `harmonic` | harmonic(best, sum): between the best and twice it, so many weak chunks never outrank one strong one | haskie's own rule; no source measures it against the other two |

A chunk the search did not rank, one a passage grew into or the fill added, scores 0 and adds
nothing under any rule. An excerpt scores its best passage. The near-duplicate fold still measures
two results' overlap as the harmonic mean of its two directions, whatever `score_fold` is: that is
an overlap, not a score of matched chunks. Every search's score lineage names the rule it used.

### Reranking whole excerpts (experiment)

Folding chunk scores judges an excerpt by its parts, never by the text an agent reads. Recent
long-document rerankers score the unit they return instead: SumRank ranks condensed documents
[7], and EBCAR scores passages with their context [8]. With `rerank_excerpts` on and a reranker
chosen, an excerpts search scores each finished excerpt, its heading path in front, against the
questions it answers, one reranker pass a question, and the best of them is its score. A single
question's excerpts are then sorted by it; several keep the order their turns gave them.

It runs only when every excerpt fits what the reranker reads, estimated at four characters a
token: the catalogue's context, or less where the loader cuts shorter. Every reranker today reads
512, though the models read 8,192: haskie cuts each pair there (`onnx_models.MAX_PAIR_TOKENS`,
`mlx_models.MAX_PAIR_TOKENS`). An excerpt cut short would be
scored on its opening alone, beside others scored whole, so if one does not fit the chunk scores
stand. Each search logs `search_rerank_excerpts` with whether it ran and how long it took. It is
off until an evaluation shows it returns more answer per character than the fold.

## Folding repeats

A pointwise reranker scores one passage at a time, so it cannot see that two results repeat each
other [1]. `search/collapse.py` folds them in the chunk and passage pipelines, once per search, as
the last fold before the answer. The fold walks the results best first and compares each one only
with the results already kept (leader clustering). Comparing only with kept results stops chains,
so A close to B and B close to C never merges A with C.

Each new result is compared with each kept result in turn:

```mermaid
flowchart TD
    pair(["new result vs one kept result"]) --> same{"same document and<br/>collection, touching?"}
    same -- yes --> skip["not a repeat of this one:<br/>one passage cut in two"]
    same -- no --> text{"the same text,<br/>whitespace aside?"}
    text -- yes --> dup["repeat: <b>duplicate</b>"]
    text -- no --> doc{"same document,<br/>other collection,<br/>one span inside the other?"}
    doc -- "kept one inside new one" --> swap["repeat: <b>contained</b>;<br/>the fuller new one takes the slot"]
    doc -- "new one inside kept one" --> cont["repeat: <b>contained</b>"]
    doc -- "no" --> contain{"containment,<br/>each way"}
    contain -- "kept one inside new one" --> swap
    contain -- "new one inside kept one" --> cont
    contain -- "both ways" --> eq["repeat: <b>equivalent</b>"]
    contain -- "neither" --> alike{"alike as a whole?"}
    alike -- yes --> eq
    alike -- no --> skip2["not a repeat of this one"]
```

A duplicate is an exact character match once whitespace is collapsed, in any document. Both texts
need at least 7 words, so two equal headings do not fold. Equivalent means the same meaning in
other words: a nearly identical vector, or nearly the same words where words decide.

A new result that repeats no kept result takes a new slot, while fewer than `limit` are taken.
A passage too short to stand alone (see "Short passages") never leads a fold. Nothing folds under
it, and it never takes a fuller passage's slot. Its section is dropped when no other passage
stands in it, and a passage under it would be dropped too.

The containment and alike tests run in up to two spaces, and the first space that finds a repeat
decides. A `vector` search decides by the embedding space alone, as it ranks by vectors alone.
Hybrid and full-text searches use both:

- **Embedding space**, used when every result has a vector and the model has duplicate
  thresholds. Containment is the mean of each chunk's best cosine against the other result.
  Alike is the cosine of the two results' mean vectors.
- **Word space**, always used. Containment is the share of one result's three-word shingles found
  in the other (threshold 0.8). Alike is the Jaccard of their words (threshold 0.5, a common
  near-duplicate line [1]). A result with fewer than 5 shingles never matches here.

Cosine thresholds are set per embedding profile (`duplicate_chunk` and `duplicate_passage` in
`catalogue/seed.sql`), because a raw cosine means different things for different models. No
profile has them yet, so every profile folds by words alone. For granite-97m no threshold could
work: two different neighbouring chunks of the Rust book have a median cosine of 0.951, and one
paragraph under two heading paths scored 0.921. The containment threshold is a judgement call, and the
code says so.

A folded result becomes an `also_in` entry under the result it repeats, and `also_in` is a tree.
When a fuller result takes a slot (the superset swap), the old one moves under it with everything
folded into it. The fuller one takes the old one's score too, and the score lineage says so. Each
place stays under the place it was measured against. Each entry carries its `relation` to its
parent. It also carries `to_parent` and `to_root`, measured by words and by embedding:
`contained`, `contains`, `alike`, and `score`, the harmonic mean of the two directions.
That score is the Dice coefficient for words and the F1 of the best chunk matches for embeddings.

## Several questions at once

When the parts of a question are answered in different places, one search of the whole question
tends to fill every slot with one part. The reranker scores one passage at a time, so it never sees
that another part went unanswered. So `search_excerpts` takes `q` as a list: one question, or 2 to
5 parts of one, each at most 500 characters. An optional `context` of at most 200 characters is the
background the parts share. The query embedding reads it in front of each part, as it reads a chunk
under its heading path, so the context steers which candidates come up. Everything that matches
words reads the part alone: full-text search, the word scores, and the reranker, which sets the
final order. The context's words would otherwise make every part match every passage that shares
them: "ddd" over a book on DDD ranks "Can I DDD?" above what a part about anti-patterns asks, and a
part the sources say nothing about would look answered. `rerank_with_context` (off by default)
puts the context in front of each part for the reranker too.

```mermaid
flowchart LR
    asked(["q: 2-5 parts<br/>+ context"]) --> each["<b>shared ranking</b><br/>per part, in parallel:<br/>context + part"]
    each --> ranges["merge neighbours,<br/>find short ones,<br/>per part"]
    ranges --> turns["<b>take turns</b><br/>round-robin<br/>over the parts"]
    turns --> fold["collapse ranges<br/>across all parts"]
    fold --> tag["tag each passage<br/>with its parts"]
    tag --> group["group by section"] --> budget["budget"] --> probe["probe"] --> fill["fill"] --> rerankx["rerank whole<br/>excerpts"] --> excerpts(["excerpts"])
```

Each part runs the shared ranking on its own, as deep as one search of that `limit` would go. Then
the parts take turns at the slots in the order they were asked. On its turn, a part takes its best
range not yet picked. Each part is owed its own best `ceil(limit / parts)` ranges. A pick made for
another part counts for this part too when it overlaps one of those ranges, and the part sits out
one round for each such pick. A range that overlaps a pick, or continues it in the same section,
joins that pick, so two parts that land on one passage get one passage. Round-robin reads ranks
alone, so it works with or without a reranker and embeddings. The Text REtrieval Conference (TREC)
retrieval-augmented generation (RAG) track pipelines give each query its slots the same way [2].
Near-duplicates then fold across all the parts, once, as in one search, and the passages group by
section, the sections taking the slots in the order the parts picked them.

Each span's `aspect_scores` gives how well its passage matched every part whose own ranking holds
its chunks, or those of a place folded into it: that part's best such chunk. The best chunk rather
than a fold over them, so a part's score stays on the reranker's 0 to 1 scale, the one
`min_rerank_score` is set on; a fold of three strong chunks would read 2.5.

With a reranker on, its score has a scale: under its floor the reranker judged a chunk no answer,
and it is dropped from the ranking before passages are built. The floor is `min_rerank_score` when
set, else the reranker's own (`reranker_calibration` in the catalogue): the average score it gives
30 to 50 pairs a person judged borderline relevant, the way Cohere sets a relevance threshold [6].
`mise run calibrate-rerankers` measures it on your own collections; until then every reranker starts
at 0.05, judged on one book with the MiniLM-L-6 the catalogue once held, and says so in the score
lineage. For Ettin it drops almost nothing: ettin-32m scores a question the shelf does not answer
at logit 5.9 on median (`docs/gaps.md`). A part's tag in
`aspects` then means the reranker judged the passage an answer to it. Each chunk of the passage
scores its best part, and the passage folds those scores by `score_fold` like any passage. Without a
reranker the scores of two parts share no scale, so `aspects` lists the parts the passage ranked
high for: the part that picked it, every part that joined it or ranks it among its own owed ranges,
and those of every place folded into it; it keeps the score it was picked with. A vector or hybrid
search finds nearest passages for any part, even one the sources say nothing about, so without a
reranker a tag is not proof of an answer. An excerpt's `aspects` joins its spans', and its
`aspect_scores` holds each part's best. A part no excerpt lists found nothing, and the answer's
`uncovered` names it. One question, or a list that deduplicates to one, is the single search, its
context read the same way, and its `aspects` is empty. A `limit` below the number of parts is
refused (422).

## Full-text search

`GET /api/search/text` is one BM25 query over every collection, or over the comma-separated
`collections`, with no model and nothing to set up. It ignores the session's selection. Scores
are raw BM25, because one scorer with one tokenizer puts every collection on one scale. It is
paged with an offset cursor bound to the query (at most 1,000 results deep). A collection still
building its first full-text index contributes nothing: LanceDB refuses a BM25 query without it.

## Settings

`limit`, `candidates`, `mode`, `fusion`, `rrf_k`, `vector_weight`, `bm25_weight`, `nprobes`,
`refine_factor`, `reranker`, `reranker_model`, `map_reranker_model`, `rerank_with_context`,
`min_rerank_score`, `rerank_excerpts`, `score_fold`, `min_passage_chars`, `max_passage_grow`,
`grow_bias`, `fill_values`, `max_section_chars` and `max_answer_chars` each have a user default and
a description in the UI, and a collection can override them. The UI groups the ones that decide how
passages expand under Expansion: `min_passage_chars`, `max_passage_grow`, `fill_values`,
`grow_bias`, `max_section_chars` and `max_answer_chars`. In the shared ranking, each collection
retrieves with its own overrides: `mode`, `nprobes` and `refine_factor`. The settings of the merged
ranking (`fusion`, `rrf_k`, the two weights, `candidates`, the rerankers) and of every step after it
come from the collection only when it is the one collection in scope, and from the user otherwise.
`limit` comes from the call, else from the same place, for chunks and passages. The other answers
have fixed defaults of their own: `search_excerpts` 10 sections (`flow.DEFAULT_EXCERPTS`) and
`search_sections` 15 sections and 10 documents. No route takes search settings per call: a search
with other settings is a search of a collection whose overrides say so.

Code: `search/flow.py`, `search/retrieval.py`, `search/passage.py`, `search/collapse.py`,
`search/aspects.py`, `search/thin.py`, `search/section.py`, `search/fill.py`, `search/probe.py`,
`search/section_map.py`, `sections/`.

## References

1. Schlatt, F. et al. "Set-Encoder: Permutation-Invariant Inter-Passage Attention for Listwise
   Passage Re-Ranking with Cross-Encoders." *ECIR*, 2025. https://arxiv.org/abs/2404.06912
2. Samuel, S. et al. "Beyond Relevance: On the Relationship Between Retrieval and RAG Information
   Coverage." *ICTIR*, 2026. https://arxiv.org/abs/2603.08819
3. D-Star AI. "dsRAG: Relevant Segment Extraction." GitHub, 2024.
   https://github.com/D-Star-AI/dsRAG/blob/main/dsrag/rse.py
4. Vespa. "Working with chunks." Vespa documentation, 2026.
   https://docs.vespa.ai/en/rag/working-with-chunks.html
5. Elastic. "semantic_text field type reference: chunking." Elasticsearch documentation, 2026.
   https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/semantic-text-reference
6. Cohere. "Best practices for using Rerank: interpreting results." Cohere documentation, 2026.
   https://docs.cohere.com/docs/reranking-best-practices
7. Feng, J. et al. "SumRank: Aligning Summarization Models for Long-Document Listwise Reranking."
   arXiv preprint, 2026. https://arxiv.org/abs/2603.24204
8. Yuan, Y. et al. "Embedding-Based Context-Aware Reranker." arXiv preprint, 2025.
   https://arxiv.org/abs/2510.13329
9. "GeoRAG." arXiv preprint, 2026. Facility location with demand weighed by relevance alone, its
   "(1,0)" ablation. https://arxiv.org/abs/2606.29328
10. Grootendorst, M. "c-TF-IDF." BERTopic documentation, 2026.
    https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html
11. Grootendorst, M. "Representation models": `KeyBERTInspired` and `MaximalMarginalRelevance`.
    BERTopic documentation, 2026.
    https://maartengr.github.io/BERTopic/getting_started/representation/representation.html
