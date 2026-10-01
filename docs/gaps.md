# Gaps

The Gaps page lists the questions your shelf did not answer, grouped by topic, with the searches
that asked them and what came closest. Replay asks them again after you add a document, so you
can see a gap close before you mark it resolved.

It reads the search log. Every search is written there, and nothing is judged when it is written.

## The search log

```mermaid
flowchart LR
    handler["<b>handler</b><br/>log.capturing"] --> plan["<b>plan</b><br/>scope, mode, limit"]
    plan --> rank["<b>rerank</b>, per question<br/>query vector, best cosine,<br/>best reranker score"]
    rank --> answer["<b>answer</b><br/>results, folded places,<br/>uncovered questions"]
    answer --> write[("searches<br/>search_questions<br/>search_results")]
```

A handler opens a capture around the search (`log.capturing`). The search fills it in through a
context variable, so no step passes it along. Two places observe what they already hold:

- the flow's plan, or the full-text listing: the collections searched, the mode that ran, and
  the limit;
- the flow's `rerank` step, which every ranked search runs once per question: the question's
  query vector, the profile it was embedded under, and its score profile. That profile holds the
  20 best cosines between the query and the rows read (`log.similarities`), and the reranker's 20
  best scores before its floor dropped any. The best cosine and the best reranker score are the
  heads of those lists.

The best cosine is measured on the vectors, not read from a score column. A hybrid query's fusion
keeps only a rank score, and a rank says nothing about how close the best row came.

The capture is written when the search ends, in one transaction: a `searches` row, one
`search_questions` row per question asked, and one `search_results` row per place returned.
An excerpts search also keeps `missing_terms`, the words of its questions no excerpt held.
`scoped` marks a search kept to some documents or sections.
Places are stored in preorder with their parent, so the `also_in` trees survive; an excerpt's
places are its passages' repeats. Each keeps its citation (`header`, `location`), so the log reads
without the document. A failed search is written with its error, then the error is raised again.
A search without a session is written too. A later page of `/api/search/text` is the same search
and writes nothing. A malformed question is a refused request, not a search, and writes nothing.

`GET /api/searches` reads the log back, newest first, and an agent reads it as the MCP tool
`list_searches`; neither ever sends a query vector. A session's history reads its searches from
the log. The Insights chart counts every search, a search without a session under "no session".
The nightly run deletes searches older than `retention.search_days` (90 by default; 0 keeps
everything).

## Which questions are gaps

`search/gaps.py` judges each stored question on read, so a bar measured again re-judges every
search already stored. A question is a gap when a detector fires. The detectors run in the order
below, and the first that fires names the signal:

| signal | when |
| --- | --- |
| `reported` | the agent that asked it said the excerpts do not answer it (`report_gap`), fully or in part. It reads the excerpts, so its verdict outranks every score |
| `empty` | its search returned nothing |
| `uncovered` | several questions were asked at once, and no excerpt answers this one |
| `weak` | its best match is under the bar. A reranked search is judged by the floor it dropped chunks under: the settings' `min_rerank_score` when one is set, else the reranker's calibrated floor (`reranker_calibration`), because the reranker reads query and passage together. A map drops nothing, so it is judged by its own `map_reranker_model`'s calibrated floor, never by `min_rerank_score`. Otherwise the profile's `weak_match` cosine decides. No bar known: no verdict |
| `borderline` | no reranker judged it, and its best cosine sits from `weak_match` up to `answered_match`. It may be answered. `list_gaps` leaves it out unless `signals` asks for it, and the page folds these topics away |

A failed search is an error, not a gap. A search kept to some documents or sections
(`document_ids`, `section_ids`; the row's `scoped`) is judged by `reported` alone: what it
missed may sit in the documents it kept out. A new signal is one `Signal` member and one detector
function.

`report_gap` takes the session, the question as asked and a verdict, `insufficient` or `partial`,
with a note on what the excerpts lacked. It lands on the newest question with those words in that
session, within the last hour, and only on a search that ran. Its wording follows the
sufficient-context test: could a careful reader answer from these excerpts alone?

Gap questions are grouped into topics by leader clustering, newest first, as `collapse` folds
results. Each joins the closest topic whose newest question it matches, so a chain of near matches
never merges two topics that do not match each other. Two questions match when both were embedded
under one profile with a `same_topic` bar and their query cosine clears it. Otherwise only the same
words match. Topics rank by how many times they were asked, then by the newest.

The page and an agent share the routes: `list_gaps`, `replay_gaps`, `review_gaps` and
`report_gap` are MCP tools too, so an agent that adds a document can check the gap closed and
resolve it.

Replay asks each question again on its own, as `search_excerpts` over every collection with the
context it had. Every collection, not the scope it had: the question is whether the shelf answers
it now, and the new document often sits in a new collection. Nothing is written to the log.
Dismiss and resolve set `search_questions.review` on every question of the topic, so one part of
a search of several stays open while another is dismissed; reopen clears it.

## The bars

The cosine bars live in the catalogue next to the duplicate cosines:
`embedding_profiles.weak_match`, `answered_match` and `same_topic`. A best cosine under
`weak_match` is `weak`; from it up to `answered_match`, `borderline`; over it, an answer. A
reranked search uses the floor it dropped chunks under instead, which the log keeps when the
settings override the reranker's, and has no band. `catalogue/seed.sql` records how each bar was
measured. Each is set so that no answered question is flagged: a false gap costs the curator's
trust, a missed one waits for the next search.

`mise run evaluate-gaps` measures them (`tests/gapeval/`). It chunks and embeds two shelves, each
read at a pinned commit: "The Rust Programming Language" (Apache-2.0 or MIT; 45 answered
questions, 40 unanswered, 30 of them near its topics) and four of haskie's docs (12 answered, 5
unanswered). For each question it takes the score profile a search would log, and scores every
predictor in its `FEATURES` by AUROC, the chance an answered question scores above an unanswered
one.

| model | answered, lowest | unanswered, highest | bars |
| --- | --- | --- | --- |
| granite-97m-multilingual, best cosine | 0.881 (Rust), 0.793 (docs) | 0.884 (Rust), 0.826 (docs) | weak 0.79, borderline to 0.885 |
| ettin-32m, best reranker score | 0.99988 (Rust), 0.9991 (docs) | 0.99982 (Rust), 0.99998 (docs) | floor 0.05, uncalibrated |

The findings below were first measured with models the catalogue no longer holds (bge-small,
arctic-embed-m, MiniLM-L-6); where the current defaults were measured too, both are given.

What the measurements say:

- **The best cosine stays.** The mean of the top 5 (`mean5`) ranks answered over unanswered a
  little better (bge-small: AUROC 0.999 against 0.998 on the Rust book, 1.000 against 0.983 on
  the docs; granite-97m: 0.999 against 0.997, and 0.967 for both). Under a bar that flags no
  answered question on either shelf, it caught 32 of 45 against 30 for bge-small, but 13 against
  18 for arctic-m. The top-two gap and the spread barely separate (AUROC 0.64 and 0.86).
- **A bar does not travel between shelves, or between versions of one.** The Rust book's lowest
  answered cosine would flag 3 of 12 answered docs questions. An edit to the docs alone moved
  their lowest answered cosine from 0.698 to 0.673. The bars sit under both shelves, each read at
  its pinned commit.
- **The band catches what the low bar misses.** For granite-97m, 0.79 to 0.885 holds the 27
  unanswered questions the low bar misses, with 8 of 57 answered (14%); for bge-small, 0.67 to
  0.775 held 15 with 5 of 57. For arctic-m the scores overlapped too far: no band held the missed
  ones under 15% of answered, so it had none.
- **The reranker ranks best, but its floor needs calibrating.** ettin-32m ranks every answered
  Rust book question over every unanswered one (AUROC 1.000), the docs shelf less well (0.733).
  Its scores crowd near 1: the Rust book's unanswered questions score 0.997 on median, because
  they sit near its topics. So at 0.05 its floor catches none of 45, where MiniLM-L-6 caught 36.
  A floor under every answered question (0.999) would catch 39, but the floor also drops chunks
  from every search, so it is left to `calibrate-rerankers` on borderline pairs.
- **Missing words do not tell a wording gap from a missing document.** The idea was that a
  borderline question whose words no near miss holds (`missing_terms`) exists under other words.
  But every unanswered question has such words (45 of 45), so the rule would label 14 of 45 true
  content gaps "wording", and would catch only 12 of 20 questions asked in words the docs do not
  use (`reworded` in `tests/gapeval`). Not shipped.
- **Shared near misses do not group topics.** Joining two gap questions by a lower cosine plus
  shared near misses joined 94% of same-topic pairs on the Rust book, against 89% by cosine alone,
  but merged 3 to 13 pairs of different topics on the docs shelf: on a small shelf, every
  unanswerable question lands on the same few chunks. Only identical near misses merge nothing,
  and they add nothing. Not shipped.

A profile without `weak_match` gives no cosine verdict; its questions can still be `reported`,
`empty` or `uncovered`.

Because a bar does not travel, a home can measure its own. `mise run calibrate-gaps sample`
writes the questions its searches logged, each with its best cosine and near misses. A person
marks each one answered or not. Then `measure --profile NAME --write` sets the two bars by the
same rules: no answered question under the low bar, and a band only while it flags at most 15% of
answered ones. It needs 10 labelled questions of each kind.

Code: `search/log.py`, `search/gaps.py`, `api/gaps.py`, `catalogue/seed.sql`,
`web/src/pages/Gaps.tsx`.
