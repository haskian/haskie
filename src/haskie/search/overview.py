"""A map of the sections a topic touches: which sections of which documents, and what each is
about, without their text.

The search scans deep (`flow.SECTION_SCAN`), groups the scanned chunks into the sections an
excerpt would quote (`section.section_of`), and picks `limit` of them to cover the scan rather
than to repeat its best part:

- **Relevance-weighted facility location** [1]. Every scanned chunk is a demand point, weighed by
  its share of the scan's relevance; each section covers a chunk as closely as its nearest chunk
  is to it, `max(cos, 0)`. The first pick is the most relevant section; each next one is the
  section that covers the most demand the picks so far leave uncovered. A section that repeats a
  pick covers nothing new, so it is not picked: it is listed under that pick as `related`, with
  the other sections whose chunks the pick covers best. Without aspects, this is the
  best-supported way to cover a question, ahead of MMR and DPP [1] (see docs/search.md).
- **Centred cosines.** Embedding cosines carry an offset that depends on the model, and facility
  location reads a cosine as an amount, so the vectors are centred on the mean chunk vector of the
  collections searched (`Collection.centre`), never on the scan's own mean, which would take out
  the topic they share. Before a collection's first maintenance there is no mean yet, and the raw
  cosine stands (logged).
- **At most `PER_DOCUMENT` sections of one document** while another document has a section left
  that covers at least `CAP_SHARE` of the best gain, so one long book does not fill the map, and
  a section barely on the topic does not take a slot for the sake of variety.
- **Without vectors** (full-text search) nothing measures how close two chunks are: the sections
  go by relevance with the same cap, and a section whose matched words are the same as a pick's
  (`collapse.TAU_WORDS`) is `related` to it.

Each section says what it is about twice: `keywords` against the other sections of its own
document, fixed at indexing (`outline.build`), and `distinct`, the few of them that set it apart
from the other sections the scan reached (`distinct`). Neither decides what is picked.

No IO here: `retrieval.map_sections` reads the outlines, the chunk spans and the corpus mean.

[1] "GeoRAG." arXiv preprint 2606.29328, 2026, the demand-weighted objective without its
    sub-queries (its "(1,0)" ablation). https://arxiv.org/abs/2606.29328
"""

import bisect
from collections import Counter
from collections.abc import Sequence
from operator import itemgetter

import msgspec
import numpy as np

from haskie.collection.index import Hit, SpanKey, location
from haskie.indexing.chunk import HEADING_SEP
from haskie.outline import keywords
from haskie.outline.build import Node
from haskie.search import collapse, section
from haskie.search.passage import fold, pages
from haskie.settings import ScoreFold

PER_DOCUMENT = 2  # sections of one document while others have some left: Google's site cap
# The cap holds a document back only for a section of another that still covers at least this
# share of the best gain left: a hard cap spent slots on sections barely on the topic. Judgement,
# measured once: on 8 questions over 3 books, 0.5 kept the coverage and relevance of no cap (0.637
# of demand, 96% of the top sections' relevance) where a hard cap lost both (0.627, 92%).
CAP_SHARE = 0.5
RELATED = 5  # related sections listed under one pick
DISTINCT = 5  # words per section that set it apart on this map


class Placed(msgspec.Struct, kw_only=True):
    """Where a section of the map is, and how well it matched."""

    collection: str
    document_id: str
    document: str  # its name, what it is cited by
    header: str  # its heading path, ready to cite; empty for a document with no headings
    location: str
    line_start: int
    line_end: int
    score: float  # its matched chunks' scores, folded by `score_fold`


class Related(Placed, kw_only=True):
    """A section the map did not pick, listed under the pick that covers it best."""

    # how closely the pick covers its matched chunks: the mean centred cosine to the pick's
    # nearest chunk, or the words both hold without vectors
    similarity: float


class MappedSection(Placed, kw_only=True):
    """One section of the map: where it is, how it matched, and what it is about."""

    depth: int  # how many headings deep: 1 for a chapter under no title, 0 for the whole document
    seq_start: int  # its first and last chunk in `collection`'s table, 1-based
    seq_end: int
    chars: int  # how long it is: what reading it with `search_excerpts` costs at most
    chunks: int  # how many of its chunks the search matched
    keywords: list[str]  # what it is about, against the other sections of its depth
    distinct: list[str]  # what sets it apart from the other sections on this map
    markdown_file: str
    related: list[Related] = []


class SectionMap(msgspec.Struct):
    """The answer of `search_sections`: the sections, in the order they were picked."""

    sections: list[MappedSection]
    collections: list[str]  # the fewest that together hold every document above


class Pooled(msgspec.Struct):
    """One section the scan reached, with the scanned chunks in it."""

    collection: str
    document_id: str
    document: str  # its name
    path: tuple[str, ...]
    entries: list[section.Entry]  # its chunks, first to last
    at: list[int]  # the positions of its scanned chunks in the scan, best first
    score: float


def pool(
    hits: list[Hit],
    outlines: dict[section.Place, section.Outline],
    max_chars: int,
    how: ScoreFold,
) -> list[Pooled]:
    """The sections the hits (best first) fall in, in the order their best hit ranks. A section
    counts once by its span of the document: one document chunked two ways in two collections
    has one section there."""
    found: dict[SpanKey, Pooled] = {}
    for at, hit in enumerate(hits):
        outline = outlines[(hit.collection, hit.document_id)]
        where = section.section_of(outline, hit.seq, max_chars)
        first = bisect.bisect_left(outline, where.seq_start, key=lambda one: one.seq)
        last = bisect.bisect_right(outline, where.seq_end, key=lambda one: one.seq)
        entries = outline[first:last]
        key = (hit.document_id, entries[0].char_start, entries[-1].char_end)
        if key not in found:
            found[key] = Pooled(
                hit.collection, hit.document_id, hit.document, where.path, entries, [], 0.0
            )
        found[key].at.append(at)
    for one in found.values():
        one.score = fold([hits[at].score for at in one.at], how)
    return list(found.values())


class Picked(msgspec.Struct):
    """Which sections were picked, in order, and which unpicked ones each covers best."""

    picks: list[int]
    related: dict[int, list[tuple[int, float]]]  # pick -> (section, similarity), closest first
    coverage: list[float]  # the share of the scan's relevance covered after each pick
    lifted: int  # picks made past the per-document cap, every other document being spent


def nearness(
    vectors: Sequence[collapse.Vector], centre: np.ndarray | None, pooled: Sequence[Pooled]
) -> np.ndarray:
    """How closely each section covers each scanned chunk [chunks, sections]: the centred cosine
    of the chunk to the section's nearest scanned chunk, clipped at 0. A section covers its own
    chunks wholly."""
    unit = collapse.unit_rows(vectors)
    if centre is not None:
        unit = collapse.unit_rows(unit - centre)
    alike = np.clip(unit @ unit.T, 0.0, 1.0)
    return np.stack([alike[:, one.at].max(axis=1) for one in pooled], axis=1)


def cover(weights: np.ndarray, near: np.ndarray, pooled: Sequence[Pooled], k: int) -> Picked:
    """Greedy relevance-weighted facility location (see the module). `weights` holds each scanned
    chunk's relevance, `near` how closely each section covers each chunk (`nearness`), `pooled`
    the sections in rank order: the first pick is the most relevant section, and a tie in gain
    goes to the better-ranked one. It stops early only when no section left covers anything new.
    """
    if not pooled:
        return Picked(picks=[], related={}, coverage=[], lifted=0)
    total = float(weights.sum())
    demand = weights / total if total > 0 else np.full(len(weights), 1 / len(weights))
    first = max(range(len(pooled)), key=lambda at: (pooled[at].score, -at))
    picks, covered = [first], near[:, first].copy()
    coverage, lifted = [float(demand @ covered)], 0
    while len(picks) < min(k, len(pooled)):
        gains = demand @ np.maximum(near - covered[:, None], 0.0)
        gains[picks] = -np.inf
        fresh = gains > 0
        within = fresh & ~_capped(pooled, picks) & (gains >= CAP_SHARE * gains.max())
        if not fresh.any():
            break  # every section left repeats what the picks cover
        lifted += not within.any()
        best = int(np.argmax(np.where(within if within.any() else fresh, gains, -np.inf)))
        picks.append(best)
        covered = np.maximum(covered, near[:, best])
        coverage.append(float(demand @ covered))
    return Picked(
        picks=picks,
        related=_related(demand, near, pooled, picks),
        coverage=coverage,
        lifted=lifted,
    )


def _capped(pooled: Sequence[Pooled], picks: list[int]) -> np.ndarray:
    """Which sections belong to a document that has `PER_DOCUMENT` picks already."""
    taken = Counter(pooled[at].document_id for at in picks)
    return np.asarray([taken[one.document_id] >= PER_DOCUMENT for one in pooled])


def _related(
    demand: np.ndarray, near: np.ndarray, pooled: Sequence[Pooled], picks: list[int]
) -> dict[int, list[tuple[int, float]]]:
    """Each section left out, under the pick that covers its scanned chunks best (demand-weighted
    mean of `near`), when that pick covers them at all; the `RELATED` closest per pick."""
    weights = np.zeros((len(pooled), len(demand)))  # each section's share of each chunk's demand
    for at, one in enumerate(pooled):
        weights[at, one.at] = demand[one.at]
    closeness = (weights @ near[:, picks]) / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    found: dict[int, list[tuple[int, float]]] = {pick: [] for pick in picks}
    chosen = set(picks)
    for at in range(len(pooled)):
        best = int(np.argmax(closeness[at]))
        if at not in chosen and closeness[at, best] > 0:
            found[picks[best]].append((at, float(closeness[at, best])))
    return {
        pick: sorted(near_, key=lambda pair: -pair[1])[:RELATED] for pick, near_ in found.items()
    }


def by_rank(texts: Sequence[str], pooled: Sequence[Pooled], k: int) -> Picked:
    """The sections by relevance, when there are no vectors to cover by: the same per-document
    cap, and a section whose matched chunks say what a pick's say (`collapse.Worded.alike` at
    `collapse.TAU_WORDS` or above) related to that pick instead. `texts` are the scanned hits'."""
    words = collapse.Worded(texts)
    order = sorted(range(len(pooled)), key=lambda at: (-pooled[at].score, at))
    picks: list[int] = []
    related: dict[int, list[tuple[int, float]]] = {}
    taken: Counter[str] = Counter()
    waiting: list[tuple[int, int]] = []  # held back by the cap or by `k`: (section, picks seen)
    lifted = 0
    for relaxed in (False, True):  # the second pass: every other document is spent
        queue = [(at, 0) for at in order] if not relaxed else waiting
        waiting = []
        for at, seen in queue:
            repeat = _repeat(words, pooled, at, picks[seen:])
            if repeat is not None:
                related[repeat[0]].append((at, repeat[1]))
            elif len(picks) < k and (relaxed or taken[pooled[at].document_id] < PER_DOCUMENT):
                picks.append(at)
                related[at] = []
                taken[pooled[at].document_id] += 1
                lifted += relaxed
            else:
                waiting.append((at, len(picks)))
    trimmed = {pick: related[pick][:RELATED] for pick in picks}
    return Picked(picks=picks, related=trimmed, coverage=[], lifted=lifted)


def _repeat(
    words: collapse.Worded, pooled: Sequence[Pooled], at: int, kept: list[int]
) -> tuple[int, float] | None:
    """The first pick section `at` repeats, and how alike the two are."""
    for other in kept:
        alike = words.alike(pooled[at].at, pooled[other].at)
        if alike >= collapse.TAU_WORDS:
            return other, alike
    return None


def distinct(found: Sequence[dict[str, int]], picks: list[int]) -> dict[int, list[str]]:
    """The `DISTINCT` keywords of each pick that set it apart from every section the scan
    reached: c-TF-IDF (`keywords.ctfidf`) with one class per section, over the keywords each
    section's outline stores with their counts (`outline.build.Node.keywords`). Its terms are the
    stored keywords, not the words of its matched chunks: those were weighed against the rest of
    their document and reranked by meaning already, while a few matched chunks are too little
    text to tell a rare word from a common one."""
    classes: list[Counter[str]] = []
    written: list[dict[str, str]] = []
    for one in found:
        counts: Counter[str] = Counter()
        forms: dict[str, str] = {}
        for form, uses in one.items():
            key = keywords.key_of(form)
            counts[key] += uses
            forms.setdefault(key, form)
        classes.append(counts)
        written.append(forms)
    weights = keywords.ctfidf(classes)
    return {
        pick: [written[pick][key] for key in keywords.best(weights[pick], DISTINCT)]
        for pick in picks
    }


def stored(pooled: Sequence[Pooled], outlines: dict[str, list[Node]]) -> list[Node | None]:
    """Each section's node in its document's outline (`outlines` by document id), None when there
    is none: the node under the
    same heading path whose span overlaps the section's most. The outline was cut from one
    chunking (`outline.store`) and the section from its collection's, which can bound a section a
    few characters apart, never onto another section of the same path."""
    by_path: dict[tuple[str, tuple[str, ...]], list[Node]] = {}
    for doc, nodes in outlines.items():
        for node in nodes:
            by_path.setdefault((doc, tuple(node.headings)), []).append(node)
    found: list[Node | None] = []
    for one in pooled:
        start, end = one.entries[0].char_start, one.entries[-1].char_end
        shared = [
            (min(end, node.char_end) - max(start, node.char_start), node)
            for node in by_path.get((one.document_id, one.path), [])
        ]
        chars, node = max(shared, key=itemgetter(0), default=(0, None))
        found.append(node if chars > 0 else None)
    return found


def mapped(
    hits: list[Hit],
    pooled: Sequence[Pooled],
    picked: Picked,
    nodes: Sequence[Node | None],
    words: dict[int, list[str]],
) -> list[MappedSection]:
    """The picks as the answer lists them, each with its node's keywords (`nodes`, one per
    section) and the few that set it apart (`words`, one list per pick)."""
    found: list[MappedSection] = []
    for pick in picked.picks:
        one, node = pooled[pick], nodes[pick]
        found.append(
            MappedSection(
                **_placed(one),
                depth=len(one.path),
                seq_start=one.entries[0].seq,
                seq_end=one.entries[-1].seq,
                chars=one.entries[-1].char_end - one.entries[0].char_start,
                chunks=len(one.at),
                keywords=list(node.keywords) if node is not None else [],
                distinct=words[pick],
                markdown_file=hits[one.at[0]].markdown_file,
                related=[
                    Related(**_placed(pooled[at]), similarity=round(closeness, 4))
                    for at, closeness in picked.related[pick]
                ],
            )
        )
    return found


def _placed(one: Pooled) -> dict:
    """The fields of `Placed` for a pooled section."""
    first, last = one.entries[0], one.entries[-1]
    return {
        "collection": one.collection,
        "document_id": one.document_id,
        "document": one.document,
        "header": HEADING_SEP.join(one.path),
        "location": location(one.document, *pages(one.entries), first.line_start, last.line_end),
        "line_start": first.line_start,
        "line_end": last.line_end,
        "score": one.score,
    }
