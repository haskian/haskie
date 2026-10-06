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

Each section says what it is about with its `descriptors`, fixed at indexing (`sections.build`).
They never decide what is picked.

Beside the sections, the map lists the documents the search reached hardest (`documents`): every
scanned chunk folded per document by `score_fold`, best first, and every other document a listed
section is in, so each section's document has its row. A document the map picked nothing from can
still lead them.

With a reranker on, the search's model (`reranker_model`) scores every scanned chunk first
and drops none: the scores are the demand weights, the section scores and the document scores.
A section that only shares a word with the topic then counts for little. Without one, the fused
retrieval scores stand, which weigh the chunks nearly alike.

No IO here: `retrieval.map_sections` reads the chunk placements, the corpus mean, the picks'
descriptors, the documents' descriptions and every listed document's memberships.

[1] "GeoRAG." arXiv preprint 2606.29328, 2026, the demand-weighted objective without its
    sub-queries (its "(1,0)" ablation). https://arxiv.org/abs/2606.29328
"""

import bisect
from collections import Counter
from collections.abc import Mapping, Sequence

import msgspec
import numpy as np

from haskie.collection.index import Hit, SpanKey, location
from haskie.indexing.chunk import HEADING_SEP
from haskie.search import collapse, section
from haskie.search.passage import document_score, fold, pages
from haskie.sections.descriptors import Description
from haskie.settings import ScoreFold

PER_DOCUMENT = 2  # sections of one document while others have some left: Google's site cap
# The cap holds a document back only for a section of another that still covers at least this
# share of the best gain left: a hard cap spent slots on sections barely on the topic. Judgement,
# measured once: on 8 questions over 3 books, 0.5 kept the coverage and relevance of no cap (0.637
# of demand, 96% of the top sections' relevance) where a hard cap lost both (0.627, 92%).
CAP_SHARE = 0.5
RELATED = 5  # related sections listed under one pick
# documents a map lists past the ones its sections are in: a shortlist, not a page of results
DOCUMENTS = 10


class Placed(msgspec.Struct, kw_only=True):
    """Where a section of the map is, and how well it matched."""

    collection: str
    document_id: str
    document: str  # its name, what it is cited by
    header: str  # its heading path, ready to cite; empty for a document with no headings
    id: str  # the section's id: what a search keeps to by `section_ids`
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
    descriptors: list[str]  # what it is about (`pipeline.descriptors`)
    description: str = ""
    related: list[Related] = []


class MappedDocument(msgspec.Struct, kw_only=True):
    """One document the search reached, and how hard: what to read, or to search alone."""

    document_id: str
    document: str  # its name, what it is cited by
    description: str  # what `describe_document` says it is about; empty when nobody did
    score: float  # every matched chunk of it the search read, folded by `score_fold`
    chunks: int  # how many of its chunks matched: a sum grows with how much a document says
    sections: int  # how many of the map's sections are in it; 0 when the map picked none
    collections: list[str]  # every searched collection holding it, in name order
    markdown_file: str  # the whole document on disk
    source_file: str  # the file it was imported from


class SectionMap(msgspec.Struct):
    """The answer of `search_sections`: the sections, in the order they were picked, and the
    documents the search reached, best first."""

    sections: list[MappedSection]
    documents: list[MappedDocument]
    collections: list[str]  # the fewest that together hold every section and document above
    searched: list[str]  # the collections the search covered, in the order they were chosen
    # the question, when its best match is under the bar its models were measured at: the map is
    # full whatever is asked, so this says it may hold nothing on the topic (`gaps.weak_questions`)
    uncovered: list[str] = []


class Candidate(msgspec.Struct):
    """One section the scan reached, with the scanned chunks in it."""

    collection: str
    document_id: str
    document: str  # its name
    path: tuple[str, ...]
    id: str  # the section's id (`section.Section.id`)
    placements: list[section.Placement]  # its chunks, first to last
    at: list[int]  # the positions of its scanned chunks in the scan, best first
    score: float


def candidates(
    hits: list[Hit],
    placed: dict[section.Place, section.Placements],
    max_chars: int,
    how: ScoreFold,
) -> list[Candidate]:
    """The sections the hits (best first) fall in, in the order their best hit ranks. A section
    counts once by its span of the document: one document chunked two ways in two collections
    has one section there."""
    found: dict[SpanKey, Candidate] = {}
    for at, hit in enumerate(hits):
        chunks = placed[(hit.collection, hit.document_id)]
        where = section.section_of(chunks, hit.seq, max_chars)
        first = bisect.bisect_left(chunks, where.seq_start, key=lambda one: one.seq)
        last = bisect.bisect_right(chunks, where.seq_end, key=lambda one: one.seq)
        placements = chunks[first:last]
        key = (hit.document_id, placements[0].char_start, placements[-1].char_end)
        if key not in found:
            found[key] = Candidate(
                hit.collection,
                hit.document_id,
                hit.document,
                where.path,
                where.id,
                placements,
                [],
                0.0,
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
    vectors: Sequence[collapse.Vector], centre: np.ndarray | None, candidates: Sequence[Candidate]
) -> np.ndarray:
    """How closely each section covers each scanned chunk [chunks, sections]: the centred cosine
    of the chunk to the section's nearest scanned chunk, clipped at 0. A section covers its own
    chunks wholly."""
    unit = collapse.unit_rows(vectors)
    if centre is not None:
        unit = collapse.unit_rows(unit - centre)
    alike = np.clip(unit @ unit.T, 0.0, 1.0)
    return np.stack([alike[:, one.at].max(axis=1) for one in candidates], axis=1)


def cover(weights: np.ndarray, near: np.ndarray, candidates: Sequence[Candidate], k: int) -> Picked:
    """Greedy relevance-weighted facility location (see the module). `weights` holds each scanned
    chunk's relevance, `near` how closely each section covers each chunk (`nearness`), `candidates`
    the sections in rank order: the first pick is the most relevant section, and a tie in gain
    goes to the better-ranked one. It stops early only when no section left covers anything new.
    """
    if not candidates:
        return Picked(picks=[], related={}, coverage=[], lifted=0)
    total = float(weights.sum())
    demand = weights / total if total > 0 else np.full(len(weights), 1 / len(weights))
    first = max(range(len(candidates)), key=lambda at: (candidates[at].score, -at))
    picks, covered = [first], near[:, first].copy()
    coverage, lifted = [float(demand @ covered)], 0
    while len(picks) < min(k, len(candidates)):
        gains = demand @ np.maximum(near - covered[:, None], 0.0)
        gains[picks] = -np.inf
        fresh = gains > 0
        within = fresh & ~_capped(candidates, picks) & (gains >= CAP_SHARE * gains.max())
        if not fresh.any():
            break  # every section left repeats what the picks cover
        lifted += not within.any()
        best = int(np.argmax(np.where(within if within.any() else fresh, gains, -np.inf)))
        picks.append(best)
        covered = np.maximum(covered, near[:, best])
        coverage.append(float(demand @ covered))
    return Picked(
        picks=picks,
        related=_related(demand, near, candidates, picks),
        coverage=coverage,
        lifted=lifted,
    )


def _capped(candidates: Sequence[Candidate], picks: list[int]) -> np.ndarray:
    """Which sections belong to a document that has `PER_DOCUMENT` picks already."""
    taken = Counter(candidates[at].document_id for at in picks)
    return np.asarray([taken[one.document_id] >= PER_DOCUMENT for one in candidates])


def _related(
    demand: np.ndarray, near: np.ndarray, candidates: Sequence[Candidate], picks: list[int]
) -> dict[int, list[tuple[int, float]]]:
    """Each section left out, under the pick that covers its scanned chunks best (demand-weighted
    mean of `near`), when that pick covers them at all; the `RELATED` closest per pick."""
    # each section's share of each chunk's demand
    weights = np.zeros((len(candidates), len(demand)))
    for at, one in enumerate(candidates):
        weights[at, one.at] = demand[one.at]
    closeness = (weights @ near[:, picks]) / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    found: dict[int, list[tuple[int, float]]] = {pick: [] for pick in picks}
    chosen = set(picks)
    for at in range(len(candidates)):
        best = int(np.argmax(closeness[at]))
        if at not in chosen and closeness[at, best] > 0:
            found[picks[best]].append((at, float(closeness[at, best])))
    return {
        pick: sorted(near_, key=lambda pair: -pair[1])[:RELATED] for pick, near_ in found.items()
    }


def by_rank(texts: Sequence[str], candidates: Sequence[Candidate], k: int) -> Picked:
    """The sections by relevance, when there are no vectors to cover by: the same per-document
    cap, and a section whose matched chunks say what a pick's say (`collapse.Worded.alike` at
    `collapse.TAU_WORDS` or above) related to that pick instead. `texts` are the scanned hits'."""
    words = collapse.Worded(texts)
    order = sorted(range(len(candidates)), key=lambda at: (-candidates[at].score, at))
    picks: list[int] = []
    related: dict[int, list[tuple[int, float]]] = {}
    taken: Counter[str] = Counter()
    waiting: list[tuple[int, int]] = []  # held back by the cap or by `k`: (section, picks seen)
    lifted = 0
    for relaxed in (False, True):  # the second pass: every other document is spent
        queue = [(at, 0) for at in order] if not relaxed else waiting
        waiting = []
        for at, seen in queue:
            repeat = _repeat(words, candidates, at, picks[seen:])
            if repeat is not None:
                related[repeat[0]].append((at, repeat[1]))
            elif len(picks) < k and (relaxed or taken[candidates[at].document_id] < PER_DOCUMENT):
                picks.append(at)
                related[at] = []
                taken[candidates[at].document_id] += 1
                lifted += relaxed
            else:
                waiting.append((at, len(picks)))
    trimmed = {pick: related[pick][:RELATED] for pick in picks}
    return Picked(picks=picks, related=trimmed, coverage=[], lifted=lifted)


def _repeat(
    words: collapse.Worded, candidates: Sequence[Candidate], at: int, kept: list[int]
) -> tuple[int, float] | None:
    """The first pick section `at` repeats, and how alike the two are."""
    for other in kept:
        alike = words.alike(candidates[at].at, candidates[other].at)
        if alike >= collapse.TAU_WORDS:
            return other, alike
    return None


def mapped(
    candidates: Sequence[Candidate],
    picked: Picked,
    described: Mapping[tuple[str, str], Description],
) -> list[MappedSection]:
    """The picks as the answer lists them, each with its descriptors (`described`, by collection
    and section id; none when the collection's cache entry has no such section)."""
    found: list[MappedSection] = []
    for pick in picked.picks:
        one = candidates[pick]
        about = described.get((one.collection, one.id), Description())
        found.append(
            MappedSection(
                **_placed(one),
                depth=len(one.path),
                seq_start=one.placements[0].seq,
                seq_end=one.placements[-1].seq,
                chars=one.placements[-1].char_end - one.placements[0].char_start,
                chunks=len(one.at),
                descriptors=about.descriptors,
                description=about.description,
                related=[
                    Related(**_placed(candidates[at]), similarity=round(closeness, 4))
                    for at, closeness in picked.related[pick]
                ],
            )
        )
    return found


def listed(ranked: list[list[Hit]], mapped: set[str]) -> list[list[Hit]]:
    """The documents a map lists, best first: the best `DOCUMENTS` of `ranked` (each a document's
    hits, `passage.top_documents`), and every other one a listed section is in (`mapped`, by
    document id), so each section's document has its row, the file to open with it."""
    return [
        group for at, group in enumerate(ranked) if at < DOCUMENTS or group[0].document_id in mapped
    ]


def documents(
    groups: list[list[Hit]],
    sections: Sequence[MappedSection],
    held: Mapping[str, list[str]],
    about: Mapping[str, str],
    how: ScoreFold,
) -> list[MappedDocument]:
    """The documents the search reached hardest (`passage.top_documents`), in the order given,
    each with how many of the map's `sections` are in it, the collections that hold it (`held`,
    by document id; the collection whose table matched it when it was not looked up) and its
    description (`about`)."""
    picked = Counter(one.document_id for one in sections)
    return [
        MappedDocument(
            document_id=best.document_id,
            document=best.document,
            description=about.get(best.document_id, ""),
            score=document_score(group, how),
            chunks=len(group),
            sections=picked[best.document_id],
            collections=held.get(best.document_id, [best.collection]),
            markdown_file=best.markdown_file,
            source_file=best.source_file,
        )
        for group in groups
        for best in group[:1]
    ]


def _placed(one: Candidate) -> dict:
    """The fields of `Placed` for a candidate."""
    first, last = one.placements[0], one.placements[-1]
    return {
        "collection": one.collection,
        "document_id": one.document_id,
        "document": one.document,
        "header": HEADING_SEP.join(one.path),
        "id": one.id,
        "location": location(one.document, *pages(one.placements), first.line_start, last.line_end),
        "line_start": first.line_start,
        "line_end": last.line_end,
        "score": one.score,
    }
