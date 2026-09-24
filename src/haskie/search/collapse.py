"""Near-duplicates folded into the result they repeat, so a slot goes to something new.

Five books making the same point would otherwise spend five slots on it. The reranker cannot see
that: it scores one result at a time. So after the ranking, the results are walked best first and
each one is compared only with the results already kept (leader clustering). One close enough to a
kept result is folded into it as an `also_in` pointer - where else to cite the same point - and
the next result down takes the freed slot. Comparing only with kept results is what stops a chain:
A close to B and B close to C never merges A with C.

It runs once per search, at the last fold before the answer: on hits for the `chunks` answer, and
on hit ranges (after `passage.ranges`) for passages. Folding twice would leave a folded chunk as a
hole in its own passage, and the pointers would depend on the order of the steps.

Two results are compared two ways, and either one is enough:
- chunk by chunk, as containment: how much of one result is found in the other. A one-chunk
  passage copied into a three-chunk one is contained in it, though their mean vectors differ.
- result by result: the cosine of their mean vectors, or the Jaccard of their words.

The comparisons are read off a `Space` built once for the whole scan: the chunks' own embedding
vectors when every row has one (`Embedded`), else their words (`Worded`), which catch copies but not
rewordings. The embedding space checks the words too, so identical text folds under any heading.
The cosine is raw, not centred on a corpus mean: for a yes-or-no threshold, centring bge or e5
vectors amounts to a shift of the threshold per model, so the threshold is set per model instead
(`EmbeddingModel.duplicate`, where the sources for its values are).

No IO: `retrieval` builds the space from the rows it read and hands it in.
"""

import re
from collections.abc import Callable, Sequence

import msgspec
import numpy as np

from haskie.collection.index import Hit, HitReference, location
from haskie.search.passage import HitRange, PassageReference
from haskie.settings import DuplicateCosine, EmbeddingModel

# word Jaccard: Set-Encoder (ECIR 2025, https://arxiv.org/abs/2404.06912) defines near-duplicate
# clusters as Jaccard > 0.5
TAU_WORDS = 0.5
# judgement, no source. Four in five of one result's words found in the other reads as
# a copy with a sentence added; the superset swap then keeps the fuller one.
TAU_CONTAINED = 0.8
# judgement, no source. Over half the smaller span shared means the two cut the same
# passage; exact, so no calibration beyond the choice of "half".
TAU_CHARS = 0.5
MAX_ALSO = 5  # pointers listed per result; `also_count` keeps the total

WORD = re.compile(r"\w+")


# --- the spaces -------------------------------------------------------------------


class Embedded:
    """The scanned chunks as unit vectors: cosine between chunks, and between the mean vectors of
    two groups of them, read off one matrix.

    It keeps their words too (`words`), and a copy the words find folds even when the cosine
    misses it. A chunk is embedded under its heading path (`chunk.framed`), so the same paragraph
    under two different headings embeds apart: 0.927 against a 0.92 bar, measured with bge-small.
    """

    kind = "embedding"

    def __init__(
        self, vectors: np.ndarray, duplicate: DuplicateCosine, texts: Sequence[str]
    ) -> None:
        self.cosines = vectors @ vectors.T
        self.tau_contained = duplicate.chunk
        self.tau_alike = duplicate.passage
        self.words = Worded(texts)

    def contained(self, inner: Sequence[int], outer: Sequence[int]) -> float:
        """How much of `inner` is in `outer`: its chunks' best cosines to any chunk of `outer`,
        averaged."""
        return float(self.cosines[np.ix_(inner, outer)].max(axis=1).mean())

    def alike(self, one: Sequence[int], other: Sequence[int]) -> float:
        """The cosine of the two groups' mean vectors. For unit vectors it is a ratio of sums over
        the cosine matrix, so the vectors are not needed again."""
        across = self.cosines[np.ix_(one, other)].sum()
        within = self.cosines[np.ix_(one, one)].sum() * self.cosines[np.ix_(other, other)].sum()
        return float(across / np.sqrt(within)) if within > 0 else 0.0


class Worded:
    """The scanned chunks as sets of words: Jaccard between two groups, and the share of one
    group's words found in the other. Catches copies and quotes, not paraphrases."""

    kind = "words"
    tau_contained = TAU_CONTAINED
    tau_alike = TAU_WORDS

    def __init__(self, texts: Sequence[str]) -> None:
        self.words = [set(WORD.findall(text.lower())) for text in texts]
        self._unions: dict[tuple[int, ...], set[str]] = {}

    def _union(self, rows: Sequence[int]) -> set[str]:
        """The words of a group of chunks, built once per group: the fold compares each result
        with every kept one, so the same groups come back again and again."""
        key = tuple(rows)
        if key not in self._unions:
            self._unions[key] = set().union(*(self.words[row] for row in rows))
        return self._unions[key]

    def contained(self, inner: Sequence[int], outer: Sequence[int]) -> float:
        words = self._union(inner)
        return len(words & self._union(outer)) / len(words) if words else 0.0

    def alike(self, one: Sequence[int], other: Sequence[int]) -> float:
        a, b = self._union(one), self._union(other)
        return len(a & b) / len(a | b) if a | b else 0.0


Space = Embedded | Worded


def space(
    texts: Sequence[str], vectors: Sequence[Sequence[float] | None], model: EmbeddingModel | None
) -> Space:
    """The space the scan is compared in: the embedding vectors when every row carries one under a
    model with thresholds, else the words. One row without a vector would leave its chunk with
    nothing to compare, so a mixed scan falls back to words as a whole."""
    duplicate = model.duplicate if model else None
    if duplicate is None or not vectors or any(vector is None for vector in vectors):
        return Worded(texts)
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return Embedded(matrix / np.where(norms == 0, 1.0, norms), duplicate, texts)


# --- the fold ---------------------------------------------------------------------


class _Item(msgspec.Struct):
    """What the fold compares of a hit or a hit range: where it sits and which chunks it holds."""

    collection: str
    document: str
    seq_start: int
    seq_end: int
    char_start: int
    char_end: int
    rows: list[int]  # its chunks, as rows of the space


class _Group(msgspec.Struct):
    """One kept slot while the fold walks: its result, and what was folded into it."""

    leader: int
    score: float  # the slot's score: the first leader's, kept through a superset swap
    folded: list[tuple[int, float]]  # (item, similarity), in walk order


class Fold(msgspec.Struct):
    """A fold's decision on one item against one kept item: fold it in, or swap it in."""

    similarity: float
    swap: bool = False  # the new item contains the kept one: it takes the slot instead


def _compare(item: _Item, kept: _Item, where: Space) -> Fold | None:
    """Whether `item` repeats `kept`, and how.

    Two results of one document in one collection that touch are never folded: they are one
    passage cut in two, not two sources. Two spans of one document that overlap in characters (one
    document chunked two ways in two collections) are compared by that overlap, which is exact.
    """
    if item.document == kept.document and item.collection == kept.collection:
        if item.seq_start <= kept.seq_end + 1 and kept.seq_start <= item.seq_end + 1:
            return None
    elif item.document == kept.document:
        overlap = min(item.char_end, kept.char_end) - max(item.char_start, kept.char_start)
        if overlap > 0:
            smaller = min(item.char_end - item.char_start, kept.char_end - kept.char_start)
            share = overlap / smaller if smaller > 0 else 0.0
            return Fold(similarity=share) if share > TAU_CHARS else None
    found = _decide(item, kept, where)
    if found is None and isinstance(where, Embedded):
        found = _decide(item, kept, where.words)
    return found


def _decide(item: _Item, kept: _Item, where: Space) -> Fold | None:
    """Whether the content of `item` repeats `kept`, in one space: contained either way, or alike
    as a whole."""
    into = where.contained(item.rows, kept.rows)
    over = where.contained(kept.rows, item.rows)
    alike = where.alike(item.rows, kept.rows)
    # the superset swap: without it the reader would get the shared text twice, once on its own
    # and once inside the fuller result further down
    if over > where.tau_contained and into <= where.tau_contained:
        return Fold(similarity=over, swap=True)
    if into > where.tau_contained or alike > where.tau_alike:
        return Fold(similarity=max(into, alike))
    return None


def _groups(items: list[_Item], scores: list[float], where: Space, limit: int) -> list[_Group]:
    """Leader clustering over `items`, best first.

    Past `limit` kept items the walk goes on, folding only: a repeat found further down still
    counts as corroboration, and it costs no slot.
    """
    groups: list[_Group] = []
    for index, item in enumerate(items):
        found = [(group, _compare(item, items[group.leader], where)) for group in groups]
        matches = [(group, fold) for group, fold in found if fold is not None]
        if not matches:
            if len(groups) < limit:
                groups.append(_Group(leader=index, score=scores[index], folded=[]))
            continue
        repeats = [(group, fold) for group, fold in matches if not fold.swap]
        if repeats:
            # it says what a kept result already says: it is on screen, under the closest one
            group, fold = max(repeats, key=lambda match: match[1].similarity)
            group.folded.append((index, fold.similarity))
            continue
        # it contains kept results: it takes the best slot among them, and every one it contains
        # joins that slot, or the reader would get the shared text twice
        (group, fold), *others = matches  # `groups` is in slot order, best first
        for other, other_fold in others:
            group.folded += [(other.leader, other_fold.similarity), *other.folded]
            groups.remove(other)
        group.folded.append((group.leader, fold.similarity))
        group.leader = index
    return groups


def _folded[R](group: _Group, reference: Callable[[int, float], R]) -> tuple[list[R], int]:
    """A group's pointers, best first, capped at `MAX_ALSO`, and how many there were. Items are
    walked best first, so the lower index is the better one."""
    ordered = sorted(group.folded)
    return [reference(index, similarity) for index, similarity in ordered[:MAX_ALSO]], len(ordered)


def hits(found: list[Hit], where: Space, limit: int) -> list[Hit]:
    """The `limit` best hits with their near-duplicates folded into them, best first."""
    items = [
        _Item(
            collection=hit.collection,
            document=hit.document,
            seq_start=hit.seq,
            seq_end=hit.seq,
            char_start=hit.char_start,
            char_end=hit.char_end,
            rows=[row],
        )
        for row, hit in enumerate(found)
    ]
    groups = _groups(items, [hit.score for hit in found], where, limit)

    def reference(index: int, similarity: float) -> HitReference:
        hit = found[index]
        return HitReference(
            collection=hit.collection,
            document=hit.document,
            seq=hit.seq,
            header=hit.header,
            location=hit.location,
            line_start=hit.line_start,
            line_end=hit.line_end,
            score=hit.score,
            similarity=similarity,
        )

    kept: list[Hit] = []
    for group in groups:
        also_in, also_count = _folded(group, reference)
        kept.append(
            msgspec.structs.replace(
                found[group.leader], score=group.score, also_in=also_in, also_count=also_count
            )
        )
    return kept


def ranges(
    hit_ranges: list[HitRange], scanned: list[Hit], where: Space, limit: int
) -> list[HitRange]:
    """The `limit` best hit ranges with their near-duplicates folded into them, best first.

    `scanned` is the list the space was built over, so each range finds its chunks' rows in it.
    """
    row_of = {(hit.collection, hit.document, hit.seq): row for row, hit in enumerate(scanned)}
    items = [
        _Item(
            collection=hit_range.hits[0].collection,
            document=hit_range.hits[0].document,
            seq_start=hit_range.seq_start,
            seq_end=hit_range.seq_end,
            char_start=hit_range.char_start,
            char_end=hit_range.char_end,
            rows=[row_of[(hit.collection, hit.document, hit.seq)] for hit in hit_range.hits],
        )
        for hit_range in hit_ranges
    ]
    groups = _groups(items, [hit_range.score for hit_range in hit_ranges], where, limit)

    def reference(index: int, similarity: float) -> PassageReference:
        hit_range = hit_ranges[index]
        best = max(hit_range.hits, key=lambda hit: (hit.score, -hit.seq))
        return PassageReference(
            collection=best.collection,
            document=best.document,
            seq_start=hit_range.seq_start,
            seq_end=hit_range.seq_end,
            header=best.header,
            # not widened: a pointer to where the match is, not a passage to quote
            location=location(
                best.document,
                best.page_start,
                best.page_end,
                hit_range.line_start,
                hit_range.line_end,
            ),
            line_start=hit_range.line_start,
            line_end=hit_range.line_end,
            score=hit_range.score,
            similarity=similarity,
        )

    kept: list[HitRange] = []
    for group in groups:
        also_in, also_count = _folded(group, reference)
        kept.append(
            msgspec.structs.replace(
                hit_ranges[group.leader], score=group.score, also_in=also_in, also_count=also_count
            )
        )
    return kept
