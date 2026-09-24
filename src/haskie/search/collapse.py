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

The comparisons are read off the `Space`s built once for the whole scan (`spaces`): the chunks'
own embedding vectors when every row has one (`Embedded`), then their words (`Worded`), which catch
copies but not rewordings. The first space that finds a repeat decides, so identical text folds
under any heading path its vectors were embedded after. The cosine is raw, not centred on a corpus
mean: for a yes-or-no threshold, centring bge or e5 vectors amounts to a shift of the threshold per
model, so the threshold is set per model instead (`EmbeddingModel.duplicate`, where the sources for
its values are).

No IO: `retrieval` hands in the texts and vectors of the rows it read.
"""

import re
from collections.abc import Callable, Sequence

import msgspec
import numpy as np

from haskie.collection.index import Hit, HitReference, Relation, location
from haskie.search.passage import HitRange, PassageReference, pages
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

WORD = re.compile(r"\w+")
# Broder, "On the resemblance and containment of documents" (1997), defines containment over
# shingles, runs of w words. w = 3 and the minimum are judgement: long enough that a match is
# the same wording, short enough that one sentence has several.
SHINGLE = 3
MIN_SHINGLES = 5  # 7 words: under this neither measure speaks - a label, not a point


# --- the spaces -------------------------------------------------------------------


class Embedded:
    """The scanned chunks as unit vectors: cosine between chunks, and between the mean vectors of
    two groups of them, read off one matrix."""

    kind = "embedding"

    def __init__(self, vectors: np.ndarray, duplicate: DuplicateCosine) -> None:
        self.cosines = vectors @ vectors.T
        self.tau_contained = duplicate.chunk
        self.tau_alike = duplicate.passage
        self._within: dict[tuple[int, ...], float] = {}

    def contained(self, inner: Sequence[int], outer: Sequence[int]) -> float:
        """How much of `inner` is in `outer`: its chunks' best cosines to any chunk of `outer`,
        averaged."""
        if len(inner) == 1 and len(outer) == 1:  # two hits: one cosine, no slicing
            return float(self.cosines[inner[0], outer[0]])
        return float(self.cosines[np.ix_(inner, outer)].max(axis=1).mean())

    def alike(self, one: Sequence[int], other: Sequence[int]) -> float:
        """The cosine of the two groups' mean vectors. For unit vectors it is a ratio of sums over
        the cosine matrix, so the vectors are not needed again."""
        if len(one) == 1 and len(other) == 1:
            return float(self.cosines[one[0], other[0]])
        across = self.cosines[np.ix_(one, other)].sum()
        within = self._within_sum(one) * self._within_sum(other)
        return float(across / np.sqrt(within)) if within > 0 else 0.0

    def _within_sum(self, rows: Sequence[int]) -> float:
        """A group's cosines with itself, summed, once per group: a kept result is compared with
        every result after it, as `Worded._union` caches its sets."""
        key = tuple(rows)
        if key not in self._within:
            self._within[key] = float(self.cosines[np.ix_(rows, rows)].sum())
        return self._within[key]


class Worded:
    """The scanned chunks as words: Jaccard of two groups' word sets, and the share of one
    group's word 3-grams (shingles) found in the other's. Catches copies and quotes, not
    paraphrases.

    Containment is over shingles, not single words, as Broder defines it for documents: a short
    chunk's few words are all found somewhere in a long passage, so on single words a heading such
    as "APPENDIX D" was 100% contained in a chapter that merely mentioned an appendix. Shingles
    only match where the same words stand in the same order.
    """

    kind = "words"
    tau_contained = TAU_CONTAINED
    tau_alike = TAU_WORDS

    def __init__(self, texts: Sequence[str]) -> None:
        tokens = [WORD.findall(text.lower()) for text in texts]
        self.words = [set(words) for words in tokens]
        self.shingles = [
            {tuple(words[at : at + SHINGLE]) for at in range(len(words) - SHINGLE + 1)}
            for words in tokens
        ]
        self._words: dict[tuple[int, ...], set] = {}
        self._shingles: dict[tuple[int, ...], set] = {}

    @staticmethod
    def _union(per_row: list[set], cache: dict[tuple[int, ...], set], rows: Sequence[int]) -> set:
        """The sets of a group of chunks joined, once per group: the fold compares each result
        with every kept one, so the same groups come back again and again."""
        if len(rows) == 1:
            return per_row[rows[0]]
        key = tuple(rows)
        if key not in cache:
            cache[key] = set().union(*(per_row[row] for row in rows))
        return cache[key]

    def contained(self, inner: Sequence[int], outer: Sequence[int]) -> float:
        shingles = self._union(self.shingles, self._shingles, inner)
        if len(shingles) < MIN_SHINGLES:
            return 0.0
        return len(shingles & self._union(self.shingles, self._shingles, outer)) / len(shingles)

    def alike(self, one: Sequence[int], other: Sequence[int]) -> float:
        groups = (one, other)
        if min(len(self._union(self.shingles, self._shingles, g)) for g in groups) < MIN_SHINGLES:
            return 0.0
        a = self._union(self.words, self._words, one)
        b = self._union(self.words, self._words, other)
        return len(a & b) / len(a | b)


Space = Embedded | Worded


def spaces(
    texts: Sequence[str], vectors: Sequence[Sequence[float] | None], model: EmbeddingModel | None
) -> list[Space]:
    """The spaces the scan is compared in, first to decide first: the embedding vectors when every
    row carries one under a model with thresholds, then the words. One row without a vector would
    leave its chunk with nothing to compare, so a mixed scan is compared by words alone."""
    words = Worded(texts)
    duplicate = model.duplicate if model else None
    if duplicate is None or not vectors or any(vector is None for vector in vectors):
        return [words]
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return [Embedded(matrix / np.where(norms == 0, 1.0, norms), duplicate), words]


# --- the fold ---------------------------------------------------------------------


class _Item(msgspec.Struct):
    """What the fold compares of a hit or a hit range: where it sits and which chunks it holds."""

    collection: str
    document: str
    seq_start: int
    seq_end: int
    char_start: int
    char_end: int
    rows: list[int]  # its chunks, as rows of the spaces


class _Fold(msgspec.Struct):
    """A fold's decision on one item against one kept item: fold it in, or swap it in."""

    relation: Relation
    similarity: float
    swap: bool = False  # the new item contains the kept one: it takes the slot instead
    via: int | None = None  # the item it was measured against, when that is not its leader


class _Group(msgspec.Struct):
    """One kept slot while the fold walks: its result, and what was folded into it."""

    leader: int
    score: float  # the slot's score: the first leader's, kept through a superset swap
    folded: list[tuple[int, _Fold]]  # (item, how it overlaps the leader), in walk order


def _compare(item: _Item, kept: _Item, where: list[Space]) -> _Fold | None:
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
            return _Fold(Relation.SAME_SPAN, share) if share > TAU_CHARS else None
    return next((fold for space in where if (fold := _decide(item, kept, space))), None)


def _decide(item: _Item, kept: _Item, space: Space) -> _Fold | None:
    """Whether the content of `item` repeats `kept`, in one space, and how.

    Containment one way only is `CONTAINED`: the smaller result sits inside the fuller one.
    Containment both ways, or a match as a whole, is `DUPLICATE`.
    """
    into = space.contained(item.rows, kept.rows)
    over = space.contained(kept.rows, item.rows)
    # the superset swap: without it the reader would get the shared text twice, once on its own
    # and once inside the fuller result further down
    if over > space.tau_contained and into <= space.tau_contained:
        # the kept one moves under `item`, and sits inside it
        return _Fold(Relation.CONTAINED, over, swap=True)
    if into > space.tau_contained and over <= space.tau_contained:
        return _Fold(Relation.CONTAINED, into)
    alike = space.alike(item.rows, kept.rows)  # only here: containment decided the rest
    if into > space.tau_contained or alike > space.tau_alike:
        # past the two lines above, containment one way means containment both ways
        return _Fold(Relation.DUPLICATE, max(min(into, over), alike))
    return None


def _groups(
    items: list[_Item], scores: list[float], where: list[Space], limit: int
) -> list[_Group]:
    """Leader clustering over `items`, best first.

    Past `limit` kept items the walk goes on, folding only: a repeat found further down still
    counts as corroboration, and it costs no slot.
    """
    groups: list[_Group] = []
    for index, item in enumerate(items):
        matches = [(g, fold) for g in groups if (fold := _compare(item, items[g.leader], where))]
        if not matches:
            if len(groups) < limit:
                groups.append(_Group(leader=index, score=scores[index], folded=[]))
            continue
        repeats = [(group, fold) for group, fold in matches if not fold.swap]
        if repeats:
            # it says what a kept result already says: it is on screen, under the closest one
            group, fold = max(repeats, key=lambda match: match[1].similarity)
            group.folded.append((index, fold))
            continue
        # it contains kept results: it takes the best slot among them, and every one it contains
        # joins that slot, or the reader would get the shared text twice
        (group, fold), *others = matches  # `groups` is in slot order, best first
        moved = _moved(group, fold, items, index, where)
        for other, other_fold in others:
            moved += _moved(other, other_fold, items, index, where)
            groups.remove(other)
        group.leader = index
        group.folded = moved
    return groups


def _moved(
    group: _Group, fold: _Fold, items: list[_Item], leader: int, where: list[Space]
) -> list[tuple[int, _Fold]]:
    """What a superset swap moves under the new `leader`: the old leader, which `fold` measured
    against it, and every member, compared with it again.

    A member the comparison no longer places keeps what was measured, against the old leader, and
    says so through `via`: leader clustering is not transitive, so nothing measured it against the
    new one."""
    moved = [(group.leader, msgspec.structs.replace(fold, swap=False))]
    for member, was in group.folded:
        again = _compare(items[member], items[leader], where)
        if again is not None and not again.swap:
            moved.append((member, again))
        else:
            moved.append(
                (
                    member,
                    was if was.via is not None else msgspec.structs.replace(was, via=group.leader),
                )
            )
    return moved


def _collapse[R: (Hit, HitRange)](
    results: list[R],
    items: list[_Item],
    where: list[Space],
    limit: int,
    reference: Callable[[R, _Fold], HitReference | PassageReference],
) -> list[R]:
    """The fold, shared by hits and ranges: the `limit` best results, each with every pointer
    folded into it, best first (items are walked best first, so the lower index is the better
    one). The list is bounded by the scan, not capped: a count of what is left out would tell a
    reader less than the places themselves."""
    kept: list[R] = []
    for group in _groups(items, [result.score for result in results], where, limit):
        folded = sorted(group.folded, key=lambda member: member[0])
        refs = {index: reference(results[index], fold) for index, fold in folded}
        # a `via` points at a place in the same list, since a swap moves the old leader with it
        also_in = [
            refs[index]
            if fold.via is None
            else msgspec.structs.replace(refs[index], via=refs[fold.via].location)
            for index, fold in folded
        ]
        kept.append(
            msgspec.structs.replace(results[group.leader], score=group.score, also_in=also_in)
        )
    return kept


def hits(found: list[Hit], where: list[Space], limit: int) -> list[Hit]:
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

    def reference(hit: Hit, fold: _Fold) -> HitReference:
        return HitReference(
            collection=hit.collection,
            document=hit.document,
            seq=hit.seq,
            header=hit.header,
            location=hit.location,
            line_start=hit.line_start,
            line_end=hit.line_end,
            score=hit.score,
            relation=fold.relation,
            similarity=fold.similarity,
        )

    return _collapse(found, items, where, limit, reference)


def ranges(
    hit_ranges: list[HitRange], scanned: list[Hit], where: list[Space], limit: int
) -> list[HitRange]:
    """The `limit` best hit ranges with their near-duplicates folded into them, best first.

    `scanned` is the list the spaces were built over, so each range finds its chunks' rows in it.
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

    def reference(hit_range: HitRange, fold: _Fold) -> PassageReference:
        best = hit_range.best
        return PassageReference(
            collection=best.collection,
            document=best.document,
            seq_start=hit_range.seq_start,
            seq_end=hit_range.seq_end,
            header=best.header,
            # not widened: a pointer to where the match is, not a passage to quote
            location=location(
                best.document,
                *pages(hit_range.hits),
                hit_range.line_start,
                hit_range.line_end,
            ),
            line_start=hit_range.line_start,
            line_end=hit_range.line_end,
            score=hit_range.score,
            relation=fold.relation,
            similarity=fold.similarity,
        )

    return _collapse(hit_ranges, items, where, limit, reference)
