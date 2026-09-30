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

Two results relate one of three ways (`Relation`), tried in this order:
- `duplicate`: the same text, character for character once whitespace is collapsed, in any
  document. Exact, so it needs no space and no threshold.
- `contained`: one sits inside the other. Within one document the character spans say so
  exactly; otherwise chunk by chunk, as containment: how much of one result is found in the
  other. A one-chunk passage copied into a three-chunk one is contained in it, though their mean
  vectors differ.
- `equivalent`: the same meaning in other words. Containment both ways, or the two alike as a
  whole: the cosine of their mean vectors, or the Jaccard of their words.

The comparisons are read off the `Space`s built once for the whole scan (`spaces`): the chunks'
own embedding vectors when every row has one (`Embedded`), then their words (`Worded`), which catch
near copies but not rewordings. The first space that finds a repeat decides. Words decide as they
rank: a `vector` search folds by its vectors alone, a hybrid or full-text one by its words too.
Every place is measured in both spaces all the same (`Overlaps`).

The cosine is raw, not centred on a corpus mean: for a yes-or-no threshold, centring bge or e5
vectors amounts to a shift of the threshold per model, so the threshold is set per model instead
(`EmbeddingModel.duplicate`; `catalogue/seed.sql` names the sources for its values).

No IO: `retrieval` hands in the texts and vectors of the rows it read.
"""

import re
from collections.abc import Callable, Sequence

import msgspec
import numpy as np

from haskie.catalogue.catalogue import DuplicateCosine, EmbeddingModel
from haskie.collection.index import (
    Hit,
    HitReference,
    Overlap,
    Overlaps,
    Relation,
)
from haskie.search.passage import HitRange, PassageReference, harmonic
from haskie.settings import SearchMode

# A chunk's vector: a row of the float32 array a read holds (`index._rows`), or a list a test builds
type Vector = Sequence[float] | np.ndarray

# word Jaccard: Set-Encoder (ECIR 2025, https://arxiv.org/abs/2404.06912) defines near-duplicate
# clusters as Jaccard > 0.5
TAU_WORDS = 0.5
# judgement, no source. Four in five of one result's words found in the other reads as
# a copy with a sentence added; the superset swap then keeps the fuller one.
TAU_CONTAINED = 0.8

WORD = re.compile(r"\w+")
# Broder, "On the resemblance and containment of documents" (1997), defines containment over
# shingles, runs of w words. w = 3 and the minimum are judgement: long enough that a match is
# the same wording, short enough that one sentence has several.
SHINGLE = 3
MIN_SHINGLES = 5  # 7 words: under this neither measure speaks - a label, not a point
# the same bar for an exact duplicate: two equal headings in two books are one label, not a point
MIN_WORDS = SHINGLE + MIN_SHINGLES - 1


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


class Scan(msgspec.Struct, frozen=True):
    """The spaces one scan is compared in: those that decide a fold, first to decide first, and
    every one a place is measured in (`to_parent`, `to_root`)."""

    deciding: list[Space]
    measured: list[Space]


def spaces(
    texts: Sequence[str],
    vectors: Sequence[Vector | None],
    model: EmbeddingModel | None,
    mode: SearchMode = SearchMode.HYBRID,
) -> Scan:
    """The spaces the scan is compared in: the embedding vectors when every row carries one under
    a model with thresholds, then the words. One row without a vector would leave its chunk with
    nothing to compare, so a mixed scan is compared by words alone. A `vector` search, which
    ranks by vectors alone, folds by them alone too."""
    words = Worded(texts)
    duplicate = model.duplicate if model else None
    if duplicate is None or not vectors or any(vector is None for vector in vectors):
        return Scan(deciding=[words], measured=[words])
    measured: list[Space] = [Embedded(unit_rows(vectors), duplicate), words]
    return Scan(deciding=measured[:1] if mode == SearchMode.VECTOR else measured, measured=measured)


def unit_rows(vectors: Sequence[Vector | None] | np.ndarray) -> np.ndarray:
    """The vectors as rows of unit length, so a matrix product of two is their cosines. A zero
    vector stays zero rather than dividing by it. The caller has checked that none is missing."""
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


# --- the fold ---------------------------------------------------------------------


class _Item(msgspec.Struct):
    """What the fold compares of a hit or a hit range: where it sits and which chunks it holds."""

    collection: str
    document_id: str
    seq_start: int
    seq_end: int
    char_start: int
    char_end: int
    rows: list[int]  # its chunks, as rows of the spaces
    text: str | None  # what a duplicate is matched by (`_exact`)
    # too short to stand alone (`HitRange.alone`): no passage, so nothing may sit under it
    alone: bool = False


class _Fold(msgspec.Struct):
    """A fold's decision on one item against one kept item: fold it in, or swap it in."""

    relation: Relation
    similarity: float  # how strongly `relation` holds: what picks the closest of several kept
    swap: bool = False  # the new item contains the kept one: it takes the slot instead


class _Node(msgspec.Struct):
    """One folded item, under the item it was folded into, with what was folded into it."""

    item: int
    relation: Relation  # how it overlaps its parent
    similarity: float  # how strongly `relation` holds
    children: list["_Node"]


class _Group(msgspec.Struct):
    """One kept slot while the fold walks: its result, and the tree folded into it."""

    leader: int
    score: float  # the slot's score: the first leader's, kept through a superset swap
    children: list[_Node]  # in walk order


def _exact(texts: Sequence[str]) -> str | None:
    """The text a duplicate is matched by: its chunks' texts, whitespace collapsed. None under
    `MIN_WORDS` words, where two equal texts are a label, not a point."""
    words = " ".join(texts).split()
    return " ".join(words) if len(WORD.findall(" ".join(words))) >= MIN_WORDS else None


def _compare(item: _Item, kept: _Item, where: list[Space]) -> _Fold | None:
    """Whether `item` repeats `kept`, and how.

    Two results of one document in one collection that touch are never folded: they are one
    passage cut in two, not two sources. Two spans of one document where one holds the other
    (one document chunked two ways in two collections) are contained, exactly; a partial overlap
    is compared like any other pair.
    """
    one_document = item.document_id == kept.document_id
    if one_document and item.collection == kept.collection:
        if item.seq_start <= kept.seq_end + 1 and kept.seq_start <= item.seq_end + 1:
            return None
    if item.text is not None and item.text == kept.text:
        return _Fold(Relation.DUPLICATE, 1.0)
    # one collection cuts a document into spans that never overlap: only two can nest
    if one_document and item.collection != kept.collection:
        if kept.char_start <= item.char_start and item.char_end <= kept.char_end:
            return _Fold(Relation.CONTAINED, 1.0)
        if item.char_start <= kept.char_start and kept.char_end <= item.char_end:
            return _Fold(Relation.CONTAINED, 1.0, swap=True)  # the kept one moves under it
    return next((fold for space in where if (fold := _decide(item, kept, space))), None)


def _decide(item: _Item, kept: _Item, space: Space) -> _Fold | None:
    """Whether the content of `item` repeats `kept`, in one space, and how.

    Containment one way only is `CONTAINED`: the smaller result sits inside the fuller one.
    Containment both ways, or a match as a whole, is `EQUIVALENT`.
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
        return _Fold(Relation.EQUIVALENT, max(min(into, over), alike))
    return None


def _groups(items: list[_Item], scores: list[float], scan: Scan, limit: int | None) -> list[_Group]:
    """Leader clustering over `items`, best first.

    An item is compared with the kept results only, and folds under the one it repeats. A
    superset swap puts the old leader under the new one with everything folded into it, so each
    item stays under the item it was measured against: leader clustering is not transitive, and
    a place the new leader would not place is still a repeat of the one above it.

    Past `limit` kept items the walk goes on, folding only: a repeat found further down still
    counts as corroboration, and it costs no slot. No `limit` keeps every item that repeats none.

    Of the two items of a fold, the one that would lead must be a passage. An item too short to
    stand alone (`_Item.alone`) is dropped with its section (`section.group`), and would take
    whatever sat under it along. So nothing folds under one, and one never swaps in.
    """
    groups: list[_Group] = []
    for index, item in enumerate(items):
        matches = [
            (g, fold)
            for g in groups
            if (fold := _compare(item, items[g.leader], scan.deciding))
            and not (item.alone if fold.swap else items[g.leader].alone)
        ]
        if not matches:
            if limit is None or len(groups) < limit:
                groups.append(_Group(leader=index, score=scores[index], children=[]))
            continue
        repeats = [(group, fold) for group, fold in matches if not fold.swap]
        if repeats:
            # it says what a kept result already says: it is on screen, under the closest one
            group, fold = max(repeats, key=lambda match: match[1].similarity)
            group.children.append(_Node(index, fold.relation, fold.similarity, []))
            continue
        # it contains kept results: it takes the best slot among them, and every one it contains
        # moves under it, or the reader would get the shared text twice
        (group, fold), *others = matches  # `groups` is in slot order, best first
        moved = [_Node(group.leader, fold.relation, fold.similarity, group.children)]
        for other, other_fold in others:
            moved.append(
                _Node(other.leader, other_fold.relation, other_fold.similarity, other.children)
            )
            groups.remove(other)
        group.leader = index
        group.children = moved
    return groups


def _overlap(space: Space, item: _Item, other: _Item) -> Overlap:
    contained = space.contained(item.rows, other.rows)
    contains = space.contained(other.rows, item.rows)
    alike = space.alike(item.rows, other.rows)
    return Overlap(contained, contains, alike, score=harmonic(contained, contains))


def _overlaps(item: _Item, other: _Item, where: list[Space]) -> Overlaps:
    """How `item` overlaps `other`, every way the fold compares: in each space, and by characters
    where both are cut from one document."""
    by_kind = {space.kind: _overlap(space, item, other) for space in where}
    chars = None
    if item.document_id == other.document_id:
        shared = min(item.char_end, other.char_end) - max(item.char_start, other.char_start)
        smaller = min(item.char_end - item.char_start, other.char_end - other.char_start)
        chars = max(0, shared) / smaller if smaller > 0 else 0.0
    return Overlaps(words=by_kind["words"], embedding=by_kind.get("embedding"), chars=chars)


def _collapse[R: (Hit, HitRange), F: (HitReference, PassageReference)](
    results: list[R],
    items: list[_Item],
    scan: Scan,
    limit: int | None,
    reference: Callable[[R, _Node, Overlaps, Overlaps, list[F]], F],
) -> list[R]:
    """The fold, shared by hits and ranges: the `limit` best results, each with the tree of
    places folded into it, every level best first (items are walked best first, so the lower
    index is the better one). The tree is bounded by the scan, not capped: a count of what is
    left out would tell a reader less than the places themselves."""
    kept: list[R] = []
    for group in _groups(items, [result.score for result in results], scan, limit):
        root = items[group.leader]

        def place(node: _Node, parent: _Item, root: _Item = root) -> F:
            item = items[node.item]
            to_parent = _overlaps(item, parent, scan.measured)
            return reference(
                results[node.item],
                node,
                to_parent,
                to_parent if parent is root else _overlaps(item, root, scan.measured),
                [place(child, item) for child in sorted(node.children, key=_item_of)],
            )

        also_in = [place(child, root) for child in sorted(group.children, key=_item_of)]
        kept.append(
            msgspec.structs.replace(results[group.leader], score=group.score, also_in=also_in)
        )
    return kept


def _item_of(node: _Node) -> int:
    return node.item


def places(references: Sequence[HitReference | PassageReference]) -> int:
    """How many places a tree of references holds, at every level."""
    return sum(1 + places(reference.also_in) for reference in references)


def hits(found: list[Hit], scan: Scan, limit: int) -> list[Hit]:
    """The `limit` best hits with their near-duplicates folded into them, best first."""
    items = [
        _Item(
            collection=hit.collection,
            document_id=hit.document_id,
            seq_start=hit.seq,
            seq_end=hit.seq,
            char_start=hit.char_start,
            char_end=hit.char_end,
            rows=[row],
            text=_exact([hit.text]),
        )
        for row, hit in enumerate(found)
    ]

    def reference(
        hit: Hit,
        node: _Node,
        to_parent: Overlaps,
        to_root: Overlaps,
        also_in: list[HitReference],
    ) -> HitReference:
        return HitReference(
            collection=hit.collection,
            document_id=hit.document_id,
            document=hit.document,
            seq=hit.seq,
            header=hit.header,
            location=hit.location,
            line_start=hit.line_start,
            line_end=hit.line_end,
            score=hit.score,
            relation=node.relation,
            similarity=node.similarity,
            to_parent=to_parent,
            to_root=to_root,
            also_in=also_in,
        )

    return _collapse(found, items, scan, limit, reference)


def ranges(
    hit_ranges: list[HitRange], scanned: list[Hit], scan: Scan, limit: int | None
) -> list[HitRange]:
    """The `limit` best hit ranges with their near-duplicates folded into them, best first.

    `scanned` is the list the spaces were built over, so each range finds its chunks' rows in it.
    """
    row_of = {(hit.collection, hit.document_id, hit.seq): row for row, hit in enumerate(scanned)}
    items = [
        _Item(
            collection=hit_range.hits[0].collection,
            document_id=hit_range.hits[0].document_id,
            seq_start=hit_range.seq_start,
            seq_end=hit_range.seq_end,
            char_start=hit_range.char_start,
            char_end=hit_range.char_end,
            rows=[row_of[(hit.collection, hit.document_id, hit.seq)] for hit in hit_range.hits],
            text=_exact([hit.text for hit in hit_range.hits]),
            alone=hit_range.alone,
        )
        for hit_range in hit_ranges
    ]

    def reference(
        hit_range: HitRange,
        node: _Node,
        to_parent: Overlaps,
        to_root: Overlaps,
        also_in: list[PassageReference],
    ) -> PassageReference:
        best = hit_range.best
        return PassageReference(
            collection=best.collection,
            document_id=best.document_id,
            document=best.document,
            seq_start=hit_range.seq_start,
            seq_end=hit_range.seq_end,
            header=best.header,
            # not read: a pointer to where the match is, not a passage to quote
            location=hit_range.location,
            line_start=hit_range.line_start,
            line_end=hit_range.line_end,
            score=hit_range.score,
            relation=node.relation,
            similarity=node.similarity,
            to_parent=to_parent,
            to_root=to_root,
            also_in=also_in,
        )

    return _collapse(hit_ranges, items, scan, limit, reference)
