"""Passages too short to stand alone: grown by the neighbouring chunks that match the question,
else dropped.

A chunk is cut where the author cut, so a short one is usually a section's lead-in ("Three
rules:", with the list in the next chunk), a section's last line, or a separator (`---`). Alone
it tells a reader little and costs a slot. So a thin range takes the chunk before or after it,
one at a time, as long as that chunk is in the same section and matches the question. A range
with no such neighbour is marked `alone`, unless it is the best range of the search: a short
exact answer ("Tim Cook, CEO") is still an answer. What `alone` costs is decided where the answer
is cut: a passage that is alone goes, and so does an excerpt whose passages all are, while a
short passage inside a section with another passage stays in that section's excerpt. A thin
range with a section end on both sides is a whole section, a short note say, not a piece of one:
it is kept as the author wrote it.

"Matches" is measured, not assumed: `retrieval.fill_thin` scores the neighbours the way it can
score the question against the scanned hits (the reranker, the query vector, or the question's
words), and a neighbour matches when it scores at least the median scanned hit. The floor is
this search's own, so no calibrated threshold is needed. No IO here.
"""

import msgspec

from haskie.collection.index import ChunkKey, Hit, chunk_key
from haskie.indexing.segment import CutReason
from haskie.search.collapse import MIN_WORDS, WORD
from haskie.search.passage import HitRange, continues, ends_section, ranges

# The words of a question that say nothing about its topic. A short list on purpose: the question
# words and the glue of English questions, so the words left are the ones a passage should share.
STOPWORDS = frozenset(
    "the and for are was were has have had how what when where which who why whom whose does did "
    "should would could can may might must shall will not but with from into onto that this these "
    "those than then there their them they its it's about over under between each any all some "
    "one two more most other such only own same very just also".split()
)

# Where a whole section begins and ends: at a heading, or at the edge of the text chunked
_BOUNDS = frozenset({CutReason.HEADING, CutReason.EDGE})


def is_thin(hits: list[Hit], min_chars: int) -> bool:
    """Consecutive hits shorter than `min_chars`, or under `MIN_WORDS` words: too little to quote
    on its own. `min_chars` 0 turns the check off."""
    if min_chars <= 0:
        return False
    words = sum(len(WORD.findall(hit.text)) for hit in hits)
    return hits[-1].char_end - hits[0].char_start < min_chars or words < MIN_WORDS


def _fragment(hit_range: HitRange, min_chars: int) -> bool:
    """Thin, with words, and part of a longer section: what grows or goes. A range with a section
    bound on both sides is a whole section, a short note say, with nothing of it to grow into."""
    whole = hit_range.hits[0].start_reason in _BOUNDS and hit_range.hits[-1].end_reason in _BOUNDS
    return is_thin(hit_range.hits, min_chars) and not _wordless(hit_range) and not whole


def _wordless(hit_range: HitRange) -> bool:
    """No word at all, a separator or a stray symbol: nothing to quote, whatever it ranked."""
    return not any(WORD.search(hit.text) for hit in hit_range.hits)


def around(hit_ranges: list[HitRange], min_chars: int, grow: int) -> set[ChunkKey]:
    """The chunks a thin range could grow into: up to `grow` on each side that no heading closes
    (`passage.ends_section`). Whether a chunk further out is still in the section is only known
    once the nearer one is read, so all of them are asked for."""
    wanted: set[ChunkKey] = set()
    for hit_range in hit_ranges:
        if grow <= 0 or not _fragment(hit_range, min_chars):
            continue
        first, last = hit_range.hits[0], hit_range.hits[-1]
        for step in range(1, grow + 1):
            if first.start_reason != CutReason.HEADING and first.seq - step >= 1:
                wanted.add((first.collection, first.document, first.seq - step))
            if not ends_section(last):
                wanted.add((last.collection, last.document, last.seq + step))
    return wanted


class Filled(msgspec.Struct):
    """The ranges after the thin ones grew, and what that took."""

    ranges: list[HitRange]  # best first, as `passage.ranges` orders them
    added: list[Hit]  # the neighbours that joined, each once
    grown: int  # thin ranges that took at least one neighbour
    dropped: int  # ranges without a word, which went


def fill(
    hit_ranges: list[HitRange],
    neighbours: dict[ChunkKey, Hit],
    scores: dict[ChunkKey, float],
    floor: float,
    min_chars: int,
    grow: int,
) -> Filled:
    """Grow each thin range by the neighbours that match (`scores[key] >= floor`), best first,
    never past a heading (`passage.continues`), until it is no longer thin or has taken `grow`. A
    thin range that took none is marked `alone`, unless it is the first of `hit_ranges`, the best
    the search found. A thin range that is a whole section is kept as it is. A range without a
    word is dropped whatever it ranked.

    A neighbour joins with score 0, so a range's score (`passage.harmonic` over its chunks) says
    how strongly it matched, not how much it grew. The ranges are rebuilt from every kept chunk at
    the end, so two ranges a neighbour now joins become one, and a rebuilt range is `alone` only
    when every chunk of it came from a range that was.
    """
    kept: dict[ChunkKey, Hit] = {}
    added: dict[ChunkKey, Hit] = {}
    lonely: set[ChunkKey] = set()
    grown = dropped = 0
    for position, hit_range in enumerate(hit_ranges):
        if _wordless(hit_range):
            dropped += 1
            continue
        if _fragment(hit_range, min_chars):
            took = _grow(hit_range.hits, neighbours, scores, floor, min_chars, grow)
            if took:
                grown += 1
                added.update((chunk_key(hit), hit) for hit in took)
            elif position > 0:
                lonely.update(chunk_key(hit) for hit in hit_range.hits)
        kept.update((chunk_key(hit), hit) for hit in hit_range.hits)
    joined = [hit for one, hit in added.items() if one not in kept]
    kept.update((chunk_key(hit), hit) for hit in joined)
    rebuilt = [
        msgspec.structs.replace(one, alone=all(chunk_key(hit) in lonely for hit in one.hits))
        for one in ranges(list(kept.values()))
    ]
    return Filled(ranges=rebuilt, added=joined, grown=grown, dropped=dropped)


def _grow(
    hits: list[Hit],
    neighbours: dict[ChunkKey, Hit],
    scores: dict[ChunkKey, float],
    floor: float,
    min_chars: int,
    grow: int,
) -> list[Hit]:
    """The neighbours one thin run of hits takes, in the order it takes them."""
    took: list[Hit] = []
    current = list(hits)
    while len(took) < grow:
        first, last = current[0], current[-1]
        before = neighbours.get((first.collection, first.document, first.seq - 1))
        after = neighbours.get((last.collection, last.document, last.seq + 1))
        matching = [
            one
            for one in (
                before if before is not None and continues(before, first) else None,
                after if after is not None and continues(last, after) else None,
            )
            if one is not None and scores[chunk_key(one)] >= floor
        ]
        if not matching:
            break
        best = max(matching, key=lambda one: (scores[chunk_key(one)], -one.seq))
        took.append(msgspec.structs.replace(best, score=0.0))
        current = [took[-1], *current] if best.seq < first.seq else [*current, took[-1]]
        if not is_thin(current, min_chars):
            break
    return took


def terms(text: str) -> set[str]:
    """The words of `text` a passage on its topic would share: lowercase, three letters or more,
    no stopwords."""
    return {word for word in WORD.findall(text.lower()) if len(word) > 2} - STOPWORDS


def overlap(question: set[str], text: str) -> float:
    """The share of the question's words that `text` holds: the match score when there is neither
    a reranker nor a vector to ask. A chunk's heading path is left out, since every chunk of a
    section shares it."""
    if not question:
        return 0.0
    return len(question & terms(text)) / len(question)
