"""The words that say what a section is about, and what sets it apart from the sections beside it.

An outline picks each section's keywords through a `Strategy`. `ClassTfidf` is the one there is,
three steps, each the one BERTopic describes for a topic, with a section in place of a topic:

- **Terms.** A text's words, lowercase, three letters or more, no stopwords, counted by their
  stem (`probe.stem`, the Snowball stemmer LanceDB's full-text index uses), and every pair of
  such words that stand next to each other. A pair never spans a stopword or a punctuation mark,
  so "aggregate root" is a term and "root of the aggregate" is not. A contraction is one word
  ("don't"), which the stop list names. A term a text uses once is no candidate (`frequent`),
  KP-Miner's rule: a pair used once is mostly two words that happen to meet ("applications
  store"), and a word used once is often half of one a PDF split over two lines ("charac
  teristics"), which the inverse frequency would rank first for its rarity. It still counts in
  how common the term is across the classes. A text too short to have enough terms used twice
  keeps them all as candidates. A word split by a hyphen at a line end is joined first
  ("pro-/cessing"). Each term keeps the form it is written in most often, for display.
- **c-TF-IDF** (`ctfidf`). Each section is one class; a term weighs how often the section uses
  it against how often every class does, with BM25's inverse frequency and the square root of the
  frequency, BERTopic's `ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)`
  [1]. So a chapter's terms are the ones its sibling chapters use less.
- **Rerank** (`rerank`). The best terms by that weight are the candidates; each is embedded, and
  the ones closest to the section's own vector win, one at a time, each against the ones already
  taken (maximal marginal relevance), so "aggregate" and "aggregates" do not both take a slot.
  BERTopic's `KeyBERTInspired` and `MaximalMarginalRelevance` [2].

MMR here picks words, not passages: which results a search returns is decided elsewhere
(`search/overview.py`), where the research this repository follows ranks MMR the weakest method.

The strategies share the term statistics below, which the search's map of sections
(`search/overview.py`) reuses to set its picks apart. No IO here.

[1] https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html
[2] https://maartengr.github.io/BERTopic/getting_started/representation/representation.html
"""

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from operator import itemgetter
from typing import NamedTuple, Protocol

import numpy as np

from haskie.search.collapse import unit_rows
from haskie.search.probe import stem

# English function words, the Snowball stop list and the modal verbs and fillers that say nothing
# about a section's topic. Broader than `thin.STOPWORDS`, which keeps the words a question hangs on.
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are aren't as at be because been
    before being below between both but by can can't cannot could couldn't did didn't do does
    doesn't doing don't down during each either else etc even ever every few for from further get
    gets got had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
    herself him himself his how how's however i i'd i'll i'm i've if in into is isn't it it's its
    itself just let's like may me might more most much must mustn't my myself neither no nor not
    now of off often on once one only or other ought our ours ourselves out over own per rather
    same shall shan't she she'd she'll she's should shouldn't since so some such than that that's
    the their theirs them themselves then there there's these they they'd they'll they're they've
    this those though through thus to too two under until up upon us use used uses using very via
    was wasn't we we'd we'll we're we've well were weren't what what's when when's where where's
    whether which while who who's whom whose why why's will with within without won't would
    wouldn't yes yet you you'd you'll you're you've your yours yourself yourselves
    """.split()
)

MIN_LETTERS = 3  # shorter words are glue or noise: "of", "id", a stray list marker
# Where a pair of words stops being a phrase: sentence and clause punctuation, brackets, table
# and link syntax, and a blank line. Emphasis marks (`*`, `_`, backticks) are not among them, so
# "**aggregate** root" is still a pair.
_BREAK = re.compile(r"[.,;:!?()\[\]{}<>\"|=/\\–—“”]|\n\s*\n")
_LETTER = re.compile(r"[^\W\d_]")
# A word, and its contraction or possessive with it ("don't", "aggregate's"): split at the
# apostrophe, "don" and "isn" would pass as words the stop list never names
_WORD = re.compile(r"\w+(?:['\u2019]\w+)*")
_HYPHENATED = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")
MIN_USES = 2  # uses of a term in one text before it counts (KP-Miner's rule, with n = 2)

type Term = tuple[str, str]  # (key: its stems joined by a space, the form it was written in)


def terms(text: str) -> list[Term]:
    """Every term of `text` in reading order, each word once and each pair of neighbours once."""
    found: list[Term] = []
    for clause in _BREAK.split(_HYPHENATED.sub(r"\1\2", text)):
        previous: tuple[str, str] | None = None  # the last word kept, if it was the last word read
        for word in _WORD.findall(clause):
            lower = word.lower().replace("\u2019", "'")
            if len(lower) < MIN_LETTERS or lower in STOPWORDS or not _LETTER.search(lower):
                previous = None
                continue
            key = stem(lower)
            found.append((key, word))
            if previous is not None:
                found.append((f"{previous[0]} {key}", f"{previous[1]} {word}"))
            previous = (key, word)
    return found


def counted(found: list[Term], forms: Counter[Term]) -> Counter[str]:
    """How often each term key occurs in `found`. Each (key, form) is counted into `forms`, which
    the caller shares across the texts whose terms it will show (`shown`)."""
    forms.update(found)
    return Counter(map(itemgetter(0), found))


def shown(keys: Iterable[str], forms: Counter[Term]) -> dict[str, str]:
    """The form each of `keys` is written in most often, the first seen when two tie."""
    wanted = set(keys)
    best: dict[str, tuple[int, str]] = {}
    for (key, form), uses in forms.items():
        if key in wanted and uses > best.get(key, (0, ""))[0]:
            best[key] = (uses, form)
    return {key: form for key, (_, form) in best.items()}


def key_of(written: str) -> str:
    """A term's key from the form it is written in: its words' stems, as `terms` keys it."""
    return " ".join(stem(word) for word in written.lower().split())


def summed(counts: Iterable[Counter[str]]) -> Counter[str]:
    """The counts added up, in place into one: `sum(counts, Counter())` copies the running total
    at every step, quadratic in the counts of a long chapter."""
    total: Counter[str] = Counter()
    for one in counts:
        total.update(one)
    return total


def frequent(counts: Counter[str], enough: int) -> set[str]:
    """The terms `counts` uses at least `MIN_USES` times, unless that leaves fewer than `enough`
    words: then the text is short, and each word it uses is a candidate."""
    kept = {key for key, n in counts.items() if n >= MIN_USES}
    return kept if sum(" " not in key for key in kept) >= enough else set(counts)


def ctfidf(classes: Sequence[Counter[str]]) -> list[dict[str, float]]:
    """Each class's terms weighed against every class's (see the module):
    `sqrt(tf / |c|) * log(1 + (A - f + 0.5) / (f + 0.5))`, where `tf` is the term's count in the
    class, `|c|` the class's count of terms, `f` the term's count over every class, and `A` the
    average class size. One class alone is weighed against itself, so a term it uses on every line
    sinks and the rest rank by how often they occur."""
    total = summed(classes)
    sizes = [one.total() for one in classes]
    average = sum(sizes) / len(classes) if classes else 0.0
    return [
        {
            term: math.sqrt(count / size)
            * math.log(1 + (average - total[term] + 0.5) / (total[term] + 0.5))
            for term, count in one.items()
        }
        for one, size in zip(classes, sizes, strict=True)
    ]


def best(weights: dict[str, float], k: int) -> list[str]:
    """The `k` heaviest term keys, heaviest first, each a term whose stems no heavier one already
    holds or is held by: "aggregate root" over "aggregate", not both."""
    kept: list[str] = []
    stems: list[frozenset[str]] = []
    for key in sorted(weights, key=lambda key: (-weights[key], key)):
        mine = frozenset(key.split())
        if any(mine <= other or other <= mine for other in stems):
            continue
        kept.append(key)
        stems.append(mine)
        if len(kept) == k:
            break
    return kept


DIVERSITY = 0.3  # BERTopic's example value for `MaximalMarginalRelevance`; judgement, not measured


def rerank(candidates: np.ndarray, section: np.ndarray, k: int) -> list[int]:
    """The `k` candidates (rows of unit vectors) to keep, by maximal marginal relevance to the
    section's unit vector: first the closest, then each time the one whose closeness, less
    `DIVERSITY` times its closeness to the nearest already kept, is highest."""
    if not len(candidates):
        return []
    relevance = candidates @ section
    alike = candidates @ candidates.T
    kept = [int(np.argmax(relevance))]
    while len(kept) < min(k, len(candidates)):
        score = (1 - DIVERSITY) * relevance - DIVERSITY * alike[:, kept].max(axis=1)
        score[kept] = -np.inf
        kept.append(int(np.argmax(score)))
    return kept


# --- strategies ---------------------------------------------------------------------

KEYWORDS = 8  # per section: enough to tell it apart, few enough to read at a glance
CANDIDATES = 20  # terms per section the embedding model reranks (`rerank`)

type Embed = Callable[[list[str]], list[list[float]]]  # document-side embeddings, one per text


class Run(NamedTuple):
    """One section as a strategy sees it: how many headings deep it sits, and the chunks it runs
    over, both ends inclusive."""

    depth: int
    first: int
    last: int


class Strategy(Protocol):
    """How an outline picks the keywords of each of a document's sections (`outline.build`)."""

    def pick(
        self,
        texts: Sequence[str],
        runs: Sequence[Run],
        vectors: np.ndarray | None,
        embed: Embed | None,
    ) -> list[dict[str, int]]:
        """Each run's keywords as written, best first, with how often the run uses each.

        `texts` are the document's chunks in order. `vectors` holds one unit vector per run, and
        `embed` embeds any text the same way; both are None without an embedding model."""
        ...


class ClassTfidf:
    """c-TF-IDF over the sections of one depth, reranked by meaning when there is a model (see
    the module).

    The sections of one depth are the classes of one c-TF-IDF, so a chapter's terms are weighed
    against the other chapters' and a section's against the other sections'. A depth with one
    section only is weighed against itself. With a model, each section's best `CANDIDATES` terms
    are embedded, in one call for the whole document, and reranked against the section's vector;
    without one, the best terms by weight are its keywords."""

    def pick(
        self,
        texts: Sequence[str],
        runs: Sequence[Run],
        vectors: np.ndarray | None,
        embed: Embed | None,
    ) -> list[dict[str, int]]:
        forms: Counter[Term] = Counter()
        per_chunk = [counted(terms(text), forms) for text in texts]
        counts = [summed(per_chunk[run.first : run.last + 1]) for run in runs]
        weights: list[dict[str, float]] = [{} for _ in runs]
        for depth in {run.depth for run in runs}:
            at = [n for n, run in enumerate(runs) if run.depth == depth]
            for n, weighed in zip(at, ctfidf([counts[n] for n in at]), strict=True):
                # every use weighs how common a term is; only a term used enough is a candidate
                kept = frequent(counts[n], KEYWORDS)
                weights[n] = {key: weight for key, weight in weighed.items() if key in kept}
        if embed is None or vectors is None:
            chosen = [best(weighed, KEYWORDS) for weighed in weights]
        else:
            chosen = _reranked(vectors, weights, forms, embed)
        written = shown([key for keys in chosen for key in keys], forms)
        return [{written[key]: counts[n][key] for key in keys} for n, keys in enumerate(chosen)]


def _reranked(
    vectors: np.ndarray,
    weights: list[dict[str, float]],
    forms: Counter[Term],
    embed: Embed,
) -> list[list[str]]:
    """Each section's keywords: its best `CANDIDATES` terms, reranked against its unit vector
    (`rerank`). Every distinct candidate of the document is embedded once."""
    candidates = [best(weighed, CANDIDATES) for weighed in weights]
    distinct = list(dict.fromkeys(key for keys in candidates for key in keys))
    if not distinct:
        return [[] for _ in weights]
    written = shown(distinct, forms)
    embedded = unit_rows(embed([written[key] for key in distinct]))
    position = {key: at for at, key in enumerate(distinct)}
    return [
        [keys[at] for at in rerank(embedded[[position[key] for key in keys]], section, KEYWORDS)]
        for keys, section in zip(candidates, vectors, strict=True)
    ]


CLASS_TFIDF: Strategy = ClassTfidf()
