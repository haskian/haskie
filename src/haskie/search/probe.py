"""The words of a question the answer never mentions, searched for once more.

A search ranks by the whole question, so the part most of the question is about fills the slots,
and a word the rest of it hangs on can go missing: "keep inventory consistent when an order is
placed" answered by five passages on orders, none with "inventory" in it. So after the sections are
kept, each question's words (`thin.terms`: no stopwords, no short ones) are looked for in their
text and headings (`vocabulary`). The words none of them holds are searched for by full text,
alone, and the best new passage that search finds joins the answer (`retrieval.probe_gaps`): in the
kept section it belongs to, else as one excerpt past `limit`. Taking a ranked section's slot instead
would trade one gap for another: a search of one excerpt would lose its whole answer. It is the
only evidence for those words, so the answer's budget gives it room first (`section.within`).

A synonym defeats this ("invariant" for "consistent"); it asks only for the words a question used,
in any form the stemmer knows as the same word. What is still missing after that is part of the
answer (`Answer.missing_terms`): the agent learns what its sources did not say. No IO here.
"""

from collections.abc import Iterable
from functools import lru_cache

import msgspec
import snowballstemmer

from haskie.search import thin
from haskie.search.collapse import WORD
from haskie.search.passage import Answer, Excerpt
from haskie.search.section import Group

PROBE_SCAN = 20  # hits the probe's full-text search reads: its best new passage is what it wants


class Question(msgspec.Struct, frozen=True):
    """One question of a search, as the steps after the ranking read it."""

    vector: list[float] | None  # the embedding of its framed form, None for a lexical search
    asked: str  # the question alone: what its words are read from, and the answer should hold
    label: str | None = None  # the question to tag passages with, when several were asked


_ENGLISH = snowballstemmer.stemmer("english")
# Distinct words kept stemmed: more than the whole vocabulary of a large English corpus, so the
# words of every search stay in it, and a bound on the memory a stream of odd tokens can take.
STEM_CACHE = 2**15


@lru_cache(maxsize=STEM_CACHE)
def stem(word: str) -> str:
    """A word's stem, by the Snowball English stemmer, the one LanceDB's full-text index uses:
    "deployment" and "deploy" are "deploy", "keeps" and "keeping" are "keep", while "category" is
    not "cat". An irregular form ("ran" for "run") has a stem of its own."""
    return _ENGLISH.stemWord(word)


def vocabulary(texts: Iterable[str]) -> set[str]:
    """The stems of every word of `texts`: a text holds a word when it holds its stem.

    Each distinct word is stemmed once, and `stem` remembers it: the stemmer is pure Python, and
    an answer of 36,000 characters repeats most of its words. Measured on one such answer (1,112
    distinct words): 11 ms with every word new, 0.65 ms once they are known, as they are for the
    second read of every search (`probe.report` over what `retrieval.probe_gaps` read) and for
    the vocabulary a running server has seen. So a search reads it on its own event loop rather
    than waiting for a slot of the CPU budget (`cpu.on_cpu`), which indexing may hold."""
    words = {word for text in texts for word in WORD.findall(text.lower())}
    return {stem(word) for word in words}


def missing(questions: list[Question], covered: Iterable[str]) -> dict[str, list[Question]]:
    """Each word of the questions (`thin.terms`) that `covered` does not hold (`vocabulary`), in
    the order asked, with the questions that used it."""
    held = vocabulary(covered)
    found: dict[str, list[Question]] = {}
    for question in questions:
        for word in thin.terms(question.asked):
            if stem(word) not in held:
                found.setdefault(word, []).append(question)
    return found


def covered(groups: list[Group]) -> list[str]:
    """What the kept sections say: their passages' text and every heading they sit under."""
    return [text for one in groups for hit in one.hits for text in (hit.text, *hit.headings)]


def placed(groups: list[Group], found: Group) -> list[Group]:
    """The groups with the probe's section in them: joined to the kept section it is part of,
    else after the others."""
    for at, one in enumerate(groups):
        if (one.collection, one.document_id, one.section) == (
            found.collection,
            found.document_id,
            found.section,
        ):
            joined = msgspec.structs.replace(one, ranges=[*one.ranges, *found.ranges])
            return [*groups[:at], joined, *groups[at + 1 :]]
    return [*groups, found]


def report(excerpts: list[Excerpt], questions: list[Question], searched: list[str]) -> Answer:
    """The answer, with what it lacks: the questions no excerpt names, when several were asked,
    and the words of any question that no excerpt's text or headings hold."""
    text = [
        piece
        for one in excerpts
        for piece in (one.text, one.header, *(span.header for span in one.spans))
    ]
    named = {label for one in excerpts for label in one.aspects}
    return Answer(
        excerpts=excerpts,
        uncovered=[
            question.label
            for question in questions
            if question.label is not None and question.label not in named
        ],
        missing_terms=list(missing(questions, text)),
        searched=searched,
    )
