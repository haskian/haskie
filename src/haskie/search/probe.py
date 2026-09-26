"""The words of a question the answer never mentions, searched for once more.

A search ranks by the whole question, so the part most of the question is about fills the slots,
and a word the rest of it hangs on can go missing: "keep inventory consistent when an order is
placed" answered by five passages on orders, none with "inventory" in it. So after the sections
are kept, each question's words (`thin.terms`: no stopwords, no short ones) are looked for in
their text and headings. The words none of them holds are searched for by full text, alone, and
the best new passage that search finds joins the answer (`retrieval.probe_gaps`): in the kept
section it belongs to, else as one excerpt past `limit`. Taking a ranked section's slot instead
would trade one gap for another: a search of one excerpt would lose its whole answer.

A synonym defeats this ("invariant" for "consistent"); it asks only for the exact words a
question used. What is still missing after that is part of the answer (`Answer.missing_terms`):
the agent learns what its sources did not say. No IO here.
"""

import bisect
from collections.abc import Iterable

import msgspec

from haskie.search import thin
from haskie.search.collapse import WORD
from haskie.search.passage import Answer, Excerpt, HitRange
from haskie.search.section import Group

PROBE_SCAN = 20  # hits the probe's full-text search reads: its best new passage is what it wants


class Question(msgspec.Struct, frozen=True):
    """One question of a search, as the steps after the ranking read it."""

    text: str  # what it was searched with, the shared context included
    vector: list[float] | None  # its embedding, None for a lexical search
    asked: str  # the question alone, whose words the answer should hold
    label: str | None = None  # the question to tag passages with, when several were asked


def terms(text: str) -> list[str]:
    """The words of `text` a passage on its topic would share (`thin.terms`), in order."""
    wanted = thin.terms(text)
    return [word for word in dict.fromkeys(WORD.findall(text.lower())) if word in wanted]


def stem(word: str) -> str:
    """A crude stem: the word less its last two letters, but at least four. "keep" is held by
    "keeps" and "keeping", "order" by "orders", "consistent" by "consistency". It knows no
    language, so an irregular form ("ran" for "run") still reads as missing."""
    return word[: max(4, len(word) - 2)]


def missing(questions: list[Question], covered: Iterable[str]) -> dict[str, list[Question]]:
    """Each word of the questions no text of `covered` holds, in the order asked, with the
    questions that used it. A text holds a word when one of its words begins with its stem."""
    held = sorted({word for text in covered for word in WORD.findall(text.lower())})
    found: dict[str, list[Question]] = {}
    for question in questions:
        for word in terms(question.asked):
            if not _holds(held, stem(word)):
                found.setdefault(word, []).append(question)
    return found


def _holds(words: list[str], prefix: str) -> bool:
    """Whether a word of `words` (sorted) begins with `prefix`."""
    at = bisect.bisect_left(words, prefix)
    return at < len(words) and words[at].startswith(prefix)


def covered(groups: list[Group]) -> list[str]:
    """What the kept sections say: their passages' text and every heading they sit under."""
    return [
        text
        for one in groups
        for hit_range in one.ranges
        for hit in hit_range.hits
        for text in (hit.text, *hit.headings)
    ]


def tags(hit_range: HitRange, wanted: dict[str, list[Question]]) -> list[str]:
    """The questions whose missing words a passage the probe found holds, in the order asked."""
    held = sorted({word for hit in hit_range.hits for word in WORD.findall(hit.text.lower())})
    labels = [
        question.label
        for word, questions in wanted.items()
        if _holds(held, stem(word))
        for question in questions
        if question.label is not None
    ]
    return list(dict.fromkeys(labels))


def placed(groups: list[Group], found: Group) -> tuple[list[Group], str]:
    """The groups with the probe's section in them, and where it went: into the kept section it
    is part of, else after the others."""
    for at, one in enumerate(groups):
        if (one.collection, one.document, one.section) == (
            found.collection,
            found.document,
            found.section,
        ):
            joined = msgspec.structs.replace(one, ranges=[*one.ranges, *found.ranges])
            return [*groups[:at], joined, *groups[at + 1 :]], "joined"
    return [*groups, found], "added"


def report(excerpts: list[Excerpt], questions: list[Question]) -> Answer:
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
    )
