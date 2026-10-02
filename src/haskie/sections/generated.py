"""Descriptors a language model writes: it reads each section and names its topics.

The `Generated` strategy of `descriptors.Strategy`. Each section is one prompt: its heading path and
an excerpt of its prose (`excerpt`), and the model answers with up to five short noun phrases
(`parse`). Judged blind on 200 sections of four technical books, Gemma-4-E2B's descriptors scored
4.04 of 5 against 2.13 for `descriptors.ClassTfidf`, both where c-TF-IDF does well and where it
does not: c-TF-IDF picks the words a section uses most distinctively, which in a technical book are
often code identifiers and names, while the model names concepts. It sees one section at a time,
so it does not set a section apart from the ones beside it. A phrase the heading path already says
is dropped (`unsaid`), unless nothing else is left.

Once every section has its descriptors, the same model writes what the whole document is about,
in a few sentences (`summarize`): no 4,096-token context holds a book, so it reads the book's
outline, each heading with the descriptors of its section, and an excerpt spread over the book.
A collection's description is written from its documents' (`summarize_collection`).

No IO here: the caller passes the function that asks the model.
"""

import re
from collections.abc import Callable, Sequence

import numpy as np

from haskie.sections.build import Section
from haskie.sections.descriptors import DESCRIPTORS, Embed, Run, said

type Reply = Callable[[str, int], str]  # the model's answer to one prompt, at most so many tokens

EXCERPT_CHARS = 6000  # of a section's prose the model reads; the judged runs read this much
EXCERPT_CHUNKS = 6  # a longer section is read as this many of its chunks, spread over it
REPLY_TOKENS = 60  # five phrases of up to three words take about 30
PROMPT = """Below is one section of a book. Write up to 5 descriptors: short noun phrases (1 to 3 \
words) that together tell a reader what topics this section covers. Be specific. Do not repeat \
the heading. Do not use generic words such as chapter, example, figure. Answer with the \
descriptors only, separated by " | ".

Heading path: {heading}

Section text:
{text}"""


def excerpt(texts: Sequence[str]) -> str:
    """A section's chunks as the model reads them: whole when they fit `EXCERPT_CHARS`, else
    `EXCERPT_CHUNKS` of them spread evenly from the first to the last, each cut to its share, so
    the model sees the whole span of a long section rather than its opening."""
    whole = "\n\n".join(texts)
    if len(whole) <= EXCERPT_CHARS:
        return whole
    picks = dict.fromkeys(np.linspace(0, len(texts) - 1, min(len(texts), EXCERPT_CHUNKS)).round())
    share = EXCERPT_CHARS // len(picks)
    return "\n\n[...]\n\n".join(texts[int(at)][:share] for at in picks)


def parse(answer: str) -> list[str]:
    """The descriptors of an answer: its first line that holds " | ", else its first line, split
    there. A small model sometimes writes a sentence before the list, or a label ahead of it on
    its line ("Descriptors: ..."), which is dropped."""
    lines = [line for line in answer.splitlines() if line.strip()]
    line = next((line for line in lines if "|" in line), lines[0] if lines else "")
    if ":" in line.split("|")[0]:
        line = line.split(":", 1)[1]
    phrases = (phrase.strip(" \t.*-\"'`") for phrase in line.split("|"))
    return [phrase for phrase in phrases if phrase][:DESCRIPTORS]


class Generated:
    """Each section described by `reply`, one prompt each (see the module)."""

    def __init__(self, reply: Reply) -> None:
        self._reply = reply

    def pick(
        self,
        texts: Sequence[str],
        runs: Sequence[Run],
        vectors: np.ndarray | None,
        embed: Embed | None,
    ) -> list[list[str]]:
        return [self._describe(run, texts[run.first : run.last + 1]) for run in runs]

    def _describe(self, run: Run, texts: Sequence[str]) -> list[str]:
        text = excerpt(texts)
        if not text.strip():
            return []  # a section of code or tables alone: nothing to read
        heading = " > ".join(run.headings) or "(the whole document)"
        found = parse(self._reply(PROMPT.format(heading=heading, text=text), REPLY_TOKENS))
        return unsaid(found, run.headings) or found


def unsaid(phrases: list[str], headings: Sequence[str]) -> list[str]:
    """The phrases that say more than the heading path: not those whose every word it holds, by
    the rule c-TF-IDF drops such terms by (`descriptors.said`). The reader sees the path beside
    them. Judged blind on 200 sections: dropping them scored as keeping them (3.76 of 5 against
    3.79, the same list in 183 sections), and leaves none of the 4.5% of phrases that were such.
    Asking the model not to use the heading's words did worse (3.52 against 3.89), as it then
    left out the section's main topic, and asking for eight phrases to drop them from was no
    better (3.73) and 30% slower."""
    header = said(headings)
    return [phrase for phrase in phrases if not (words := said([phrase])) or not words <= header]


# --- the whole document -------------------------------------------------------------

OUTLINE_CHARS = 4000  # with an excerpt of `EXCERPT_CHARS`, about 2,500 of the context's tokens
SUMMARY_SENTENCES = 5  # at most; the prompt asks for 2 to 5
SUMMARY_TOKENS = 250  # five sentences of about 30 words take about 200
# Tried on ten books. A prompt that gave the file name and asked what the document is about
# opened every answer with "This document is a book titled ...", and named the title and author.
# Sentences that start with a verb, without the file name, kept both out of all ten; topics as
# subjects did too, but read worse. Both then listed front and back matter, which the excerpt's
# first and last chunks often are, until the last rule but one.
SUMMARY_PROMPT = """Below are the outline and excerpts of a book. Write 2 to 5 plain sentences on \
what it teaches. The first sentence says its main subject. Start every sentence with a verb, \
with no subject, as in "Explains how ..." or "Covers ...". Never name the book, its title or its \
author, and never write "This document" or "This book". Leave out the front and back matter: \
preface, contributors, conventions, how to use the examples, contact details, the index. Say \
only what the outline and the excerpts show. Answer with the sentences only.

Outline (each heading, with the topics its section covers):
{outline}

Excerpts:
{text}"""
_LABEL = re.compile(r"^(?:summary|description)\s*:\s*", re.IGNORECASE)
_LIST_MARK = re.compile(r"^(?:[-*#]+|\d+[.)])\s*")  # "- ", "## ", "1. ", "2) "
# after a sentence's end mark, but not after "e.g." or "i.e.", which a sentence goes on past
_SENTENCE_END = re.compile(r"(?<=[.!?])(?<!e\.g\.)(?<!i\.e\.)\s+")


def outline(sections: Sequence[Section]) -> str:
    """The document's headings, indented by depth, each with its section's descriptors. Deeper
    headings are left out until it fits `OUTLINE_CHARS`, then it is cut there: a book whose one
    first-level heading is its title still shows its chapters, and a long book its parts."""
    depth = max((len(one.headings) for one in sections), default=0)
    text = ""
    for deepest in range(depth, 0, -1):
        text = "\n".join(
            "  " * (len(one.headings) - 1)
            + f"- {one.headings[-1]}"
            + (f" ({', '.join(one.descriptors)})" if one.descriptors else "")
            for one in sections
            if 1 <= len(one.headings) <= deepest
        )
        if len(text) <= OUTLINE_CHARS:
            return text
    return text[:OUTLINE_CHARS]


def sentences(answer: str, most: int = SUMMARY_SENTENCES) -> str:
    """The summary in an answer: its lines as one paragraph, without bold marks, list marks or a
    label ahead of it ("Summary: ..."), cut to `most` sentences. A last sentence cut off by the
    token limit is dropped, unless it is the only one."""
    lines = (_LIST_MARK.sub("", line.strip()) for line in answer.replace("**", "").splitlines())
    text = _LABEL.sub("", " ".join(line for line in lines if line))
    kept = _SENTENCE_END.split(text)[:most]
    if len(kept) > 1 and not kept[-1].endswith((".", "!", "?")):
        kept.pop()
    return " ".join(kept)


def summarize(sections: Sequence[Section], texts: Sequence[str], reply: Reply) -> str:
    """What a document is about, in a few sentences, from its described `sections` and every
    chunk's prose in `texts` (see the module); empty for a document with neither headings nor
    prose, which gives the model nothing to read."""
    text = excerpt(texts)
    headings = outline(sections)
    if not text.strip() and not headings:
        return ""
    prompt = SUMMARY_PROMPT.format(outline=headings or "(none)", text=text)
    return sentences(reply(prompt, SUMMARY_TOKENS))


# --- a collection ------------------------------------------------------------------

# its documents' descriptions the model reads: about 2,500 tokens, which leaves the prompt and
# the reply room in the context. A collection whose descriptions do not fit reads each cut to an
# equal share: a few hundred documents leave each a phrase, the ceiling of one prompt.
COLLECTION_CHARS = 10_000
COLLECTION_SENTENCES = 7  # at most; the prompt asks for 3 to 7
COLLECTION_TOKENS = 350  # seven sentences of about 30 words take about 280
# Tried on two shelves of three and ten books, their descriptions written by `summarize`. A
# prompt that listed them and asked what the collection covers made the model copy the first ones
# through, sentence by sentence; asking it to step back, not to go one document at a time and not
# to reuse their wording, and to name the field and the themes that run through several of them,
# gave a summary of the whole on both. Another that only asked it to generalize repeated itself on
# three books.
COLLECTION_PROMPT = """A person collected {count} documents. Their descriptions:

{descriptions}

Step back and describe the collection as a whole in 3 to 7 short sentences. Do not summarize the \
documents one by one, and do not reuse their wording. Say the field they belong to, the 3 to 5 \
themes that run through several of them, and what a reader of the whole collection learns. Start \
every sentence with a verb, with no subject, as in "Covers ..." or "Spans ...". Never name a \
document, a title or an author, and never write "This collection". Answer with the sentences \
only."""


def summarize_collection(descriptions: Sequence[str], reply: Reply) -> str:
    """What a collection is about, in a few sentences, from its documents' `descriptions`; empty
    when none has one, which gives the model nothing to read."""
    kept = [one.strip() for one in descriptions if one.strip()]
    if not kept:
        return ""
    share = COLLECTION_CHARS // len(kept)
    listed = "\n".join(f"{at}. {one[:share]}" for at, one in enumerate(kept, 1))
    prompt = COLLECTION_PROMPT.format(count=len(kept), descriptions=listed)
    answer = reply(prompt, COLLECTION_TOKENS)
    return sentences(answer, COLLECTION_SENTENCES)
