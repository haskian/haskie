"""Descriptors a language model writes: it reads each section and names its topics.

The `Generated` strategy of `descriptors.Strategy`. Each section is one prompt: its heading path and
an excerpt of its prose (`excerpt`), and the model answers with up to six short noun phrases
(`parse`). Section descriptions are written separately (`describe_section`). With the earlier
prompt, judged blind on 200 sections of four technical books, Gemma-4-E2B's descriptors scored
4.04 of 5 against 2.13 for `descriptors.ClassTfidf`, both where c-TF-IDF does well and where it
does not: c-TF-IDF picks the words a section uses most distinctively, which in a technical book are
often code identifiers and names, while the model names concepts. It sees one section at a time,
so it does not set a section apart from the ones beside it. Topics already named in the heading
path remain eligible descriptors.

Once every section has its descriptors, the same model writes what the whole document is about,
in a few sentences (`summarize`): no 4,096-token context holds a book, so it reads the book's
section descriptions and descriptors, summarizing groups before combining them when needed.
A collection's description is written from its documents' (`summarize_collection`).

No IO here: the caller passes the function that asks the model.
"""

import re
from collections.abc import Callable, Sequence

import numpy as np

from haskie.sections.build import Section
from haskie.sections.descriptors import DESCRIPTORS, Description, Embed, Run

type Reply = Callable[[str, int], str]  # the model's answer to one prompt, at most so many tokens

EXCERPT_CHARS = 6000  # of a section's prose the model reads; the judged runs read this much
EXCERPT_CHUNKS = 6  # a longer section is read as this many of its chunks, spread over it
REPLY_TOKENS = 90  # six short descriptors, including their label
SECTION_SENTENCES = 2
SECTION_TOKENS = 120
PROMPT = f"""Below is one section of a book. Write up to {DESCRIPTORS} descriptors: short noun \
phrases (1 to 3 words) that together tell a reader what topics this section covers. Be specific. \
Include relevant topics even when the heading already names them. Do not use generic words such \
as chapter, example, figure. Answer with the descriptors only, separated by " | ".

Heading path: {{heading}}

Section text:
{{text}}"""
SECTION_PROMPT = """Below is one section of a book. Write 1 to 2 short plain sentences describing \
what this section teaches. Start each sentence with a verb, as in "Explains how ..." or \
"Covers ...". Say only what the section text shows. Do not name the book or author, and do not \
write "This section". Answer with the sentences only.

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


def describe_section(run: Run, texts: Sequence[str], reply: Reply) -> str:
    """One section's prose summary, with no model call when there is no prose."""
    text = excerpt(texts[run.first : run.last + 1])
    if not text.strip():
        return ""
    heading = " > ".join(run.headings) or "(the whole document)"
    return sentences(
        reply(SECTION_PROMPT.format(heading=heading, text=text), SECTION_TOKENS), SECTION_SENTENCES
    )


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
    ) -> list[Description]:
        return [self._describe(run, texts[run.first : run.last + 1]) for run in runs]

    def _describe(self, run: Run, texts: Sequence[str]) -> Description:
        text = excerpt(texts)
        if not text.strip():
            return Description()  # a section of code or tables alone: nothing to read
        heading = " > ".join(run.headings) or "(the whole document)"
        return Description(
            descriptors=parse(self._reply(PROMPT.format(heading=heading, text=text), REPLY_TOKENS))
        )


# --- the whole document -------------------------------------------------------------

SUMMARY_CHARS = 10_000  # leaves room for the prompt and reply in the describer's context
SUMMARY_ITEMS = 16  # bounds each reduction and guarantees fewer summaries at the next level
SUMMARY_SENTENCES = 5
SUMMARY_TOKENS = 250
SUMMARY_PROMPT = """Below are descriptions and descriptors of sections of a book, or summaries \
of groups of its sections. Describe what the book teaches as a whole in 2 to 5 plain sentences. \
Combine the themes rather than listing sections one by one. The first sentence says its main \
subject. Start every sentence with a verb, with no subject, as in "Explains how ..." or \
"Covers ...". Never name the book, its title or author. Never write "This document" or \
"This book". Leave out front and back matter such as the preface, contributors, conventions, \
contact details and index. Say only what the section information shows. Answer with the \
sentences only.

Section information:
{sections}"""
_LABEL = re.compile(r"^(?:summary|description)\s*:\s*", re.IGNORECASE)
_LIST_MARK = re.compile(r"^(?:[-*#]+|\d+[.)])\s*")  # "- ", "## ", "1. ", "2) "
# after a sentence's end mark, but not after "e.g." or "i.e.", which a sentence goes on past
_SENTENCE_END = re.compile(r"(?<=[.!?])(?<!e\.g\.)(?<!i\.e\.)\s+")


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


def summarize(sections: Sequence[Section], reply: Reply) -> str:
    """Summarize saved section descriptions and descriptors, including every section. A long
    document is reduced in bounded groups until their summaries fit one prompt."""
    records = [
        f"Heading: {' > '.join(one.headings) or '(the whole document)'}\n"
        f"Description: {one.description}\nDescriptors: {' | '.join(one.descriptors)}"
        for one in sections
        if one.description.strip() or one.descriptors
    ]
    while records:
        summaries: list[str] = []
        start = 0
        while start < len(records):
            group: list[str] = []
            size = 0
            while start < len(records) and len(group) < SUMMARY_ITEMS:
                record = records[start][:SUMMARY_CHARS]
                added = len(record) + (2 if group else 0)
                if group and size + added > SUMMARY_CHARS:
                    break
                group.append(record)
                size += added
                start += 1
            text = "\n\n".join(group)
            summary = sentences(reply(SUMMARY_PROMPT.format(sections=text), SUMMARY_TOKENS))
            if not summary:
                return ""  # do not publish a summary silently missing one group
            summaries.append(summary)
        if len(summaries) == 1:
            return summaries[0]
        records = summaries
    return ""


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
