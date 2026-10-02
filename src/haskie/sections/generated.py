"""Descriptors a language model writes: it reads each section and names its topics.

The `Generated` strategy of `descriptors.Strategy`. Each section is one prompt: its heading path and
an excerpt of its prose (`excerpt`), and the model answers with up to five short noun phrases
(`parse`). Judged blind on 200 sections of four technical books, Gemma-4-E2B's descriptors scored
4.04 of 5 against 2.13 for `descriptors.ClassTfidf`, both where c-TF-IDF does well and where it
does not: c-TF-IDF picks the words a section uses most distinctively, which in a technical book are
often code identifiers and names, while the model names concepts. It sees one section at a time,
so it does not set a section apart from the ones beside it. A phrase the heading path already says
is dropped (`unsaid`), unless nothing else is left.

No IO here: the caller passes the function that asks the model.
"""

from collections.abc import Callable, Sequence

import numpy as np

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
