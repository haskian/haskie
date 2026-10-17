"""Arm H's `UserPromptSubmit` hook: before the agent reads a prompt, search haskie with it and put
the best few passages in front of the agent - steering from the library on every prompt, with no
tool call the agent has to think of making.

Claude Code runs this with the hook's JSON on stdin (`prompt` among it) and adds what it prints to
the prompt's context. It never blocks the prompt: on any failure it prints nothing and exits 0,
so the arm degrades to arm G (haskie connected, not mentioned) rather than to a broken run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request

MAX_QUERY = 500  # what `search_excerpts` takes per question
MAX_QUESTIONS = 5  # and how many questions at once
MIN_WORDS = 6  # a shorter sentence is an instruction ("Standard library only."), not a topic
PASSAGES = 3
SHOWN = 300  # characters of each passage the hook shows
TIMEOUT = 20.0
# `run.prompt_for`'s own first paragraph: about the eval, not the request
HEADER = re.compile(r"\AYou are completing the \S+ evaluation task\..*?\n\n", re.DOTALL)
CODE = re.compile(r"```.*?```|^(?: {4}|\t).*$", re.DOTALL | re.MULTILINE)
SENTENCE = re.compile(r"(?<=[.!?:])\s+|\n\s*\n")


def questions(prompt: str) -> list[str]:
    """The request as search questions: its prose sentences, the eval's header and any code
    aside. A whole prompt as one question reads as noise to a reranker, which then keeps nothing
    above its floor; one sentence a question, several at once, finds what each is about."""
    prose = CODE.sub(" ", HEADER.sub("", prompt))
    found: list[str] = []
    for sentence in SENTENCE.split(prose):
        text = " ".join(sentence.split())
        if len(text.split()) >= MIN_WORDS and text not in found:
            found.append(text if len(text) <= MAX_QUERY else text[:MAX_QUERY].rsplit(" ", 1)[0])
    return found[:MAX_QUESTIONS]


def search(api: str, collection: str, asked: list[str]) -> list[dict]:
    """The best `PASSAGES` excerpts for any of `asked`, best first. The questions take turns at
    the slots (a search needs one slot each at least), so the answer is in turn order: sorted by
    score here, an instruction's weak match does not lead just for being asked first."""
    limit = max(PASSAGES, len(asked))
    params = urllib.parse.urlencode(
        {"q": asked, "collections": collection, "limit": limit}, doseq=True
    )
    url = f"{api.rstrip('/')}/api/search/excerpts?{params}"
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:  # noqa: S310
        excerpts = json.loads(response.read())["excerpts"]
    return sorted(excerpts, key=lambda e: e.get("score", 0.0), reverse=True)[:PASSAGES]


def nudge(collection: str, excerpts: list[dict]) -> str:
    """What the agent reads before the prompt: where each passage is, and its start."""
    if not excerpts:
        return ""
    lines = [
        f"Your document library (haskie, collection `{collection}`) has passages that may bear "
        "on this request:",
        "",
    ]
    for excerpt in excerpts:
        text = " ".join(excerpt.get("text", "").split())
        text = text if len(text) <= SHOWN else text[:SHOWN].rsplit(" ", 1)[0] + " ..."
        lines.append(f"- {excerpt.get('header', '')} ({excerpt.get('location', '')}): {text}")
    lines += ["", "The haskie search tools can show more of any of these."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True)
    parser.add_argument("--collection", required=True)
    args = parser.parse_args(argv)
    try:
        asked = questions(json.load(sys.stdin).get("prompt", ""))
        if asked:
            print(nudge(args.collection, search(args.api, args.collection, asked)))
    except Exception:  # a hook that fails must not fail the prompt
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
