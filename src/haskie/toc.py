"""Markdown table of contents via pyromark."""

import msgspec
import pyromark


class Heading(msgspec.Struct):
    level: int
    text: str
    offset: int  # byte offset of the heading in the markdown


def headings(markdown: str) -> list[Heading]:
    result: list[Heading] = []
    current: Heading | None = None
    for event, span in pyromark.events_with_range(markdown):
        match event:
            case {"Start": {"Heading": {"level": level}}}:
                current = Heading(level=int(str(level)[1]), text="", offset=span["start"])
            case {"Text": str(text)} | {"Code": str(text)} if current is not None:
                current.text += text
            case {"End": {"Heading": _}} if current is not None:
                current.text = current.text.strip()
                result.append(current)
                current = None
    return result
