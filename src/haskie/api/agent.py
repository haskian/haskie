"""What an agent reads: the MCP tools' answers, with fewer fields than the REST routes'.

A search answers the web UI with every field it draws from: offsets, chunk numbers, lines and
pages apart. An agent cites a result by `header` and `location`, which already name the document,
the pages and the lines, so the rest only costs it tokens. Each tool here has its own handler, the
REST route's twin with the same arguments and the same search behind it, and answers with one of
these views instead (`view`). An empty list or map is left out (`omit_defaults`): an agent reads
a missing `also_in` as none.

The views name only what they keep; converting a full answer drops the rest, so a field added to
an answer stays out of the tools until it is added here.
"""

import msgspec

from haskie.collection.index import Overlaps, Relation


class _View(msgspec.Struct, omit_defaults=True):
    pass


class Place(_View):
    """A place that says what a passage says (`PassageReference`)."""

    collection: str
    document_id: str
    document: str
    header: str
    location: str
    score: float
    relation: Relation
    similarity: float
    to_parent: Overlaps
    to_root: Overlaps
    also_in: "list[Place]" = []


class Span(_View):
    """One passage of an excerpt (`passage.Span`)."""

    header: str
    section_id: str
    location: str
    score: float
    also_in: list[Place] = []
    aspects: list[str] = []
    aspect_scores: dict[str, float] = {}


class Excerpt(_View):
    """One section of a document, as an agent quotes it (`passage.Excerpt`)."""

    collection: str
    document_id: str
    document: str
    header: str
    section_id: str
    location: str
    text: str
    score: float
    markdown_file: str
    spans: list[Span] = []
    aspects: list[str] = []


class Answer(msgspec.Struct):
    """`search_excerpts` (`passage.Answer`); every field always there, empty or not."""

    excerpts: list[Excerpt]
    uncovered: list[str]
    missing_terms: list[str]


class HotSection(_View):
    header: str
    location: str
    score: float
    chunks: int


class Source(_View):
    """One document that answers (`passage.Source`)."""

    collection: str
    document_id: str
    document: str
    score: float
    chunks: int
    header: str
    location: str
    text: str
    markdown_file: str
    description: str = ""
    collections: list[str] = []
    sections: list[HotSection] = []


class Sources(msgspec.Struct):
    documents: list[Source]
    collections: list[str]


class Related(_View):
    """A section the map did not pick, under the pick that covers it (`section_map.Related`)."""

    collection: str
    document_id: str
    document: str
    header: str
    id: str
    location: str
    score: float
    similarity: float


class MappedSection(_View):
    """One section of the map (`section_map.MappedSection`)."""

    collection: str
    document_id: str
    document: str
    header: str
    id: str
    location: str
    score: float
    chars: int
    chunks: int
    markdown_file: str
    descriptors: list[str] = []
    related: list[Related] = []


class SectionMap(msgspec.Struct):
    sections: list[MappedSection]
    collections: list[str]


def view[T](found: object, as_type: type[T]) -> T:
    """`found`, a full answer, as the view `as_type`: the fields it does not name are dropped."""
    return msgspec.convert(msgspec.to_builtins(found), as_type)
