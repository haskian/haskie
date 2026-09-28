"""Structure-Aware Chunking (`indexing/chunk.py`, `indexing/segment.py`).

Every case runs through `chunk.split`, the one entry the indexer calls, and every chunk of every
case is held to the same contract by `_check` - the offsets cut out exactly its text, the text is
its sentences joined, no chunk is longer than the chunk size, and no visible character of the
input is lost - so a case only has to state what is particular to it.
"""

import pytest
from conftest import MD

from haskie.document import render
from haskie.document.convert import PAGE_MARKER, without_markers
from haskie.indexing import chunk, segment
from haskie.indexing.chunk import Chunk, Piece
from haskie.indexing.segment import CutReason, PieceType
from haskie.settings import Chunker, ChunkSettings

WIDE = ChunkSettings()  # 1200: every short case fits one chunk
SMALL = ChunkSettings(chunk_size=40)


def _check(text: str, settings: ChunkSettings, chunks: list[Chunk], byte_offset: int = 0) -> None:
    data = text.encode()
    for c in chunks:
        starts = [*(p.position for p in c.layout), len(c.text)]
        cut = [
            Piece(p.type, c.text[a:b])
            for p, a, b in zip(c.layout, starts, starts[1:], strict=False)
        ]
        assert cut == c.pieces, "the layout cuts the text back into the pieces, types and all"
        source = text[c.char_start : c.char_end]
        assert without_markers(source) == c.text == "".join(_texts(c))
        assert not PAGE_MARKER.search(c.text), "a page marker is a page, never text"
        assert data[c.byte_start - byte_offset : c.byte_end - byte_offset].decode() == source
        assert c.text == c.text.strip() and c.pieces and all(_texts(c))
        assert len(chunk.framed(c.frame, c.text)) <= settings.chunk_size, "the frame counts"
        if settings.chunker == "markdown":
            alone = all(p.type == "heading" for p in c.pieces)
            text_pieces = [p for p in c.pieces if p.type != "heading"]
            assert alone or len(text_pieces) == len(c.pieces), "headings only in their own chunk"
            assert not any(render.headings(p.text) for p in text_pieces), "headings are the frame"
            assert not (alone and c.frame and c.frame[-1] == c.headings[-1]), "never read twice"
        assert c.line_start == text.count("\n", 0, c.char_start) + 1
        assert c.line_end == text.count("\n", 0, c.char_end) + 1
    spans = [(c.char_start, c.char_end) for c in chunks]
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False)), "no overlap, in order"
    covered = set().union(*(range(c.char_start, c.char_end) for c in chunks))
    content = PAGE_MARKER.sub(lambda m: " " * len(m[0]), text)  # a lone marker is dropped
    if settings.chunker == "markdown":  # a heading is read from the frame, never the text
        for block in segment.blocks(text):
            if block.kind == "heading":
                content = (
                    content[: block.start] + " " * (block.end - block.start) + content[block.end :]
                )
    lost = "".join(c for i, c in enumerate(content) if not c.isspace() and i not in covered)
    # a chunk without a word is never made (`chunk._worded`), so a separator may be in none
    assert not any(c.isalnum() for c in lost), f"text in no chunk: {lost!r}"


def _texts(c: Chunk) -> list[str]:
    return [piece.text for piece in c.pieces]


def _split(text: str, settings: ChunkSettings) -> list[Chunk]:
    chunks = chunk.split(text, settings)
    _check(text, settings, chunks)
    return chunks


# --- one table: what each case is cut into ------------------------------------------


@pytest.mark.parametrize(
    ("name", "text", "settings", "expected"),
    [
        # nothing to chunk
        ("empty text", "", WIDE, []),
        ("only whitespace", " \n\n\t\n", WIDE, []),
        ("only page markers", "<!-- page 1 -->\n\n<!-- page 2: needs OCR, skipped -->\n", WIDE, []),
        # sentences
        ("one line, no heading", "plain text", WIDE, [([], ["plain text"])]),
        (
            "sentences on one line",
            "One is here. Two is here! Three?",
            WIDE,
            [([], ["One is here. ", "Two is here! ", "Three?"])],
        ),
        (
            "each sentence on its own line",
            "First one.\nSecond one.\nThird one.",
            WIDE,
            [([], ["First one.\n", "Second one.\n", "Third one."])],
        ),
        (
            "a hard-wrapped sentence is not cut at the line break",
            "This sentence is\nwrapped across\nlines. Next one.",
            WIDE,
            [([], ["This sentence is\nwrapped across\nlines. ", "Next one."])],
        ),
        (
            "abbreviations and decimals do not end a sentence",
            "Use e.g. this one. Version 3.14 works.",
            WIDE,
            [([], ["Use e.g. this one. ", "Version 3.14 works."])],
        ),
        (
            "a sentence inside parentheses",
            "It fails. (The retry happens later.) Then it works.",
            WIDE,
            [([], ["It fails. ", "(The retry happens later.) ", "Then it works."])],
        ),
        (
            "Chinese full stops need no spaces",
            "我們需要一個數據庫。它必須可靠！你同意嗎？",
            WIDE,
            [([], ["我們需要一個數據庫。", "它必須可靠！", "你同意嗎？"])],
        ),
        (
            "an opening bracket goes to the sentence it opens",
            "寄美訪價。」《日記》此後一段時間。",
            WIDE,
            [([], ["寄美訪價。」", "《日記》此後一段時間。"])],
        ),
        (
            "hard-wrapped Chinese is not cut at every line",
            "可見原亮三郎作為教科\n書的大龍頭。難怪他雖\n然在銀行界十分活躍。",
            WIDE,
            [([], ["可見原亮三郎作為教科\n書的大龍頭。", "難怪他雖\n然在銀行界十分活躍。"])],
        ),
        # inline markdown is not punctuation
        (
            "bold opening a sentence goes with it",
            "Ends here. **Bold starts** the next one.",
            WIDE,
            [([], ["Ends here. ", "**Bold starts** the next one."])],
        ),
        (
            "a link's dots and brackets do not cut",
            "See [the docs](http://a.b/c.d). Done.",
            WIDE,
            [([], ["See [the docs](http://a.b/c.d). ", "Done."])],
        ),
        (
            "a footnote marker stays with its sentence",
            "It fails.<sup>ii</sup> Next sentence.",
            WIDE,
            [([], ["It fails.<sup>ii</sup> ", "Next sentence."])],
        ),
        (
            "a tag whose name only starts like sup or sub hides nothing",
            "Fill in the <subject> line.\n\nOne is here. Two is here. Three is here.",
            ChunkSettings(chunk_merge_below=0),
            [
                ([], ["Fill in the <subject> line."]),
                ([], ["One is here. ", "Two is here. ", "Three is here."]),
            ],
        ),
        (
            "an unclosed footnote marker hides no more than the rest of its own block",
            "Note.<sup>1 Unclosed footnote\n\nOne is here. Two is here.",
            ChunkSettings(chunk_merge_below=0),
            [([], ["Note.<sup>1 Unclosed footnote"]), ([], ["One is here. ", "Two is here."])],
        ),
        (
            "inline code keeps its dots",
            "Call `os.path.join()` here. Then stop.",
            WIDE,
            [([], ["Call `os.path.join()` here. ", "Then stop."])],
        ),
        # blocks kept whole
        (
            "a table is one piece, its cells' sentences uncut",
            "Before it.\n\n| a | b |\n|---|---|\n| 1. x | Two. Three. |\n\nAfter it.",
            WIDE,
            [
                (
                    [],
                    [
                        "Before it.\n\n",
                        "| a | b |\n|---|---|\n| 1. x | Two. Three. |\n\n",
                        "After it.",
                    ],
                )
            ],
        ),
        (
            "a code block is one piece",
            "Run this. Now.\n\n```py\nfoo(). bar(). baz()\n```\n\nIt works.",
            WIDE,
            [([], ["Run this. ", "Now.\n\n", "```py\nfoo(). bar(). baz()\n```\n\n", "It works."])],
        ),
        (
            "a rule and raw HTML are pieces of their own",
            "One. Two.\n\n---\n\n<div>Three. Four.</div>\n\nFive.",
            WIDE,
            [([], ["One. ", "Two.\n\n", "---\n\n", "<div>Three. Four.</div>\n\n", "Five."])],
        ),
        # lists and quotes
        (
            "tight list: the marker opens the item's first sentence",
            "- One. Two.\n- Three.\n",
            WIDE,
            [([], ["- One. ", "Two.\n", "- Three."])],
        ),
        (
            "numbered list whose items are sentences",
            "1. First step. Do it.\n2. Second step.\n",
            WIDE,
            [([], ["1. First step. ", "Do it.\n", "2. Second step."])],
        ),
        (
            "nested list: the parent's own text is its own prose",
            "- Parent one. Parent two.\n  - Child one. Child two.\n",
            WIDE,
            [([], ["- Parent one. ", "Parent two.\n  ", "- Child one. ", "Child two."])],
        ),
        (
            "loose list: each item a paragraph",
            "- Loose one. More.\n\n- Loose two.\n",
            WIDE,
            [([], ["- Loose one. ", "More.\n\n", "- Loose two."])],
        ),
        (
            "blockquote: the continuation `>` does not cut",
            "> Quoted one. Quoted\n> two.\n",
            WIDE,
            [([], ["> Quoted one. ", "Quoted\n> two."])],
        ),
        # headings
        (
            "stacked headings with nothing under them make no chunk",
            "# A\n## B\n### C\n#### D\n##### E\n",
            WIDE,
            [],
        ),
        (
            "stacked headings open one section, filed under the deepest",
            "# A\n## B\n### C\nOne is here.\nTwo is here.\nThree is here.\n",
            WIDE,
            [(["A", "B", "C"], ["One is here.\n", "Two is here.\n", "Three is here."])],
        ),
        (
            "stacked headings longer than a chunk are the path, not the text",
            "# A\n## B\n### C\n#### D\n##### E\n\nThe body.",
            ChunkSettings(chunk_size=20),
            [(["A", "B", "C", "D", "E"], ["The body."])],
        ),
        (
            "sibling headings with nothing under them are separate sections, not a stack",
            "<!-- page 1 -->\n\n## Chapter 1\n\n<!-- page 2 -->\n\n## Chapter 2\n\nText.",
            WIDE,
            [(["Chapter 2"], ["Text."])],
        ),
        (
            "a shallower heading after an empty one closes it too",
            "## Sub\n# Top\nText.",
            WIDE,
            [(["Top"], ["Text."])],
        ),
        (
            "an empty heading, then a deeper one with text: stacked, so one section under both",
            "# A\n\n## B\n\nText.",
            WIDE,
            [(["A", "B"], ["Text."])],
        ),
        (
            "an empty heading, then a sibling with text: only the text is a chunk",
            "# A\n\n# B\n\nText.",
            WIDE,
            [(["B"], ["Text."])],
        ),
        (
            "a document of headings alone has no chunk",
            "# Book\n\n# Index\n\n## Terms\n",
            WIDE,
            [],
        ),
        (
            "every heading after text starts a new chunk, even when everything fits",
            "# A\nOne is here.\n## B\nTwo is here.\n### C\nThree is here.\n",
            WIDE,
            [
                (["A"], ["One is here."]),
                (["A", "B"], ["Two is here."]),
                (["A", "B", "C"], ["Three is here."]),
            ],
        ),
        (
            "heading path: siblings replace, deeper ones append",
            "# A\n## B\n### C\nc.\n## D\nd.\n",
            ChunkSettings(chunk_size=8),
            [
                (["A", "B", "C"], ["c."]),
                (["A", "D"], ["d."]),
            ],
        ),
        (
            "a section's first chunk starts at its first sentence, its headings in the path",
            MD,
            SMALL,
            [
                (["Title"], ["intro text"]),
                (["Title", "Alpha"], ["alpha body about lancedb"]),
                (["Title", "Beta"], ["beta body"]),
            ],
        ),
        (
            "text before the first heading has no heading",
            "pre\n# H\nbody",
            ChunkSettings(chunk_size=8),
            [([], ["pre"]), (["H"], ["body"])],
        ),
        (
            "text before the first heading is its own chunk, even when everything fits",
            "pre\n# H\nbody",
            WIDE,
            [([], ["pre"]), (["H"], ["body"])],
        ),
        (
            "a page marker ahead of a heading does not hide it",
            "<!-- page 1 -->\n\n# H\nbody",
            WIDE,
            [(["H"], ["body"])],
        ),
        (
            "a page marker between paragraphs is taken out, the blank line kept",
            "One. Two.\n\n<!-- page 2 -->\n\nThree. Four.",
            WIDE,
            [([], ["One. ", "Two.\n\n", "Three. ", "Four."])],
        ),
        (
            "a page marker with nothing after it is in no chunk",
            "# H\n\nOne. Two.\n\n<!-- page 2 -->\n",
            WIDE,
            [(["H"], ["One. ", "Two."])],
        ),
        (
            "text chunker: a page marker is no paragraph and no text",
            "One.\n\n<!-- page 2 -->\n\nTwo.\n\n<!-- page 3 -->",
            ChunkSettings(chunker=Chunker.TEXT),
            [([], ["One.\n\n", "Two."])],
        ),
        (
            "setext headings are headings",
            "Title\n=====\n\nOne. Two.\n\nPart\n----\n\nThree.",
            WIDE,
            [
                (["Title"], ["One. ", "Two."]),
                (["Title", "Part"], ["Three."]),
            ],
        ),
        (
            "a heading straight into a table",
            "# Prices\n| a | b |\n|---|---|\n| 1 | 2 |\n",
            WIDE,
            [(["Prices"], ["| a | b |\n|---|---|\n| 1 | 2 |"])],
        ),
        (
            "a heading as the last thing in the text makes no chunk",
            "Body one.\n\n# Tail",
            ChunkSettings(chunk_size=12),
            [([], ["Body one."])],
        ),
        (
            "an HTML comment that is not a page marker is content",
            "<!-- a note. two -->\n\nText.",
            WIDE,
            [([], ["<!-- a note. two -->\n\n", "Text."])],
        ),
        (
            "a paragraph with no words, only markup",
            "![a diagram](fig.png)\n\nText.",
            WIDE,
            [([], ["![a diagram](fig.png)\n\n", "Text."])],
        ),
        # the text chunker
        (
            "text chunker: markdown is prose too",
            "# Not a heading. Second.\n\n| a. | b. |",
            ChunkSettings(chunker=Chunker.TEXT),
            [(["Not a heading. Second."], ["# Not a heading. ", "Second.\n\n", "| a. | b. |"])],
        ),
        (
            "text chunker: whole document in one chunk",
            MD,
            ChunkSettings(chunker=Chunker.TEXT, chunk_size=1000),
            [
                (
                    ["Title"],
                    [
                        "# Title\n\n",
                        "intro text\n\n",
                        "## Alpha\n\n",
                        "alpha body about lancedb\n\n",
                        "## Beta\n\n",
                        "beta body",
                    ],
                )
            ],
        ),
        (
            "windows line endings",
            "# H\r\n\r\nOne.\r\nTwo.\r\n",
            WIDE,
            [(["H"], ["One.\r\n", "Two."])],
        ),
    ],
)
def test_chunk(name: str, text: str, settings: ChunkSettings, expected: list) -> None:
    chunks = _split(text, settings)
    assert [(c.headings, _texts(c)) for c in chunks] == expected, name


# --- paragraphs: one chunk each, short ones merged ---------------------------------------

# At chunk size 100 and the default 33%, a paragraph under 33 characters is short.
MID = "This paragraph is long enough that it is no longer short at all."  # 64
MID_TWO = "Another paragraph that is long enough not to count as short here."  # 65
LONG = " ".join(f"Sentence {i} of a paragraph too long for one chunk." for i in range(5))  # 249
LONG_CHUNKS = [  # two whole sentences fit a chunk
    " ".join(f"Sentence {i} of a paragraph too long for one chunk." for i in pair)
    for pair in ((0, 1), (2, 3), (4,))
]


def _paragraphs(merge_below: int = 33) -> ChunkSettings:
    return ChunkSettings(chunk_size=100, chunk_merge_below=merge_below)


@pytest.mark.parametrize(
    ("name", "text", "settings", "expected"),
    [
        (
            "a short paragraph goes into the paragraph below it",
            f"Lead in.\n\n{MID}",
            _paragraphs(),
            [f"Lead in.\n\n{MID}"],
        ),
        (
            "short paragraphs ahead of one too long for a chunk merge with each other",
            f"One here.\n\nTwo here.\n\n{LONG}",
            _paragraphs(),
            ["One here.\n\nTwo here.", *LONG_CHUNKS],
        ),
        (
            "two paragraphs that are not short stay apart, though they would fit",
            f"{MID}\n\n{MID_TWO}",
            _paragraphs(),
            [MID, MID_TWO],
        ),
        (
            "a run of short paragraphs longer than a chunk breaks between paragraphs",
            "\n\n".join(f"Short paragraph {i}." for i in range(8)),
            _paragraphs(),
            [
                "\n\n".join(f"Short paragraph {i}." for i in range(5)),
                "\n\n".join(f"Short paragraph {i}." for i in range(5, 8)),
            ],
        ),
        (
            "a short paragraph with nothing below to go into joins the chunk above",
            f"{MID}\n\nA short one.\n\n{LONG}",
            _paragraphs(),
            [f"{MID}\n\nA short one.", *LONG_CHUNKS],
        ),
        (
            "a short paragraph at the end joins the chunk above",
            f"{MID}\n\nThe end.",
            _paragraphs(),
            [f"{MID}\n\nThe end."],
        ),
        (
            "a short paragraph that fits nowhere is a chunk of its own",
            f"{LONG}\n\nThe end.",
            _paragraphs(),
            [*LONG_CHUNKS, "The end."],
        ),
        (
            "lines with no blank line between are one paragraph, though none is short",
            "Sentence a is a line of about forty characters.\n"
            "Sentence b is another line, also forty.\nthis",
            _paragraphs(),
            [
                "Sentence a is a line of about forty characters.\n"
                "Sentence b is another line, also forty.\nthis"
            ],
        ),
        (
            "a tight list is one paragraph, though none of its items is short",
            "- An item of forty-five characters, give or take.\n"
            "- Another item, also about forty-five of them.",
            _paragraphs(),
            [
                "- An item of forty-five characters, give or take.\n"
                "- Another item, also about forty-five of them."
            ],
        ),
        (
            "a loose list is one whole too: blank lines between items do not part them",
            "- An item of forty-five characters, give or take.\n\n"
            "- Another item, also about forty-five of them.\n\n"
            "A paragraph after the list that is long enough, yes.",
            _paragraphs(),
            [
                "- An item of forty-five characters, give or take.\n\n"
                "- Another item, also about forty-five of them.",
                "A paragraph after the list that is long enough, yes.",
            ],
        ),
        (
            "an item's own paragraphs and nested lists stay with the list",
            "- one\n\n  continued para of item one.\n\n  - nested\n\n- two",
            _paragraphs(merge_below=0),
            ["- one\n\n  continued para of item one.\n\n  - nested\n\n- two"],
        ),
        (
            "a line leading straight into a list is one paragraph with it",
            "A lead line that is long enough not to be short:\n- one item here\n- two item here",
            _paragraphs(),
            ["A lead line that is long enough not to be short:\n- one item here\n- two item here"],
        ),
        (
            "a blank line keeps a short paragraph out of a long one below it",
            f"this sentence.\n\n{LONG}",
            _paragraphs(),
            ["this sentence.", *LONG_CHUNKS],
        ),
        (
            "a list longer than a chunk is cut between items, never inside one",
            f"Intro.\n\n- one\n- two\n- {MID}\n- {MID}\n- three\n\nAfter the list.",
            _paragraphs(),
            ["Intro.", f"- one\n- two\n- {MID}", f"- {MID}\n- three", "After the list."],
        ),
        (
            "short list items merge like short paragraphs",
            f"- one\n- two\n- three\n\n{MID}",
            _paragraphs(),
            [f"- one\n- two\n- three\n\n{MID}"],
        ),
        (
            "a page marker is no text: the paragraphs around it chunk as they would without it",
            f"{MID}\n\n<!-- page 2 -->\n\nLead in.\n\n{MID_TWO}",
            _paragraphs(),
            [MID, f"Lead in.\n\n{MID_TWO}"],
        ),
        (
            "0% never merges",
            f"Lead in.\n\n{MID}",
            _paragraphs(merge_below=0),
            ["Lead in.", MID],
        ),
        (
            "100% merges every paragraph that fits",
            "Medium paragraph number one is here.\n\nMedium paragraph number two is here.",
            _paragraphs(merge_below=100),
            ["Medium paragraph number one is here.\n\nMedium paragraph number two is here."],
        ),
    ],
)
def test_paragraphs_are_chunks_and_short_ones_merge(
    name: str, text: str, settings: ChunkSettings, expected: list[str]
) -> None:
    chunks = _split(text, settings)
    assert [c.text for c in chunks] == expected, name


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        ("one chunk: the text's own edges", "Just this.", [("edge", "edge")]),
        (
            "a heading opens the next chunk",
            f"# A\n\n{MID}\n\n# B\n\n{MID_TWO}",
            [("edge", "heading"), ("heading", "edge")],
        ),
        (
            "a page marker ahead of the heading does not hide it",
            f"# A\n\n{MID}\n\n<!-- page 2 -->\n\n# B\n\n{MID_TWO}",
            [("edge", "heading"), ("heading", "edge")],
        ),
        (
            "a blank line between two paragraphs that did not merge",
            f"{MID}\n\n{MID_TWO}",
            [("edge", "paragraph"), ("paragraph", "edge")],
        ),
        (
            "a full chunk cut between two list items",
            f"- one\n- two\n- {MID}\n- {MID}",
            [("edge", "length_block"), ("length_block", "edge")],
        ),
        (
            "a full chunk cut between two sentences",
            LONG,
            [("edge", "length_sentence"), ("length_sentence", "length_sentence")]
            + [("length_sentence", "edge")],
        ),
        (
            "a sentence longer than a chunk cut inside itself",
            "word " * 30,
            [("edge", "length_oversize"), ("length_oversize", "edge")],
        ),
        (
            "a separator alone at a section's end makes no chunk: the one before ends there",
            f"# A\n\n{MID}\n\n{MID_TWO}\n\n---\n\n# B\n\n{MID}",
            [("edge", "paragraph"), ("paragraph", "heading"), ("heading", "edge")],
        ),
        (
            "a section of a separator alone makes no chunk: its heading rides on",
            f"# A\n\n{MID}\n\n# B\n\n---\n\n# C\n\n{MID_TWO}",
            [("edge", "heading"), ("heading", "edge")],
        ),
    ],
)
def test_each_chunk_says_why_it_starts_and_ends_where_it_does(
    name: str, text: str, expected: list[tuple[str, str]]
) -> None:
    chunks = _split(text, _paragraphs())
    assert [(c.start_reason, c.end_reason) for c in chunks] == expected, name


@pytest.mark.parametrize(
    ("name", "text", "settings", "expected"),
    [
        ("a heading over text is no piece: it is the frame", "# Title\n\nBody.", WIDE, ["text"]),
        ("sentences of a plain paragraph", "One. Two.", WIDE, ["text", "text"]),
        ("sentences of a list item", "- One. Two.\n- Three.", WIDE, ["list", "list", "list"]),
        ("a list item's own paragraphs", "- One.\n\n  More.\n", WIDE, ["list", "list"]),
        ("sentences of a blockquote", "> One. Two.", WIDE, ["quote", "quote"]),
        ("a list inside a blockquote: the innermost", "> - One.\n> - Two.", WIDE, ["list", "list"]),
        (
            "a blockquote inside a list item: the innermost",
            "- Item.\n\n  > Quoted.",
            WIDE,
            ["list", "quote"],
        ),
        ("a table", "| a |\n|---|\n| 1 |", WIDE, ["table"]),
        ("a code block", "```\nx = 1\n```", WIDE, ["code"]),
        ("raw HTML", "<div>Hi.</div>", WIDE, ["html"]),
        ("a rule", "One.\n\n---\n\nTwo.", WIDE, ["text", "rule", "text"]),
        (
            "a page marker is no piece",
            "<!-- page 2 -->\n\n- One.",
            WIDE,
            ["list"],
        ),
        (
            "a cut of an oversize piece keeps its type",
            "```\n" + "x = 1\n" * 20 + "```",
            ChunkSettings(chunk_size=40),
            ["code"] * 4,
        ),
        (
            "the text chunker: every piece is text",
            "# Not. A heading.\n\n- one",
            ChunkSettings(chunker=Chunker.TEXT),
            ["text", "text", "text"],
        ),
    ],
)
def test_every_piece_is_typed_by_the_markdown_it_came_from(
    name: str, text: str, settings: ChunkSettings, expected: list[str]
) -> None:
    chunks = _split(text, settings)
    assert [piece.type for c in chunks for piece in c.pieces] == expected, name


@pytest.mark.parametrize(
    ("name", "text", "end_reason", "expected"),
    [
        ("nothing to pack", "", "edge", []),
        ("a section of headings alone: nothing", "# A\n## B\n", "heading", []),
        (
            "two paragraphs apart: a paragraph, then the section's end",
            f"# H\n\n{MID}\n\n{MID_TWO}",
            "heading",
            [(MID, ["H"], "paragraph"), (MID_TWO, [], "heading")],
        ),
        (
            "a list cut between items, then the text's edge",
            f"- one\n- two\n- {MID}\n- {MID}",
            "edge",
            [(f"- one\n- two\n- {MID}", [], "length_block"), (f"- {MID}", [], "edge")],
        ),
        (
            "a paragraph cut between sentences, the last chunk ends with the group",
            f"{LONG}\n\n{MID}",
            "edge",
            [(LONG_CHUNKS[0], [], "length_sentence"), (LONG_CHUNKS[1], [], "length_sentence")]
            + [(LONG_CHUNKS[2], [], "paragraph"), (MID, [], "edge")],
        ),
        (
            "a sentence longer than a chunk, cut inside itself",
            "word " * 30,
            "heading",
            [(("word " * 20).strip(), [], "length_oversize")]
            + [(("word " * 10).strip(), [], "heading")],
        ),
    ],
)
def test_pack_names_each_cut_where_it_makes_it(
    name: str, text: str, end_reason: segment.CutReason, expected: list[tuple[str, list, str]]
) -> None:
    pieces = segment.fit(text, segment.sentences(text, segment.blocks(text)), 100)
    packed = segment.pack(pieces, 100, 33.0, end_reason)
    got = [
        (
            text[c.pieces[0].start : c.pieces[-1].visible_end] if c.pieces else "",
            [heading.title for heading in c.headings],
            c.end_reason,
        )
        for c in packed
    ]
    assert got == expected, name


@pytest.mark.parametrize(
    ("name", "text", "size", "expected"),
    [
        ("sentences with no blank line: one paragraph", "One. Two.\nThree.", 100, [0, 0, 0]),
        ("a blank line starts the next", "One. Two.\n\nThree.", 100, [0, 0, 1]),
        ("a tight list is one", "- One.\n- Two.\n\nAfter.", 100, [0, 0, 1]),
        ("a loose list is one too", "- One.\n\n- Two.\n\nAfter.", 100, [0, 0, 1]),
        ("two lists a paragraph apart are two", "- One.\n\nText.\n\n- Two.", 100, [0, 1, 2]),
        ("a blockquote's `>` line is no blank line", "> One.\n>\n> Two.\n\nAfter.", 100, [0, 0, 1]),
        ("two blockquotes a blank line apart are two", "> One.\n\n> Two.", 100, [0, 1]),
        ("a line straight into a table is one", "Lead:\n| a |\n|---|\n| 1 |", 100, [0, 0]),
        (
            "`fit` keeps a cut piece in its paragraph, blank lines inside it or not",
            "```\n" + "x = 1\n\n" * 8 + "```\n\nAfter.",
            30,
            [0, 0, 0, 1],
        ),
    ],
)
def test_every_piece_is_numbered_with_its_paragraph(
    name: str, text: str, size: int, expected: list[int]
) -> None:
    pieces = segment.fit(text, segment.sentences(text, segment.blocks(text)), size)
    assert [piece.paragraph for piece in pieces] == expected, name


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        ("ascii: bytes are chars", "# H\n\nplain *words*", ["# H\n", "plain *words*"]),
        (
            "non-ascii: bytes decoded to chars",
            "# 見\n\n見出し 🌍 *text*",
            ["# 見\n", "見出し 🌍 *text*"],
        ),
        ("a page marker is no block", "<!-- page 2 -->\n\ncafé", ["café"]),
    ],
)
def test_blocks_are_in_char_offsets(name: str, text: str, expected: list[str]) -> None:
    found = segment.blocks(text)
    assert [text[block.start : block.end] for block in found] == expected, name
    prose = found[-1]
    words = [text[start:end] for start, end in prose.words]
    assert "".join(words) == expected[-1].replace("*", ""), name


@pytest.mark.parametrize(
    ("name", "text", "settings", "expected"),
    [
        ("only page markers: nothing", "<!-- page 1 -->\n\n<!-- page 2 -->\n", WIDE, []),
        (
            "between two short paragraphs merged into one chunk",
            f"<!-- page 1 -->\n\nLead in.\n\n<!-- page 2 -->\n\n{MID}",
            _paragraphs(),
            [(f"Lead in.\n\n{MID}", 1, 2)],
        ),
        (
            "mid-paragraph: a sentence that runs onto the next page",
            "<!-- page 1 -->\n\nIt starts on one page\n<!-- page 2 -->\nand ends on the next.",
            WIDE,
            [("It starts on one page\nand ends on the next.", 1, 2)],
        ),
        (
            "mid-paragraph, the text chunker: one sentence across the page",
            "<!-- page 1 -->\n\nIt starts on one page\n<!-- page 2 -->\nand ends on the next.",
            ChunkSettings(chunker=Chunker.TEXT),
            [("It starts on one page\nand ends on the next.", 1, 2)],
        ),
        (
            "where one chunk ends and the next starts: in neither",
            f"<!-- page 1 -->\n\n{MID}\n\n<!-- page 2 -->\n\n{MID_TWO}\n\n<!-- page 3 -->",
            _paragraphs(),
            [(MID, 1, 1), (MID_TWO, 2, 2)],
        ),
        (
            "before a heading: the section is on the marker's page",
            "<!-- page 1 -->\n\n# A\n\nOne.\n\n<!-- page 2 -->\n\n# B\n\nTwo.",
            WIDE,
            [("One.", 1, 1), ("Two.", 2, 2)],
        ),
        (
            "ahead of a piece longer than a chunk: no cut is the marker alone",
            "One.\n\n<!-- page 2 -->\n\n" + "word " * 30,
            ChunkSettings(chunk_size=59),
            [("One.", None, None)]
            + [(("word " * 12).strip(), 2, 2)] * 2
            + [(("word " * 6).strip(), 2, 2)],
        ),
    ],
)
def test_page_markers_are_pages_not_text(
    name: str, text: str, settings: ChunkSettings, expected: list[tuple[str, int, int]]
) -> None:
    chunks = _split(text, settings)
    assert [(c.text, c.page_start, c.page_end) for c in chunks] == expected, name
    assert all("<!--" not in c.text for c in chunks), name


@pytest.mark.parametrize(
    ("name", "part", "opened", "expected"),
    [
        ("the first part: nothing open", "Text.\n\n## B\n\nMore.", [], [[], ["B"]]),
        (
            "text ahead of the part's first heading sits under what was open",
            "Text.\n\n## C\n\nMore.",
            [(1, "A"), (2, "B")],
            [["A", "B"], ["A", "C"]],
        ),
        (
            "a heading at the top level closes everything carried",
            "Text.\n\n# Z\n\nMore.",
            [(1, "A"), (2, "B")],
            [["A", "B"], ["Z"]],
        ),
    ],
)
def test_a_later_part_is_chunked_under_the_headings_still_open(
    name: str, part: str, opened: list[tuple[int, str]], expected: list[list[str]]
) -> None:
    chunks = chunk.split(part, WIDE, opened=opened)
    _check(part, WIDE, chunks)
    assert [c.headings for c in chunks] == expected, name


@pytest.mark.parametrize(
    ("name", "parts", "expected"),
    [
        ("no headings anywhere", ["a", "b"], [[], []]),
        ("a chapter carries into the next part", ["# A\n\n## B\n\ntext", "more"], [[], ["A", "B"]]),
        ("a sibling replaces, then carries", ["# A\n## B\n## C\n", "x"], [[], ["A", "C"]]),
        (
            "every part folds onto what the one before carried",
            ["# A\n", "## B\n", "# Z\n### D\n", "x"],
            [[], ["A"], ["A", "B"], ["Z", "D"]],
        ),
    ],
)
def test_open_headings_fold_part_by_part(
    name: str, parts: list[str], expected: list[list[str]]
) -> None:
    """What `plan_embed` hands each part: the headings open where it starts."""
    opened: list[chunk.Opened] = []
    seen: list[list[str]] = []
    for part in parts:
        seen.append([text for _, text in opened])
        opened = chunk.open_headings(part, opened)
    assert seen == expected, name


# --- what a table cannot show: sizes, offsets, pages -----------------------------------


def test_long_text_without_punctuation_or_newlines_falls_back_to_words() -> None:
    """A sentence longer than a chunk is the one place a chunk ends mid-sentence: cut on words."""
    text = "word " * 500  # 2500 chars, no sentence end, no line break
    chunks = _split(text, ChunkSettings(chunk_size=1200))
    assert len(chunks) == 3
    assert all(c.text.startswith("word") and c.text.endswith("word") for c in chunks)


def test_a_long_sentence_among_short_ones_is_cut_alone() -> None:
    long = "And so on " + "and so on " * 29  # one 300-char sentence
    text = f"Short one. {long.strip()}. Short two."
    chunks = _split(text, ChunkSettings(chunk_size=100))
    assert _texts(chunks[0]) == ["Short one."]
    assert _texts(chunks[-1])[-1] == "Short two."
    assert all(len(c.text) <= 100 for c in chunks)


def test_a_table_bigger_than_a_chunk_is_cut_between_rows() -> None:
    rows = "".join(f"| row {i} | value {i}. More. |\n" for i in range(40))
    text = f"| a | b |\n|---|---|\n{rows}"
    chunks = _split(text, ChunkSettings(chunk_size=200))
    assert len(chunks) > 1
    assert all(c.text.startswith("|") and c.text.endswith("|") for c in chunks)


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("a code block longer than a chunk", "```\n" + "x = 1  # set x\n" * 30 + "```\n"),
        ("one URL longer than a chunk, no spaces", "See " + "https://a.b/" + "x" * 150 + " now."),
        ("a sentence exactly the chunk size", "A" * 58 + ". Next."),
        ("a Chinese paragraph longer than a chunk", "資料很重要。" * 40),
    ],
)
def test_oversized_pieces_still_fit(name: str, text: str) -> None:
    """Whatever cannot be cut at a sentence is still cut to size, and nothing is lost."""
    chunks = _split(text, ChunkSettings(chunk_size=59))
    assert chunks, name


def test_a_page_break_inside_a_section_goes_with_what_follows() -> None:
    text = "<!-- page 1 -->\n\n# A\n\nOne here.\n\n<!-- page 2 -->\n\nTwo here."
    chunks = _split(text, ChunkSettings(chunk_size=30))
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (2, 2)]
    assert chunks[1].pieces[0] == Piece(PieceType.TEXT, "Two here.")


@pytest.mark.parametrize(
    ("name", "text", "offset"),
    [
        ("ascii: a byte offset is the char offset", "# H\n\nplain words here\n", 0),
        ("two-byte characters push the bytes past the chars", "# Ü\n\ncafé näher dabei\n", 0),
        ("three- and four-byte characters too", "# 見\n\n見出し 🌍 text after\n", 0),
        ("a part of a batched document starts where the one before it ended", "# H\n\né\n", 97),
    ],
)
def test_byte_offsets_index_the_encoded_markdown(name: str, text: str, offset: int) -> None:
    """What a search seeks to. A char offset is not a file position once a document leaves ASCII,
    so the byte range is carried beside it and has to cut the same text out of the bytes."""
    settings = ChunkSettings(chunk_size=12)
    chunks = chunk.split(text, settings, byte_offset=offset)
    _check(text, settings, chunks, byte_offset=offset)
    assert chunks, name
    assert all(c.byte_start >= c.char_start for c in chunks), f"{name}: bytes never run short"
    assert chunks[0].byte_start == offset + len(text[: chunks[0].char_start].encode()), name


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("the whole document: both ends are its edges", CutReason.EDGE),
        ("a middle part: both ends meet another part", CutReason.PART),
    ],
)
def test_a_part_boundary_is_not_the_documents_edge(name: str, reason: CutReason) -> None:
    """A section may go on across a part boundary, so the cut says which one it is; the cut
    between the two sections inside the part stays a heading."""
    text = "Intro.\n\n# H\n\nBody."
    first, last = chunk.split(text, WIDE, start_reason=reason, end_reason=reason)

    assert (first.start_reason, last.end_reason) == (reason, reason), name
    assert (first.end_reason, last.start_reason) == ("heading", "heading"), name


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        ("a heading first", "# Chapter 4\n\nMore.", True),
        ("a page marker, then a heading", "<!-- page 11 -->\n\n## Chapter 4\n\nMore.", True),
        ("text first, a heading later", "More of it.\n\n# Chapter 4\n\nMore.", False),
        ("no heading at all", "More of it.", False),
        ("nothing", "", False),
    ],
)
def test_a_part_opens_with_a_heading_only_before_any_word(
    name: str, text: str, expected: bool
) -> None:
    assert chunk.opens_with_heading(text) is expected, name


def test_offsets_of_a_later_part() -> None:
    """A part of a batched document reports lines and chars of the whole document."""
    (c,) = chunk.split("# H\n\nOne.", WIDE, line_offset=10, char_offset=500)
    assert (c.line_start, c.line_end, c.char_start, c.char_end) == (13, 13, 505, 509)


def test_a_heading_longer_than_a_chunk_is_cut_in_the_frame_only() -> None:
    """A conversion can turn a whole paragraph into a heading. It stays whole in the path, and
    is cut in the frame to half a chunk, so the text under it keeps the other half."""
    heading = "long heading words " * 10
    text = f"# {heading}\n\nBody."
    (c,) = _split(text, ChunkSettings(chunk_size=59))
    assert c.text == "Body."
    assert c.headings == [heading.strip()]
    assert c.frame == [heading[:27].rstrip()], "29 of 59: the heading and the blank line after it"


def test_a_heading_is_never_a_chunk_alone() -> None:
    """Every chunk under a heading but the first misses it in its text; the models read it from
    the breadcrumb put in front (`framed`) instead."""
    text = "# Costs\n\n## Europe\n\n" + " ".join(f"They rose by {i}%." for i in range(30))
    chunks = _split(text, ChunkSettings(chunk_size=100))
    assert len(chunks) > 2
    assert _texts(chunks[0])[0].startswith("They rose"), "the first chunk starts at its text"
    assert all(c.header == "Costs > Europe" for c in chunks)
    assert chunk.framed(chunks[1].frame, chunks[1].text).startswith("Costs > Europe\n\nThey")
    assert chunk.framed([], "before any heading") == "before any heading"


@pytest.mark.parametrize(
    ("name", "text", "settings", "expected"),
    [
        (
            "a deep heading path leaves the section less room, so it needs more chunks",
            "# Part One\n## Chapter Two\n### Section Three\n\n" + LONG,
            ChunkSettings(chunk_size=150),
            (["Part One", "Chapter Two", "Section Three"], 3),
        ),
        (
            "the same text under no heading fits fewer",
            LONG,
            ChunkSettings(chunk_size=150),
            ([], 2),
        ),
        (
            "a path longer than half a chunk loses its outermost steps first",
            "# The Whole Book Title\n## A Long Part Name\n### Retries\n\nThey back off.",
            ChunkSettings(chunk_size=60),
            (["A Long Part Name", "Retries"], 1),
        ),
        (
            "the text chunker frames nothing, so its text keeps the whole size",
            "# Part One\n\n" + "x" * 80,
            ChunkSettings(chunker=Chunker.TEXT, chunk_size=100),
            ([], 1),
        ),
    ],
)
def test_the_frame_counts_toward_the_chunk_size(
    name: str, text: str, settings: ChunkSettings, expected: tuple[list[str], int]
) -> None:
    """What the models read is the frame and the text: together they fit the chunk size."""
    chunks = _split(text, settings)
    assert (chunks[-1].frame, len(chunks)) == expected, name
    longest = max(len(chunk.framed(c.frame, c.text)) for c in chunks)
    assert longest <= settings.chunk_size, name


# Under a path of 38 characters and its blank line, 110 of 150 are left: two sentences a chunk,
# where the whole 150 takes three.
DEEP = "# Part One\n## Chapter Two\n### Section Three\n\n" + LONG
DEEP_PATH = ["Part One", "Chapter Two", "Section Three"]


@pytest.mark.parametrize(
    ("name", "framed", "expected_frame", "expected_chunks"),
    [
        ("off: no frame, and the text has the whole size", False, [], 2),
        ("on: the frame is the path, and the text has what it leaves", True, DEEP_PATH, 3),
    ],
)
def test_the_frames_step_frames_each_chunk_and_shrinks_its_budget(
    name: str, framed: bool, expected_frame: list[str], expected_chunks: int
) -> None:
    settings = ChunkSettings(chunk_size=150, chunk_frame=framed)
    chunks = _split(DEEP, settings)
    assert [c.frame for c in chunks] == [expected_frame] * expected_chunks, name
    assert all(c.headings == DEEP_PATH for c in chunks), f"{name}: the path is cited either way"
    if not framed:
        assert all(chunk.framed(c.frame, c.text) == c.text for c in chunks), name


@pytest.mark.parametrize(
    ("name", "settings", "expected"),
    [
        (
            "markdown, framed",
            ChunkSettings(),
            ["blocks", "sentences", "sections", "frames", "pack", "locate"],
        ),
        (
            "markdown, unframed",
            ChunkSettings(chunk_frame=False),
            ["blocks", "sentences", "sections", "pack", "locate"],
        ),
        (
            "text: no headings to frame with, whatever the setting",
            ChunkSettings(chunker=Chunker.TEXT),
            ["paragraphs", "sentences", "sections", "pack", "locate"],
        ),
    ],
)
def test_the_pipeline_is_composed_from_the_settings(
    name: str, settings: ChunkSettings, expected: list[str]
) -> None:
    assert chunk.pipeline(settings) == tuple(getattr(chunk, step) for step in expected), name


@pytest.mark.parametrize(
    ("name", "text", "settings", "expected"),
    [
        ("headings alone at the top: no chunk", "# A\n## B\n", WIDE, []),
        (
            "an empty part and chapter still head the chapter after them",
            "# Part II\n\n## Chapter 5\n\n## Chapter 6\n\nText.",
            WIDE,
            [(["Part II", "Chapter 6"], ["Part II", "Chapter 6"])],
        ),
        (
            "headings alone under a chapter: no chunk, and closed by the next chapter",
            "# Book\n\nIntro.\n\n## Part\n### Empty\n\n# Next\n\nMore.",
            WIDE,
            [(["Book"], ["Book"]), (["Next"], ["Next"])],
        ),
        (
            "a trailing chapter of headings alone: no chunk after the last text",
            "# Part I\n\n## Leaders\n\nLeaders take writes.\n\n# Part II\n",
            WIDE,
            [(["Part I", "Leaders"], ["Part I", "Leaders"])],
        ),
    ],
)
def test_headings_alone_make_no_chunk_but_head_the_ones_after(
    name: str, text: str, settings: ChunkSettings, expected: list
) -> None:
    chunks = _split(text, settings)
    assert [(c.headings, c.frame) for c in chunks] == expected, name


def test_frame_is_what_the_models_read_ahead_of_the_text() -> None:
    assert chunk.frame(["Costs", "Europe"]) == "Costs > Europe\n\n"
    assert chunk.frame([]) == ""
    assert chunk.framed(["Costs"], "They rose.") == "Costs\n\nThey rose."


def test_record_carries_pieces_for_the_cache_and_text_and_layout_for_the_index() -> None:
    (c,) = chunk.split("# H\n\nOne. Two.", WIDE)
    values = chunk.record(c, None, None, seq=1)
    assert values["pieces"] == [{"type": "text", "text": "One. "}, {"type": "text", "text": "Two."}]
    assert values["text"] == "One. Two."
    assert values["layout"] == [{"type": "text", "position": 0}, {"type": "text", "position": 5}]
    assert (values["headings"], values["frame"]) == (["H"], ["H"])


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        (
            "a separator between two paragraphs, alone, goes",
            f"{MID}\n\n---\n\n{MID_TWO}",
            [[], []],
        ),
        (
            "a section of a separator alone: its heading opens the next section's path",
            f"# A\n\n{MID}\n\n## B\n\n* * *\n\n## C\n\n{MID_TWO}",
            [["A"], ["A", "C"]],
        ),
        (
            "a page marker and a separator alone say nothing either",
            f"{MID}\n\n<!-- page 2 -->\n\n---\n\n{MID_TWO}",
            [[], []],
        ),
        ("a document of separators alone makes no chunk", "---\n\n***\n\n- - -", []),
    ],
)
def test_a_chunk_without_a_word_is_never_made(
    name: str, text: str, expected: list[list[str]]
) -> None:
    """A separator or a stray symbol packed alone would be found by its heading path alone, and
    say nothing: it makes no chunk, and every chunk made holds a word."""
    chunks = _split(text, _paragraphs())

    assert [c.headings for c in chunks] == expected, name
    assert all(any(ch.isalnum() for ch in c.text) for c in chunks), (
        f"{name}: every chunk says a word"
    )
