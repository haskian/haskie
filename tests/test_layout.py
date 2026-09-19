"""Sharding: where a document and a collection sit under the haskie home.

`layout` is two values now — `shard()` and `PART_DIGITS`. A fresh home is sharded from the first
write, so there is no layout migration left to test.
"""

import hashlib

import pytest

from haskie import document, embed_cache, home, layout
from haskie.collection import Collection
from haskie.document import Document

DOC = "guide.md"


def test_shard_hashes_the_utf8_bytes_of_the_name() -> None:
    """A name is text, a hash is bytes: the encoding is pinned so the shard never depends on the
    platform or the locale."""
    name = "résumé.md"

    assert layout.shard(name) == hashlib.sha1(name.encode("utf-8")).hexdigest()[:2]
    assert len(layout.shard(name)) == 2
    assert layout.shard(name) == layout.shard(name), "the same name always lands in one place"


def test_shard_spreads_names_over_the_whole_byte() -> None:
    """One directory per name would make every listing pay for every document, so the point of
    the shard is the spread: a thousand names have to use most of the 256 directories."""
    shards = {layout.shard(f"doc-{i}.md") for i in range(1000)}

    assert len(shards) > 200, "a thousand names fall into more than 200 of the 256 shards"
    assert all(len(prefix) == 2 and int(prefix, 16) >= 0 for prefix in shards), "two hex digits"


def test_every_document_path_sits_under_the_same_shard() -> None:
    """A document owns one folder: the upload, the markdown, the parts, the preview and the
    embedding cache are all inside it, so one `remove_tree` deletes everything it owns."""
    doc = Document(name=DOC, suffix=".md", size=1, status="imported")
    prefix = layout.shard(DOC)
    root = home.DOCUMENT_ROOT / prefix / DOC

    assert document.root(DOC) == root
    assert doc.root == root
    assert doc.original == root / "original.md"
    assert doc.markdown == root / "original.md.md"
    assert doc.parts_dir == root / "parts"
    assert doc.preview_dir == root / "preview"
    assert doc.embeddings_dir == root / "embeddings"
    assert doc.part_path(7) == doc.parts_dir / "000007.md"
    assert embed_cache.file_path(DOC, "abc").parent == doc.embeddings_dir
    assert {path.parent.parent for path in (doc.original, doc.markdown)} == {root.parent}


def test_part_numbers_are_wide_enough_for_a_long_document() -> None:
    doc = Document(name=DOC, suffix=".md", size=1, status="imported")

    assert layout.PART_DIGITS == 6, "four digits would cap a document at ten thousand parts"
    assert doc.part_path(0).name == "000000.md"
    assert doc.part_path(123456).name == "123456.md"


def test_a_collection_is_sharded_by_its_own_name() -> None:
    collection = Collection("notes")
    root = home.COLLECTION_ROOT / layout.shard("notes") / "notes"

    assert collection.root == root
    assert collection.index_dir == root / "index"


@pytest.mark.anyio
async def test_a_document_lands_in_its_shard_and_is_removed_from_it(tmp_path) -> None:
    source = tmp_path / "incoming" / DOC
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# A\n")

    doc = await document.import_path(str(source))

    assert doc.original.read_text() == "# A\n"
    assert doc.source_path() == doc.original
    assert [p.name for p in home.DOCUMENT_ROOT.iterdir()] == [layout.shard(DOC)]

    await document.remove_files(doc.name)

    assert not doc.root.exists(), "the whole folder goes, not only the upload"
