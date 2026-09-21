"""The embedding cache: the URN that keys it, the id that addresses it, and the parquet round trip.

Nothing here launches DBOS. The cache is what makes a document reusable across collections, so
the key has to be exact (field order, every field part of it) and the write has to be atomic and
idempotent.
"""

import hashlib
import itertools
from pathlib import Path

import msgspec
import pytest
from conftest import import_row

from haskie import db, document, embed_cache
from haskie.chunk import CHUNK_VERSION, Chunk
from haskie.document import Document
from haskie.embed_cache import NO_MODEL, Params
from haskie.index import Row
from haskie.settings import ChunkSettings, EmbeddingModel

pytestmark = pytest.mark.anyio  # most cases await; the pure ones ignore the marker

DOC = "guide.md"
BODY = "# Title\n\nbody about lancedb\n"
TINY = EmbeddingModel("test/tiny", 4)

BASE = Params(
    doc=DOC,
    model="BAAI/bge-small-en-v1.5",
    chunk_size=1200,
    chunk_overlap=150,
    chunker="markdown",
    chunk_version=1,
    parser="anydoc",
    skip_ocr_pages=True,
)
# Pinned, not recomputed: the URN is the cache key, so a change to its shape must fail a test
# rather than silently retire every entry on disk.
BASE_URN = (
    "document:guide.md;model:BAAI/bge-small-en-v1.5;chunk_size:1200;chunk_overlap:150;"
    "chunker:markdown;chunk_version:1;parser:anydoc;skip_ocr_pages:true"
)
BASE_ID = "5b753e61445f55f1c72657b07433c4f470c359deeb6057b6286fa9d6e1f89b46"


def _row(text: str, vector: list[float] | None = None) -> Row:
    return Row(
        chunk=Chunk(
            heading="Title",
            text=text,
            line_start=1,
            line_end=3,
            char_start=0,
            char_end=len(text),
            parents=["Book", "Part I"],
            page_start=2,
            page_end=3,
        ),
        vector=vector,
    )


def _parts(directory: Path, groups: list[list[Row]]) -> list[Path]:
    """One `NNNNNN.rows.json` per part, in part order: what the embed slices leave behind."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for seq, rows in enumerate(groups):
        path = directory / f"{seq:06d}.rows.json"
        path.write_bytes(msgspec.json.encode(rows))
        paths.append(path)
    return paths


# --- the key ----------------------------------------------------------------------


def test_urn_is_the_fields_in_one_fixed_order() -> None:
    assert embed_cache.urn(BASE) == BASE_URN
    assert embed_cache.urn(BASE) == embed_cache.urn(msgspec.structs.replace(BASE)), "deterministic"


def test_the_key_is_the_full_sha256_of_the_urn() -> None:
    assert embed_cache.key(BASE) == hashlib.sha256(BASE_URN.encode()).hexdigest() == BASE_ID
    assert len(BASE_ID) == 64, "not truncated: a collision would serve another document's vectors"


@pytest.mark.parametrize(
    ("name", "field", "value"),
    [
        ("another document", "doc", "other.md"),
        ("another embedding model", "model", "BAAI/bge-large-en-v1.5"),
        ("another chunk size", "chunk_size", 900),
        ("another chunk overlap", "chunk_overlap", 0),
        ("another chunker", "chunker", "text"),
        ("another chunking version", "chunk_version", 2),
        ("another parser", "parser", "plain"),
        ("ocr pages kept instead of skipped", "skip_ocr_pages", False),
    ],
)
def test_every_field_of_params_changes_the_id(name: str, field: str, value) -> None:
    """Whatever the cached rows depend on is in the key, or a changed setting would be served
    stale rows from the entry it was computed under."""
    other = msgspec.structs.replace(BASE, **{field: value})

    assert embed_cache.urn(other) != BASE_URN, name
    assert embed_cache.key(other) != BASE_ID, name


@pytest.mark.parametrize(
    ("name", "embedding", "expected_model"),
    [
        ("a profile with an embedding model", TINY, "test/tiny"),
        ("a profile without one indexes text only", None, NO_MODEL),
    ],
)
def test_params_reads_the_document_and_the_collections_chunk_settings(
    name: str, embedding: EmbeddingModel | None, expected_model: str
) -> None:
    doc = Document(
        name="book.pdf",
        suffix=".pdf",
        size=10,
        status="imported",
        parser="plain",
        skip_ocr_pages=False,
    )
    chunking = ChunkSettings(chunker="text", chunk_size=400, chunk_overlap=40)

    params = embed_cache.params(doc, chunking, embedding)

    assert params == Params(
        doc="book.pdf",
        model=expected_model,
        chunk_size=400,
        chunk_overlap=40,
        chunker="text",
        chunk_version=CHUNK_VERSION,
        parser="plain",
        skip_ocr_pages=False,
    ), name


def test_paths_are_derived_from_the_document_and_the_id() -> None:
    root = document.root(DOC)
    assert embed_cache.file_path(DOC, BASE_ID) == root / "embeddings" / f"{BASE_ID}.parquet"
    assert embed_cache.scratch_dir(DOC, BASE_ID) == root / "embeddings" / f"{BASE_ID}.tmp"
    assert embed_cache.rows_path(DOC, BASE_ID, 7).name == "000007.rows.json"
    assert embed_cache.rows_path(DOC, BASE_ID, 7).parent == embed_cache.scratch_dir(DOC, BASE_ID)


# --- write, lookup, read ------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "dims", "vector"),
    [
        ("with vectors", 4, [0.25, 0.5, -0.75, 1.0]),
        ("full text only, no vector column", None, None),
    ],
)
async def test_write_lookup_read_round_trip(
    tmp_path: Path, name: str, dims: int | None, vector: list[float] | None
) -> None:
    """Three parts, the middle one empty: every part is a row group, so group `n` is always part
    `n`, and an empty group simply yields no rows."""
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, doc=doc.name)
    rows = [_row("alpha lancedb", vector), _row("beta lancedb", vector)]
    parts = _parts(tmp_path / "scratch", [rows, [], [_row("gamma lancedb", vector)]])

    assert await embed_cache.lookup(params) is None, f"{name}: nothing cached yet"

    cache_id = await embed_cache.write(params, parts, dims)

    assert cache_id == embed_cache.key(params), name
    assert await embed_cache.lookup(params) == cache_id, name
    assert embed_cache.file_path(doc.name, cache_id).is_file(), name
    assert await embed_cache.row_groups(doc.name, cache_id) == 3, f"{name}: one group per part"

    read = [group async for group in embed_cache.read(doc.name, cache_id, 0, 3)]

    assert [part for part, _ in read] == [0, 2], f"{name}: the empty part yields no rows"
    assert [len(group) for _, group in read] == [2, 1], name
    first = read[0][1][0]
    assert first.chunk == rows[0].chunk, f"{name}: every chunk column round trips"
    if vector is None:
        assert first.vector is None, name
    else:
        assert first.vector == pytest.approx(vector), name
    (entry,) = await embed_cache.entries(doc.name)
    assert (entry.id, entry.doc, entry.urn) == (cache_id, doc.name, embed_cache.urn(params))
    assert (entry.rows, entry.chunk_size, entry.chunker) == (3, params.chunk_size, "markdown")
    assert entry.bytes == embed_cache.file_path(doc.name, cache_id).stat().st_size, name
    wire = msgspec.json.decode(msgspec.json.encode(entry))
    assert set(wire) == set(embed_cache.ENTRY_COLUMNS), "the row is the wire shape, `doc` renamed"


async def test_read_of_a_range_returns_only_that_range(tmp_path: Path) -> None:
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, doc=doc.name)
    parts = _parts(tmp_path / "scratch", [[_row(f"part {i}")] for i in range(4)])
    cache_id = await embed_cache.write(params, parts, None)

    read = [part async for part, _ in embed_cache.read(doc.name, cache_id, 1, 3)]

    assert read == [1, 2], "the parts the index group asked for, numbered as they are stored"
    past_the_end = [part async for part, _ in embed_cache.read(doc.name, cache_id, 4, 6)]
    assert past_the_end == [], "a group beyond the last part reads nothing instead of raising"


async def test_write_consumes_the_scratch_directory_of_the_computation(tmp_path: Path) -> None:
    """The scratch rows go last, after the file and the row: a retry before the row was written
    still finds its input."""
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, doc=doc.name)
    cache_id = embed_cache.key(params)
    parts = _parts(embed_cache.scratch_dir(doc.name, cache_id), [[_row("alpha")]])
    assert all(path.exists() for path in parts)

    await embed_cache.write(params, parts, None)

    assert not embed_cache.scratch_dir(doc.name, cache_id).exists(), "the scratch files are gone"
    assert embed_cache.file_path(doc.name, cache_id).is_file(), "the cache file is not"


async def test_a_second_write_of_the_same_params_is_a_no_op_row(tmp_path: Path) -> None:
    """A retried write after a crash, or the loser of two concurrent writers, must not raise."""
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, doc=doc.name)
    parts = _parts(tmp_path / "scratch", [[_row("alpha")]])

    first = await embed_cache.write(params, parts, None)
    (before,) = await embed_cache.entries(doc.name)
    again = await embed_cache.write(params, parts, None)

    assert again == first, "the same params, so the same id"
    entries = await embed_cache.entries(doc.name)
    assert len(entries) == 1, "insert or ignore: one row, not two and not an IntegrityError"
    assert entries[0].created_at == before.created_at, "the first row stands"


async def test_entries_lists_every_cache_of_one_document_newest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = await import_row(DOC, BODY)
    other = await import_row("other.md", BODY)
    # a clock that ticks once per call, so "newest first" is not a race with the wall clock
    monkeypatch.setattr(embed_cache.time, "time", itertools.count(1000.0).__next__)
    parts = _parts(tmp_path / "scratch", [[_row("alpha")]])
    wanted = [
        msgspec.structs.replace(BASE, doc=doc.name, chunk_size=size) for size in (400, 800, 1200)
    ]
    written = [await embed_cache.write(params, parts, None) for params in wanted]
    await embed_cache.write(msgspec.structs.replace(BASE, doc=other.name), parts, None)

    entries = await embed_cache.entries(doc.name)

    assert [entry.id for entry in entries] == list(reversed(written)), "newest first"
    assert [entry.chunk_size for entry in entries] == [1200, 800, 400]
    assert len({entry.id for entry in entries}) == 3, "one entry per distinct chunk size"
    assert await embed_cache.entries("never-imported.md") == [], "a document with no cache"


@pytest.mark.parametrize(
    ("name", "keep_row", "keep_file", "hit"),
    [
        ("neither the row nor the file", False, False, False),
        ("a row whose file never landed", True, False, False),
        ("a file whose row never landed", False, True, False),
        ("both -> a hit", True, True, True),
    ],
)
async def test_lookup_answers_a_hit_only_when_the_row_and_the_file_agree(
    tmp_path: Path, name: str, keep_row: bool, keep_file: bool, hit: bool
) -> None:
    """Either half alone is an interrupted write: a miss to recompute, never an error."""
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, doc=doc.name)
    parts = _parts(tmp_path / "scratch", [[_row("alpha")]])
    cache_id = await embed_cache.write(params, parts, None)
    if not keep_row:
        async with db.connect() as conn:
            await conn.execute("delete from embeddings where id = ?", (cache_id,))
    if not keep_file:
        embed_cache.file_path(doc.name, cache_id).unlink()

    assert await embed_cache.lookup(params) == (cache_id if hit else None), name


async def test_row_groups_of_a_missing_cache_file_raises() -> None:
    doc = await import_row(DOC, BODY)
    with pytest.raises(FileNotFoundError, match=BASE_ID):
        await embed_cache.row_groups(doc.name, BASE_ID)


async def test_a_failed_merge_leaves_no_partial_cache_file(tmp_path: Path) -> None:
    """The parquet file is written through a `.tmp` and one replace, so a reader never sees a
    half-written cache — and a failure leaves nothing to mistake for one."""
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, doc=doc.name)
    missing = tmp_path / "scratch" / "000000.rows.json"  # never written by any embed slice

    with pytest.raises(FileNotFoundError):
        await embed_cache.write(params, [missing], None)

    directory = doc.embeddings_dir
    assert list(directory.glob("*.parquet.tmp")) == [], "the temp file is removed by the failure"
    assert list(directory.glob("*.parquet")) == [], "and no cache file was published"
    assert await embed_cache.lookup(params) is None, "so the next attempt recomputes"
    assert await embed_cache.entries(doc.name) == [], "the row is only written after the file"


async def test_writing_vectors_the_rows_do_not_carry_is_refused(tmp_path: Path) -> None:
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, doc=doc.name)
    parts = _parts(tmp_path / "scratch", [[_row("alpha")]])  # no vector on the row

    with pytest.raises(ValueError, match="carries no vector"):
        await embed_cache.write(params, parts, 4)

    assert await embed_cache.lookup(params) is None


async def test_the_cache_row_goes_when_the_document_does(tmp_path: Path) -> None:
    """`embeddings.document` cascades: deleting the document takes its whole cache with it."""
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, doc=doc.name)
    parts = _parts(tmp_path / "scratch", [[_row("alpha")]])
    await embed_cache.write(params, parts, None)
    assert len(await embed_cache.entries(doc.name)) == 1

    await document.remove_row(doc.name)

    assert await embed_cache.entries(doc.name) == []
