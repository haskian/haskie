"""The embedding cache: the URN that keys it, the id that addresses it, and the parquet round trip.

Nothing here launches DBOS. The cache is what makes a document reusable across collections, so
the key has to be exact (field order, every field part of it) and the write has to be atomic and
idempotent.
"""

import hashlib
import itertools
from pathlib import Path

import lancedb
import msgspec
import numpy as np
import pytest
from conftest import id_of, import_row
from sqlalchemy import delete, select

from haskie import db, home
from haskie.catalogue.catalogue import EmbeddingModel, Matryoshka
from haskie.collection.index import Row
from haskie.document import document
from haskie.document.document import Document, DocumentStatus
from haskie.indexing import embed_cache
from haskie.indexing.chunk import CHUNK_VERSION, Chunk, Piece
from haskie.indexing.embed_cache import NO_MODEL, Params
from haskie.indexing.segment import PieceType
from haskie.outline import store
from haskie.settings import Chunker, ChunkSettings, Parser
from haskie.tables import embeddings

pytestmark = pytest.mark.anyio  # most cases await; the pure ones ignore the marker

DOC = "guide.md"
BODY = "# Title\n\nbody about lancedb\n"
TINY = EmbeddingModel("test/tiny", 4)

DOC_ID = "d41d8cd98f00b204e9800998ecf8427e"  # an MD5, as a document's id is
BASE = Params(
    document_id=DOC_ID,
    model="BAAI/bge-small-en-v1.5",
    chunk_size=1200,
    chunk_merge_below=33,
    chunk_frame=True,
    chunker=Chunker.MARKDOWN,
    chunk_version=1,
    parser=Parser.ANYDOC,
    skip_ocr_pages=True,
)
# Pinned, not recomputed: the URN is the cache key, so a change to its shape must fail a test
# rather than silently retire every entry on disk.
BASE_URN = (
    "document_id:d41d8cd98f00b204e9800998ecf8427e;model:BAAI/bge-small-en-v1.5;chunk_size:1200;"
    "chunk_merge_below:33;chunk_frame:true;chunker:markdown;chunk_version:1;parser:anydoc;"
    "skip_ocr_pages:true"
)
BASE_ID = "16c5d8b180e8c8967c03684ea4f881a0632d4186f11aa20c1fc1cec8687dffcc"


def _row(text: str, vector: list[float] | None = None) -> Row:
    """An embed slice's row: `seq` is left at 0, the way a slice writes it. `_merge` numbers it."""
    return Row(
        chunk=Chunk(
            headings=["Book", "Part I", "Title"],
            frame=["Book", "Part I", "Title"],
            pieces=[Piece(PieceType.TEXT, text)],
            line_start=1,
            line_end=3,
            char_start=0,
            char_end=len(text),
            byte_start=0,
            byte_end=len(text.encode()),
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
        ("another document", "document_id", "1" * 32),
        ("another embedding model", "model", "BAAI/bge-large-en-v1.5"),
        ("another chunk size", "chunk_size", 900),
        ("another merge share", "chunk_merge_below", 50),
        ("no heading path prepended", "chunk_frame", False),
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
    ("name", "change"),
    [
        ("another document prefix", {"document_prefix": "passage: "}),
        ("a Matryoshka cut", {"matryoshka": Matryoshka()}),
        ("another vector size", {"dims": 2}),
    ],
)
def test_a_model_that_embeds_differently_is_another_cache_entry(name: str, change: dict) -> None:
    """The entry id is what a collection reads vectors back by: one model embedding another way
    must not be served the vectors the first way wrote."""
    doc = Document(id="0" * 32, name=DOC, suffix=".md", size=10, status=DocumentStatus.IMPORTED)
    other = msgspec.structs.replace(TINY, **change)

    before = embed_cache.params(doc, ChunkSettings(), TINY)
    after = embed_cache.params(doc, ChunkSettings(), other)

    assert embed_cache.key(after) != embed_cache.key(before), name


@pytest.mark.parametrize(
    ("name", "embedding", "expected_model"),
    [
        ("a profile with an embedding model", TINY, TINY.cache_name),
        ("a profile without one indexes text only", None, NO_MODEL),
    ],
)
def test_params_reads_the_document_and_the_collections_chunk_settings(
    name: str, embedding: EmbeddingModel | None, expected_model: str
) -> None:
    doc = Document(
        id=DOC_ID,
        name="book.pdf",
        suffix=".pdf",
        size=10,
        status=DocumentStatus.IMPORTED,
        parser=Parser.PLAIN,
        skip_ocr_pages=False,
    )
    chunking = ChunkSettings(
        chunker=Chunker.TEXT, chunk_size=400, chunk_merge_below=25, chunk_frame=False
    )

    params = embed_cache.params(doc, chunking, embedding)

    assert params == Params(
        document_id=DOC_ID,
        model=expected_model,
        chunk_size=400,
        chunk_merge_below=25,
        chunk_frame=False,
        chunker=Chunker.TEXT,
        chunk_version=CHUNK_VERSION,
        parser=Parser.PLAIN,
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
    params = msgspec.structs.replace(BASE, document_id=doc.id)
    rows = [_row("alpha lancedb", vector), _row("beta lancedb", vector)]
    parts = _parts(tmp_path / "scratch", [rows, [], [_row("gamma lancedb", vector)]])

    assert await embed_cache.lookup(params) is None, f"{name}: nothing cached yet"

    cache_id = await embed_cache.write(params, parts, dims)

    assert cache_id == embed_cache.key(params), name
    assert await embed_cache.lookup(params) == cache_id, name
    assert embed_cache.file_path(doc.id, cache_id).is_file(), name
    assert await embed_cache.row_groups(doc.id, cache_id) == 3, f"{name}: one group per part"

    read = [group async for group in embed_cache.read(doc.id, cache_id, 0, 3)]

    assert [part for part, _ in read] == [0, 2], f"{name}: the empty part yields no rows"
    assert [len(group) for _, group in read] == [2, 1], name
    first = read[0][1][0]
    assert first.chunk == rows[0].chunk, f"{name}: every chunk column round trips"
    if vector is None:
        assert first.vector is None, name
    else:
        assert first.vector == pytest.approx(vector), name
    (entry,) = await embed_cache.entries(doc.id)
    assert (entry.id, entry.document_id, entry.urn) == (cache_id, doc.id, embed_cache.urn(params))
    assert (entry.rows, entry.chunk_size, entry.chunker) == (3, params.chunk_size, "markdown")
    assert entry.bytes == embed_cache.file_path(doc.id, cache_id).stat().st_size, name
    wire = msgspec.json.decode(msgspec.json.encode(entry))
    assert set(wire) == {column.name for column in embed_cache.ENTRY_COLUMNS}, (
        "the row is the wire shape, `doc` renamed"
    )


async def test_seq_numbers_the_whole_document_across_its_parts(tmp_path: Path) -> None:
    """The parts are chunked in parallel and each one numbers its chunks from zero, so `seq` is
    the merge's job: 1..N over every part in order, with an empty part consuming no number."""
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, document_id=doc.id)
    groups = [
        [_row("alpha lancedb"), _row("beta lancedb")],
        [],
        [_row("gamma lancedb")],
        [_row("delta lancedb"), _row("epsilon lancedb")],
    ]
    assert all(row.seq == 0 for group in groups for row in group), "unnumbered going in"
    parts = _parts(tmp_path / "scratch", groups)

    cache_id = await embed_cache.write(params, parts, None)

    read = [group async for group in embed_cache.read(doc.id, cache_id, 0, 4)]

    assert [part for part, _ in read] == [0, 2, 3], "the empty part yields no rows"
    numbered = [(row.seq, row.chunk.text) for _, group in read for row in group]
    assert numbered == [
        (1, "alpha lancedb"),
        (2, "beta lancedb"),
        (3, "gamma lancedb"),
        (4, "delta lancedb"),
        (5, "epsilon lancedb"),
    ], "1..N in document order, through the parquet round trip"


async def test_read_of_a_range_returns_only_that_range(tmp_path: Path) -> None:
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, document_id=doc.id)
    parts = _parts(tmp_path / "scratch", [[_row(f"part {i}")] for i in range(4)])
    cache_id = await embed_cache.write(params, parts, None)

    read = [part async for part, _ in embed_cache.read(doc.id, cache_id, 1, 3)]

    assert read == [1, 2], "the parts the index group asked for, numbered as they are stored"
    past_the_end = [part async for part, _ in embed_cache.read(doc.id, cache_id, 4, 6)]
    assert past_the_end == [], "a group beyond the last part reads nothing instead of raising"


async def test_write_consumes_the_scratch_directory_of_the_computation(tmp_path: Path) -> None:
    """The scratch rows go last, after the file and the row: a retry before the row was written
    still finds its input."""
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, document_id=doc.id)
    cache_id = embed_cache.key(params)
    parts = _parts(embed_cache.scratch_dir(doc.id, cache_id), [[_row("alpha")]])
    assert all(path.exists() for path in parts)

    await embed_cache.write(params, parts, None)

    assert not embed_cache.scratch_dir(doc.id, cache_id).exists(), "the scratch files are gone"
    assert embed_cache.file_path(doc.id, cache_id).is_file(), "the cache file is not"


async def test_a_second_write_of_the_same_params_is_a_no_op_row(tmp_path: Path) -> None:
    """A retried write after a crash, or the loser of two concurrent writers, must not raise."""
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, document_id=doc.id)
    parts = _parts(tmp_path / "scratch", [[_row("alpha")]])

    first = await embed_cache.write(params, parts, None)
    (before,) = await embed_cache.entries(doc.id)
    again = await embed_cache.write(params, parts, None)

    assert again == first, "the same params, so the same id"
    entries = await embed_cache.entries(doc.id)
    assert len(entries) == 1, "insert or ignore: one row, not two and not an IntegrityError"
    assert entries[0].created_at == before.created_at, "the first row stands"


async def _vector_of(doc: str) -> list[float] | None:
    """The document vector the one cache row of `doc` stores, decoded."""
    async with db.connect() as conn:
        stored = await conn.scalar(
            select(embeddings.c.vector).where(embeddings.c.document_id == doc)
        )
    return None if stored is None else np.frombuffer(stored, dtype=np.float32).tolist()


@pytest.mark.parametrize(
    ("name", "dims", "groups", "expected"),
    [
        (
            "each chunk counts once, however long its vector; parts add up, an empty one adds 0;"
            " the mean is not normalized",
            4,
            [[_row("alpha", [3.0, 0.0, 0.0, 0.0])], [], [_row("beta", [0.0, 4.0, 0.0, 0.0])]],
            [0.5, 0.5, 0.0, 0.0],
        ),
        ("no embedding model: no vector", None, [[_row("alpha")]], None),
        (
            "vectors that cancel out have no direction",
            4,
            [[_row("alpha", [1.0, 0.0, 0.0, 0.0]), _row("beta", [-1.0, 0.0, 0.0, 0.0])]],
            None,
        ),
    ],
)
async def test_write_stores_the_document_vector(
    tmp_path: Path,
    name: str,
    dims: int | None,
    groups: list[list[Row]],
    expected: list[float] | None,
) -> None:
    """The mean of the unit chunk vectors, not normalized: its length is how tightly the chunks
    point one way, which a mean over a collection needs (`corpus_sum`)."""
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, document_id=doc.id)

    await embed_cache.write(params, _parts(tmp_path / "scratch", groups), dims)

    stored = await _vector_of(doc.id)
    if expected is None:
        assert stored is None, name
    else:
        assert stored == pytest.approx(expected), name


async def _embedded(
    tmp_path: Path, name: str, vector: list[float], model: str = TINY.cache_name, size: int = 1200
) -> None:
    """One cache entry of `name`, one chunk whose vector is `vector`."""
    params = msgspec.structs.replace(
        BASE, document_id=await id_of(name), model=model, chunk_size=size
    )
    parts = _parts(tmp_path / f"{name}-{model}-{size}", [[_row(name, vector)]])
    await embed_cache.write(params, parts, len(vector))


async def test_nearest_ranks_imported_documents_by_their_newest_vector(tmp_path: Path) -> None:
    """Closest first; the document itself, one under another model and one not imported are
    left out; a document with two entries is compared by its newest."""
    for name in ("a.md", "close.md", "far.md", "other-model.md", "importing.md"):
        await import_row(name, f"# {name}\n\nbody\n")
        if name != "importing.md":
            await document.set_status(await id_of(name), DocumentStatus.IMPORTED)
    await _embedded(tmp_path, "a.md", [1.0, 0.0, 0.0, 0.0])
    await _embedded(tmp_path, "far.md", [1.0, 1.0, 0.0, 0.0], size=900)  # older: not read
    await _embedded(tmp_path, "far.md", [0.0, 1.0, 0.0, 0.0])
    await _embedded(tmp_path, "close.md", [3.0, 1.0, 0.0, 0.0])
    await _embedded(tmp_path, "other-model.md", [1.0, 0.0, 0.0, 0.0], model="other/model")
    await _embedded(tmp_path, "importing.md", [1.0, 0.0, 0.0, 0.0])

    found = await embed_cache.nearest(await id_of("a.md"), TINY.cache_name, 3)

    assert [(one.document, one.similarity) for one in found] == [
        ("close.md", pytest.approx(0.9487, abs=1e-4)),
        ("far.md", pytest.approx(0.0, abs=1e-4)),
    ]
    assert [
        one.document for one in await embed_cache.nearest(await id_of("a.md"), TINY.cache_name, 1)
    ] == ["close.md"], "cut to the limit"
    assert await embed_cache.nearest(await id_of("importing.md"), TINY.cache_name, 3) == [], (
        "no vector yet"
    )
    assert await embed_cache.nearest(await id_of("a.md"), "other/model", 3) == [], (
        "none under that model"
    )


async def test_entries_lists_every_cache_of_one_document_newest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = await import_row(DOC, BODY)
    other = await import_row("other.md", BODY)
    # a clock that ticks once per call, so "newest first" is not a race with the wall clock
    monkeypatch.setattr(embed_cache.time, "time", itertools.count(1000.0).__next__)
    parts = _parts(tmp_path / "scratch", [[_row("alpha")]])
    wanted = [
        msgspec.structs.replace(BASE, document_id=doc.id, chunk_size=size)
        for size in (400, 800, 1200)
    ]
    written = [await embed_cache.write(params, parts, None) for params in wanted]
    await embed_cache.write(msgspec.structs.replace(BASE, document_id=other.id), parts, None)

    entries = await embed_cache.entries(doc.id)

    assert [entry.id for entry in entries] == list(reversed(written)), "newest first"
    assert [entry.chunk_size for entry in entries] == [1200, 800, 400]
    assert len({entry.id for entry in entries}) == 3, "one entry per distinct chunk size"
    assert await embed_cache.entries("0" * 32) == [], "a document with no cache"


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
    params = msgspec.structs.replace(BASE, document_id=doc.id)
    parts = _parts(tmp_path / "scratch", [[_row("alpha")]])
    cache_id = await embed_cache.write(params, parts, None)
    if not keep_row:
        async with db.connect() as conn:
            await conn.execute(delete(embeddings).where(embeddings.c.id == cache_id))
    if not keep_file:
        embed_cache.file_path(doc.id, cache_id).unlink()

    assert await embed_cache.lookup(params) == (cache_id if hit else None), name


async def test_row_groups_of_a_missing_cache_file_raises() -> None:
    doc = await import_row(DOC, BODY)
    with pytest.raises(FileNotFoundError, match=BASE_ID):
        await embed_cache.row_groups(doc.id, BASE_ID)


async def test_a_failed_merge_leaves_no_partial_cache_file(tmp_path: Path) -> None:
    """The parquet file is written through a `.tmp` and one replace, so a reader never sees a
    half-written cache — and a failure leaves nothing to mistake for one."""
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, document_id=doc.id)
    missing = tmp_path / "scratch" / "000000.rows.json"  # never written by any embed slice

    with pytest.raises(FileNotFoundError):
        await embed_cache.write(params, [missing], None)

    directory = doc.embeddings_dir
    assert list(directory.glob("*.parquet.tmp")) == [], "the temp file is removed by the failure"
    assert list(directory.glob("*.parquet")) == [], "and no cache file was published"
    assert await embed_cache.lookup(params) is None, "so the next attempt recomputes"
    assert await embed_cache.entries(doc.id) == [], "the row is only written after the file"


async def test_writing_vectors_the_rows_do_not_carry_is_refused(tmp_path: Path) -> None:
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, document_id=doc.id)
    parts = _parts(tmp_path / "scratch", [[_row("alpha")]])  # no vector on the row

    with pytest.raises(ValueError, match="carries no vector"):
        await embed_cache.write(params, parts, 4)

    assert await embed_cache.lookup(params) is None


async def test_the_cache_row_goes_when_the_document_does(tmp_path: Path) -> None:
    """`embeddings.document` cascades: deleting the document takes its whole cache with it."""
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, document_id=doc.id)
    parts = _parts(tmp_path / "scratch", [[_row("alpha")]])
    await embed_cache.write(params, parts, None)
    assert len(await embed_cache.entries(doc.id)) == 1

    await document.remove_row(doc.id)

    assert await embed_cache.entries(doc.id) == []


# --- the outline built from an entry --------------------------------------------------


def _headed(text: str, heading: str, vector: list[float] | None = None) -> Row:
    """A chunk under `Book > heading`, at its own offsets."""
    row = _row(text, vector)
    row.chunk.headings = ["Book", heading]
    return row


SAGAS_TEXT = (
    "A saga splits a long transaction into local steps. When a step fails, the saga runs the "
    "compensating step of every step before it. An orchestrator can drive the saga, or each "
    "service can listen for the events of the others."
)
QUORUMS_TEXT = (
    "A quorum write waits for a majority of replicas. A quorum read asks a majority too, so the "
    "two quorums overlap and the read sees the latest write, unless a replica is stale."
)


def _book() -> list[Row]:
    return [
        _headed(SAGAS_TEXT, "Sagas", [1.0, 0.0, 0.0, 0.0]),
        _headed(QUORUMS_TEXT, "Quorums", [0.0, 1.0, 0.0, 0.0]),
    ]


async def _index(model: str = BASE.model):
    conn = await lancedb.connect_async(str(home.OUTLINE_ROOT))
    return await conn.open_table(store.table_name(model))


async def test_build_outline_reads_the_entrys_file(tmp_path: Path) -> None:
    """Its nodes go beside the markdown and into the outline index, with their vectors; the
    keyword candidates are embedded in one call for the whole document."""
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, document_id=doc.id)
    await embed_cache.write(params, _parts(tmp_path / "scratch", [_book()]), 4)
    assert await store.read([doc.id]) == {}, "the write alone builds no outline"
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[1.0, 0.0, 0.0, 0.0] if "saga" in text else [0.0, 1.0, 0.0, 0.0] for text in texts]

    await embed_cache.build_outline(params, embed)

    assert await store.current(doc.id, params.model)
    nodes = (await store.read([doc.id]))[doc.id]
    assert [node.header for node in nodes] == ["", "Book", "Book > Sagas", "Book > Quorums"]
    assert store.path(doc.id).parent == doc.markdown.parent, "beside the markdown"
    sagas = nodes[2]
    assert sagas.keywords["saga"] == 3, "each keyword with how often the section uses it"
    assert not {"quorum", "majority", "replica"} & set(sagas.keywords), "its own words only"
    assert (sagas.line_start, sagas.page_start, sagas.page_end) == (1, 2, 3)
    assert len(calls) == 1, "one embedding call per document"
    rows = await (await _index()).query().where(f"document_id = '{doc.id}'").to_list()
    by_position = {row["position"]: row for row in rows}
    assert sorted(by_position) == [0, 1, 2, 3]
    assert by_position[2]["headings"] == ["Book", "Sagas"]
    assert list(by_position[2]["vector"]) == [1.0, 0.0, 0.0, 0.0], "the node's unit vector"
    assert by_position[2]["keywords"][0] == {"keyword": next(iter(sagas.keywords)), "uses": 3}


async def test_build_outline_of_an_entry_without_vectors(tmp_path: Path) -> None:
    doc = await import_row(DOC, BODY)
    params = msgspec.structs.replace(BASE, document_id=doc.id, model=NO_MODEL)
    rows = [_headed(SAGAS_TEXT, "Sagas"), _headed(QUORUMS_TEXT, "Quorums")]
    await embed_cache.write(params, _parts(tmp_path / "scratch", [rows]), None)

    await embed_cache.build_outline(params, None)

    assert len((await store.read([doc.id]))[doc.id]) == 4
    assert "vector" not in (await (await _index(NO_MODEL)).schema()).names, "no model: no vectors"


async def test_corpus_sum_adds_the_indexed_members_by_their_chunks(tmp_path: Path) -> None:
    """Each indexed member's mean times its chunks: the sum a collection's centre divides."""
    from haskie.collection.collection import Collection, MemberStatus

    notes = await Collection.create("notes")
    for name, vectors in (
        ("a.md", [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        ("b.md", [[0.0, 0.0, 2.0, 0.0]]),
        ("pending.md", [[0.0, 0.0, 0.0, 1.0]]),
    ):
        doc = await import_row(name, BODY)
        await document.set_status(doc.id, DocumentStatus.IMPORTED)
        params = msgspec.structs.replace(BASE, document_id=doc.id, model=TINY.cache_name)
        rows = [_row(name, vector) for vector in vectors]
        await embed_cache.write(params, _parts(tmp_path / name, [rows]), 4)
        await notes.add(doc.id)
        if name != "pending.md":
            await notes.set_member_status(doc.id, MemberStatus.INDEXED)

    found = await embed_cache.corpus_sum("notes", TINY.cache_name)

    assert found is not None
    total, rows = found
    assert rows == 3, "a pending member is not in the table yet"
    assert total == pytest.approx([1.0, 1.0, 1.0, 0.0]), "unit vectors, summed"
    assert await embed_cache.corpus_sum("notes", "other/model") is None

    await notes.set_centre(found, TINY.cache_name)
    await Collection.create("empty")
    centre = await Collection.centre(["notes", "empty"], TINY.cache_name)
    assert centre is not None and centre == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0.0])
    assert await Collection.centre(["notes"], "other/model") is None, "a centre is per model"
    await notes.set_centre(None, TINY.cache_name)
    assert await Collection.centre(["notes"], TINY.cache_name) is None, "cleared"
