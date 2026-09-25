"""What DBOS records of a workflow survives a change to the structs inside it.

Every case records the input an embed slice is really given, then swaps one struct class for a
changed one under the name the record points at, the way a new release changes it, and reads the
record back.
"""

from collections.abc import Callable
from typing import Any

import msgspec
import pytest
from dbos._serialization import DefaultSerializer

from haskie.catalogue.catalogue import DuplicateCosine, EmbeddingModel
from haskie.document import document
from haskie.document.document import Document, DocumentStatus
from haskie.indexing import pipeline, workflows
from haskie.indexing.pipeline import Batch
from haskie.indexing.serializer import SERIALIZER
from haskie.indexing.workflows import Context, Stage
from haskie.settings import ChunkSettings, Parser, PipelineSettings

DOCUMENT = Document(
    name="patterns.pdf",
    suffix=".pdf",
    size=48_213,
    status=DocumentStatus.EMBEDDING,
    parser=Parser.ANYDOC,
    skip_ocr_pages=True,
)
CONTEXT = Context(
    document=DOCUMENT,
    chunking=ChunkSettings(chunk_size=900),
    # bge-small as the catalogue seeds it, with every nested struct a record can hold
    embedding=EmbeddingModel(
        "BAAI/bge-small-en-v1.5", 384, duplicate=DuplicateCosine(chunk=0.92, passage=0.95)
    ),
    pipeline=PipelineSettings(cpu_budget=4, batch_pages=2),
    collection=None,
    cache_id="3f1c9a",
)
BATCHES = [Batch(seq=0, start=0, end=2), Batch(seq=1, start=2, end=4, line_offset=87)]
INPUT = {"args": (Stage.EMBED, BATCHES, CONTEXT), "kwargs": {}}

Field = tuple[str, Any] | tuple[str, Any, Any]


def _changed(cls: type[msgspec.Struct], fields: list[Field], frozen: bool = False) -> type:
    """`cls` as a later release might define it: the same name and module, other fields."""
    return msgspec.defstruct(cls.__name__, fields, module=cls.__module__, frozen=frozen)


# Context's own fields; defaults from `embedding` on, so a case can insert one before them
CONTEXT_FIELDS: list[Field] = [
    ("document", Document),
    ("chunking", ChunkSettings),
    ("embedding", object, None),
    ("pipeline", object, None),
    ("collection", str | None, None),
    ("cache_id", str, ""),
]


def _context(args: tuple) -> dict:
    return msgspec.structs.asdict(args[2])


def _batches(args: tuple) -> list[tuple]:
    return [(type(b).__name__, b.seq, b.pages, b.start, b.end) for b in args[1]]


def _document(args: tuple) -> tuple:
    found = args[2].document
    return type(found).__struct_config__.frozen, found.name, found.size, hasattr(found, "error")


@pytest.mark.parametrize(
    ("name", "module", "cls", "observe", "expected"),
    [
        (
            "a field removed since is dropped",
            workflows,
            _changed(Context, CONTEXT_FIELDS[:-1]),
            _context,
            {k: v for k, v in msgspec.structs.asdict(CONTEXT).items() if k != "cache_id"},
        ),
        (
            "a field added since takes its default, and every later field keeps its value",
            workflows,
            _changed(Context, [*CONTEXT_FIELDS[:2], ("slot", int, 7), *CONTEXT_FIELDS[2:]]),
            _context,
            {**msgspec.structs.asdict(CONTEXT), "slot": 7},
        ),
        (
            "fields reordered keep their values",
            workflows,
            _changed(
                Context, [CONTEXT_FIELDS[1], CONTEXT_FIELDS[0], *reversed(CONTEXT_FIELDS[2:])]
            ),
            _context,
            msgspec.structs.asdict(CONTEXT),
        ),
        (
            "every struct in a list is rebuilt, one field removed and one added",
            pipeline,
            _changed(Batch, [("seq", int), ("start", int), ("end", int), ("pages", int, 1)]),
            _batches,
            [("Batch", 0, 1, 0, 2), ("Batch", 1, 1, 2, 4)],
        ),
        (
            "a struct nested in another is rebuilt as its new class says: frozen, less a field",
            document,
            _changed(
                Document,
                [(field, Any, None) for field in Document.__struct_fields__ if field != "error"],
                frozen=True,
            ),
            _document,
            (True, "patterns.pdf", 48_213, False),
        ),
    ],
)
def test_a_record_reads_back_after_a_struct_in_it_changed(
    name: str, module: Any, cls: type, observe: Callable[[tuple], Any], expected: Any, monkeypatch
) -> None:
    recorded = SERIALIZER.serialize(INPUT)
    monkeypatch.setattr(module, cls.__name__, cls)

    loaded = SERIALIZER.deserialize(recorded)

    assert loaded["args"][0] == Stage.EMBED and loaded["kwargs"] == {}, name
    assert observe(loaded["args"]) == expected, name


def test_a_record_reads_back_unchanged_when_nothing_changed() -> None:
    assert SERIALIZER.deserialize(SERIALIZER.serialize(INPUT)) == INPUT
    assert SERIALIZER.deserialize(SERIALIZER.serialize([3, None, "x"])) == [3, None, "x"]


def test_a_field_added_without_a_default_fails_the_load(monkeypatch) -> None:
    """Nothing to fill it with: guessing a value would run the workflow on made-up settings."""
    recorded = SERIALIZER.serialize(INPUT)
    monkeypatch.setattr(workflows, "Context", _changed(Context, [("new", int), *CONTEXT_FIELDS]))

    with pytest.raises(TypeError, match="Missing required argument 'new'"):
        SERIALIZER.deserialize(recorded)


def test_dbos_own_pickle_breaks_on_the_same_changes(monkeypatch) -> None:
    """Why this serializer exists: DBOS's pickle keeps a struct by position. A field removed fails
    the load, and a field added moves every later value into the field before it."""
    recorded = DefaultSerializer().serialize(INPUT)

    monkeypatch.setattr(workflows, "Context", _changed(Context, CONTEXT_FIELDS[:-1]))
    with pytest.raises(TypeError, match="Extra positional arguments"):
        DefaultSerializer().deserialize(recorded)

    inserted = [*CONTEXT_FIELDS[:2], ("slot", Any, 7), *CONTEXT_FIELDS[2:]]
    monkeypatch.setattr(workflows, "Context", _changed(Context, inserted))
    shifted = DefaultSerializer().deserialize(recorded)["args"][2]
    assert shifted.slot == CONTEXT.embedding, "the model landed in the new field"
    assert shifted.collection == "3f1c9a", "the cache id landed in the collection"


def test_its_records_are_tagged_apart_from_dbos_own() -> None:
    """DBOS reads a `py_pickle` record with its own pickle, whatever serializer is configured, so
    the two formats need two names."""
    assert SERIALIZER.name() == "haskie_pickle" != DefaultSerializer().name()
