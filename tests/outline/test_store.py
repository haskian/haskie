"""Where an outline is kept: the file beside the markdown and the one index across collections.

Each case writes through `store.save` into the test's own home, and reads back the file and the
LanceDB table as the search and the API would.
"""

from datetime import timedelta

import lancedb
import msgspec
import numpy as np
import pytest

from haskie import home
from haskie.document import document
from haskie.outline import store
from haskie.outline.build import Node

pytestmark = pytest.mark.anyio

MODEL = "test/tiny:4"
WHOLE = Node(
    headings=[],
    line_start=1,
    line_end=9,
    char_start=0,
    char_end=400,
    byte_start=0,
    byte_end=400,
    page_start=1,
    page_end=3,
    keywords={"saga": 4, "compensating step": 2},
)
SAGAS = msgspec.structs.replace(
    WHOLE, headings=["Sagas"], line_start=3, char_start=20, byte_start=20, page_start=2
)


@pytest.fixture(autouse=True)
def _folders() -> None:
    """Each document's folder, which its import made before any outline is saved into it."""
    for doc in ("book.md", "notes.md", "a.md", "b.md", "c.md"):
        document.root(doc).mkdir(parents=True)


def _outline(*nodes: Node, model: str = MODEL) -> store.Outline:
    return store.Outline(model=model, nodes=list(nodes))


def _vectors(count: int) -> np.ndarray:
    return np.eye(4)[:count]


async def _rows(doc: str, model: str = MODEL) -> list[dict]:
    conn = await lancedb.connect_async(str(home.OUTLINE_ROOT))
    table = await conn.open_table(store.table_name(model))
    rows = await table.query().where(f"document_id = '{doc}'").to_list()
    return sorted(rows, key=lambda row: row["position"])


@pytest.mark.parametrize(
    ("name", "saved", "expected"),
    [
        ("no outline yet", None, False),
        ("built under another model", "other/model:8", False),
        ("built under this model", MODEL, True),
    ],
)
async def test_current(name: str, saved: str | None, expected: bool) -> None:
    if saved is not None:
        await store.save("book.md", _outline(WHOLE, model=saved), None)

    assert await store.current("book.md", MODEL) is expected, name


async def test_save_puts_the_nodes_in_the_file_and_the_index() -> None:
    await store.save("book.md", _outline(WHOLE, SAGAS), _vectors(2))

    assert await store.read(["book.md", "ghost.md"]) == {"book.md": [WHOLE, SAGAS]}, (
        "a document without an outline is absent"
    )
    rows = await _rows("book.md")
    assert [(row["position"], row["headings"]) for row in rows] == [(0, []), (1, ["Sagas"])]
    assert rows[1]["keywords"] == [
        {"keyword": "saga", "uses": 4},
        {"keyword": "compensating step", "uses": 2},
    ], "best first, each with its uses"
    assert (rows[1]["char_start"], rows[1]["page_start"]) == (20, 2)
    assert list(rows[1]["vector"]) == [0.0, 1.0, 0.0, 0.0]


async def test_a_new_outline_replaces_the_documents_nodes_only() -> None:
    await store.save("book.md", _outline(WHOLE, SAGAS), _vectors(2))
    await store.save("notes.md", _outline(WHOLE, SAGAS), _vectors(2))

    await store.save("book.md", _outline(WHOLE), _vectors(1))

    assert [row["position"] for row in await _rows("book.md")] == [0], "the node it lost is gone"
    assert len(await _rows("notes.md")) == 2, "another document's nodes stay"


async def test_an_outline_without_nodes_empties_the_documents_rows() -> None:
    await store.save("book.md", _outline(WHOLE, SAGAS), _vectors(2))

    await store.save("book.md", _outline(), np.empty((0, 4)))

    assert await _rows("book.md") == []
    assert await store.read(["book.md"]) == {"book.md": []}


async def test_each_model_has_a_table_of_its_own() -> None:
    """A model change drops nothing: a document not yet embedded under the new model keeps its
    rows, and its file still names the old model, so both agree when the model changes back."""
    other = "other/model:8"
    await store.save("book.md", _outline(WHOLE), _vectors(1))
    await store.save("notes.md", _outline(WHOLE), _vectors(1))

    await store.save("book.md", _outline(WHOLE, model=other), np.eye(8)[:1])

    assert len(await _rows("notes.md")) == 1 and await store.current("notes.md", MODEL)
    assert len(list((await _rows("book.md", other))[0]["vector"])) == 8
    assert len(await _rows("book.md")) == 1, "the old model's rows wait for a change back"
    assert store.table_name(other) != store.table_name(MODEL)


async def test_forget_drops_the_file_and_the_rows_under_every_model() -> None:
    await store.forget("book.md")  # nothing to drop: no table, no file
    await store.save("book.md", _outline(WHOLE), None)
    await store.save("book.md", _outline(WHOLE, model="other/model:8"), None)
    await store.save("notes.md", _outline(WHOLE), None)

    await store.forget("book.md")

    assert not store.path("book.md").exists()
    assert await _rows("book.md") == [] and await _rows("book.md", "other/model:8") == []
    assert len(await _rows("notes.md")) == 1


async def test_compact_merges_the_fragments_each_write_leaves() -> None:
    assert await store.compact(timedelta(0)) == 0, "no table yet"
    for doc in ("a.md", "b.md", "c.md"):
        await store.save(doc, _outline(WHOLE), None)

    assert await store.compact(timedelta(0)) > 0
    assert len(await _rows("b.md")) == 1
