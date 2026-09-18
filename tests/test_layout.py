"""Sharded document paths, and the one-time migration of a home written in the flat layout."""

import hashlib
import os
from pathlib import Path

import pytest

from haskie import home, layout
from haskie.library import Library, shard

pytestmark = pytest.mark.anyio  # every test here awaits, except the one over `shard` itself

DOC = "a.md"
BODY = "# A\n"


def _events(caplog) -> list[str]:
    """Our loggers pass structlog's event dict as the record message."""
    return [r.msg["event"] for r in caplog.records if isinstance(r.msg, dict)]


def _flat_library(name: str = "old") -> Path:
    """The layout-1 folder an older build wrote, built by hand: nothing produces it any more.

    One document with everything it can own: the upload, the assembled markdown, a micro-batch
    parts directory numbered with four digits, and a preview directory.
    """
    root = home.LIBRARY_ROOT / name
    (root / "files").mkdir(parents=True)
    (root / "files" / DOC).write_text(BODY)
    parts = root / "markdown" / f"{DOC}.parts"
    parts.mkdir(parents=True)
    (root / "markdown" / f"{DOC}.md").write_text(BODY)
    (parts / "0000.md").write_text(BODY)
    (parts / "0000.rows.json").write_text("[]")
    preview = root / "preview" / DOC
    preview.mkdir(parents=True)
    (preview / "source").write_text(BODY)
    (preview / "preview.md").write_text(BODY)
    return root


def _tree(root: Path) -> list[str]:
    """Every path under `root`, home-relative and sorted: the snapshot a rerun must not change."""
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def _migrated_tree(prefix: str) -> list[str]:
    """Exactly what `_flat_library` must look like once it is sharded: names it, and by being a
    whole-tree comparison it also rules out a leftover flat copy."""
    return sorted(
        [
            "files",
            f"files/{prefix}",
            f"files/{prefix}/{DOC}",
            "markdown",
            f"markdown/{prefix}",
            f"markdown/{prefix}/{DOC}.md",
            f"markdown/{prefix}/{DOC}.parts",
            f"markdown/{prefix}/{DOC}.parts/000000.md",
            f"markdown/{prefix}/{DOC}.parts/000000.rows.json",
            "preview",
            f"preview/{prefix}",
            f"preview/{prefix}/{DOC}",
            f"preview/{prefix}/{DOC}/preview.md",
            f"preview/{prefix}/{DOC}/source",
        ]
    )


# --- path construction -------------------------------------------------------------


async def test_every_document_path_sits_under_the_same_shard() -> None:
    lib = await Library.create("shards")
    doc = "guide.md"
    prefix = hashlib.sha1(doc.encode("utf-8")).hexdigest()[:2]

    assert shard(doc) == prefix
    assert lib.file_path(doc) == lib.files / prefix / doc
    assert lib.markdown_path(doc) == lib.markdown / prefix / f"{doc}.md"
    assert lib.parts_dir(doc) == lib.markdown / prefix / f"{doc}.parts"
    assert lib.preview_dir(doc) == lib.previews / prefix / doc
    assert lib.part_path(doc, 7).name == "000007.md"
    assert lib.rows_path(doc, 7).name == "000007.rows.json"
    assert lib.part_path(doc, 7).parent == lib.parts_dir(doc)


def test_shard_hashes_the_utf8_bytes_of_the_name() -> None:
    """A name is text, a hash is bytes: the encoding is pinned so the shard never depends on the
    platform or the locale."""
    name = "résumé.md"

    assert shard(name) == hashlib.sha1(name.encode("utf-8")).hexdigest()[:2]
    assert len(shard(name)) == 2


async def test_an_upload_lands_in_its_shard_and_is_removed_from_it() -> None:
    lib = await Library.create("round-trip")

    doc = await lib.save("guide.md", BODY.encode())

    assert lib.file_path(doc.name).read_text() == BODY
    assert lib.source_path(doc.name) == lib.file_path(doc.name)
    assert [p.name for p in lib.files.iterdir()] == [shard(doc.name)]
    await lib.remove_files(doc.name)
    assert not lib.file_path(doc.name).exists()


# --- migration ---------------------------------------------------------------------


async def test_a_fresh_home_only_records_the_layout_version() -> None:
    assert await layout.migrate_layout() == 0, "nothing to move"
    assert await layout._layout_version() == str(layout.LAYOUT_VERSION)
    assert await layout.migrate_layout() == 0


async def test_a_flat_home_is_migrated_once() -> None:
    root = _flat_library()
    prefix = shard(DOC)

    moved = await layout.migrate_layout()

    assert moved == 4, "the upload, the markdown, the parts directory and the preview"
    assert _tree(root) == _migrated_tree(prefix)
    assert (root / "markdown" / prefix / f"{DOC}.parts" / "000000.rows.json").read_text() == "[]"
    lib = Library("old")
    assert lib.file_path(DOC).read_text() == BODY, "the new path finds the moved file"
    assert lib.part_path(DOC, 0).read_text() == BODY, "the part kept its number"

    snapshot = _tree(root)
    assert await layout.migrate_layout() == 0, "the recorded version stops a second walk"
    assert _tree(root) == snapshot


@pytest.mark.parametrize(
    ("name", "replacements"),
    [
        ("after the first entry moved", 1),
        ("while widening the part numbers", 3),
        ("after widening, before the parts directory moved", 4),
    ],
)
async def test_an_interrupted_migration_is_finished_by_the_next_run(
    monkeypatch: pytest.MonkeyPatch, name: str, replacements: int
) -> None:
    root = _flat_library()
    prefix = shard(DOC)
    real_replace = os.replace
    done = 0

    def interrupted(src, dst) -> None:
        nonlocal done
        if done >= replacements:
            raise OSError("interrupted")
        done += 1
        real_replace(src, dst)

    monkeypatch.setattr(layout.os, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        await layout.migrate_layout()
    assert await layout._layout_version() is None, f"the version is not recorded {name}"
    monkeypatch.setattr(layout.os, "replace", real_replace)

    assert await layout.migrate_layout() > 0, "the run that was interrupted is still owed"
    assert _tree(root) == _migrated_tree(prefix), f"no duplicate and nothing lost {name}"


async def test_an_entry_that_already_exists_under_its_shard_is_skipped(caplog) -> None:
    """Both layouts hold the document. Choosing for the user could lose the newer copy, so the
    move is refused and logged, and the flat entry is left where it is."""
    root = _flat_library()
    prefix = shard(DOC)
    (root / "files" / prefix).mkdir(parents=True)
    (root / "files" / prefix / DOC).write_text("# sharded\n")

    with caplog.at_level("WARNING"):
        moved = await layout.migrate_layout()

    assert moved == 3, "the three entries with no conflict"
    assert _events(caplog) == ["layout_conflict"]
    assert (root / "files" / prefix / DOC).read_text() == "# sharded\n"
    assert (root / "files" / DOC).read_text() == BODY, "the flat copy is left for the user"
    assert str(home.HOME) not in caplog.text, "an absolute path identifies the user"


async def test_an_index_row_from_the_flat_layout_resolves_to_the_sharded_path() -> None:
    """A stored path is never rewritten: a hit recomputes both paths from the document name, so
    ten thousand rows of an old library cost nothing to keep."""
    import lancedb

    lib = await Library.create("legacy")
    doc = await lib.save("g.md", b"# Hi\n\nhello world\n")
    lib.markdown_path(doc.name).parent.mkdir(parents=True, exist_ok=True)
    lib.markdown_path(doc.name).write_text("# Hi\n\nhello world\n")
    lib.index_dir.mkdir(parents=True)
    table = lancedb.connect(str(lib.index_dir)).create_table(
        "chunks",
        data=[
            {
                "doc": "g.md",
                "source_path": "library/legacy/files/g.md",  # flat: written before the shards
                "markdown_path": "library/legacy/markdown/g.md.md",
                "part": 0,
                "chunk_id": 0,
                "line_start": 1,
                "line_end": 3,
                "char_start": 0,
                "char_end": 18,
                "page_start": None,
                "page_end": None,
                "parents": "",
                "heading": "Hi",
                "text": "hello world",
            }
        ],
    )
    table.create_fts_index("text", replace=True)

    (hit,) = await lib.search("hello")

    assert hit.source_path == lib.relative(lib.file_path("g.md")) != "library/legacy/files/g.md"
    assert hit.markdown_path == lib.relative(lib.markdown_path("g.md"))
    assert (Path(hit.home) / hit.source_path).read_bytes().startswith(b"# Hi")
    assert (Path(hit.home) / hit.markdown_path).read_text() == "# Hi\n\nhello world\n"
