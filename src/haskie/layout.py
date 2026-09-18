"""Filesystem layout of the library folder, and the one-time move to the sharded one.

Layout 1 put every document of a library straight into three directories: `files/<doc>`,
`markdown/<doc>.md`, `markdown/<doc>.parts/`, `preview/<doc>/`. Ten thousand documents made ten
thousand entries in each of them, which every lookup and every listing pays for. It also numbered
the micro-batch parts of a document with four digits, capping one document at ten thousand parts.

Layout 2 inserts a shard directory keyed by the document name (`shard`), so a library
spreads over 256 directories per base, and widens part numbers to `PART_DIGITS` digits.

`migrate_layout` brings an existing home to layout 2, once, at startup. It is idempotent and safe
to interrupt: an entry already moved is inside a shard directory, where the walk does not look
again, and the version flag is written only after every library is done, so a run that dies half
way is simply finished by the next one.

The walk itself is blocking file IO, so it runs in one worker thread (`_migrate_home`, and the
helpers it calls); only `migrate_layout` and the two touches of the version flag are async.
"""

import hashlib
import os
import re
from pathlib import Path

import anyio.to_thread

from haskie import db, home
from haskie.errors import scrub
from haskie.logs import get_logger

PART_DIGITS = 6  # width of a micro-batch sequence number; four capped a document at 10k parts


def shard(doc: str) -> str:
    """The prefix directory a document lives in: the first byte of the SHA-1 of its name, hex.

    Every per-document file sits under `<base>/<shard>/`, so a library with ten thousand documents
    spreads over 256 directories instead of filling three with ten thousand entries each. The hash
    is over the UTF-8 bytes of the name, so it is stable across platforms and locales, and it is a
    name, not a secret: SHA-1 is used for its spread, not for its strength.
    `layout.migrate_layout` moves a home written by an older build into this shape, once.
    """
    return hashlib.sha1(doc.encode("utf-8")).hexdigest()[:2]


LAYOUT_VERSION = 2
LAYOUT_KEY = "layout_version"

BASES = ("files", "markdown", "preview")  # the per-document directories of one library
MARKDOWN_SUFFIXES = (".parts", ".md")  # decorations `markdown/` adds to a document name
SHARD_DIR = re.compile(r"^[0-9a-f]{2}$")  # what `shard` produces, and nothing else:
# every document name carries a file suffix (`library.save` enforces it), so no document entry
# can be mistaken for a shard directory.
PART_FILE = re.compile(r"^(\d+)(\.rows\.json|\.md)$")

_log = get_logger(__name__)


async def migrate_layout() -> int:
    """Move every library of the home to the sharded layout; returns how many entries moved.

    Call once at startup, after the database migrations and before anything reads a document
    (in particular before DBOS recovers workflows, which resume mid-pipeline).
    """
    if await _layout_version() == str(LAYOUT_VERSION):
        return 0
    moved = await anyio.to_thread.run_sync(_migrate_home)
    await _set_layout_version()  # last: an interrupted run leaves the flag unset and is redone
    _log.info("layout_migrated", version=LAYOUT_VERSION, entries=moved)
    return moved


def _migrate_home() -> int:
    """The whole move, in one worker thread: it is a burst of `iterdir` and `os.replace` calls that
    would otherwise block the event loop. It runs once per home, at boot, before anything else
    reads a document, so nothing observes the tree while it changes."""
    moved = 0
    if home.LIBRARY_ROOT.is_dir():
        for library_dir in sorted(home.LIBRARY_ROOT.iterdir()):
            if library_dir.is_dir():
                moved += _migrate_library(library_dir)
    return moved


def _migrate_library(library_dir: Path) -> int:
    moved = 0
    for base_name in BASES:
        base = library_dir / base_name
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and SHARD_DIR.fullmatch(entry.name):
                continue  # a shard directory: its contents are already where they belong
            moved += _move(base_name, base, entry, library_dir.name)
    return moved


def _document_name(base_name: str, entry_name: str) -> str:
    """The document an entry belongs to. `files/` and `preview/` name it directly; `markdown/`
    appends `.md` to the assembled file and `.parts` to the micro-batch directory."""
    if base_name != "markdown":
        return entry_name
    for suffix in MARKDOWN_SUFFIXES:
        if entry_name.endswith(suffix):
            return entry_name[: -len(suffix)]
    return entry_name


def _move(base_name: str, base: Path, entry: Path, library: str) -> int:
    """Move one flat entry under its shard; returns 1 when it moved, 0 when it was skipped."""
    if entry.is_dir() and entry.name.endswith(".parts"):
        _widen_parts(entry)  # before the move, so an interrupt leaves the whole entry to redo
    target = base / shard(_document_name(base_name, entry.name)) / entry.name
    if target.exists():
        # both layouts hold this document: refuse to choose, and leave the flat copy for the user
        _log.warning("layout_conflict", library=library, path=scrub(str(target)))
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(entry, target)
    return 1


def _widen_parts(parts_dir: Path) -> None:
    """`0000.md` -> `000000.md`, `0000.rows.json` -> `000000.rows.json`. The part number keeps its
    value, so the pipeline reads the same parts in the same order. Idempotent: a name that is
    already wide enough is rewritten to itself."""
    for part in sorted(parts_dir.iterdir()):
        match = PART_FILE.fullmatch(part.name)
        if match is None:
            continue
        widened = part.with_name(f"{int(match[1]):0{PART_DIGITS}d}{match[2]}")
        if widened != part:
            os.replace(part, widened)


async def _layout_version() -> str | None:
    async with db.connect() as conn:
        cursor = await conn.execute("select value from meta where key = ?", (LAYOUT_KEY,))
        row = await cursor.fetchone()
    return row[0] if row else None


async def _set_layout_version() -> None:
    async with db.connect() as conn:
        await conn.execute(
            "insert into meta (key, value) values (?, ?) "
            "on conflict (key) do update set value = excluded.value",
            (LAYOUT_KEY, str(LAYOUT_VERSION)),
        )
