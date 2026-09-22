"""Documents: one row in `documents`, one folder under ~/.haskie/documents/<shard>/<name>/.

A document is first class and belongs to no collection: it is imported once, under a name that
never changes, and any number of collections may then hold it (`collection/collection.py`). What a
document owns lives in its folder — `original.<ext>` (the file as uploaded), `original.<ext>.md`
(the markdown assembled from it once, at import), `parts/` (the per-micro-batch markdown that every
collection re-chunks from), `preview/` (built lazily on first open) and `embeddings/` (the cache
`indexing/embed_cache.py` writes) — so deleting the folder deletes everything but the rows, and the
rows cascade from the document's own.

Two-phase intake: `stage` writes an upload into `staging/` with a `staging` row beside it, and
commits no document — no name is taken and no `documents` row exists yet.
`import_staged` / `import_path` are the import: they fix the name (`safe_name`, suffix kept),
refuse a name already taken, create the row and move the file into its folder. Conversion happens
once, at import, so `parser` and `skip_ocr_pages` are chosen then and stored on the row, not on a
collection. Lifecycle: queued -> converting -> embedding -> imported, ending in error or cancelled
instead; `deleting` while a delete runs, so nothing attaches the document meanwhile.

Every row read, row write and file touch is awaited: the database goes through `db.connect()`
(aiosqlite), the files through `anyio.Path` and `home`, and the one piece of CPU work here — the
preview build — through `cpu.on_cpu`. The pure parts (paths, name cleaning, row decoding) stay
sync.
"""

import re
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Protocol, get_args
from uuid import uuid4

import anyio
import anyio.to_thread
import msgspec

from haskie import cpu, db, home
from haskie.document import convert
from haskie.errors import Conflict, InvalidInput, NotFound, NotReady, PermanentError
from haskie.paging import Page, PageRequest, key_reader, keyset, resolve_sort
from haskie.settings import Parser, load_user_settings

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
UPLOAD_MAX_BYTES = 512 * 1024 * 1024  # also the HTTP request body cap (see app.create_app)
# What `stage` produces, and the only thing `staging_path` accepts: a path is built from it.
STAGING_ID = re.compile(r"^[0-9a-f]{32}\.[a-z0-9]+\Z")

DocStatus = Literal[
    "queued", "converting", "embedding", "imported", "error", "cancelled", "deleting"
]
DOCUMENT_STATUSES: tuple[DocStatus, ...] = get_args(DocStatus)
# in the import pipeline right now: the states a poll waits on
ACTIVE_DOCUMENT_STATUSES: tuple[DocStatus, ...] = ("queued", "converting", "embedding")

# Public sort name -> SQL expression. The whitelist is the only source of column identifiers a
# listing can order by, so a request can never name a column (see paging.resolve_sort).
DOCUMENT_SORTS = {"name": "name", "size": "size", "status": "status", "updated_at": "updated_at"}


class Document(msgspec.Struct):
    """One row of `documents`, plus the paths that follow from its name and suffix.

    The path properties are sync and IO-free: everything a document owns is derived from the
    two immutable columns, so a pipeline step that holds the row holds every path it needs."""

    name: str
    suffix: str  # of the original file, lower-case, with the dot: ".pdf"
    size: int
    status: DocStatus
    error: str | None = None
    preview: convert.Preview | None = None
    parser: Parser = "anydoc"
    skip_ocr_pages: bool = True
    created_at: float = 0.0  # unix seconds
    updated_at: float = 0.0
    description: str = ""  # what the document is, in the importer's words

    @property
    def root(self) -> Path:
        return root(self.name)

    @property
    def original(self) -> Path:
        return self.root / f"original{self.suffix}"

    @property
    def markdown(self) -> Path:
        return self.root / f"original{self.suffix}.md"

    @property
    def preview_dir(self) -> Path:
        return self.root / "preview"

    @property
    def parts_dir(self) -> Path:
        """The per-micro-batch markdown. Durable, not scratch: a collection that chunks the
        document differently re-chunks from the same part boundaries (see `pipeline`)."""
        return self.root / "parts"

    @property
    def embeddings_dir(self) -> Path:
        return embeddings_dir(self.name)

    def part_path(self, seq: int) -> Path:
        return self.parts_dir / f"{home.part_name(seq)}.md"

    def source_path(self) -> Path:
        """`original`, for the readers that need the file to exist."""
        if not self.original.is_file():
            raise NotFound(f"document file missing: {self.name}")
        return self.original

    @staticmethod
    def relative(path: Path) -> str:
        """Path as stored in an index: relative to the haskie home, so the folder is portable.
        Static: it reads the home, not the document."""
        return path.relative_to(home.HOME).as_posix()


# The struct's field order is the column order, so the SELECT, the row unpack and the keyset key
# reader cannot drift apart.
DOCUMENT_COLUMNS: tuple[str, ...] = tuple(f.encode_name for f in msgspec.structs.fields(Document))
DOCUMENT_SELECT = ", ".join(DOCUMENT_COLUMNS)


class Listed(Document):
    """A document as the API lists it: the row, plus how many collections hold it. A read model
    for the gallery, not a column: `DOCUMENT_COLUMNS` reads the base class alone."""

    collections: int = 0


async def listed(docs: list[Document]) -> list[Listed]:
    """The same documents with their collection counts, from one query."""
    names_ = [doc.name for doc in docs]
    counts: dict[str, int] = {}
    if names_:
        async with db.connect() as conn:
            cursor = await conn.execute(
                "select document, count(*) from collection_documents "
                f"where document in ({db.placeholders(len(names_))}) group by document",
                names_,
            )
            rows: list[Any] = list(await cursor.fetchall())
        counts = dict(rows)
    return [
        Listed(**msgspec.structs.asdict(doc), collections=counts.get(doc.name, 0)) for doc in docs
    ]


class Staged(msgspec.Struct):
    """An upload waiting in `staging/`: bytes on disk and one row in `staging`, no document yet."""

    staging_id: str
    filename: str
    size: int


def root(name: str) -> Path:
    return home.DOCUMENT_ROOT / home.shard(name) / name


def embeddings_dir(name: str) -> Path:
    """The document's embedding cache folder (see `embed_cache`), for callers that hold the name
    rather than the row."""
    return root(name) / "embeddings"


def safe_name(name: str) -> str:
    cleaned = SAFE_NAME.sub("-", name).strip("-.")
    if not cleaned:
        raise InvalidInput(f"invalid name: {name!r}")
    return cleaned


def stored_name(filename: str, rename_to: str | None = None) -> str:
    """The name a file is imported under, or the reason it is refused.

    `rename_to` names the document; the suffix decides how it is parsed, so a rename that drops
    or changes it keeps the original's: the importer is naming the document, not choosing a
    parser.
    """
    chosen = Path(rename_to).name if rename_to else Path(filename).name
    suffix = Path(filename).suffix.lower()
    if rename_to and Path(chosen).suffix.lower() != suffix:
        chosen += suffix
    name = safe_name(chosen)
    if Path(name).suffix.lower() not in convert.SUPPORTED_SUFFIXES:
        raise PermanentError(f"unsupported file type: {name}")
    return name


def _document(row: tuple) -> Document:
    return db.row_to(Document, DOCUMENT_COLUMNS, row, preview=convert.Preview)


# --- staging ------------------------------------------------------------------


def staging_path(staging_id: str) -> Path:
    """The file behind a staging id. The id is a trust boundary: only what `stage` produces
    becomes a path, so a caller cannot name a file outside `staging/`."""
    if not STAGING_ID.match(staging_id):
        raise InvalidInput(f"invalid staging id: {staging_id!r}")
    return home.STAGING_ROOT / staging_id


async def stage(filename: str, content: bytes) -> Staged:
    """Keep an upload until it is imported. The name is only checked here, not committed: the
    import may still rename the document, and a staged upload nobody imports is swept away.

    Bytes first, row second: a crash in between leaves a file the sweep removes as an orphan,
    where a row without bytes would be an upload the import cannot read.
    """
    stored_name(filename)  # refuse what could never be imported, before writing anything
    if len(content) > UPLOAD_MAX_BYTES:
        raise InvalidInput(f"file larger than {UPLOAD_MAX_BYTES} bytes: {len(content)}")
    staging_id = f"{uuid4().hex}{Path(filename).suffix.lower()}"
    name = Path(filename).name
    await anyio.Path(home.STAGING_ROOT).mkdir(parents=True, exist_ok=True, mode=home.DIR_MODE)
    await home.atomic_write(home.STAGING_ROOT / staging_id, content)
    async with db.connect() as conn:
        await conn.execute(
            "insert into staging (staging_id, filename, size, created_at) values (?, ?, ?, ?)",
            (staging_id, name, len(content), time.time()),
        )
    return Staged(staging_id=staging_id, filename=name, size=len(content))


async def sweep_staging(max_age_seconds: float) -> int:
    """Delete staged uploads older than `max_age_seconds`, rows and bytes; returns how many went.

    A file with no row goes too, once it is that old: it is an interrupted `stage`, and nobody can
    import it because the import reads the row. An upload that was never imported is not a
    document, so nothing but its bytes is lost.
    """
    cutoff = time.time() - max_age_seconds
    async with db.connect() as conn:
        cursor = await conn.execute("select staging_id, created_at from staging")
        rows = await cursor.fetchall()
    staged = {staging_id for staging_id, _ in rows}
    expired = [staging_id for staging_id, created_at in rows if created_at < cutoff]
    for staging_id in expired:
        await anyio.Path(staging_path(staging_id)).unlink(missing_ok=True)
    if expired:
        async with db.connect() as conn:
            await conn.execute(
                f"delete from staging where staging_id in ({db.placeholders(len(expired))})",
                expired,
            )
    deleted = len(expired)
    directory = anyio.Path(home.STAGING_ROOT)
    if not await directory.is_dir():
        return deleted
    async for file in directory.iterdir():
        orphan = STAGING_ID.match(file.name) and file.name not in staged
        if orphan and (await file.stat()).st_mtime < cutoff:
            await file.unlink(missing_ok=True)
            deleted += 1
    return deleted


# --- import -------------------------------------------------------------------


class ImportOptions(msgspec.Struct):
    """How an import stores the file, whichever source it came from."""

    name: str | None = None  # store it under this name instead of the file's own
    description: str = ""
    parser: Parser | None = None
    skip_ocr_pages: bool | None = None


async def _create(name: str, size: int, options: ImportOptions) -> Document:
    """The row, before the file: a name already taken is refused with nothing on disk to undo.
    `parser` / `skip_ocr_pages` default to the user settings at the moment of import."""
    user = await load_user_settings()
    parser = options.parser or user.conversion.parser
    skip = (
        user.conversion.skip_ocr_pages if options.skip_ocr_pages is None else options.skip_ocr_pages
    )
    now = time.time()
    async with db.connect() as conn:
        cursor = await conn.execute(
            "insert into documents (name, suffix, size, status, parser, skip_ocr_pages, "
            "created_at, updated_at, description) values (?, ?, ?, 'queued', ?, ?, ?, ?, ?) "
            "on conflict (name) do nothing",
            (name, Path(name).suffix.lower(), size, parser, skip, now, now, options.description),
        )
        created = cursor.rowcount == 1  # read on the open connection, before it is closed
    if not created:
        raise Conflict(f"document already exists: {name}")
    return await get(name)


async def _place(document: Document, move: bool, source: Path) -> Document:
    """Put the file where the row says it is; the row goes if that fails, so a failed import
    leaves neither a phantom row nor a name that cannot be used again."""
    try:
        await anyio.Path(document.root).mkdir(parents=True, exist_ok=True)
        if move:
            await anyio.to_thread.run_sync(shutil.move, source, document.original)
        else:
            await anyio.to_thread.run_sync(shutil.copyfile, source, document.original)
    except BaseException:
        await remove_files(document.name)
        await remove_row(document.name)
        raise
    return document


async def import_staged(staging_id: str, options: ImportOptions | None = None) -> Document:
    """Turn a staged upload into a document: the final name, the row, the file in its folder.

    The staging row carries the name the user uploaded, because the staging id holds only the
    suffix. The staged file is moved, not copied, and its row goes once the file is placed, so an
    import consumes the staging entry; a failed import leaves it to be retried or swept.
    """
    options = options or ImportOptions()
    source = staging_path(staging_id)  # a trust boundary: the id is validated into a path here
    async with db.connect() as conn:
        cursor = await conn.execute(
            "select filename from staging where staging_id = ?", (staging_id,)
        )
        row = await cursor.fetchone()
    if row is None or not await anyio.Path(source).is_file():
        raise NotFound(f"staged upload not found: {staging_id}")
    final = stored_name(row[0], options.name)
    size = (await anyio.Path(source).stat()).st_size
    document = await _create(final, size, options)
    placed = await _place(document, True, source)
    async with db.connect() as conn:
        await conn.execute("delete from staging where staging_id = ?", (staging_id,))
    return placed


async def import_path(path: str, options: ImportOptions | None = None) -> Document:
    """Import by absolute path: a trust boundary, so the path is checked before it is read.

    The file is copied rather than read into memory: an import may be as large as the upload
    cap allows, and `shutil.copyfile` streams it (in a worker thread, so nothing blocks).
    """
    options = options or ImportOptions()
    source = Path(path).expanduser()
    if not source.is_absolute():
        raise InvalidInput(f"path must be absolute: {home.scrub(str(source))}")
    if not await anyio.Path(source).is_file():
        raise InvalidInput(f"file not found: {home.scrub(str(source))}")
    final = stored_name(source.name, options.name)
    size = (await anyio.Path(source).stat()).st_size
    if size > UPLOAD_MAX_BYTES:
        raise InvalidInput(f"file larger than {UPLOAD_MAX_BYTES} bytes: {size}")
    document = await _create(final, size, options)
    return await _place(document, False, source)


# --- rows ---------------------------------------------------------------------


async def page(request: PageRequest, status: DocStatus | None = None) -> Page[Document]:
    """One page of every document, optionally of one status. `total` counts the filtered rows,
    so it is what the page is a page of."""
    sort, expression = resolve_sort(request.sort, DOCUMENT_SORTS, "name")
    walk = keyset(sort, expression, request)
    filters, params = [], []
    if status is not None:
        filters.append("status = ?")
        params.append(status)
    boundary, boundary_params = walk.where()
    clauses = [*filters, *([boundary] if boundary else [])]
    where = f"where {' and '.join(clauses)} " if clauses else ""
    filtered = f"where {' and '.join(filters)}" if filters else ""
    async with db.connect() as conn:
        cursor = await conn.execute(
            f"select {DOCUMENT_SELECT} from documents {where}"
            f"{walk.order_by()} limit {walk.limit()}",
            [*params, *boundary_params],
        )
        rows: list[Any] = list(await cursor.fetchall())
        cursor = await conn.execute(f"select count(*) from documents {filtered}", params)
        (total,) = await cursor.fetchone() or (0,)  # a count always returns its one row
    return walk.page(
        rows,
        build=_document,
        key=key_reader(sort, expression, list(DOCUMENT_COLUMNS)),
        total=total,
    )


async def get(name: str) -> Document:
    async with db.connect() as conn:
        cursor = await conn.execute(
            f"select {DOCUMENT_SELECT} from documents where name = ?", (name,)
        )
        row: Any = await cursor.fetchone()
    if row is None:
        raise NotFound(f"document not found: {name}")
    return _document(row)


async def set_status(name: str, status: DocStatus, error: str | None = None) -> None:
    """A lifecycle step is a change to the document, so it stamps `updated_at`: that is the
    column the "recently touched" listing sorts on. Building the preview is not (see
    `ensure_preview`), it only fills in what the row always described."""
    async with db.connect() as conn:
        await conn.execute(
            "update documents set status = ?, error = ?, updated_at = ? where name = ?",
            (status, error, time.time(), name),
        )


async def describe(name: str, description: str) -> Document:
    """Replace the document's description; empty clears it. Returns the row as it now stands.

    One statement: `returning` gives back the updated row, so the write and the read a caller
    needs cannot see two different versions of it."""
    async with db.connect() as conn:
        cursor = await conn.execute(
            f"update documents set description = ? where name = ? returning {DOCUMENT_SELECT}",
            (description, name),
        )
        row: Any = await cursor.fetchone()
    if row is None:
        raise NotFound(f"document not found: {name}")
    return _document(row)


async def describe_of(docs: set[str]) -> dict[str, str]:
    """The descriptions of several documents in one query, keyed by name. A document with none
    is absent from the result. Batched because the caller is a search shortlist."""
    if not docs:
        return {}
    wanted = sorted(docs)
    async with db.connect() as conn:
        cursor = await conn.execute(
            "select name, description from documents "
            f"where name in ({db.placeholders(len(wanted))}) and description != ''",
            wanted,
        )
        rows = await cursor.fetchall()
    return {name: description for name, description in rows}


class Described(Protocol):
    """A search row carrying the description of the document it points at."""

    doc: str
    description: str


def fill_descriptions(rows: Iterable[Described], described: dict[str, str]) -> None:
    """Put each row's description on it, empty for a document that has none. A description
    belongs to the document rather than to the row, so every search fills it the same way, from
    one `describe_of` over its whole shortlist."""
    for row in rows:
        row.description = described.get(row.doc, "")


async def collections_of(name: str) -> list[str]:
    """Every collection holding the document, in name order."""
    async with db.connect() as conn:
        cursor = await conn.execute(
            "select collection from collection_documents where document = ? order by collection",
            (name,),
        )
        rows = await cursor.fetchall()
    return [collection for (collection,) in rows]


async def memberships(docs: set[str], collections: list[str]) -> dict[str, list[str]]:
    """Which of `collections` hold each of `docs`, in name order; a document none of them hold is
    absent. One query for a whole search result, the way `describe_of` is one for its
    descriptions: a search that folds hits to documents needs every membership at once."""
    if not docs or not collections:
        return {}
    wanted = list(docs)
    names = list(set(collections))
    async with db.connect() as conn:
        cursor = await conn.execute(
            "select document, collection from collection_documents "
            f"where document in ({db.placeholders(len(wanted))}) "
            f"and collection in ({db.placeholders(len(names))}) "
            "order by document, collection",
            [*wanted, *names],
        )
        rows = await cursor.fetchall()
    held: dict[str, list[str]] = {}
    for name, collection in rows:
        held.setdefault(name, []).append(collection)
    return held


# --- preview ------------------------------------------------------------------

# One lock per document being built, so two readers of the same document build the preview once.
# An `anyio.Lock` rather than a threading one: a preview is only ever built on Litestar's event
# loop (the handler that opens a document), so one set of loop primitives covers every builder, and
# a waiting reader yields its loop instead of blocking it. Made on first use, because no lock may
# exist before there is a loop to await it on.
_preview_locks: dict[str, anyio.Lock] = {}


# A preview build parses a whole document, so a burst of opens would otherwise start one parse per
# request. The semaphore admits `pipeline.preview_workers` of them; the rest wait, and a reader
# that waited this long is told to retry instead of holding its request open forever.
PREVIEW_WAIT_SECONDS = 60
# 2 is the default of `PipelineSettings.preview_workers`. An anyio semaphore, unlike the CPU
# budget's: every preview waits on Litestar's event loop, as `_preview_locks` describes.
_preview_slots = cpu.ResizableSemaphore(anyio.Semaphore, 2)


def configure_preview_slots(workers: int) -> None:
    """Resize the pool of preview builders (from `workflows.apply_settings`)."""
    _preview_slots.resize(workers)


async def ensure_preview(name: str) -> tuple[Document, convert.Preview]:
    """Build the side-by-side preview (first pages only for PDF) once, on first open.

    Returns the preview beside the row so a caller never has to re-check `Document.preview` for
    a `None` this has just ruled out.

    Two locks, always in this order: the document's own (build this document once), then a
    slot in the process-wide pool (build at most `preview_workers` documents at a time).
    The parse itself is CPU work, so it runs in a worker thread under the CPU budget.
    """
    info = await get(name)
    if info.preview is not None:
        return info, info.preview
    # setdefault, with no await in between, so two readers of one document take the same lock
    lock = _preview_locks.setdefault(name, anyio.Lock())
    async with lock:
        try:
            info = await get(name)  # another reader may have built it while we waited
            if info.preview is not None:
                return info, info.preview
            slots = _preview_slots.current  # the object to release, even if the pool is resized
            try:
                with anyio.fail_after(PREVIEW_WAIT_SECONDS):
                    await slots.acquire()
            except TimeoutError:
                raise NotReady("preview queue is full; retry") from None
            try:
                preview = await cpu.on_cpu(
                    convert.build_preview,
                    info.source_path(),
                    info.preview_dir,
                    info.parser,
                    info.skip_ocr_pages,
                )
                async with db.connect() as conn:
                    await conn.execute(
                        "update documents set preview = ? where name = ?",
                        (db.dumps(preview), name),
                    )
            finally:
                slots.release()
            return await get(name), preview
        finally:
            # Still holding the lock, so a queued reader keeps it: dropping it here would send
            # that reader and a newcomer into the same failing build at once. With nobody waiting
            # the build is committed (or failed) and the next reader needs no lock at all.
            if lock.statistics().tasks_waiting == 0:
                _preview_locks.pop(name, None)


# --- removal ------------------------------------------------------------------
# Two steps so a workflow can run each one durably; the row goes last, so a crash leaves a
# document that can be removed again instead of orphaned files. The index rows in every
# collection that held the document are removed by that collection first (see `workflows`).


async def remove_files(name: str) -> None:
    await home.remove_tree(root(name))


async def remove_row(name: str) -> None:
    """Cascades to `collection_documents` and `embeddings`; `pragma foreign_keys = on` is set on
    every connection."""
    async with db.connect() as conn:
        await conn.execute("delete from documents where name = ?", (name,))
