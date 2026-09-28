"""Documents: one row in `documents`, one folder under ~/.haskie/documents/<shard>/<name>/.

A document is first class and belongs to no collection: it is imported once, under a name that
never changes, and any number of collections may then hold it (`collection/collection.py`). What a
document owns lives in its folder — `original.<ext>` (the file as uploaded), `original.<ext>.md`
(the markdown assembled from it once, at import), `parts/` (one markdown file per part, which every
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
preview build — through `cpu.off_interpreter` for a PDF and `cpu.on_cpu` otherwise. The pure
parts (paths, name cleaning, row decoding) stay sync.
"""

import hashlib
import re
import shutil
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import anyio
import anyio.to_thread
import msgspec
from sqlalchemy import Row, delete, func, literal, select, update
from sqlalchemy.dialects.sqlite import insert

from haskie import cpu, db, home
from haskie.document import convert
from haskie.errors import Conflict, InvalidInput, NotFound, NotReady, PermanentError
from haskie.paging import Page, PageRequest, count_of, keyset, resolve_sort
from haskie.settings import Parser, load_user_settings
from haskie.tables import collection_documents, documents, staging

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
UPLOAD_MAX_BYTES = 512 * 1024 * 1024  # also the HTTP request body cap (see app.create_app)
# What `stage` produces, and the only thing `staging_path` accepts: a path is built from it.
STAGING_ID = re.compile(r"^[0-9a-f]{32}\.[a-z0-9]+\Z")


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    CONVERTING = "converting"
    EMBEDDING = "embedding"
    IMPORTED = "imported"
    ERROR = "error"
    CANCELLED = "cancelled"
    DELETING = "deleting"


DOCUMENT_STATUSES: tuple[DocumentStatus, ...] = tuple(DocumentStatus)
# in the import pipeline right now: the states a poll waits on
ACTIVE_DOCUMENT_STATUSES: tuple[DocumentStatus, ...] = (
    DocumentStatus.QUEUED,
    DocumentStatus.CONVERTING,
    DocumentStatus.EMBEDDING,
)

# Public sort name -> column. The whitelist is the only source of columns a listing can order by,
# so a request can never name one (see paging.resolve_sort).
DOCUMENT_SORTS = {
    "name": documents.c.name,
    "size": documents.c.size,
    "status": documents.c.status,
    "updated_at": documents.c.updated_at,
}


class Document(msgspec.Struct):
    """One row of `documents`, plus the paths that follow from its name and suffix.

    The path properties are sync and IO-free: everything a document owns is derived from the
    two immutable columns, so a pipeline step that holds the row holds every path it needs."""

    name: str
    suffix: str  # of the original file, lower-case, with the dot: ".pdf"
    size: int
    status: DocumentStatus
    error: str | None = None
    preview: convert.Preview | None = None
    parser: Parser = Parser.ANYDOC
    skip_ocr_pages: bool = True
    created_at: float = 0.0  # unix seconds
    updated_at: float = 0.0
    description: str = ""  # what the document is, in the importer's words
    md5: str = ""  # of the original file's bytes: the same hash is the same file imported again

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
        """One markdown file per part. Durable, not scratch: a collection that chunks the
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


DOCUMENT_COLUMNS = db.columns_of(documents, Document)


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
            rows = await conn.execute(
                select(collection_documents.c.document, func.count())
                .where(collection_documents.c.document.in_(names_))
                .group_by(collection_documents.c.document)
            )
            counts = dict(rows.tuples().all())
    return [
        Listed(**msgspec.structs.asdict(doc), collections=counts.get(doc.name, 0)) for doc in docs
    ]


class Staged(msgspec.Struct):
    """An upload waiting in `staging/`: bytes on disk and one row in `staging`, no document yet."""

    staging_id: str
    filename: str
    size: int
    # documents already holding these exact bytes: importing it again only adds a copy
    duplicates: list[str]


def _md5_of_file(path: Path) -> str:
    """What `md5` holds: a fingerprint for spotting the same file twice, not a security check.
    Streamed, since an import may be as large as the upload cap."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()


async def identical(md5: str, but: str | None = None) -> list[str]:
    """The documents whose original has these bytes, in name order, `but` left out, and one being
    deleted too: it is on its way out, so the file is no repeat of it."""
    same = select(documents.c.name).where(
        documents.c.md5 == md5, documents.c.status != DocumentStatus.DELETING
    )
    if but is not None:
        same = same.where(documents.c.name != but)
    async with db.connect() as conn:
        return list(await conn.scalars(same.order_by(documents.c.name)))


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


def from_row(row: Row[Any]) -> Document:
    """A row that selected `DOCUMENT_COLUMNS`, and maybe more, as a `Document`."""
    return db.row_to(Document, row, preview=convert.Preview)


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
    # refuse what could never be imported, before writing anything
    importable = stored_name(filename)
    if len(content) > UPLOAD_MAX_BYTES:
        raise InvalidInput(f"file larger than {UPLOAD_MAX_BYTES} bytes: {len(content)}")
    # The cleaned name's suffix, not the raw one: `report.md.` or `notes.md~` would give an id
    # `STAGING_ID` refuses, which neither the import nor the sweep could turn back into a path.
    staging_id = f"{uuid4().hex}{Path(importable).suffix.lower()}"
    name = Path(filename).name
    await anyio.Path(home.STAGING_ROOT).mkdir(parents=True, exist_ok=True, mode=home.DIR_MODE)
    await home.atomic_write(home.STAGING_ROOT / staging_id, content)
    # hashed once, here, and carried to the import in the row
    md5 = await anyio.to_thread.run_sync(_md5_of_file, home.STAGING_ROOT / staging_id)
    async with db.connect() as conn:
        await conn.execute(
            insert(staging).values(
                staging_id=staging_id,
                filename=name,
                size=len(content),
                md5=md5,
                created_at=time.time(),
            )
        )
    duplicates = await identical(md5)
    return Staged(staging_id=staging_id, filename=name, size=len(content), duplicates=duplicates)


async def sweep_staging(max_age_seconds: float) -> int:
    """Delete staged uploads older than `max_age_seconds`, rows and bytes; returns how many went.

    A file with no row goes too, once it is that old: it is an interrupted `stage`, and nobody can
    import it because the import reads the row. An upload that was never imported is not a
    document, so nothing but its bytes is lost.

    The bytes go by the files in `staging/`, never by turning a row's id into a path: every file
    there is ours, and one a live row names is kept. So a row an older build staged under an id
    `staging_path` now refuses (`notes.md~`) is cleared like any other.
    """
    cutoff = time.time() - max_age_seconds
    async with db.connect() as conn:
        rows = (await conn.execute(select(staging.c.staging_id, staging.c.created_at))).all()
    live = {staging_id for staging_id, created_at in rows if created_at >= cutoff}
    expired = {staging_id for staging_id, created_at in rows if created_at < cutoff}
    if expired:
        async with db.connect() as conn:
            await conn.execute(delete(staging).where(staging.c.staging_id.in_(expired)))
    deleted = len(expired)
    directory = anyio.Path(home.STAGING_ROOT)
    if not await directory.is_dir():
        return deleted
    async for file in directory.iterdir():
        if file.name in live:
            continue
        if file.name in expired:
            await file.unlink(missing_ok=True)
        elif (await file.stat()).st_mtime < cutoff:  # an orphan, once it is that old
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


async def _create(name: str, size: int, md5: str, options: ImportOptions) -> Document:
    """The row, before the file: a name already taken is refused with nothing on disk to undo.
    `parser` / `skip_ocr_pages` default to the user settings at the moment of import.

    Taken ignoring case: a document's folder is named after it, and on a case-insensitive disk
    (macOS by default) `Notes.md` and `notes.md` are one folder, so the second import would
    overwrite the first's files. Names are ASCII (`SAFE_NAME`), which `NOCASE` compares exactly,
    and the check is in the insert itself, so two imports at once cannot both pass it."""
    user = await load_user_settings()
    now = time.time()
    row = {
        "name": name,
        "suffix": Path(name).suffix.lower(),
        "size": size,
        "status": DocumentStatus.QUEUED,
        "parser": options.parser or user.conversion.parser,
        "skip_ocr_pages": (
            user.conversion.skip_ocr_pages
            if options.skip_ocr_pages is None
            else options.skip_ocr_pages
        ),
        "created_at": now,
        "updated_at": now,
        "description": options.description,
        "md5": md5,
    }
    taken = select(documents.c.name).where(documents.c.name.collate("NOCASE") == name)
    async with db.connect() as conn:
        result = await conn.execute(
            insert(documents).from_select(
                list(row),
                select(
                    *(literal(value, documents.c[key].type) for key, value in row.items())
                ).where(~taken.exists()),
            )
        )
        created = result.rowcount == 1  # read on the open connection, before it is closed
        existing = None if created else await conn.scalar(taken)
    if not created:
        raise Conflict(f"document already exists: {existing or name}")
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
        row = (
            await conn.execute(
                select(staging.c.filename, staging.c.md5).where(staging.c.staging_id == staging_id)
            )
        ).first()
    if row is None or not await anyio.Path(source).is_file():
        raise NotFound(f"staged upload not found: {staging_id}")
    final = stored_name(row.filename, options.name)
    size = (await anyio.Path(source).stat()).st_size
    document = await _create(final, size, row.md5, options)
    placed = await _place(document, True, source)
    async with db.connect() as conn:
        await conn.execute(delete(staging).where(staging.c.staging_id == staging_id))
    return placed


async def import_path(path: str, options: ImportOptions | None = None) -> Document:
    """Import by absolute path: a trust boundary, so the path is checked before it is read.

    The file is copied rather than read into memory: an import may be as large as the upload
    cap allows, and `shutil.copyfile` streams it (in a worker thread, so nothing blocks).
    """
    options = options or ImportOptions()
    source = Path(path).expanduser()
    # Every refusal names the file alone: the audit trail copies the error, and it never holds
    # the folder an import came from (see `api.documents.import_document`).
    if not source.is_absolute():
        raise InvalidInput(f"path must be absolute: {source.name}")
    if not await anyio.Path(source).is_file():
        raise InvalidInput(f"file not found: {source.name}")
    final = stored_name(source.name, options.name)
    with _reading(source):
        size = (await anyio.Path(source).stat()).st_size
        if size > UPLOAD_MAX_BYTES:
            raise InvalidInput(f"file larger than {UPLOAD_MAX_BYTES} bytes: {size}")
        md5 = await anyio.to_thread.run_sync(_md5_of_file, source)
    document = await _create(final, size, md5, options)
    with _reading(source):
        return await _place(document, False, source)


@contextmanager
def _reading(source: Path) -> Iterator[None]:
    """Refuse a source the process cannot read, naming the file but not its folder.

    A file can pass `is_file` and still fail to open: its mode, or macOS privacy protection on
    a folder like Documents. The `OSError` then carries the full path. A failure about any
    other file, such as the document's own folder filling the disk, passes through unchanged."""
    try:
        yield
    except OSError as exc:
        if exc.filename is None or Path(exc.filename) != source:
            raise
        raise InvalidInput(f"cannot read file: {source.name}: {exc.strerror}") from None


# --- rows ---------------------------------------------------------------------


async def page(request: PageRequest, status: DocumentStatus | None = None) -> Page[Document]:
    """One page of every document, optionally of one status. `total` counts the filtered rows,
    so it is what the page is a page of."""
    sort, column = resolve_sort(request.sort, DOCUMENT_SORTS, "name")
    walk = keyset(sort, column, request, documents.c.name)
    filters = [documents.c.status == status] if status is not None else []
    listing = select(*DOCUMENT_COLUMNS).where(*filters)
    async with db.connect() as conn:
        rows = (await conn.execute(walk.apply(listing))).all()
        total = await conn.scalar(count_of(listing))
    return walk.page(rows, build=from_row, total=total)


async def get(name: str) -> Document:
    async with db.connect() as conn:
        row = (
            await conn.execute(select(*DOCUMENT_COLUMNS).where(documents.c.name == name))
        ).first()
    if row is None:
        raise NotFound(f"document not found: {name}")
    return from_row(row)


async def set_status(name: str, status: DocumentStatus, error: str | None = None) -> None:
    """A lifecycle step is a change to the document, so it stamps `updated_at`: that is the
    column the "recently touched" listing sorts on. Building the preview is not (see
    `ensure_preview`), it only fills in what the row always described."""
    async with db.connect() as conn:
        await conn.execute(
            update(documents)
            .where(documents.c.name == name)
            .values(status=status, error=error, updated_at=time.time())
        )


async def describe(name: str, description: str) -> Document:
    """Replace the document's description; empty clears it. Returns the row as it now stands.

    One statement: `returning` gives back the updated row, so the write and the read a caller
    needs cannot see two different versions of it."""
    async with db.connect() as conn:
        row = (
            await conn.execute(
                update(documents)
                .where(documents.c.name == name)
                .values(description=description)
                .returning(*DOCUMENT_COLUMNS)
            )
        ).first()
    if row is None:
        raise NotFound(f"document not found: {name}")
    return from_row(row)


async def descriptions_of(docs: set[str]) -> dict[str, str]:
    """The descriptions of several documents in one query, keyed by name. A document with none
    is absent from the result. Batched because the caller is a search shortlist."""
    if not docs:
        return {}
    wanted = sorted(docs)
    async with db.connect() as conn:
        rows = await conn.execute(
            select(documents.c.name, documents.c.description).where(
                documents.c.name.in_(wanted), documents.c.description != ""
            )
        )
        return dict(rows.tuples().all())


class Described(Protocol):
    """A search row carrying the description of the document it points at."""

    document: str
    description: str


def fill_descriptions(rows: Iterable[Described], described: dict[str, str]) -> None:
    """Put each row's description on it, empty for a document that has none. A description
    belongs to the document rather than to the row, so every search fills it the same way, from
    one `descriptions_of` over its whole shortlist."""
    for row in rows:
        row.description = described.get(row.document, "")


async def collections_of(name: str) -> list[str]:
    """Every collection holding the document, in name order."""
    async with db.connect() as conn:
        held = await conn.scalars(
            select(collection_documents.c.collection)
            .where(collection_documents.c.document == name)
            .order_by(collection_documents.c.collection)
        )
        return list(held)


# --- preview ------------------------------------------------------------------

# One lock per document being built, so two readers of the same document build the preview once.
# An `anyio.Lock` rather than a threading one: a preview is only ever built on Litestar's event
# loop (the handler that opens a document), so one set of loop primitives covers every builder, and
# a waiting reader yields its loop instead of blocking it. Made on first use, because no lock may
# exist before there is a loop to await it on.
_preview_locks: dict[str, anyio.Lock] = {}


# A preview build parses a whole document, so a burst of opens would otherwise start one parse per
# request. The limiter admits `pipeline.preview_workers` of them; the rest wait, and a reader
# that waited this long is told to retry instead of holding its request open forever.
PREVIEW_WAIT_SECONDS = 60
# 2 is the default of `PipelineSettings.preview_workers`. An anyio limiter, unlike the CPU budget:
# every preview waits on Litestar's event loop, as `_preview_locks` describes. Its size may change
# under running builds, and it counts them: from 2 to 3 under load admits one more build, not three.
_preview_slots = anyio.CapacityLimiter(2)


def configure_preview_slots(workers: int) -> None:
    """Resize the pool of preview builders (from `workflows.apply_settings`)."""
    _preview_slots.total_tokens = workers


async def ensure_preview(name: str) -> tuple[Document, convert.Preview]:
    """Build the side-by-side preview (first pages only for PDF) once, on first open.

    Returns the preview beside the row so a caller never has to re-check `Document.preview` for
    a `None` this has just ruled out.

    Two locks, always in this order: the document's own (build this document once), then a
    slot in the process-wide pool (build at most `preview_workers` documents at a time).
    The parse itself is CPU work, so it runs under the CPU budget: a PDF in the extraction pool,
    anything else in a worker thread.
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
            try:
                with anyio.fail_after(PREVIEW_WAIT_SECONDS):
                    await _preview_slots.acquire()
            except TimeoutError:
                raise NotReady("preview queue is full; retry") from None
            # PDF extraction holds the GIL (see `cpu`), so on a thread it would stall this loop,
            # and every request on it, for the whole parse. It leaves the interpreter, as the
            # import's does; every other kind releases the GIL and stays on a thread.
            run = cpu.off_interpreter if info.suffix == ".pdf" else cpu.on_cpu
            try:
                preview = await run(
                    convert.build_preview,
                    info.source_path(),
                    info.preview_dir,
                    info.parser,
                    info.skip_ocr_pages,
                )
                async with db.connect() as conn:
                    await conn.execute(
                        update(documents)
                        .where(documents.c.name == name)
                        .values(preview=db.dumps(preview))
                    )
            finally:
                _preview_slots.release()
            return await get(name), preview
        finally:
            # Still holding the lock, so a queued reader keeps it: dropping it here would send
            # that reader and a newcomer into the same failing build at once. With nobody waiting
            # the build is committed (or failed) and the next reader needs no lock at all.
            if lock.statistics().tasks_waiting == 0:
                _preview_locks.pop(name, None)


# --- removal ------------------------------------------------------------------
# Two steps so a delete operation can run each one as a durable step. The row goes last, so a
# crash leaves a document that can be removed again instead of orphaned files. The index rows in
# every collection that held the document are removed by that collection first (see `workflows`).


async def remove_files(name: str) -> None:
    await home.remove_tree(root(name))


async def remove_row(name: str) -> None:
    """Cascades to `collection_documents` and `embeddings`; `pragma foreign_keys = on` is set on
    every connection."""
    async with db.connect() as conn:
        await conn.execute(delete(documents).where(documents.c.name == name))
