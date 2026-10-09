"""Cover images: the picture a document or collection card shows behind its name.

Every cover is a low-poly picture, a JPEG of triangles placed by a seed (`low_poly`). A
document's triangles take their colours from its cover page when it has one: a PDF's first page,
an EPUB's cover image, an image file itself. Every other document's take theirs from a random
gradient. A document's seed is its id, the MD5 of its bytes: the same file always draws the same
picture.

A document's cover is built once, on first request, and kept in its folder (`Document.cover`,
`of_document`). A collection's cover is made of its first documents' covers by name: one alone,
two stacked, or four in a grid, by how many it holds. An empty one gets a low-poly gradient seeded
by its name (`of_collection`).
"""

import colorsys
import io
import posixpath
import random
import threading
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote
from xml.etree import ElementTree

import anyio

from haskie import cpu, home, logs
from haskie.document.convert import RASTER_SUFFIXES
from haskie.document.document import Document

log = logs.get_logger(__name__)

COVER_PX = 480  # its side: a gallery card is 228 CSS pixels square, twice that on a retina screen
JPEG_QUALITY = 85
JPEG = "image/jpeg"
# the suffixes whose files can carry a cover page: an SVG has no pixels for Pillow to read
PAGE_SUFFIXES = {".pdf", ".epub"} | RASTER_SUFFIXES
# an EPUB is a zip: a cover image that inflates past this is refused, not read into memory
MAX_EPUB_IMAGE_BYTES = 64 * 1024 * 1024
FACET_LIGHT = 0.12  # how far a facet is lightened or darkened, as a share of its colour
POLY_CELLS = 7  # a 7 x 7 grid, two triangles a cell: few enough to read as facets
# A collection's cover changes with its members, and a document's name, once its document is
# deleted, can be taken by another file. So the browser asks again each time: tens of kilobytes.
FRESH = {"Cache-Control": "no-cache"}
# pdfium is not thread-safe, even across documents (pypdfium2's own warning)
_pdfium_lock = threading.Lock()
# one build per document at a time, as two would write the same temporary file; held only while
# someone waits on it, as `document._preview_locks` are
_building: dict[str, anyio.Lock] = {}


def cover_page(source: Path) -> Any:
    """The cover page of the file at `source` as an RGB Pillow image, or None when it has none.

    A file that cannot be read has none either: the cover only decorates its card, and the import
    reports what is wrong with the file. Sync, for a worker thread or the extraction pool."""
    from PIL import Image
    from pypdfium2 import PdfiumError

    try:
        page = _page(source)
        return None if page is None else page.convert("RGB")
    except (
        OSError,  # Pillow's UnidentifiedImageError among them
        KeyError,  # a member the EPUB names but does not hold
        ElementTree.ParseError,
        zipfile.BadZipFile,
        Image.DecompressionBombError,
        PdfiumError,
    ) as exc:
        log.warning("cover_page_unreadable", file=source.name, error=type(exc).__name__)
    return None


def _page(source: Path) -> Any:
    """The cover page as Pillow opened it, in its own mode; None for a file that has none."""
    from PIL import Image

    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return _pdf_first_page(source)
    if suffix == ".epub":
        found = _epub_cover(source)
        image = None if found is None else Image.open(io.BytesIO(found))
    elif suffix in PAGE_SUFFIXES:
        image = Image.open(source)
    else:
        return None
    if image is not None:
        # a JPEG decodes at a fraction of its size: a 24 MP photo takes 1 MB, not 72
        image.draft("RGB", (COVER_PX, COVER_PX))
    return image


def _pdf_first_page(source: Path) -> Any:
    import pypdfium2

    with _pdfium_lock, pypdfium2.PdfDocument(source) as pdf:
        page = pdf[0]
        # one PDF point is one pixel at scale 1
        return page.render(scale=COVER_PX / max(page.get_size())).to_pil()


def _epub_cover(source: Path) -> bytes | None:
    """The bytes of the image an EPUB names as its cover: the manifest item marked
    `cover-image` (EPUB 3), else the one a `<meta name="cover">` points at (EPUB 2)."""
    with zipfile.ZipFile(source) as epub:
        container = ElementTree.fromstring(epub.read("META-INF/container.xml"))
        rootfile = container.find(".//{*}rootfile")
        if rootfile is None:
            return None
        opf_path = rootfile.get("full-path", "")
        opf = ElementTree.fromstring(epub.read(opf_path))
        meta = opf.find(".//{*}metadata/{*}meta[@name='cover']")
        cover_id = None if meta is None else meta.get("content")
        for item in opf.findall(".//{*}manifest/{*}item"):
            marked = "cover-image" in item.get("properties", "").split()
            if marked or (cover_id is not None and item.get("id") == cover_id):
                # an href is a relative, percent-encoded URI: `../Images/my%20cover.png`
                href = unquote(item.get("href", ""))
                member = epub.getinfo(
                    posixpath.normpath(posixpath.join(posixpath.dirname(opf_path), href))
                )
                if member.file_size > MAX_EPUB_IMAGE_BYTES:
                    return None
                return epub.read(member)
    return None


def draw(source: Path, seed: str) -> bytes:
    """The cover as a JPEG: the cover page of the file at `source` drawn low-poly, else a low-poly
    gradient. `seed` places the triangles, and picks the gradient's colours. Sync, for the
    extraction pool."""
    return low_poly(seed, cover_page(source))


def low_poly(seed: str, page: Any = None) -> bytes:
    """A low-poly picture as a JPEG: a grid of triangles, its points shifted at random, each
    triangle one colour.

    With a `page`, a triangle takes the mean colour of the page under it, the page cropped square
    from its top, where a book's title is. Without, it takes a colour on the line between two
    random ones: where it sits picks most of it, so the picture reads as a gradient, and chance
    shifts it along the line. Either way chance lightens or darkens it, as a light would a facet
    tilted its own way, so each triangle shows. The same seed and page draw the same picture."""
    from PIL import Image, ImageDraw, ImageOps, ImageStat

    pick = random.Random(seed)  # a str seed is hashed with SHA-512: the same in every process
    hue = pick.random()
    # the second hue a quarter turn on at most: further, the line between them runs through grey
    hues = (hue, (hue + pick.uniform(0.08, 0.25)) % 1)
    first, last = (colorsys.hls_to_rgb(one, pick.uniform(0.4, 0.65), 0.75) for one in hues)
    size = (COVER_PX, COVER_PX)
    if page is not None:
        page = ImageOps.fit(page, size, centering=(0.5, 0.0))
    step = COVER_PX / POLY_CELLS
    jitter = step * 0.4

    def shift(index: int) -> float:
        # the border stays put, so the triangles fill the picture to its edges
        return pick.uniform(-jitter, jitter) if 0 < index < POLY_CELLS else 0

    def colour(triangle: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
        """The triangle's colour before its light, each channel from 0 to 1."""
        if page is not None:
            mask = Image.new("L", size)
            ImageDraw.Draw(mask).polygon(triangle, fill=255)
            return tuple(channel / 255 for channel in ImageStat.Stat(page, mask).mean)
        centre = sum(px + py for px, py in triangle) / 3
        share = min(1, max(0, centre / (2 * COVER_PX) + pick.uniform(-0.2, 0.2)))
        return tuple(one + (two - one) * share for one, two in zip(first, last, strict=True))

    corners = range(POLY_CELLS + 1)
    points = [[(x * step + shift(x), y * step + shift(y)) for x in corners] for y in corners]
    image = Image.new("RGB", size)
    canvas = ImageDraw.Draw(image)
    for y in range(POLY_CELLS):
        for x in range(POLY_CELLS):
            a, b, c, d = points[y][x], points[y][x + 1], points[y + 1][x + 1], points[y + 1][x]
            halves = ((a, b, c), (a, c, d)) if pick.random() < 0.5 else ((a, b, d), (b, c, d))
            for triangle in halves:
                light = pick.uniform(1 - FACET_LIGHT, 1 + FACET_LIGHT)
                rgb = tuple(min(255, round(255 * light * one)) for one in colour(triangle))
                # outlined in its own colour, so no seam shows between two triangles
                canvas.polygon(triangle, fill=rgb, outline=rgb)
    return _jpeg(image)


def _jpeg(image: Any) -> bytes:
    out = io.BytesIO()
    image.save(out, "JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


async def of_document(row: Document) -> Path:
    """The document's cover, built on first request: its cover page drawn low-poly, else a
    low-poly gradient."""
    # setdefault, with no await in between, so two readers of one document take the same lock
    lock = _building.setdefault(row.id, anyio.Lock())
    try:
        async with lock:
            if not await anyio.Path(row.cover).is_file():
                # Pillow and pdfium are native code reading an untrusted file: in the extraction
                # pool, a crash takes one worker, not the server (see `cpu`)
                drawn = await cpu.off_interpreter(draw, row.source_path(), row.id)
                await home.atomic_write(row.cover, drawn)
    finally:
        if lock.statistics().tasks_waiting == 0:
            _building.pop(row.id, None)
    return row.cover


def mosaic(covers: list[Path]) -> bytes:
    """One cover alone, two stacked top and bottom, or four in a 2 by 2 grid, as one JPEG. Each
    is cropped to its cell around its centre. Sync, for a worker thread."""
    from PIL import Image, ImageOps

    columns = 2 if len(covers) == 4 else 1
    rows = 1 if len(covers) == 1 else 2
    cell = (COVER_PX // columns, COVER_PX // rows)
    image = Image.new("RGB", (COVER_PX, COVER_PX))
    for index, path in enumerate(covers):
        with Image.open(path) as one:
            tile = ImageOps.fit(one, cell)
        image.paste(tile, ((index % columns) * cell[0], (index // columns) * cell[1]))
    return _jpeg(image)


async def of_collection(found: list[Document], name: str) -> bytes:
    """A collection's cover from its first documents by name (`Collection.first_members`): one
    alone, the first two stacked from two or three, all four from four. An empty collection gets a
    low-poly gradient seeded by its name."""
    if not found:
        return await cpu.on_cpu(low_poly, name)
    shown = found if len(found) == 4 else found[:2]
    return await cpu.on_cpu(mosaic, [await of_document(one) for one in shown])
