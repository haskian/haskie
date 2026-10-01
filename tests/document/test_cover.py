"""Cover pages and low-poly pictures: the pictures behind the document and collection cards.

A cover only decorates a card, so a file with no cover page, or one that cannot be read, gets
None and the caller draws a low-poly picture instead. Never an exception: the import reports a
bad file.
"""

import io
import posixpath
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from haskie.document import cover

from conftest import text_pdf  # isort: skip

CONTAINER = (
    '<?xml version="1.0"?><container version="1.0" '
    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
    '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
    "</rootfiles></container>"
)


def png(width: int, height: int, mode: str = "RGB") -> bytes:
    out = io.BytesIO()
    Image.new(mode, (width, height), (200, 40, 10)).save(out, "PNG")
    return out.getvalue()


def epub(
    metadata: str, manifest: str, files: dict[str, bytes], container: str = CONTAINER
) -> bytes:
    """An EPUB as a reader opens it: the container names the package, whose manifest lists the
    files, one of them maybe the cover."""
    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Book</dc:title>'
        f"{metadata}</metadata><manifest>"
        '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
        f"{manifest}</manifest></package>"
    )
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as book:
        book.writestr("mimetype", "application/epub+zip")
        book.writestr("META-INF/container.xml", container)
        book.writestr("OEBPS/content.opf", opf)
        book.writestr("OEBPS/ch1.xhtml", "<html><body>One</body></html>")
        for name, body in files.items():
            book.writestr(posixpath.normpath(f"OEBPS/{name}"), body)
    return out.getvalue()


EPUB3_ITEM = '<item id="c" href="images/c.png" media-type="image/png" properties="cover-image"/>'
EPUB2_ITEM = '<item id="art" href="images/c.png" media-type="image/png"/>'
EPUB2_META = '<meta name="cover" content="art"/>'
COVER_FILE = {"images/c.png": png(600, 900)}
# beside the package's folder, not in it, as many EPUBs keep their images
UP_A_FOLDER = {"../Images/my c.png": png(600, 900)}


@pytest.mark.parametrize(
    ("name", "body", "size"),
    [
        pytest.param("paper.pdf", text_pdf(["Title"]), (480, 240), id="pdf-first-page-rendered"),
        pytest.param("photo.png", png(960, 640, "RGBA"), (960, 640), id="image-is-its-own-cover"),
        pytest.param(
            "book.epub", epub("", EPUB3_ITEM, COVER_FILE), (600, 900), id="epub3-cover-image"
        ),
        pytest.param(
            "book.epub", epub(EPUB2_META, EPUB2_ITEM, COVER_FILE), (600, 900), id="epub2-meta"
        ),
        pytest.param("book.epub", epub("", EPUB2_ITEM, COVER_FILE), None, id="epub-no-cover"),
        pytest.param("book.epub", epub("", EPUB3_ITEM, {}), None, id="epub-cover-missing"),
        pytest.param(
            "book.epub",
            epub("", EPUB3_ITEM.replace("images/c.png", "../Images/my%20c.png"), UP_A_FOLDER),
            (600, 900),
            id="epub-href-up-a-folder-and-percent-encoded",
        ),
        pytest.param(
            "book.epub",
            epub("", EPUB3_ITEM, COVER_FILE, container="<container><rootfiles/></container>"),
            None,
            id="epub-names-no-package",
        ),
        pytest.param("book.epub", b"not a zip", None, id="epub-not-a-zip"),
        pytest.param("broken.pdf", b"%PDF-1.4 nonsense", None, id="pdf-unreadable"),
        pytest.param("fake.png", b"not an image", None, id="image-unreadable"),
        pytest.param("notes.md", b"# Notes", None, id="markdown-has-none"),
    ],
)
def test_cover_page(tmp_path: Path, name: str, body: bytes, size: tuple | None) -> None:
    source = tmp_path / name
    source.write_bytes(body)

    found = cover.cover_page(source)

    if size is None:
        assert found is None
    else:
        assert (found.mode, found.size) == ("RGB", size), "read whole, in the colours drawn from"


def test_an_epub_cover_too_large_to_inflate_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.epub"
    source.write_bytes(epub("", EPUB3_ITEM, COVER_FILE))
    monkeypatch.setattr(cover, "MAX_EPUB_IMAGE_BYTES", len(COVER_FILE["images/c.png"]) - 1)

    assert cover.cover_page(source) is None


def test_a_low_poly_picture_is_drawn_from_its_seed() -> None:
    one = cover.low_poly("rust-book.pdf")

    image = Image.open(io.BytesIO(one))
    assert (image.format, image.size) == ("JPEG", (cover.COVER_PX, cover.COVER_PX))
    colours = image.getcolors(maxcolors=cover.COVER_PX**2)
    assert colours is not None and len(colours) > 2 * cover.POLY_CELLS**2, "a colour per facet"
    assert cover.low_poly("rust-book.pdf") == one, "the same seed draws the same picture"
    assert cover.low_poly("ostep.pdf") != one, "another seed draws another"


def test_a_cover_page_lends_its_colours_from_the_top() -> None:
    """Every facet takes the page's colour under it, and the page is cropped square from its top,
    where a book's title is: a tall page blue above and red below draws blue alone."""
    page = Image.new("RGB", (480, 960), (20, 40, 200))
    page.paste((200, 40, 10), (0, 480, 480, 960))

    drawn = Image.open(io.BytesIO(cover.low_poly("rust-book.pdf", page)))

    pixels = np.asarray(drawn)
    assert pixels[..., 0].max() < 60 and pixels[..., 2].min() > 150, "blue, lit or shaded, no red"
    colours = np.unique(pixels.reshape(-1, 3), axis=0)
    assert len(colours) > 2 * cover.POLY_CELLS**2, "still a colour per facet"


def test_a_cover_is_drawn_from_its_page_or_a_gradient(tmp_path: Path) -> None:
    source = tmp_path / "photo.png"
    source.write_bytes(png(600, 600))

    notes = tmp_path / "notes.md"
    notes.write_text("# Notes")

    assert cover.draw(source, "seed") == cover.low_poly("seed", cover.cover_page(source))
    assert cover.draw(notes, "seed") == cover.low_poly("seed"), "no cover page: the gradient"


RED, BLUE, GREEN, WHITE = (220, 30, 30), (30, 30, 220), (30, 200, 30), (240, 240, 240)


@pytest.mark.parametrize(
    ("colours", "cells"),
    [
        pytest.param([RED], {(120, 120): RED, (360, 360): RED}, id="one-alone"),
        pytest.param([RED, BLUE], {(240, 120): RED, (240, 360): BLUE}, id="two-stacked"),
        pytest.param(
            [RED, BLUE, GREEN, WHITE],
            {(120, 120): RED, (360, 120): BLUE, (120, 360): GREEN, (360, 360): WHITE},
            id="four-in-a-grid",
        ),
    ],
)
def test_a_mosaic_lays_out_its_covers(tmp_path: Path, colours: list, cells: dict) -> None:
    covers = []
    for index, colour in enumerate(colours):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (cover.COVER_PX, cover.COVER_PX), colour).save(path)
        covers.append(path)

    drawn = Image.open(io.BytesIO(cover.mosaic(covers)))

    assert (drawn.format, drawn.size) == ("JPEG", (cover.COVER_PX, cover.COVER_PX))
    for point, colour in cells.items():
        found = drawn.getpixel(point)
        assert isinstance(found, tuple)
        assert all(abs(a - b) < 8 for a, b in zip(found, colour, strict=True)), point
