"""On-device OCR: the PDF pages the converter finds no text on, and raster images.

Every test but the `network` one stands `pdf_inspector`'s OCR in with `ocr_reads` (conftest):
the real one needs its model, a download.
"""

import io
import os
from pathlib import Path

import pdf_inspector
import pytest
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader

from conftest import ocr_reads, text_pdf, use_ocr  # isort: skip
from haskie.document import convert, ocr
from haskie.document.bookmarks import Bookmark
from haskie.errors import PermanentError
from haskie.settings import Parser

SCANNED = "Raft elects a leader per term"


def _scan(lines: list[str], size: tuple[int, int] = (1700, 900)) -> Image.Image:
    """A page as a scanner gives it: black text on white pixels, nothing extractable."""
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for at, line in enumerate(lines):
        draw.text((100, 100 + at * 80), line, fill="black", font=ImageFont.load_default(size=48))
    return image


@pytest.mark.parametrize(
    ("name", "pages", "texts", "on", "skip", "unread", "expected", "refused"),
    [
        ("text pages only: OCR never runs", ["one", "two"], {}, True, True, [], ["one"], None),
        (
            "a scan OCR reads: its text stands, the policy has nothing to judge",
            ["one", None],
            {2: SCANNED},
            True,
            False,
            [],
            ["one", f"<!-- page 2 -->\n\n{SCANNED}"],
            None,
        ),
        (
            "two scans, OCR reads the second: the first is skipped, the second matched by page",
            ["one", None, None],
            {3: SCANNED},
            True,
            True,
            [2],
            ["<!-- page 2: needs OCR, skipped -->", f"<!-- page 3 -->\n\n{SCANNED}"],
            None,
        ),
        (
            "scans OCR reads nothing on, skip off: their empty text stays, the policy refuses",
            ["one", None, "two", None],
            {},
            True,
            False,
            [2, 4],
            ["one", "<!-- page 2 -->", "two", "<!-- page 4 -->"],
            "2 of 4 pages need OCR and OCR read no text",
        ),
        (
            "OCR off: a scan is left unread, and skipped",
            ["one", None],
            {2: SCANNED},
            False,
            True,
            [2],
            ["one", "<!-- page 2: needs OCR, skipped -->"],
            None,
        ),
        (
            "every page unread, skip on: the policy still refuses",
            [None, None],
            {},
            True,
            True,
            [1, 2],
            ["<!-- page 1: needs OCR, skipped -->", "<!-- page 2: needs OCR, skipped -->"],
            "all 2 pages need OCR and OCR read no text",
        ),
    ],
)
def test_a_pdf_page_with_no_text_is_read_by_ocr_or_left_to_the_policy(
    name: str,
    pages: list,
    texts: dict[int, str],
    on: bool,
    skip: bool,
    unread: list[int],
    expected: list[str],
    refused: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = use_ocr(monkeypatch, texts)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(text_pdf(pages))
    scans = [page for page, text in enumerate(pages, start=1) if text is None]

    markdown, found, total = convert.pdf_pages_markdown(pdf, skip_ocr_pages=skip, ocr=on)

    assert (found, total) == (unread, len(pages)), name
    assert all(part in markdown for part in expected), (name, markdown)
    assert ("skipped" in markdown) == (skip and bool(unread)), name
    assert calls == ([(scans, True)] if on and scans else []), "one offline call, for the scans"
    if refused is None:
        convert.check_ocr_policy(len(found), total, skip)  # no raise: the policy accepts it
    else:
        with pytest.raises(PermanentError, match=refused):
            convert.check_ocr_policy(len(found), total, skip)


def test_a_batch_reads_the_scanned_page_before_it_to_claim_its_bookmarks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch converts the page before it only to learn which bookmarks it claims. When that page
    is a scan, OCR reads it too: the heading there claims the bookmark, so the running header that
    repeats it on the next page stays text, as it does in a conversion of the whole document."""
    calls = use_ocr(monkeypatch, {2: "# Consensus\n\nbody"})
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(text_pdf(["Front", None, [(24, "Consensus"), (12, "running text")]]))
    marks = [Bookmark(level=1, title="Book", page=1), Bookmark(2, "Consensus", 2)]

    markdown, _, total = convert.pdf_pages_markdown(pdf, [2], marks=marks, ocr=True)
    whole, _, _ = convert.pdf_pages_markdown(pdf, marks=marks, ocr=True)

    assert calls == [([2], True)] * 2, "the page before, a bookmark's, is read with the batch's own"
    assert total == 1, "the page before is not part of the batch"
    assert "# Consensus" not in markdown and "Consensus" in markdown, markdown
    assert markdown in whole, "the batch reads as the same pages of the whole document"


def test_ocr_that_cannot_run_reads_nothing_and_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The model is not there, or the runtime does not load: the pages fall to the
    `skip_ocr_pages` policy, as before there was OCR."""

    def broken(*_args: object, **_kwargs: object) -> None:
        raise ValueError(f"failed to load ONNX Runtime from {tmp_path}/libonnxruntime.dylib")

    monkeypatch.setattr(pdf_inspector, "process_pdf_with_ocr", broken)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(text_pdf(["one", None]))

    with caplog.at_level("WARNING"):
        markdown, found, _ = convert.pdf_pages_markdown(pdf, skip_ocr_pages=True, ocr=True)

    assert found == [2] and "<!-- page 2: needs OCR, skipped -->" in markdown
    (record,) = [r.msg for r in caplog.records if isinstance(r.msg, dict)]
    assert record["event"] == "ocr_unavailable"
    assert record["error"].startswith("ValueError: failed to load ONNX Runtime from $HASKIE_HOME")


@pytest.mark.parametrize(
    ("name", "suffix", "texts", "on", "expected", "asked"),
    [
        ("a raster image OCR reads", ".png", {1: SCANNED}, True, SCANNED, [(None, True)]),
        ("a raster image OCR reads nothing on", ".jpg", {}, True, "", [(None, True)]),
        ("OCR off: a raster image has no text", ".webp", {1: SCANNED}, False, "", []),
        ("a vector image: never read", ".svg", {1: SCANNED}, True, "", []),
    ],
)
def test_an_image_is_read_by_ocr(
    name: str,
    suffix: str,
    texts: dict[int, str],
    on: bool,
    expected: str,
    asked: list,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = use_ocr(monkeypatch, texts)
    path = tmp_path / f"shot{suffix}"
    if suffix == ".svg":
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>Raft</text></svg>')
    else:
        _scan([SCANNED]).save(path)

    assert convert.to_markdown(path, Parser.ANYDOC, on) == expected, name
    assert calls == asked, name


def _sixteen_bit_scan() -> Image.Image:
    """A 16-bit grayscale scan: ink at 4000 of 65535 on a white page."""
    image = Image.new("I;16", (300, 100), 65535)
    ImageDraw.Draw(image).rectangle((10, 30, 60, 60), fill=4000)
    return image


def _transparent_logo() -> Image.Image:
    """Dark ink on transparent pixels."""
    image = Image.new("RGBA", (300, 100), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((10, 30, 60, 60), fill=(0, 0, 0, 255))
    return image


@pytest.mark.parametrize(
    ("name", "image"),
    [
        ("transparent pixels laid on white, not black", _transparent_logo()),
        ("16-bit gray scaled to 8 bits, its ink kept", _sixteen_bit_scan()),
    ],
)
def test_an_image_is_laid_out_for_ocr_as_it_looks(
    name: str, image: Image.Image, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page OCR reads shows the image as a viewer does, at one image pixel per rendered pixel:
    at the 72 dpi Pillow writes by default, OCR would render it at 4.3 times its pixels."""
    sent: list[bytes] = []

    def stand_in(source: bytes, **options: object) -> object:
        sent.append(source)
        assert options["dpi"] == ocr.DPI, "rendered at the dpi the page is laid out at"
        return ocr_reads({})[0](source, **options)

    monkeypatch.setattr(pdf_inspector, "process_pdf_with_ocr_bytes", stand_in)
    path = tmp_path / "picture.png"
    image.save(path)

    convert.to_markdown(path, Parser.ANYDOC, ocr=True)

    (page,) = PdfReader(io.BytesIO(sent[0])).pages
    points = [round(float(side) * ocr.DPI / 72) for side in page.mediabox[2:]]
    assert points == [300, 100], name
    (picture,) = page.images
    with Image.open(io.BytesIO(picture.data)) as flat:
        rgb = flat.convert("RGB")
        assert rgb.getpixel((0, 0)) == (255, 255, 255), name
        ink = rgb.getpixel((30, 45))
        assert isinstance(ink, tuple) and max(ink) < 32, name


def test_an_image_that_cannot_be_opened_fails_the_document(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n not an image")

    with pytest.raises(PermanentError, match="could not convert broken.png"):
        convert.to_markdown(path, Parser.ANYDOC, ocr=True)


CONVERTED_PDF = (
    "<!-- page 1 -->\n\none\n\n<!-- page 2: needs OCR, skipped -->\n\n"
    f"<!-- page 3 -->\n\n{SCANNED}\n\n<!-- page 4 -->\n\nfour\n"
)


@pytest.mark.parametrize(
    ("name", "suffix", "converted", "shown", "right_pane", "unread"),
    [
        (
            "a converted PDF: its first pages, the skipped ones noted",
            ".pdf",
            CONVERTED_PDF,
            3,
            f"<!-- page 1 -->\n\none\n\n<!-- page 2: needs OCR, skipped -->\n\n"
            f"<!-- page 3 -->\n\n{SCANNED}\n",
            [2],
        ),
        (
            "a converted PDF shorter than the preview: all of it",
            ".pdf",
            CONVERTED_PDF,
            10,
            CONVERTED_PDF,
            [2],
        ),
        (
            "a PDF not converted yet: converted here, without OCR",
            ".pdf",
            None,
            10,
            "<!-- page 1 -->\n\none",
            [2],
        ),
        ("a converted image: what OCR read at import", ".png", SCANNED, 10, SCANNED, None),
        ("an image not converted yet: no text, without OCR", ".png", None, 10, "", None),
    ],
)
def test_a_preview_shows_the_converted_markdown_and_never_runs_ocr(
    name: str,
    suffix: str,
    converted: str | None,
    shown: int,
    right_pane: str,
    unread: list[int] | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = use_ocr(monkeypatch, {1: SCANNED, 2: SCANNED})
    monkeypatch.setattr(convert, "PREVIEW_PAGES", shown)
    source = tmp_path / f"original{suffix}"
    if suffix == ".pdf":
        source.write_bytes(text_pdf(["one", None, "three", "four"]))
    else:
        _scan([SCANNED]).save(source)
    markdown = source.with_name(source.name + ".md")
    if converted is not None:
        markdown.write_text(converted)

    preview = convert.build_preview(source, tmp_path / "preview", Parser.ANYDOC, True, markdown)

    pane = (tmp_path / "preview" / "preview.md").read_text()
    if converted is None:
        assert pane.startswith(right_pane), (name, pane)
    else:
        assert pane == right_pane, (name, pane)
    assert calls == [], "the import read the scans; a preview never reads them again"
    if unread is not None:
        assert (preview.kind, preview.ocr_pages) == (convert.PreviewKind.PDF, unread), name


@pytest.mark.parametrize(
    ("name", "variable", "named", "library"),
    [
        ("ONNX Runtime from its package", "ORT_DYLIB_PATH", "libonnxruntime.", True),
        ("PDFium from its package", "PDFIUM_LIB_PATH", "libpdfium.", True),
        ("the model beside the others", "PDF_INSPECTOR_MODEL_CACHE", "haskie-ocr", False),
    ],
)
def test_ocr_finds_its_runtime_and_model_folder(
    name: str, variable: str, named: str, library: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from huggingface_hub import constants

    ocr._runtime.cache_clear()
    monkeypatch.delenv(variable, raising=False)
    try:
        ocr._runtime()
    finally:
        ocr._runtime.cache_clear()

    found = Path(os.environ[variable])
    assert found.name.startswith(named) and found.is_file() == library, name
    assert library or found.parent == Path(constants.HF_HOME), "a folder of the HF cache"


def test_a_runtime_path_already_set_stands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORT_DYLIB_PATH", "/opt/ort/libonnxruntime.so")
    ocr._runtime.cache_clear()
    try:
        ocr._runtime()
    finally:
        ocr._runtime.cache_clear()

    assert os.environ["ORT_DYLIB_PATH"] == "/opt/ort/libonnxruntime.so"


def test_the_model_is_fetched_online_by_reading_one_blank_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`pdf_inspector` downloads its model on the first page it reads online, and offers no
    download of its own."""
    calls = use_ocr(monkeypatch, {})

    ocr.fetch(ocr.MODEL)

    assert calls == [(None, False)], "online, to download"
    with pytest.raises(AssertionError, match="pins pp-ocrv6-small"):
        ocr.fetch("another-model")


@pytest.mark.network
def test_real_ocr_reads_a_scanned_page_and_an_image(tmp_path: Path) -> None:
    """PP-OCRv6 small itself: fetched first (about 31 MB), then read offline."""
    ocr.fetch(ocr.MODEL)
    pdf = tmp_path / "scan.pdf"
    _scan(["Chapter 1", SCANNED], size=(1700, 2200)).save(pdf, format="PDF", resolution=200)
    image = tmp_path / "shot.png"
    _scan([SCANNED]).save(image)

    markdown, found, total = convert.pdf_pages_markdown(pdf, skip_ocr_pages=False, ocr=True)

    assert (found, total) == ([], 1), "OCR read the scanned page"
    assert SCANNED in markdown and "Chapter 1" in markdown, markdown
    assert SCANNED in convert.to_markdown(image, Parser.ANYDOC, ocr=True)
