"""Optical character recognition (OCR) on the device, inside `pdf_inspector`: PP-OCRv6 small on
ONNX Runtime, the page rendered by PDFium. `convert` asks it to read the PDF pages the converter
finds no text on, and raster images.

Its model is fetched by the model lifecycle (`fetch`, run by `indexing.models` while the `ocr`
setting is on), so a conversion never touches the network: it reads offline (`read`).

Every function here runs in a worker thread or a pool worker, as `convert`'s do.
"""

import importlib.util
import io
import os
from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING

from haskie import home
from haskie.logs import get_logger

if TYPE_CHECKING:
    from PIL import Image

MODEL = "pp-ocrv6-small"  # the model `pdf_inspector` pins
# What OCR renders a page at, and so what an image is laid out at: one image pixel per rendered
# pixel. At the 72 dpi Pillow writes by default, a photo would be rendered at 4.3 times its pixels.
DPI = 150.0

_log = get_logger(__name__)


@cache
def _runtime() -> None:
    """Point OCR at the ONNX Runtime and PDFium libraries the installed Python packages ship, and
    its model at `haskie-ocr` in the Hugging Face cache, beside the other model weights
    (docs/storage.md), unless the environment already names others. Once per process: a pool
    worker sets its own."""
    from huggingface_hub import constants

    os.environ.setdefault("PDF_INSPECTOR_MODEL_CACHE", str(Path(constants.HF_HOME) / "haskie-ocr"))
    for variable, package, names in (
        ("ORT_DYLIB_PATH", "onnxruntime", ("libonnxruntime.", "onnxruntime.dll")),
        ("PDFIUM_LIB_PATH", "pypdfium2_raw", ("libpdfium.", "pdfium.dll")),
    ):
        spec = importlib.util.find_spec(package)
        if variable in os.environ or spec is None or spec.origin is None:
            continue
        folder = Path(spec.origin).parent
        # "libonnxruntime." leaves out its execution providers, "libonnxruntime_providers_*"
        found = [file for file in folder.glob("**/*") if file.name.startswith(names)]
        if found:
            os.environ[variable] = str(found[0])


def fetch(name: str, _accelerator: object = None) -> None:
    """Download the model (`_runtime` says where), or do nothing when it is there. The model
    lifecycle runs it for `ModelKind.OCR`, so it takes what the other models' loaders take.
    `pdf_inspector` offers no download of its own, so this reads one blank page, online. Raises
    when the download fails, so the lifecycle retries it."""
    import pdf_inspector

    assert name == MODEL, f"pdf_inspector pins {MODEL}, not {name}"
    _runtime()
    pdf_inspector.process_pdf_with_ocr_bytes(_blank_page(), mode="force")


@cache
def _blank_page() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PDF")
    return buffer.getvalue()


def read(source: Path | bytes, pages: list[int] | None = None) -> dict[int, str]:
    """The markdown OCR reads on each of the 1-based `pages` of a PDF (every page when None),
    by page; a page it reads no text on is left out.

    Offline: the model is fetched by `fetch`. OCR that cannot run (the model is not there, the
    runtime does not load) reads nothing, so the pages fall to the `skip_ocr_pages` policy, as
    before there was OCR."""
    import pdf_inspector

    _runtime()
    run = (
        partial(pdf_inspector.process_pdf_with_ocr_bytes, source)
        if isinstance(source, bytes)
        else partial(pdf_inspector.process_pdf_with_ocr, str(source))
    )
    try:
        result = run(mode="force", page_numbers=pages, dpi=DPI, offline=True)
    except Exception as exc:
        _log.warning("ocr_unavailable", error=home.scrub(f"{type(exc).__name__}: {exc}"))
        return {}
    return {page.page_number: page.markdown for page in result.pages if page.markdown.strip()}


def image_page(image: "Image.Image") -> bytes:
    """A raster image as the one-page PDF OCR reads, laid out at `DPI` on white: OCR reads PDFs
    only, and transparent pixels would otherwise turn black under dark text."""
    from PIL import Image

    rgba = _eight_bit(image).convert("RGBA")
    page = Image.new("RGB", rgba.size, "white")
    page.paste(rgba, mask=rgba)
    buffer = io.BytesIO()
    page.save(buffer, format="PDF", resolution=DPI)
    return buffer.getvalue()


def _eight_bit(image: "Image.Image") -> "Image.Image":
    """A 16- or 32-bit grayscale image scaled to 8 bits: Pillow's own conversion clips every value
    above 255 to white, so a 16-bit scan would lose all but its blackest ink."""
    if image.mode not in ("I", "I;16", "I;16B", "I;16L", "I;16N"):
        return image
    import numpy as np
    from PIL import Image

    pixels = np.asarray(image)  # its own width: 2 bytes a pixel for 16 bits, 4 for 32
    return Image.fromarray((pixels.clip(0, 65535) >> 8).astype(np.uint8))  # 8-bit gray: "L"
