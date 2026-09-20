"""PDF extraction in a separate interpreter.

The rest of the suite sets `cpu.CONVERT_WORKERS = 0` and extracts inline, because a pool per
xdist worker costs more to start than the tests save. This module is the one place the pool itself
runs, so the pickling boundary and the forkserver context stay covered.
"""

from pathlib import Path

import pytest

from haskie import convert, cpu

from conftest import text_pdf  # isort: skip

pytestmark = pytest.mark.anyio

PAGES: list[str | None] = ["page one text", "page two text", "page three text"]


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "doc.pdf"
    path.write_bytes(text_pdf(PAGES))
    return path


async def test_extraction_in_a_pool_matches_extraction_inline(pdf: Path, monkeypatch) -> None:
    """Same answer either way: `off_interpreter` only changes where the work runs."""
    inline = await cpu.off_interpreter(convert.pdf_pages_markdown, pdf)

    monkeypatch.setattr(cpu, "CONVERT_WORKERS", 2)
    monkeypatch.setattr(cpu, "_pool", None)
    try:
        pooled = await cpu.off_interpreter(convert.pdf_pages_markdown, pdf)
    finally:
        cpu.shutdown_pool()

    assert pooled == inline
    markdown, ocr_pages, pages = pooled
    assert (pages, ocr_pages) == (len(PAGES), [])
    assert "<!-- page 1 -->" in markdown, "the page markers the chunker reads survive the boundary"


async def test_an_error_in_the_pool_reaches_the_caller(tmp_path: Path, monkeypatch) -> None:
    """A parser failure is a `PermanentError` whether or not it crossed a process boundary."""
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    monkeypatch.setattr(cpu, "CONVERT_WORKERS", 2)
    monkeypatch.setattr(cpu, "_pool", None)
    try:
        with pytest.raises(convert.PermanentError, match="could not convert bad.pdf"):
            await cpu.off_interpreter(convert.pdf_pages_markdown, bad)
    finally:
        cpu.shutdown_pool()
