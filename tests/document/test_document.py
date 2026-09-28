"""Documents: import by path, staging, and the preview build."""

import errno
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from haskie import errors
from haskie.document import document

from conftest import MD, NO_MODELS, audit_lines, document_names  # isort: skip

pytestmark = pytest.mark.anyio


# --- import by path ---------------------------------------------------------------


def _source(folder: Path, name: str = "salary.md", content: str = MD) -> Path:
    """A file the user already has, in a folder of their own."""
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / name
    source.write_text(content)
    return source


def _relative(folder: Path) -> str:
    return "private/salary.md"


def _missing(folder: Path) -> str:
    return str(folder / "salary.md")


def _unreadable(folder: Path) -> str:
    source = _source(folder)
    source.chmod(0)
    return str(source)


def _unsupported(folder: Path) -> str:
    return str(_source(folder, "salary.exe"))


def _too_large(folder: Path) -> str:
    return str(_source(folder, content="x" * 128))


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file of mode 000")
@pytest.mark.parametrize(
    ("name", "path", "message"),
    [
        ("a relative path", _relative, "path must be absolute: salary.md"),
        ("a file that is not there", _missing, "file not found: salary.md"),
        ("a file the process cannot read", _unreadable, "cannot read file: salary.md: "),
        ("a type it cannot parse", _unsupported, "unsupported file type: salary.exe"),
        ("a file over the upload cap", _too_large, "file larger than 100 bytes: 128"),
    ],
)
async def test_a_refused_path_import_names_the_file_but_never_its_folder(
    client,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    path: Callable[[Path], str],
    message: str,
) -> None:
    """The error body and the audit line both carry the refusal, and the audit trail never holds
    where an import came from."""
    monkeypatch.setattr(document, "UPLOAD_MAX_BYTES", 100)  # above MD, below the large file
    folder = tmp_path / "private"
    await client.post("/api/init", json=NO_MODELS)
    given = path(folder)

    response = await client.post("/api/documents/import", json={"path": given})

    assert response.status_code == 422, name
    assert response.json()["detail"].startswith(message), name
    (record,) = [line for line in audit_lines() if line["event"] == "document.import"]
    assert (record["outcome"], record["detail"]["source"]) == ("error", Path(given).name)
    assert message in record["error"], name
    assert str(folder) not in json.dumps(record), name
    assert "private/salary" not in json.dumps(record), name
    assert await document_names() == [], name


@pytest.mark.parametrize(
    ("name", "filename"),
    [
        ("an error naming no file", None),
        ("an error about the document's own folder", "{destination}"),
    ],
)
async def test_a_path_import_passes_on_a_failure_that_is_not_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, filename: str | None
) -> None:
    """Only a failure to read the user's file is turned into a refusal; the disk filling up under
    the document's folder stays the error it is, and the row goes with it."""
    source = _source(tmp_path / "private")

    def copyfile(src: Path, dst: Path) -> None:
        text = None if filename is None else filename.format(destination=dst)
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), text)

    monkeypatch.setattr(shutil, "copyfile", copyfile)

    with pytest.raises(OSError) as raised:
        await document.import_path(str(source))

    assert not isinstance(raised.value, errors.InvalidInput), name
    assert raised.value.errno == errno.ENOSPC, name
    assert await document_names() == [], name
