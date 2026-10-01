"""Ids: an MD5 in base58, one width for all."""

import hashlib
from pathlib import Path

import pytest

from haskie import ids


@pytest.mark.parametrize(
    ("name", "digest", "expected"),
    [
        ("zero pads with the zero digit", bytes(16), "1" * 22),
        ("one is the digit after it", bytes(15) + b"\x01", "1" * 21 + "2"),
        ("58 carries into the next digit", bytes(15) + b"\x3a", "1" * 20 + "21"),
        ("the largest MD5 still fits the width", b"\xff" * 16, "YcVfxkQb6JRzqk5kF2tNLv"),
    ],
)
def test_base58(name: str, digest: bytes, expected: str) -> None:
    assert ids.base58(digest) == expected, name
    assert ids.ID.fullmatch(expected), f"{name}: an id's shape"


def test_an_id_is_the_md5_of_the_bytes_whether_read_whole_or_from_a_file(tmp_path: Path) -> None:
    path = tmp_path / "book.md"
    path.write_bytes(b"# Sagas\n")

    found = ids.md5(b"# Sagas\n")

    assert found == ids.md5_of_file(path) == ids.base58(hashlib.md5(b"# Sagas\n").digest())
    assert len(found) == ids.WIDTH and not set(found) & set("0OIl"), "no digit to misread"
