"""Ids: an MD5, written in base58.

A document's id is one, of its bytes (`Document.id`), and so are a section's and a chunk's, of
their place in the document (`sections.build`). Base58 (Bitcoin's alphabet) leaves
out `0`, `O`, `I` and `l`, so an id reads back unambiguously, and holds letters and digits alone,
so it is safe in a path, a URN and a SQL literal. Every id is `WIDTH` characters: the digest as
one number in base 58, padded with the zero digit `1`, so an id's shape can be checked (`ID`).
"""

import hashlib
import re
from pathlib import Path

ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
WIDTH = 22  # 58**22 > 2**128: every MD5 fits
ID = re.compile(f"[{ALPHABET}]{{{WIDTH}}}")


def base58(digest: bytes) -> str:
    """`digest` as one number in base 58, `WIDTH` digits, most significant first."""
    number = int.from_bytes(digest, "big")
    digits: list[str] = []
    while number:
        number, digit = divmod(number, 58)
        digits.append(ALPHABET[digit])
    return "".join(reversed(digits)).rjust(WIDTH, ALPHABET[0])


def md5(data: bytes) -> str:
    """The id of `data`."""
    return base58(hashlib.md5(data, usedforsecurity=False).digest())


def md5_of_file(path: Path) -> str:
    """The id of a file's bytes: a fingerprint that makes the same file the same document, not a
    security check. Read in blocks, since an import may be as large as the upload cap."""
    with path.open("rb") as handle:
        return base58(
            hashlib.file_digest(handle, lambda: hashlib.md5(usedforsecurity=False)).digest()
        )
