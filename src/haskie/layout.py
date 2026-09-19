"""Where a document or a collection sits on disk: the shard directory keyed by its name.

`~/.haskie/documents/` and `~/.haskie/collections/` each hold one folder per entry. Ten thousand
documents would make ten thousand entries in one directory, which every lookup and every listing
pays for, so both roots insert a shard directory (`shard`) between them and the entry: an entry
lives at `<root>/<shard>/<name>/`, and a root spreads over 256 directories.
"""

import hashlib

PART_DIGITS = 6  # width of a micro-batch sequence number; four would cap a document at 10k parts


def shard(name: str) -> str:
    """The prefix directory an entry lives in: the first byte of the SHA-1 of its name, hex.

    The hash is over the UTF-8 bytes of the name, so it is stable across platforms and locales,
    and it is a name, not a secret: SHA-1 is used for its spread, not for its strength.
    """
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:2]
