"""The open corpora as `manifest.json` lists them: each book downloaded, checked against its hash,
and indexed into one haskie collection per subject.

A book's file never enters the repository: the manifest pins where it came from and its SHA-256,
and `fetch` downloads it into `evals/corpus/open/<collection>/`, the gitignored corpus directory. A
file whose hash no longer matches is an error, not something to re-download silently: the
questions written from it would no longer hold.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

import msgspec

from evals import setup

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
DIRECTORY = setup.CORPUS_DIR / "open"
QUESTIONS = HERE / "questions"  # one reviewed bookqa dataset per collection, versioned
USER_AGENT = "Mozilla/5.0 (haskie eval corpora)"  # some hosts refuse urllib's default agent


class Document(msgspec.Struct):
    name: str  # the file name haskie imports it under, unique across the corpora
    title: str  # as the list names it
    url: str  # the file itself
    sha256: str
    bytes: int
    page: str = ""  # the listed page `url` was found on, when the list links a page, not the file


class Collection(msgspec.Struct):
    name: str
    subject: str  # the list's subject heading
    documents: list[Document]


class Skipped(msgspec.Struct):
    subject: str
    title: str
    url: str
    reason: str


class Manifest(msgspec.Struct):
    source: str  # the list the books were picked from, pinned to a commit
    collections: list[Collection]
    skipped: list[Skipped] = []  # links tried and left out, and why


class HashMismatch(RuntimeError):
    pass


def load(path: Path = MANIFEST) -> Manifest:
    return msgspec.json.decode(path.read_bytes(), type=Manifest)


def dump(manifest: Manifest, path: Path = MANIFEST) -> None:
    path.write_bytes(msgspec.json.format(msgspec.json.encode(manifest), indent=2) + b"\n")


def find(name: str, path: Path | None = None) -> Collection:
    """The manifest's collection `name`; `KeyError` naming the ones there are."""
    path = path or MANIFEST
    collections = {c.name: c for c in load(path).collections}
    if name not in collections:
        raise KeyError(f"no collection {name} in {path}; there are {', '.join(collections)}")
    return collections[name]


def questions(name: str) -> Path:
    """Where the reviewed questions on collection `name` live: a bookqa dataset."""
    return QUESTIONS / f"{name}.jsonl"


def stems(collection: Collection) -> set[str]:
    """Its books' file names without extension: the names of their bookqa candidate folders."""
    return {Path(d.name).stem for d in collection.documents}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, timeout: float = 120) -> bytes:
    request = urllib.request.Request(url, headers={"user-agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def fetch(collection: Collection, directory: Path = DIRECTORY) -> list[Path]:
    """Every book of `collection` on disk, downloading what is missing, each checked against its
    hash."""
    target = directory / collection.name
    target.mkdir(parents=True, exist_ok=True)
    files = []
    for document in collection.documents:
        path = target / document.name
        if not path.exists():
            print(f"fetching {document.name}", flush=True)
            path.write_bytes(download(document.url))
        if sha256(path.read_bytes()) != document.sha256:
            raise HashMismatch(f"{path} does not match the manifest: {document.url} changed")
        files.append(path)
    return files


def description(collection: Collection) -> str:
    return f"Open books on {collection.subject}, from free-programming-books."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api", default=os.environ.get("HASKIE_EVAL_CORPORA_URL", "http://127.0.0.1:8126")
    )
    parser.add_argument(
        "--profile", default=os.environ.get("HASKIE_EVAL_EMBED_PROFILE", "granite-small-english")
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--collection", action="append", help="default: every collection")
    args = parser.parse_args(argv)
    manifest = load(args.manifest)
    wanted = [c for c in manifest.collections if c.name in (args.collection or [c.name])]
    setup.ensure_profile(args.profile, args.api)
    ok = True
    for collection in wanted:
        files = fetch(collection)
        ok = setup.load(files, collection.name, description(collection), args.api) and ok
    if not ok:
        print("not every book imported and indexed", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
