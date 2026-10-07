"""Build `manifest.json`, by hand: for each subject asked for, the first `PER_COLLECTION` books of
free-programming-books' subject list (pinned to `LIST_COMMIT`) that download as one file and hold
text haskie can index.

The list mixes links to a book's file with links to a website that serves it chapter by chapter;
only a link to one PDF or EPUB can be pinned by hash, so the others are not candidates. A link
that answers with an HTML page - a book's landing page - is followed one step, to the PDF or EPUB
on it named most like the book, and the manifest pins that file and names the page. A
candidate is tried in the list's order and kept when it downloads, is under `MAX_BYTES`, and is a
real PDF whose first pages hold text (not a scan, not an HTML page behind a .pdf link) or a real
EPUB. Every link tried and left out is recorded in the manifest with the reason, and every kept
file lands in the corpus directory, so `corpus.py` downloads nothing again. A subject built again
replaces its own collection in the manifest and leaves the others as they are.
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import time
import urllib.parse
import zipfile
from pathlib import Path

import msgspec
import pypdf

from evals.corpora import corpus
from evals.corpora.corpus import Collection, Document, Manifest, Skipped

LIST_COMMIT = "0ddd8749a993efeac1644ea9f47af0bb6f47b955"  # 2026-10-05
LIST_URL = (
    "https://raw.githubusercontent.com/EbookFoundation/free-programming-books/"
    f"{LIST_COMMIT}/books/free-programming-books-subjects.md"
)
PILOT = ("Operating Systems", "Machine Learning", "Software Architecture")
PER_COLLECTION = 10
MAX_BYTES = 100_000_000
TEXT_PAGES = 15  # pages of a PDF read for text
MIN_TEXT = 2_000  # characters those pages must hold: under it, a scan or a cover
# Books haskie cannot read though they pass the checks here: pypdf and haskie's converter extract
# text differently, so only an import tells. `eval:corpora:setup` fails on such a book; add it.
EXCLUDED = {
    "https://markburgess.org/os/os.pdf": "haskie finds no text: all 140 pages need OCR",
}
RETRY_SECONDS = 5

logging.getLogger("pypdf").setLevel(logging.ERROR)


class Candidate(msgspec.Struct, frozen=True):
    subject: str
    title: str
    url: str


class Rejected(Exception):
    pass


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def subjects(markdown: str) -> dict[str, str]:
    """Each `### ` subject's heading, with any anchors stripped, and the text under it."""
    found = {}
    for part in re.split(r"^### ", markdown, flags=re.M)[1:]:
        heading, _, body = part.partition("\n")
        found[re.sub(r"<[^>]+>", "", heading).strip()] = body
    return found


def candidates(subject: str, body: str) -> list[Candidate]:
    """The subject's links to one PDF or EPUB, in the list's order: by the URL's extension or by
    the `(PDF)`/`(EPUB)` the list writes after it."""
    found = []
    for title, url, rest in re.findall(r"^\s*\* \[([^\]]+)\]\(([^)\s]+)\)(.*)$", body, re.M):
        if re.search(r"\.(pdf|epub)([?#]|$)", url, re.I) or re.search(r"\((PDF|EPUB)\)", rest):
            found.append(Candidate(subject, title.strip(), url))
    return found


def kind(data: bytes) -> str:
    """`pdf` or `epub` for a file haskie can index, else `Rejected` with the reason."""
    if len(data) > MAX_BYTES:
        raise Rejected(f"{len(data) / 1e6:.0f} MB, over {MAX_BYTES / 1e6:.0f} MB")
    if data.startswith(b"%PDF"):
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            text = "".join(page.extract_text() or "" for page in reader.pages[:TEXT_PAGES])
        except Exception as error:  # pypdf raises many kinds on a broken file
            raise Rejected(f"unreadable PDF: {type(error).__name__}") from error
        if len(text.strip()) < MIN_TEXT:
            raise Rejected(f"{len(text.strip())} characters of text in {TEXT_PAGES} pages: a scan?")
        return "pdf"
    if data.startswith(b"PK"):
        try:
            mimetype = zipfile.ZipFile(io.BytesIO(data)).read("mimetype").strip()
        except (zipfile.BadZipFile, KeyError) as error:
            raise Rejected("a zip, not an EPUB") from error
        if mimetype != b"application/epub+zip":
            raise Rejected(f"a zip of {mimetype.decode(errors='replace')}, not an EPUB")
        return "epub"
    raise Rejected(f"not a PDF or EPUB: starts {data[:15]!r}")


def _download(url: str) -> bytes:
    """`corpus.download`, tried once more after a failure: a run that tries dozens of hosts
    otherwise loses books to one passing DNS or network error."""
    try:
        return corpus.download(url)
    except Exception:
        time.sleep(RETRY_SECONDS)
        return corpus.download(url)


def linked_files(html: str, page: str) -> list[str]:
    """The PDF and EPUB links of an HTML page, absolute, in page order, without repeats."""
    found: list[str] = []
    for href in re.findall(r"""href=["']([^"']+?\.(?:pdf|epub)(?:[?#][^"']*)?)["']""", html, re.I):
        url = urllib.parse.urljoin(page, href)
        if url not in found:
            found.append(url)
    return found


def pick(links: list[str], title: str) -> str | None:
    """The link whose file name shares the most words with the book's title, the first on a tie:
    a book's page often links its chapters too, and the whole book is named like the book."""
    words = set(slug(title).split("-")) - {"a", "an", "the", "of", "and", "to", "in", "for"}

    def shared(url: str) -> int:
        name = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
        return len(words & set(slug(name).split("-")))

    return max(links, key=shared) if links else None


def resolve(candidate: Candidate) -> tuple[bytes, str, str]:
    """The book's bytes, its kind and the URL they came from: the listed link's, or, when that is
    an HTML page, the PDF or EPUB on it named most like the book (`pick`). `Rejected` otherwise."""
    data = _download(candidate.url)
    try:
        return data, kind(data), candidate.url
    except Rejected:
        if not data.lstrip()[:1] == b"<":
            raise
        link = pick(linked_files(data.decode(errors="replace"), candidate.url), candidate.title)
        if link is None:
            raise
    data = _download(link)
    return data, kind(data), link


def collect(
    subject: str, found: list[Candidate], directory: Path, per_collection: int = PER_COLLECTION
) -> tuple[Collection, list[Skipped]]:
    """The first `per_collection` usable books of `found`, written into `directory`, and every
    candidate tried and left out."""
    name = f"open-{slug(subject)}"
    target = directory / name
    target.mkdir(parents=True, exist_ok=True)
    kept: list[Document] = []
    skipped: list[Skipped] = []
    seen: set[str] = set()
    for candidate in found:
        if len(kept) == per_collection:
            break
        if candidate.url in EXCLUDED:
            reason = EXCLUDED[candidate.url]
            skipped.append(Skipped(subject, candidate.title, candidate.url, reason))
            continue
        try:
            data, extension, url = resolve(candidate)
        except Rejected as reason:
            skipped.append(Skipped(subject, candidate.title, candidate.url, str(reason)))
            continue
        except Exception as error:  # a dead link fails in many ways: HTTP, DNS, TLS, timeout
            reason = f"download failed: {type(error).__name__}: {error}"[:200]
            skipped.append(Skipped(subject, candidate.title, candidate.url, reason))
            continue
        digest = corpus.sha256(data)
        if digest in seen:
            skipped.append(
                Skipped(subject, candidate.title, candidate.url, "same file as a kept book")
            )
            continue
        seen.add(digest)
        file = f"{slug(candidate.title)[:80]}.{extension}"
        (target / file).write_bytes(data)
        page = candidate.url if url != candidate.url else ""
        kept.append(Document(file, candidate.title, url, digest, len(data), page))
        print(f"  kept {file} ({len(data) / 1e6:.1f} MB)", flush=True)
    return Collection(name, subject, kept), skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", action="append", help=f"default: {', '.join(PILOT)}")
    parser.add_argument("--per-collection", type=int, default=PER_COLLECTION)
    parser.add_argument("--manifest", type=Path, default=corpus.MANIFEST)
    args = parser.parse_args(argv)
    listed = subjects(corpus.download(LIST_URL).decode())
    unknown = [s for s in args.subject or PILOT if s not in listed]
    if unknown:
        print(
            f"no such subject: {', '.join(unknown)}; listed: {', '.join(listed)}", file=sys.stderr
        )
        return 1
    asked = args.subject or list(PILOT)
    previous = corpus.load(args.manifest) if args.manifest.exists() else Manifest(LIST_URL, [])
    collections = [c for c in previous.collections if c.subject not in asked]
    skipped = [s for s in previous.skipped if s.subject not in asked]
    for subject in asked:
        found = candidates(subject, listed[subject])
        print(f"{subject}: {len(found)} candidates", flush=True)
        collection, left = collect(subject, found, corpus.DIRECTORY, args.per_collection)
        collections.append(collection)
        skipped += left
        print(f"  {len(collection.documents)} kept, {len(left)} left out", flush=True)
    collections.sort(key=lambda c: c.name)
    corpus.dump(Manifest(LIST_URL, collections, skipped), args.manifest)
    print(f"written to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
