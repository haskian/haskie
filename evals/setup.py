"""Idempotent setup for the eval corpus: download once, import once, index once. Re-running this
is always safe - it converges to the same end state rather than duplicating or erroring.

Idempotent means checking first, not writing and catching the conflict: every write below is
preceded by a read that decides whether the write is needed at all, so a re-run's server log
reads the same as a first run's - no 409s logged and swallowed, because none are ever sent.

Starting the isolated haskie instance itself is `mise run eval:setup`'s job (it shells out to
`haskie ensure --home ...`), not this module's - this file only talks to whatever server `api`
points at, over HTTP, and never touches the filesystem of a haskie home directly. That the
instance it talks to has its own `--home` and its own port, separate from any real personal
collection, is a hard requirement on the caller, not something enforced here.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"  # downloaded PDFs live here, never under the project's own library/
DEFAULT_HOME = ROOT / ".haskie-eval"
DEFAULT_API = "http://127.0.0.1:8123"
COLLECTION = "eval-programming-books"
DESCRIPTION = "Open-source books used to evaluate whether an agent reaches for haskie search."
POLL_SECONDS = 3.0


@dataclass(frozen=True)
class Source:
    name: str
    url: str


# One book per task, "various programming topics" per the original brief: OS scheduling,
# concurrency, and version control. Extend this list to grow past the first three tasks.
SOURCES = (
    Source("ostep-cpu-sched-mlfq.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-sched-mlfq.pdf"),
    Source(
        "little-book-of-semaphores.pdf",
        "https://greenteapress.com/semaphores/LittleBookOfSemaphores.pdf",
    ),
    Source("pro-git.pdf", "https://github.com/progit/progit2/releases/download/2.1.443/progit.pdf"),
)


def fetch(sources: tuple[Source, ...] = SOURCES, directory: Path = CORPUS_DIR) -> list[Path]:
    """Download whatever is missing. A file already on disk is left alone, so a partial or
    completed download costs nothing to re-run."""
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for source in sources:
        target = directory / source.name
        if not target.exists():
            print(f"fetching {source.name}", flush=True)
            with urllib.request.urlopen(source.url, timeout=120) as response:  # noqa: S310
                target.write_bytes(response.read())
        files.append(target)
    return files


def call(method: str, path: str, api: str, body: dict | None = None) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(  # noqa: S310
        f"{api.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.loads(response.read() or b"{}")


def get_or_none(path: str, api: str) -> Any | None:
    """The parsed body of `GET path`, or `None` if the server says it does not exist yet.

    This is what makes every write below idempotent by checking rather than by catching a
    conflict: a 404 here is an ordinary, expected answer, not an error path."""
    try:
        return call("GET", path, api)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise RuntimeError(f"GET {path} -> {error.code}: {error.read().decode()[:300]}") from error


def ensure_collection(collection: str, description: str, api: str) -> None:
    if get_or_none(f"/api/collections/{collection}", api) is None:
        call("POST", "/api/collections", api, {"name": collection, "description": description})


def import_all(files: list[Path], api: str) -> list[str]:
    """The name of each file once imported - already there, or freshly imported."""
    names = []
    for file in files:
        existing = get_or_none(f"/api/documents/{file.name}", api)
        if existing is not None:
            names.append(existing["name"])
            continue
        row = call("POST", "/api/documents/import", api, {"path": str(file.resolve())})
        names.append(row["name"])
    return names


def await_status(names: list[str], wanted: str, api: str, limit: float = 1800) -> list[str]:
    deadline = time.monotonic() + limit
    pending = list(dict.fromkeys(names))
    ready: list[str] = []
    while pending and time.monotonic() < deadline:
        rows = [call("GET", f"/api/documents/{name}", api) for name in pending]
        ready += [r["name"] for r in rows if r.get("status") == wanted]
        failed = [r["name"] for r in rows if r.get("status") == "error"]
        for name in failed:
            print(f"import failed: {name}", flush=True)
        pending = [r["name"] for r in rows if r.get("status") not in (wanted, "error")]
        if pending:
            time.sleep(POLL_SECONDS)
    return ready


def attach_all(names: list[str], collection: str, api: str) -> None:
    for name in names:
        held_by = call("GET", f"/api/documents/{name}/collections", api)
        if collection not in held_by:
            call("POST", f"/api/collections/{collection}/documents", api, {"document": name})


def await_indexed(expected: int, collection: str, api: str, limit: float = 1800) -> bool:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        counts = call("GET", f"/api/collections/{collection}", api).get("counts", {})
        done, errors = counts.get("indexed", 0), counts.get("error", 0)
        if done + errors >= expected:
            return errors == 0
        time.sleep(POLL_SECONDS)
    return False


def main(api: str = DEFAULT_API, collection: str = COLLECTION) -> int:
    files = fetch()
    ensure_collection(collection, DESCRIPTION, api)
    names = import_all(files, api)
    ready = await_status(names, "imported", api)
    attach_all(ready, collection, api)
    ok = await_indexed(len(ready), collection, api)
    print(f"{collection}: {len(ready)}/{len(files)} imported and indexed")
    return 0 if ok and len(ready) == len(files) else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(*sys.argv[1:]))
