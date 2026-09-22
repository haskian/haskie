"""The coding corpus: open-licensed books and chapters, fetched and imported into a collection.

Public documents on purpose. The web reaches them too, and that is what keeps the comparison
honest: the question is not whether the library holds something the internet does not, it is how
many steps and how much money each route costs to reach the same passage. A corpus the baseline
could not reach would answer a question nobody asked.

Chapters rather than whole books, because a chapter is one topic and a retrieval hit that names
it says something. Standard library only - this runs against a server over HTTP, and a download
script is not worth a dependency.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import msgspec

COLLECTION = "coding-books"
DESCRIPTION = (
    "Open-licensed programming books and chapters: OS internals (scheduling, virtual memory, "
    "concurrency, file systems), JavaScript, and Git."
)
API = "http://127.0.0.1:8000"
OSTEP = "https://pages.cs.wisc.edu/~remzi/OSTEP/"
POLL_SECONDS = 5.0


class Source(msgspec.Struct):
    name: str
    url: str


SOURCES = (
    Source("ostep-cpu-sched-mlfq.pdf", f"{OSTEP}cpu-sched-mlfq.pdf"),
    Source("ostep-cpu-sched-lottery.pdf", f"{OSTEP}cpu-sched-lottery.pdf"),
    Source("ostep-vm-paging.pdf", f"{OSTEP}vm-paging.pdf"),
    Source("ostep-vm-tlbs.pdf", f"{OSTEP}vm-tlbs.pdf"),
    Source("ostep-threads-locks.pdf", f"{OSTEP}threads-locks.pdf"),
    Source("ostep-threads-cv.pdf", f"{OSTEP}threads-cv.pdf"),
    Source("ostep-threads-sema.pdf", f"{OSTEP}threads-sema.pdf"),
    Source("ostep-file-implementation.pdf", f"{OSTEP}file-implementation.pdf"),
    Source("ostep-file-ffs.pdf", f"{OSTEP}file-ffs.pdf"),
    Source("ostep-file-journaling.pdf", f"{OSTEP}file-journaling.pdf"),
    Source(
        "little-book-of-semaphores.pdf",
        "https://greenteapress.com/semaphores/LittleBookOfSemaphores.pdf",
    ),
    Source("eloquent-javascript.pdf", "https://eloquentjavascript.net/Eloquent_JavaScript.pdf"),
    Source(
        "pro-git.pdf",
        "https://github.com/progit/progit2/releases/download/2.1.443/progit.pdf",
    ),
)


def fetch(destination: Path, sources: tuple[Source, ...] = SOURCES) -> list[Path]:
    """Download whatever is missing and return every file. Already-present files are left alone,
    so re-running costs nothing and a half-finished download is retried by deleting it."""
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for source in sources:
        target = destination / source.name
        if not target.exists():
            print(f"fetching {source.name}", flush=True)
            with urllib.request.urlopen(source.url, timeout=120) as response:  # noqa: S310
                target.write_bytes(response.read())
        files.append(target)
    return files


def call(method: str, path: str, body: dict | None = None, api: str = API) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(  # noqa: S310
        f"{api}{path}", data=data, method=method, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        return json.loads(response.read() or b"{}")


def create(collection: str = COLLECTION, api: str = API) -> None:
    """Creating a collection that is already there is not a failure; nothing else here may
    swallow a conflict, because a refused attach looks exactly the same and is one."""
    try:
        call("POST", "/api/collections", {"name": collection, "description": DESCRIPTION}, api)
    except urllib.error.HTTPError as error:
        if error.code != 409:
            raise


def import_all(files: list[Path], api: str = API) -> list[str]:
    """Import each file, or adopt the row that is already there.

    A path import is refused with a conflict once the document exists, and for a corpus that is
    the state we wanted: the task has to be re-runnable without deleting the library first.
    """
    names = []
    for file in files:
        try:
            row = call("POST", "/api/documents/import", {"path": str(file.resolve())}, api)
        except urllib.error.HTTPError as error:
            if error.code != 409:
                raise
            row = call("GET", f"/api/documents/{file.name}", api=api)
            print(f"have {row['name']}", flush=True)
        else:
            print(f"importing {row['name']}", flush=True)
        names.append(row["name"])
    return names


def await_status(names: list[str], wanted: str, api: str, limit: float) -> list[str]:
    """Poll until every document has reached `wanted`, and return the ones that did.

    Import is a background pipeline: convert, then embed. Attaching a document that has not come
    out of it yet is refused, which is how the first run of this script quietly created an empty
    collection.
    """
    deadline = time.monotonic() + limit
    pending = list(names)
    ready, failed = [], []
    while pending and time.monotonic() < deadline:
        rows = [call("GET", f"/api/documents/{name}", api=api) for name in pending]
        ready += [r["name"] for r in rows if r["status"] == wanted]
        failed += [r["name"] for r in rows if r["status"] == "error"]
        pending = [r["name"] for r in rows if r["status"] not in (wanted, "error")]
        print(f"{wanted} {len(ready)}/{len(names)}, {len(failed)} failed", flush=True)
        if pending:
            time.sleep(POLL_SECONDS)
    for name in failed:
        print(f"skipping {name}: import failed", flush=True)
    return ready


def attach_all(names: list[str], collection: str = COLLECTION, api: str = API) -> None:
    for name in names:
        call("POST", f"/api/collections/{collection}/documents", {"document": name}, api)
        print(f"attached {name}", flush=True)


def indexed(
    expected: int, collection: str = COLLECTION, api: str = API, limit: float = 3600
) -> bool:
    """Poll until every member is indexed: an attach is accepted, not done."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        counts = call("GET", f"/api/collections/{collection}", api=api).get("counts", {})
        done, errors = counts.get("indexed", 0), counts.get("error", 0)
        print(f"indexed {done}/{expected}, {errors} failed", flush=True)
        if done + errors >= expected:
            return errors == 0
        time.sleep(POLL_SECONDS)
    return False


def main() -> int:
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("library/coding-books")
    files = fetch(destination)
    create()
    ready = await_status(import_all(files), "imported", API, limit=3600)
    attach_all(ready)
    return 0 if indexed(len(ready)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
