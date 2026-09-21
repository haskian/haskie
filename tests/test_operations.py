"""`jobs.fold_operations`: a page of pipeline jobs read as operations with stages."""

import msgspec
import pytest

from haskie import jobs

TAIL = "c3680a02207a41f89078486d1b3a4c90"
DOC = "principles.pdf"


def job(action: jobs.JobAction, status: str = "SUCCESS", **patch) -> jobs.Job:
    prefix = {"import": "imp", "embed": "emb", "index": "idx-col:asd"}[action]
    base = jobs.Job(
        id=f"{prefix}:{DOC}:{TAIL}",
        action=action,
        collection="asd" if action == "index" else None,
        doc=DOC,
        status=status,
        created_at=1_000.0,
        updated_at=1_009.0,
        error=None,
        tasks_done=27,
        tasks_running=0,
        tasks_total=27,
    )
    return msgspec.structs.replace(base, **patch)


def shape(rows: list[jobs.JobRow]) -> list[tuple]:
    return [
        (
            row.id.split(":")[0],
            [(s.stage, s.job_id.split(":")[0], s.status, s.tasks_total) for s in row.stages],
        )
        for row in rows
    ]


@pytest.mark.parametrize(
    ("name", "page", "expected"),
    [
        ("nothing", [], []),
        (
            "an import and its embed become one operation of two stages, convert first",
            [job("embed"), job("import")],
            [("imp", [("convert", "imp", "SUCCESS", 27), ("embed", "emb", "SUCCESS", 27)])],
        ),
        (
            "an index and its embed: embed first, index last",
            [job("embed", tasks_total=0, tasks_done=0), job("index", tasks_total=2, tasks_done=2)],
            [("idx-col", [("embed", "emb", "SUCCESS", 0), ("index", "idx-col", "SUCCESS", 2)])],
        ),
        (
            "an embed whose parent is not on the page keeps a row of its own",
            [job("embed")],
            [("emb", [("embed", "emb", "SUCCESS", 27)])],
        ),
        (
            "an import without its embed on the page has one stage",
            [job("import", status="PENDING")],
            [("imp", [("convert", "imp", "PENDING", 27)])],
        ),
        (
            "a running import whose embed exists has converted already",
            [job("embed", status="PENDING", tasks_done=3), job("import", status="PENDING")],
            [("imp", [("convert", "imp", "SUCCESS", 27), ("embed", "emb", "PENDING", 27)])],
        ),
        (
            "a running index waits for its embed, then writes",
            [
                job("embed", status="PENDING"),
                job("index", status="PENDING", tasks_total=2, tasks_done=0),
            ],
            [("idx-col", [("embed", "emb", "PENDING", 27), ("index", "idx-col", "ENQUEUED", 2)])],
        ),
        (
            "a running index whose embed is done is writing",
            [job("embed"), job("index", status="PENDING", tasks_total=2, tasks_done=1)],
            [("idx-col", [("embed", "emb", "SUCCESS", 27), ("index", "idx-col", "PENDING", 2)])],
        ),
        (
            "a failed import whose embed failed: convert was over, the embed carries the error",
            [
                job("embed", status="ERROR", error="model gone"),
                job("import", status="ERROR", error="model gone"),
            ],
            [("imp", [("convert", "imp", "SUCCESS", 27), ("embed", "emb", "ERROR", 27)])],
        ),
        (
            "another document's embed is not this operation's",
            [job("embed", id=f"emb:other.pdf:{TAIL}", doc="other.pdf"), job("import")],
            [
                ("emb", [("embed", "emb", "SUCCESS", 27)]),
                ("imp", [("convert", "imp", "SUCCESS", 27)]),
            ],
        ),
    ],
)
def test_fold_operations(name: str, page: list[jobs.Job], expected: list[tuple]) -> None:
    assert shape(jobs.fold_operations(page)) == expected, name


def test_fold_operations_sums_the_counters() -> None:
    (row,) = jobs.fold_operations(
        [
            job("embed", tasks_done=5, tasks_total=27, tasks_running=1),
            job("import", tasks_done=3, tasks_total=3),
        ]
    )
    assert row.detail == {"tasks_done": 8, "tasks_running": 1, "tasks_total": 30}
    assert row.status == "SUCCESS" and row.title == f"import {DOC}"


@pytest.mark.parametrize(
    ("name", "page", "expected"),
    [
        (
            "an import converts until it spawns its embed, which runs on its own clock",
            [
                job("embed", created_at=1_003.0, updated_at=1_020.0),
                job("import", updated_at=1_021.0),
            ],
            [3.0, 17.0],
        ),
        (
            "an index waits for its embed, then writes",
            [
                job("embed", created_at=1_001.0, updated_at=1_004.0),
                job("index", updated_at=1_009.0),
            ],
            [3.0, 5.0],
        ),
        ("a stage on its own is its whole workflow", [job("embed")], [9.0]),
        (
            "a running stage has no duration yet",
            [job("embed", status="PENDING"), job("import", status="PENDING")],
            [0.0, None],
        ),
        (
            "a clock that went backwards reads as zero, never negative",
            [job("embed", created_at=999.0, updated_at=1_009.0), job("import")],
            [0.0, 10.0],
        ),
    ],
)
def test_stage_seconds(name: str, page: list[jobs.Job], expected: list[float | None]) -> None:
    (row,) = jobs.fold_operations(page)
    assert [s.seconds for s in row.stages] == expected, name
