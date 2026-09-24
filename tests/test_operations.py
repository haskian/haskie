"""`operations.fold_operations`: a page of pipeline runs read as operations with their jobs."""

import msgspec
import pytest

from haskie.indexing import operations
from haskie.indexing.dbos_names import RunStatus
from haskie.indexing.workflows import PipelineAction

TAIL = "c3680a02207a41f89078486d1b3a4c90"
DOC = "principles.pdf"


def run(
    action: operations.PipelineAction, status: RunStatus = RunStatus.SUCCESS, **patch
) -> operations._StageRun:
    prefix = {"import": "imp", "embed": "emb", "index": "idx-col:asd"}[action]
    base = operations._StageRun(
        id=f"{prefix}:{DOC}:{TAIL}",
        action=action,
        collection="asd" if action == "index" else None,
        document=DOC,
        status=status,
        created_at=1_000.0,
        updated_at=1_009.0,
        error=None,
        tasks_done=27,
        tasks_running=0,
        tasks_total=27,
    )
    return msgspec.structs.replace(base, **patch)


def shape(rows: list[operations.Operation]) -> list[tuple]:
    return [
        (
            row.id.split(":")[0],
            [(j.stage, j.id.split(":")[0], j.status, j.tasks_total) for j in row.jobs],
        )
        for row in rows
    ]


@pytest.mark.parametrize(
    ("name", "page", "expected"),
    [
        ("nothing", [], []),
        (
            "an import and its embed become one operation of two jobs, convert first",
            [run(PipelineAction.EMBED), run(PipelineAction.IMPORT)],
            [("imp", [("convert", "imp", "SUCCESS", 27), ("embed", "emb", "SUCCESS", 27)])],
        ),
        (
            "an index and its embed: embed first, index last",
            [
                run(PipelineAction.EMBED, tasks_total=0, tasks_done=0),
                run(PipelineAction.INDEX, tasks_total=2, tasks_done=2),
            ],
            [("idx-col", [("embed", "emb", "SUCCESS", 0), ("index", "idx-col", "SUCCESS", 2)])],
        ),
        (
            "an embed whose parent is not on the page keeps an operation of its own",
            [run(PipelineAction.EMBED)],
            [("emb", [("embed", "emb", "SUCCESS", 27)])],
        ),
        (
            "an import without its embed on the page has one job",
            [run(PipelineAction.IMPORT, status=RunStatus.PENDING)],
            [("imp", [("convert", "imp", "PENDING", 27)])],
        ),
        (
            "a running import whose embed exists has converted already",
            [
                run(PipelineAction.EMBED, status=RunStatus.PENDING, tasks_done=3),
                run(PipelineAction.IMPORT, status=RunStatus.PENDING),
            ],
            [("imp", [("convert", "imp", "SUCCESS", 27), ("embed", "emb", "PENDING", 27)])],
        ),
        (
            "a running index waits for its embed, then writes",
            [
                run(PipelineAction.EMBED, status=RunStatus.PENDING),
                run(PipelineAction.INDEX, status=RunStatus.PENDING, tasks_total=2, tasks_done=0),
            ],
            [("idx-col", [("embed", "emb", "PENDING", 27), ("index", "idx-col", "ENQUEUED", 2)])],
        ),
        (
            "a running index whose embed is done is writing",
            [
                run(PipelineAction.EMBED),
                run(PipelineAction.INDEX, status=RunStatus.PENDING, tasks_total=2, tasks_done=1),
            ],
            [("idx-col", [("embed", "emb", "SUCCESS", 27), ("index", "idx-col", "PENDING", 2)])],
        ),
        (
            "a failed import whose embed failed: convert was over, the embed carries the error",
            [
                run(PipelineAction.EMBED, status=RunStatus.ERROR, error="model gone"),
                run(PipelineAction.IMPORT, status=RunStatus.ERROR, error="model gone"),
            ],
            [("imp", [("convert", "imp", "SUCCESS", 27), ("embed", "emb", "ERROR", 27)])],
        ),
        (
            "another document's embed is not this operation's",
            [
                run(PipelineAction.EMBED, id=f"emb:other.pdf:{TAIL}", document="other.pdf"),
                run(PipelineAction.IMPORT),
            ],
            [
                ("emb", [("embed", "emb", "SUCCESS", 27)]),
                ("imp", [("convert", "imp", "SUCCESS", 27)]),
            ],
        ),
    ],
)
def test_operations(name: str, page: list, expected: list[tuple]) -> None:
    assert shape(operations.fold_operations(page)) == expected, name


def test_a_job_declared_done_carries_no_error() -> None:
    (row,) = operations.fold_operations(
        [
            run(PipelineAction.EMBED, status=RunStatus.ERROR, error="model gone"),
            run(PipelineAction.IMPORT, status=RunStatus.ERROR, error="model gone"),
        ]
    )
    convert, embed = row.jobs
    assert (convert.status, convert.error) == ("SUCCESS", None), "the convert stage was over"
    assert (embed.status, embed.error) == ("ERROR", "model gone")
    assert row.error == "model gone", "the operation still reports what failed"


def test_operations_sum_the_counters() -> None:
    (row,) = operations.fold_operations(
        [
            run(PipelineAction.EMBED, tasks_done=5, tasks_total=27, tasks_running=1),
            run(PipelineAction.IMPORT, tasks_done=3, tasks_total=3),
        ]
    )
    assert row.detail == {"tasks_done": 8, "tasks_running": 1, "tasks_total": 30}
    assert row.status == "SUCCESS" and row.title == DOC


@pytest.mark.parametrize(
    ("name", "page", "expected"),
    [
        (
            "an import converts until it spawns its embed, which runs on its own clock",
            [
                run(PipelineAction.EMBED, created_at=1_003.0, updated_at=1_020.0),
                run(PipelineAction.IMPORT, updated_at=1_021.0),
            ],
            [3.0, 17.0],
        ),
        (
            "an index waits for its embed, then writes",
            [
                run(PipelineAction.EMBED, created_at=1_001.0, updated_at=1_004.0),
                run(PipelineAction.INDEX, updated_at=1_009.0),
            ],
            [3.0, 5.0],
        ),
        ("a job on its own is its whole run", [run(PipelineAction.EMBED)], [9.0]),
        (
            "a running job has no duration yet",
            [
                run(PipelineAction.EMBED, status=RunStatus.PENDING),
                run(PipelineAction.IMPORT, status=RunStatus.PENDING),
            ],
            [0.0, None],
        ),
        (
            "a clock that went backwards reads as zero, never negative",
            [
                run(PipelineAction.EMBED, created_at=999.0, updated_at=1_009.0),
                run(PipelineAction.IMPORT),
            ],
            [0.0, 10.0],
        ),
    ],
)
def test_job_seconds(name: str, page: list, expected: list[float | None]) -> None:
    (row,) = operations.fold_operations(page)
    assert [j.seconds for j in row.jobs] == expected, name
