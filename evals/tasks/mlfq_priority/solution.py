"""Reference for `mlfq_priority`, from ostep-cpu-sched-mlfq.pdf p.10 (the final rule set)."""


def on_enter(levels: int) -> int:
    """Rule 3: a job entering the system starts at the topmost queue."""
    return 0


def after_run(
    priority: int, used_at_level: float, allotment: float, gave_up_cpu: bool, levels: int
) -> tuple[int, float]:
    """Rule 4: once a job uses up its allotment at a level, regardless of how many times it has
    given up the CPU, its priority is reduced. `gave_up_cpu` therefore changes nothing; the
    chapter's earlier Rule 4b, which reset the allotment on an I/O, is what this replaces."""
    if used_at_level >= allotment:
        return min(priority + 1, levels - 1), 0.0
    return priority, used_at_level


def boost(priorities: list[int]) -> list[int]:
    """Rule 5: after some time period S, every job moves to the topmost queue."""
    return [0 for _ in priorities]
