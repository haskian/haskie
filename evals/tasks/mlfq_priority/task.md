Write `mlfq.py` in the current directory. Standard library only.

Implement the priority-adjustment rule of a multi-level feedback queue scheduler, exactly as the
final version in "Scheduling: The Multi-Level Feedback Queue" states it. Priorities are integers,
where 0 is the topmost (highest-priority) queue and `levels - 1` is the bottom.

    def on_enter(levels: int) -> int:
        """The priority a job is given when it enters the system."""

    def after_run(
        priority: int, used_at_level: float, allotment: float, gave_up_cpu: bool, levels: int
    ) -> tuple[int, float]:
        """The job's priority and its time counted at that level, once it stops running.

        `used_at_level` is the total time the job has spent at its current priority, including
        the run that just finished. `gave_up_cpu` is whether it stopped by blocking (an I/O, say)
        rather than by being preempted. Returns `(priority, used_at_level)` going forward."""

    def boost(priorities: list[int]) -> list[int]:
        """Every job's priority after the periodic boost."""

The chapter revises its own rule partway through - use the version it ends on, not an earlier
draft of it. Say which rule each branch of your code implements.
