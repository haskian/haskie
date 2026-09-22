Write `mlfq.py` in the current directory, implementing the priority rules of the multi-level
feedback queue exactly as the OSTEP chapter "Scheduling: The Multi-Level Feedback Queue" finally
states them — the chapter revises its own rules partway through, so use the set it ends on.
Standard library only.

Priorities are integers where 0 is the topmost (highest-priority) queue and `levels - 1` is the
bottom. Define exactly these names:

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

Follow the chapter's final rules rather than the version you would write from memory, and say
which rule each branch implements. If you cannot confirm the rules against a source, write your
best attempt rather than stopping, and mark it `# unconfirmed`. Write the file in this session
either way; do not stop to ask.
