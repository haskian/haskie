Write `revision_ranges.py` in the current directory. Standard library only. Do not run Git or
inspect a repository - these functions only build argument lists for a Git command to be run
later, elsewhere.

Each function returns a complete argument list, including the Git subcommand itself as the first
element (what you would pass to `subprocess.run(["git", *args])`), from Pro Git's "Commit
Ranges" material:

    def commits_in_two_dot(base: str, tip: str) -> list[str]:
        """`git log` arguments for the double-dot range: commits reachable from `tip` that are
        not reachable from `base`."""

    def commits_in_three_dot(left: str, right: str) -> list[str]:
        """`git log` arguments for the triple-dot range: commits reachable from either `left` or
        `right`, but not both - the symmetric difference."""

    def diff_from_merge_base(left: str, right: str) -> list[str]:
        """`git diff` arguments using triple-dot notation: the changes on `right` since its
        common ancestor with `left`."""

    def commits_reachable_excluding(include: list[str], exclude: list[str]) -> list[str]:
        """`git log` arguments for Pro Git's "Multiple Points" syntax: commits reachable from
        any ref in `include`, excluding any reachable from any ref in `exclude`, using `^ref`
        rather than `--not`. `include` first, in order, then the excluded refs."""

Preserve revision strings exactly, including expressions like `HEAD~2`.
