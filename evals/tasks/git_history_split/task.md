Write `git_history_split.py` in the current directory. Standard library only. Do not run Git or
inspect a repository - these functions only build argument lists for a Git command to be run
later, elsewhere.

`git history split` is an experimental Git subcommand, added in 2026, that interactively splits a
commit into two: hunks you choose become a new commit that becomes the *parent* of the original
commit, which keeps whatever hunks weren't selected. Each function returns a complete argument
list, including the Git subcommand itself as the first two elements (what you would pass to
`subprocess.run(["git", *args])`):

    def split_commit(commit: str) -> list[str]:
        """Split `commit` apart, using every default."""

    def split_commit_dry_run(commit: str) -> list[str]:
        """Same split, but preview the reference updates instead of making them."""

    def split_commit_head_only(commit: str) -> list[str]:
        """Same split, but rewrite only HEAD rather than every reference this command would
        otherwise update by default."""

    def split_commit_limited_to(commit: str, paths: list[str]) -> list[str]:
        """Same split, limited to changes touching the given paths."""

Preserve revision strings and paths exactly.
