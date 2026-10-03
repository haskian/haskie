"""Scored against `github_api_breaking_changes.py` as the agent wrote it.

Grounded in GitHub's own REST API breaking-changes changelog
(github-api-breaking-changes.md), specifically its `2026-03-10` section - GitHub's first
calendar API version to ship breaking changes, released in March 2026, after this suite's
model's training cutoff. Each fact below directly reverses a years-old, heavily-documented API
shape, so a guess made without searching doesn't just risk being uninformed - it's biased toward
confidently reproducing the *old*, now-wrong value.
"""

import github_api_breaking_changes
import pytest


@pytest.mark.discriminating
def test_submodule_entries_now_report_their_own_type() -> None:
    """Before this version, a submodule listed by the contents API had `type: "file"` - the
    shape most git/GitHub tooling and documentation still describes. 2026-03-10 gives it its own
    type instead."""
    assert github_api_breaking_changes.submodule_content_type() == "submodule"


@pytest.mark.discriminating
def test_installation_deletion_is_now_accepted_not_immediately_done() -> None:
    """The endpoint used to return 204 No Content, meaning "already done." 2026-03-10 moves the
    deletion to the background and returns 202 Accepted instead."""
    assert github_api_breaking_changes.installation_deletion_status() == 202


@pytest.mark.discriminating
def test_javascript_and_typescript_share_one_enum_value_now() -> None:
    """Before this version, the `languages` enum listed "javascript" and "typescript"
    separately. 2026-03-10 merges them into one value, reflecting that CodeQL always analyzed
    them together."""
    assert github_api_breaking_changes.code_scanning_combined_language() == "javascript-typescript"
