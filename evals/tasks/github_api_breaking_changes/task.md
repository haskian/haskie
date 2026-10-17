Write `github_api_breaking_changes.py` in the current directory. Standard library only. Do not
make any network requests - report only the documented facts below as literal values.

GitHub's REST API uses calendar versioning: each version is a date, sent as the
`X-GitHub-Api-Version` header, and a new version can ship breaking changes relative to the one
before it. Version `2026-03-10` is one such version. For each function, report the value that
version documents - not whatever an older version of the API used to return, which is very
likely what you already "know" about these endpoints from before this version existed.

    def submodule_content_type() -> str:
        """Under API version 2026-03-10, the `type` value the "Get repository content" endpoint
        (`GET /repos/{owner}/{repo}/contents/{path}`) reports for an entry that is a
        submodule."""

    def installation_deletion_status() -> int:
        """Under API version 2026-03-10, the HTTP status code `DELETE
        /app/installations/{installation_id}` returns on success."""

    def code_scanning_combined_language() -> str:
        """Under API version 2026-03-10, the single value in the `languages` enum of code
        scanning's default-setup response that now covers what used to be two separate values,
        for JavaScript and TypeScript."""
