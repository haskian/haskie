"""A pytest plugin that writes each test's node id, outcome and marker to one JSON file.

Scoring a task means running its tests against whatever the agent wrote, and the summary line
pytest prints is not something to parse. `EVAL_REPORT` names the file to write; without it the
plugin does nothing, so a task's tests still run by hand the ordinary way.
"""

import json
import os
from pathlib import Path
from typing import Any

KEY = "EVAL_REPORT"
DISCRIMINATING = "discriminating"
_MARKER_HELP = (
    f"{DISCRIMINATING}: the assertion that separates having read the source from remembering the "
    "gist. Reported apart from the rest, because that is the number the eval is after."
)


# Module level rather than on the config: a `TestReport` carries no way back to it, and one
# pytest process is one scoring run.
_OUTCOMES: dict[str, dict[str, Any]] = {}


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", _MARKER_HELP)
    _OUTCOMES.clear()


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    for item in items:
        _OUTCOMES[item.nodeid] = {
            "outcome": "notrun",  # a collection error leaves the test never reported on
            DISCRIMINATING: item.get_closest_marker(DISCRIMINATING) is not None,
        }


def pytest_runtest_logreport(report: Any) -> None:
    known = _OUTCOMES.get(report.nodeid)
    if known is None:
        return
    # A failure in setup is a failure of the task, not a missing result, so the first non-passing
    # phase wins and a later phase does not overwrite it.
    if report.when == "call" or report.outcome != "passed":
        if known["outcome"] in ("notrun", "passed"):
            known["outcome"] = report.outcome


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    destination = os.environ.get(KEY)
    if destination:
        Path(destination).write_text(json.dumps(_OUTCOMES), encoding="utf-8")
