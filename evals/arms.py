"""The four configurations under test.

A and D answer "is haskie worth it"; B, C and D answer the question that actually decides where
the effort goes - whether the tools alone are enough, what the skill adds, and whether a
user-level rule adds anything on top of it.

The skill is the one the product installs, rendered from the collections a home holds, rather
than a copy written for the eval: an arm that tests an invented skill measures the invention.
"""

import json
import urllib.request

import msgspec

from evals.corpus import API
from evals.runner import Arm
from haskie import claude
from haskie.collection import CollectionSummary

# The candidate user-level rule for arm D targets the observed retrieval-path failure without
# forbidding legitimate filesystem inspection after discovery.
MEMORY = """\
# Working with the user's documents

When the user's document collections can answer a question, use haskie as the retrieval interface.
Start with `search_text` when you do not already know the relevant document. Do not use Grep, Read,
or Bash on the haskie document store as the first knowledge lookup. If a haskie result identifies a
document, prefer `document_passages` for the relevant passage; use the backing Markdown file only
when you genuinely need information that the haskie result cannot provide. Do not search merely
because a document happens to contain the same topic: for ordinary programming tasks that do not
require the user's sources, answer directly. When an answer depends on a user source, name the
document you used.
"""


def summaries(api: str = API) -> list[CollectionSummary]:
    with urllib.request.urlopen(f"{api}/api/collections?page_size=1000", timeout=60) as response:  # noqa: S310
        page = json.loads(response.read())
    return [msgspec.convert(row, CollectionSummary) for row in page["items"]]


def arms(api: str = API) -> tuple[Arm, ...]:
    skill = claude.render_skill(summaries(api))
    return (
        Arm(name="a-no-library", mcp=False),
        Arm(name="b-mcp-only", mcp=True),
        Arm(name="c-mcp-skill", mcp=True, skill=skill),
        Arm(name="d-mcp-skill-memory", mcp=True, skill=skill, memory=MEMORY, memory_scope="user"),
    )
