"""Request and response shapes shared by more than one feature module."""

from typing import Annotated

import msgspec
from litestar.params import Parameter


class BulkStarted(msgspec.Struct):
    """A job was accepted and runs in the background; follow it at /api/jobs/{job_id}/progress."""

    job_id: str


# A search returns at least one result or none at all; the bound rides along into the schema.
Limit = Annotated[int | None, Parameter(ge=1)]
