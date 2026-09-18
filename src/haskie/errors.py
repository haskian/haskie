"""Typed errors: the contract between the domain and the HTTP layer.

Every message here is written for the user, so it must not carry absolute paths; `scrub` removes
the ones that slip in from library exceptions.
"""

from pathlib import Path
from typing import ClassVar

from haskie import home


class HaskieError(Exception):
    """Base for expected failures; the message is safe to show the user (paths scrubbed).

    `status_code` is part of the contract, so it lives with the error rather than in a map the
    HTTP layer keeps in step by hand.
    """

    status_code: ClassVar[int] = 400


class NotFound(HaskieError):
    status_code = 404


class LibraryNotFound(NotFound):
    pass


class DocumentNotFound(NotFound):
    pass


class JobNotFound(NotFound):
    pass


class Conflict(HaskieError):
    status_code = 409


class InvalidInput(HaskieError, ValueError):
    """Also a ValueError: msgspec only turns a ValueError raised in `__post_init__` into a
    decode-time `ValidationError`, and callers that predate this module catch ValueError."""

    status_code = 422


class NotReady(HaskieError):
    """A model is still loading."""

    status_code = 503


class PermanentError(HaskieError):
    """Pipeline failure that must not be retried: the file cannot be processed as it is."""

    status_code = 422


class UnsupportedFileType(PermanentError):
    pass


class NeedsOcr(PermanentError):
    pass


class ConversionError(PermanentError):
    pass


def scrub(text: str) -> str:
    """Replace absolute paths with their symbolic root, for log lines and error bodies.
    The home directory is replaced first because it usually sits inside the user directory."""
    return text.replace(str(home.HOME), "$HASKIE_HOME").replace(str(Path.home()), "~")
