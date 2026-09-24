"""Typed errors: the contract between the domain and the HTTP layer.

Every message here is written for the user, so it must not carry absolute paths; `home.scrub`
removes the ones that slip in from library exceptions.
"""

from typing import ClassVar


class HaskieError(Exception):
    """Base for expected failures; the message is safe to show the user (paths scrubbed).

    `status_code` and `headers` are part of the contract, so they live with the error rather than
    in a map the HTTP layer keeps in step by hand.
    """

    status_code: ClassVar[int] = 400
    headers: ClassVar[dict[str, str]] = {}


class NotFound(HaskieError):
    status_code = 404


class Conflict(HaskieError):
    status_code = 409


class InvalidInput(HaskieError, ValueError):
    """Also a ValueError: msgspec only turns a ValueError raised in `__post_init__` into a
    decode-time `ValidationError`, and callers that predate this module catch ValueError."""

    status_code = 422


class NotReady(HaskieError):
    """Not ready yet: a model is still loading, or every preview builder is busy."""

    status_code = 503
    headers = {"Retry-After": "2"}


class PermanentError(HaskieError):
    """Pipeline failure that must not be retried: the file cannot be processed as it is."""

    status_code = 422
