"""Structured logging: one JSON line per event on stdout, for our code and for the libraries.

structlog renders our events; a `ProcessorFormatter` renders foreign `logging` records through the
same processor chain, so litestar, uvicorn and DBOS lines carry the same field names. Every record
reaches the one root handler: litestar gets `logging_config=None` and `haskie run` gives uvicorn
`log_config=None`. Two libraries install a text handler of their own anyway, so `configure` takes
their loggers over: uvicorn's CLI (under `litestar run --reload`) sets its up before it imports the
app, and DBOS adds its own as it initializes, unless its logger already has one.
"""

import logging
import logging.config
import os
from contextlib import AbstractContextManager
from typing import Any

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

from haskie import home

AUDIT = 25  # between INFO and WARNING: kept at the default level, quiet enough not to be noise
logging.addLevelName(AUDIT, "AUDIT")

_LEVEL_VAR = "HASKIE_LOG_LEVEL"
_FORMAT_VAR = "HASKIE_LOG_FORMAT"
_DEFAULT_LEVEL = "INFO"
_DEFAULT_FORMAT = "json"


def level() -> str:
    """The level every logger in this process runs at; DBOS is told the same one."""
    return os.environ.get(_LEVEL_VAR, _DEFAULT_LEVEL).upper()


def drop_color_message(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """uvicorn passes a coloured copy of some messages in `extra`, for a terminal; in a log line it
    is the message again, with escape codes and unfilled `%` placeholders."""
    event_dict.pop("color_message", None)
    return event_dict


def scrub_paths(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Absolute paths identify the user, so scrub every string value, not only the message."""
    return {k: home.scrub(v) if isinstance(v, str) else v for k, v in event_dict.items()}


SHARED_PROCESSORS: list[Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.stdlib.ExtraAdder(),
    drop_color_message,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    scrub_paths,
]


def _renderer(log_format: str) -> Processor:
    if log_format == "console":
        return structlog.dev.ConsoleRenderer()
    return structlog.processors.JSONRenderer()


def formatter(log_format: str) -> structlog.stdlib.ProcessorFormatter:
    """What renders every line, ours and the libraries': the shared chain, then JSON or console."""
    return structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            _renderer(log_format),
        ],
        foreign_pre_chain=SHARED_PROCESSORS,
    )


_configured = False


def configure() -> None:
    """Idempotent: the app factory, the DBOS worker and the tests all call it."""
    global _configured
    if _configured:
        return
    log_format = os.environ.get(_FORMAT_VAR, _DEFAULT_FORMAT).lower()
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"standard": {"()": formatter, "log_format": log_format}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "standard",
                },
                # DBOS installs its text handler only on a logger that has none
                "null": {"class": "logging.NullHandler"},
            },
            "root": {"handlers": ["console"], "level": level()},
            # Configuring a logger here removes the handlers it already has.
            "loggers": {
                "uvicorn": {"handlers": [], "propagate": True},
                "uvicorn.access": {"handlers": [], "propagate": True},
                "dbos": {"handlers": ["null"], "propagate": True},
            },
        }
    )
    structlog.configure(
        processors=[*SHARED_PROCESSORS, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        # stdlib levels do the filtering; AUDIT goes through a plain stdlib logger (see audit.py)
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.stdlib.get_logger(name)


def bind(**values: Any) -> None:
    """Add fields to every log line of this task or thread until it is cleared."""
    structlog.contextvars.bind_contextvars(**values)


def bound(**values: Any) -> AbstractContextManager[None]:
    """`bind` scoped to a block."""
    return structlog.contextvars.bound_contextvars(**values)


def clear() -> None:
    """Drop every bound field; a request middleware calls this before it binds its own."""
    structlog.contextvars.clear_contextvars()
