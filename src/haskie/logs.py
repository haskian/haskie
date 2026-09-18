"""Structured logging: one JSON line per event on stdout, for our code and for the libraries.

structlog renders our events; a `ProcessorFormatter` renders foreign `logging` records through the
same processor chain, so litestar, uvicorn and DBOS lines carry the same field names. Litestar gets
`logging_config` so it configures structlog exactly the way `configure()` does.
"""

import logging
import os
from contextlib import AbstractContextManager
from typing import Any

import structlog
from litestar.logging.config import LoggingConfig, StructLoggingConfig
from structlog.typing import EventDict, Processor, WrappedLogger

from haskie import errors

AUDIT = 25  # between INFO and WARNING: kept at the default level, quiet enough not to be noise
logging.addLevelName(AUDIT, "AUDIT")

LEVEL_VAR = "HASKIE_LOG_LEVEL"
FORMAT_VAR = "HASKIE_LOG_FORMAT"
DEFAULT_LEVEL = "INFO"
DEFAULT_FORMAT = "json"

# Libraries that install their own handler or format. Empty handlers plus propagate routes their
# records to the root handler, which renders them like ours.
ADOPTED_LOGGERS: tuple[str, ...] = (
    "litestar",
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "dbos",
)


def scrub_paths(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Absolute paths identify the user, so scrub every string value, not only the message."""
    return {k: errors.scrub(v) if isinstance(v, str) else v for k, v in event_dict.items()}


SHARED_PROCESSORS: list[Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.stdlib.ExtraAdder(),
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    scrub_paths,
]

logging_config = StructLoggingConfig(
    processors=[*SHARED_PROCESSORS, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,  # stdlib levels decide, so AUDIT works
    cache_logger_on_first_use=True,
    log_exceptions="always",
    pretty_print_tty=False,
)


def _renderer(log_format: str) -> Processor:
    if log_format == "console":
        return structlog.dev.ConsoleRenderer()
    return structlog.processors.JSONRenderer()


def _stdlib_config(level: str, log_format: str) -> LoggingConfig:
    """dictConfig for the stdlib side: one stdout handler rendering every record."""
    stream: dict[str, Any] = {
        "class": "logging.StreamHandler",
        "stream": "ext://sys.stdout",
        "formatter": "standard",
    }
    return LoggingConfig(
        formatters={
            "standard": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processors": [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    _renderer(log_format),
                ],
                "foreign_pre_chain": SHARED_PROCESSORS,
            }
        },
        # LoggingConfig injects its own `console` and `queue_listener` when either is missing;
        # declaring both keeps every record on one plain stdout handler and starts no listener.
        handlers={"console": dict(stream), "queue_listener": dict(stream)},
        loggers={name: {"handlers": [], "propagate": True} for name in ADOPTED_LOGGERS},
        root={"handlers": ["console"], "level": level},
    )


_configured = False


def configure() -> None:
    """Idempotent: the app factory, the DBOS worker and the tests all call it."""
    global _configured
    if _configured:
        return
    level = os.environ.get(LEVEL_VAR, DEFAULT_LEVEL).upper()
    log_format = os.environ.get(FORMAT_VAR, DEFAULT_FORMAT).lower()
    logging_config.standard_lib_logging_config = _stdlib_config(level, log_format)
    logging_config.standard_lib_logging_config.configure()
    logging_config.configure()  # same structlog call Litestar makes with `logging_config`
    _configured = True


def adopt_dbos_logger() -> None:
    """DBOS attaches its own text handler in `dbos/_logger.py` when it launches; drop it so its
    records reach our formatter instead of printing a second format alongside."""
    dbos_logger = logging.getLogger("dbos")
    dbos_logger.handlers.clear()
    dbos_logger.propagate = True


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
