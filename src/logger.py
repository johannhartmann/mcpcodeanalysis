"""Structured logging configuration using structlog."""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from structlog.processors import CallsiteParameter

from src.config import settings

if TYPE_CHECKING:
    from structlog.types import Processor


def setup_logging() -> None:
    """Configure structured logging with structlog."""
    # Determine log level
    log_level = getattr(logging, settings.logging.level.upper())

    # Configure timestamper
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    # Shared processors
    shared_processors: list["Processor"] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.ExtraAdder(),
        timestamper,
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                CallsiteParameter.PATHNAME,
                CallsiteParameter.FILENAME,
                CallsiteParameter.MODULE,
                CallsiteParameter.FUNC_NAME,
                CallsiteParameter.LINENO,
            ]
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Configure structlog based on format
    processors: list["Processor"]
    if settings.logging.format == "json":
        # JSON output
        processors = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Human-readable output
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(
                colors=settings.logging.console_colorized,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # File logging if enabled
    if settings.logging.file_enabled:
        setup_file_logging(log_level)

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def setup_file_logging(log_level: int) -> None:
    """Set up file logging with rotation."""
    from logging.handlers import TimedRotatingFileHandler

    log_path = Path(settings.logging.file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create file handler with rotation
    file_handler = TimedRotatingFileHandler(
        filename=str(log_path),
        when=settings.logging.file_rotation[0],  # 'd' for daily
        interval=1,
        backupCount=settings.logging.file_retention_days,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)

    # Add formatter based on format setting
    if settings.logging.format == "json":
        # Recreate shared processors for file handler
        file_processors: list["Processor"] = [
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.ExtraAdder(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    CallsiteParameter.PATHNAME,
                    CallsiteParameter.FILENAME,
                    CallsiteParameter.MODULE,
                    CallsiteParameter.FUNC_NAME,
                    CallsiteParameter.LINENO,
                ]
            ),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        formatter: logging.Formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=file_processors,
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    file_handler.setFormatter(formatter)

    # Add handler to root logger
    logging.getLogger().addHandler(file_handler)


def get_logger(name: str | None = None, **context: Any) -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (defaults to module name)
        **context: Additional context to bind to the logger

    Returns:
        Bound logger instance
    """
    logger: structlog.BoundLogger = structlog.get_logger(name)
    if context:
        logger = logger.bind(**context)
    return logger


# Initialize logging when module is imported
setup_logging()
