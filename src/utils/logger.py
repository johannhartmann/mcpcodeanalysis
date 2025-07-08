"""Logging configuration and utilities."""

import json
import logging
import sys
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from src.mcp_server.config import LoggingConfig, get_settings


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_data[key] = value

        return json.dumps(log_data)


class StructuredLogger:
    """Structured logger with context support."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self._context: dict[str, Any] = {}

    def with_context(self, **kwargs) -> "StructuredLogger":
        """Create a new logger with additional context."""
        new_logger = StructuredLogger(self.logger)
        new_logger._context = {**self._context, **kwargs}
        return new_logger

    def _log(self, level: int, msg: str, **kwargs) -> None:
        """Log with context."""
        extra = {**self._context, **kwargs}
        self.logger.log(level, msg, extra=extra)

    def debug(self, msg: str, **kwargs) -> None:
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs) -> None:
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs) -> None:
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs) -> None:
        self._log(logging.CRITICAL, msg, **kwargs)

    def exception(self, msg: str, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(msg, extra={**self._context, **kwargs})


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Set up logging configuration."""
    if config is None:
        settings = get_settings()
        config = settings.logging

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if config.console_enabled:
        if config.console_colorized:
            console = Console(stderr=True)
            console_handler = RichHandler(
                console=console,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
            )
        else:
            console_handler = logging.StreamHandler(sys.stderr)
            if config.format == "json":
                console_handler.setFormatter(JSONFormatter())
            else:
                console_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    ),
                )

        root_logger.addHandler(console_handler)

    # File handler
    if config.file_enabled:
        config.file_path.parent.mkdir(parents=True, exist_ok=True)

        if config.file_rotation == "daily":
            from logging.handlers import TimedRotatingFileHandler

            file_handler = TimedRotatingFileHandler(
                filename=str(config.file_path),
                when="midnight",
                interval=1,
                backupCount=config.file_retention_days,
            )
        else:
            file_handler = logging.FileHandler(str(config.file_path))

        if config.format == "json":
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                ),
            )

        root_logger.addHandler(file_handler)

    # Set specific logger levels
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(logging.getLogger(name))


# Initialize logging on import
setup_logging()
