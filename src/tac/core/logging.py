"""Structured logging configuration for the Twilio Agent Connect."""

import json
import logging
import re
import sys
from typing import Any

# Standard logging attributes that should not be included in structured output
_RESERVED_LOG_ATTRS = {
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
    "message",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "thread",
    "threadName",
    "exc_info",
    "exc_text",
    "stack_info",
    "taskName",
}


_PHONE_RE = re.compile(
    r"(?<!\w)"
    r"(?:\+\d[\d\-\s()]{6,}\d"
    r"|(?:\(\d{3}\)\s*|\d{3}[\-.\s]?)\d{3}[\-.\s]?\d{4}"
    r"|\d{10})"
    r"(?!\w)"
)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_REDACTED = "***"


def _scrub_pii(value: str) -> str:
    """Replace phone-number and email patterns with ``***``."""
    value = _PHONE_RE.sub(_REDACTED, value)
    value = _EMAIL_RE.sub(_REDACTED, value)
    return value


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging using only stdlib."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize JSON formatter."""
        super().__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": _scrub_pii(record.getMessage()),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = _scrub_pii(self.formatException(record.exc_info))

        # Add extra fields from LogRecord
        # Skip standard fields and internal fields
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_ATTRS and not key.startswith("_"):
                log_data[key] = _scrub_pii(str(value)) if isinstance(value, str) else value

        return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with context support."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record for console output with context.

        Args:
            record: Log record to format

        Returns:
            Formatted log string
        """
        # Build context string from extra attributes
        context_parts = []
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_ATTRS and not key.startswith("_"):
                context_parts.append(f"{key}={value}")

        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""

        # Format base message
        formatted = super().format(record)

        return _scrub_pii(f"{formatted}{context_str}")


class ContextLogger:
    """Logger wrapper that binds context to all log calls."""

    def __init__(self, logger: logging.Logger, **context: Any):
        """
        Initialize context logger.

        Args:
            logger: Base logger instance
            **context: Context fields to bind to all log calls
        """
        self.logger = logger
        self.context = context

    def _log(self, level: int, msg: str, exc_info: bool = False, **extra: Any) -> None:
        """
        Internal log method that merges context with extra fields.

        Args:
            level: Log level
            msg: Log message
            exc_info: Include exception info
            **extra: Additional fields to log
        """
        merged_extra = {**self.context, **extra}
        self.logger.log(level, msg, extra=merged_extra, exc_info=exc_info)

    def debug(self, msg: str, **extra: Any) -> None:
        """
        Log debug message with context.

        Args:
            msg: Log message
            **extra: Additional fields
        """
        self._log(logging.DEBUG, msg, **extra)

    def info(self, msg: str, **extra: Any) -> None:
        """
        Log info message with context.

        Args:
            msg: Log message
            **extra: Additional fields
        """
        self._log(logging.INFO, msg, **extra)

    def warning(self, msg: str, **extra: Any) -> None:
        """
        Log warning message with context.

        Args:
            msg: Log message
            **extra: Additional fields
        """
        self._log(logging.WARNING, msg, **extra)

    def error(self, msg: str, exc_info: bool = False, **extra: Any) -> None:
        """
        Log error message with context.

        Args:
            msg: Log message
            exc_info: Include exception traceback
            **extra: Additional fields
        """
        self._log(logging.ERROR, msg, exc_info=exc_info, **extra)

    def critical(self, msg: str, exc_info: bool = False, **extra: Any) -> None:
        """
        Log critical message with context.

        Args:
            msg: Log message
            exc_info: Include exception traceback
            **extra: Additional fields
        """
        self._log(logging.CRITICAL, msg, exc_info=exc_info, **extra)

    def bind(self, **context: Any) -> "ContextLogger":
        """
        Create new logger with additional context.

        Args:
            **context: Additional context fields to bind

        Returns:
            New ContextLogger with merged context
        """
        return ContextLogger(self.logger, **{**self.context, **context})

    def isEnabledFor(self, level: int) -> bool:  # noqa: N802 - matches logging.Logger API
        """
        Check if logger is enabled for the given level.

        Note: Method name uses camelCase to match the standard library's logging.Logger API
        for drop-in compatibility with code expecting a Logger interface.

        Args:
            level: Logging level to check

        Returns:
            True if logger is enabled for the level
        """
        return self.logger.isEnabledFor(level)


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> logging.Logger:
    """
    Configure structured logging for TAC framework.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format - 'json' for structured logs, 'console' for human-readable

    Returns:
        Configured logger instance
    """
    # Get the root logger for TAC
    logger = logging.getLogger("tac")

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Create handler and formatter based on format preference
    handler = logging.StreamHandler(sys.stdout)

    formatter: logging.Formatter
    if log_format == "json":
        formatter = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    else:
        formatter = ConsoleFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


def get_logger(name: str, **context: Any) -> ContextLogger:
    """
    Get a context-aware logger instance for a specific module.

    Args:
        name: Logger name (typically __name__ from the calling module)
        **context: Initial context to bind (e.g., conversation_id, channel)

    Returns:
        ContextLogger instance with bound context
    """
    # Create child logger under the tac namespace
    base_logger = logging.getLogger(f"tac.{name}")
    return ContextLogger(base_logger, **context)
