"""Logging configuration for the TelecomPlus system.

Provides consistent logging across all modules.
"""

import logging
import sys
from typing import Optional

from .config import LOG_FORMAT, LOG_LEVEL


def setup_logger(
    name: str,
    level: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__ of the module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Set level
    log_level = level or LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid adding multiple handlers
    if not logger.handlers:
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, log_level.upper()))

        # Format
        formatter = logging.Formatter(format_string or LOG_FORMAT)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


# Default logger for the package
default_logger = setup_logger("telecomplus")
