"""Utility functions for logging."""

import logging
import sys

PACKAGE_LOGGER = "paras-fast"

STANDARD_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
STANDARD_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str | int = "INFO",
    *,
    fmt: str = STANDARD_FMT,
    datefmt: str = STANDARD_DATEFMT,
    stream: None | int | str | object = None,
) -> None:
    """
    Set up logging for the package.

    :param level: logging level (e.g., "INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL")
    :param fmt: log message format
    :param datefmt: date format for log messages
    :param stream: stream to which logs will be written (default: sys.stderr)
    """
    if stream is None:
        stream = sys.stderr

    if isinstance(level, str):
        level = level.upper()

    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    # Avoid duplicate handlers if called repeatedly
    # Keep simple: remove existing handlers created by previous setup calls
    root.handlers = [handler]

    # Make sure package logger propagates to root
    logging.getLogger(PACKAGE_LOGGER).propagate = True


def add_file_handler(
    logfile: str,
    *,
    level: str | int = "INFO",
    fmt: str = STANDARD_FMT,
    datefmt: str = STANDARD_DATEFMT,
) -> None:
    """
    Add a file handler to the root logger.

    :param logfile: path to log file
    :param level: logging level for the file handler (e.g., "INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL")
    :param fmt: log message format for the file handler
    :param datefmt: date format for log messages in the file handler
    .. note:: intended to be called after setup_logging(); safe to call multiple times for same logfile
    """
    if isinstance(level, str):
        level = level.upper()

    root = logging.getLogger()

    # Prevent duplicate file handlers for the same path
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and h.baseFilename == logfile:
            return
        
    fh = logging.FileHandler(logfile)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root.addHandler(fh)
