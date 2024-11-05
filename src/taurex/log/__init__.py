"""Main logging module."""

import logging

from .logger import Loggable
from .logger import Loggable as Logger
from .logger import setup_log

last_log = logging.INFO


def setLogLevel(level: int) -> None:  # noqa: N802
    """Set the log level.

    Parameters
    ----------
    level : int
        Log level to set.

    """
    global last_log
    from .logger import root_logger

    root_logger.setLevel(level)
    last_log = level


def disableLogging() -> None:  # noqa: N802
    """Disable logging."""
    import logging

    from .logger import root_logger

    global last_log
    last_log = root_logger.level
    root_logger.setLevel(logging.CRITICAL)


def enableLogging() -> None:  # noqa: N802
    """Enable logging."""
    global last_log
    import logging

    if last_log is None:
        last_log = logging.INFO
    setLogLevel(last_log)


__all__ = [
    "Logger",
    "Loggable",
    "setLogLevel",
    "disableLogging",
    "enableLogging",
    "setup_log",
]
