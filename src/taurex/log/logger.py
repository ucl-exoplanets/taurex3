"""Taurex logging module."""
import io
import logging
import typing as t

logging.getLogger("taurex").addHandler(logging.NullHandler())
"""Root logger for taurex"""

root_logger = logging.getLogger("taurex")


def setup_log(name: str) -> logging.Logger:
    """Build a logger class for the given name.

    Parameters
    ----------
    name : str
        Name of logger

    Returns
    -------
    logging.Logger
        Logger class

    """
    return logging.getLogger(name)


class TauRexHandler(logging.StreamHandler):
    """Logging Handler for Taurex 3.

    Prevents other MPI threads from writing to log
    unless they are in trouble (>=ERROR)

    Parameters
    ----------
    stream : stream-object , optional
        Stream to write to otherwise defaults to ``stderr``

    """

    def __init__(self, stream: io.IOBase = None) -> None:
        from taurex.mpi import get_rank

        super().__init__(stream=stream)

        self._rank = get_rank()

    def emit(self, record: logging.LogRecord) -> None:
        # print(record)
        if self._rank == 0 or record.levelno >= logging.ERROR:
            # msg = '[{}] {}'.format(self._rank,record.msg)
            # record.msg = msg
            return super().emit(record)
        else:
            pass


class Loggable:
    """Base class for loggable objects."""

    def __init__(self, name: t.Optional[str] = None) -> None:
        """Initialise logger."""
        name = name or self.__class__.__name__
        self._logger = setup_log(f"taurex.{name}")

    def info(self, message: str, *args: t.Any, **kwargs: t.Any) -> None:
        """See :class:`logging.Logger`."""
        self._logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args: t.Any, **kwargs: t.Any) -> None:
        """See :class:`logging.Logger`."""
        self._logger.warning(message, *args, **kwargs)

    def debug(self, message: str, *args: t.Any, **kwargs: t.Any) -> None:
        """See :class:`logging.Logger`."""
        import inspect

        frame = inspect.currentframe()

        new_message = message
        if frame is not None:
            f_back = frame.f_back
            if f_back is not None:
                f_code = f_back.f_code
                new_message = f"In: {f_code.co_name}() - {message}"

        self._logger.debug(new_message, *args, **kwargs)

    def error(self, message: str, *args: t.Any, **kwargs: t.Any) -> None:
        """See :class:`logging.Logger`."""
        self._logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args: t.Any, **kwargs: t.Any) -> None:
        """See :class:`logging.Logger`."""
        self._logger.critical(message, *args, **kwargs)

    def error_and_raise(
        self, exception: t.Type[Exception], message: str, *args: t.Any, **kwargs: t.Any
    ) -> Exception:
        """Print error message and raises exception."""
        self._logger.error(message, *args, **kwargs)
        raise exception(message, *args, **kwargs)
