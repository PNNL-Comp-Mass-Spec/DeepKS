"""Module building on top of the logging module to provide a custom logger with custom logging functions and colors"""
from __future__ import annotations
import logging, inspect
import os
import time, tqdm, pathlib
from typing import Literal
from termcolor import colored

logging.DEBUG = 5
VANISHING_STATUS = 9
"""A status that is overwritten by the next logging statement."""
STATUS = 10
"""A status that is not overwritten by the next logging statement.""" ""
PROGRESS = 15
"""Whether or not to show progress bars."""
TRAIN_INFO = 21
"""Information about performance in the training process."""
VAL_INFO = 23
"""Information about performance in the validation process."""
TEST_INFO = RESINFO = 25
"""Information about performance in the testing process."""
USER_ERROR = 35
"""An error that is caused and/or partially expected by the user."""
ERROR = 45
"""A totally unexpected error."""

logging.addLevelName(logging.DEBUG, "Debug")
logging.addLevelName(VANISHING_STATUS, "Status")
logging.addLevelName(STATUS, "Status")
logging.addLevelName(PROGRESS, "Progress")
logging.addLevelName(logging.INFO, "Info")
logging.addLevelName(TRAIN_INFO, "Train Info")
logging.addLevelName(VAL_INFO, "Validation Info")
logging.addLevelName(TEST_INFO, "Test Result Info")
logging.addLevelName(RESINFO, "Results Info")
logging.addLevelName(logging.WARNING, "Warning")
logging.addLevelName(USER_ERROR, "User Error")
logging.addLevelName(logging.ERROR, "Unexpected Non-Accounted-For Error")

logging.VANISHING_STATUS = VANISHING_STATUS  # type: ignore
logging.STATUS = STATUS  # type: ignore
logging.PROGRESS = PROGRESS  # type: ignore
logging.TRAIN_INFO = TRAIN_INFO  # type: ignore
logging.VAL_INFO = VAL_INFO  # type: ignore
logging.TEST_INFO = TEST_INFO  # type: ignore
logging.RESINFO = RESINFO  # type: ignore
logging.USER_ERROR = USER_ERROR  # type: ignore


class TqdmStreamHandler(logging.StreamHandler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class CustomLogger(logging.Logger):
    def __init__(
        self,
        logging_level: int = logging.DEBUG,
        output_method: Literal["console", "logfile"] = "console",
        upper_logging_level: int | float = float("inf"),
    ) -> None:
        super().__init__("MainCustomLogger")
        self._level = logging_level
        self._upper_level = upper_logging_level

        # Create a logger
        self.setLevel(self._level)

        # Create console handler and set level
        if output_method == "console":
            self.handler = TqdmStreamHandler()
        else:
            self.handler = logging.FileHandler("logfile.log")
        if isinstance(self.handler, logging.StreamHandler):
            self.handler.setLevel(logging_level)
        else:
            self.handler.setLevel(max(self._level, STATUS))

        # Create formatter
        formatter = CustomFormatter()

        # Add formatter to console_handler
        self.handler.setFormatter(formatter)

        # Add console_handler to logger
        self.addHandler(self.handler)

        self.last_log = None

        self.VANISHING_STATUS = VANISHING_STATUS
        """A status that is overwritten by the next logging statement."""
        self.STATUS = STATUS
        """A status that is not overwritten by the next logging statement."""
        self.PROGRESS = PROGRESS
        """Whether or not to show progress bars."""
        self.TRAIN_INFO = TRAIN_INFO
        """Information about performance in the training process."""
        self.VAL_INFO = VAL_INFO
        """Information about performance in the validation process."""
        self.TEST_INFO = TEST_INFO
        """Information about performance in the testing process."""
        self.RESINFO = RESINFO
        """Information about performance in the testing process."""
        self.USER_ERROR = USER_ERROR
        """An error that is caused and/or partially expected by the user."""
        self.ERROR = ERROR
        """A totally unexpected error."""

    def _update_logging_level(self, new_level: int):
        self.setLevel(new_level)
        self.handler.setLevel(new_level)

    def _update_logging_method(self, new_method: Literal["console", "logfile"]):
        if new_method == "console":
            self.handler = logging.StreamHandler()
        else:
            self.handler = logging.FileHandler("logfile.log")
            if isinstance(self.handler, logging.StreamHandler):
                self.handler.setLevel(self._level)
            else:
                self.handler.setLevel(max(self._level, STATUS))

    def _blankit(self):
        # if self.last_log and self.last_log == "vstatus":
        print(" " * os.get_terminal_size().columns, end="\r")

    def debug(self, msg, *args, **kwargs):
        """Log debugging statements."""
        if self._upper_level >= logging.DEBUG:
            self._blankit()
            if self.isEnabledFor(logging.DEBUG):
                self._log(logging.DEBUG, msg, args, **kwargs)
            self.last_log = "debug"

    def vstatus(self, msg, *args, **kwargs):
        """Log a status that is overwritten by the next logging statement."""
        if self._upper_level >= VANISHING_STATUS:
            self._blankit()
            if self.isEnabledFor(VANISHING_STATUS):
                self._log(VANISHING_STATUS, msg, args, **kwargs)
            self.last_log = "vstatus"

    def status(self, msg, *args, **kwargs):
        """Log a status that is not overwritten by the next logging statement."""
        if self._upper_level >= STATUS:
            self._blankit()
            if self.isEnabledFor(STATUS):
                self._log(STATUS, msg, args, **kwargs)
            self.last_log = "status"

    def info(self, msg, *args, **kwargs):
        """Log information not related to program progress."""
        if self._upper_level >= logging.INFO:
            self._blankit()
            if self.isEnabledFor(logging.INFO):
                self._log(logging.INFO, msg, args, **kwargs)
            self.last_log = "info"

    def trinfo(self, msg, *args, **kwargs):
        """Log information about performance in the training process."""
        if self._upper_level >= TRAIN_INFO:
            self._blankit()
            if self.isEnabledFor(TRAIN_INFO):
                self._log(TRAIN_INFO, msg, args, **kwargs)
            self.last_log = "trinfo"

    def valinfo(self, msg, *args, **kwargs):
        """Log information about performance in the validation process."""
        if self._upper_level >= VAL_INFO:
            self._blankit()
            if self.isEnabledFor(VAL_INFO):
                self._log(VAL_INFO, msg, args, **kwargs)
            self.last_log = "valinfo"

    def teinfo(self, msg, *args, **kwargs):
        """Log information about performance in the testing process."""
        if self._upper_level >= TEST_INFO:
            self._blankit()
            if self.isEnabledFor(TEST_INFO):
                self._log(TEST_INFO, msg, args, **kwargs)
            self.last_log = "teinfo"

    def resinfo(self, msg, *args, **kwargs):
        """Log information that are prediction results."""
        if self._upper_level >= RESINFO:
            self._blankit()
            if self.isEnabledFor(RESINFO):
                self._log(RESINFO, msg, args, **kwargs)
            self.last_log = "resinfo"

    def warning(self, msg, *args, **kwargs):
        """Log warnings."""
        if self._upper_level >= logging.WARNING:
            self._blankit()
            if self.isEnabledFor(logging.WARNING):
                frame = inspect.currentframe()
                assert frame is not None
                back_frame = frame.f_back
                assert back_frame is not None
                lineno = back_frame.f_lineno
                finame = back_frame.f_code.co_filename
                loc_msg = kwargs.get("loc_msg", f"{finame}:{lineno}")
                msg = f"{msg} ({kwargs.get('loc_msg', loc_msg)})"
                if "loc_msg" in kwargs:
                    del kwargs["loc_msg"]

                self._log(logging.WARNING, msg, args, **kwargs)
            self.last_log = "warning"

    def error(self, msg, *args, **kwargs):
        """Log errors."""
        if self._upper_level >= logging.ERROR:
            self._blankit()
            if self.isEnabledFor(logging.ERROR):
                self._log(logging.ERROR, msg, args, **kwargs)
            self.last_log = "error"

    def uerror(self, msg, *args, **kwargs):
        """Log errors that are caused and/or partially anticipated by the user."""
        if self._upper_level >= USER_ERROR:
            self._blankit()
            if self.isEnabledFor(USER_ERROR):
                self._log(USER_ERROR, msg, args, **kwargs)
            self.last_log = "uerror"


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and styles for easy identification."""

    format_vanish = "{levelname}: {message}\033[F"
    format_neutral = "{levelname}: {message}"
    format_danger = "{levelname}: {message}"

    FORMATS = {
        logging.DEBUG: colored(format_neutral, "grey"),
        logging.VANISHING_STATUS: colored(format_vanish, "light_green"),  # type: ignore
        logging.STATUS: colored(format_neutral, "green"),  # type: ignore
        logging.INFO: colored(format_neutral, "blue"),
        logging.TRAIN_INFO: colored(format_neutral, "cyan"),  # type: ignore
        logging.VAL_INFO: colored(format_neutral, "cyan", attrs=["bold"]),  # type: ignore
        logging.TEST_INFO: colored(format_neutral, "cyan", attrs=["bold"]),  # type: ignore
        logging.RESINFO: colored(format_neutral, "cyan", attrs=["bold"]),  # type: ignore
        logging.WARNING: colored(format_danger, "yellow"),
        logging.USER_ERROR: colored(format_danger, "red"),  # type: ignore
        logging.ERROR: colored(format_danger, "red", attrs=["bold"]),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, style="{")
        return formatter.format(record)


if __name__ == "__main__":  # pragma: no cover
    logger = CustomLogger(logging_level=logging.DEBUG)
    logger.debug("This is a debug message.")
    logger.vstatus("This is a vanishing status message.")
    time.sleep(1)
    logger.status("This is a status message.")
    logger.trinfo("This is a training info message.")
    logger.valinfo("This is a validation info message.")
    logger.teinfo("This is a test info message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.uerror("This is a user error message.")
    logger.error("This is an error message.")
