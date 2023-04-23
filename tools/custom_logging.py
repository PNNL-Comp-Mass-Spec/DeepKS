import logging
import os
import time
from typing import Literal
from termcolor import colored

logging.DEBUG = 5
VANISHING_STATUS = 9
STATUS = 10
TRAIN_INFO = 21
VAL_INFO = 23
TEST_INFO = 25
USER_ERROR = 35
ERROR = 45

logging.addLevelName(logging.DEBUG, "Debug")
logging.addLevelName(VANISHING_STATUS, "Status")
logging.addLevelName(STATUS, "Status")
logging.addLevelName(logging.INFO, "Info")
logging.addLevelName(TRAIN_INFO, "Train Info")
logging.addLevelName(VAL_INFO, "Validation Info")
logging.addLevelName(TEST_INFO, "Test Result Info")
logging.addLevelName(logging.WARNING, "Warning")
logging.addLevelName(USER_ERROR, "User Error")
logging.addLevelName(logging.ERROR, "Unexpected Non-Accounted-For Error")

logging.VANISHING_STATUS = VANISHING_STATUS  # type: ignore
logging.STATUS = STATUS  # type: ignore
logging.TRAIN_INFO = TRAIN_INFO  # type: ignore
logging.VAL_INFO = VAL_INFO  # type: ignore
logging.TEST_INFO = TEST_INFO  # type: ignore
logging.USER_ERROR = USER_ERROR  # type: ignore


class CustomLogger(logging.Logger):
    def __init__(
        self, logging_level: int = logging.DEBUG, output_method: Literal["console", "logfile"] = "console"
    ) -> None:
        super().__init__("MainCustomLogger")
        self._level = logging_level

        # Create a logger
        self.setLevel(self._level)

        # Create console handler and set level
        self.handler = logging.StreamHandler() if output_method == "console" else logging.FileHandler("logfile.log")
        self.handler.setLevel(
            logging_level if isinstance(self.handler, logging.StreamHandler) else max(self._level, STATUS)
        )

        # Create formatter
        formatter = CustomFormatter()

        # Add formatter to console_handler
        self.handler.setFormatter(formatter)

        # Add console_handler to logger
        self.addHandler(self.handler)

        self.last_log = None

    def _update_logging_level(self, new_level: int):
        self.setLevel(new_level)
        self.handler.setLevel(new_level)

    def _update_logging_method(self, new_method: Literal["console", "logfile"]):
        self.handler = logging.StreamHandler() if new_method == "console" else logging.FileHandler("logfile.log")
        self.handler.setLevel(
            self._level if isinstance(self.handler, logging.StreamHandler) else max(self._level, STATUS)
        )

    def _blankit(self):
        if self.last_log and self.last_log == "vstatus":
            print(" " * os.get_terminal_size().columns, end="\r")

    def debug(self, msg, *args, **kwargs):
        self._blankit()
        if self.isEnabledFor(logging.DEBUG):
            self._log(logging.DEBUG, msg, args, **kwargs)
        self.last_log = "debug"

    def vstatus(self, msg, *args, **kwargs):
        self._blankit()
        if self.isEnabledFor(VANISHING_STATUS):
            self._log(VANISHING_STATUS, msg, args, **kwargs)
        self.last_log = "vstatus"

    def status(self, msg, *args, **kwargs):
        self._blankit()
        if self.isEnabledFor(STATUS):
            self._log(STATUS, msg, args, **kwargs)
        self.last_log = "status"

    def info(self, msg, *args, **kwargs):
        self._blankit()
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)
        self.last_log = "info"

    def trinfo(self, msg, *args, **kwargs):
        self._blankit()
        if self.isEnabledFor(TRAIN_INFO):
            self._log(TRAIN_INFO, msg, args, **kwargs)
        self.last_log = "trinfo"

    def valinfo(self, msg, *args, **kwargs):
        self._blankit()
        if self.isEnabledFor(VAL_INFO):
            self._log(VAL_INFO, msg, args, **kwargs)
        self.last_log = "valinfo"

    def teinfo(self, msg, *args, **kwargs):
        self._blankit()
        if self.isEnabledFor(TEST_INFO):
            self._log(TEST_INFO, msg, args, **kwargs)
        self.last_log = "teinfo"

    def warning(self, msg, *args, **kwargs):
        self._blankit()
        if self.isEnabledFor(logging.WARNING):
            self._log(logging.WARNING, msg, args, **kwargs)
        self.last_log = "warning"

    def error(self, msg, *args, **kwargs):
        self._blankit()
        if self.isEnabledFor(logging.ERROR):
            self._log(logging.ERROR, msg, args, **kwargs)
        self.last_log = "error"

    def uerror(self, msg, *args, **kwargs):
        self._blankit()
        if self.isEnabledFor(USER_ERROR):
            self._log(USER_ERROR, msg, args, **kwargs)
        self.last_log = "uerror"


class CustomFormatter(logging.Formatter):
    format_vanish = "{levelname}: {message}\033[F"
    format_neutral = "{levelname}: {message}"
    format_danger = "{levelname}: {message} ({filename}:{lineno})"

    FORMATS = {
        logging.DEBUG: colored(format_neutral, "grey"),
        logging.VANISHING_STATUS: colored(format_vanish, "light_green"),  # type: ignore
        logging.STATUS: colored(format_neutral, "green"),  # type: ignore
        logging.INFO: colored(format_neutral, "blue"),
        logging.TRAIN_INFO: colored(format_neutral, "cyan"),  # type: ignore
        logging.VAL_INFO: colored(format_neutral, "cyan", attrs=["underline"]),  # type: ignore
        logging.TEST_INFO: colored(format_neutral, "cyan", attrs=["bold"]),  # type: ignore
        logging.WARNING: colored(format_danger, "yellow"),
        logging.USER_ERROR: colored(format_danger, "red"),  # type: ignore
        logging.ERROR: colored(format_danger, "red", attrs=["bold"]),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, style="{")
        return formatter.format(record)


if __name__ == "__main__":
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
