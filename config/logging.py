"""Contains logging and printing functions that draw from `DeepKS.tools.custom_logging` and `DeepKS.tools.splash`."""

from ..tools import custom_logging
from ..tools.splash.write_splash import write_splash
import pathlib, os, json

from typing import Any


from .join_first import join_first
import warnings


def get_logger():
    """Wrapper for `custom_logging.CustomLogger`, that can be configured by a JSON file in the same directory as this file.
    """
    try:
        with open(f"{join_first('logging_config.json', 0, __file__)}") as f:
            kwargs = json.load(f)
            good_keys = {"logging_level", "upper_logging_level", "output_method"}
            for k, v in kwargs.items():
                assert k in good_keys
                if k == "logging_level":
                    assert isinstance(v, int) and v >= 0
                elif k == "output_method":
                    assert isinstance(v, str) and k in {"console", "logfile"}
    except json.decoder.JSONDecodeError as e:
        if "Expecting value:" in str(e):
            kwargs = {}
        else:
            raise e from None

    logger = custom_logging.CustomLogger(**kwargs)

    # Define a custom warning handler
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logger.warning(f"{category.__name__} — {message} ({filename}:{lineno})")

    # Register the custom warning handler with the warnings module
    warnings.showwarning = warning_handler

    return logger
