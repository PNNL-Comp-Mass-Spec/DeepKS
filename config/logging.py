"""Contains logging and printing functions that draw from `DeepKS.tools.custom_logging` and `DeepKS.tools.splash`."""

from ..tools import custom_logging
from ..tools.splash.write_splash import write_splash
import pathlib, os, json

from typing import Any


def join_first(levels: int = 1, x: Any = "/"):
    """Helper function to join a target path to a pseudo-root path derived from the location of this file.

    Parameters
    ----------
    levels : optional
        How many directories out of the directory of this file the "new root" should start, by default 1
    x :
        The target path, by default "/"

    Returns
    -------
    str
        The joined path

    Examples
    --------
    >>> join_first(1, "images/Phylo Families/phylo_families_Cairo.pdf")
    "/Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/api/../images/Phylo Families/phylo_families_Cairo.pdf"
    """
    x = str(x)
    if os.path.isabs(x):
        return x
    else:
        return os.path.join(pathlib.Path(__file__).parent.resolve(), *[".."] * levels, x)


import warnings


def splash(splash_file):
    """Wrapper for `write_splash`. Same Parameters."""

    write_splash(splash_file)


def get_logger():
    """Wrapper for `custom_logging.CustomLogger`, that can be configured by a JSON file in the same directory as this file.
    """
    try:
        with open(f"{join_first(0, 'logging_config.json')}") as f:
            kwargs = json.load(f)
            good_keys = {"logging_level", "output_method"}
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
        logger.warning(f"{category.__name__}--{message} ({filename}:{lineno})")

    # Register the custom warning handler with the warnings module
    warnings.showwarning = warning_handler

    return logger
