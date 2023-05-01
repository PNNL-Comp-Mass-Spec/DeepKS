"""Contains functionality to automatically glean the latest pre-trained neural network and group classifier binaries."""

import json, os, pathlib, re, warnings
from termcolor import colored

PRE_TRAINED_NN: str
"""The path to the latest pre-trained neural network binary."""
PRE_TRAINED_GC: str
"""The path to the latest pre-trained group classifier binary."""

from ..config.logging import get_logger

logger = get_logger()
"""The logger for this module."""


def smart_get_latest():
    """Automatically get the latest pre-trained neural network and group classifier binary file names

    Returns
    -------
    None
        Does not return anything, but sets the global variables PRE_TRAINED_NN and PRE_TRAINED_GC
    """
    global PRE_TRAINED_NN, PRE_TRAINED_GC
    # List bin directory
    set_vars = []
    bin_ = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "bin")
    for pttrn in ["nn_weights", "gc_weights"]:
        max_version = -1
        one_matched = False
        for file in os.listdir(bin_):
            if v := re.search(r"(UNITTESTVERSION|)deepks_" + pttrn + r"\.((|-)\d+)\.cornichon", file):
                one_matched = True
                max_version = max(max_version, int(v.group(2)))
        if not one_matched:
            warnings.warn(colored(f"No pre-trained {pttrn} file found in {bin_}!", "yellow"), ResourceWarning)
        set_vars.append(os.path.join(bin_, "deepks_" + pttrn + "." + str(max_version) + ".cornichon"))
    PRE_TRAINED_NN, PRE_TRAINED_GC = tuple(set_vars)
    logger.info(f"Using latest pre-trained neural network: {PRE_TRAINED_NN}")
    logger.info(f"Using latest pre-trained group classifier: {PRE_TRAINED_GC}")


smart_get_latest()
