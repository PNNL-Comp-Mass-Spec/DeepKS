import json, os, pathlib, re, warnings
from termcolor import colored

# where_am_i = pathlib.Path(__file__).parent.resolve()
# os.chdir(where_am_i)

# with open("default_paths.cfg.json") as f:
#     default_paths = json.load(f)

# PRE_TRAINED_NN = default_paths["PRE_TRAINED_NN"]
# PRE_TRAINED_GC = default_paths["PRE_TRAINED_GC"]

PRE_TRAINED_NN = PRE_TRAINED_GC = ""

from ..config.root_logger import get_logger

logger = get_logger()


def smart_get_latest():
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
