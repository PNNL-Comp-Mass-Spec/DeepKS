"""Default entry point for running examples."""
from copy import deepcopy
from ..config.join_first import join_first
import logging
from ..tools import custom_logging
import json, atexit


def exit_fn():
    with open(join_first("config/logging_config.json", 1, __file__), "w") as f:
        json.dump(init_config_dict, f, indent=3)


atexit.register(exit_fn)

with open(join_first("config/logging_config.json", 1, __file__)) as f:
    cd = json.load(f)
init_config_dict = deepcopy(cd)
cd["logging_level"] = custom_logging.STATUS
cd["upper_logging_level"] = logging.WARNING - 1
with open(join_first("config/logging_config.json", 1, __file__), "w") as f:
    json.dump(cd, f, indent=3)

from . import examples
from ..tools.splash.write_splash import write_splash

import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn")

write_splash("examples")

examples._main()
