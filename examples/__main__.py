"""Default entry point for running examples."""
from ..config.join_first import join_first
from ..tools import custom_logging
import json, atexit


# def exit_fn():
#     with open(join_first("config/logging_config.json", 1, __file__), "w") as f:
#         json.dump({}, f, indent=3)


# atexit.register(exit_fn)

# with open(join_first("config/logging_config.json", 1, __file__)) as f:
#     cd = json.load(f)
# cd["logging_level"] = custom_logging.STATUS
# cd["upper_logging_level"] = custom_logging.RESINFO
# with open(join_first("config/logging_config.json", 1, __file__), "w") as f:
#     json.dump(cd, f, indent=3)

from . import examples
from ..tools.splash.write_splash import write_splash

import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn")

write_splash("examples")

examples._main()
