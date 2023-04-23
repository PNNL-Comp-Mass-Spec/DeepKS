from ..tools import custom_logging
import pathlib, os, json

join_first = lambda levels, x: os.path.join(pathlib.Path(__file__).parent.resolve(), *[".."] * levels, x)


def get_logger():
    with open(join_first(1, __file__)) as f:
        kwargs = json.load(f)
        good_keys = {"logging_level", "output_method"}
        for k, v in kwargs.items():
            assert k in good_keys
            if k == "logging_level":
                assert isinstance(v, int) and v >= 0
            elif k == "output_method":
                assert isinstance(v, str) and k in {"console", "logfile"}
        logger = custom_logging.CustomLogger(**kwargs)
    return logger
