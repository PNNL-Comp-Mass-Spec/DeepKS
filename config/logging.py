from ..tools import custom_logging
import pathlib, os, json

join_first = lambda levels, x: os.path.join(pathlib.Path(__file__).parent.resolve(), *[".."] * levels, x)

import warnings


def splash(splash_file):
    from ..tools.splash.write_splash import write_splash

    write_splash(splash_file)


def get_logger():
    try:
        with open(join_first(1, __file__)) as f:
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
