"""The main example script"""

import logging
import sys, os, pathlib, warnings, jstyleson as json
from ..api import main
from ..config.join_first import join_first
from ..config.logging import get_logger
import atexit
from termcolor import colored

logger = get_logger()
"""The logger for this script."""

DEVICE = os.environ.get("DEVICE", "cpu")
"""The device to use for training. Can be configured with the `DEVICE` environment variable."""

with open(join_first("tests/examples.jsonc", 1, __file__)) as f:
    EXAMPLES = json.load(f)
    """Global list of lists that are command line arguments, each of which is an example."""
    for ex in EXAMPLES:
        if "DEVICE_PLACEHOLDER" in ex:
            ex[ex.index("DEVICE_PLACEHOLDER")] = DEVICE


def _main():
    """Main example function that runs examples"""
    os.chdir(pathlib.Path(__file__).parent.resolve())

    logger.info(
        "This is an example script for DeepKS. To inspect the sample input files, check the"
        " 'DeepKS/tests/sample_inputs/' directory.\n\n"
    )

    if "--ex-list" not in sys.argv:
        inds = range(len(EXAMPLES))
    else:
        inds = eval(sys.argv[sys.argv.index("--ex-list") + 1])

    for i, example in zip(inds, [EXAMPLES[i] for i in inds]):
        print()
        logger.status(
            f"[Example {i+1}/{len(EXAMPLES)}] Simulating the following command line from the directory"
            f" '/':\n{colored('$', 'black', attrs=['bold'])} {colored(' '.join(example), 'black')}\n"
        )
        sys.argv = example
        with warnings.catch_warnings():
            main.setup()

    logger.info("All Examples Complete.")


if __name__ == "__main__":  # pragma: no cover
    _main()
