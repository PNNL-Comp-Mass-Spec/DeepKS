import sys, termcolor, os, pathlib, warnings, json
from ..api import main
from termcolor import colored

from ..config.logging import get_logger

logger = get_logger()

DEVICE = os.environ.get("DEVICE", "cpu")

EXAMPLES = []
with open(os.path.join(pathlib.Path(__file__).parent.resolve(), "tests/examples.json")) as f:
    EXAMPLES = json.load(f)
    for ex in EXAMPLES:
        if "DEVICE_PLACEHOLDER" in ex:
            ex[ex.index("DEVICE_PLACEHOLDER")] = DEVICE


def _main():
    os.chdir(pathlib.Path(__file__).parent.resolve())

    print(
        colored(
            (
                "Info: This is an example script for DeepKS. To inspect the sample input files, check the"
                " 'DeepKS/tests/sample_inputs/' directory."
            ),
            "blue",
        )
    )

    inds = range(len(EXAMPLES)) if "--ex-list" not in sys.argv else eval(sys.argv[sys.argv.index("--ex-list") + 1])

    for i, example in zip(inds, [EXAMPLES[i] for i in inds]):
        if i > 0:
            for _ in range(4):
                print()
        print(colored(f"[Example {i+1}/{len(EXAMPLES)}] Simulating the following command line from `/`:", "yellow"))
        print()
        print(termcolor.colored(" ".join(example) + "\n", "yellow"))
        sys.argv = example
        with warnings.catch_warnings():
            main.setup()

    logger.info("All Examples Complete.")


if __name__ == "__main__":
    _main()
