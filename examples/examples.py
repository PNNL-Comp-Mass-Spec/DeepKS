import sys, termcolor, os, pathlib, warnings
from ..api import main
from termcolor import colored


def _main():
    os.chdir(pathlib.Path(__file__).parent.resolve())

    print(
        colored(
            (
                "Info: This is an example script for DeepKS. To inspect the sample input files, check the"
                " 'examples/sample_inputs' directory."
            ),
            "blue",
        )
    )
    examples = [
        [
            "python3 -m DeepKS.api.main",
            "-k",
            "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD",
            "-s",
            "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA",
            "-p",
            "in_order",
            "-v"
        ],
        [
            "python3 -m DeepKS.api.main",
            "-kf",
            "tests/sample_inputs/kins.txt",
            "-sf",
            "tests/sample_inputs/sites.txt",
            "-p",
            "dictionary",
            "-v"
        ],
        [
            "python3 -m DeepKS.api.main",
            "-kf",
            "tests/sample_inputs/kins.txt",
            "-sf",
            "tests/sample_inputs/sites-prod.txt",
            "-p",
            "dictionary_json",
            "--kin-info",
            "tests/sample_inputs/kin-info.json",
            "--site-info",
            "tests/sample_inputs/site-info.json",
            "--scores",
            "--normalize-scores",
            "--cartesian-product",
        ],
    ]

    for i, example in enumerate(examples):
        if i > 0:
            for _ in range(4):
                print()
        print(colored(f"[Example {i+1}/{len(examples)}] Simulating the following command line from `DeepKS/`:", "yellow"))
        print()
        print(termcolor.colored(" ".join(example) + "\n", "yellow"))
        sys.argv = example
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            main.pre_main()

    print(colored("Info: All Examples Complete.", "blue"))
