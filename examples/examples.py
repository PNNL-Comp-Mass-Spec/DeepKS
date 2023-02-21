import sys, termcolor, os, pathlib, warnings
from ..api import main
from termcolor import colored

EXAMPLES = [
        [
            "python3 -m DeepKS.api.main",
            "-k",
            "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD",
            "-s",
            "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA",
            "-p",
            "inorder",
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
    

    for i, example in enumerate(EXAMPLES):
        if i > 0:
            for _ in range(4):
                print()
        print(colored(f"[Example {i+1}/{len(EXAMPLES)}] Simulating the following command line from `DeepKS/`:", "yellow"))
        print()
        print(termcolor.colored(" ".join(example) + "\n", "yellow"))
        sys.argv = example
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            main.setup()

    print(colored("Info: All Examples Complete.", "blue"))
