import sys, termcolor, os, pathlib, warnings
from ..api import main
from termcolor import colored

DEVICE = os.environ.get("DEVICE", "cpu")

EXAMPLES = [
        [
            "python3 -m DeepKS.api.main",
            "-k",
            "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD",
            "-s",
            "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA",
            "-p",
            "inorder",
            "-v",
            "--device", 
            DEVICE
        ],
        [
            "python3 -m DeepKS.api.main",
            "-kf",
            "tests/sample_inputs/kins.txt",
            "-sf",
            "tests/sample_inputs/sites.txt",
            "-p",
            "dictionary",
            "-v",
            "--device", 
            DEVICE
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
            "--device", 
            DEVICE
        ],
        [
            "python3 -m DeepKS.api.main",
            "-kf",
            "tests/sample_inputs/kins.txt",
            "-sf",
            "tests/sample_inputs/sites-prod.txt",
            "-p",
            "csv",
            "--kin-info",
            "tests/sample_inputs/kin-info-known-groups.json",
            "--site-info",
            "tests/sample_inputs/site-info.json",
            "--scores",
            "--normalize-scores",
            "--cartesian-product",
            "--groups",
            "--bypass-group-classifier",
            "--device", 
            DEVICE
        ],
    ]

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
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            main.setup()

    print(colored("Info: All Examples Complete.", "blue"))

if __name__ == "__main__":
    _main()