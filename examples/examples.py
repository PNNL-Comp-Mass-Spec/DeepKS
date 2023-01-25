from ..api import main
import sys, termcolor

print("Info: This is an example script for DeepKS. To inspect the sample input files, check the 'examples/sample_inputs' directory.")
print()
print("Info: [Example 1/3] Simulating the following command line from `DeepKS/`:")
print()
print(termcolor.colored(" ".join(example_1 := ["python3 -m DeepKS.api.main", "-kf", "../tests/sample_inputs/kins.txt", "-sf", "../tests/sample_inputs/sites.txt", "-p", "dictionary", "-v"]) + "\n", "blue"))
example_1 = [x.replace("../tests/sample_inputs/", "examples/sample_inputs/") for x in example_1]
sys.argv = example_1
main._cmd_testing_simulator()

print("TODO: Put in the rest of the example code here.")