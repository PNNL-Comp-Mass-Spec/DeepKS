import unittest
import sys, os

class TestPreProcessing(unittest.TestCase):
    def setUp(self):
        global main
        from ..data.preprocessing import main as this_main
        self.main = this_main
    
    def test_run_main_no_mtx_to_add(self):
        if "DEBUGGING" in os.environ:
            del os.environ["DEBUGGING"]
        self.main.main()

class TestTrainingProcess(unittest.TestCase):
    pass

class TestGenerateFigures(unittest.TestCase):
    def test_sunburst(self):
        from ..images.Sunburst import sunburst
        sunburst.make_sunburst()

class TestMainAPIFromCMDL(unittest.TestCase):
    def setUp(self):
        from ..api import main as this_main
        self.main = this_main

    def test_dict(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-k", "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD", "-s", "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA", "-p", "dictionary", "--scores"]
        self.main.setup()

    def test_in_order(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-k", "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD", "-s", "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA", "-p", "in_order", "-v", "--normalize-scores", "--suppress-seqs-in-output"]
        self.main.setup()

    def test_in_order_json(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-k", "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD", "-s", "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA", "-p", "in_order_json", "-v"]
        self.main.setup()

    def test_dict_json(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-k", "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD", "-s", "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA", "-p", "dictionary_json"]
        self.main.setup()

    def test_load_from_file(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-kf", "tests/sample_inputs/kins.txt", "-sf", "tests/sample_inputs/sites.txt", "-p", "dictionary", "-v"]
        self.main.setup()

    def test_cartesian_product(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-kf", "tests/sample_inputs/kins.txt", "-sf", "tests/sample_inputs/sites-prod.txt", "-p", "dictionary_json", "-v", "--dry-run", "--kin-info", "tests/sample_inputs/kin-info.json", "--site-info", "tests/sample_inputs/site-info.json", "--cartesian-product"]
        self.main.setup()

        sys.argv = ["python3 -m DeepKS.api.main", "-kf", "tests/sample_inputs/kins.txt", "-sf", "tests/sample_inputs/sites-prod.txt", "-p", "dictionary_json", "-v", "--kin-info", "tests/sample_inputs/kin-info.json", "--site-info", "tests/sample_inputs/site-info.json", "--scores", "--normalize-scores", "--cartesian-product"]
        self.main.setup()
    
    def test_bad_devices(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-kf", "tests/sample_inputs/kins.txt", "-sf", "tests/sample_inputs/sites.txt", "-p", "dictionary", "-v", "--device", "cpuu"]
        self.assertRaises(SystemExit, self.main.setup) # TODO - add error code check

        sys.argv = ["python3 -m DeepKS.api.main", "-kf", "tests/sample_inputs/kins.txt", "-sf", "tests/sample_inputs/sites.txt", "-p", "dictionary", "-v", "--device", "cudaa"]
        self.assertRaises(SystemExit, self.main.setup)

        sys.argv = ["python3 -m DeepKS.api.main", "-kf", "tests/sample_inputs/kins.txt", "-sf", "tests/sample_inputs/sites.txt", "-p", "dictionary", "-v", "--device", "cuda:999"]
        self.assertRaises(SystemExit, self.main.setup)

    def test_get_help(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-h"]
        self.assertRaises(SystemExit, self.main.setup)

class TestExamples(unittest.TestCase):
    def setUp(self):
        from ..api import main as this_main
        self.main = this_main
        from ..examples.examples import EXAMPLES

        self.examples = EXAMPLES
        
    def test_examples(self):
        for ex in self.examples:
            sys.argv = ex
            self.main.setup()


# if __name__ == '__main__':
#     runner = unittest.TextTestRunner()
#     tmapifcmdl = unittest.TestLoader().loadTestsFromTestCase(TestMainAPIFromCMDL)
#     runner.run(tmapifcmdl)