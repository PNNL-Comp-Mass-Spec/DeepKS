import unittest
import sys
import argparse


class TestMainAPIFromCMDL(unittest.TestCase):
    def setUp(self):
        global main
        from ..api import main
        pass

    def test_dict(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-k", "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD", "-s", "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA", "-p", "dictionary", "--scores"]
        main.pre_main()

    def test_in_order(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-k", "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD", "-s", "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA", "-p", "in_order", "-v", "--normalize-scores", "--suppress-seqs-in-output"]
        main.pre_main()

    def test_in_order_json(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-k", "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD", "-s", "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA", "-p", "in_order_json", "-v"]
        main.pre_main()

    def test_dict_json(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-k", "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD", "-s", "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA", "-p", "dictionary_json"]
        main.pre_main()

    def test_load_from_file(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-kf", "../tests/sample_inputs/kins.txt", "-sf", "../tests/sample_inputs/sites.txt", "-p", "dictionary", "-v"]
        main.pre_main()

    def test_cartesian_product(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-kf", "../tests/sample_inputs/kins.txt", "-sf", "../tests/sample_inputs/sites-prod.txt", "-p", "dictionary_json", "-v", "--dry-run", "--kin-info", "../tests/sample_inputs/kin-info.json", "--site-info", "../tests/sample_inputs/site-info.json", "--cartesian-product"]
        main.pre_main()

        sys.argv = ["python3 -m DeepKS.api.main", "-kf", "../tests/sample_inputs/kins.txt", "-sf", "../tests/sample_inputs/sites-prod.txt", "-p", "dictionary_json", "-v", "--kin-info", "../tests/sample_inputs/kin-info.json", "--site-info", "../tests/sample_inputs/site-info.json", "--scores", "--normalize-scores", "--cartesian-product"]
        main.pre_main()
    
    def test_bad_devices(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-kf", "../tests/sample_inputs/kins.txt", "-sf", "../tests/sample_inputs/sites.txt", "-p", "dictionary", "-v", "--device", "cpuu"]
        self.assertRaises(SystemExit, main.pre_main) # TODO - add error code check

        sys.argv = ["python3 -m DeepKS.api.main", "-kf", "../tests/sample_inputs/kins.txt", "-sf", "../tests/sample_inputs/sites.txt", "-p", "dictionary", "-v", "--device", "cudaa"]
        self.assertRaises(SystemExit, main.pre_main)

        sys.argv = ["python3 -m DeepKS.api.main", "-kf", "../tests/sample_inputs/kins.txt", "-sf", "../tests/sample_inputs/sites.txt", "-p", "dictionary", "-v", "--device", "cuda:999"]
        self.assertRaises(SystemExit, main.pre_main)

    def test_get_help(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-h"]
        self.assertRaises(SystemExit, main.pre_main)

    

# if __name__ == '__main__':
#     runner = unittest.TextTestRunner()
#     tmapifcmdl = unittest.TestLoader().loadTestsFromTestCase(TestMainAPIFromCMDL)
#     runner.run(tmapifcmdl)