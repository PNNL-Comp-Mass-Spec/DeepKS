import unittest
import sys


class TestMainAPIFromCMDL(unittest.TestCase):
    def setUp(self):
        global main
        from DeepKS.api import main
        pass

    def test_dict(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-k", "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD", "-s", "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA", "-p", "dictionary", "--scores"]
        main._cmd_testing_simulator()

    def test_in_order(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-k", "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD", "-s", "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA", "-p", "in_order", "-v"]
        main._cmd_testing_simulator()

    def test_in_order_json(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-k", "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD", "-s", "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA", "-p", "in_order_json", "-v"]
        main._cmd_testing_simulator()

    def test_dict_json(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-k", "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD", "-s", "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA", "-p", "dictionary_json"]
        main._cmd_testing_simulator()

    def test_load_from_file(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-kf", "../tests/sample_inputs/kins.txt", "-sf", "../tests/sample_inputs/sites.txt", "-p", "dictionary", "-v"]
        main._cmd_testing_simulator()

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    tmapifcmdl = unittest.TestLoader().loadTestsFromTestCase(TestMainAPIFromCMDL)
    runner.run(tmapifcmdl)