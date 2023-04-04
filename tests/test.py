import sys, os, argparse, unittest, pathlib, json, inspect
from parameterized import parameterized

DEVICE = os.environ.get("DEVICE", "cpu")


class UsesR:
    pass


class TestMisc(unittest.TestCase):
    # def setUp(self):
    #     from ..models.individual_classifiers import smart_save_nn as this_smart_save_nn, IndividualClassifiers

    #     self.smart_save_nn = this_smart_save_nn
    #     self.IndividualClassifiers = IndividualClassifiers
    #     self.real_file = "UNITTESTVERSIONdeepks_nn_weights.-1.cornichon"
    #     self.parent = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "bin")
    #     # make backup
    #     assert os.path.exists(os.path.join(self.parent, self.real_file))
    #     with open(os.path.join(self.parent, self.real_file), "rb") as fl:
    #         self.bu = fl.read()

    def test_identity_placeholder(self):
        assert True, "Identity"

    # def test_smart_save_nn(self):
    #     self.sample_files = ["UNITTESTVERSIONdeepks_nn_weights.0.cornichon",
    #                     "UNITTESTVERSIONdeepks_nn_weights.2.cornichon",
    #                     "UNITTESTVERSIONdeepks_nn_weights.5.cornichon"]
    #     self.old_dir = os.getcwd()
    #     os.chdir(self.parent)
    #     assert os.path.exists(self.real_file)
    #     for f in self.sample_files:
    #         with open(f, "w") as fl:
    #             fl.write("This is a test nn weight file.")

    #     self.smart_save_nn(self.IndividualClassifiers.load_all(self.real_file))
    #     assert os.path.exists(os.path.join(self.parent, "deepks_nn_weights.6.cornichon"))

    # def tearDown(self):
    #     [os.unlink(f) for f in self.sample_files + ["deepks_nn_weights.6.cornichon"]]
    #     os.chdir(self.old_dir)
    #     # restore backup
    #     with open(os.path.join(self.parent, self.real_file), "wb") as restore:
    #         restore.write(self.bu)


class TestPreprocessing(unittest.TestCase, UsesR):
    def setUp(self):
        global main
        from ..data.preprocessing import main as this_main

        self.main = this_main

    def test_run_main_no_mtx_to_add(self):
        if "DEBUGGING" in os.environ:
            del os.environ["DEBUGGING"]
        self.main.main()


class TestTrainingIndividualClassifiers(unittest.TestCase):
    def setUp(self):
        from ..models.individual_classifiers import main as this_main

        self.main = this_main

    def test_a_train_nn_small(self):
        sys.argv = [
            "python3 -m DeepKS.models.individual_classifiers",
            "--train",
            "tests/sample_inputs/small_train.csv",
            "--val",
            "tests/sample_inputs/small_val_or_test.csv",
            "--device",
            "cpu",
            "-s",
        ]
        self.main()

    def test_b_test_nn_small(self):
        sys.argv = [
            "python3 -m DeepKS.models.multi_state_classifier",
            "--test",
            "tests/sample_inputs/small_train.csv",  # TODO may want to change this in future
            "--load",
            "/home/dockeruser/DeepKS/bin/deepks_nn_weights.1.cornichon",  # TODO Fix quick stop-gap
            "--device",
            "cpu",
            "-s",
        ]
        self.main()

    def test_c_train_nn_small_from_another_dir(self):
        old_dir = os.getcwd()
        os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve())
        sys.argv = [
            "python3 -m DeepKS.models.individual_classifiers",
            "--train",
            "tests/sample_inputs/small_train.csv",
            "--val",
            "tests/sample_inputs/small_val_or_test.csv",
            "--device",
            "cpu",
            "-s",
        ]
        self.main()
        os.chdir(old_dir)


class TestTrainingGroupClassifier(unittest.TestCase):
    def setUp(self):
        from ..models.multi_stage_classifier import main as this_main

        self.main = this_main

    def test_train_gc_small(self):
        from DeepKS.api.cfg import PRE_TRAINED_NN

        sys.argv = [
            "python3 -m DeepKS.models.multi_stage_classifier",
            "--load",
            PRE_TRAINED_NN,
            "--device",
            "cpu",
            "-c",
            "--test",
            "tests/sample_inputs/small_val_or_test.csv",
        ]
        self.main()


class TestGenerateFigures(unittest.TestCase):
    def test_sunburst(self):
        from ..images.Sunburst import sunburst

        sunburst.make_sunburst()


class TestMainAPIFromCMDL(unittest.TestCase):
    def setUp(self):
        from ..api import main as this_main

        self.main = this_main

    def test_convert_raw_to_prob(self):
        sys.argv = [
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
            "--cartesian-product",
            "--groups",
            "--convert-raw-to-prob",
            "--device",
            DEVICE,
        ]
        self.main.setup()

    def test_dict(self):
        sys.argv = [
            "python3 -m DeepKS.api.main",
            "-k",
            "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD",
            "-s",
            "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA",
            "-p",
            "dictionary",
            "--scores",
            "--device",
            DEVICE,
        ]
        self.main.setup()

    def test_in_order(self):
        sys.argv = [
            "python3 -m DeepKS.api.main",
            "-k",
            "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD",
            "-s",
            "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA",
            "-p",
            "inorder",
            "-v",
            "--normalize-scores",
            "--suppress-seqs-in-output",
            "--device",
            DEVICE,
        ]
        self.main.setup()

    def test_in_order_json(self):
        sys.argv = [
            "python3 -m DeepKS.api.main",
            "-k",
            "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD",
            "-s",
            "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA",
            "-p",
            "in_order_json",
            "-v",
            "--device",
            DEVICE,
        ]
        self.main.setup()

    def test_dict_json(self):
        sys.argv = [
            "python3 -m DeepKS.api.main",
            "-k",
            "TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD",
            "-s",
            "VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA",
            "-p",
            "dictionary_json",
            "--device",
            DEVICE,
        ]
        self.main.setup()

    def test_load_from_file(self):
        sys.argv = [
            "python3 -m DeepKS.api.main",
            "-kf",
            "tests/sample_inputs/kins.txt",
            "-sf",
            "tests/sample_inputs/sites.txt",
            "-p",
            "dictionary",
            "-v",
            "--device",
            DEVICE,
        ]
        self.main.setup()

    def test_cartesian_product(self):
        sys.argv = [
            "python3 -m DeepKS.api.main",
            "-kf",
            "tests/sample_inputs/kins.txt",
            "-sf",
            "tests/sample_inputs/sites-prod.txt",
            "-p",
            "dictionary_json",
            "-v",
            "--dry-run",
            "--kin-info",
            "tests/sample_inputs/kin-info.json",
            "--site-info",
            "tests/sample_inputs/site-info.json",
            "--cartesian-product",
            "--device",
            DEVICE,
        ]
        self.main.setup()

        sys.argv = [
            "python3 -m DeepKS.api.main",
            "-kf",
            "tests/sample_inputs/kins.txt",
            "-sf",
            "tests/sample_inputs/sites-prod.txt",
            "-p",
            "dictionary_json",
            "-v",
            "--kin-info",
            "tests/sample_inputs/kin-info.json",
            "--site-info",
            "tests/sample_inputs/site-info.json",
            "--scores",
            "--normalize-scores",
            "--cartesian-product",
            "--device",
            DEVICE,
        ]
        self.main.setup()

    def test_cartesian_product_csv(self):
        sys.argv = [
            "python3 -m DeepKS.api.main",
            "-kf",
            "tests/sample_inputs/kins.txt",
            "-sf",
            "tests/sample_inputs/sites-prod.txt",
            "-p",
            "csv",
            "--kin-info",
            "tests/sample_inputs/kin-info.json",
            "--site-info",
            "tests/sample_inputs/site-info.json",
            "--scores",
            "--groups",
            "--normalize-scores",
            "--cartesian-product",
            "--device",
            DEVICE,
        ]
        self.main.setup()

    def test_cartesian_product_db(self):
        sys.argv = [
            "python3 -m DeepKS.api.main",
            "-kf",
            "tests/sample_inputs/kins.txt",
            "-sf",
            "tests/sample_inputs/sites-prod.txt",
            "-p",
            "sqlite",
            "--kin-info",
            "tests/sample_inputs/kin-info.json",
            "--site-info",
            "tests/sample_inputs/site-info.json",
            "--scores",
            "--groups",
            "--normalize-scores",
            "--cartesian-product",
            "--device",
            DEVICE,
        ]
        self.main.setup()

    def test_bad_devices(self):
        sys.argv = [
            "python3 -m DeepKS.api.main",
            "-kf",
            "tests/sample_inputs/kins.txt",
            "-sf",
            "tests/sample_inputs/sites.txt",
            "-p",
            "dictionary",
            "-v",
            "--device",
            "cpuu",
        ]
        self.assertRaises(SystemExit, self.main.setup)

        sys.argv = [
            "python3 -m DeepKS.api.main",
            "-kf",
            "tests/sample_inputs/kins.txt",
            "-sf",
            "tests/sample_inputs/sites.txt",
            "-p",
            "dictionary",
            "-v",
            "--device",
            "cudaa",
        ]
        self.assertRaises(SystemExit, self.main.setup)

        sys.argv = [
            "python3 -m DeepKS.api.main",
            "-kf",
            "tests/sample_inputs/kins.txt",
            "-sf",
            "tests/sample_inputs/sites.txt",
            "-p",
            "dictionary",
            "-v",
            "--device",
            "cuda:999",
        ]
        self.assertRaises(SystemExit, self.main.setup)

    def test_get_help(self):
        sys.argv = ["python3 -m DeepKS.api.main", "-h"]
        with self.assertRaises(SystemExit) as ar:
            self.main.setup()
            self.assertEqual(ar.exception.code, 0)


with open(os.path.join(pathlib.Path(__file__).parent.resolve(), "examples.json")) as f:
    EXAMPLES = json.load(f)
    for ex in EXAMPLES:
        if "DEVICE_PLACEHOLDER" in ex:
            ex[ex.index("DEVICE_PLACEHOLDER")] = DEVICE


class TestExamples(unittest.TestCase):
    def setUp(self):
        from ..api import main as this_main

        self.main = this_main

    @parameterized.expand([[f"ex_{i}"] + EXAMPLES[i] for i in range(len(EXAMPLES))])
    def test_examples(self, name, *ex):
        # print(f"{name=}")
        # print(f"{ex=}")
        sys.argv = list(ex)
        self.main.setup()


api_suite = unittest.TestSuite()
training_suite = unittest.TestSuite()
non_r_suite = unittest.TestSuite()

testloader = unittest.TestLoader()

api_suite.addTests(
    [
        testloader.loadTestsFromTestCase(TestExamples),
        testloader.loadTestsFromTestCase(TestMainAPIFromCMDL),
    ]
)

training_suite.addTests(
    [
        testloader.loadTestsFromTestCase(TestTrainingIndividualClassifiers),
        testloader.loadTestsFromTestCase(TestTrainingGroupClassifier),
    ]
)

is_non_r_test = lambda x: isinstance(x, type) and issubclass(x, unittest.TestCase) and not issubclass(x, UsesR)

non_r_tests = [obj_type for _, obj_type in inspect.getmembers(sys.modules[__name__]) if is_non_r_test(obj_type)]

if any(".non_r_tests" in a for a in sys.argv):
    print(f"Running the following non-r-tests:")
    for test in non_r_tests:
        print(f"  * {test.__name__}")

non_r_suite.addTests([testloader.loadTestsFromTestCase(test) for test in non_r_tests])

pass
pass
