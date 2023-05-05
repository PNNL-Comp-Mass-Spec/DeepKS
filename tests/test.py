"""All the tests for DeepKS"""
import sys, os, functools, unittest, pathlib, json, inspect
from parameterized import parameterized

DEVICE = os.environ.get("DEVICE", "cpu")
"""Device `torch` will use. Can be set with the `DEVICE` environment variable."""

from ..config.join_first import join_first


class UsesR:
    """Empty class used to mark tests that use R."""

    pass


from ..models.GroupClassifier import (
    GroupClassifier,
    GCPrediction,
    SiteGroupClassifier,
    KinGroupClassifier,
    PseudoSiteGroupClassifier,
)

import __main__

setattr(__main__, "PseudoSiteGroupClassifier", PseudoSiteGroupClassifier)


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


class TestAAAPreprocessing(unittest.TestCase, UsesR):
    def setUp(self):
        global main
        from ..data.preprocessing import main as this_main
        from ..data.preprocessing.main import step_1_download_psp, step_2_download_uniprot
        from ..data.preprocessing.main import step_3_get_kin_to_fam_to_grp, step_4_get_pairwise_mtx
        from ..data.preprocessing.main import step_5_get_train_val_test_split

        self.step_1 = step_1_download_psp
        self.step_2 = step_2_download_uniprot
        self.step_3 = step_3_get_kin_to_fam_to_grp
        self.step_4 = step_4_get_pairwise_mtx
        self.step_5_a = functools.partial(step_5_get_train_val_test_split, eval_or_train_on_all="E")
        self.step_5_b = functools.partial(step_5_get_train_val_test_split, eval_or_train_on_all="T")

    def test_step_1_preproc(self):
        self.step_1()

    def test_step_2_preproc(self):
        a, r = self.step_2()
        setattr(self.__class__, "seq_filename_A", a)
        setattr(self.__class__, "raw_data_filename", r)

    def test_step_3_preproc(self):
        self.kin_fam_grp_filename = self.step_3(
            getattr(self.__class__, "seq_filename_A", "../raw_data/kinase_seq_833.csv")
        )

    def test_step_4_preproc(self):
        new_mtx_filename = self.step_4(
            getattr(self.__class__, "seq_filename_A", "../raw_data/kinase_seq_833.csv"),
        )
        setattr(self.__class__, "new_mtx_file", new_mtx_filename)

    def test_step_5_a_preproc(self):
        self.step_5_a(
            getattr(self.__class__, "kin_fam_grp_filename", "../kin_to_fam_to_grp_828.csv"),
            getattr(self.__class__, "raw_data_filename", "../raw_data/raw_data_22769.csv"),
            getattr(self.__class__, "seq_filename_A", "../raw_data/kinase_seq_833.csv"),
            getattr(self.__class__, "new_mtx_file", "../preprocessing/pairwise_mtx_828.csv"),
        )

    def test_step_5_b_preproc(self):
        self.step_5_b(
            getattr(self.__class__, "kin_fam_grp_filename", "../kin_to_fam_to_grp_828.csv"),
            getattr(self.__class__, "raw_data_filename", "../raw_data/raw_data_22769.csv"),
            getattr(self.__class__, "seq_filename_A", "../raw_data/kinase_seq_833.csv"),
            getattr(self.__class__, "new_mtx_file", "../preprocessing/pairwise_mtx_828.csv"),
        )


class TestTrainingIndividualClassifiers(unittest.TestCase):
    def setUp(self):
        from ..models.individual_classifiers import main as this_main

        self.main = this_main

    def test_train_nn_small(self):
        sys.argv = [
            "python3 -m DeepKS.models.individual_classifiers",
            "--train",
            "tests/sample_inputs/small_train.csv",
            "--val",
            "tests/sample_inputs/small_val_or_test.csv",
            "--device",
            DEVICE,
            "--pre-trained-gc",
            "bin/deepks_gc_weights.-1.cornichon",
            "--s-test",
        ]
        self.main()

    def test_train_nn_small_from_another_dir(self):
        old_dir = os.getcwd()
        os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve())
        sys.argv = [
            "python3 -m DeepKS.models.individual_classifiers",
            "--train",
            "tests/sample_inputs/small_train.csv",
            "--val",
            "tests/sample_inputs/small_val_or_test.csv",
            "--device",
            DEVICE,
            "--pre-trained-gc",
            "bin/deepks_gc_weights.-1.cornichon",
            "--s-test",
        ]
        self.main()
        os.chdir(old_dir)


class TestTrainingGroupClassifier(unittest.TestCase):
    def setUp(self):
        pass

    def test_train_gc_small(self):
        PseudoSiteGroupClassifier.package(
            join_first("data/raw_data/raw_data_45176_formatted_65.csv", 1, __file__),
            join_first("data/preprocessing/kin_to_fam_to_grp_826.csv", 1, __file__),
            is_testing=True,
        )


class TestGenerateFigures(unittest.TestCase):
    def test_sunburst(self):
        from ..images.Sunburst import sunburst

        sunburst.make_sunburst(
            "../../data/kin_to_fam_to_grp_828.csv", "../../data/raw_data/raw_data_22769.csv"
        )  # TODO: Make this automatic or have config json file


class TestMainAPIFromCMDL(unittest.TestCase):
    def setUp(self):
        from ..api import main as this_main

        self.main = this_main

    # def test_convert_raw_to_prob(self):
    #     sys.argv = [
    #         "python3 -m DeepKS.api.main",
    #         "-kf",
    #         "tests/sample_inputs/kins.txt",
    #         "-sf",
    #         "tests/sample_inputs/sites-prod.txt",
    #         "-p",
    #         "csv",
    #         "--kin-info",
    #         "tests/sample_inputs/kin-info-known-groups.json",
    #         "--site-info",
    #         "tests/sample_inputs/site-info.json",
    #         "--scores",
    #         "--cartesian-product",
    #         "--groups",
    #         "--convert-raw-to-prob",
    #         "--pre-trained-nn",
    #         "bin/deepks_nn_weights.-1.cornichon",
    #         "--device",
    #         DEVICE,
    #         "--pre-trained-nn",
    #         "bin/deepks_nn_weights.-1.cornichon",
    #         "--pre-trained-gc",
    #         "bin/deepks_gc_weights.-1.cornichon"
    #     ]
    #     self.main.setup()

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
            "--pre-trained-nn",
            "bin/deepks_nn_weights.-1.cornichon",
            "--pre-trained-gc",
            "bin/deepks_gc_weights.-1.cornichon",
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
            "--pre-trained-nn",
            "bin/deepks_nn_weights.-1.cornichon",
            "--pre-trained-gc",
            "bin/deepks_gc_weights.-1.cornichon",
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
            "--pre-trained-nn",
            "bin/deepks_nn_weights.-1.cornichon",
            "--pre-trained-gc",
            "bin/deepks_gc_weights.-1.cornichon",
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
            "--pre-trained-nn",
            "bin/deepks_nn_weights.-1.cornichon",
            "--pre-trained-gc",
            "bin/deepks_gc_weights.-1.cornichon",
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
            "--pre-trained-nn",
            "bin/deepks_nn_weights.-1.cornichon",
            "--pre-trained-gc",
            "bin/deepks_gc_weights.-1.cornichon",
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
            "--pre-trained-nn",
            "bin/deepks_nn_weights.-1.cornichon",
            "--pre-trained-gc",
            "bin/deepks_gc_weights.-1.cornichon",
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
            "--pre-trained-nn",
            "bin/deepks_nn_weights.-1.cornichon",
            "--pre-trained-gc",
            "bin/deepks_gc_weights.-1.cornichon",
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
            "--pre-trained-nn",
            "bin/deepks_nn_weights.-1.cornichon",
            "--pre-trained-gc",
            "bin/deepks_gc_weights.-1.cornichon",
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
            "--pre-trained-nn",
            "bin/deepks_nn_weights.-1.cornichon",
            "--pre-trained-gc",
            "bin/deepks_gc_weights.-1.cornichon",
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
            "--pre-trained-nn",
            "bin/deepks_nn_weights.-1.cornichon",
            "--pre-trained-gc",
            "bin/deepks_gc_weights.-1.cornichon",
        ]
        with self.assertRaises(SystemExit) as ar:
            self.main.setup()

        self.assertEqual(ar.exception.code, 1)

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
            "--pre-trained-nn",
            "bin/deepks_nn_weights.-1.cornichon",
            "--pre-trained-gc",
            "bin/deepks_gc_weights.-1.cornichon",
        ]
        with self.assertRaises(SystemExit) as ar:
            self.main.setup()

        self.assertEqual(ar.exception.code, 1)

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
            "--pre-trained-nn",
            "bin/deepks_nn_weights.-1.cornichon",
            "--pre-trained-gc",
            "bin/deepks_gc_weights.-1.cornichon",
        ]
        with self.assertRaises(SystemExit) as ar:
            self.main.setup()

        self.assertEqual(ar.exception.code, 1)

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


api_and_training_suite = unittest.TestSuite()
"""Suite of all tests that test the API and training."""
api_suite = unittest.TestSuite()
"""Suite of tests that test the API."""
training_suite = unittest.TestSuite()
"""Suite of tests that test the training."""
non_r_suite = unittest.TestSuite()
"""Suite of all tests that test the non-R functionality."""

testloader = unittest.TestLoader()
"""Test loader for all tests. Just using `unittest.TestLoader.loadTestsFromTestCase`."""

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

api_and_training_suite.addTests([api_suite, training_suite])

is_non_r_test = lambda x: isinstance(x, type) and issubclass(x, unittest.TestCase) and not issubclass(x, UsesR)
"""Simple lambda to determine if the given object is a non-R test."""

non_r_tests = [obj_type for _, obj_type in inspect.getmembers(sys.modules[__name__]) if is_non_r_test(obj_type)]
"""List of all non-R tests."""

if any(".non_r_tests" in a for a in sys.argv):
    print(f"Running the following non-r-tests:")
    for test in non_r_tests:
        print(f"  * {test.__name__}")

non_r_suite.addTests([testloader.loadTestsFromTestCase(test) for test in non_r_tests])
