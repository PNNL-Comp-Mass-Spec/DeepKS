"""All the tests for DeepKS"""
import pickle, atexit, re, sys, os, functools, unittest, pathlib, json, inspect, warnings
from types import ModuleType
from parameterized import parameterized
from watchdog.utils.dirsnapshot import DirectorySnapshot, DirectorySnapshotDiff

from DeepKS.models.multi_stage_classifier import MultiStageClassifier
from ..config.join_first import join_first
from ..models.GroupClassifier import PseudoSiteGroupClassifier
import __main__

setattr(__main__, "PseudoSiteGroupClassifier", PseudoSiteGroupClassifier)

DEVICE = os.environ.get("DEVICE", "cpu")
"""Device `torch` will use. Can be set with the `DEVICE` environment variable."""

atexit.register(lambda: tearDownModule())


class UsesR:
    """Empty class used to mark tests that use R."""

    pass


def setUpModule():
    # print("Set Up")
    pre_setUp(sys.modules[__name__])


def tearDownModule():
    # print("Tear Down")
    post_tearDown(sys.modules[__name__])


def pre_setUp(self: unittest.TestCase | ModuleType, root: str = join_first("", 1, __file__)):
    init_snap = DirectorySnapshot(root)
    setattr(self, "init_snap", init_snap)


def post_tearDown(self: unittest.TestCase | ModuleType, root: str = join_first("", 1, __file__)):
    after_snap = DirectorySnapshot(root)
    try:
        init_snap = getattr(self, "init_snap")
    except AttributeError:
        warnings.warn(
            f"The Object {self} did not have an init_snap attribute (a snapshot of a directory before running the"
            " Object). Will not clean up any files that may have been created during the Object."
        )
        init_snap = after_snap
    diff = DirectorySnapshotDiff(init_snap, after_snap)
    for file in diff.files_created:
        os.remove(file)
    for dir in diff.dirs_created:
        os.rmdir(dir)
    do_not_warn_files = {
        r"__pycache__",
        r"ipynb_checkpoints",
        r"\.DS_Store",
        r"pytest_cache",
        r"\.pyc",
    }
    for file in diff.files_modified:
        for ptn in do_not_warn_files:
            if re.search(ptn, file):
                warnings.warn(f"File `{file}` was modified during the running of the Object {self}.")
                continue


class TestDiscovery(unittest.TestCase):
    """Test the `DeepKS.discovery` module."""

    ...


class TestEvaluation(unittest.TestCase):
    """Test the `DeepKS.models.DeepKS_evaluation` module."""

    def setUp(self):
        from ..models.DeepKS_evaluation import eval_and_roc_workflow as this_eval_and_roc_workflow
        from ..models.multi_stage_classifier import MultiStageClassifier as this_MultiStageClassifier
        from ..models.individual_classifiers import IndividualClassifiers

        self.eval_and_roc_workflow = this_eval_and_roc_workflow
        self.MultiStageClassifier = this_MultiStageClassifier
        this_load_all = IndividualClassifiers.load_all
        self.load_all = this_load_all

    def testEvaluation(self):
        with open(join_first("bin/deepks_gc_weights.-1.cornichon", 1, __file__), "rb") as gcf:
            gc = pickle.load(gcf)
        nn = self.load_all(join_first("bin/deepks_nn_weights.-1.cornichon", 1, __file__), DEVICE)
        msc = MultiStageClassifier(gc, nn)
        self.eval_and_roc_workflow(
            msc,
            join_first("data/preprocessing/kin_to_fam_to_grp_828.csv", 1, __file__),
            join_first("tests/sample_inputs/small_val_or_test.csv", 1, __file__),
            join_first("tests/sample_inputs/msc_resave.-1.cornichon", 1, __file__),
            force_recompute=True,
            bypass_gc=False,
        )


class TestTuning(unittest.TestCase):
    """Test the `DeepKS.tools.Tuner` module."""

    def setUp(self):
        from ..models.individual_classifiers import main as train_main
        from ..tools.Tuner import main as this_main

        self.train_main = train_main
        self.main = this_main
        self.cmdl = [
            "--train",
            "tests/sample_inputs/small_train.csv",
            "--val",
            "tests/sample_inputs/small_val_or_test.csv",
            "--device",
            DEVICE,
            "--pre-trained-gc",
            "bin/deepks_gc_weights.-1.cornichon",
            "--groups",
            "NON-TK",
        ]

    def testTuning(self):
        self.main(self.cmdl, self.train_main, num_samples=3, max_epoch=5)


class TestMisc(unittest.TestCase):
    """Test Miscellaneous functions."""

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
    """Run the preprocessing pipeline. (AAA is in the name to make sure this runs first.)"""

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
        self.step_5_a = functools.partial(step_5_get_train_val_test_split, eval_or_train_on_all="E", num_restarts=6)
        self.step_5_b = functools.partial(step_5_get_train_val_test_split, eval_or_train_on_all="T")

    # def test_all_preproc(self):
    #     self.step_1()
    #     seq_filename_A, raw_data_filename = self.step_2()
    #     kin_fam_grp_filename = self.step_3(seq_filename_A)
    #     new_mtx_filename = self.step_4(seq_filename_A)
    #     step_5_args = kin_fam_grp_filename, raw_data_filename, seq_filename_A, new_mtx_filename
    #     self.step_5_a(*step_5_args)
    #     self.step_5_b(*step_5_args)

    def test_step_1_preproc(self):
        self.step_1()

    def test_step_2_preproc(self):
        a, r = self.step_2()
        setattr(self.__class__, "seq_filename_A", a)
        setattr(self.__class__, "raw_data_filename", r)

    def test_step_3_preproc(self):
        kin_fam_grp_filename = self.step_3(getattr(self.__class__, "seq_filename_A", "../raw_data/kinase_seq_833.csv"))
        setattr(self.__class__, "kin_fam_grp_filename", kin_fam_grp_filename)

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
    """Test training different neural network models"""

    def setUp(self):
        from ..models.individual_classifiers import main as this_main

        self.main = this_main

    def test_train_nn(self):
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

    def test_train_nn_another_dir_LSTM(self):
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
            "--ksr-params",
            "tests/sample_inputs/sample_hp_configs/KSR_params_RNN.json",
        ]
        self.main()
        os.chdir(old_dir)

    def test_train_nn_ATTNWSELF(self):
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
            "--ksr-params",
            "tests/sample_inputs/sample_hp_configs/KSR_params_ATTNWSELF.json",
        ]
        self.main()


class TestTrainingGroupClassifier(unittest.TestCase):
    """Test training the group classifier"""

    def setUp(self):
        pass

    def test_train_gc_small(self):
        PseudoSiteGroupClassifier.package(
            join_first("data/raw_data/raw_data_45176_formatted_65.csv", 1, __file__),
            join_first("data/preprocessing/kin_to_fam_to_grp_826.csv", 1, __file__),
            is_testing=True,
        )


class TestGenerateFigures(unittest.TestCase):
    """Test generating figures"""

    def test_sunburst(self):
        from ..images.Sunburst import sunburst

        sunburst.make_sunburst(
            "data/preprocessing/kin_to_fam_to_grp_828.csv", "data/raw_data/raw_data_22769.csv"
        )  # TODO: Make this automatic or have config json file


class TestMainAPIFromCMDL(unittest.TestCase):
    """Test the main API (the main purpose) of `DeepKS`."""

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
    """Test the provided DeepKS examples."""

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
