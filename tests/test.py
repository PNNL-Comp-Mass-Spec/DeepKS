"""All the tests for DeepKS"""
import pickle, atexit, sys, os, functools, unittest, pathlib, json, inspect, time, signal
from git.repo import Repo
from parameterized import parameterized

from DeepKS.models.multi_stage_classifier import MultiStageClassifier
from ..config.join_first import join_first
from ..models.GroupClassifier import PseudoSiteGroupClassifier
import __main__

setattr(__main__, "PseudoSiteGroupClassifier", PseudoSiteGroupClassifier)

DEVICE = os.environ.get("DEVICE", "cpu")
"""Device `torch` will use. Can be set with the `DEVICE` environment variable."""


# atexit.register(lambda: tearDownModule())
def int_handler(signal, frame):
    tearDownModule()
    raise KeyboardInterrupt


def term_handler(signal, frame):
    tearDownModule()
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, int_handler)
signal.signal(signal.SIGTERM, term_handler)


class UsesR:
    """Empty class used to mark tests that use R."""

    pass


def setUpModule():
    pre_setUp()


def tearDownModule():
    post_tearDown()


def pre_setUp(root: str = join_first("", 1, __file__)):
    repo = Repo(root)
    repo.git.add("--all")


def post_tearDown(root: str = join_first("", 1, __file__)):
    if "KEEP" in os.environ:
        return
    repo = Repo(root)
    for item in repo.index.diff(None):
        # check if item is a file
        if os.path.isfile(os.path.join(root, item.a_path)):
            # discard the changes to the file
            repo.git.checkout("--", item.a_path)
    # discard any untracked, new files in the repo
    repo.git.clean("-df")


class TestDiscovery(unittest.TestCase):
    """Test the `DeepKS.discovery` module."""

    ...


class TestEvaluationn(unittest.TestCase):
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

    class TestTreeMaker(unittest.TestCase):
        """Test Tree Maker."""

        def setUp(self) -> None:
            from ..tools.new_treemaker import main as this_main

            self.main = this_main

        def test_tree_maker(self):
            self.main()

    def test_TestTreeMaker(self):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(self.TestTreeMaker)
        unittest.TextTestRunner().run(suite)


class TestAAAPreprocessing(unittest.TestCase, UsesR):
    """Run the preprocessing pipeline. (AAA is in the name to make sure this runs first.)"""

    def setUp(self):
        global main
        from ..data.preprocessing import main as this_main
        from ..data.preprocessing.main import step_1_download_psp, step_2_download_uniprot
        from ..data.preprocessing.main import step_3_get_kin_to_fam_to_grp, step_4_get_pairwise_mtx
        from ..data.preprocessing.main import step_5_get_train_val_test_split
        from ..data.preprocessing.main import step_6_drop_overlapping_sites

        self.step_1 = step_1_download_psp
        self.step_2 = step_2_download_uniprot
        self.step_3 = step_3_get_kin_to_fam_to_grp
        self.step_4 = step_4_get_pairwise_mtx
        # functools.partial means "run the function that is in the first argument, but always pass the given additional arguments to it"
        self.step_5_a = functools.partial(step_5_get_train_val_test_split, part="a", num_restarts=6)
        self.step_5_b = functools.partial(step_5_get_train_val_test_split, part="b")
        self.step_5_c = functools.partial(step_5_get_train_val_test_split, part="c")
        self.step_6 = step_6_drop_overlapping_sites

        self.backup_kinase_seq_filename = join_first("data/raw_data/kinase_seq_833.csv", 1, __file__)
        self.backup_kin_fam_grp_filename = join_first("data/preprocessing/kin_to_fam_to_grp_828.csv", 1, __file__)
        self.backup_raw_data_filename = join_first("data/raw_data/raw_data_22769.csv", 1, __file__)
        self.backup_new_mtx_filename = join_first("data/preprocessing/pairwise_mtx_833.csv", 1, __file__)
        self.backup_tr_fi_vl_fi_te_fi = tuple(
            map(
                lambda fn: join_first(fn, 1, __file__),
                [
                    "data/raw_data/raw_data_30070_formatted_65.csv",
                    "data/raw_data/raw_data_6896_formatted_95.csv",
                    "data/raw_data/raw_data_8364_formatted_95.csv",
                ],
            )
        )

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
        kin_fam_grp_filename = self.step_3(getattr(self.__class__, "seq_filename_A", self.backup_kinase_seq_filename))
        setattr(self.__class__, "kin_fam_grp_filename", kin_fam_grp_filename)

    def test_step_4_preproc(self):
        new_mtx_filename = self.step_4(
            getattr(self.__class__, "seq_filename_A", self.backup_kinase_seq_filename),
        )
        setattr(self.__class__, "new_mtx_file", new_mtx_filename)

    def step_5_args(self):
        ls = (
            getattr(self.__class__, "kin_fam_grp_filename", self.backup_kin_fam_grp_filename),
            getattr(self.__class__, "raw_data_filename", self.backup_raw_data_filename),
            getattr(self.__class__, "seq_filename_A", self.backup_kinase_seq_filename),
            getattr(self.__class__, "new_mtx_file", self.backup_new_mtx_filename),
        )
        for fn in ls:
            assert os.path.exists(fn), f"{fn} does not exist."
        return ls

    def test_step_5_a_preproc(self):
        self.step_5_a(*self.step_5_args())

    def test_step_5_b_preproc(self):
        self.step_5_b(*self.step_5_args())

    def test_step_5_c_preproc(self):
        setattr(self.__class__, "tr_fi_vl_fi_te_fi", self.step_5_c(*self.step_5_args()))

    def test_step_6_preproc(self):
        self.step_6(*getattr(self.__class__, "tr_fi_vl_fi_te_fi", self.backup_tr_fi_vl_fi_te_fi))


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

    def test_train_nn_classic(self):
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
            "tests/sample_inputs/sample_hp_configs/KSR_params_Classic.json",
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


with open(join_first("examples.json", 0, __file__)) as f:
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

    def test_examples_as_module(self):
        from ..examples import __main__


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
