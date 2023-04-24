"""Contains functions that are an interface to the DeepKS project.
"""
import sys
from nbformat import convert
from termcolor import colored
from typing import Literal

if (len(sys.argv) >= 2 and sys.argv[1] not in ["--help", "-h", "--usage", "-u"]) or len(sys.argv) < 2:
    from ..splash import write_splash

    if __name__ == "__main__":
        write_splash.write_splash("main_api")

from ..config.root_logger import get_logger

logger = get_logger()
logger.status("Loading Modules...")

import os, pathlib, typing, argparse, textwrap, re, json, warnings, jsonschema, jsonschema.exceptions, socket, io, torch


from .cfg import PRE_TRAINED_NN, PRE_TRAINED_GC

join_first = lambda levels, x: (
    x if os.path.isabs(x) else os.path.join(pathlib.Path(__file__).parent.resolve(), *[".."] * levels, x)
)

with open(str(pathlib.Path(__file__).parent.resolve()) + "/../config/API_IMPORT_MODE.json", "r") as apiim:
    API_IMPORT_MODE = "true" in apiim.read()

from ..tools import schema_validation
from ..tools.informative_tb import informative_exception


class Capturing(list):
    """https://stackoverflow.com/a/16571630/16158339"""

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self._stringio = io.StringIO()
        sys.stderr = self._stringio2 = io.StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        self.extend(self._stringio2.getvalue().splitlines())
        del self._stringio  # free up some memory
        del self._stringio2  # free up some memory
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def make_predictions(
    kinase_seqs: list[str],
    kin_info: dict,
    site_seqs: list[str],
    site_info: dict,
    predictions_output_format: typing.Literal[
        "inorder", "dictionary", "inorder_json", "dictionary_json", "csv", "sqlite"
    ] = "csv",
    suppress_seqs_in_output: bool = False,
    verbose: bool = True,
    pre_trained_gc: str = PRE_TRAINED_GC,
    pre_trained_nn: str = PRE_TRAINED_NN,
    device: str = "cpu",
    scores: bool = False,
    normalize_scores: bool = False,
    dry_run: bool = False,
    cartesian_product: bool = False,
    group_output: bool = False,
    bypass_group_classifier: list[str] = [],
    convert_raw_to_prob: bool = True,
    group_on: Literal["site", "kin"] = "site",
):
    """Make a target/decoy prediction for a kinase-substrate pair.

    @arg kinase_seqs: The kinase sequences. Each must be >= 1 and <= 4128 residues long.
    @arg kin_info: The kinase (meta-) information. See C{./kin-info_file_format.txt} for the required format.
    @arg site_seqs: The site sequences. Each must be 15 residues long.
    @arg site_info: The site (meta-) information. See C{./site-info_file_format.txt} for the required format.
    @arg predictions_output_format: The format of the output.
        - C{inorder} returns a list of predictions in the same order as the input kinases and sites.
        - C{dictionary} returns a dictionary of predictions, where the keys are the input kinases and sites
            and the values are the predictions.
        - C{in_order_json} outputs a JSON string (filename = C{"DeepKS/out/current-date-and-time.json"}) of a list of
            predictions in the same order as the input kinases and sites.
        - C{dictionary_json} outputs a JSON string (filename = C{"DeepKS/out/current-date-and-time.json"}) of a
            dictionary of predictions, where the keys are the input kinases and sites and the values are the
            predictions.
        - C{csv} outputs a CSV table (filename = C{"DeepKS/out/current-date-and-time.csv"}), where the columns include
            the input kinases, sites, sequence, metadata and predictions.
        - C{sqlite} outputs a sqlite database (filename = C{"DeepKS/out/current-date-and-time.sqlite"}), where the
            columns include the input kinases, sites, sequence, metadata and predictions.
    @arg suppress_seqs_in_output: Whether to include the input sequences in the output.
    @arg verbose: Whether to print predictions.
    @arg pre_trained_gc: Path to previously trained group classifier model state.
    @arg pre_trained_nn: Path to previously trained neural network model state.
    @arg device: Device to use for predictions.
    @arg scores: Whether to return scores.
    @arg normalize_scores: Whether to normalize scores.
    @arg dry_run: Whether to run a dry run (make sure input parameters work).
    @arg cartesian_product: Whether to make predictions for all combinations of kinases and sites.
    @arg group_output: Whether to return group predictions.
    @arg bypass_group_classifier: List of known kinase groups in the same order in which they appear in kinase_seqs.
                                  See C{./kin-info_file_format.txt} for instructions on how to specify groups.
    @arg convert_raw_to_prob: Whether to convert raw scores to empirical probabilities. The neural network save file's object must have a C{emp_eqn} attribute (i.e., a mapping from raw score to empirical probability).
    """
    config.cfg.set_mode("no_alin")
    try:
        logger.status("Validating inputs.")
        assert predictions_output_format in (
            ["inorder", "dictionary", "in_order_json", "dictionary_json", "csv", "sqlite"]
        ), f"Output format was {predictions_output_format}, which is not allowed."

        assert len(kinase_seqs) == len(site_seqs) or cartesian_product, (
            f"The number of kinases and sites must be equal. (There are {len(kinase_seqs)} kinases and"
            f" {len(site_seqs)} sites.)"
        )
        for i, kinase_seq in enumerate(kinase_seqs):
            assert 1 <= len(kinase_seq) <= 4128, (
                "Warning: DeepKS currently only accepts kinase sequences of length <= 4128. The input kinase at index"
                f" {i} --- {kinase_seq[0:10]}... is {len(kinase_seq)}. (It was trained on sequences of length <= 4128.)"
            )
            f"Kinase sequences must only contain letters. The input kinase at index {i} --- {kinase_seq[:10]}..."
            " is problematic."
        for i, site_seq in enumerate(site_seqs):
            assert len(site_seq) == 15, (
                f"DeepKS currently only accepts site sequences of length 15. The input site at index {i} ---"
                f" {site_seq[i][:10]}... is {len(site_seq)}. (It was trained on sequences of length 15.)"
            )
            assert site_seq.isalpha(), (
                f"Site sequences must only contain letters. The input site at index {i} --- {site_seq[i]} is"
                " problematic."
            )

        logger.info("Inputs are valid!")

        # Create (load) multi-stage classifier
        logger.status("Loading previously trained models...")
        with open(pre_trained_gc, "rb") as f:
            group_classifier: SKGroupClassifier = pickle.load(f)
        individual_classifiers: IndividualClassifiers = IndividualClassifiers.load_all(
            pre_trained_nn, target_device=device
        )
        msc = MultiStageClassifier(group_classifier, individual_classifiers)
        # nn_sample = list(individual_classifiers.interfaces.values())[0]
        # summary_stringio = io.StringIO()
        # FAKE_BATCH_SIZE = 101
        # nn_sample.device = torch.device(device); nn_sample.inp_types = [torch.int32, torch.int32]; nn_sample.inp_size = [(FAKE_BATCH_SIZE, 15), (FAKE_BATCH_SIZE, 4128)]; nn_sample.model_summary_name = summary_stringio
        # nn_sample.write_model_summary()
        # summary_stringio.seek(0)
        # summary_string = re.sub(str(FAKE_BATCH_SIZE), "BaS", summary_stringio.read())
        # summary_string = re.sub(r"=+\nInput size \(MB\):.*\nForward\/backward pass size \(MB\):.*\nParams size \(MB\):.*\nEstimated Total Size \(MB\):.*\n=+", "", summary_string)
        # with open(os.path.abspath("") + "/../architectures/nn_summary.txt", "w") as f:
        #     f.write(summary_string)

        if dry_run:
            logger.status("Dry run successful!")
            return

        logger.status("Beginning Prediction Process...")
        try:
            res = msc.predict(  #### This is the meat of the prediction process!
                kinase_seqs,
                kin_info,
                site_seqs,
                site_info,
                predictions_output_format=predictions_output_format,
                suppress_seqs_in_output=suppress_seqs_in_output,
                device=device,
                scores=scores,
                normalize_scores=normalize_scores,
                cartesian_product=cartesian_product,
                group_on=group_on,
                group_output=group_output,
                bypass_group_classifier=bypass_group_classifier,
                convert_raw_to_prob=convert_raw_to_prob,
            )
        except Exception as e:
            informative_exception(e, print_full_tb=True, top_message="Error: Prediction process failed!")

        assert res is not None or re.search(
            "(json|csv|sqlite)", predictions_output_format
        ), f"Issue with results. ({res=}; {predictions_output_format=})"

        if verbose:
            assert res is not None
            msg = "\n<" * 16 + " REQUESTED RESULTS " + ">" * 16 + "\n"
            if all(isinstance(r, dict) for r in res):
                msg += pprint.pformat([dict(collections.OrderedDict(sorted(r.items()))) for r in res], sort_dicts=False)
            else:
                msg += pprint.pformat(res, sort_dicts=False)
            msg += "\n" + "<" * int(np.floor(len(msg) / 2)) + ">" * int(np.ceil(len(msg) / 2)) + "\n"
            logger.info(msg)
        logger.status("Done!\n")
        return res
    except Exception as e:
        print(informative_exception(e, print_full_tb=True))


def parse_api() -> dict[str, typing.Any]:
    """Parse the command line arguments.

    @returns: Dictionary mapping the argument name to the argument value.
    """

    logger.status("Parsing Arguments")

    def wrap(s) -> str:
        if "\n" in s:
            wrapped = []
            for line in s.split("\n"):
                num_leading_indents = len(re.findall("^\t+", line))
                new_lines = wrap(line).split("\n")
                for i in range(1, len(new_lines)):
                    new_lines[i] = "\t" * num_leading_indents + new_lines[i]
                wrapped.append("\n".join(new_lines))
            return "\n".join(wrapped)
        else:
            return textwrap.fill(
                s, width=60, subsequent_indent="        ", expand_tabs=True, tabsize=4, replace_whitespace=False
            )

    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        def __init__(self, prog):
            super().__init__(prog, max_help_position=60, width=100)

        def _split_lines(self, text, width):
            return super()._split_lines(wrap(text), width) + [""]

        def _get_help_string(self, action):
            return str(action.help) + "\n\t" + f"{f'(Default: {action.default}'})"

    ap = argparse.ArgumentParser(prog="python -m DeepKS.api.main", formatter_class=CustomFormatter, exit_on_error=False)

    def new_exit(status=0, message=None):
        # print(f"@@@ {sys.argv=}")
        if sys.argv[1] in ["-h", "--help"] and len(sys.argv) == 2:
            sys.exit(0)
        raise ValueError()

    error_orig = ap.error
    ap.exit = new_exit

    def new_err(message):
        output = ""
        try:
            with Capturing() as output:
                error_orig(message)
        except Exception:
            raise argparse.ArgumentError(None, "\t\n".join(output))

    ap.error = new_err

    k_group = ap.add_mutually_exclusive_group(required=True)
    k_group.add_argument(
        "-k",
        help="Comma-delimited kinase sequences (no spaces). Each must be <= 4128 residues long.",
        metavar="<kinase sequences>",
    )
    k_group.add_argument(
        "-kf",
        help="The file containing line-delimited kinase sequences. Each must be <= 4128 residues long.",
        metavar="<kinase sequences file>",
    )

    s_group = ap.add_mutually_exclusive_group(required=True)
    s_group.add_argument(
        "-s",
        help="Comma-delimited site sequences (no spaces). Each must be 15 residues long.",
        metavar="<site sequences>",
    )
    s_group.add_argument(
        "-sf",
        help="The file containing line-delimited site sequences. Each must be 15 residues long.",
        metavar="<site sequences file>",
    )

    ap.add_argument(
        "--kin-info",
        required=False,
        help=f"Kinase information file. See DeepKS/api/kin-info_file_format.txt for details about the correct format.",
        metavar="<kinase info file>",
    )
    ap.add_argument(
        "--site-info",
        required=False,
        help=f"Site information file. See DeepKS/api/site-info_file_format.txt for details about the correct format.",
        metavar="<site info file>",
    )

    output_choices_helper = {
        "inorder": (
            "prints (if verbose == True) a list of predictions in the same order as the input kinases and sites."
        ),
        "dictionary": (
            "prints (if verbose == True) a dictionary of predictions, where the keys are the input kinases and sites"
            " and the values are the predictions."
        ),
        "in_order_json": (
            "outputs a JSON string (filename = ../out/current-date-and-time.json) of a list of predictions in the same"
            " order as the input kinases and sites."
        ),
        "dictionary_json": (
            "outputs a JSON string (filename = ../out/current-date-and-time.json) of a dictionary of predictions, where"
            " the keys are the input kinases and sites and the values are the predictions."
        ),
        "csv": (
            "outputs a CSV table (filename = ../out/current-date-and-time.csv), where the columns include the input"
            " kinases, sites, sequence, metadata and predictions."
        ),
        "sqlite": (
            "outputs an sqlite database (filename = ../out/current-date-and-time.sqlite), where the fields (columns)"
            " include the input kinases, sites, sequence, metadata and predictions."
        ),
    }

    ap.add_argument(
        "--cartesian-product",
        default=False,
        action="store_true",
        help="Whether to perform a cartesian product of the input kinases and sites.",
    )

    ap.add_argument(
        "-p",
        default="inorder",
        choices=output_choices_helper,
        help="\n".join(wrap("* " + f"{k}: {v}") for k, v in output_choices_helper.items()),
    )
    ap.add_argument(
        "--suppress-seqs-in-output",
        default=False,
        required=False,
        action="store_true",
        help="Whether to include the input sequences in the output.",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        help="Whether to print predictions.",
        default=False,
        required=False,
        action="store_true",
    )

    ap.add_argument(
        "--pre-trained-nn",
        help="The path to the pre-trained neural network.",
        default=PRE_TRAINED_NN,
        required=False,
        metavar="<pre-trained neural network file>",
    )
    ap.add_argument(
        "--pre-trained-gc",
        help="The path to the pre-trained group classifier.",
        default=PRE_TRAINED_GC,
        required=False,
        metavar="<pre-trained group classifier file>",
    )

    ap.add_argument(
        "--group-on",
        help="Whether to group on kinases or sites.",
        choices=["kinase", "site"],
        default=None,
        required=False,
        metavar="<group on>",
    )

    def device_eligibility(arg_value):
        try:
            assert bool(re.search("^cuda(:|)[0-9]*$", arg_value)) or bool(re.search("^cpu$", arg_value))
            if "cuda" in arg_value:
                if arg_value == "cuda":
                    return arg_value
                cuda_num = int(re.findall("([0-9]+)", arg_value)[0])
                assert 0 <= cuda_num <= torch.cuda.device_count()
        except Exception:
            raise argparse.ArgumentTypeError(
                f"Device '{arg_value}' does not exist on this machine (hostname: {socket.gethostname()}).\n"
                f"Choices are {sorted(set(['cpu']).union([f'cuda:{i}' for i in range(torch.cuda.device_count())]))}."
            )

        return arg_value

    ap.add_argument(
        "--device",
        type=str,
        help="Specify device. Choices are {'cpu', 'cuda:<gpu number>'}.",
        metavar="<device>",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    ap.add_argument(
        "--scores",
        help="Whether to obtain the scores of the predictions.",
        default=False,
        required=False,
        action="store_true",
    )
    ap.add_argument(
        "--normalize-scores",
        help="Whether to normalize the scores in the predictions between 0 and 1",
        default=False,
        required=False,
        action="store_true",
    )
    ap.add_argument(
        "--groups",
        help="Whether to obtain the groups of the predictions.",
        default=False,
        required=False,
        action="store_true",
    )
    ap.add_argument(
        "--bypass-group-classifier",
        help=(
            "Whether or not to bypass the group classifier (due to having known groups). See"
            " C{./kin-info_file_format.txt} for instructions on how to specify groups."
        ),
        default=False,
        required=False,
        action="store_true",
    )

    ap.add_argument(
        "--dry-run",
        help="Only validates command line parameters; does not do any computations",
        default=False,
        required=False,
        action="store_true",
    )

    ap.add_argument(
        "--convert-raw-to-prob",
        help="Attempt to convert raw scores into empirical probabilities.",
        default=False,
        required=False,
        action="store_true",
    )

    try:
        args = ap.parse_args()  #### ^ Argument collecting v Argument processing ####
    except Exception as e:
        informative_exception(e, "Issue with arguments provided.", print_full_tb=False)
    args_dict = vars(args)
    device_eligibility(args_dict["device"])
    ii = ll = None
    if args_dict["k"] is not None:
        args_dict["kinase_seqs"] = args_dict.pop("k").split(",")
        del args_dict["kf"]
    elif "kf" in args_dict:
        with open(join_first(1, args_dict["kf"])) as f:
            args_dict["kinase_seqs"] = [line.strip() for line in f]
        try:
            fn_relevant = args_dict["kf"].split("/")[-1].split(".")[0]
            assert not re.search(r"[0-9^]", fn_relevant) or (ii := int(re.sub(r"[^0-9]", "", fn_relevant))) == (
                ll := len(args_dict["kinase_seqs"])
            )
        except AssertionError:
            warnings.warn(
                f"The number of kinases in the input file name ({ll}) does not match the number of kinases in the input"
                f" list ({ii}). This may cause unintended behavior."
            )
        del args_dict["k"]
        del args_dict["kf"]
    else:
        raise AssertionError("Should never be in this case.")
    if args_dict["s"] is not None:
        args_dict["site_seqs"] = args_dict.pop("s").split(",")
        del args_dict["sf"]
    elif "sf" in args_dict:
        with open(join_first(1, args_dict["sf"])) as f:
            args_dict["site_seqs"] = [line.strip() for line in f]
        try:
            fn_relevant = args_dict["sf"].split("/")[-1].split(".")[0]
            assert not re.search(r"[0-9^]", fn_relevant) or (ii := int(re.sub(r"[^0-9]", "", fn_relevant))) == (
                ll := len(args_dict["site_seqs"])
            )
        except AssertionError:
            warnings.warn(
                f"The number of sites in the input file name ({ll}) does not match the number of sites in the input"
                f" list ({ii}). This may cause unintended behavior."
            )
        del args_dict["s"]
        del args_dict["sf"]
    else:
        raise AssertionError("Should never be in this case.")

    args_dict["predictions_output_format"] = args_dict.pop("p")
    # if "json" not in args_dict["predictions_output_format"] and not args_dict["verbose"]:
    #         args_dict["verbose"] = True
    #         print(
    #                 colored(
    #                         'Info: Verbose mode is being set to "True" because the predictions output format is not JSON.', "blue"
    #                 )
    #         )
    if "json" in args_dict["predictions_output_format"] and args_dict["verbose"]:
        args_dict["verbose"] = False
        print(
            colored('Info: Verbose mode is being set to "False" because the predictions output format is JSON.', "blue")
        )
    if not (
        re.search(r"(json|csv|sql)", args_dict["predictions_output_format"]) and args_dict["suppress_seqs_in_output"]
    ):
        print(
            colored(
                (
                    "Info: `--suppress-seqs-in-output` is being ignored because the predictions output format is not"
                    " json/csv/sqlite."
                ),
                "blue",
            )
        )

    args_dict["kinase_seqs"] = [x.strip() for x in args_dict["kinase_seqs"] if x != ""]
    args_dict["site_seqs"] = [x.strip() for x in args_dict["site_seqs"] if x != ""]
    args_dict["group_output"] = args_dict.pop("groups")
    if not args_dict["scores"] and args_dict["normalize_scores"]:
        logger.info("Ignoring `--normalize-scores` since `--scores` was not set.")

    if args_dict["kin_info"] is None:
        kinase_info_dict = {
            kinase_seq: {"Gene Name": "?", "Uniprot Accession ID": "?"} for kinase_seq in args_dict["kinase_seqs"]
        }
    else:
        with open(join_first(1, args_dict["kin_info"])) as f:
            kinase_info_dict = json.load(f)
            try:
                jsonschema.validate(
                    kinase_info_dict,
                    (schema_validation.KinSchema),
                )
            except jsonschema.exceptions.ValidationError as e:
                print("", file=sys.stderr)
                print(colored(f"Error: Kinase information format is incorrect.", "red"), file=sys.stderr)
                print(colored("\nFor reference, the jsonschema.exceptions.ValidationError was:", "magenta"))
                print(colored(str(e), "magenta"))
                print(colored("\n\nMore info:\n\n", "magenta"))
                with open(join_first(0, "kin-info_file_format.txt")) as f:
                    print(colored(f.read(), "magenta"), file=sys.stderr)

                informative_exception(e, "", False)

    args_dict["kin_info"] = kinase_info_dict

    if args_dict["site_info"] is None:
        site_info_dict = {
            site_seq: {"Gene Name": "?", "Uniprot Accession ID": "?", "Location": "?"}
            for site_seq in args_dict["site_seqs"]
        }
    else:
        with open(join_first(1, args_dict["site_info"])) as f:
            site_info_dict = json.load(f)
            try:
                jsonschema.validate(
                    site_info_dict,
                    schema_validation.SiteSchema
                    if not args_dict["bypass_group_classifier"]
                    else schema_validation.SiteSchemaBypassGC,
                )
            except jsonschema.exceptions.ValidationError:
                emsg = f"\nError: Site information format is incorrect."
                with open(join_first(0, "./site-info_file_format.txt")) as f:
                    emsg += f.read()
                logger.uerror(emsg)
                sys.exit(1)
    args_dict["site_info"] = site_info_dict

    try:
        for kinase_seq in args_dict["kinase_seqs"]:
            assert kinase_seq in kinase_info_dict, f"Kinase sequence {kinase_seq} not found in kinase info file."
        for site_seq in args_dict["site_seqs"]:
            assert site_seq in site_info_dict, f"Site sequence {site_seq} not found in site info file."
    except AssertionError as e:
        if args_dict["bypass_group_classifier"] and re.search(
            r"Kinase sequence.*not found in kinase info file\.", str(e)
        ):
            raise AssertionError(f"{e} (Since `--bypass-group-classifier` was set, this is a fatal error.)")
        else:
            warnings.warn(colored(f"{e} (The output will not contain info for this sequence.)", "yellow"))

    args_dict["bypass_group_classifier"] = (
        [site_info_dict[ss]["Known Group"] for ss in args_dict["site_seqs"]]
        if args_dict["bypass_group_classifier"]
        else []
    )

    if args_dict["group_on"] and not args_dict["cartesian_product"]:
        logger.info("Ignoring `--group-on` since `--cartesian-product` was not set.")

    if not args_dict["group_on"] and args_dict["cartesian_product"]:
        logger.info("Setting `--group-on` to 'site' as default since `--cartesian-product` was set.")
        args_dict["group_on"] = "site"

    return args_dict


def setup(args: dict[str, typing.Any] = {}):
    """Optionally parses command line arguments for DeepKS, imports necessary modules, and calls make_predictions with `args` (either passed in or from the commandline).

    Args:
        @arg args: DeepKS arguments.
    """
    global pickle, pprint, np, IndividualClassifiers, MultiStageClassifier, SKGroupClassifier, informative_exception, tqdm, itertools, collections, json, config, jsonschema
    from ..tools.informative_tb import informative_exception

    os.chdir(pathlib.Path(__file__).parent.resolve())
    if not isinstance(args, dict) or (hasattr(args, "__len__") and len(args) == 0):
        try:
            args = parse_api()
        except Exception as e:
            informative_exception(e, "Error: DeepKS API failed to parse arguments.")

    import cloudpickle as pickle, pprint, numpy as np, tqdm, itertools, collections, json, jsonschema
    from ..models.individual_classifiers import IndividualClassifiers
    from ..models.multi_stage_classifier import MultiStageClassifier
    from ..models.group_classifier_definitions import SKGroupClassifier
    from .. import config

    make_predictions(**args)


if __name__ == "__main__":
    args = parse_api()
    import cloudpickle as pickle, pprint, numpy as np, tqdm, itertools, collections, json, jsonschema
    from ..models.individual_classifiers import IndividualClassifiers
    from ..models.multi_stage_classifier import MultiStageClassifier
    from ..models.group_classifier_definitions import SKGroupClassifier
    from .. import config

    make_predictions(**args)

if API_IMPORT_MODE:
    import cloudpickle as pickle, pprint, numpy as np, tqdm, itertools, collections, json, jsonschema, torch
    from ..models.individual_classifiers import IndividualClassifiers
    from ..models.multi_stage_classifier import MultiStageClassifier
    from ..models.group_classifier_definitions import SKGroupClassifier
    from .. import config
