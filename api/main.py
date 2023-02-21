import sys
if __name__ == "__main__" and (len(sys.argv) >= 2 and sys.argv[1] not in ["--help", "-h", "--usage", "-u"]):
    from ..splash import write_splash

    write_splash.write_splash("main_api")

import os, sys, pathlib, typing, argparse, textwrap, re, json
from termcolor import colored
from .cfg import PRE_TRAINED_NN, PRE_TRAINED_GC


def make_predictions(
    kinase_seqs: list[str],
    kin_info: dict,
    site_seqs: list[str],
    site_info: dict,
    predictions_output_format: str = "inorder",
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
):
    """Make a target/decoy prediction for a kinase-substrate pair.

    Args:
        kinase_seqs (list[str]): The kinase sequences. Each must be >= 1 and <= 4128 residues long.
        kin_info (dict): The kinase (meta-) information.
        site_seqs ([str]): The site sequences. Each must be 15 residues long.
        site_info (dict): The site (meta-) information.
        predictions_output_format (str, optional): The format of the output. Defaults to "inorder"
            - "inorder"returns a list of predictions in the same order as the input kinases and sites.
            - "dictionary" returns a dictionary of predictions, where the keys are the input kinases and sites and the values are the predictions.
            - "in_order_json" outputs a JSON string (filename = ../out/current-date-and-time.json of a list of predictions in the same order as the input kinases and sites.
            - "dictionary_json" outputs a JSON string (filename = ../out/current-date-and-time.json) of a dictionary of predictions, where the keys are the input kinases and sites and the values are the predictions.
            - "csv" outputs a CSV table (filename = ../out/current-date-and-time.csv), where the columns include the input kinases, sites, sequence, metadata and predictions.
            - "sqlite" outputs a sqlite database (filename = ../out/current-date-and-time.sqlite), where the columns include the input kinases, sites, sequence, metadata and predictions.
        suppress_seqs_in_output (bool, optional): Whether to include the input sequences in the output. Defaults to False.
        verbose (bool, optional): Whether to print predictions. Defaults to True.
        pre_trained_gc (str, optional): Path to previously trained group classifier model state. Defaults to PRE_TRAINED_GC.
        pre_trained_nn (str, optional): Path to previously trained neural network model state. Defaults to PRE_TRAINED_NN.
        device (str, optional): Device to use for predictions. Defaults to "cpu".
        scores (bool, optional): Whether to return scores. Defaults to False.
        normalize_scores (bool, optional): Whether to normalize scores. Defaults to False.
        dry_run (bool, optional): Whether to run a dry run (make sure input parameters work). Defaults to False.
        cartesian_product (bool, optional): Whether to make predictions for all combinations of kinases and sites. Defaults to False.
        group_output (bool, optional): Whether to return group predictions. Defaults to False.
    """
    config.cfg.set_mode('no_alin')
    try:
        print(colored("Status: Validating inputs.", "green"))
        assert predictions_output_format in (["inorder" "dictionary", "in_order_json", "dictionary_json", 
                                            "csv", "sqlite"]
        )

        assert len(kinase_seqs) == len(site_seqs) or cartesian_product, (
            f"The number of kinases and sites must be equal. (There are {len(kinase_seqs)} kinases and"
            f" {len(site_seqs)} sites.)"
        )
        for i, kinase_seq in enumerate(kinase_seqs):
            assert 1 <= len(kinase_seq) <= 4128, (
                f"Warning: DeepKS currently only accepts kinase sequences of length <= 4128. The input kinase at index {i} ---"
                f" {kinase_seq[0:10]}... is {len(kinase_seq)}. (It was trained on sequences of length <= 4128.)"
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

        if dry_run:
            print(colored("Status: Dry run successful!", "green"))
            return
        print(colored("Info: Inputs are valid!", "blue"))
        # Create (load) multi-stage classifier
        print(colored("Status: Loading previously trained models...", "green"))
        group_classifier: SKGroupClassifier = pickle.load(open(pre_trained_gc, "rb"))
        individual_classifiers: IndividualClassifiers = IndividualClassifiers.load_all(pre_trained_nn, device=device)
        msc = MultiStageClassifier(group_classifier, individual_classifiers)

        print(colored("Status: Beginning Prediction Process...", "green"))
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
                group_output=group_output,
            )
        except Exception as e:
            informative_exception(e, print_full_tb=True, top_message="Error: Prediction process failed!")

        assert res is not None or "json" in predictions_output_format

        if verbose:
            assert res is not None
            print()
            print(first_msg := "<" * 16 + " REQUESTED RESULTS " + ">" * 16 + "\n")
            if all(isinstance(r, dict) for r in res):
                pprint.pprint([dict(collections.OrderedDict(sorted(r.items()))) for r in res], sort_dicts=False)  # type: ignore
            else:
                pprint.pprint(res)
            print("\n" + "<" * int(np.floor(len(first_msg) / 2)) + ">" * int(np.ceil(len(first_msg) / 2)) + "\n")
        print(colored("Status: Done!\n", "green"))
        return res
    except Exception as e:
        print(informative_exception(e, print_full_tb=False))

def parse_api() -> dict[str, typing.Any]:
    """Parse the command line arguments.

    Returns:
        dict[str, Any]: Dictionary mapping the argument name to the argument value.
    """
    def wrap(s) -> str:
        if "\n" in s:
            wrapped = []
            for line in s.split("\n"):
                num_leading_indents = len(re.findall("^\t+", line))
                new_lines = wrap(line).split("\n")
                for i in range(1, len(new_lines)):
                    new_lines[i] = "\t"*num_leading_indents + new_lines[i]
                wrapped.append("\n".join(new_lines))
            return "\n".join(wrapped)
        else:
            return textwrap.fill(s, width=60, subsequent_indent="    ", expand_tabs=True, tabsize=4, replace_whitespace=False)

    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        def __init__(self, prog):
            super().__init__(prog, max_help_position=60, width=100)

        def _split_lines(self, text, width):
            return super()._split_lines(wrap(text), width) + [""]
        
        def _get_help_string(self, action):
            return str(action.help) + "\n\t" + f"{f'(Default: {action.default}'})"

    ap = argparse.ArgumentParser(prog="python -m DeepKS.api.main", formatter_class=CustomFormatter)
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

    with open(f"{pathlib.Path(__file__).parent.resolve()}/info_file_format.txt") as info_format_f:
        info_format_str = info_format_f.read()
    ap.add_argument("--kin-info", required=False, help=info_format_str, metavar="<kinase info file>")
    ap.add_argument("--site-info", required=False, help="Site information file. Must be able to be read as JSON. Same structure as `info_format_str`.", metavar="<site info file>")

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
            "outputs a CSV table (filename = ../out/current-date-and-time.csv), where the columns include the input kinases, sites, sequence, metadata and predictions."
        ),
        "sqlite": (
            "outputs an sqlite database (filename = ../out/current-date-and-time.sqlite), where the fields (columns) include the input kinases, sites, sequence, metadata and predictions."
        )
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
        required = False,
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
        "--pre_trained_nn",
        help="The path to the pre-trained neural network.",
        default=PRE_TRAINED_NN,
        required=False,
        metavar="<pre-trained neural network file>",
    )
    ap.add_argument(
        "--pre_trained_gc",
        help="The path to the pre-trained group classifier.",
        default=PRE_TRAINED_GC,
        required=False,
        metavar="<pre-trained group classifier file>",
    )

    import torch

    def device(arg_value):
        try:
            assert bool(re.search("^cuda(:|)[0-9]*$", arg_value)) or bool(re.search("^cpu$", arg_value))
            if "cuda" in arg_value:
                assert torch.cuda.is_available()
                if arg_value == "cuda":
                    return arg_value
                cuda_num = int(re.findall("([0-9]+)", arg_value)[0])
                assert 0 <= cuda_num <= torch.cuda.device_count()
        except AssertionError:
            raise argparse.ArgumentTypeError(
                f"Device '{arg_value}' does not exist. Choices are {'cpu', 'cuda[:[0-9]+]'}."
            )

        return arg_value

    ap.add_argument(
        "--device",
        type=device,
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
        "--dry-run",
        help="Only validates command line parameters; does not do any computations",
        default=False,
        required=False,
        action="store_true",
    )

    args = ap.parse_args()  #### ^ Argument collecting v Argument processing ####
    args_dict = vars(args)
    if args_dict["k"] is not None:
        args_dict["kinase_seqs"] = args_dict.pop("k").split(",")
        del args_dict["kf"]
    elif "kf" in args_dict:
        args_dict["kinase_seqs"] = [line.strip() for line in open("../" + args_dict.pop("kf"))]
        del args_dict["k"]
    if args_dict["s"] is not None:
        args_dict["site_seqs"] = args_dict.pop("s").split(",")
        del args_dict["sf"]
    elif "sf" in args_dict:
        args_dict["site_seqs"] = [line.strip() for line in open("../" + args_dict.pop("sf"))]
        del args_dict["s"]

    args_dict["predictions_output_format"] = args_dict.pop("p")
    if "json" not in args_dict["predictions_output_format"] and not args_dict["verbose"]:
        args_dict["verbose"] = True
        print(
            colored(
                'Info: Verbose mode is being set to "True" because the predictions output format is not JSON.', "blue"
            )
        )
    if "json" in args_dict["predictions_output_format"] and args_dict["verbose"]:
        args_dict["verbose"] = False
        print(
            colored('Info: Verbose mode is being set to "False" because the predictions output format is JSON.', "blue")
        )
    if args_dict["suppress_seqs_in_output"] and "json" not in args_dict["predictions_output_format"]:
        print(
            colored(
                "Info: `--suppress-seqs-in-output` is being ignored because the predictions output format is not JSON.",
                "blue",
            )
        )

    args_dict["kinase_seqs"] = [x.strip() for x in args_dict["kinase_seqs"] if x != ""]
    args_dict["site_seqs"] = [x.strip() for x in args_dict["site_seqs"] if x != ""]
    args_dict["group_output"] = args_dict.pop("groups")
    if not args_dict["scores"] and args_dict["normalize_scores"]:
        print(colored("Info: Ignoring `--normalize-scores` since `--scores` was not set.", "blue"))

    import pandas as pd

    if args_dict["kin_info"] is None:
        kinase_info_dict = {
            kinase_seq: {"Gene Name": "<UNK>", "Uniprot Accession ID": "<UNK>"}
            for kinase_seq in args_dict["kinase_seqs"]
        }
    else:
        kinase_info_dict = json.load(open("../" + args_dict["kin_info"]))
        assert isinstance(kinase_info_dict, dict), f"Kinase information format is incorrect. Correct format is {info_format_str}"
        assert all([isinstance(kinase_seq, str) for kinase_seq in kinase_info_dict.keys()]), f"Kinase information format is incorrect. Correct format is {info_format_str}"
        assert all([isinstance(kinase_info_dict[kinase_seq], dict) for kinase_seq in kinase_info_dict.keys()]), f"Kinase information format is incorrect. Correct format is {info_format_str}"
        assert all([isinstance(info_key, str) for kinase_seq in kinase_info_dict.keys() for info_key in kinase_info_dict[kinase_seq].keys()]), f"Kinase information format is incorrect. Correct format is {info_format_str}"
        assert all([isinstance(kinase_info_dict[kinase_seq][info_key], str) for kinase_seq in kinase_info_dict.keys() for info_key in kinase_info_dict[kinase_seq].keys()]), f"Kinase information format is incorrect. Correct format is {info_format_str}"
    args_dict['kin_info'] = kinase_info_dict

    if args_dict["site_info"] is None:
        site_info_dict = {
            site_seq: {"Gene Name": "<UNK>", "Uniprot Accession ID": "<UNK>", "Location": "<UNK>"}
            for site_seq in args_dict["site_seqs"]
        }
    else:
        site_info_dict = json.load(open("../" + args_dict["site_info"]))
        assert isinstance(site_info_dict, dict), f"Site information format is incorrect. Correct format is {info_format_str}"
        assert all([isinstance(site_seq, str) for site_seq in site_info_dict.keys()]), f"Site information format is incorrect. Correct format is {info_format_str}"
        assert all([isinstance(site_info_dict[site_seq], dict) for site_seq in site_info_dict.keys()]), f"Site information format is incorrect. Correct format is {info_format_str}"
        assert all([isinstance(info_key, str) for site_seq in site_info_dict.keys() for info_key in site_info_dict[site_seq].keys()]), f"Site information format is incorrect. Correct format is {info_format_str}"
        assert all([isinstance(site_info_dict[site_seq][info_key], str) for site_seq in site_info_dict.keys() for info_key in site_info_dict[site_seq].keys()]), f"Site information format is incorrect. Correct format is {info_format_str}"
    args_dict['site_info'] = site_info_dict

    assert all({"Uniprot Accession ID", "Gene Name"}.issubset(set(kinase_info_dict[kinase_seq].keys())) for kinase_seq in args_dict["kinase_seqs"]), (
        "Kinase information file must have the columns 'Gene Name' and 'Uniprot Accession ID'"
    )

    assert all({"Uniprot Accession ID", "Gene Name", "Location"}.issubset(set(site_info_dict[site_seq].keys())) for site_seq in args_dict["site_seqs"]), (
        "Site information file must have the columns 'Gene Name', 'Uniprot Accession ID', and 'Location'"
    )

    assert(all(kinase_seq in kinase_info_dict for kinase_seq in args_dict["kinase_seqs"]))
    assert(all(site_seq in site_info_dict for site_seq in args_dict["site_seqs"]))

    # args_dict["kinase_gene_names"] = list(kinase_info_dict.keys())
    # args_dict["kinase_uniprot_accessions"] = [kinase_info_dict[kinase_seq]["Uniprot Accession ID"] for kinase_seq in args_dict["kinase_seqs"]]
    # args_dict["site_gene_names"] = list(site_info_dict.keys())
    # args_dict["site_uniprot_accessions"] = [site_info_dict[site_seq]["Uniprot Accession ID"] for site_seq in args_dict["site_seqs"]]
    # args_dict["site_locations"] = [site_info_dict[site_seq]["Location"] for site_seq in args_dict["site_seqs"]]

    # del args_dict["kin_info"]
    # del args_dict["site_info"]

    return args_dict


def setup():
    global pickle, pprint, np, IndividualClassifiers, MultiStageClassifier, SKGroupClassifier, informative_exception, tqdm, itertools, collections, json, config
    os.chdir(pathlib.Path(__file__).parent.resolve())
    args = parse_api()

    print(colored("Status: Loading Modules...", "green"))
    import cloudpickle as pickle, pprint, numpy as np, tqdm, itertools, collections, json
    from ..models.individual_classifiers import IndividualClassifiers
    from ..models.multi_stage_classifier import MultiStageClassifier
    from ..models.group_classifier_definitions import SKGroupClassifier
    from .. import config
    from ..tools.informative_tb import informative_exception

    make_predictions(**args)


if __name__ in ["__main__"]:
    setup()
