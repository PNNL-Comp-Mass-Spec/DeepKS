import collections
import os, pathlib, typing, argparse, textwrap, re, itertools
import sys

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)

from .cfg import PRE_TRAINED_NN, PRE_TRAINED_GC

def make_predictions(
    kinase_seqs: list[str],
    site_seqs: list[str],
    predictions_output_format: str = "in_order",
    verbose: bool = True,
    pre_trained_gc: str = PRE_TRAINED_GC,
    pre_trained_nn: str = PRE_TRAINED_NN,
    device: str = "cpu",
    scores: bool = False,
    dry_run: bool = False,
    cartesian_product: bool = False
):
    """Make a target/decoy prediction for a kinase-substrate pair.

    Args:
        kinase_seqs (list[str]): The kinase sequences. Each must be >= 1 and <= 4128 residues long.
        site_seqs ([str]): The site sequences. Each must be 15 residues long.
        predictions_output_format (str, optional): The format of the output. Defaults to "in_order".
            - "in_order" returns a list of predictions in the same order as the input kinases and sites.
            - "dictionary" returns a dictionary of predictions, where the keys are the input kinases and sites and the values are the predictions.
            - "in_order_json" outputs a JSON string (filename = ../out/current-date-and-time.json of a list of predictions in the same order as the input kinases and sites.
            - "dictionary_json" outputs a JSON string (filename = ../out/current-date-and-time.json) of a dictionary of predictions, where the keys are the input kinases and sites and the values are the predictions.
        verbose (bool, optional): Whether to print predictions. Defaults to True.
        pre_trained_gc (str, optional): Path to previously trained group classifier model state. Defaults to PRE_TRAINED_GC.
        pre_trained_nn (str, optional): Path to previously trained neural network model state. Defaults to PRE_TRAINED_NN.
        device (str, optional): Device to use for predictions. Defaults to "cpu".
        scores (bool, optional): Whether to return scores. Defaults to False.
        dry_run (bool, optional): Whether to run a dry run (make sure input parameters work). Defaults to False.
        cartesian_product (bool, optional): Whether to make predictions for all combinations of kinases and sites. Defaults to False.
    """
    # try:
        # Input validation
    if True: # FIXME: Use real exception handling
        assert predictions_output_format in ["in_order", "dictionary", "in_order_json", "dictionary_json"]

        assert len(kinase_seqs) == len(site_seqs) or cartesian_product, (
            f"The number of kinases and sites must be equal. (There are {len(kinase_seqs)} kinases and"
            f" {len(site_seqs)} sites.)"
        )
        for i, kinase_seq in enumerate(kinase_seqs):
            assert 1 <= len(kinase_seq) <= 4128, (
                f"DeepKS currently only accepts kinase sequences of length <= 4128. The input kinase at index {i} --- {kinase_seq[i][:10]} is"
                f" {len(kinase_seq)}. (It was trained on sequences of length <= 4128.)"
            )
            assert kinase_seq.isalpha(), f"Kinase sequences must only contain letters. The input kinase at index {i} --- {kinase_seq[i][:10]}... is problematic."
        for i, site_seq in enumerate(site_seqs):
            assert len(site_seq) == 15, (
                f"DeepKS currently only accepts site sequences of length 15. The input site at index {i} --- {site_seq[i][:10]}... is"
                f" {len(site_seq)}. (It was trained on sequences of length 15.)"
            )
            assert site_seq.isalpha(), f"Site sequences must only contain letters. The input site at index {i} --- {site_seq[i]} is problematic."

        if cartesian_product:
            cart_prod = list(itertools.product(kinase_seqs, site_seqs))
            kinase_seqs = [x[0] for x in cart_prod]
            site_seqs = [x[1] for x in cart_prod]


        if dry_run:
            print("Status: Dry run successful!")
            return

        # Create (load) multi-stage classifier
        print("Status: Loading previously trained models...")
        group_classifier: SKGroupClassifier = pickle.load(open(pre_trained_gc, "rb"))
        individual_classifiers: IndividualClassifiers = pickle.load(open(pre_trained_nn, "rb"))
        msc = MultiStageClassifier(group_classifier, individual_classifiers)

        print("Status: Beginning Prediction Process...")
        try:
            res = msc.predict(kinase_seqs, site_seqs, predictions_output_format=predictions_output_format, device=device, scores=scores)
        except Exception as e:
            print("Status: Predicting Failed!\n\n")
            raise e

        assert res is not None or "json" in predictions_output_format

        if verbose:
            assert res is not None
            print()
            print(first_msg := "<"*16 + " REQUESTED RESULTS " + ">"*16 + "\n")
            if all(isinstance(r, dict) for r in res):
                order = {'kinase': 0, 'site': 1, 'prediction': 2, 'score': 3}
                sortkey = lambda x: order[x[0]]
                pprint.pprint([dict(collections.OrderedDict(sorted(r.items(), key=sortkey))) for r in res], sort_dicts=False) # type: ignore
            else:
                pprint.pprint(res)
            print("\n" + "<"* int(np.floor(len(first_msg)/2)) + ">"*int(np.ceil(len(first_msg)/2))+"\n")
        print("Status: Done!\n")
        return res

    # except Exception as e:
    #     print("Status: Something went wrong!\n\n")
    #     raise e


def parse_api() -> dict[str, typing.Any]:
    """Parse the command line arguments.

    Returns:
        dict[str, Any]: Dictionary mapping the argument name to the argument value.
    """
    wrap = lambda s: textwrap.fill(s, width = 60, subsequent_indent="    ")

    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        def __init__(self, prog):
            super().__init__(prog, max_help_position=60, width=120)
        def _split_lines(self, text, width):
            return super()._split_lines(text, width) + ['']


    ap = argparse.ArgumentParser(prog = 'python -m DeepKS.api.main', formatter_class=CustomFormatter)
    k_group = ap.add_mutually_exclusive_group(required=True)
    k_group.add_argument(
        "-k", help=wrap("Comma-delimited kinase sequences (no spaces). Each must be <= 4128 residues long."),
        metavar="<kinase sequences>"
    )
    k_group.add_argument( # TODO: Add FASTA support
        "-kf", help=wrap("The file containing line-delimited kinase sequences. Each must be <= 4128 residues long."),
        metavar="<kinase sequences file>"
    )

    s_group = ap.add_mutually_exclusive_group(required=True)
    s_group.add_argument(
        "-s", help=wrap("Comma-delimited site sequences (no spaces). Each must be 15 residues long."), metavar="<site sequences>"
    )
    s_group.add_argument(
        "-sf",
        help=wrap("The file containing line-delimited site sequences. Each must be 15 residues long."), metavar="<site sequences file>"
    )

    output_choices_helper = {
        "in_order": (
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
    }

    ap.add_argument("--cartesian-product", default=False, action="store_true", help=wrap("Whether to perform a cartesian product of the input kinases and sites. Defaults to False."))

    ap.add_argument(
        "-p",
        default="in_order",
        choices=output_choices_helper,
        help='\n'.join(wrap("* " + f"{k}: {v}") for k, v in output_choices_helper.items())
    )
    ap.add_argument("-v", "--verbose", help="Whether to print predictions. Defaults to True.", default=False, required=False, action="store_true")

    ap.add_argument("--pre_trained_nn", help=wrap("The path to the pre-trained neural network."), default=PRE_TRAINED_NN, required=False, metavar="<pre-trained neural network file>")
    ap.add_argument("--pre_trained_gc", help=wrap("The path to the pre-trained group classifier."), default=PRE_TRAINED_GC, required=False, metavar="<pre-trained group classifier file>") #FIXME: Max width

    import torch
    def device(arg_value):
        try:
            assert(bool(re.search("^cuda(:|)[0-9]*$", arg_value)) or bool(re.search("^cpu$", arg_value)))
            if "cuda" in arg_value:
                if arg_value == "cuda":
                    return arg_value
                cuda_num = int(re.findall("([0-9]+)", arg_value)[0])
                assert(0 <= cuda_num <= torch.cuda.device_count())
        except Exception:
            raise argparse.ArgumentTypeError(f"Device '{arg_value}' does not exist. Choices are {'cpu', 'cuda[:<gpu #>]'}.")
        
        return arg_value
        
    ap.add_argument("--device", type=device, help="Specify device. Choices are {'cpu', 'cuda:<gpu number>'}.", metavar='<device>', default='cuda:0' if torch.cuda.is_available() else 'cpu')

    ap.add_argument("--scores", help=wrap("Whether to print the scores of the predictions."), default=False, required=False, action="store_true")

    ap.add_argument("--dry-run", help=wrap("Only validates command line parameters; does not do any computations"), default=False, required=False, action="store_true")

    args = ap.parse_args()
    args_dict = vars(args)
    if args_dict['k'] is not None:
        args_dict['kinase_seqs'] = args_dict.pop('k').split(',')
        del args_dict['kf']
    elif 'kf' in args_dict:
        args_dict['kinase_seqs'] = [line.strip() for line in open("../" + args_dict.pop('kf'))]
        del args_dict['k']
    if args_dict['s'] is not None:
        args_dict['site_seqs'] = args_dict.pop('s').split(',')
        del args_dict['sf']
    elif 'sf' in args_dict:
        args_dict['site_seqs'] = [line.strip() for line in open("../" +args_dict.pop('sf'))]
        del args_dict['s']
    if 'v' in args_dict:
        args_dict['verbose'] = args_dict.pop('v')

    args_dict['predictions_output_format'] = args_dict.pop('p')
    if 'json' not in args_dict['predictions_output_format'] and not args_dict['verbose']:
        args_dict['verbose'] = True
        print("Info: Verbose mode is being set to \"True\" because the predictions output format is not JSON.")
    if 'json' in args_dict['predictions_output_format'] and args_dict['verbose']:
        args_dict['verbose'] = False
        print("Info: Verbose mode is being set to \"False\" because the predictions output format is JSON.")

    args_dict['kinase_seqs'] = [x.strip() for x in args_dict['kinase_seqs'] if x != '']
    args_dict['site_seqs'] = [x.strip() for x in args_dict['site_seqs'] if x != '']

    return args_dict

def _cmd_testing_simulator():
    global pickle, pprint, np, IndividualClassifiers, MultiStageClassifier, SKGroupClassifier
    args = parse_api()

    print("Status: Loading Modules...")
    import cloudpickle as pickle, pprint, numpy as np
    from ..models.individual_classifiers import IndividualClassifiers
    from ..models.multi_stage_classifier import MultiStageClassifier
    from ..models.group_classifier_definitions import SKGroupClassifier

    make_predictions(**args)

if __name__ in ["__main__"]:
    args = parse_api()

    print("Status: Loading Modules...")
    import cloudpickle as pickle, pprint, numpy as np
    from ..models.individual_classifiers import IndividualClassifiers
    from ..models.multi_stage_classifier import MultiStageClassifier
    from ..models.group_classifier_definitions import SKGroupClassifier

    make_predictions(**args)
