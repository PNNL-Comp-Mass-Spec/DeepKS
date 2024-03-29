"""Defines the `MultiStageClassifier` class, which allows group pre-classification before neural network classification."""
if __name__ == "__main__":  # pragma: no cover
    from ..tools.splash.write_splash import write_splash
    from termcolor import colored

    write_splash("main_gc_trainer")

from xmlrpc.client import boolean
from ..config.logging import get_logger

logger = get_logger()

if __name__ == "__main__":  # pragma: no cover
    logger.status("Loading Modules")


import pandas as pd, numpy as np, tempfile as tf, json, cloudpickle as pickle, pathlib, os, tqdm, re, sqlite3, warnings
import torch, argparse, socket
from typing import Any, Callable, Literal, Union
from ..tools.get_needle_pairwise import get_needle_pairwise_mtx
from .individual_classifiers import IndividualClassifiers
from . import individual_classifiers
from ..tools.file_names import get as get_file_name
from ..tools import make_fasta as dist_mtx_maker
from sklearn.neural_network import MLPClassifier
from termcolor import colored
from .GroupClassifier import GroupClassifier, SiteGroupClassifier

from .individual_classifiers import IndividualClassifiers

# pd.set_option("display.max_columns", 100)
# pd.set_option("display.max_rows", 100000)
# pd.set_option("display.width", 240)

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)

from ..config.join_first import join_first


class MultiStageClassifier:
    """A class that allows group pre-classification before neural network classification."""

    def __init__(
        self,
        group_classifier: GroupClassifier,
        individual_classifiers: individual_classifiers.IndividualClassifiers,
    ):
        """Initialize a MultiStageClassifier object.

        Parameters
        ----------
        group_classifier : GroupClassifier
            The group classifier to use
        individual_classifiers : IndividualClassifiers
            The individual (neural network) classifiers to use
        """
        self.group_classifier = group_classifier
        self.individual_classifiers = individual_classifiers

    def __str__(self):
        """Return a string representation of this object.

        Returns
        -------
            Simple string representation of this object: prints both the group classifier and the individual classifier string representations.
        """
        return (
            f"MultiStageClassifier(group_classifier={self.group_classifier},"
            f" individual_classifiers={self.individual_classifiers})"
        )

    def group_classify_then_evaluate(
        self,
        addl_args: dict[str, Any],
        Xy_formatted_input_file: str,
        predict_mode: bool = False,
        bypass_group_classifier: dict | list = {},
        get_emp_eqn: bool = True,
        cartesian_product: bool = False,
        device: str = "cpu",
        group_on: Literal["kin", "site"] = "site",
    ):
        """

        Parameters
        ----------
        addl_args : dict[str, Any]
            _description_
        Xy_formatted_input_file : str
            _description_
        predict_mode : bool, optional
            _description_, by default False
        bypass_group_classifier : dict, optional
            _description_, by default {}
        get_emp_eqn : bool, optional
            _description_, by default True
        cartesian_product : bool, optional
            _description_, by default False
        device : str, optional
            _description_, by default "cpu"
        group_on : Literal[&quot;kin&quot;, &quot;site&quot;], optional
            _description_, by default "site"

        Returns
        -------
        _type_
            _description_
        """
        logger.status("Prediction Step [1/2]: Sending input kinases to group classifier")
        # Open XY_formatted_input_file
        Xy_formatted_input_file = join_first(Xy_formatted_input_file, 1, __file__)
        with open(Xy_formatted_input_file) as f:
            XY_chars = f.read()
        try:
            input_file = pd.read_csv(Xy_formatted_input_file)
        except pd.errors.ParserError:
            input_file = {k: pd.Series([x for x in v]) for k, v in json.loads(XY_chars).items()}
        if not predict_mode:
            _, _, true_symbol_to_grp_dict = individual_classifiers.IndividualClassifiers.get_symbol_to_grp_dict(
                "../data/preprocessing/kin_to_fam_to_grp_826.csv"
            )
        else:
            true_symbol_to_grp_dict = None

        ### Get group predictions
        if group_on == "kin":
            opposite_grp = "Kinase Sequence"
        else:
            opposite_grp = "Site Sequence"
        unq_symbols = input_file[opposite_grp].drop_duplicates()
        unq_symbols_inds = unq_symbols.index.tolist()
        unq_symbols = unq_symbols.tolist()

        if bypass_group_classifier:
            true_symbol_to_grp_dict = dict(zip(unq_symbols, [input_file["known_groups"][i] for i in unq_symbols_inds]))

        if not predict_mode or bypass_group_classifier:
            assert true_symbol_to_grp_dict is not None
            # groups_true = [true_symbol_to_grp_dict[u] for u in unq_symbols]
        else:
            # groups_true = [None for _ in unq_symbols]
            pass

        ### Perform Real Accuracy
        # embedded_unq_symbols = [SiteClassifier.one_hot_aa_embedding(s) for s in unq_symbols]
        groups_prediction = self.group_classifier.__class__.predict(
            self.group_classifier, [str(x) for x in unq_symbols]
        )
        pred_grp_dict = {symbol: grp for symbol, grp in zip(unq_symbols, groups_prediction)}
        self.pred_grp_dict = pred_grp_dict
        # true_grp_dict = {symbol: grp for symbol, grp in zip(unq_symbols, groups_true)}

        ### Perform Simulated 100% Accuracy
        # sim_ac = 1
        # wrong_inds = set()
        # if sim_ac != 1:
        #     wrong_inds = set([round(i / (1 - sim_ac), 0) for i in range(int(len(groups_true) // (1 / sim_ac)))])
        # print("Simulated accuracy:", sim_ac)
        # random.seed(0)
        # groups_prediction = [
        #     groups_true[i]
        #     if (sim_ac == 1 or i not in wrong_inds)
        #     else random.choice(list(set(groups_true) - set([groups_true[i]])))
        #     for i in range(len(groups_true))
        # ]

        # Report performance
        res = {}
        self.individual_classifiers.evaluations = {}
        if bypass_group_classifier:
            logger.info("Info: Group Classifier Accuracy (since we have known groups)")
        if predict_mode:
            if "test_json" not in addl_args:
                key = "test"
            else:
                key = "test_json"
            addl_args[key] = Xy_formatted_input_file
            print(
                colored(
                    (
                        "Status: Prediction Step [2/2]: Sending input kinases to individual group classifiers, based on"
                        " step [1/2]"
                    ),
                    "green",
                )
            )

            res = self.individual_classifiers.evaluation(
                addl_args,
                self.group_classifier,
                True,
                torch.device(device),
                get_emp_eqn=get_emp_eqn,
                cartesian_product=cartesian_product,
            )

        return res

    @staticmethod
    def _package_results(
        predictions_output_format,
        kin_info,
        site_info,
        kinase_seqs,
        site_seqs,
        cartesian_product,
        boolean_predictions,
        numerical_scores,
        group_predictions,
        scores,
        group_output,
        suppress_seqs_in_output,
    ):
        if "dict" in predictions_output_format or re.search(r"(sqlite|csv)", predictions_output_format):
            logger.status("Copying Results to Dictionary.")

            base_kinase_gene_names = []
            base_kinase_uniprot_accession = []
            for k in kinase_seqs:
                if k in kin_info:
                    base_kinase_gene_names.append(kin_info[k]["Gene Name"])
                    base_kinase_uniprot_accession.append(kin_info[k]["Uniprot Accession ID"])
                else:
                    base_kinase_gene_names.append("?")
                    base_kinase_uniprot_accession.append("?")

            base_site_gene_names = []
            base_site_location = []
            base_site_uniprot_accession = []
            for s in site_seqs:
                if s in site_info:
                    base_site_gene_names.append(site_info[s]["Gene Name"])
                    base_site_location.append(site_info[s]["Location"])
                    base_site_uniprot_accession.append(site_info[s]["Uniprot Accession ID"])
                else:
                    base_site_gene_names.append("?")
                    base_site_location.append("?")
                    base_site_uniprot_accession.append("?")

            kinase_gene_names = (
                base_kinase_gene_names
                if not cartesian_product
                else [x for x in base_kinase_gene_names for _ in range(len(site_seqs))]
            )
            site_gene_names = (
                base_site_gene_names
                if not cartesian_product
                else [x for _ in range(len(kinase_seqs)) for x in base_site_gene_names]
            )
            kinase_uniprot_accession = (
                base_kinase_uniprot_accession
                if not cartesian_product
                else [x for x in base_kinase_uniprot_accession for _ in range(len(site_seqs))]
            )
            site_uniprot_accession = (
                base_site_uniprot_accession
                if not cartesian_product
                else [x for _ in range(len(kinase_seqs)) for x in base_site_uniprot_accession]
            )
            site_location = (
                base_site_location
                if not cartesian_product
                else [x for _ in range(len(kinase_seqs)) for x in base_site_location]
            )
            orig_kin_seq_len = len(kinase_seqs)
            if cartesian_product:
                kinase_seqs = [x for x in kinase_seqs for _ in range(len(site_seqs))]
            if not cartesian_product:
                site_seqs = site_seqs
            else:
                site_seqs = [x for _ in range(orig_kin_seq_len) for x in site_seqs]
            assert all(
                [
                    len(x) == len(kinase_seqs)
                    for x in [
                        kinase_seqs,
                        site_seqs,
                        boolean_predictions,
                        numerical_scores,
                        group_predictions,
                        kinase_gene_names,
                        site_gene_names,
                        kinase_uniprot_accession,
                        site_uniprot_accession,
                        site_location,
                    ]
                ]
            ), (
                f"Error: Results lists are not all the same length (Debug: {len(kinase_seqs)=} vs"
                f" {len(kinase_seqs)=} vs {len(site_seqs)=} vs {len(boolean_predictions)=} vs"
                f" {len(numerical_scores)=} vs {len(group_predictions)=} vs {len(kinase_gene_names)=} vs"
                f" {len(site_gene_names)=} vs {len(kinase_uniprot_accession)=} vs {len(site_uniprot_accession)=} vs"
                f" {len(site_location)=})"
            )

            ret = []
            for k, s, sc, gp, p, kgn, sgn, kua, sua, sl in zip(
                kinase_seqs,
                site_seqs,
                numerical_scores,
                group_predictions,
                boolean_predictions,
                kinase_gene_names,
                site_gene_names,
                kinase_uniprot_accession,
                site_uniprot_accession,
                site_location,
            ):
                ret.append(
                    {
                        "Kinase Uniprot Accession": kua,
                        "Site Uniprot Accession": sua,
                        "Site Location": sl,
                        "Kinase Gene Name": kgn,
                        "Site Gene Name": sgn,
                        "Prediction": p,
                    }
                )
                if scores:
                    ret[-1]["Score"] = sc
                if group_output:
                    ret[-1]["Kinase Group Prediction"] = gp
                if not suppress_seqs_in_output:
                    ret[-1]["Kinase Sequence"] = k
                    ret[-1]["Site Sequence"] = s

        else:
            if scores:
                ret = [(n, b) for n, b in zip(numerical_scores, boolean_predictions)]
            else:
                ret = boolean_predictions
        if predictions_output_format not in ["inorder", "dictionary"]:
            file_name = (
                f"{str(pathlib.Path(__file__).parent.resolve())}/"
                f"../out/{get_file_name('results', re.sub(r'.*?_json', 'json', predictions_output_format))}"
            )
            logger.info(f"Writing results to {file_name}")
            table = pd.DataFrame(ret)
            if "json" in predictions_output_format:
                table.to_json(open(file_name, "w"), orient="records", indent=3)
            elif predictions_output_format == "csv":
                table.to_csv(file_name, index=False)
            else:
                assert predictions_output_format == "sqlite"
                for col in table.columns:
                    if table[col].dtype == "object":
                        table[col] = table[col].astype("string")
                table.to_sql(name="DeepKS Results", con=sqlite3.connect(file_name), index=False, if_exists="fail")
        else:
            return ret

    def predict(
        self,
        kinase_seqs: list[str],
        kin_info: dict,
        site_seqs: list[str],
        site_info: dict,
        predictions_output_format: str = "inorder",
        suppress_seqs_in_output: bool = False,
        device: str = "cpu",
        scores: bool = False,
        normalize_scores: bool = False,
        cartesian_product: bool = False,
        group_on: Literal["site", "kin"] = "site",
        group_output: bool = False,
        bypass_group_classifier: list[str] = [],
        convert_raw_to_prob=True,
    ):
        temp_df = pd.DataFrame({"kinase": kinase_seqs}).drop_duplicates(keep="first").reset_index()
        seq_to_id = {seq: "KINID" + str(idx) for idx, seq in zip(temp_df.index, temp_df["kinase"])}
        id_to_seq = {v: k for k, v in seq_to_id.items()}
        assert len(seq_to_id) == len(id_to_seq), "Error: seq_to_id and id_to_seq are not the same length"
        site_seq_to_id = {seq: "SITEID" + str(idx) for idx, seq in enumerate(site_seqs)}
        site_id_to_seq = {v: k for k, v in site_seq_to_id.items()}
        assert len(site_seq_to_id) == len(
            site_id_to_seq
        ), "Error: site_seq_to_id and site_id_to_seq are not the same length"

        if cartesian_product:
            pair_id_range = range(len(kinase_seqs) * len(site_seqs))
        else:
            if isinstance(self.group_classifier, SiteGroupClassifier):
                pair_id_range = range(len(site_seqs))
            else:
                pair_id_range = range(len(kinase_seqs))
        data_dict = {
            "Gene Name of Provided Kin Seq": [seq_to_id[k] for k in kinase_seqs],
            "Gene Name of Kin Corring to Provided Sub Seq": [site_seq_to_id[s] for s in site_seqs],
            "Kinase Sequence": kinase_seqs,
            "Site Sequence": site_seqs,
            "pair_id": [f"Pair # {i}" for i in pair_id_range],
            "Class": [-1],
            "Num Seqs in Orig Kin": ["N/A"],
        }

        if bypass_group_classifier:
            data_dict.update({"known_groups": bypass_group_classifier})

        with tf.NamedTemporaryFile("w") as f:
            if cartesian_product:
                with open(f.name, "w") as f2:
                    f2.write(json.dumps(data_dict, indent=3))
            else:
                efficient_to_csv(data_dict, f.name)

            # The "meat" of the prediction process.
            res = self.group_classify_then_evaluate(
                {"test": f.name, "device": device},
                f.name,
                predict_mode=True,
                bypass_group_classifier=bypass_group_classifier,
                get_emp_eqn=convert_raw_to_prob,
                cartesian_product=cartesian_product,
                device=device,
                group_on=group_on,
            )

            assert isinstance(res, list), "res is not a list"
            assert res is not None, "res is None"
            emp_eqns = [x[1][3] for x in res]
            group_predictions = [x[1][2] for x in res]
            numerical_scores = [x[1][1] for x in res]
            pred_ids = [x[0] for x in res]
            if normalize_scores:
                max_ = max(numerical_scores)
                min_ = min(numerical_scores)
                numerical_scores = [(x - min_) / (max_ - min_) for x in numerical_scores]
            if convert_raw_to_prob:
                assert len(emp_eqns) == len(numerical_scores), "emp_eqns and numerical_scores are not the same length"
                mapped_numerical_scores = []
                for pred_id, x, emp_eqn in zip(pred_ids, numerical_scores, emp_eqns):
                    if isinstance(emp_eqn, Callable):
                        mapped_numerical_scores.append(emp_eqn([x])[0])
                    else:
                        warnings.warn(
                            colored(f"No empirical equation found for query {pred_id}. Returning raw score.", "yellow"),
                            UserWarning,
                        )
                        mapped_numerical_scores.append(x)
            boolean_predictions = []
            for x in res:
                if x[1][0]:
                    boolean_predictions.append("True Phos. Pair")
                else:
                    boolean_predictions.append("False Phos. Pair")
        logger.status("Predictions Complete!")
        return self._package_results(
            predictions_output_format,
            kin_info,
            site_info,
            kinase_seqs,
            site_seqs,
            cartesian_product,
            boolean_predictions,
            numerical_scores,
            group_predictions,
            scores,
            group_output,
            suppress_seqs_in_output,
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


def smart_save_msc(msc: MultiStageClassifier):
    bin_ = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "bin")
    max_version = 0
    for file in os.listdir(bin_):
        if v := re.search(r"(UNITTESTVERSION|)deepks_msc_weights\.((|-)\d+)\.cornichon", file):
            max_version = max(max_version, int(v.group(2)) + 1)
    save_path = os.path.join(bin_, f"deepks_msc_weights.{max_version}.cornichon")
    logger.status(f"Serializing and Saving Group Classifier to Disk. ({save_path})")
    with open(save_path, "wb") as f:
        pickle.dump(msc, f)


def efficient_to_csv(data_dict: dict, outfile: str):
    """Efficiently writes a dictionary of lists to a csv file by not storing the entire data object in memory.

    Parameters
    ----------
    data_dict : dict
        A dictionary of lists to be written to a csv file. The keys are the column names and the values are the column values.
    outfile : str
        The path to the output csv file.

    Notes
    -----
    Because ``data_dict`` is a dictionary of lists, the lists must all be the same length. If they are not, an `AssertionError` will be raised.
    """
    assert all([isinstance(x, list) for x in data_dict.values()])
    headers = ",".join(data_dict.keys())
    max_len = max(len(x) for x in data_dict.values())
    dl = {k: len(v) for k, v in data_dict.items()}
    assert all(
        [x == 1 or x == max_len for x in [len(x) for x in data_dict.values()]]
    ), f"Tried to write uneven length lists to csv. Data Dict lengths: {dl}"
    for k in data_dict:
        if len(data_dict[k]) == 1:
            data_dict[k] = data_dict[k] * max_len
    with open(outfile, "w") as f:
        f.write(headers + "\n")
        lines_written = 1
        logger.status("Writing prediction queries to tempfile.")
        for _, row_tuple in enumerate(zip(*data_dict.values())):
            f.write(",".join([str(x) for x in row_tuple]) + "\n")
            lines_written += 1


if __name__ == "__main__":  # pragma: no cover
    nn = IndividualClassifiers.load_all(join_first("bin/deepks_nn_weights.11.cornichon", 1, __file__))
    with open(join_first("bin/deepks_gc_weights.2.cornichon", 1, __file__), "rb") as f:
        gc: GroupClassifier = pickle.load(f)

    smart_save_msc(MultiStageClassifier(gc, nn))
