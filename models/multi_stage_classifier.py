if __name__ == "__main__":
    from ..splash.write_splash import write_splash
    from termcolor import colored

    write_splash("main_gc_trainer")
    print(colored("Progress: Loading Modules", "green"), flush=True)
import pandas as pd, numpy as np, tempfile as tf, json, cloudpickle as pickle, pathlib, os, tqdm, re, sqlite3, warnings
import torch, argparse, socket
from typing import Callable, Union
from ..tools.get_needle_pairwise import get_needle_pairwise_mtx
from .individual_classifiers import IndividualClassifiers
from . import group_classifier_definitions as grp_pred
from .group_classifier_definitions import check_is_fitted
from . import individual_classifiers
from ..tools.file_names import get as get_file_name
from ..tools import make_fasta as dist_mtx_maker
from sklearn.neural_network import MLPClassifier
from termcolor import colored
from .site_classifier import SiteClassifier

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)

warnings.filterwarnings(action="always", category=UserWarning)

join_first = lambda levels, x: os.path.join(pathlib.Path(__file__).parent.resolve(), *[".."] * levels, x)


def smart_save_gc(individual_classifier: grp_pred.SKGroupClassifier):
    bin_ = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "bin")
    max_version = -1
    for file in os.listdir(bin_):
        if v := re.search(r"(UNITTESTVERSION|)deepks_gc_weights\.((|-)\d+)\.cornichon", file):
            max_version = max(max_version, int(v.group(2)) + 1)
    save_path = os.path.join(bin_, f"deepks_gc_weights.{max_version}.cornichon")
    print(colored(f"Status: Serializing and Saving Group Classifier to Disk. ({save_path})", "green"))
    with open(save_path, "wb") as f:
        pickle.dump(individual_classifier, f)


class MultiStageClassifier:
    def __init__(
        self,
        group_classifier: grp_pred.SKGroupClassifier,
        individual_classifiers: individual_classifiers.IndividualClassifiers,
    ):
        self.group_classifier = group_classifier
        self.individual_classifiers = individual_classifiers

    def __str__(self):
        return "MultiStageClassifier(group_classifier=%s, individual_classifiers=%s)" % (
            self.group_classifier,
            self.individual_classifiers,
        )

    def evaluation_preparation(
        self, addl_args, Xy_formatted_input_file: str, predict_mode=False, bypass_group_classifier={}, get_emp_eqn=True
    ):
        print(colored("Status: Prediction Step [1/2]: Sending input kinases to group classifier", "green"))
        # Open XY_formatted_input_file
        Xy_formatted_input_file = join_first(1, Xy_formatted_input_file)
        if "test_json" in addl_args:
            with open(Xy_formatted_input_file) as f:
                input_file = json.load(f)
            input_file = {k: pd.Series(v) for k, v in input_file.items()}
        else:
            input_file = pd.read_csv(Xy_formatted_input_file)
        if not predict_mode:
            _, _, true_symbol_to_grp_dict = individual_classifiers.IndividualClassifiers.get_symbol_to_grp_dict(
                "../data/preprocessing/kin_to_fam_to_grp_826.csv"
            )
        else:
            true_symbol_to_grp_dict = None

        ### Get group predictions
        unq_symbols = input_file["Site Sequences"].drop_duplicates()
        unq_symbols_inds = unq_symbols.index.tolist()
        unq_symbols = unq_symbols.tolist()

        if bypass_group_classifier:
            true_symbol_to_grp_dict = dict(zip(unq_symbols, [input_file["known_groups"][i] for i in unq_symbols_inds]))

        if not predict_mode or bypass_group_classifier:
            assert true_symbol_to_grp_dict is not None
            groups_true = [true_symbol_to_grp_dict[u] for u in unq_symbols]
        else:
            groups_true = [None for _ in unq_symbols]

        ### Perform Real Accuracy
        groups_prediction = self.group_classifier.predict(unq_symbols)
        pred_grp_dict = {symbol: grp for symbol, grp in zip(unq_symbols, groups_prediction)}
        self.pred_grp_dict = pred_grp_dict
        true_grp_dict = {symbol: grp for symbol, grp in zip(unq_symbols, groups_true)}

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
        if predict_mode is False or bypass_group_classifier:
            print(
                colored(
                    (
                        "Info: Group Classifier Accuracy"
                        f" {'(since we have known groups) ' if bypass_group_classifier else ''}—"
                        f" {self.group_classifier.test(groups_true, groups_prediction)}"
                    ),
                    "blue",
                )
            )
            self.individual_classifiers.evaluation(
                addl_args, pred_grp_dict, true_grp_dict, predict_mode, get_emp_eqn=get_emp_eqn
            )
            # assert isinstance(emp_eqn, Callable)
        if predict_mode:
            addl_args["test" if "test_json" not in addl_args else "test_json"] = Xy_formatted_input_file
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
                pred_grp_dict if not bypass_group_classifier else true_grp_dict,
                true_groups={},
                predict_mode=True,
                get_emp_eqn=get_emp_eqn,
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
            print(colored("Status: Copying Results to Dictionary.", "green"))

            base_kinase_gene_names = [kin_info[k]["Gene Name"] if k in kin_info else "?" for k in kinase_seqs]
            base_kinase_uniprot_accession = [
                kin_info[k]["Uniprot Accession ID"] if k in kin_info else "?" for k in kinase_seqs
            ]
            base_site_gene_names = [site_info[s]["Gene Name"] if s in site_info else "?" for s in site_seqs]
            base_site_uniprot_accession = [
                site_info[s]["Uniprot Accession ID"] if s in site_info else "?" for s in site_seqs
            ]
            base_site_location = [site_info[s]["Location"] if s in site_info else "?" for s in site_seqs]

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
            kinase_seqs = (
                kinase_seqs if not cartesian_product else [x for x in kinase_seqs for _ in range(len(site_seqs))]
            )
            site_seqs = site_seqs if not cartesian_product else [x for _ in range(orig_kin_seq_len) for x in site_seqs]
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

            ret = [
                {
                    "Kinase Uniprot Accession": kua,
                    "Site Uniprot Accession": sua,
                    "Site Location": sl,
                    "Kinase Gene Name": kgn,
                    "Site Gene Name": sgn,
                    "Prediction": p,
                }
                | (
                    (
                        (
                            ({} if not scores else {"Score": sc})
                            | ({} if not group_output else {"Kinase Group Prediction": gp})
                        )
                        | ({} if suppress_seqs_in_output else {"Kinase Sequence": k, "Site Sequence": s})
                    )
                )
                for k, s, p, sc, gp, kgn, sgn, kua, sua, sl in zip(
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
                )
            ]
        else:
            ret = [(n, b) for n, b in zip(numerical_scores, boolean_predictions)] if scores else boolean_predictions
        if predictions_output_format not in ["inorder", "dictionary"]:
            file_name = (
                f"{str(pathlib.Path(__file__).parent.resolve())}/"
                f"../out/{get_file_name('results', re.sub(r'.*?_json', 'json', predictions_output_format))}"
            )
            print(colored(f"Info: Writing results to {file_name}", "blue"))
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
        group_output: bool = False,
        bypass_group_classifier: list[str] = [],
        convert_raw_to_prob=True,
    ):
        temp_df = pd.DataFrame({"kinase": kinase_seqs}).drop_duplicates(keep="first").reset_index()
        seq_to_id = {seq: "KINID" + str(idx) for idx, seq in zip(temp_df.index, temp_df["kinase"])}
        id_to_seq = {v: k for k, v in seq_to_id.items()}
        assert len(seq_to_id) == len(id_to_seq), "Error: seq_to_id and id_to_seq are not the same length"

        data_dict = {
            "Gene Name of Provided Kin Seq": [seq_to_id[k] for k in kinase_seqs],
            "Gene Name of Kin Corring to Provided Sub Seq": ["N/A"],
            "Kinase Sequence": kinase_seqs,
            "Site Sequence": site_seqs,
            "pair_id": [
                f"Pair # {i}" for i in range(len(kinase_seqs) * len(site_seqs) if cartesian_product else len(site_seqs))
            ],
            "Class": [-1],
            "Num Seqs in Orig Kin": ["N/A"],
        }

        data_dict = (data_dict | {"known_groups": bypass_group_classifier}) if bypass_group_classifier else data_dict

        with tf.NamedTemporaryFile("w") as f:
            if cartesian_product:
                with open(f.name, "w") as f2:
                    f2.write(json.dumps(data_dict, indent=3))
            else:
                efficient_to_csv(data_dict, f.name)

            # The "meat" of the prediction process.
            res = self.evaluation_preparation(
                {"test": f.name, "device": device},
                f.name,
                predict_mode=True,
                bypass_group_classifier=bypass_group_classifier,
                get_emp_eqn=convert_raw_to_prob,
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
            boolean_predictions = ["False Phos. Pair" if not x[1][0] else "True Phos. Pair" for x in res]
        print(colored("Status: Predictions Complete!", "green"))
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

    def align_novel_kin_seqs(
        self,
        kin_id_to_seq: dict[str, str],
        existing_seqs=["../data/raw_data/kinase_seq_826.csv", "../data/raw_data/kinase_seq_494.csv"],
    ) -> None:
        train_kin_list = self.group_classifier.X_train
        existing_seqs_to_known_ids = pd.concat(
            [pd.read_csv(existing_seq) for existing_seq in existing_seqs], ignore_index=True
        )  # TODO - fix variable names
        existing_seqs_to_known_ids.drop(columns=["symbol"], inplace=True)
        existing_seqs_to_known_ids.drop_duplicates(inplace=True, keep="first")
        existing_seqs_to_known_ids["Symbol"] = (
            existing_seqs_to_known_ids["gene_name"] + "|" + existing_seqs_to_known_ids["kinase"]
        )
        new_filename = existing_seqs = f"../data/raw_data/kinase_seq_{len(existing_seqs_to_known_ids)}.csv"
        existing_seqs_to_known_ids.to_csv(new_filename, index=False)
        existing_seqs_to_known_ids = existing_seqs_to_known_ids.set_index("kinase_seq").to_dict()["Symbol"]
        val_kin_list = [x for x in kin_id_to_seq if kin_id_to_seq[x] not in existing_seqs_to_known_ids]
        if len(val_kin_list) == 0:
            print(colored("Info: Leveraging pre-computed pairwise alignment scores!", "blue"))
        additional_name_dict = {
            x: existing_seqs_to_known_ids[kin_id_to_seq[x]]
            for x in kin_id_to_seq
            if kin_id_to_seq[x] in existing_seqs_to_known_ids
        }
        novel_df = None
        with tf.NamedTemporaryFile() as temp_df_in_file:
            df_in = pd.DataFrame(
                {"kinase": "", "kinase_seq": [kin_id_to_seq[v] for v in val_kin_list], "gene_name": val_kin_list}
            )
            df_in.to_csv(temp_df_in_file.name, index=False)
            with tf.NamedTemporaryFile() as temp_fasta_known, tf.NamedTemporaryFile() as temp_fasta_novel, tf.NamedTemporaryFile() as combined_temp_fasta:
                dist_mtx_maker.make_fasta(existing_seqs, temp_fasta_known.name)
                dist_mtx_maker.make_fasta(temp_df_in_file.name, temp_fasta_novel.name)
                with open(combined_temp_fasta.name, "w") as combined_fasta:
                    with open(temp_fasta_known.name, "r") as known_fasta, open(
                        temp_fasta_novel.name, "r"
                    ) as novel_fasta:
                        combined_fasta.write(known_fasta.read() + novel_fasta.read())

                with tf.NamedTemporaryFile() as temp_mtx_out:
                    novel_df = get_needle_pairwise_mtx(
                        combined_fasta.name,
                        temp_mtx_out.name,
                        num_procs=6,
                        restricted_combinations=[train_kin_list, val_kin_list],
                    )
        # novel_df.rename(
        #     columns={"RET|PTC2|Q15300": "RET/PTC2|Q15300"} if "RET|PTC2|Q15300" in novel_df.columns else {},
        #     inplace=True,
        # )
        for additional_name in additional_name_dict:
            grp_pred.MTX[additional_name] = grp_pred.MTX[additional_name_dict[additional_name]]
            grp_pred.MTX.loc[additional_name] = grp_pred.MTX.loc[additional_name_dict[additional_name]]

        grp_pred.MTX = pd.concat([grp_pred.MTX[train_kin_list], novel_df])


def get_group_classifier():
    args = parse_args()
    with open(str(args["gc_params"]), "r") as f:
        hps = json.load(f)
    gc = SiteClassifier.get_group_classifier(join_first(1, args["train"]), join_first(1, args["kin_fam_grp"]), {}, hps)
    if args.get("val"):
        (vl,) = SiteClassifier.format_data(join_first(1, args["val"]), kin_fam_grp=args["kin_fam_grp"])
        print(colored(f"Val Info: Val Performance of GC: {SiteClassifier.performance(gc, *vl)}", "blue"))
    if args.get("s"):
        smart_save_gc(gc)
    return gc


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


def parse_args() -> dict[str, Union[str, None]]:
    print(colored("Progress: Parsing Arguments", "green"))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        help=(
            "Specify device. Choices are"
            f" {sorted(set(['cpu']).union([f'cuda:{i}' for i in range(torch.cuda.device_count())]))}."
        ),
        metavar="<device>",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--train", type=str, help="Specify train file name", required=True, metavar="<train_file_name.csv>"
    )
    parser.add_argument(
        "--val", type=str, help="Specify validation file name", required=False, metavar="<val_file_name.csv>"
    )

    parser.add_argument(
        "--gc-params",
        type=str,
        help="Specify Group Classifier and its hyperparameters file name",
        required=False,
        default=join_first(0, "GC_params.json"),
        metavar="gc_params.json>",
    )

    # parser.add_argument(
    #     "--gc-training-params",
    #     type=str,
    #     help="Specify Kinase Substrate Relationship training options file name",
    #     required=False,
    #     default=join_first(0, "KSR_training_params.json"),
    #     metavar="<gc_params.json>",
    # )

    # parser.add_argument(
    #     "--nni-params",
    #     type=str,
    #     help="Specify Nerual Net Interface options file name",
    #     required=False,
    #     default=join_first(0, "NNI_params.json"),
    #     metavar="<gc_params.json>",
    # )

    parser.add_argument(
        "--kin-fam-grp",
        type=str,
        help="Specify Kinase-Family-Group file name",
        required=False,
        default=join_first(1, "data/preprocessing/kin_to_fam_to_grp_826.csv"),
        metavar="<kin_fam_grp.csv>",
    )

    parser.add_argument("-s", action="store_true", help="Include to save state", required=False)

    try:
        args = vars(parser.parse_args())
    except Exception as e:
        print(e)
        exit(1)

    device_eligibility(args["device"])
    extra_files = (["val"] if args.get("val") else []) + (["gc_params"] if args.get("gc_params") else [])
    for fn in ["train", "kin_fam_grp"] + extra_files:
        f = str(args[fn])
        try:
            assert "formatted" in f or f.endswith(
                ".json"
            ), "'formatted' is not in the train or filename. Did you select the correct file?"
        except AssertionError as e:
            warnings.warn(str(e), UserWarning)
        assert os.path.exists(join_first(1, f)), f"Input file '{join_first(1, f)}' does not exist."

    return args


def get_multi_stage_classifier(load_gc=False):
    if not load_gc:
        train_kins, val_kins, _, train_true, val_true, _ = grp_pred.get_ML_sets(
            *(
                [
                    join_first(0, x)
                    for x in [
                        "../data/preprocessing/pairwise_mtx_826.csv",
                        "../data/preprocessing/tr_kins.json",
                        "../data/preprocessing/vl_kins.json",
                        "../data/preprocessing/te_kins.json",
                        "../data/preprocessing/kin_to_fam_to_grp_826.csv",
                    ]
                ]
                + [None]
            ),
            verbose=False,
        )

        train_kins, train_true = np.array(train_kins + val_kins), np.array(train_true + val_true)

        group_classifier = grp_pred.SKGroupClassifier(
            X_train=train_kins,
            y_train=train_true,
            classifier=MLPClassifier,
            hyperparameters={
                "activation": "identity",
                "max_iter": 500,
                "learning_rate": "adaptive",
                "hidden_layer_sizes": (1000, 500, 100, 50),
                "random_state": 42,
                "alpha": 1e-7,
            },
        )

        ### Use KNN instead
        # group_classifier = grp_pred.SKGroupClassifier(
        #     X_train=train_kins,
        #     y_train=train_true,
        #     classifier=KNeighborsClassifier,
        #     hyperparameters={"metric": "chebyshev", "n_neighbors": 1},
        # )

    else:
        group_classifier = pickle.load(
            open(join_first(1, "bin/deepks_gc_weights.-1.cornichon"), "rb")
        )  # FIXME Not general enough

    return group_classifier


def efficient_to_csv(data_dict, outfile):
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
        for _, row_tuple in tqdm.tqdm(
            enumerate(zip(*data_dict.values())),
            total=max_len,
            desc=colored("Status: Writing prediction queries to tempfile.", "cyan"),
            colour="cyan",
        ):
            f.write(",".join([str(x) for x in row_tuple]) + "\n")
            lines_written += 1


def smart_save_msc(msc: MultiStageClassifier):
    bin_ = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "bin")
    max_version = 0
    for file in os.listdir(bin_):
        if v := re.search(r"(UNITTESTVERSION|)deepks_msc_weights\.((|-)\d+)\.cornichon", file):
            max_version = max(max_version, int(v.group(2)) + 1)
    save_path = os.path.join(bin_, f"deepks_msc_weights.{max_version}.cornichon")
    print(colored(f"Status: Serializing and Saving Group Classifier to Disk. ({save_path})", "green"))
    with open(save_path, "wb") as f:
        pickle.dump(msc, f)


def combine_ic_and_gc(nn: IndividualClassifiers, gc: SiteClassifier) -> MultiStageClassifier:
    return MultiStageClassifier(gc, nn)


if __name__ == "__main__":
    gc = get_multi_stage_classifier()
    # smart_save_gc(gc)

    nn = IndividualClassifiers.load_all(join_first(1, "bin/deepks_nn_weights.11.cornichon"))

    smart_save_msc(combine_ic_and_gc(nn, gc))
