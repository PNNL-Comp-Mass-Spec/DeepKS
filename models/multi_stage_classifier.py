if __name__ == "__main__":
    from ..splash.write_splash import write_splash

    write_splash("gc_trainer")
    print("Progress: Loading Modules", flush=True)
from copy import deepcopy
import pandas as pd, numpy as np, tempfile as tf, json, cloudpickle as pickle, pathlib, os, tqdm
import re, sqlite3
from ..tools.get_needle_pairwise import get_needle_pairwise_mtx
from .individual_classifiers import IndividualClassifiers
from . import group_classifier_definitions as grp_pred
from . import individual_classifiers
from ..tools.parse import parsing
from ..tools.file_names import get as get_file_name
from ..tools import make_fasta as dist_mtx_maker
from sklearn.neural_network import MLPClassifier
from ..api.cfg import PRE_TRAINED_NN, PRE_TRAINED_GC
from sklearn.utils.validation import check_is_fitted
from termcolor import colored

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)


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

    def evaluate(
        self,
        newargs,
        Xy_formatted_input_file: str,
        evaluation_kwargs=None,
        predict_mode=False,
        bypass_group_classifier={},
    ):
        print(colored("Status: Prediction Step [1/2]: Sending input kinases to group classifier", "green"))
        # Open XY_formatted_input_file
        if "test_json" in newargs:
            input_file = json.load(open(Xy_formatted_input_file))
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
        unq_symbols = input_file["orig_lab_name"].drop_duplicates()
        unq_symbols_inds = unq_symbols.index.tolist()
        unq_symbols = unq_symbols.tolist()

        if bypass_group_classifier:
            true_symbol_to_grp_dict = dict(zip(unq_symbols, [input_file['known_groups'][i] for i in unq_symbols_inds]))

        if not predict_mode or bypass_group_classifier:
            assert true_symbol_to_grp_dict is not None
            groups_true = [true_symbol_to_grp_dict[u] for u in unq_symbols]
        else:
            groups_true = [None for _ in unq_symbols]

        ### Perform Real Accuracy
        groups_prediction = self.group_classifier.predict(unq_symbols)
        pred_grp_dict = {symbol: grp for symbol, grp in zip(unq_symbols, groups_prediction)}
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
            self.individual_classifiers.roc_evaluation(newargs, pred_grp_dict, true_grp_dict, predict_mode)
        if predict_mode:
            newargs["test" if "test_json" not in newargs else "test_json"] = Xy_formatted_input_file
            res = self.individual_classifiers.roc_evaluation(
                newargs, pred_grp_dict if not bypass_group_classifier else true_grp_dict, true_groups=None, predict_mode=True
            )

        print(
            colored(
                (
                    "Status: Prediction Step [2/2]: Sending input kinases to individual group classifiers, based on"
                    " step [1/2]"
                ),
                "green"
            )
        )
        
        return res

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
        cartesian_product: bool = False,  # TODO - Use this if there's a better way of handing cartesian products
        group_output: bool = False,
        bypass_group_classifier: list[str] = [],
    ):
        temp_df = pd.DataFrame({"kinase": kinase_seqs}).drop_duplicates(keep="first").reset_index()
        seq_to_id = {seq: "KINID" + str(idx) for idx, seq in zip(temp_df.index, temp_df["kinase"])}
        id_to_seq = {v: k for k, v in seq_to_id.items()}
        assert len(seq_to_id) == len(id_to_seq), "Error: seq_to_id and id_to_seq are not the same length"

        data_dict = {
            "lab_name": ["N/A"],
            "orig_lab_name": [seq_to_id[k] for k in kinase_seqs],
            "lab": kinase_seqs,
            "seq": site_seqs,
            "pair_id": [
                f"Pair # {i}" for i in range(len(kinase_seqs) * len(site_seqs) if cartesian_product else len(site_seqs))
            ],
            "class": [-1],
            "num_seqs": ["N/A"],
        }

        data_dict = (data_dict | {"known_groups": bypass_group_classifier}) if bypass_group_classifier else data_dict

        with tf.NamedTemporaryFile("w") as f:
            print(
                colored("Status: Aligning Novel Kinase Sequences (for the purpose of the group classifier).", "green")
            )
            self.align_novel_kin_seqs(
                id_to_seq
            )  # existing_seqs = pd.read_csv("../data/raw_data/kinase_seqs_494.csv").index.tolist()
            print(colored("Status: Done Aligning Novel Kinase Sequences.", "green"))
            if cartesian_product:
                with open(f.name, "w") as f2:
                    f2.write(json.dumps(data_dict, indent=3))
                res = self.evaluate(
                    {"test_json": f.name, "device": device},
                    f.name,
                    predict_mode=True,
                    bypass_group_classifier=bypass_group_classifier,
                )
            else:
                efficient_to_csv(data_dict, f.name)
                # The "meat" of the prediction process.
                res = self.evaluate({"test": f.name, "device": device}, f.name, predict_mode=True)
            assert res is not None, "Error: res is None"
            group_predictions = [x[1][2] for x in res]
            numerical_scores = [x[1][1] for x in res]
            if normalize_scores:
                max_ = max(numerical_scores)
                min_ = min(numerical_scores)
                numerical_scores = [(x - min_) / (max_ - min_) for x in numerical_scores]
            boolean_predictions = ["False Phos. Pair" if not x[1][0] else "True Phos. Pair" for x in res]
        print(colored("Status: Predictions Complete!", "green"))
        if "dict" in predictions_output_format or re.search(r"(sqlite|csv)", predictions_output_format):
            print(colored("Status: Copying Results to Dictionary.", "green"))

            # if cartesian_product:
            #     kinase_seqs = list(itertools.chain(*[[x]*len(site_seqs) for x in kinase_seqs]))
            #     site_seqs = list(itertools.chain(*[site_seqs]*len(kinase_seqs)))

            base_kinase_gene_names = [kin_info[k]["Gene Name"] for k in kinase_seqs]
            base_kinase_uniprot_accession = [kin_info[k]["Uniprot Accession ID"] for k in kinase_seqs]
            base_site_gene_names = [site_info[s]["Gene Name"] for s in site_seqs]
            base_site_uniprot_accession = [site_info[s]["Uniprot Accession ID"] for s in site_seqs]
            base_site_location = [site_info[s]["Location"] for s in site_seqs]

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
                    combined_fasta.write(
                        open(temp_fasta_known.name, "r").read() + open(temp_fasta_novel.name, "r").read()
                    )

                with tf.NamedTemporaryFile() as temp_mtx_out:
                    novel_df = get_needle_pairwise_mtx(
                        combined_fasta.name,
                        temp_mtx_out.name,
                        num_procs=6,
                        restricted_combinations=[train_kin_list, val_kin_list],
                    )
        novel_df.rename(
            columns={"RET|PTC2|Q15300": "RET/PTC2|Q15300"} if "RET|PTC2|Q15300" in novel_df.columns else {},
            inplace=True,
        )
        for additional_name in additional_name_dict:
            grp_pred.MTX[additional_name] = grp_pred.MTX[additional_name_dict[additional_name]]
            grp_pred.MTX.loc[additional_name] = grp_pred.MTX.loc[additional_name_dict[additional_name]]

        grp_pred.MTX = pd.concat([grp_pred.MTX[train_kin_list], novel_df])


def main(run_args):
    train_kins, val_kins, test_kins, train_true, val_true, test_true = grp_pred.get_ML_sets(
        "../data/preprocessing/pairwise_mtx_826.csv",
        "../data/preprocessing/tr_kins.json",
        "../data/preprocessing/vl_kins.json",
        "../data/preprocessing/te_kins.json",
        "../data/preprocessing/kin_to_fam_to_grp_826.csv",
    )

    train_kins, eval_kins, train_true, eval_true = (  # type: ignore
        np.array(train_kins + val_kins),
        np.array(test_kins),
        np.array(train_true + val_true),
        np.array(test_true),
    )
    # train_kins = np.char.replace(train_kins, "RET/PTC2|Q15300", "RET|PTC2|Q15300")
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

    individual_classifiers = IndividualClassifiers.load_all(
        run_args["load_include_eval"] if run_args["load_include_eval"] is not None else run_args["load"],
        run_args["device"],
    )
    if run_args["c"]:
        check_is_fitted(group_classifier.model)
        pickle.dump(individual_classifiers, open(f"{PRE_TRAINED_NN}", "wb"))
        open(f"{PRE_TRAINED_GC}", "wb").write(pickle.dumps(group_classifier))
        print(colored("Info: Saved pre-trained classifiers to disk with the following paths:", "blue"))
        print(f"* {PRE_TRAINED_NN}")
        print(f"* {PRE_TRAINED_GC}")
        print(colored("Status: Exiting Successfully.", "green"))
        return

    msc = MultiStageClassifier(group_classifier, individual_classifiers)
    msc.evaluate(run_args, run_args["test"])


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


if __name__ == "__main__":
    run_args = parsing()
    assert (
        run_args["load"] is not None or run_args["load_include_eval"]
    ), "For now, multi-stage classifier must be run with --load or --load-include-eval."
    main(run_args)
