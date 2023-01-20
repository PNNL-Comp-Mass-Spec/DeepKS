if __name__ == "__main__":
    from .write_splash import write_splash

    write_splash()
    print("Progress: Loading Modules", flush=True)
import pandas as pd, numpy as np, tempfile as tf, random, json, datetime, dateutil.tz
from typing import Union
from ..tools.get_needle_pairwise import get_needle_pairwise_mtx
from ..models.group_prediction_from_hc import get_coordinates
from .individual_classifiers import IndividualClassifiers
from . import group_prediction_from_hc as grp_pred
from . import individual_classifiers
from ..tools.parse import parsing
from ..data.preprocessing.PreprocessingSteps import get_labeled_distance_matrix as dist_mtx_maker
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


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

    def evaluate(self, newargs, Xy_formatted_input_file: str, evaluation_kwargs=None, predict_mode=False) -> None:
        print("Progress: Sending input kinases to group classifier")
        # Open XY_formatted_input_file
        input_file = pd.read_csv(Xy_formatted_input_file)
        _, _, true_symbol_to_grp_dict = individual_classifiers.IndividualClassifiers.get_symbol_to_grp_dict(
            "../data/preprocessing/kin_to_fam_to_grp_817.csv"
        )

        ### Get group predictions
        unq_symbols = input_file["orig_lab_name"].unique()
        if predict_mode is False:
            groups_true = [true_symbol_to_grp_dict[u] for u in unq_symbols]
        else:
            groups_true = [None for u in unq_symbols]

        ### Perform Real Accuracy
        groups_prediction = self.group_classifier.predict(unq_symbols)

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

        pred_grp_dict = {symbol: grp for symbol, grp in zip(unq_symbols, groups_prediction)}
        if predict_mode is False:
            true_grp_dict = {symbol: grp for symbol, grp in zip(unq_symbols, groups_true)}
        else:
            true_grp_dict = {symbol: None for symbol in unq_symbols}

        # Report performance
        if predict_mode is False:
            print(f"Info: Group Classifier Accuracy — {self.group_classifier.test(groups_true, groups_prediction)}")

        # Send to individual classifiers
        print("Progress: Sending input kinases to individual classifiers (with group predictions)")
        self.individual_classifiers.evaluations = {}
        self.individual_classifiers.roc_evaluation(newargs, pred_grp_dict, true_grp_dict, predict_mode)

    def predict(
        self,
        kinase_seqs: list[str],
        site_seqs: list[str],
        predictions_output_format: str = "in_order",
        verbose: bool = True,
    ) -> Union[None, list[bool], list[dict[str, Union[str, bool]]]]:

        temp_df = pd.DataFrame({"kinase": kinase_seqs}).drop_duplicates(keep="first")
        seq_to_id = {seq: "KinID" + str(idx) for idx, seq in zip(temp_df.index, temp_df["kinase"])}
        id_to_seq = {v: k for k, v in seq_to_id.items()}
        assert len(seq_to_id) == len(id_to_seq), "Error: seq_to_id and id_to_seq are not the same length"

        df_post_predict = pd.DataFrame(
            {
                "orig_lab_name": "N/A -- Predict Mode",
                "lab_name": [seq_to_id[k] for k in kinase_seqs],
                "lab": kinase_seqs,
                "seq": site_seqs,
                "class": "N/A -- Predict Mode",
                "num_seqs": "N/A -- Predict Mode",
            }
        )

        with tf.NamedTemporaryFile("w") as f:
            df_post_predict.to_csv(f.name, index=False)
            self.evaluate({"test": f.name}, f.name, predict_mode=True)

        res: list[bool] = []  # TODO!

        if "dict" in predictions_output_format:
            ret = [{"kinase": k, "site": s, "prediction": p} for k, s, p in zip(kinase_seqs, site_seqs, res)]
        else:
            ret = res
        if verbose:
            print(ret)
        if "json" in predictions_output_format:
            now = datetime.datetime.now(tz=dateutil.tz.tzlocal())
            file_name = (
                f"../out/{now.isoformat(timespec='milliseconds', sep='@')}.json"
            )
            json.dump(ret, open(file_name, "w"))
        else:
            return ret

    def align_novel_kin_seqs(self, kin_id_to_seq: dict[str, str]) -> np.ndarray:
        train_kin_list = self.group_classifier.X_train
        val_kin_list = list(kin_id_to_seq.keys())
        novel_df = None
        with tf.NamedTemporaryFile() as temp_df_in_file:
            df_in = pd.DataFrame(
                {"kinase": val_kin_list, "kinase_seq": list(kin_id_to_seq.values()), "gene_name": val_kin_list}
            )
            df_in.to_csv(temp_df_in_file.name, index=False)
            with tf.NamedTemporaryFile() as temp_fasta_file:
                dist_mtx_maker.make_fasta(temp_df_in_file.name, temp_fasta_file.name)
                with tf.NamedTemporaryFile() as temp_mtx_out:
                    novel_df = get_needle_pairwise_mtx(
                        temp_fasta_file.name,
                        temp_mtx_out.name,
                        num_procs=1,
                        restricted_combinations=[train_kin_list, val_kin_list],
                    )

        grp_pred.MTX = pd.concat([grp_pred.MTX, novel_df])
        return grp_pred.get_coordinates(train_kin_list, val_kin_list)[0]


def main(run_args):
    train_kins, val_kins, test_kins, train_true, val_true, test_true = grp_pred.get_ML_sets(
        "../tools/pairwise_mtx_822.csv",
        "json/tr.json",
        "json/vl.json",
        "json/te.json",
        "../data/preprocessing/kin_to_fam_to_grp_817.csv",
    )

    train_kins, eval_kins, train_true, eval_true = (  # type: ignore
        np.array(train_kins + val_kins),
        np.array(test_kins),
        np.array(train_true + val_true),
        np.array(test_true),
    )

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
        run_args["load_include_eval"] if run_args["load_include_eval"] is not None else run_args["load"]
    )

    msc = MultiStageClassifier(group_classifier, individual_classifiers)
    msc.evaluate(run_args, run_args["test"])


if __name__ == "__main__":
    run_args = parsing()
    assert (
        run_args["load"] is not None or run_args["load_include_eval"]
    ), "For now, multi-stage classifier must be run with --load or --load-include-eval."
    main(run_args)
