if __name__ == "__main__":
    from .write_splash import write_splash

    write_splash()
    print("Progress: Loading Modules", flush=True)
import pandas as pd, numpy as np, tempfile as tf, random, json, datetime, dateutil.tz, cloudpickle as pickle, pathlib, os
import sklearn.utils.validation
from typing import Union
from ..tools.get_needle_pairwise import get_needle_pairwise_mtx
from .individual_classifiers import IndividualClassifiers
from . import group_classifier_definitions as grp_pred
from . import individual_classifiers
from ..tools.parse import parsing
from ..data.preprocessing.PreprocessingSteps import get_labeled_distance_matrix as dist_mtx_maker
from sklearn.neural_network import MLPClassifier
from ..api.cfg import PRE_TRAINED_NN, PRE_TRAINED_GC
from sklearn.utils.validation import check_is_fitted

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

    def evaluate(self, newargs, Xy_formatted_input_file: str, evaluation_kwargs=None, predict_mode=False):
        print("Progress: Sending input kinases to group classifier")
        # Open XY_formatted_input_file
        input_file = pd.read_csv(Xy_formatted_input_file)
        if predict_mode is False:
            _, _, true_symbol_to_grp_dict = individual_classifiers.IndividualClassifiers.get_symbol_to_grp_dict(
                "../data/preprocessing/kin_to_fam_to_grp_817.csv"
            )
        else:
            true_symbol_to_grp_dict = None

        ### Get group predictions
        unq_symbols = input_file["orig_lab_name"].unique()
        if predict_mode is False:
            assert true_symbol_to_grp_dict is not None
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
            return {}
        else:
            return pred_grp_dict

    def predict(
        self,
        kinase_seqs: list[str],
        site_seqs: list[str],
        predictions_output_format: str = "in_order",
    ) -> Union[None, list[bool], list[dict[str, Union[str, bool]]]]:

        temp_df = pd.DataFrame({"kinase": kinase_seqs}).drop_duplicates(keep="first")
        seq_to_id = {seq: "KINID" + str(idx) for idx, seq in zip(temp_df.index, temp_df["kinase"])}
        id_to_seq = {v: k for k, v in seq_to_id.items()}
        assert len(seq_to_id) == len(id_to_seq), "Error: seq_to_id and id_to_seq are not the same length"

        df_post_predict = pd.DataFrame(
            {
                "lab_name": "N/A -- Predict Mode",
                "orig_lab_name": [seq_to_id[k] for k in kinase_seqs],
                "lab": kinase_seqs,
                "seq": site_seqs,
                "class": "N/A -- Predict Mode",
                "num_seqs": "N/A -- Predict Mode",
            }
        )

        with tf.NamedTemporaryFile("w") as f:
            df_post_predict.to_csv(f.name, index=False)
            self.align_novel_kin_seqs(df_post_predict.set_index("orig_lab_name").to_dict()["lab"])
            res = self.evaluate({"test": f.name}, f.name, predict_mode=True)


        if "dict" in predictions_output_format:
            ret = [{"kinase": k, "site": s, "prediction": p} for k, s, p in zip(kinase_seqs, site_seqs, res.values())]
        else:
            ret = res
        if "json" in predictions_output_format:
            now = datetime.datetime.now(tz=dateutil.tz.tzlocal())
            file_name = (
                f"../out/{now.isoformat(timespec='milliseconds', sep='@')}.json"
            )
            json.dump(ret, open(file_name, "w"))
        else:
            return ret

    def align_novel_kin_seqs(self, kin_id_to_seq: dict[str, str]) -> None:
        train_kin_list = self.group_classifier.X_train
        val_kin_list = list(kin_id_to_seq.keys())
        novel_df = None
        with tf.NamedTemporaryFile() as temp_df_in_file:
            df_in = pd.DataFrame(
                {"kinase": val_kin_list, "kinase_seq": list(kin_id_to_seq.values()), "gene_name": val_kin_list}
            )
            df_in.to_csv(temp_df_in_file.name, index=False)
            with tf.NamedTemporaryFile() as temp_fasta_known, tf.NamedTemporaryFile() as temp_fasta_novel, tf.NamedTemporaryFile() as combined_temp_fasta:
                dist_mtx_maker.make_fasta("../data/raw_data/kinase_seq_822.txt", temp_fasta_known.name)
                dist_mtx_maker.make_fasta(temp_df_in_file.name, temp_fasta_novel.name)
                with open(combined_temp_fasta.name, "w") as combined_fasta:
                    combined_fasta.write(open(temp_fasta_known.name, "r").read() + open(temp_fasta_novel.name, "r").read())
                
                with tf.NamedTemporaryFile() as temp_mtx_out:
                    novel_df = get_needle_pairwise_mtx(
                        combined_fasta.name,
                        temp_mtx_out.name,
                        num_procs=1,
                        restricted_combinations=[train_kin_list, val_kin_list],
                    )
        novel_df.rename(columns={"RET|PTC2|Q15300": "RET/PTC2|Q15300"}, inplace=True)
        grp_pred.MTX = pd.concat([grp_pred.MTX[train_kin_list], novel_df])


def main(run_args):
    train_kins, val_kins, test_kins, train_true, val_true, test_true = grp_pred.get_ML_sets(
        "../tools/pairwise_mtx_822.csv",
        "../data/preprocessing/tr_kins.json",
        "../data/preprocessing/vl_kins.json",
        "../data/preprocessing/te_kins.json",
        "../data/preprocessing/kin_to_fam_to_grp_817.csv",
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
        run_args['device']
    )
    if run_args["c"]:
        check_is_fitted(group_classifier.model)
        pickle.dump(individual_classifiers, open(f"{PRE_TRAINED_NN}", "wb"))
        open(f"{PRE_TRAINED_GC}", "wb").write(pickle.dumps(group_classifier))
        print("Info: Saved pre-trained classifiers to disk with the following paths:")
        print(f"* {PRE_TRAINED_NN}")
        print(f"* {PRE_TRAINED_GC}")
        print("Status: Exiting Successfully.")
        return

    msc = MultiStageClassifier(group_classifier, individual_classifiers)
    msc.evaluate(run_args, run_args["test"])


if __name__ == "__main__":
    run_args = parsing()
    assert (
        run_args["load"] is not None or run_args["load_include_eval"]
    ), "For now, multi-stage classifier must be run with --load or --load-include-eval."
    main(run_args)
