if __name__ == "__main__":
    from .write_splash import write_splash

    write_splash()
    print("Progress: Loading Modules", flush=True)
import collections, random
import pandas as pd, numpy as np
from .individual_classifiers import IndividualClassifiers
from . import group_prediction_from_hc as grp_pred
from . import individual_classifiers
from ..tools.parse import parsing
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

    def evaluate(
        self,
        newargs,
        Xy_formatted_input_file: str,
        evaluation_kwargs=None,
    ) -> None:
        print("Progress: Sending input kinases to group classifier")
        # Open XY_formatted_input_file
        input_file = pd.read_csv(Xy_formatted_input_file)
        _, _, true_symbol_to_grp_dict = individual_classifiers.IndividualClassifiers.get_symbol_to_grp_dict(
            "../data/preprocessing/kin_to_fam_to_grp_817.csv"
        )

        ### Get group predictions
        unq_symbols = input_file["orig_lab_name"].unique()
        groups_true = [true_symbol_to_grp_dict[u] for u in unq_symbols]

        ### Perform Real Accuracy
        groups_prediction = self.group_classifier.predict(unq_symbols)

        ### Perform Simulated 100% Accuracy
        # sim_ac = 1
        # wrong_inds = set()
        # if sim_ac != 1:
        #     wrong_inds = set([round(i/(1-sim_ac), 0) for i in range(int(len(groups_true)//(1/sim_ac)))])
        # print("Simulated accuracy:", sim_ac)
        # random.seed(0)
        # groups_prediction = [groups_true[i] if (sim_ac == 1 or i not in wrong_inds) else random.choice(list(set(groups_true) - set([groups_true[i]]))) for i in range(len(groups_true))]

        pred_grp_dict = {symb: grp for symb, grp in zip(unq_symbols, groups_prediction)}
        true_grp_dict = {symb: grp for symb, grp in zip(unq_symbols, groups_true)}

        # Report performance
        print(f"Info: Group Classifier Accuracy — {self.group_classifier.test(groups_true, groups_prediction)}")

        # Send to individual classifiers
        print("Progress: Sending input kinases to individual classifiers (with group predictions)")
        self.individual_classifiers.evaluations = {}
        self.individual_classifiers.roc_evaluation(newargs, pred_grp_dict, true_grp_dict)

    def predict(self, X: np.ndarray, verbose=False):
        raise RuntimeError("Not implemented yet.")
        predicted_groups = self.group_classifier.predict(X)
        if verbose:
            print(f"Predicted groups: {predicted_groups}")
        classifier_to_list_of_inds = collections.defaultdict(list)
        for i in range(len(X)):
            classifier_to_list_of_inds[predicted_groups[i]].append(i)

        predictions = [-1 for _ in range(len(X))]
        for indiv_classif in classifier_to_list_of_inds:
            predictions = self.individual_classifiers.predict(X[classifier_to_list_of_inds[indiv_classif]])

        return predictions


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
