from ast import main
import collections
import pandas as pd, numpy as np
from models.individual_classifiers import IndividualClassifiers
from . import group_prediction_from_hc as grp_pred
from . import individual_classifiers


class MultiStageClassifier:
    def __init__(
        self,
        group_classifier: grp_pred.KNNGroupClassifier,
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
        Xy_formatted_input_file: str,
        evaluation_kwargs=None,
    ) -> None:
        print("Progress: Sending input kinases to group classifier")
        # Open XY_formatted_input_file
        input_file = pd.read_csv(Xy_formatted_input_file)
        _, _, true_symbol_to_grp_dict = individual_classifiers.IndividualClassifiers.get_symbol_to_grp_dict(
            "../data/preprocessing/kin_to_fam_to_grp_817.csv"
        )
        _, _, true_groups = [true_symbol_to_grp_dict[x["Kinase"]] for _, x in input_file.iterrows()]

        # Get group predictions
        group_predictions = self.group_classifier.predict(input_file["seq"])

        # Report performance
        print(f"Info: Group Classifier Accuracy — {self.group_classifier.test(true_groups, group_predictions)}")

        # Send to individual classifiers
        print("Progress: Sending input kinases to individual classifiers (with group predictions)")
        self.individual_classifiers.evaluate(
            list(set(group_predictions)),
            Xy_formatted_input_file,
            evaluation_kwargs=({} if evaluation_kwargs is None else evaluation_kwargs),
            pred_groups={symb: grp for symb, grp in zip(group_predictions, input_file["seq"])}
        )

        # Run test and ROC curves

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


def main():

    train_kins, val_kins, test_kins, train_true, val_true, test_true = grp_pred.get_ML_sets(
        "../tools/pairwise_mtx_822.csv",
        "../data/preprocessing/tr_kins.json",
        "../data/preprocessing/vl_kins.json",
        "../data/preprocessing/te_kins.json",
        "../data/preprocessing/kin_to_fam_to_grp_817.csv",
    )

    train_kins, eval_kins, train_true, eval_true = ( # type: ignore
        np.array(train_kins + val_kins),
        np.array(test_kins),
        np.array(train_true + val_true),
        np.array(test_true),
    )

    group_classifier = grp_pred.KNNGroupClassifier(X_train=train_kins, y_train=train_true)

    individual_classifiers = IndividualClassifiers.load_all(
        "../bin/indivudial_classifiers_2022-12-20T17:24:06.760770.pkl"
    )

    msc = MultiStageClassifier(group_classifier, individual_classifiers)
    # tr = "../data/raw_data_31834_formatted_65_26610.csv"
    # vl = "../data/raw_data_6500_formatted_65_5698.csv"
    te = "../data/raw_data_6406_formatted_65_5616.csv"
    msc.evaluate(te)


if __name__ == "__main__":
    splash_screen = open("../tools/splash_screen.txt", "r").read()
    print(splash_screen)
    print()
    main()
