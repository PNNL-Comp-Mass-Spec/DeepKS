import collections
import functools, warnings, scipy, pandas as pd, json, re, os, pathlib, numpy as np, sklearn.model_selection, matplotlib
from . import group_prediction_from_hc as grp_pred
from . import individual_classifiers


class MultiStageClassifier:
    def __init__(
        self,
        group_classifier: grp_pred.KNNGroupClassifier,
        individual_classifiers: dict[str, individual_classifiers.IndividualClassifiers],
    ):
        self.group_classifier = group_classifier
        self.individual_classifiers = individual_classifiers

    def __str__(self):
        return "MultiStageClassifier(group_classifier=%s, individual_classifiers=%s)" % (
            self.group_classifier,
            self.individual_classifiers,
        )

    def evaluate(self, X, y):
        pass

    def predict(self, X: np.ndarray, verbose=False):
        predicted_groups = self.group_classifier.predict(X)
        if verbose:
            print(f"Predicted groups: {predicted_groups}")
        classifier_to_list_of_inds = collections.defaultdict(list)
        for i in range(len(X)):
            classifier_to_list_of_inds[predicted_groups[i]].append(i)
        
        predictions = [-1 for _ in range(len(X))]
        for indiv_classif in classifier_to_list_of_inds:
            predictions = self.individual_classifiers[indiv_classif].predict(X[classifier_to_list_of_inds[indiv_classif]])

        return predictions
