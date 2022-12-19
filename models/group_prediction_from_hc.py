import functools, warnings, scipy, pandas as pd, json, re, os, pathlib, numpy as np, sklearn.model_selection, matplotlib
from random import seed, shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from typing import Callable, ClassVar, TypeVar, Union, Protocol
import typing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt
from pprint import pprint  # type: ignore
from sklearn.utils.validation import check_is_fitted

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)

matplotlib.rcParams["font.family"] = "Palatino"

KIN_SEQS = pd.read_csv("../data/raw_data/kinase_seq_822.txt", sep="\t").set_index("kinase")
MTX = pd.read_csv("../tools/pairwise_mtx_822.csv", index_col=0)

class AcceptableClassifier(typing.Protocol):
    def fit(self, X, y, *args) -> typing.Any:
        ...
    def predict(self, X, *args) -> typing.Any:
        ...
    def predict_proba(self, X, *args) -> typing.Any:
        ...
    def score(self, X, y, *args) -> typing.Any:
        ...
    def __init__(self, *args):
        ...

def factory(acceptable_classifier: AcceptableClassifier) -> AcceptableClassifier:
    class NewClass(acceptable_classifier.__class__):
        def fit(self, X, y, *args) -> typing.Any:
            print("Yup. You're using the right method.")
            return super().fit(X, y)
        def predict(self, X, *args) -> typing.Any:
            return super().predict(X)
        def predict_proba(self, X, *args) -> typing.Any:
            return super().predict_proba(X)
        def score(self, X, y, *args) -> typing.Any:
            return super().score(X, y)
    
    NewClass.__name__ = acceptable_classifier.__class__.__name__ + "New"
    acceptable_classifier.__class__ = NewClass
    
    return acceptable_classifier

def perform_hyperparameter_tuning(
    X: list,
    y: list,
    classifiers: list,
    hyperparameterses: list[dict[str, list[Union[int, float, str]]]],
):
    assert len(X) == len(y), "X_train and y_train must be the same length."
    assert len(classifiers) == len(hyperparameterses), "classifiers and hyperparameterses must be the same length."

    pipe = Pipeline(steps=[("clf", None)])
    params_grid = [
        {"clf": [classifier]} | {"clf" + "__" + hp: hpv for hp, hpv in hyperparameters.items()}
        for classifier, hyperparameters in zip(classifiers, hyperparameterses)
    ]

    split_inds = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=False)
    gscv = GridSearchCV(pipe, params_grid, n_jobs=1, verbose=0, scoring="accuracy", cv=split_inds)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The least populated class in y has only")
        gscv.fit(X, y)
    print("Best...")
    print("\tEstimator:", gscv.best_params_["clf"])
    print("\tScore:", gscv.best_score_)
    print("\tParams:", gscv.best_params_)
    return gscv.best_params_["clf"], gscv.best_params_


def get_ML_sets(dist_matrix_file, json_tr, json_vl, json_te, kin_fam_grp_file, kin_seqs, verbose=False):
    del_decor = lambda x: re.sub(r"[\(\)\*]", "", x)
    dist_matrix = pd.read_csv(dist_matrix_file, index_col=0)
    symbol_to_grp = pd.read_csv(kin_fam_grp_file)
    symbol_to_grp["Symbol"] = symbol_to_grp["Kinase"].apply(del_decor) + "|" + symbol_to_grp["Uniprot"]
    symbol_to_grp_dict = symbol_to_grp.set_index("Symbol").to_dict()["Group"]
    train_kins = [del_decor(x) for x in json.load(open(json_tr, "r"))]
    val_kins = [del_decor(x) for x in json.load(open(json_vl, "r"))]
    test_kins = [del_decor(x) for x in json.load(open(json_te, "r"))]

    train_true = [symbol_to_grp_dict[x] for x in dist_matrix.loc[train_kins, train_kins].index]
    val_true = [symbol_to_grp_dict[x] for x in dist_matrix.loc[val_kins, val_kins].index]
    test_true = [symbol_to_grp_dict[x] for x in dist_matrix.loc[test_kins, test_kins].index]
    return train_kins, val_kins, test_kins, train_true, val_true, test_true


def get_coordinates(train_kin_list, val_kin_list):
    train_mtx, eval_mtx = recluster(train_kin_list, val_kin_list)
    X_train = train_mtx.values
    X_val = eval_mtx.values
    return X_train, X_val


recluster = lambda train_symbols, eval_symbols: (
    MTX.loc[train_symbols, train_symbols],
    MTX.loc[eval_symbols, train_symbols],
)


def run_hp_tuning():
    kin_seqs = pd.read_csv("../data/raw_data/kinase_seq_822.txt", sep="\t").set_index(["kinase"])

    train_kins, val_kins, test_kins, train_true, val_true, test_true = get_ML_sets(
        "../tools/pairwise_mtx_822.csv",
        "../data/preprocessing/tr_kins.json",
        "../data/preprocessing/vl_kins.json",
        "../data/preprocessing/te_kins.json",
        "../data/preprocessing/kin_to_fam_to_grp_817.csv",
        kin_seqs,
    )
    
    all_kins, all_true = np.array(train_kins + val_kins + test_kins), np.array(train_true + val_true + test_true)
    scores = []
    all_kins, all_true = train_kins + val_kins + test_kins, train_true + val_true + test_true
    splitter = sklearn.model_selection.StratifiedKFold(shuffle=True, n_splits=10, random_state=0)
    scores = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The least populated class in y has only")
        for i, (train, test) in enumerate(splitter.split(all_kins, all_true)):
            seed(i)
            shuffle(train)
            seed(i)
            shuffle(test)
            print(test)
            print(f"------ Fold {i} ----------------------")
            train_kins, train_true = [all_kins[x] for x in train], [all_true[x] for x in train]
            test_kins, test_true = [all_kins[x] for x in test], [all_true[x] for x in test]
            best_est, _ = perform_hyperparameter_tuning(
                train_kins,
                train_true,
                [factory(KNeighborsClassifier()), factory(RandomForestClassifier())],
                [
                    {
                        "n_neighbors": list(range(1, 4)),
                        "metric": ["euclidean", "manhattan", "chebyshev", "minkowski", "cosine"],
                    },
                    {"n_estimators": [10, 50], "max_depth": [50, 10]},
                ],
            )
            model = best_est
            model.fit(pd.Series(train_kins), pd.Series(train_true))
            test_pred = model.predict(test_kins)
            print(f"Test Acc: {sum(test_pred == test_true)/len(test_true):.3f}\n")
            scores.append(sum(test_pred == test_true) / len(test_true))
        if len(scores) < 1:
            raise ValueError("No scores were calculated")
        print(f"Mean Acc: {100*np.mean(scores):.3f} +/-95%CI {(confIntMean(scores)[1] - np.mean(scores))*100:.4f}")


class KNNGroupClassifier:
    def __init__(self, X_train, y_train, n_neighbors=1, metric="euclidean"):
        self.X_train = X_train
        self.y_train = y_train
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        self.model = factory(self.model)
        self.model.fit(X_train, y_train)

    def predict(self, X_test) -> list[str]:
        return self.model.predict(X_test).tolist()

    def make_roc(self, X_test, y_test, fig_kwargs={"figsize": (10, 10)}, filename=None):
        plt.figure(**fig_kwargs)
        y_pred = self.model.predict(X_test)
        for pos_class in list(set(y_test)):
            fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=pos_class)
            plt.plot(fpr, tpr, label=f"Class {pos_class}")
        plt.plot([0, 1], [0, 1], color="red", lw=0.5, linestyle="--", label="Random Model")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()

    def make_confusion_matrix(self, X_test, y_test, fig_kwargs={"figsize": (10, 10)}, filename=None):
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.xticks(plt.xticks()[0], plt.xticks()[1], rotation=45, ha="right", va="top")  # type: ignore
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()

    @staticmethod
    def k_fold_evaluation(model, X, y, k=5):
        splitter = sklearn.model_selection.StratifiedKFold(shuffle=True, n_splits=k, random_state=0)
        scores = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "The least populated class in y has only")
            for i, (train, test) in enumerate(splitter.split(X, y)):
                seed(i)
                shuffle(train)
                seed(i)
                shuffle(test)
                print(test)
                print(f"------ Fold {i} ----------------------")
                train_kins, train_true = [X[x] for x in train], [y[x] for x in train]
                test_kins, test_true = [X[x] for x in test], [y[x] for x in test]
                model.fit(pd.Series(train_kins), pd.Series(train_true))
                test_pred = model.predict(test_kins)
                print(f"Test Acc: {sum(test_pred == test_true)/len(test_true):.3f}\n")
                scores.append(sum(test_pred == test_true) / len(test_true))
            if len(scores) < 1:
                raise ValueError("No scores were calculated")
            print(f"Mean Acc: {100*np.mean(scores):.3f} +/-95%CI {(confIntMean(scores)[1] - np.mean(scores))*100:.4f}")


def confIntMean(a, conf=0.95):
    """Get the confidence interval of the mean of a list of numbers.

    Args:
        a (ArrayLike): list of numbers
        conf (float, optional): 1 - alpha value. Defaults to 0.95.

    Returns:
        tuple(float, float): The confidence interval

    Notes:
        code from https://stackoverflow.com/a/33374673.
    """
    mean, sem, m = np.mean(a), scipy.stats.sem(a), scipy.stats.t.ppf((1 + conf) / 2.0, len(a) - 1)
    return mean - m * sem, mean + m * sem


if __name__ == "__main__":

    # single_run()
    run_hp_tuning()
