import typing, multiprocessing, warnings, scipy, pandas as pd, json, re, os, pathlib, numpy as np
import sklearn.model_selection, matplotlib, matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor as Pool
from random import seed, shuffle
from typing import Union
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.exceptions import NotFittedError
from pprint import pprint
from itertools import chain, combinations
from termcolor import colored
from itertools import product
from ..config.logging import get_logger

logger = get_logger()


def powerset(iterable):  # From https://stackoverflow.com/a/5228294/16158339 ---
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def my_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)

matplotlib.rcParams["font.family"] = "monospace"

MTX = pd.read_csv("../data/preprocessing/pairwise_mtx_918.csv", index_col=0)

join_first = lambda levels, x: os.path.join(pathlib.Path(__file__).parent.resolve(), *[".."] * levels, x)


class AcceptableClassifier(typing.Protocol):
    def fit(self, X, y) -> typing.Any:
        ...

    def predict(self, X) -> typing.Any:
        ...

    def predict_proba(self, X) -> typing.Any:
        ...

    def score(self, X, y) -> typing.Any:
        ...

    def __init__(self):
        ...


def check_is_fitted(estimator: AcceptableClassifier):
    fitted = [v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")]
    if not fitted:
        raise NotFittedError("Factory AcceptableClassifier not fitted.")


def factory(acceptable_classifier: AcceptableClassifier) -> AcceptableClassifier:
    """
    def fit_(self, X: list[str], y: list[str]):
        self.training_X = X
        self.training_y = y
    def predict_(self, X):
        X_train, X_val = get_coordinates(self.training_X, X)
        KNeighborsClassifier.fit(X_train, self.training_y)
        return KNeighborsClassifier.predict(X_val)
    def predict_proba_(self, X):
        X_train, X_val = get_coordinates(self.training_X, X)
        KNeighborsClassifier.fit(X_train, self.training_y)
        return KNeighborsClassifier.predict_proba(X_val)
    def score_(self, X, y):
        X_train, X_val = get_coordinates(self.training_X, X)
        KNeighborsClassifier.fit(X_train, self.training_y)
        return KNeighborsClassifier.score(X_val, y)
    """

    class NewClass(acceptable_classifier.__class__):
        def fit(self, X, y) -> typing.Any:
            self.training_X = [x for x in X]
            self.training_y = y
            logger.status("Fitting SKlearn classifier in Factory.")
            coords = get_coordinates(self.training_X, self.training_X)[0]
            super().fit(coords, self.training_y)

        def predict(self, X) -> typing.Any:
            self.predict_called = True
            _, X_val = get_coordinates(self.training_X, X)
            return super().predict(X_val)

        def predict_proba(self, X) -> typing.Any:
            if not self.predict_called:
                _, X_val = get_coordinates(self.training_X, X)
            else:
                X_val = X
            check_is_fitted(self)
            return super().predict_proba(X_val)

        def score(self, X, y) -> typing.Any:
            if not self.predict_called:
                _, X_val = get_coordinates(self.training_X, X)
            else:
                X_val = X
            check_is_fitted(self)
            return super().score(X_val, y)

    NewClass.__name__ = acceptable_classifier.__class__.__name__ + "Customized"
    acceptable_classifier.__class__ = NewClass

    return acceptable_classifier


# def perform_hyperparameter_tuning_no_k_fold


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
        gscv.fit(X, y)
    print("Best...")
    print("\tEstimator:", gscv.best_params_["clf"])
    print("\tScore:", gscv.best_score_)
    print("\tParams:", gscv.best_params_)
    return gscv.best_params_["clf"], gscv.best_params_


def get_ML_sets(dist_matrix_file, json_tr, json_vl, json_te, kin_fam_grp_file, kin_seqs=None, verbose=False):
    del_decor = lambda x: re.sub(r"[\(\)\*]", "", x)
    dist_matrix = pd.read_csv(join_first(0, dist_matrix_file), index_col=0)
    symbol_to_grp = pd.read_csv(kin_fam_grp_file)
    symbol_to_grp["Symbol"] = symbol_to_grp["Kinase"].apply(del_decor) + "|" + symbol_to_grp["Uniprot"]
    symbol_to_grp_dict = symbol_to_grp.set_index("Symbol").to_dict()["Group"]
    with open(json_tr, "r") as ftr, open(json_vl, "r") as fvl, open(json_te, "r") as fte:
        train_kins = [del_decor(x) for x in json.load(ftr)]
        val_kins = [del_decor(x) for x in json.load(fvl)]
        test_kins = [del_decor(x) for x in json.load(fte)]

    train_true = [symbol_to_grp_dict[x] for x in dist_matrix.loc[train_kins, train_kins].index]
    val_true = [symbol_to_grp_dict[x] for x in dist_matrix.loc[val_kins, val_kins].index]
    test_true = [symbol_to_grp_dict[x] for x in dist_matrix.loc[test_kins, test_kins].index]
    return train_kins, val_kins, test_kins, train_true, val_true, test_true


def get_coordinates(train_kin_list, val_kin_list) -> tuple[np.ndarray, np.ndarray]:
    train_mtx, eval_mtx = recluster(list(train_kin_list), list(val_kin_list))
    X_train = train_mtx.values
    X_val = eval_mtx.values
    return X_train, X_val


recluster = lambda train_symbols, eval_symbols: (
    MTX.loc[train_symbols, train_symbols],
    MTX.loc[eval_symbols, train_symbols],
)


def grid_search_worker(classifier_type, hp_dict, managed_dict, iall, train_kins, train_true, val_kins, val_true, lock):
    model = factory(classifier_type(**hp_dict))
    model.fit(train_kins, train_true)
    predictions = model.predict(val_kins)
    pass_through_score = [0]
    # print("about to aquire")
    # lock.aquire()
    # print("lock about to release", flush=True)
    # lock.release()
    with lock:
        descstr = str(classifier_type.__name__) + " " + f"({hp_dict}) --- {managed_dict['done']} done/{iall}"
        print(descstr, ">>>", KNNGroupClassifier.test(val_true, predictions, pass_through_score), flush=True)
        if pass_through_score[0] > managed_dict["best_score"]:
            managed_dict["best_score"] = pass_through_score
            managed_dict["best_descstr"] = descstr
        managed_dict["done"] += 1
    # print("end of worker")
    return pass_through_score[0], descstr


def grid_search_no_cv(
    train_kins,
    train_true,
    val_kins,
    val_true,
    classifier_types: list[type],
    hyperparameterses: list[dict[str, list[Union[int, float, str, tuple]]]],
):
    argslist = []
    manager = multiprocessing.Manager()
    managed_dict = {}
    managed_lock = manager.Lock()

    assert len(classifier_types) == len(hyperparameterses), "classifiers and hyperparameterses must be the same length."
    for classifier_type, hyperparameters in list(zip(classifier_types, hyperparameterses)):
        for _, hp_dict in enumerate(list(my_product(hyperparameters))):
            argslist.append(
                [classifier_type, hp_dict, managed_dict, train_kins, train_true, val_kins, val_true, managed_lock]
            )

    managed_dict["best_score"] = 0
    managed_dict["best_descstr"] = "No classifier was more than 0% accurate."
    managed_dict["done"] = 0
    argslist = [a[:3] + [len(argslist)] + a[3:] for a in argslist]
    with Pool(8):
        for a in argslist:
            grid_search_worker(*a)

    print()
    print(f"~~~~~\nThe best score was {managed_dict['best_score']}% achieved by {managed_dict['best_descstr']}.\n~~~~~")


def custom_run():
    kin_seqs = pd.read_csv("../data/raw_data/kinase_seq_826.csv", sep="\t").set_index(["kinase"])

    train_kins, val_kins, _, train_true, val_true, _ = get_ML_sets(
        "../tools/pairwise_mtx_826.csv",
        "json/tr.json",
        "json/vl.json",
        "json/te.json",
        "../data/preprocessing/kin_to_fam_to_grp_817.csv",
        kin_seqs,
    )
    # cctr = collections.Counter(train_true)
    # print(sorted([(x, np.round(cctr[x]/sum([list(cctr.values())[i] for i in range(len(cctr))]), 2)) for x in cctr]))
    # ccvl = collections.Counter(val_true)
    # print(sorted([(x, np.round(ccvl[x]/sum([list(ccvl.values())[i] for i in range(len(ccvl))]), 2)) for x in ccvl]))
    # ccte = collections.Counter(test_true)
    # print(sorted([(x, np.round(ccte[x]/sum([list(ccte.values())[i] for i in range(len(ccte))]), 2)) for x in ccte]))

    train_kins = train_kins  # + val_kins # + test_kins
    eval_kins = val_kins  # val_kins
    train_true = train_true  # + val_true # + test_true
    eval_true = val_true  # test_true

    grid_search_no_cv(
        train_kins,
        train_true,
        eval_kins,
        eval_true,
        [KNeighborsClassifier, MLPClassifier],  # , SVC, RandomForestClassifier],
        [
            {"n_neighbors": [1, 2, 3, 4, 5, 6, 7], "metric": ["chebyshev", "correlation"], "weights": ["uniform"]},
            # {'activation': ['identity'], 'max_iter': [500], 'learning_rate': ['adaptive'], 'hidden_layer_sizes': [x for x in list(set(list(chain(*[list(permutations(x)) for x in powerset([50, 100, 500, 1000])]))) - set([()])) if len(x) >= 3], 'random_state': [0]}
            {
                "activation": ["identity"],
                "max_iter": [500],
                "learning_rate": ["adaptive"],
                "hidden_layer_sizes": [(1000, 500, 100, 50)],
                "random_state": [0, 1, 2],
                "alpha": [1e-7, 1e-4, 1e-2],
            }
            # {"C": [0.1, 1, 10], "kernel": ["linear", "rbf", "poly"], "gamma": ["scale", "auto"]},
            # {"n_estimators": [100, 500], "max_depth": [50, 100, 250], "min_samples_split": [5, 10]}
        ],
    )

    exit(0)

    model = factory(KNeighborsClassifier(n_neighbors=3, metric="cosine"))
    model.fit(train_kins, train_true)
    predictions = model.predict(eval_kins)
    print(KNNGroupClassifier.test(eval_true, predictions))
    # M = KNNGroupClassifier(train_kins, train_true)
    # M.k_fold_evaluation(model, train_kins + eval_kins, train_true + eval_true)


def run_hp_tuning():
    kin_seqs = pd.read_csv("../data/raw_data/kinase_seq_826.csv", sep="\t").set_index(["kinase"])

    train_kins, val_kins, test_kins, train_true, val_true, test_true = get_ML_sets(
        "../tools/pairwise_mtx_826.csv",
        "json/tr.json",
        "json/vl.json",
        "json/te.json",
        "../data/preprocessing/kin_to_fam_to_grp_817.csv",
        kin_seqs,
    )

    all_kins, all_true = np.array(train_kins + val_kins + test_kins), np.array(train_true + val_true + test_true)
    scores = []
    all_kins, all_true = train_kins + val_kins + test_kins, train_true + val_true + test_true
    splitter = sklearn.model_selection.StratifiedKFold(shuffle=False, n_splits=10)  # , random_state=0)
    scores = []
    with warnings.catch_warnings():
        for i, (train, test) in enumerate(splitter.split(all_kins, all_true)):
            seed(i)
            shuffle(train)
            seed(i)
            shuffle(test)
            print(f"------ Fold {i} ----------------------")
            train_kins, train_true = [all_kins[x] for x in train], [all_true[x] for x in train]
            test_kins, test_true = [all_kins[x] for x in test], [all_true[x] for x in test]
            # best_est, _ = perform_hyperparameter_tuning(
            #     train_kins,
            #     train_true,
            #     [factory(KNeighborsClassifier()), factory(RandomForestClassifier())],
            #     [
            #         {
            #             "n_neighbors": list(range(1, 4)),
            #             "metric": ["euclidean", "manhattan", "chebyshev", "minkowski", "cosine"],
            #         },
            #         {"n_estimators": [10, 50], "max_depth": [50, 10]},
            #     ],
            # )
            # model = best_est
            # model.fit(pd.Series(train_kins), pd.Series(train_true))
            # test_pred = model.predict(test_kins)
            # print(f"Test Acc: {sum(test_pred == test_true)/len(test_true):.3f}\n")
            # scores.append(sum(test_pred == test_true) / len(test_true))
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
        splitter = sklearn.model_selection.StratifiedKFold(shuffle=False, n_splits=k)  # , random_state=0)
        scores = []
        with warnings.catch_warnings():
            for i, (train, test) in enumerate(splitter.split(X, y)):
                seed(i)
                shuffle(train)
                seed(i)
                shuffle(test)
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

    @staticmethod
    def test(true, pred, pass_through=[None]):
        assert len(pass_through) == 1, "Length of `pass_through` needs to be 1."
        # print("@@@true", true, len(true))
        # print("\n---------------------------------------------------------------\n")
        # print("@@@pred", pred, len(pred))
        # print()
        # print()
        pass_through[0] = round(100 * np.sum(np.array(pred) == np.array(true)) / len(true), 2)
        return f"{100*np.sum(np.array(pred, dtype=None) == np.array(true, dtype=None))/len(true):3.2f}%"


class SKGroupClassifier:
    def __init__(self, X_train, y_train, classifier: type[AcceptableClassifier], hyperparameters={}):
        self.X_train = X_train
        self.y_train = y_train
        self.model = classifier(**hyperparameters)
        self.model = factory(self.model)
        logger.status("Fitting Group Classifier Model")
        self.model.fit(X_train, y_train)
        check_is_fitted(self.model)

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
        splitter = sklearn.model_selection.StratifiedKFold(shuffle=False, n_splits=k)  # , random_state=0)
        scores = []
        with warnings.catch_warnings():
            for i, (train, test) in enumerate(splitter.split(X, y)):
                seed(i)
                shuffle(train)
                seed(i)
                shuffle(test)
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

    @staticmethod
    def test(true, pred, pass_through=[None]):
        assert len(pass_through) == 1, "Length of `pass_through` needs to be 1."
        # print("@@@true", true, len(true))
        # print("\n---------------------------------------------------------------\n")
        # print("@@@pred", pred, len(pred))
        # print()
        # print()
        pass_through[0] = round(100 * np.sum(np.array(pred) == np.array(true)) / len(true), 2)
        return f"{100*np.sum(np.array(pred) == np.array(true))/len(true):3.2f}%"


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
    custom_run()
    # run_hp_tuning()
