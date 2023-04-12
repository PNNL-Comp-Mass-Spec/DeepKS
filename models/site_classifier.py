import re
from matplotlib import pyplot as plt
import numpy as np
from sklearn import neighbors, svm, ensemble, neural_network, naive_bayes
import sklearn.metrics
import os, pandas as pd
import sklearn
from termcolor import colored

num_aa = 21
seq_len = 15
aas = list("ACDEFGHIKLMNPQRSTVWYX")


class SiteClassifier:
    @staticmethod
    def one_hot_aa_embedding(seq):
        assert len(seq) == seq_len
        patch = np.zeros((len(seq), num_aa))
        aa_dict = {aa: i for i, aa in enumerate(aas)}
        for i, aa in enumerate(seq):
            patch[i, aa_dict[aa]] = 1
        embedding = patch.ravel()
        return embedding

    @staticmethod
    def format_data(*fns, kin_fam_grp, sample_size=None, grp_to_cls={}):
        X_y_tuples: list[tuple[np.ndarray, np.ndarray]] = []
        kin_to_grp = pd.read_csv(kin_fam_grp)[["Kinase", "Uniprot", "Group"]]
        kin_to_grp["Symbol"] = (
            kin_to_grp["Kinase"].apply(lambda x: re.sub(r"[\(\)\*]", "", x)) + "|" + kin_to_grp["Uniprot"]
        )
        kin_to_grp = kin_to_grp.set_index("Symbol").to_dict()["Group"]
        grp_to_cls.update({grp: i for i, grp in enumerate(sorted(list(set(kin_to_grp.values()))))})
        for fn in fns:
            dff = pd.read_csv(fn)[["Site Sequence", "Gene Name of Kin Corring to Provided Sub Seq"]]
            df = dff.values
            X = []
            y = [grp_to_cls[kin_to_grp[y]] for y in df[:, 1]]
            for seq in df[:, 0]:
                X.append(SiteClassifier.one_hot_aa_embedding(seq))
            Xnd: np.ndarray = np.asarray(X).astype(int)
            ynd: np.ndarray = np.asarray(y).astype(int)
            np.random.seed(42)
            if isinstance(sample_size, int):
                shuffled_ids = np.random.permutation(len(y))[:sample_size]
            else:
                shuffled_ids = np.random.permutation(len(y))
            Xnd = Xnd[shuffled_ids]
            ynd = ynd[shuffled_ids]
            X_y_tuples.append((Xnd, ynd))
        return tuple(X_y_tuples)

    @staticmethod
    def get_group_classifier(
        train_file,
        kin_fam_grp_file,
        grp_to_cls={},
        gc_hyperparameters={
            "base_classifier": "ensemble.RandomForestClassifier",
            "n_estimators": 10000,
            "max_depth": 3,
        },
    ):
        cpu_count_raw = os.cpu_count()
        cpu_count = max(1, cpu_count_raw // 2) if cpu_count_raw is not None else 1
        exec(f"import sklearn.{gc_hyperparameters['base_classifier'].split('.')[0]}")
        hps = {k: v for k, v in gc_hyperparameters.items() if k != "base_classifier"}
        gc = eval(gc_hyperparameters["base_classifier"])(**hps, random_state=42, n_jobs=cpu_count)
        (tr,) = SiteClassifier.format_data(train_file, kin_fam_grp=kin_fam_grp_file, grp_to_cls=grp_to_cls)
        print(colored("Status: Training Group Classifier", "green"))
        gc.fit(tr[0], tr[1])
        return gc

    @staticmethod
    def performance(model, X, y, metric_fn=sklearn.metrics.accuracy_score):
        y_true, y_pred = y, model.predict(X)
        return round(metric_fn(y_true, y_pred), 4)


if __name__ == "__main__":
    fd = (
        "data/raw_data_31834_formatted_65_26610.csv",
        "data/raw_data_6500_formatted_95_5698.csv",
        "data/raw_data_6406_formatted_95_5616.csv",
    )
    kfg = "data/preprocessing/kin_to_fam_to_grp_826.csv"

    tr, vl, te = SiteClassifier.format_data(*fd, kin_fam_grp=kfg, grp_to_cls=(grp_to_cls := {}))
    clf = SiteClassifier.get_group_classifier(fd[0], kfg)

    print("Train Acc: ", sklearn.metrics.accuracy_score(tr[1], clf.predict(tr[0])))
    print("Val Acc: ", sklearn.metrics.accuracy_score(vl[1], clf.predict(vl[0])))
    print("Test Acc: ", sklearn.metrics.accuracy_score(te[1], clf.predict(te[0])))
    cm = sklearn.metrics.confusion_matrix(te[1], clf.predict(te[0]), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 7))
    sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=[x for x in list(grp_to_cls.keys())]).plot(ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()
