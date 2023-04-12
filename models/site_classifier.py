import re
from matplotlib import pyplot as plt
import numpy as np
from sklearn import neighbors, svm, ensemble, neural_network, naive_bayes
import sklearn.metrics
import os, pandas as pd
import sklearn

os.chdir("DeepKS")

num_aa = 21
seq_len = 15
aas = list("ACDEFGHIKLMNPQRSTVWYX")


def one_hot_aa_embedding(seq):
    assert len(seq) == seq_len
    patch = np.zeros((len(seq), num_aa))
    aa_dict = {aa: i for i, aa in enumerate(aas)}
    for i, aa in enumerate(seq):
        patch[i, aa_dict[aa]] = 1
    embedding = patch.ravel()
    return embedding


def format_data(tr, vl, te, kin_fam_grp, sample_size=1000, grp_to_cls={}):
    fns = [tr, vl, te]
    X_y_tuples = []
    kin_to_grp = pd.read_csv(kin_fam_grp)[["Kinase", "Uniprot", "Group"]]
    kin_to_grp["Symbol"] = (
        kin_to_grp["Kinase"].apply(lambda x: re.sub(r"[\(\)\*]", "", x)) + "|" + kin_to_grp["Uniprot"]
    )
    kin_to_grp = kin_to_grp.set_index("Symbol").to_dict()["Group"]
    # kin_to_grp = {k:v if v not in {'CMGC', 'AGC'} else '(CM|A)GC' for k, v in kin_to_grp.items()}
    grp_to_cls.update({grp: i for i, grp in enumerate(sorted(list(set(kin_to_grp.values()))))})
    for fn in fns:
        dff = pd.read_csv(fn)[["Site Sequence", "Gene Name of Kin Corring to Provided Sub Seq"]]
        df = dff.values
        X = []
        y = [grp_to_cls[kin_to_grp[y]] for y in df[:, 1]]
        for seq in df[:, 0]:
            X.append(one_hot_aa_embedding(seq))
        X = np.asarray(X).astype(int)
        y = np.asarray(y).astype(int)
        shuffled_ids = np.random.permutation(len(y))[:sample_size]
        X = X[shuffled_ids]
        y = y[shuffled_ids]
        X_y_tuples.append((X, y))
    return X_y_tuples


def classifier():
    # return neural_network.MLPClassifier((100, 100, 100), activation='relu')
    # return neighbors.KNeighborsClassifier(11, metric='euclidean', weights='distance')
    # naive bayes classifier below
    # return naive_bayes.GaussianNB()
    return ensemble.RandomForestClassifier(random_state=42, n_estimators=10000, max_depth=3)


def fto(vec):
    # return [x if x != 5 else 1 for x in vec]
    return vec


if __name__ == "__main__":
    fd = (
        "data/raw_data_31834_formatted_65_26610.csv",
        "data/raw_data_6500_formatted_95_5698.csv",
        "data/raw_data_6406_formatted_95_5616.csv",
        "data/preprocessing/kin_to_fam_to_grp_826.csv",
    )
    tr, vl, te = format_data(*fd, grp_to_cls=(grp_to_cls := {}))
    clf = classifier()
    print("Fitting")
    clf.fit(tr[0], tr[1])
    print("Train Acc: ", sklearn.metrics.accuracy_score(fto(tr[1]), fto(clf.predict(tr[0]))))
    print("Val Acc: ", sklearn.metrics.accuracy_score(fto(vl[1]), fto(clf.predict(vl[0]))))
    print("Test Acc: ", sklearn.metrics.accuracy_score(fto(te[1]), fto(clf.predict(te[0]))))
    cm = sklearn.metrics.confusion_matrix(fto(te[1]), fto(clf.predict(te[0])), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 7))
    sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=[x for x in list(grp_to_cls.keys())]).plot(ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()
