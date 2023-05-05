import collections
import numpy as np
import typing, json
import pandas as pd
from matplotlib import pyplot as plt
import sklearn.metrics

np.set_printoptions(linewidth=1000, edgeitems=25)


def exhaustive_pairwise_knn(
    known_true_classes: np.ndarray,
    queries: np.ndarray,
    k: int = 5,
    tie_policy: typing.Literal["random", "alpha", "raise"] = "random",
    best_score_is_higher: bool = True,
):
    assert queries.shape[1] == len(known_true_classes), "Classes and queries must have same number of columns."
    assert len(queries.shape) == 2, "Query must be 2D."
    assert len(known_true_classes.shape) == 1, "Classes must be 1D."
    assert queries.shape[0] >= 1, "Query must have at least one row."

    res = []
    for query in queries:
        if best_score_is_higher:
            mult = -1
        else:
            mult = 1
        top_k_order = np.argsort(query * mult)[:k]
        top_k_classes = known_true_classes[top_k_order]
        counts = collections.Counter(top_k_classes)
        count_items = sorted(list(counts.items()), key=lambda x: -x[1])
        first_appearance = count_items[0][1]
        i = 0
        how_many_appearances = 0
        while i < len(count_items) and count_items[i][1] == first_appearance:
            how_many_appearances += 1
            i += 1
        if tie_policy == "random":
            res.append(count_items[np.random.randint(how_many_appearances)][0])
        if tie_policy == "alpha":
            count_items = sorted([x for x in counts.items() if x[1] == first_appearance], key=lambda x: x[0])
            res.append(count_items[0][0])
        if tie_policy == "raise":
            assert how_many_appearances == 1, "Tie detected for best distance, but tie_policy is set to raise."
    return np.array(res)


if __name__ == "__main__":
    with open("../data/preprocessing/tr_kins.json") as trf, open("../data/preprocessing/vl_kins.json") as vlf, open(
        "../data/preprocessing/te_kins.json"
    ) as tef:
        pseudo_training_symbols = [
            x.replace("(", "").replace(")", "").replace("*", "") for x in json.load(trf) + json.load(vlf)
        ]
        pseudo_testing_symbols = [x.replace("(", "").replace(")", "").replace("*", "") for x in json.load(tef)]

    pairwise_mtx = pd.read_csv("../data/preprocessing/pairwise_mtx_822.csv", index_col=0)

    class_df = pd.read_csv("../data/kin_to_fam_to_grp_821.csv")
    symbols = class_df["Kinase"] + "|" + class_df["Uniprot"]
    class_df["Symbol"] = [x.replace("(", "").replace(")", "").replace("*", "") for x in symbols]
    class_df.set_index("Symbol", inplace=True)
    class_df_all = class_df.copy()
    class_df = class_df.loc[pseudo_training_symbols]
    pairwise_mtx_train_block = pairwise_mtx[class_df.index].loc[class_df.index].copy()

    pairwise_mtx_train_block.columns = pd.MultiIndex.from_arrays([class_df.index, class_df["Group"]])
    queries_test_block = pairwise_mtx[pseudo_training_symbols].loc[pseudo_testing_symbols].copy()
    # print(pairwise_mtx_train_block)
    # print(queries)
    predictions = exhaustive_pairwise_knn(
        np.array(class_df["Group"].tolist()), queries_test_block.values, k=15, tie_policy="random"
    )
    truths = class_df_all.loc[pseudo_testing_symbols]["Group"].values
    print("Accuracy:", np.sum(predictions == truths) / len(predictions))
    cm = sklearn.metrics.confusion_matrix(truths, predictions)
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(5, 5))
    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(truths, predictions, ax=ax, normalize="true")
    locs, labs = ax.get_xticks(), ax.get_xticklabels()
    ax.set_xticks(locs, labs, rotation=90)
    plt.show()
    pass
