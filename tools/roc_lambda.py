import itertools
import matplotlib
import random
import numpy as np
from matplotlib import pyplot as plt


def get_avg_roc(fprs, tprs, aucs = None, plot = True):
    total_n = len(list(itertools.chain(*fprs))) + 1
    X = np.linspace(0, 1, num=total_n, endpoint=False).tolist()
    X = [X[i // 2] for i in range(len(X) * 2)]
    Y = [get_tpr(fprs, tprs, x) for x in X]
    micro_avg = 0
    if aucs is not None:
        for i, a in enumerate(aucs):
            micro_avg += a * len(fprs[i])/total_n
    if plot:
        plt.gca().plot(X, Y[1:] + [1], color="black", linewidth=3, **({} if aucs is None else {'label': f"{'Average Value':>13} ┆ ROC Micro Average  {micro_avg:3.3f}"}))

    return X, Y


def get_tpr(fprs, tprs, fpr_x, weighted=True):
    assert len(fprs) == len(tprs)
    total_n = len(list(itertools.chain(*fprs)))
    running_sum = 0
    for i in range(len(fprs)):
        rel_weight = len(fprs[i]) / total_n if weighted else 1 / len(fprs)
        tpr = -1
        for j in range(len(fprs[i])):
            if fpr_x == 0:
                tpr = 0
                break
            if fprs[i][j] < fpr_x:
                continue
            if fprs[i][j] == fpr_x:
                above = tprs[i][j + 1]
                below = tprs[i][j]
                tpr = (above + below) / 2
                break
            if fprs[i][j] > fpr_x:
                if not np.allclose(
                    [tprs[i][j]], [tprs[i][j - 1]]
                ):  # and not np.allclose([fprs[i][j]], [fprs[i][j - 1]]):
                    tpr = tprs[i][j - 1] + (tprs[i][j] - tprs[i][j - 1]) / (fprs[i][j] - fprs[i][j - 1]) * (
                        fpr_x - fprs[i][j - 1]
                    )
                else:
                    tpr = tprs[i][j]
                break
        assert tpr != -1
        running_sum += tpr * rel_weight
    assert 0 <= running_sum <= 1, "Avg. ROC cannot be lower than 0 or higher than 1."
    return running_sum


if __name__ == "__main__":
    random.seed(0)

    fprs = []
    tprs = []

    for i in range(8):
        h = random.choice([1.5, 8, 10, 15, 50, 100])
        size = int(random.random() * 50) + 1
        X = np.linspace(0, 1, size, endpoint=True)
        fpr = [X[i // 2] for i in range(len(X) * 2)][1:]
        tpr = [0] + [(1 - abs(x - 1) ** h) ** 0.9 for x in fpr][:-1]
        if i == 7:
            fpr = np.linspace(0, 1, 11, endpoint=True).tolist()
            fpr = [fpr[i // 2] for i in range(2 * len(fpr))][1:]
            tpr = [
                0,
                0,
                0.2,
                0.2,
                0.25,
                0.55,
                0.6,
                0.6,
                0.9,
                0.9,
                0.93,
                0.95,
                0.95,
                0.95,
                0.95,
                0.96,
                0.98,
                0.98,
                0.99,
                0.99,
                1,
            ]
            print(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=1 if i != 7 else 2, markersize=2 if i == 7 else 0, marker="o")
        fprs.append(fpr)
        tprs.append(tpr)

    X, Y = get_avg_roc(fprs, tprs)
