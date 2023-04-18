import collections, math
import cloudpickle as pickle
from matplotlib.rcsetup import cycler
import numpy as np
import sklearn
from .multi_stage_classifier import MultiStageClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
import sklearn.metrics
from scipy.stats import gaussian_kde

with open(
    "/Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/bin/deepks_msc_weights_resaved_fake.cornichon",
    "rb",
) as f:
    data: MultiStageClassifier = pickle.load(f)


cte = list(
    zip(
        data.completed_test_evaluations["ground_truth_groups"],  # type: ignore
        data.completed_test_evaluations["scores"],  # type: ignore
        data.completed_test_evaluations["ground_truth_labels"],  # type: ignore
    )
)


grp_to_scores = collections.defaultdict(list)
grp_to_gt = collections.defaultdict(list)
for g, s, gt in cte:
    grp_to_scores[g].append(s)
    grp_to_gt[g].append(gt)
# get viridis color for each group
colss = ["orange", "blue"]
cols = lambda i: plt.get_cmap(["Oranges", "Blues"][i])  # plt.get_cmap('viridis', len(grp_to_scores))  # type: ignore
# hatchs = ['///', '\\\\\\', '|||', '---', 'xx', 'oo', '**']

all_labs = []
all_old_scores = []
all_transformed_scores = []

for i, grp in enumerate(["OTHER", "CMGC"]):
    # Calculate the point density

    # lr = GaussianMixture()
    lr = LogisticRegression(penalty=None)
    lr.fit(np.asarray(grp_to_scores[grp]).reshape(-1, 1), grp_to_gt[grp])
    Xo = grp_to_scores[grp]
    Yo = [y - i / 10 for y in grp_to_gt[grp]]
    xy = np.vstack([Xo, Yo])
    z = gaussian_kde(xy)(xy)
    plt.scatter(Xo, Yo, c=z, cmap=cols(i), s=2)

    # print(grp_to_scores[grp], grp_to_gt[grp])
    # lr_eqn = lambda x: 1/(1+math.e**(lr.intercept_ + lr.coef_[0] * x))
    X = np.linspace(0, 1, 1000)
    plt.plot(X, lr.predict_proba(X.reshape(-1, 1))[:, -1] - i / 10, "-", color=colss[i], alpha=0.5, label=grp)
    # plt.hist(grp_to_scores[grp], bins=np.arange(0, 1.01, 0.005), color=cols(i), alpha=0.25, label = grp)
    # plt.xlim(0.25, 0.35)
    plt.xlim(0, 1)
    # print(sklearn.metrics.roc_auc_score(grp_to_gt[grp], grp_to_scores[grp]))
    # print(sklearn.metrics.roc_auc_score(grp_to_gt[grp], np.asarray(grp_to_scores[grp]).reshape(-1, 1)))
    # plt.xlim(min(grp_to_scores[grp]), max(grp_to_scores[grp]))
    all_labs += grp_to_gt[grp]
    all_old_scores += grp_to_scores[grp]
    all_transformed_scores += lr.predict_proba(np.asarray(grp_to_scores[grp]).reshape(-1, 1))[:, -1].tolist()

print(sklearn.metrics.roc_auc_score(all_labs, all_old_scores))
print(sklearn.metrics.roc_auc_score(all_labs, all_transformed_scores))

plt.legend()
plt.show()
