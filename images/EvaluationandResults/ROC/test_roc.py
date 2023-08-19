# %%
import torch
import random
import sys

sys.path.append("../../../tools")
sys.path.append("../../../data/preprocessing/")
from ....tools.NNInterface import NNInterface as NNI


# %%
class RandomModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = torch.nn.Linear(input_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


# %%
tl = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.rand(100, 100).float(), torch.randint(0, 2, (100, 1))), batch_size=10
)
vl = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.rand(100, 100).float(), torch.randint(0, 2, (100, 1))), batch_size=100
)
te = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.rand(100, 100).float(), torch.randint(0, 2, (100, 1))), batch_size=100
)
ho = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.rand(15, 100).float(), torch.randint(0, 2, (15, 1))), batch_size=15
)

# %%
model = RandomModel(100, 1)

# %%
nni = NNI(
    model,
    torch.nn.BCEWithLogitsLoss(),
    torch.optim.Adam(model.parameters()),
    inp_size=NNI.get_input_size(tl),
    inp_types=NNI.get_input_types(tl),
)

# %%
_ = nni.train(tl, 5, val_dl=vl)

# %%
_ = nni.test(te, verbose=True, cutoff=0.59)

# %%
import os

os.chdir("/Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS/images/ROC")

# %%
nni.get_all_rocs(tl, vl, te, ho, savefile="./ROCC")

# %%
n = 100
fake_probs = [x / n + random.random() * random.choice([-1, 1]) * 0.5 for x in range(n)]
mi = min(fake_probs)
ma = max(fake_probs)
fake_probs = [(x - mi) / ma for x in fake_probs]
fake_labels = [0] * (n // 2) + [1] * (n // 2)

# %%
nni.get_all_conf_mats(tl, vl, te, ho, "./All_CM_initial", cutoffs=[0.3, 0.4, 0.5, 0.6])

# %%
import sklearn.metrics

# %%
fpr, tpr, _ = sklearn.metrics.roc_curve(fake_labels, fake_probs)
sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

# %%
sklearn.metrics.roc_auc_score(fake_labels, fake_probs)

# %%


# %%
fake_preds = []
for cutoff in (co := [0.2, 0.4, 0.5, 0.6, 0.8]):
    fake_preds.append([1 if x > cutoff else 0 for x in fake_probs])

# %%
from matplotlib import pyplot as plt

for i, fp in enumerate(fake_preds):
    cm = sklearn.metrics.confusion_matrix(fake_labels, fp)
    sklearn.metrics.ConfusionMatrixDisplay(cm).plot()
    plt.title(f"Cutoff = {co[i]}| Acc = {(cm[0, 0] + cm[1, 1])/sum(cm.ravel())}")
    print(f"")

# %%
sys.path.append("../roc_comparison/")
import compare_auc_delong_xu as delong

# %%


# %%
random_preds = [random.random() for _ in range(n)]

# %%
sklearn.metrics.roc_auc_score(fake_labels, random_preds)

# %%
import numpy as np

delong.delong_roc_test(np.array(fake_labels), np.array(fake_probs), np.array(random_preds))
