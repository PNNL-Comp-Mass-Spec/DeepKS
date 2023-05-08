# %%
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import torch
import pickle
from attention_one_layer_k_fold import KinaseSubstrateRelationshipNN
from NNInterface import NNInterface
rcParams['font.family'] = 'P052-Roman'
rcParams['font.size'] = 13
import sklearn

# %%
config = {
        "learning_rate": 0.003,
        "batch_size": 57,
        "ll_size": 20,
        "emb_dim": 22,
        "num_epochs": 15,
        "n_gram": 1,
        "lr_decay_amt": 0.95,
        "lr_decay_freq": 1,
        "num_conv_layers": 1,
        "dropout_pr": 0.37,
        "site_param_dict": {"kernels": [8], "out_lengths": [8], "out_channels": [20]},
        "kin_param_dict": {"kernels": [80], "out_lengths": [8], "out_channels": [20]},
    }

learning_rate = config['learning_rate']
batch_size = config['batch_size']
ll_size = config['ll_size']
emb_dim = config['emb_dim']
num_epochs = config['num_epochs']
lr_decay_amt = config['lr_decay_amt']
lr_decay_freq = config['lr_decay_freq']
n = n_gram = config['n_gram']
num_conv_layers = config['num_conv_layers']
site_param_dict = config['site_param_dict']
kin_param_dict = config['kin_param_dict']
dropout_pr = config['dropout_pr']


# %%
# Read in model and data

torch.load("bin/current_best_model.pkl")
train_loader = pickle.load(open("bin/train_loader.pkl", "rb"))
val_loader = pickle.load(open("bin/val_loader.pkl", "rb"))

num_classes = 1
device = "cpu"

model = KinaseSubstrateRelationshipNN(num_classes=num_classes, inp_size=NNInterface.get_input_size(train_loader), ll_size=ll_size, emb_dim=emb_dim, num_conv_layers=num_conv_layers, site_param_dict=site_param_dict, kin_param_dict=kin_param_dict).to(device)
model.load_state_dict(torch.load("bin/current_best_model.pkl", map_location=torch.device(device)))

# %%
data = val_loader.dataset.data[:500]
target = val_loader.dataset.target[:500]
class_ = val_loader.dataset.class_[:500]

# %%
# Get predictions

# with torch.no_grad():
#     model.eval()
#     preds = torch.sigmoid(model(data, target)).cpu().numpy().tolist()
#     labels = class_.cpu().numpy().tolist()

probs = pickle.load(open("bin/probs.pkl", "rb"))
labels_ = pickle.load(open("bin/labels.pkl", "rb"))

# %%
probs

# %%
colors = ['r' if label == 1 else 'b' for label in labels_]
xlm = [-0.05, 1.05]
ylm = [-0.025*len(probs), 1.025*len(probs)]
plt.scatter(probs, range(len(probs)), c=colors, s=3)
plt.axhspan(ylm[0], ylm[1], -2, 0.395, facecolor='b', alpha=0.2)
plt.axhspan(ylm[0], ylm[1], 0.395, 2, facecolor='r', alpha=0.2)
plt.xlim(xlm)
plt.ylim(ylm)
plt.xlabel("Probability that input is a target")
plt.ylabel(f"Input index (total = {len(probs)})")
plt.title("Model Predictions")

# %%
fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels_, probs)

p = sum(labels_)
n = len(labels_) - p

fp = p*fpr
tp = p*tpr
fn = n*(1-tpr)
tn = n*(1-fpr)
acc = (tp + tn)/(fp + tp + fn + tn)


# %%
# sklearn.metrics.ConfusionMatrixDisplay(sklearn.metrics.confusion_matrix(labels, [1 if pred > 0.5 else 0 for pred in preds])).plot()
fig, ax = plt.subplots()
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")

rax = ax.twinx()
ax.plot(fpr, tpr, label = "ROC Curve (left axis)", color = "blue", linewidth=2)
plt.ylabel("Accuracy")
rax.plot(fpr, acc, label = "Accuracy", color = "orange", linewidth=1)
rax.scatter(fpr[[round(x, 3) for x in thresholds.tolist()].index(0.501)], 0.739, label = "Model's Validation Accuracy (50% Prob. Thresh.)", color = "violet", s = 10)
rax.scatter(fpr[np.argmax(acc)], acc[np.argmax(acc)], s = 10, label = "Point of Best Accuracy", color = "red")


# plt.hlines(acc[np.argmax(acc)], fpr[np.argmax(acc)], 2, color = "red", linestyles=['dashed'], linewidth = 0.15)
# plt.vlines(fpr[np.argmax(acc)], -1, acc[np.argmax(acc)], color = "red", linestyles=['dashed'], linewidth = 0.15)
rax.plot(fpr, thresholds, color = "lime", label = "Probability Threshold to Attain Given Accuracy", linewidth = 1)

handles, labels = ax.get_legend_handles_labels()
handlesr, labelsr = rax.get_legend_handles_labels()
handles = handles + handlesr
labels = labels + labelsr
plt.legend(handles, labels, loc="lower right", prop={'size': 8})

ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
rax.set_xlim(-0.01, 1.01)
rax.set_ylim(-0.01, 1.01)

# Grid
ax.vlines([x/100 for x in range(0, 101, 5)], -1, 2, color = 'grey', linewidth = 0.1)
ax.hlines([x/100 for x in range(0, 101, 5)], -1, 2, color = 'grey', linewidth = 0.1)

plt.title(f"ROC Curve and Accuracy Curve\n(Area under ROC = {round(sklearn.metrics.roc_auc_score(labels_, probs), 3)})")

# %%
thresholds

# %%
cm = sklearn.metrics.confusion_matrix(labels_, probs)

sklearn.metrics.ConfusionMatrixDisplay()

# %%
thresholds[503]

# %%
v = val_loader.dataset.target.numpy()

# %%
sv = set(tuple([tuple(x) for x in np.unique(v, axis = 0).tolist()]))

# %%
tv = set(tuple([tuple(x) for x in np.unique(t, axis = 0).tolist()]))

# %%
set.intersection(sv, tv)


