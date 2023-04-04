import re, matplotlib, pandas as pd, collections, sklearn, numpy as np, matplotlib.pyplot as plt, sklearn.metrics
from matplotlib.axes import Axes
from typing import Literal

from DeepKS.data.preprocessing.PreprocessingSteps.get_kin_fam_grp import HELD_OUT_FAMILY


def get_all_rocs(self, tl, vl, tel, ho, savefile=""):
    fig = plt.figure(figsize=(12, 12))
    ii = 0
    for i, loader in enumerate([tl, vl, tel, ho]):
        eval_res = self.eval(dataloader=loader)
        outputs = eval_res[2]
        labels = eval_res[3]
        self.roc_core(outputs, labels, i, linecolor=None)
        ii = i
    if savefile:
        fig.savefig(savefile + "_" + str(ii + 1) + ".pdf", bbox_inches="tight")


def get_all_rocs_by_group(
    self, loader, kinase_order, savefile="", kin_fam_grp_file="../data/preprocessing/kin_to_fam_to_grp_817.csv"
):
    kin_to_grp = pd.read_csv(kin_fam_grp_file).applymap(lambda c: re.sub(r"[\(\)\*]", "", c))
    kin_to_grp["Kinase"] = [f"{r['Kinase']}|{r['Uniprot']}" for _, r in kin_to_grp.iterrows()]
    kin_to_grp = kin_to_grp.set_index("Kinase").to_dict()["Group"]
    fig = plt.figure(figsize=(12, 12))
    plt.plot([0, 1], [0, 1], color="r", linestyle="--", alpha=0.5, linewidth=0.5, label="Random Model")
    eval_res = self.eval(dataloader=loader)
    outputs: list[float] = eval_res[-1]
    labels: list[int] = eval_res[3]
    assert len(outputs) == len(labels) == len(kinase_order), (
        "Something is wrong in NNInterface.get_all_rocs_by_group; the length of outputs, the number of labels, and"
        f" the number of kinases in `kinase_order` are not equal. (Respectively, {len(outputs)}, {len(labels)},"
        f" {len(kinase_order)}.)"
    )
    grp_to_indices = collections.defaultdict(list[int])
    for i in range(len(outputs)):
        grp_to_indices[kin_to_grp[kinase_order[i]]].append(i)

    outputs_dd = collections.defaultdict(list[float])
    labels_dd = collections.defaultdict(list[int])
    for g, inds in grp_to_indices.items():
        outputs_dd[g] = [outputs[i] for i in inds]
        labels_dd[g] = [labels[i] for i in inds]

    for i, g in enumerate(grp_to_indices):
        self.roc_core(outputs_dd[g], labels_dd[g], i, line_labels=[g], linecolor=None)

    if savefile:
        fig.savefig(savefile + "_" + str(len(outputs_dd)) + ".pdf", bbox_inches="tight")


def get_all_conf_mats(self, *loaders, savefile="", cutoffs=None, metric: Literal["roc", "acc"] = "roc", cm_labels=None):
    set_labels = ["Train", "Validation", "Test", f"Held Out Family — {HELD_OUT_FAMILY}"]
    for li, l in enumerate([*loaders]):
        preds = []
        eval_res = self.eval(dataloader=l, metric=metric)
        outputs = [x if not isinstance(x, list) else x[0] for x in eval_res[2]]
        labels = eval_res[3]

        if cutoffs is not None:
            for cutoff in cutoffs:
                preds.append([1 if x > cutoff else 0 for x in outputs])
        else:
            cutoffs = list(range(len(loaders)))
            preds = [eval_res[4]]

        nr = int(max(1, len(cutoffs) ** 1 / 2))
        nc = int(max(1, (len(cutoffs) / nr)))

        # fig: Figure
        # axs: list[list[Axes]]

        fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(12, 12))

        axes_list: list[Axes] = np.array(axs).ravel().tolist()
        for i, fp in enumerate(preds):
            cur_ax = axes_list[i]
            plt.sca(cur_ax)
            kwa = {"display_labels": sorted(list(cm_labels.values()))} if cm_labels is not None else {}
            cm = sklearn.metrics.confusion_matrix(labels, fp)
            sklearn.metrics.ConfusionMatrixDisplay(cm, **kwa).plot(
                ax=cur_ax,
                im_kw={
                    "vmin": 0,
                    "vmax": int(
                        np.ceil(np.max(cm) / 10 ** int(np.log10(np.max(cm)))) * 10 ** int(np.log10(np.max(cm)))
                    ),
                },
                cmap="Blues",
            )
            plt.xticks(plt.xticks()[0], plt.xticks()[1], rotation=45, ha="right", va="top")  # type: ignore
            # plt.xticks(f"Cutoff = {cutoffs[i]} | Acc = {(cm[0, 0] + cm[1, 1])/sum(cm.ravel()):3.3f}")

        if savefile:
            fig.savefig(savefile + "_" + set_labels[li] + ".pdf", bbox_inches="tight")
