"""Old plotting functions that may be useful to take from in the future."""  # pragma: no cover
# pragma: no cover
import re, matplotlib, pandas as pd, collections, sklearn, numpy as np, matplotlib.pyplot as plt, sklearn.metrics  # pragma: no cover
from matplotlib.axes import Axes  # pragma: no cover
from typing import Literal  # pragma: no cover

# pragma: no cover
from DeepKS.data.preprocessing.PreprocessingSteps.get_kin_fam_grp import HELD_OUT_FAMILY  # pragma: no cover


# pragma: no cover
# pragma: no cover
def get_all_rocs(self, tl, vl, tel, ho, savefile=""):  # pragma: no cover
    fig = plt.figure(figsize=(12, 12))  # pragma: no cover
    ii = 0  # pragma: no cover
    for i, loader in enumerate([tl, vl, tel, ho]):  # pragma: no cover
        eval_res = self.eval(dataloader=loader)  # pragma: no cover
        outputs = eval_res[2]  # pragma: no cover
        labels = eval_res[3]  # pragma: no cover
        self.roc_core(outputs, labels, i, linecolor=None)  # pragma: no cover
        ii = i  # pragma: no cover
    if savefile:  # pragma: no cover
        fig.savefig(savefile + "_" + str(ii + 1) + ".pdf", bbox_inches="tight")  # pragma: no cover


# pragma: no cover
# pragma: no cover
def get_all_rocs_by_group(  # pragma: no cover
    self,
    loader,
    kinase_order,
    savefile="",
    kin_fam_grp_file="../data/preprocessing/kin_to_fam_to_grp_817.csv",  # pragma: no cover
):  # pragma: no cover
    kin_to_grp = pd.read_csv(kin_fam_grp_file).applymap(lambda c: re.sub(r"[\(\)\*]", "", c))  # pragma: no cover
    kin_to_grp["Kinase"] = [f"{r['Kinase']}|{r['Uniprot']}" for _, r in kin_to_grp.iterrows()]  # pragma: no cover
    kin_to_grp = kin_to_grp.set_index("Kinase").to_dict()["Group"]  # pragma: no cover
    fig = plt.figure(figsize=(12, 12))  # pragma: no cover
    plt.plot(
        [0, 1], [0, 1], color="r", linestyle="--", alpha=0.5, linewidth=0.5, label="Random Model"
    )  # pragma: no cover
    eval_res = self.eval(dataloader=loader)  # pragma: no cover
    outputs: list[float] = eval_res[-1]  # pragma: no cover
    labels: list[int] = eval_res[3]  # pragma: no cover
    assert len(outputs) == len(labels) == len(kinase_order), (  # pragma: no cover
        "Something is wrong in NNInterface.get_all_rocs_by_group; the length of outputs, the number of labels, and"  # pragma: no cover
        f" the number of kinases in `kinase_order` are not equal. (Respectively, {len(outputs)}, {len(labels)},"  # pragma: no cover
        f" {len(kinase_order)}.)"  # pragma: no cover
    )  # pragma: no cover
    grp_to_indices = collections.defaultdict(list[int])  # pragma: no cover
    for i in range(len(outputs)):  # pragma: no cover
        grp_to_indices[kin_to_grp[kinase_order[i]]].append(i)  # pragma: no cover
    # pragma: no cover
    outputs_dd = collections.defaultdict(list[float])  # pragma: no cover
    labels_dd = collections.defaultdict(list[int])  # pragma: no cover
    for g, inds in grp_to_indices.items():  # pragma: no cover
        outputs_dd[g] = [outputs[i] for i in inds]  # pragma: no cover
        labels_dd[g] = [labels[i] for i in inds]  # pragma: no cover
    # pragma: no cover
    for i, g in enumerate(grp_to_indices):  # pragma: no cover
        self.roc_core(outputs_dd[g], labels_dd[g], i, line_labels=[g], linecolor=None)  # pragma: no cover
    # pragma: no cover
    if savefile:  # pragma: no cover
        fig.savefig(savefile + "_" + str(len(outputs_dd)) + ".pdf", bbox_inches="tight")  # pragma: no cover


# pragma: no cover
# pragma: no cover
def get_all_conf_mats(
    self, *loaders, savefile="", cutoffs=None, metric: Literal["roc", "acc"] = "roc", cm_labels=None
):  # pragma: no cover
    set_labels = ["Train", "Validation", "Test", f"Held Out Family — {HELD_OUT_FAMILY}"]  # pragma: no cover
    for li, l in enumerate([*loaders]):  # pragma: no cover
        preds = []  # pragma: no cover
        eval_res = self.eval(dataloader=l, metric=metric)  # pragma: no cover
        outputs = [x if not isinstance(x, list) else x[0] for x in eval_res[2]]  # pragma: no cover
        labels = eval_res[3]  # pragma: no cover
        # pragma: no cover
        if cutoffs is not None:  # pragma: no cover
            for cutoff in cutoffs:  # pragma: no cover
                preds.append([1 if x > cutoff else 0 for x in outputs])  # pragma: no cover
        else:  # pragma: no cover
            cutoffs = list(range(len(loaders)))  # pragma: no cover
            preds = [eval_res[4]]  # pragma: no cover
        # pragma: no cover
        nr = int(max(1, len(cutoffs) ** 1 / 2))  # pragma: no cover
        nc = int(max(1, (len(cutoffs) / nr)))  # pragma: no cover
        # pragma: no cover
        # fig: Figure # pragma: no cover
        # axs: list[list[Axes]] # pragma: no cover
        # pragma: no cover
        fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(12, 12))  # pragma: no cover
        # pragma: no cover
        axes_list: list[Axes] = np.array(axs).ravel().tolist()  # pragma: no cover
        for i, fp in enumerate(preds):  # pragma: no cover
            cur_ax = axes_list[i]  # pragma: no cover
            plt.sca(cur_ax)  # pragma: no cover
            kwa = (
                {"display_labels": sorted(list(cm_labels.values()))} if cm_labels is not None else {}
            )  # pragma: no cover
            cm = sklearn.metrics.confusion_matrix(labels, fp)  # pragma: no cover
            sklearn.metrics.ConfusionMatrixDisplay(cm, **kwa).plot(  # pragma: no cover
                ax=cur_ax,  # pragma: no cover
                im_kw={  # pragma: no cover
                    "vmin": 0,  # pragma: no cover
                    "vmax": int(  # pragma: no cover
                        np.ceil(np.max(cm) / 10 ** int(np.log10(np.max(cm))))
                        * 10 ** int(np.log10(np.max(cm)))  # pragma: no cover
                    ),  # pragma: no cover
                },  # pragma: no cover
                cmap="Blues",  # pragma: no cover
            )  # pragma: no cover
            plt.xticks(plt.xticks()[0], plt.xticks()[1], rotation=45, ha="right", va="top")  # type: ignore # pragma: no cover
            # plt.xticks(f"Cutoff = {cutoffs[i]} | Acc = {(cm[0, 0] + cm[1, 1])/sum(cm.ravel()):3.3f}") # pragma: no cover
        # pragma: no cover
        if savefile:  # pragma: no cover
            fig.savefig(savefile + "_" + set_labels[li] + ".pdf", bbox_inches="tight")  # pragma: no cover
