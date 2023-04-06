from abc import ABC, abstractmethod

from matplotlib import patheffects, pyplot as plt, patches as ptch, collections as clct
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
import warnings, matplotlib, numpy as np, pandas as pd, json
from matplotlib.text import Text
import sklearn, sklearn.metrics, os
from numpy.typing import ArrayLike
import random

random.seed(42)


class AnnotationHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Get the text from the original handle (the annotation)
        text = orig_handle.get_text()

        # Create a new text object to use in the legend
        # Add an example of the annotation to the legend text
        label_text = f"{text} (example)"
        label = Text(xdescent, ydescent, label_text, fontsize=fontsize, ha="left", va="center", transform=trans)
        return [label]


class ROC(ABC):
    def __init__(self, fig_kwargs={"figsize": (10, 10)}) -> None:
        self._roc_is_made = False
        matplotlib.rcParams["font.family"] = "P052"
        # matplotlib.interactive(False)
        self.fig, self.ax = plt.subplots(**fig_kwargs)

    def make_roc(self, scores, labels, *args, **kwargs):
        self._make_roc_core(scores, labels, *args, **kwargs)
        self._roc_is_made = True

    def display_roc(self):
        if self._roc_is_made:
            # matplotlib.interactive(True)
            prev_ax = plt.gca()
            plt.sca(self.ax)
            plt.show()
            plt.sca(prev_ax)
            # matplotlib.interactive(False)
        else:
            warnings.warn("ROC is not made yet. Ignoring Request. Call `make_roc(scores, labels)` first.")

    def save_roc(self, save_name, save_fig_kwargs={"bbox_inches": "tight"}):
        if self._roc_is_made:
            self.fig.savefig(save_name, **save_fig_kwargs)
        else:
            warnings.warn("ROC is not made yet. Ignoring Request. Call `make_roc(scores, labels)` first.")

    def _get_roc_info(self, scores, labels):
        return sklearn.metrics.roc_curve(labels, scores)

    def _make_roc_core(self, scores: ArrayLike, labels, *args, **kwargs):
        pass


class SplitIntoKinasesROC(ROC):
    def __init__(self, fig_kwargs={"figsize": (10, 10)}) -> None:
        super().__init__(fig_kwargs=fig_kwargs)

    def make_roc(
        self,
        scores,
        labels,
        kinase_identities,
        kinase_group,
        plotting_kwargs={
            "plot_markers": True,
            "plot_unified_line": True,
            "jitter_amount": 0.0,
            "diff_by_color": True,
            "diff_by_opacity": False,
            "show_zones": False,
        },
    ):
        return super().make_roc(scores, labels, kinase_identities, kinase_group, plotting_kwargs=plotting_kwargs)

    def cache_prepared_data(self, scores, labels, kinase_identities, kinase_group, plotting_kwargs):
        cache = {
            "scores": [x for x in scores],
            "labels": [x for x in labels],
            "kinase_identities": [x for x in kinase_identities],
            "kinase_group": kinase_group,
            "plotting_kwargs": plotting_kwargs,
        }
        input("Press any key to overwrite the cache file... (ctrl+c to cancel)")
        with open("roc-cache.json", "w") as f:
            json.dump(cache, f)

    def _make_roc_core(self, scores, labels, kinase_identities: list[str], kinase_group, plotting_kwargs):
        assert len(scores) == len(labels) == len(kinase_identities), "Lengths of lists must be equal"

        plot_unified_line = plotting_kwargs.get("plot_unified_line", True)
        jitter_amount = plotting_kwargs.get("jitter_amount", 0.0)
        diff_by_color = plotting_kwargs.get("diff_by_color", True)
        diff_by_opacity = plotting_kwargs.get("diff_by_opacity", False)
        show_zones = plotting_kwargs.get("show_zones", False)
        show_cutoff = plotting_kwargs.get("show_cutoff", False)
        show_acc = plotting_kwargs.get("show_acc", False)
        focus_on = plotting_kwargs.get("focus_on", None)
        print(focus_on)

        jitter = lambda x: (
            x if float(jitter_amount) == 0.0 else [xi + jitter_amount * random.random() - jitter_amount / 2 for xi in x]
        )

        def gradient(rgb1, rgb2, length):
            r1, g1, b1 = rgb1
            r2, g2, b2 = rgb2
            res = []
            for i in range(length):
                r = r1 + (r2 - r1) * i / (length - 1)
                g = g1 + (g2 - g1) * i / (length - 1)
                b = b1 + (b2 - b1) * i / (length - 1)
                res.append((r, g, b))
            return res

        if diff_by_color:
            y_grad = gradient(
                (220 / 255, 220 / 255, 0 / 255), (60 / 255, 60 / 255, 0 / 255), len(set(kinase_identities))
            )
            c_grad = gradient(
                (0 / 255, 255 / 255, 255 / 255), (0 / 255, 80 / 255, 80 / 255), len(set(kinase_identities))
            )
            m_grad = gradient(
                (240 / 255, 15 / 255, 255 / 255), (80 / 255, 0 / 255, 80 / 255), len(set(kinase_identities))
            )
        else:
            y_grad = [(45 / 255, 45 / 255, 0 / 255)] * len(set(kinase_identities))
            c_grad = [(0 / 255, 127 / 255, 127 / 255)] * len(set(kinase_identities))
            m_grad = [(127 / 255, 0 / 255, 127 / 255)] * len(set(kinase_identities))

        if plot_unified_line:
            y_grad.append((0, 0, 0))
            c_grad.append((0, 0, 0))
            m_grad.append((0, 0, 0))

        marker_list = [x for x in Line2D.markers.keys()]
        del_list = [",", "o", "^", "<", ">", "1", "2", "3", "4", "|", "_", "", " ", "None", "none"] + list(range(12))
        marker_list = sorted(list(set(marker_list) - set(del_list)))

        def roc_acc_thsh(fprs, tprs, thresholds, datas, plot_markers=True, roc_labels_list=kinase_identities):
            assert len(fprs) == len(tprs) == len(thresholds) == len(datas), "Lengths of lists must be equal"
            assert len(roc_labels_list) == 0 or len(roc_labels_list) == len(
                fprs
            ), "Length of roc_labels_list must be 0 or equal to length of fprs"
            pat2 = None
            ntr = []
            for i in range(len(fprs)):
                mn = np.min(datas[i]["Score"].values) - 0.025
                ma = np.max(datas[i]["Score"].values) + 0.025
                nt = np.clip(thresholds[i], mn, ma).tolist()
                ntr.append(nt)
            thresholds = ntr

            roc_labels_list = roc_labels_list if len(roc_labels_list) > 0 else [f"ROC {i}" for i in range(len(fprs))]
            roc_col = c_grad[-1]
            cut_col = y_grad[-1]
            acc_col = m_grad[-1]

            fig, ax = self.fig, self.ax
            ax.set_xticks(np.arange(0, 1.05, 0.05))
            ax.set_xticklabels([f"{x:.2f}" for x in np.arange(0, 1.05, 0.05)], rotation=90)
            ax.set_yticks(np.arange(0, 1.05, 0.05))
            ax.set_title(f"ROC Curves Stratified by Kinase (Kinase Group: {kinase_group})")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            if show_cutoff:
                tax = ax.twinx()
                tax.set_ylabel("Cutoff")
                tax.set_ylim(0, 1.1)
                tax.yaxis.label.set_color(cut_col)
                tax.tick_params(axis="y", colors=cut_col)
            if show_acc:
                taxacc = ax.twinx()
                taxacc.spines.right.set_position(("axes", 1.15))
                taxacc.set_ylabel("Accuracy")
                taxacc.yaxis.label.set_color(acc_col)
                taxacc.tick_params(axis="y", colors=acc_col)
                taxacc.set_ylim(0, 1)

            tax_handle_list = []
            roc_handle_list = []
            taxacc_handle_list = []

            for i, (fpr, tpr, thrr, data) in enumerate(zip(fprs, tprs, thresholds, datas)):
                mar = f"{roc_labels_list[i]}" if roc_labels_list[i] != "Unified" else None
                col = c_grad[i]
                alp = 0.4 if not diff_by_opacity else max(0.05, min(1, 1 / len(fprs)))
                unif = False
                is_focus = False
                linew = 2
                if roc_labels_list[i] == "Unified":
                    mar = None
                    col = "black"
                    alp = 1
                    unif = True
                    linew = 1
                if focus_on is not None and roc_labels_list[i] == focus_on:
                    linew = 3
                    is_focus = True
                (hand,) = ax.plot(
                    jitter(fpr) if not unif else fpr,
                    tpr,
                    linewidth=linew,
                    marker=None,
                    markersize=7,
                    color=col,
                    alpha=alp,
                    markeredgewidth=0,
                    zorder=2,
                    **(
                        dict(label=f"ROC curve of Kinase in {kinase_group}")
                        if i == 0
                        else dict(label=f"ROC curve of all kinases {kinase_group}")
                        if unif
                        else {}
                    ),
                )
                if is_focus:
                    (hand,) = ax.plot(
                        jitter(fpr) if not unif else fpr,
                        tpr,
                        linestyle="dashed",
                        linewidth=linew,
                        marker=None,
                        markersize=7,
                        color="red",
                        alpha=0.15,
                        markeredgewidth=0,
                        zorder=3,
                        label="ROC curve of emphasized kinase",
                    )
                if plot_markers and mar is not None:
                    r = random.uniform(0.01, 0.06)
                    random_angle = 2 * np.pi * i / len(fprs)  # random.uniform(0, 2 * np.pi) + np.pi/6
                    offset_x = r * np.cos(random_angle)
                    offset_y = r * np.sin(random_angle)
                    text_angle_degrees = random_angle * 180 / np.pi
                    if 270 >= text_angle_degrees >= 90:
                        text_angle_degrees -= 180
                    for j, (fp, tp) in enumerate(zip(fpr, tpr)):
                        if j % max(2, len(fpr) // 15) == 0 or (fp, tp) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                            ax.annotate(
                                mar,
                                (fp, tp),
                                xytext=(fp + offset_x, tp + offset_y),
                                fontsize=1 if not is_focus else 4,
                                color="red" if is_focus else "black",
                                arrowprops=dict(
                                    arrowstyle="-" if not is_focus else "->",
                                    shrinkA=0,
                                    shrinkB=0,
                                    linewidth=0.1 if not is_focus else 0.4,
                                    alpha=0.5,
                                ),
                                bbox=dict(boxstyle="square", fc="w", ec="w", pad=-0.1, alpha=0.0),
                                verticalalignment="center",
                                rotation=text_angle_degrees,
                                rotation_mode="anchor",
                                **dict(label="Kinase|Uniprot Label") if i == 0 else {},
                            )
                roc_handle_list.append(hand)
                if show_acc:
                    accs = []
                    for thr in thrr:
                        targ_labs = ["Target", 1]
                        decoy_labs = ["Decoy", 0]
                        correctly_predicted_actual_pos = len(
                            data[(data["Class"].isin(targ_labs)) & (data["Score"] >= thr)]
                        )
                        correctly_predicted_actual_neg = len(
                            data[(data["Class"].isin(decoy_labs)) & (data["Score"] < thr)]
                        )
                        acc = (correctly_predicted_actual_pos + correctly_predicted_actual_neg) / len(data)
                        accs.append(acc)

                    (taxacc_plot_handle,) = taxacc.plot(
                        jitter(fpr),
                        accs,
                        label="accuracy",
                        marker=None,
                        markersize=6,
                        color=y_grad[i],
                        linewidth=linew / 3,
                        markeredgewidth=0,
                        alpha=alp / 2 if not unif else 1,
                    )
                    taxacc_handle_list.append(taxacc_plot_handle)
                if show_cutoff:
                    (tax_plot_handle,) = tax.plot(
                        jitter(fpr),
                        thrr,
                        color=m_grad[i],
                        label="cutoff",
                        marker=None,
                        markersize=6,
                        linewidth=linew / 3,
                        markeredgewidth=0,
                        alpha=alp / 2 if not unif else 1,
                    )
                    tax_handle_list.append(tax_plot_handle)

                if show_zones and show_acc:
                    pat = ptch.Rectangle((fpr[np.argmax(accs)] - 0.01, -1), 0.02, 3, color="#f0f0f030", linewidth=0)
                    pat2 = ptch.Rectangle(
                        (-1, np.max(accs) - 0.01), 3, 0.02, color="#f0f0f030", linewidth=0, label="Best Accuracy"
                    )
                    collect = clct.PatchCollection([pat, pat2], match_original=True, zorder=-10)
                    ax.add_collection(collect)  # type: ignore

            (hand,) = ax.plot([0, 1], [0, 1], color="black", lw=0.4, linestyle="--", label="Random Model")
            roc_handle_list.append(hand)
            roc_labels_list.append("Random Model")

            ax.yaxis.label.set_color(roc_col)
            ax.tick_params(axis="y", colors=roc_col)
            # tax.legend(
            #     handles=taxacc_handle_list + [pat2] if pat2 is not None else [] + tax_handle_list,
            #     labels=[f"Accuracy ({x})" for x in roc_labels_list[:-1]]
            #     + ["Zones of Best Accuracy"]
            #     + [f"Cutoff ({x})" for x in roc_labels_list[:-1]],
            #     loc=(0.7, 0.05),
            # )
            # ax.legend(handles=roc_handle_list, labels=roc_labels_list, loc=(0.15, 0.05))
            for a in [ax] + ([tax] if show_cutoff else []) + ([taxacc] if show_acc else []):
                a.set_xlim(-0.05, 1.05)
                a.set_ylim(-0.05, 1.05)

            ax.legend(loc="best", handler_map={tuple: AnnotationHandler()})
            self.fig = fig

        all_datas = pd.DataFrame(
            {"Score": scores, "Class": [bool(x) for x in labels], "Kinase Label": kinase_identities}
        )

        datas = [df for _, df in all_datas.groupby("Kinase Label")]

        fprs, tprs, thresholds = [], [], []

        for data in datas:
            if set(data["Kinase Label"]) == {"ABL1|P00519"}:
                print(data)
            fpr, tpr, threshold = sklearn.metrics.roc_curve(data["Class"], data["Score"])
            fprs.append(fpr)
            tprs.append(tpr)
            thresholds.append(threshold)

        roc_labels_list = [list(set(data["Kinase Label"]))[0] for data in datas]
        if plot_unified_line:
            unified_fpr, unified_tpr, unified_threshold = sklearn.metrics.roc_curve(
                all_datas["Class"], all_datas["Score"]
            )
            fprs.append(unified_fpr)
            tprs.append(unified_tpr)
            thresholds.append(unified_threshold)
            datas.append(all_datas)
            roc_labels_list.append("Unified")

        roc_acc_thsh(
            fprs,
            tprs,
            thresholds,
            datas,
            roc_labels_list=roc_labels_list,
            plot_markers=plotting_kwargs.get("plot_markers", True),
        )

        self.cache_prepared_data(scores, labels, kinase_identities, kinase_group, plotting_kwargs)


class SplitIntoGroupsROC(ROC):
    def __init__(self) -> None:
        super().__init__()

    def _make_roc_core(self, scores, labels):
        pass


def test():
    test_dat = pd.read_csv(
        "/Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/tools/dff_sample_test_data.csv"
    )
    the_roc = SplitIntoKinasesROC()
    the_roc.make_roc(
        test_dat["Score"],
        test_dat["Class"],
        test_dat["Kin"].tolist(),
        "TESTGRP",
        plotting_kwargs={
            "plot_markers": True,
            "plot_unified_line": True,
            "jitter_amount": 0,
            "diff_by_color": False,
            "diff_by_opacity": True,
            "focus_on": "HIPK2|Q9H2X6",
        },
    )
    # the_roc.display_roc()
    the_roc.save_roc("test_roc.pdf", save_fig_kwargs={"bbox_inches": "tight", "dpi": 300})


def main():
    if os.path.exists("roc-cache.json"):
        with open("roc-cache.json", "r") as f:
            cache: dict = json.load(f)
            del cache["plotting_kwargs"]
            roc = SplitIntoKinasesROC(fig_kwargs={"figsize": (10, 10)})
            roc.make_roc(
                plotting_kwargs={
                    "plot_markers": True,
                    "plot_unified_line": True,
                    "jitter_amount": 0,
                    "diff_by_color": False,
                    "diff_by_opacity": True,
                    "focus_on": "ABL1|P00519",
                },
                **cache,
            )
            roc.save_roc("roc_tk.pdf", save_fig_kwargs={"bbox_inches": "tight", "dpi": 300})
    else:
        raise FileNotFoundError("No cache found")


if __name__ == "__main__":
    main()
    # test()
