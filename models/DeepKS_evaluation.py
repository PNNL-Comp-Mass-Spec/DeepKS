from functools import cache
from math import sqrt
import warnings, matplotlib, numpy as np, pandas as pd, json, traceback, sys, os
import sklearn, sklearn.metrics, tempfile, collections, random, cloudpickle as pickle, re, torch, pathlib

# from sklearn._typing import float
from ..tools.roc_helpers import ROCHelpers
from ..models.multi_stage_classifier import MultiStageClassifier
from multipledispatch import dispatch

random.seed(42)
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt, patches as ptch, collections as clct
from matplotlib import colormaps  # type: ignore
from matplotlib import axes
from typing import Callable, Hashable, Literal, Union, Any
from termcolor import colored

get_cmap = colormaps.get_cmap
from pydash import _
from numpy.typing import ArrayLike
from ..tools.file_names import get as get_file_name

from ..config.logging import get_logger

logger = get_logger()

warnings.simplefilter("once", Warning)


join_first = lambda levels, x: os.path.join(pathlib.Path(__file__).parent.resolve(), *[".."] * levels, x)


def warning_handler(message, category, filename, lineno, file=None, line=None):
    if hasattr(file, "write"):
        log = file
    else:
        log = sys.stderr
    traceback.print_stack(file=log)
    if log is not None:
        log.write(warnings.formatwarning(message, category, filename, lineno, line=""))
    else:
        print("log is None")


warnings.showwarning = warning_handler


class PerformancePlot(ABC):  # ABC = Abstract Base Class
    def __init__(self, fig_kwargs={"figsize": (10, 10)}, pass_in_ax=None) -> None:
        self._is_made = False
        matplotlib.rcParams["font.family"] = "P052"
        self.cn = self.__class__.__name__
        self._not_made_error_msg = (
            f"{self.cn} Plot is not made yet. Ignoring Request. Call `my_{_.snake_case(self.cn)}_instance.make_plot()`"
            " first."
        )
        self.fig_kwargs = fig_kwargs
        self.pass_in_ax: axes.Axes | None = pass_in_ax

    def make_plot(self, *args, **kwargs):
        if self.pass_in_ax is None:
            self.fig, self.ax = plt.subplots(**self.fig_kwargs)
        else:
            self.ax = self.pass_in_ax
            self.fig = self.ax.figure
        self._make_plot_core(*args, **kwargs)
        self._is_made = True

    def display_plot(self):
        if self._is_made:
            prev_ax = plt.gca()
            plt.sca(self.ax)
            plt.show()
            plt.sca(prev_ax)
        else:
            warnings.warn(self._not_made_error_msg)

    def save_plot(
        self,
        save_name,
        file_name_kwargs=dict(
            directory=join_first(1, "images/EvaluationandResults/ROC"),
            prefix=f"<Class Name>_plot",
            suffix="pdf",
            prefix_sep="_",
            suffix_sep=".",
            win_compat=True,
        ),
        save_fig_kwargs={"bbox_inches": "tight"},
    ):
        if file_name_kwargs.get("prefix") == "<Class Name>_plot":
            file_name_kwargs["prefix"] = f"{self.cn}_plot"

        filename = get_file_name(**file_name_kwargs)
        if self._is_made:
            self.fig.savefig(filename, **save_fig_kwargs)
        else:
            warnings.warn(self._not_made_error_msg)

    @abstractmethod
    def _make_plot_core(self, *args, **kwargs):
        pass


class CM(PerformancePlot, ABC):
    def __init__(self, **fig_kwargs) -> None:
        super().__init__(**fig_kwargs)

    def _get_cm(self, scores, labels, **cm_kwargs):
        self._cm = sklearn.metrics.confusion_matrix(labels, scores, **cm_kwargs)


class ROC(PerformancePlot, ABC):
    def __init__(self, pass_in_ax=None, **fig_kwargs) -> None:
        super().__init__(pass_in_ax=pass_in_ax, **fig_kwargs)

    def _get_roc_info(self, truthses: list[list[Any]], scoreses: list[list[float]], plot_unified_line: bool):
        fprses = []
        tprses = []
        thresholdses = []
        for truths, scores in zip(truthses, scoreses):
            fprs, tprs, thresholds = sklearn.metrics.roc_curve(truths, scores)
            fprses.append(fprs)
            tprses.append(tprs)
            thresholdses.append(thresholds)
        if plot_unified_line:
            fprs, tprs, thresholds = sklearn.metrics.roc_curve(np.concatenate(truthses), np.concatenate(scoreses))
            fprses.append(fprs)
            tprses.append(tprs)
            thresholdses.append(thresholds)
        return fprses, tprses, thresholdses

    def _roc_core_components(
        self,
        truthses,
        scoreses,
        roc_labels_list=[],
        plotting_kwargs={},
        title_extra="",
        roc_lab_extra="",
    ):
        def jitter(x):
            if float(jitter_amount) == 0.0:
                return x
            else:
                return [xi + jitter_amount * random.random() - jitter_amount / 2 for xi in x]

        jitter_amount = plotting_kwargs.get("jitter_amount", 0.0)
        diff_by_opacity = plotting_kwargs.get("diff_by_opacity", False)
        show_zones = plotting_kwargs.get("show_zones", False)
        show_cutoff = plotting_kwargs.get("show_cutoff", False)
        show_acc = plotting_kwargs.get("show_acc", False)
        focus_on = plotting_kwargs.get("focus_on", None)
        plot_pointer_labels = plotting_kwargs.get("plot_pointer_labels", False)
        fancy_legend = plotting_kwargs.get("fancy_legend", True)
        plot_unified_line = plotting_kwargs.get("plot_unified_line", False)
        plot_mean_value_line = plotting_kwargs.get("plot_mean_value_line", False)
        positive_label = plotting_kwargs.get("positive_label", 1.0)

        fprses, tprses, thresholdses = self._get_roc_info(truthses, scoreses, plot_unified_line=plot_unified_line)

        if plot_unified_line:
            scoreses.append(np.concatenate(scoreses))
            truthses.append(np.concatenate(truthses))
            roc_labels_list.append("All Data")

        # roc_colors = plotting_kwargs.get("roc_colors", "plasma")
        # cut_colors = plotting_kwargs.get("cut_colors", "plasma")
        # acc_colors = plotting_kwargs.get("acc_colors", "plasma")

        # cmap_roc, cmap_cut, cmap_acc = get_cmap(roc_colors), get_cmap(cut_colors), get_cmap(acc_colors)

        # color_places = np.linspace(0, 1, len(fprses), endpoint=True)
        # convert = lambda ls: [matplotlib.colors.to_hex(c) for c in ls]
        # roc_colors = convert(cmap_roc(color_places))
        # cut_colors = convert(cmap_cut(color_places))
        # acc_colors = convert(cmap_acc(color_places))

        assert len(fprses) == len(tprses) == len(thresholdses), "Lengths of lists must be equal"

        assert len(roc_labels_list) in [
            0,
            len(fprses),
            len(fprses) - 1,
        ], "Length of roc_labels_list must be 0 or equal to length of fprs or to length of fprs + 1"

        pat2 = None
        ntr = []
        for i in range(len(fprses)):
            mn = np.min(scoreses[i]) - 0.025
            ma = np.max(scoreses[i]) + 0.025
            nt = np.clip(thresholdses[i], mn, ma).tolist()
            ntr.append(nt)
        thresholdses = ntr

        if len(roc_labels_list) > 0:
            roc_labels_list = roc_labels_list
        else:
            roc_labels_list = [f"ROC {i}" for i in range(len(fprses))]
        # roc_col = roc_colors[0]
        # cut_col = acc_colors[0]
        # acc_col = cut_colors[0]

        fig, ax = self.fig, self.ax
        ax.set_xticks(np.arange(0, 1.05, 0.1))
        ax.set_xticklabels([f"{_:.2f}" for _ in np.arange(0, 1.05, 0.1)], rotation=90)
        ax.set_yticks(np.arange(0, 1.05, 0.1))
        ax.set_title(f"ROC Curves{title_extra}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        if show_cutoff:
            tax = ax.twinx()
            tax.set_ylabel("Cutoff")
            tax.set_ylim(0, 1)
            # tax.yaxis.label.set_color(cut_col)
            # tax.tick_params(axis="y", colors=cut_col)
        aucs = []
        if plot_unified_line:
            pct_mult = 2
        else:
            pct_mult = 1
        sd = sum([len(d) for d in scoreses])

        addl_roc_labels = ["Random"]
        if plot_mean_value_line:
            addl_roc_labels.append("Mean Value")
        else:
            addl_roc_labels.append("")
        if focus_on is not None:
            addl_roc_labels.append("Emph. Curve")
        else:
            addl_roc_labels.append("")
        indent_amt = max([len(r) for r in roc_labels_list + addl_roc_labels])
        for i, (fprs, tprs, thresholds, truths, scores) in enumerate(
            zip(fprses, tprses, thresholdses, truthses, scoreses)
        ):
            assert len(fprs) == len(tprs) == len(thresholds), "Lengths of lists must be equal"
            if roc_labels_list[i] == "All Data":
                mar = None
            else:
                mar = f"{roc_labels_list[i]}"

            # col = roc_colors[i]
            if not diff_by_opacity:
                alp = 0.65
            else:
                alp = max(0.05, min(1, 1 / len(fprses)))
            unif = False
            is_focus = False
            linew = 1
            if roc_labels_list[i] == "All Data":
                mar, col, alp, unif, linew = None, "black", 1, True, 2
            if focus_on is not None and roc_labels_list[i] == focus_on:
                linew, is_focus = 1, True

            # roc_style_kwargs = dict(linewidth=linew, markersize=7, color=col, alpha=alp, zorder=2)
            roc_style_kwargs = dict(linewidth=linew, markersize=7, alpha=alp, zorder=2)
            # if i == 0 and not legend_all_lines:
            #     roc_style_kwargs.update(dict(label=f"ROC curve of a{roc_lab_extra}"))
            # elif unif:
            #     roc_style_kwargs.update(dict(label=f"ROC curve of all{roc_lab_extra}s"))
            # elif legend_all_lines:
            #     roc_style_kwargs.update(dict(label=roc_labels_list[i]))

            roc_style_kwargs |= {"label": roc_labels_list[i]}
            aucscore = ROCHelpers.protected_roc_auc_score(truths, scores)
            alpha_level = plotting_kwargs.get("alpha_level", 0.05)
            if fancy_legend:
                alpha = 0.05
                if len(set([x for x in truths])) == 1:
                    ci, p = (0, 1), 1
                else:
                    ci, p = ROCHelpers.auc_confidence_interval(
                        np.array(truths).astype(int), np.array(scores), alpha_level
                    )
                is_signif = p < alpha_level

                if plot_unified_line:
                    pct_mult = 2
                else:
                    pct_mult = 1
                fancy_lab = (
                    f"{roc_labels_list[i]:>{indent_amt}}\n"
                    f"• AUC = {aucscore:1.3f}\n"
                    f"• {100-int(alpha*100)}% CI of AUC = [{ci[0]:1.3f}, {ci[1]:1.3f}]\n"
                    # f" {ROCHelpers.get_p_stars(p)}\n"
                    f"• n = {len(scores)} = {len(scores)*100/sd*pct_mult:.2f}% of all data"
                )
                roc_style_kwargs.update(dict(label=fancy_lab))

                if not is_signif:
                    roc_style_kwargs.update(dict(alpha=0.3, linestyle="dotted", dash_capstyle="round"))

            ax.plot(fprs, tprs, **roc_style_kwargs)
            if is_focus:
                roc_style_kwargs.update(
                    dict(
                        linestyle="dashed",
                        linewidth=linew * 2,
                        color="red",
                        alpha=1,
                        zorder=3,
                        label=f"{'Emph. Curve':>{indent_amt}}",
                    )
                )
                (focused,) = ax.plot(fprs, tprs, **roc_style_kwargs)
                focused.set_dashes([3, 1.5, 2, 1.5])  # type: ignore
                focused.set_dash_capstyle("round")
            aucs.append(aucscore)
            if plot_pointer_labels and mar is not None:
                r = random.uniform(0.01, 0.06)
                random_angle = 2 * np.pi * i / len(fprses)  # random.uniform(0, 2 * np.pi) + np.pi/6
                offset_x = r * np.cos(random_angle)
                offset_y = r * np.sin(random_angle)
                text_angle_degrees = random_angle * 180 / np.pi
                if 270 >= text_angle_degrees >= 90:
                    text_angle_degrees -= 180
                for j, (fp, tp) in enumerate(zip(fprs, tprs)):
                    if j % max(2, len(fprs) // 15) == 0 or (fp, tp) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        if is_focus:
                            fontsize = 1
                            color = "red"
                            arrowstyle = "->"
                            linewidth = 0.4
                        else:
                            fontsize = 4
                            color = "black"
                            arrowstyle = "-"
                            linewidth = 0.1
                        ax.annotate(
                            mar,
                            (fp, tp),
                            xytext=(fp + offset_x, tp + offset_y),
                            fontsize=fontsize,
                            color=color,
                            arrowprops=dict(
                                arrowstyle=arrowstyle,
                                shrinkA=0,
                                shrinkB=0,
                                linewidth=linewidth,
                                alpha=0.5,
                            ),
                            bbox=dict(boxstyle="square", fc="w", ec="w", pad=-0.1, alpha=0.0),
                            verticalalignment="center",
                            rotation=text_angle_degrees,
                            rotation_mode="anchor",
                        )

            if show_acc:
                accs = []
                for threshold in thresholds:
                    correctly_predicted_actual_pos = len(
                        [None for truth, score in zip(truths, scores) if truth == positive_label and score >= threshold]
                    )
                    correctly_predicted_actual_neg = len(
                        [None for truth, score in zip(truths, scores) if truth != positive_label and score < threshold]
                    )
                    acc = (correctly_predicted_actual_pos + correctly_predicted_actual_neg) / len(scores)
                    accs.append(acc)

                taxacc = ax.twinx()
                taxacc.spines.right.set_position(("axes", 1.15))
                taxacc.set_ylabel("Accuracy")
                # taxacc.yaxis.label.set_color(acc_col)
                # taxacc.tick_params(axis="y", colors=acc_col)
                taxacc.set_ylim(0, 1)

                if unif:
                    alpha = 1
                else:
                    alpha = alp / 2

                taxacc.plot(
                    jitter(fprs),
                    accs,
                    label="accuracy",
                    marker=None,
                    markersize=6,
                    # color=acc_colors[i],
                    linewidth=linew / 3,
                    markeredgewidth=0,
                    alpha=alpha,
                )

                if show_zones:
                    pat = ptch.Rectangle((fprs[np.argmax(accs)] - 0.01, -1), 0.02, 3, color="#f0f0f030", linewidth=0)
                    pat2 = ptch.Rectangle(
                        (-1, np.max(accs) - 0.01), 3, 0.02, color="#f0f0f030", linewidth=0, label="Best Accuracy"
                    )
                    collect = clct.PatchCollection([pat, pat2], match_original=True, zorder=-10)
                    ax.add_collection(collect)  # type: ignore
            if show_cutoff:
                if unif:
                    alpha = 1
                else:
                    alpha = alp / 2
                tax = ax.twinx()
                tax.set_ylabel("Cutoff")
                tax.set_ylim(0, 1)
                # tax.yaxis.label.set_color(cut_col)
                # tax.tick_params(axis="y", colors=cut_col)
                tax.plot(
                    jitter(fprs),
                    thresholds,
                    # color=cut_colors[i],
                    label="cutoff",
                    marker=None,
                    markersize=6,
                    linewidth=linew / 3,
                    markeredgewidth=0,
                    alpha=alpha,
                )
                tax.set_ylim(-0.05, 1.05)

        if plot_mean_value_line:
            total_obs = sum([len(scores) for scores in scoreses])
            weights = [len(scores) / total_obs for scores in scoreses]
            Xavg, Yavg, AUCavg = ROCHelpers.RocAvgValue.get_avg_line_pnts(fprses, tprses, weights)
            lab = f"{'Mean Value':>{indent_amt}} | AUC {AUCavg:1.3f}"
            ax.plot(Xavg, Yavg, color="grey", linewidth=2, label=lab)

        if fancy_legend:
            extra = "\n• AUC = 0.5\n• n = ∞"
        else:
            extra = ""
        random_model_label = f"{'Random Model':>{indent_amt}}{extra}"
        ax.plot([0, 1], [0, 1], color="black", lw=0.4, linestyle="--", label=random_model_label)

        # ax.yaxis.label.set_color(roc_col)
        # ax.tick_params(axis="y", colors=roc_col)

        ax.legend(
            loc="lower right",
            prop={"family": ["Fira Code", "monospace"], "size": 5},
            # bbox_to_anchor=(1, 0.5),
            labelspacing=2,
        )
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlim(-0.01, 1.01)
        self.fig = fig


class SplitIntoKinasesROC(ROC):
    def __init__(self, fig_kwargs={"figsize": (10, 10)}) -> None:
        super().__init__(fig_kwargs=fig_kwargs)

    def make_plot(
        self,
        scores,
        labels,
        kinase_identities,
        kinase_group,
        plotting_kwargs={
            "plot_pointer_labels": True,
            "plot_unified_line": True,
            "jitter_amount": 0.0,
            "diff_by_color": True,
            "diff_by_opacity": False,
            "show_zones": False,
        },
    ):
        return super().make_plot(scores, labels, kinase_identities, kinase_group, plotting_kwargs=plotting_kwargs)

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

    def _make_plot_core(self, truthses, scorses, kinase_identities: list[str], kinase_group, plotting_kwargs):
        plotting_kwargs.update({"plot_pointer_labels": True})

        self._roc_core_components(
            truthses,
            scorses,
            roc_labels_list=kinase_identities,
            plotting_kwargs=plotting_kwargs,
            title_extra=f" Stratified by Kinase (Kinase Group: {kinase_group}",
            roc_lab_extra=f" group-{kinase_group} kinase",
        )


class SplitIntoGroupsROC(ROC):
    def __init__(self, fig_kwargs={"figsize": (10, 10)}) -> None:
        super().__init__(fig_kwargs=fig_kwargs)

    def make_plot(self, evaluations, _ic_ref, plotting_kwargs):
        return super().make_plot(evaluations, _ic_ref, plotting_kwargs)

    @staticmethod
    def compute_test_evaluations(
        test_filename: str,
        multi_stage_classifier: MultiStageClassifier,
        resave_loc: str,
        kin_to_fam_to_grp_file: str,
        get_emp_eqn=False,
        device="cpu",
        cartesian_product=False,
        bypass_gc=False,
    ):
        assert multi_stage_classifier.__class__.__name__ == "MultiStageClassifier"
        kin_to_grp = pd.read_csv(kin_to_fam_to_grp_file)[["Kinase", "Uniprot", "Group"]]
        kin_to_grp["Symbol"] = (
            kin_to_grp["Kinase"].apply(lambda x: re.sub(r"[\(\)\*]", "", x)) + "|" + kin_to_grp["Uniprot"]
        )
        kin_to_grp = kin_to_grp.set_index("Symbol").to_dict()["Group"]
        gene_order = pd.read_csv(test_filename)["Gene Name of Kin Corring to Provided Sub Seq"]
        true_grps = [kin_to_grp[x] for x in gene_order]
        grp_to_interface = multi_stage_classifier.individual_classifiers.interfaces
        groups = list(grp_to_interface.keys())
        seen_groups = []
        all_predictions_outputs = {}
        info_dict_passthrough = {}
        pred_items = ()
        simulated_bypass_acc = 1
        random.seed(42)
        kin_to_grp_simulated_acc = {}
        for k, v in kin_to_grp.items():
            if random.random() < simulated_bypass_acc:
                kin_to_grp_simulated_acc[k] = v
            else:
                kin_to_grp_simulated_acc[k] = "CMGC"

        kwargs = {}
        if bypass_gc:
            kwargs["bypass_gc"] = kin_to_grp_simulated_acc
        for grp, loader in multi_stage_classifier.individual_classifiers.obtain_group_and_loader(
            which_groups=groups,
            Xy_formatted_input_file=test_filename,
            group_classifier=multi_stage_classifier.group_classifier,
            info_dict_passthrough=info_dict_passthrough,
            seen_groups_passthrough=seen_groups,
            device=torch.device(device),
            cartesian_product=cartesian_product,
            **kwargs,
        ):
            jumbled_predictions = multi_stage_classifier.individual_classifiers.interfaces[grp].predict(
                loader,
                int(info_dict_passthrough["on_chunk"] + 1),
                int(info_dict_passthrough["total_chunks"]),
                cutoff=0.5,
                group=grp,
            )  # TODO: Make adjustable cutoff
            # jumbled_predictions = list[predictions], list[output scores], list[group]
            del loader
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            new_info = info_dict_passthrough[grp]["PairIDs"]
            try:
                all_predictions_outputs.update(
                    {
                        pair_id: (
                            jumbled_predictions[0][i],
                            jumbled_predictions[1][i],
                            jumbled_predictions[2][i],
                            jumbled_predictions[3][i],
                            (
                                multi_stage_classifier.individual_classifiers.__dict__["grp_to_emp_eqn"].get(grp)
                                if get_emp_eqn
                                else None
                            ),
                        )
                        for pair_id, i in zip(new_info, range(len(new_info)))
                    }
                )
            except KeyError:
                raise AttributeError(
                    "`--convert-raw-to-prob` was set but the attribute `grp_to_emp_eqn` (aka a"
                    " dictionary that maps a kinase group to a function capible of converting raw"
                    " scores to empirical probabilities) was not found in the pretrained neural"
                    " network file. (To change the neural network file, use the `--pre-trained-nn`"
                    " command line option.)"
                ) from None
        key_lambda = lambda x: int(re.sub("Pair # ([0-9]+)", "\\1", x[0]))
        pred_items = sorted(all_predictions_outputs.items(), key=key_lambda)  # Pair # {i}
        pred_items = dict(
            ground_truth_labels=[x[1][3] for x in pred_items],
            scores=[x[1][1] for x in pred_items],
            ground_truth_groups=true_grps,
        )

        assert len(pred_items) != 0
        setattr(multi_stage_classifier, "completed_test_evaluations", pred_items)
        with open(resave_loc, "wb") as f:
            pickle.dump(multi_stage_classifier, f)

    def _make_plot_core(self, evaluations, _ic_ref, plotting_kwargs):
        get_emp_eqn = plotting_kwargs.get("get_emp_eqn", False)
        orig_order = ["CAMK", "AGC", "CMGC", "CK1", "OTHER", "TK", "TKL", "STE", "ATYPICAL", "<UNANNOTATED>"]
        groups = sorted(list(set(evaluations["ground_truth_groups"])))
        grp_to_emp_eqn = {}
        ground_truth_groups_to_outputs_labels = collections.defaultdict(lambda: collections.defaultdict(list))
        for gtl, o, gtg in zip(*evaluations.values()):
            ground_truth_groups_to_outputs_labels[gtg]["outputs"].append(o)
            ground_truth_groups_to_outputs_labels[gtg]["labels"].append(gtl)
        for grp in groups:
            outputs = ground_truth_groups_to_outputs_labels[grp]["outputs"]
            labels = ground_truth_groups_to_outputs_labels[grp]["labels"]
            agst = np.argsort(outputs)
            booled_labels = [bool(x) for x in labels]
            if get_emp_eqn:
                emp_prob_lambda = raw_score_to_probability(sorted(outputs), [booled_labels[i] for i in agst], **emp_eqn_kwargs)  # type: ignore
                grp_to_emp_eqn[grp] = emp_prob_lambda
            else:

                def emp_prob_lambda(l: list[Union[float, int]]):
                    raise AssertionError(
                        "For some reason, get_emp_eqn was set to False, but the emp_prob_lambda was called."
                    )

            # for i in range(len(true_group_to_results[grp])):
            #     true_grp_to_outputs_and_labels[true_group_to_results[grp][i]]["outputs"].append(outputs[i])
            #     true_grp_to_outputs_and_labels[true_group_to_results[grp][i]]["labels"].append(labels[i])
            #     true_grp_to_outputs_and_labels[true_group_to_results[grp][i]]["group_model_used"].append(grp)
            #     if get_emp_eqn:
            #         true_grp_to_outputs_and_labels[true_group_to_results[grp][i]]["outputs_emp_prob"].append(
            #             emp_prob_lambda([outputs[i]])[0]
            #         )

            assert len(outputs) == len(labels), (
                "Something is wrong in DeepKS_evaluation --- the length of outputs, the number of labels,"
                f" and the number of kinases in `kinase_order` are not equal. (Respectively, {len(outputs)},"
                f" {len(labels)}"
            )

        if get_emp_eqn:
            setattr(_ic_ref, "grp_to_emp_eqn", grp_to_emp_eqn)
            # Repickle
            if _ic_ref is not None:
                try:
                    with tempfile.NamedTemporaryFile() as tf:
                        _ic_ref.save_all(_ic_ref, tf.name)
                        print(colored("Info Re-saved Individual Classifiers with empirical equation.", "blue"))

                        with open(_ic_ref.repickle_loc.replace(".pkl", "_with_emp_eqn.pkl"), "wb") as f:
                            f.write(tf.read())
                except Exception as e:
                    warnings.warn(f"Couldn't repickle Individual Classifiers with empirical equation: {e}", UserWarning)
            else:
                warnings.warn(
                    "No repickle location found for Individual Classifiers. Not repickling; can't save empirical"
                    " equation.",
                    UserWarning,
                )

        if plotting_kwargs.get("plot_with_orig_order", False):
            groups = orig_order
        else:
            groups = sorted(groups)

        truthses = []
        scorses = []
        if get_emp_eqn:
            out_key = "outputs_emp_prob"
        else:
            out_key = "outputs"
        for group in groups:
            truths, scores = (
                ground_truth_groups_to_outputs_labels[group]["labels"],
                ground_truth_groups_to_outputs_labels[group][out_key],
            )
            truthses.append(truths)
            scorses.append(scores)

        self._roc_core_components(truthses, scorses, groups, plotting_kwargs)


class PlainROC(ROC):
    def __init__(self, fig_kwargs={"figsize": (5, 5)}, pass_in_ax=None) -> None:
        super().__init__(fig_kwargs=fig_kwargs, pass_in_ax=pass_in_ax)

    def make_plot_multiple(
        self,
        outputs_multiple: list[list[float]],
        labels_multiple: list[list[int]],
        roc_line_labels: list[str] = [],
        plotting_kwargs={"fancy_legend": False},
    ):
        if roc_line_labels == []:
            roc_line_labels = [f"Test Set {i}" for i in range(1, 1 + len(outputs_multiple))]
        assert len(outputs_multiple) == len(labels_multiple) == len(roc_line_labels)
        args = [outputs_multiple, labels_multiple, roc_line_labels, plotting_kwargs]
        super().make_plot(*args)

    def make_plot(self, outputs: list[float], labels: list[int], plotting_kwargs={"fancy_legend": False}):
        args = [[outputs], [labels], [], plotting_kwargs]
        super().make_plot(*args)

    def _make_plot_core(self, scoreses, truthses, roc_line_labels, plotting_kwargs):
        ret = self._roc_core_components(
            truthses, scoreses, roc_labels_list=roc_line_labels, plotting_kwargs=plotting_kwargs
        )
        self._is_made = True
        return ret

    def compute_test_evaluations(
        self,
        test_filename: str,
        multi_stage_classifier: MultiStageClassifier,
        resave_loc,
        device="cpu",
        cartesian_product=False,
    ):
        grp_to_interface = multi_stage_classifier.individual_classifiers.interfaces
        groups = list(grp_to_interface.keys())
        seen_groups = []
        all_predictions_outputs = {}
        input_df = None
        info_dict_passthrough = {}

        kwargs = {}
        for grp, loader in multi_stage_classifier.individual_classifiers.obtain_group_and_loader(
            which_groups=groups,
            Xy_formatted_input_file=test_filename,
            group_classifier=multi_stage_classifier.group_classifier,
            info_dict_passthrough=info_dict_passthrough,
            seen_groups_passthrough=seen_groups,
            device=torch.device(device),
            cartesian_product=cartesian_product,
            **kwargs,
        ):
            jumbled_predictions = multi_stage_classifier.individual_classifiers.interfaces[grp].predict(
                loader,
                int(info_dict_passthrough["on_chunk"] + 1),
                int(info_dict_passthrough["total_chunks"]),
                group=grp,
            )
            # del loader

            new_info = info_dict_passthrough[grp]["PairIDs"]
            try:
                partial_orig_data = pd.DataFrame.from_dict(info_dict_passthrough[grp]["original_data"])
                if input_df is None:
                    input_df = partial_orig_data
                else:
                    input_df = pd.concat([input_df, partial_orig_data], axis="rows")
                all_predictions_outputs.update(
                    {
                        pair_id: tuple([jumbled_predictions[j][i] for j in range(len(jumbled_predictions))])
                        for pair_id, i in zip(new_info, range(len(new_info)))
                    }
                )
            except KeyError:
                raise AttributeError(
                    "`--convert-raw-to-prob` was set but the attribute `grp_to_emp_eqn` (aka a"
                    " dictionary that maps a kinase group to a function capible of converting raw"
                    " scores to empirical probabilities) was not found in the pretrained neural"
                    " network file. (To change the neural network file, use the `--pre-trained-nn`"
                    " command line option.)"
                ) from None

        key_lambda = lambda x: int(re.sub("Pair # ([0-9]+)", "\\1", x))
        key_lambda_vec = lambda X: [int(re.sub("Pair # ([0-9]+)", "\\1", x)) for x in X]

        all_predictions_outputs_sorted = sorted(list(all_predictions_outputs.items()), key=lambda X: key_lambda(X[0]))
        scores = [x[1][1] for x in all_predictions_outputs_sorted]
        truths = [x[1][3] for x in all_predictions_outputs_sorted]

        assert isinstance(input_df, pd.DataFrame)
        input_df.sort_values(by="pair_id", key=key_lambda_vec, inplace=True)
        input_df["DeepKS-Score"] = scores
        assert truths == input_df["Class"].values.tolist()

        setattr(multi_stage_classifier, "completed_test_evaluations", pred_items)
        setattr(multi_stage_classifier, "input_df", input_df)

        with open(resave_loc, "wb") as f:
            pickle.dump(multi_stage_classifier, f)


# def calibrate(truths, scores):
#     sklearn.


def eval_and_roc_workflow(multi_stage_classifier, test_filename, resave_loc, force_recompute=False):
    roc = PlainROC()
    if hasattr(multi_stage_classifier, "completed_test_evaluations") and not force_recompute:
        logger.info("Using pre-computed test evaluations.")
    else:
        logger.info("Computing test evaluations.")
        PlainROC.compute_test_evaluations(roc, test_filename, multi_stage_classifier, resave_loc)

    test_evaluations = multi_stage_classifier.completed_test_evaluations
    original_data = multi_stage_classifier.original_data
    original_data_df = multi_stage_classifier.original_data_df
    org_pred_data = organize_pred_items(test_evaluations, original_data, by="Group")
    # org_pred_data = sorted(list(org_pred_data.items()), key = lambda x: -len(x[1][0]))
    # pd.DataFrame(list(zip(*org_pred_data[0][1])), columns = ["truths", "scores"]).to_csv("~/ts.csv")

    scores_multiple = [org_pred_data[grp][1] for grp in org_pred_data]
    labels_multiple = [[int(x) for x in org_pred_data[grp][0]] for grp in org_pred_data]

    # roc.make_plot_multiple(
    #     scores_multiple,
    #     labels_multiple,
    #     roc_line_labels=[f"Test Set (Group = {group})" for group in org_pred_data],
    #     plotting_kwargs={"fancy_legend": True},
    # )
    # roc.save_plot(join_first(1, "images/EvaluationandResults/"))

    org_pred_data_by_kinase = organize_pred_items(test_evaluations, original_data, by="Kinase Sequence")
    original_data = pd.DataFrame.from_dict(original_data)

    orig_data_dict = original_data[original_data["Class"] == 1].to_dict(orient="list")
    kin_to_grp = dict(zip(orig_data_dict["Kinase Sequence"], orig_data_dict["Group"]))
    kin_to_name_token = dict(zip(orig_data_dict["Kinase Sequence"], orig_data_dict["Gene Name of Provided Kin Seq"]))
    kin_to_uniprot = {k: re.sub(r"^.*\|(.*)", "\\1", v) for k, v in kin_to_name_token.items()}
    psp_avail_kins = set(kin_to_uniprot.values())
    atlas_avail_kins = set(
        pd.read_excel("/Users/druc594/Downloads/41586_2022_5575_MOESM3_ESM.xlsx", sheet_name="Table S1 Data")[
            "Uniprot id"
        ]
    )
    mutually_avail_kins = set.intersection(psp_avail_kins, atlas_avail_kins)
    org_pred_data_by_uniprot = {re.sub(r"^.*\|(.*)", "\\1", k): v for k, v in org_pred_data_by_kinase.items()}
    org_pred_data_by_kinase = {
        kin_to_uniprot[k]: v
        for k, v in org_pred_data_by_uniprot.items()
        if kin_to_uniprot.get(k) in mutually_avail_kins and kin_to_grp.get(k) == "NON-TK"
    }
    # roc_map3k14 = PlainROC(fig_kwargs={"figsize": (8, 8)})
    # scores_map3k14 = org_pred_data_by_kinase["MAVMEMACPGAPGSAVGQQKELPKAKEKTPPLGKKQSSVYKLEAVEKSPVFCGKWEILNDVITKGTAKEGSEAGPAAISIIAQAECENSQEFSPTFSERIFIAGSKQYSQSESLDQIPNNVAHATEGKMARVCWKGKRRSKARKKRKKKSSKSLAHAGVALAKPLPRTPEQESCTIPVQEDESPLGAPYVRNTPQFTKPLKEPGLGQLCFKQLGEGLRPALPRSELHKLISPLQCLNHVWKLHHPQDGGPLPLPTHPFPYSRLPHPFPFHPLQPWKPHPLESFLGKLACVDSQKPLPDPHLSKLACVDSPKPLPGPHLEPSCLSRGAHEKFSVEEYLVHALQGSVSSGQAHSLTSLAKTWAARGSRSREPSPKTEDNEGVLLTEKLKPVDYEYREEVHWATHQLRLGRGSFGEVHRMEDKQTGFQCAVKKVRLEVFRAEELMACAGLTSPRIVPLYGAVREGPWVNIFMELLEGGSLGQLVKEQGCLPEDRALYYLGQALEGLEYLHSRRILHGDVKADNVLLSSDGSHAALCDFGHAVCLQPDGLGKSLLTGDYIPGTETHMAPEVVLGRSCDAKVDVWSSCCMMLHMLNGCHPWTQFFRGPLCLKIASEPPPVREIPPSCAPLTAQAIQEGLRKEPIHRVSAAELGGKVNRALQQVGGLKSPWRGEYKEPRHPPPNQANYHQTLHAQPRELSPRAPGPRPAEETTGRAPKLQPPLPPEPPEPNKSPPLTLSKEESGMWEPLPLSSLEPAPARNPSSPERKATVPEQELQQLEIELFLNSLSQPFSLEEQEQILSCLSIDSLSLSDDSEKNPSKASQSSRDTLSSGVHSWSSQAEARSSSWNMVLARGRPTDTPSYFNGVKVQIQSLNGEHLHIREFHRVKVGDIATGISSQIPAAAFSLVTKDGQPVRYDMEVPDSGIDLQCTLAPDGSFAWSWRVKHGQLENRP"][1]
    # labels_map3k14 = org_pred_data_by_kinase["MAVMEMACPGAPGSAVGQQKELPKAKEKTPPLGKKQSSVYKLEAVEKSPVFCGKWEILNDVITKGTAKEGSEAGPAAISIIAQAECENSQEFSPTFSERIFIAGSKQYSQSESLDQIPNNVAHATEGKMARVCWKGKRRSKARKKRKKKSSKSLAHAGVALAKPLPRTPEQESCTIPVQEDESPLGAPYVRNTPQFTKPLKEPGLGQLCFKQLGEGLRPALPRSELHKLISPLQCLNHVWKLHHPQDGGPLPLPTHPFPYSRLPHPFPFHPLQPWKPHPLESFLGKLACVDSQKPLPDPHLSKLACVDSPKPLPGPHLEPSCLSRGAHEKFSVEEYLVHALQGSVSSGQAHSLTSLAKTWAARGSRSREPSPKTEDNEGVLLTEKLKPVDYEYREEVHWATHQLRLGRGSFGEVHRMEDKQTGFQCAVKKVRLEVFRAEELMACAGLTSPRIVPLYGAVREGPWVNIFMELLEGGSLGQLVKEQGCLPEDRALYYLGQALEGLEYLHSRRILHGDVKADNVLLSSDGSHAALCDFGHAVCLQPDGLGKSLLTGDYIPGTETHMAPEVVLGRSCDAKVDVWSSCCMMLHMLNGCHPWTQFFRGPLCLKIASEPPPVREIPPSCAPLTAQAIQEGLRKEPIHRVSAAELGGKVNRALQQVGGLKSPWRGEYKEPRHPPPNQANYHQTLHAQPRELSPRAPGPRPAEETTGRAPKLQPPLPPEPPEPNKSPPLTLSKEESGMWEPLPLSSLEPAPARNPSSPERKATVPEQELQQLEIELFLNSLSQPFSLEEQEQILSCLSIDSLSLSDDSEKNPSKASQSSRDTLSSGVHSWSSQAEARSSSWNMVLARGRPTDTPSYFNGVKVQIQSLNGEHLHIREFHRVKVGDIATGISSQIPAAAFSLVTKDGQPVRYDMEVPDSGIDLQCTLAPDGSFAWSWRVKHGQLENRP"][0]
    # roc_map3k14.make_plot_multiple([scores_map3k14], [[int(x) for x in labels_map3k14]], roc_line_labels=["MAP3K14_HUMAN (Test Set)"], plotting_kwargs={"fancy_legend": True})
    # roc_map3k14.save_plot("map3k14")
    # with open(os.path.expanduser("~/by_kinase_perf.csv"), "w") as fp:
    df = SqrtPerfScorer.get_perf(org_pred_data_by_kinase)
    #     df.to_csv(fp)
    return df, org_pred_data_by_uniprot, original_data_df

    # roc.make_plot(
    #     multi_stage_classifier.completed_test_evaluations,
    #     multi_stage_classifier.individual_classifiers,
    #     plotting_kwargs={
    #         "plot_pointer_labels": False,
    #         "legend_all_lines": True,
    #         "plot_unified_line": True,
    #         "diff_by_opacity": False,
    #         "focus_on": None,
    #         "plot_mean_value_line": True,
    #         "plot_unified_line": True,
    #     },
    # )
    #


def test():
    test_dat = pd.read_csv(join_first(1, "tools/dff_sample_test_data.csv"))
    the_roc = SplitIntoKinasesROC()
    the_roc.make_plot(
        test_dat["Score"],
        test_dat["Class"],
        test_dat["Kin"].tolist(),
        "TESTGRP",
        plotting_kwargs={
            "plot_pointer_labels": False,
            "legend_all_lines": True,
            "plot_unified_line": True,
            "diff_by_opacity": False,
            "focus_on": "Kin-B",
            # "plot_mean_value_line": False,
        },
    )
    the_roc.display_plot()
    # the_roc.save_plot("test_roc.pdf", save_fig_kwargs={"bbox_inches": "tight", "dpi": 300})


def get_multi_stage_classifier():
    if os.path.exists("roc-cache.json"):
        with open("roc-cache.json", "r") as f:
            cache: dict = json.load(f)
            del cache["plotting_kwargs"]
            roc = SplitIntoKinasesROC(fig_kwargs={"figsize": (10, 10)})
            roc.make_plot(
                plotting_kwargs={
                    "plot_pointer_labels": False,
                    "legend_all_lines": True,
                    "plot_unified_line": True,
                    "diff_by_opacity": False,
                    "focus_on": "Kin-B",
                },
                **cache,
            )
            roc.display_plot()
            # roc.save_plot("roc_tk.pdf", save_fig_kwargs={"bbox_inches": "tight", "dpi": 300})
    else:
        raise FileNotFoundError("No cache found")


def organize_pred_items(
    pred_items: dict[str, list[float]],
    input_data: dict[Hashable, list],
    by: Literal["Site Sequence", "Kinase Sequence", "Group"] = "Site Sequence",
) -> dict[str, tuple[list[float], list[float]]]:
    """Obtains a mapping from desired ``by`` specification to a tuple of truths, scores for that specifications

    Parameters
    ----------
    pred_items :
        Dictionary with keys 'ground_truth_labels' and 'scores', each of which maps to a list of such ``float``s. The order must correspond to that in ``input_data``
    input_data :
        A dictionary with keys 'Site Sequence', 'Kinase Sequence', 'Group', and 'Class', corresponding to the results in ``pred_items``
    by :
        The column (specification) by which to organize the ``pred_items``

    Returns
    -------
        Mapping from each ``by`` to a tuple of (truths, scores)
    """
    scores = pred_items["scores"]
    truths = pred_items["ground_truth_labels"]
    spec_to_indexes: dict[str, list[int]] = collections.defaultdict(list[int])
    for i, s in enumerate(input_data[by]):
        spec_to_indexes[s].append(i)
    spec_to_results: dict[str, tuple[list[float], list[float]]] = {
        s: ([truths[li] for li in l], [scores[li] for li in l]) for s, l in spec_to_indexes.items()
    }
    return spec_to_results


class PerfScorer(ABC):
    @staticmethod
    @abstractmethod
    def _get_perf_core(metric_score: float, num_obs: int) -> float: ...

    @staticmethod
    @abstractmethod
    def get_perf(org_pred_items: dict[str, list[float]]) -> dict[str, float]: ...


class SqrtPerfScorer(PerfScorer):
    @staticmethod
    def step_fn(x) -> float:
        if x < 30:
            return 3.0
        elif x < 60:
            return 6.0
        elif x < 120:
            return 7.0
        elif x < 240:
            return 8.0
        else:
            return 9.0

    @staticmethod
    def _get_perf_core(metric_score: float | float, num_obs: int) -> float:
        return (metric_score**0.5) * SqrtPerfScorer.step_fn(num_obs)

    @staticmethod
    def get_perf(
        org_pred_items: dict[str, tuple[list[float], list[float]]],
        base_score_metric: Callable[[list[float | int], list[float | int]], float] = lambda *args: float(
            sklearn.metrics.roc_auc_score(*args)
        ),
    ) -> pd.DataFrame:
        perf_table = pd.DataFrame(columns=["Orged by Item", "Metric Score", "Num Obs", "Perf"])
        for orged_by in org_pred_items:
            truths, scores = org_pred_items[orged_by][0], org_pred_items[orged_by][1]
            if len(set(truths)) == 1:
                continue
            metric_score = base_score_metric(truths, scores)
            num_obs = len(org_pred_items[orged_by][0])
            perf = SqrtPerfScorer._get_perf_core(metric_score, num_obs)
            perf_table.loc[len(perf_table), :] = [orged_by, metric_score, num_obs, perf]
        perf_table = perf_table.sort_values(by="Perf", ascending=[False], inplace=False).reset_index(drop=True)
        return perf_table


def main(test_filename, force_recompute: bool):
    with open(
        join_first(1, "bin/deepks_msc_weights.resaved.3.cornichon"),
        "rb",
    ) as mscfp:
        msc = pickle.load(mscfp)
        df, d, df2 = eval_and_roc_workflow(
            msc,
            join_first(1, test_filename),
            join_first(1, "bin/deepks_msc_weights.resaved.3.cornichon"),
            force_recompute=force_recompute,
        )
        df.rename(columns={"Orged by Item": "Kinase"}, inplace=True)
        df = df[[c for c in df.columns if c != "Perf"]].sort_values(by="Metric Score", inplace=False, ascending=False)
        return df, d, df2


if __name__ == "__main__":  # pragma: no cover
    pass
    # main()
    # import cloudpickle

    # # main()
    # # test()

    # fake_scores = [max(min(x / 100 - 0.5 + random.random(), 1), 0) for x in range(100)]
    # fake_truths = [0 for _ in range(50)] + [1 for _ in range(50)]

    # roc = PlainROC()
    # roc.make_plot(fake_scores, fake_truths)
    # roc.display_plot()
