import warnings, matplotlib, numpy as np, pandas as pd, json, traceback, sys, itertools, scipy, scipy.special, os
import sklearn, sklearn.metrics
import random

random.seed(42)
from abc import ABC, abstractmethod
from roc_comparison_modified.auc_delong import delong_roc_variance
from matplotlib import pyplot as plt, patches as ptch, collections as clct
from matplotlib import colormaps  # type: ignore

get_cmap = colormaps.get_cmap
from pydash import _
from numpy.typing import ArrayLike


warnings.simplefilter("once", Warning)


def warning_handler(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    if log is not None:
        log.write(warnings.formatwarning(message, category, filename, lineno, line=""))
    else:
        print("log is None")


warnings.showwarning = warning_handler


class ROCHelpers:
    @staticmethod
    def get_avg_roc(fprs, tprs, aucs=None):
        total_n = len(list(itertools.chain(*fprs))) + 1
        X = np.linspace(0, 1, num=total_n, endpoint=False).tolist()
        X = [X[i // 2] for i in range(len(X) * 2)]
        Y = [ROCHelpers.get_tpr(fprs, tprs, x) for x in X]
        micro_avg = 0
        if aucs is not None:
            for i, a in enumerate(aucs):
                micro_avg += a * len(fprs[i]) / total_n

        return X, Y, micro_avg

    @staticmethod
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
        assert 0 <= running_sum <= 1 or np.isnan(
            running_sum
        ), "Avg. ROC cannot be lower than 0 or higher than 1. Alternatively, it must be NaN."
        return running_sum

    @staticmethod
    def protected_roc_auc_score(y_true: ArrayLike, y_score: ArrayLike, *args, **kwargs) -> float:
        """Wrapper for sklearn.metrics.roc_auc_score that handles edge cases

        Args:
            @arg y_true: Iterable of (integer) true labels
            @arg y_score: Iterable of predicted scores
            @arg *args: Additional arguments to pass to roc_auc_score
            @arg **kwargs: Additional keyword arguments to pass to roc_auc_score

        Raises:
            e: Error that is not a multi class error or single class present error

        Returns:
            float: The roc_auc_score
        """
        try:
            return float(sklearn.metrics.roc_auc_score(y_true, y_score, *args, **kwargs))
        except Exception as e:
            match str(e):
                case "Only one class present in y_true. ROC AUC score is not defined in that case.":
                    warnings.warn(f"Setting roc_auc_score to 0.0 since there is only one class present in y_true.")
                    return 0.0
                case "multi_class must be in ('ovo', 'ovr')":
                    # softmax in case of multi-class
                    y_score = np.asarray(scipy.special.softmax(y_score, 1))
                    return ROCHelpers.protected_roc_auc_score(
                        y_true,
                        y_score,
                        *args,
                        **(
                            {"multi_class": "ovo", "average": "macro", "labels": list(range(y_score.shape[-1]))}
                            | kwargs
                        ),
                    )
                case _:
                    raise e

    @staticmethod
    def auc_confidence_interval(truths, pred_scores, alpha=0.05) -> tuple[tuple[float, float], float]:
        """Based off of https://gist.github.com/RaulSanchezVazquez/d338c271ace3e218d83e3cb6400a769c"""
        auc, auc_covariance = delong_roc_variance(truths, pred_scores)
        # auc, auc_covariance = 0.8, (10**random.uniform(-1, 0))**2  # For testing purposes only
        auc_std = auc_covariance**0.5
        quant = -scipy.stats.norm.ppf(alpha / 2)
        half_width = quant * auc_std
        ci = (max(0, auc - half_width), min(1, auc + half_width))
        p_val = 2 - 2 * scipy.stats.norm.cdf(abs(auc - 0.5) / auc_std)
        return ci, p_val

    @staticmethod
    def get_p_stars(p_val):
        if np.isnan(p_val):
            return "?"
        assert p_val >= 0 and p_val <= 1, p_val
        if p_val < 0.001:
            return "***"
        elif p_val < 0.01:
            return "**"
        elif p_val < 0.05:
            return "*"
        if p_val < 0.1:
            return "."
        else:
            return ""


class PerformancePlot(ABC):  # ABC = Abstract Base Class
    def __init__(self, fig_kwargs={"figsize": (10, 10)}) -> None:
        self._is_made = False
        matplotlib.rcParams["font.family"] = "P052"
        self.fig, self.ax = plt.subplots(**fig_kwargs)
        clsname = self.__class__.__name__
        self._not_made_error_msg = (
            f"{clsname} Plot is not made yet. Ignoring Request. Call `my_{_.snake_case(clsname)}_instance.make_plot()`"
            " first."
        )

    def make_plot(self, scores, labels, *args, **kwargs):
        self._make_plot_core(scores, labels, *args, **kwargs)
        self._is_made = True

    def display_plot(self):
        if self._is_made:
            prev_ax = plt.gca()
            plt.sca(self.ax)
            plt.show()
            plt.sca(prev_ax)
        else:
            warnings.warn(self._not_made_error_msg)

    def save_plot(self, save_name, save_fig_kwargs={"bbox_inches": "tight"}):
        if self._is_made:
            self.fig.savefig(save_name, **save_fig_kwargs)
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
    def __init__(self, **fig_kwargs) -> None:
        super().__init__(**fig_kwargs)

    def _roc_core_components(
        self,
        fprs,
        tprs,
        thresholds,
        datas,
        roc_labels_list=[],
        plotting_kwargs={},
        title_extra="",
        roc_lab_extra="",
    ):
        jitter = lambda x: (
            x if float(jitter_amount) == 0.0 else [xi + jitter_amount * random.random() - jitter_amount / 2 for xi in x]
        )
        jitter_amount = plotting_kwargs.get("jitter_amount", 0.0)
        diff_by_opacity = plotting_kwargs.get("diff_by_opacity", False)
        show_zones = plotting_kwargs.get("show_zones", False)
        show_cutoff = plotting_kwargs.get("show_cutoff", False)
        show_acc = plotting_kwargs.get("show_acc", False)
        focus_on = plotting_kwargs.get("focus_on", None)
        legend_all_lines = plotting_kwargs.get("legend_all_lines", False)
        plot_pointer_labels = plotting_kwargs.get("plot_pointer_labels", False)
        fancy_legend = plotting_kwargs.get("fancy_legend", True)
        plot_mean_value_line = plotting_kwargs.get("plot_mean_value_line", False)

        roc_colors = plotting_kwargs.get("roc_colors", "plasma")
        cut_colors = plotting_kwargs.get("cut_colors", "plasma")
        acc_colors = plotting_kwargs.get("acc_colors", "plasma")

        cmap_roc, cmap_cut, cmap_acc = get_cmap(roc_colors), get_cmap(cut_colors), get_cmap(acc_colors)

        color_places = np.linspace(0, 1, len(fprs), endpoint=True)
        convert = lambda ls: [matplotlib.colors.to_hex(c) for c in ls]
        roc_colors = convert(cmap_roc(color_places))
        cut_colors = convert(cmap_cut(color_places))
        acc_colors = convert(cmap_acc(color_places))

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
        roc_col = roc_colors[0]
        cut_col = acc_colors[0]
        acc_col = cut_colors[0]

        fig, ax = self.fig, self.ax
        ax.set_xticks(np.arange(0, 1.05, 0.05))
        ax.set_xticklabels([f"{_:.2f}" for _ in np.arange(0, 1.05, 0.05)], rotation=90)
        ax.set_yticks(np.arange(0, 1.05, 0.05))
        ax.set_title(f"ROC Curves{title_extra})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        if show_cutoff:
            tax = ax.twinx()
            tax.set_ylabel("Cutoff")
            tax.set_ylim(0, 1.1)
            tax.yaxis.label.set_color(cut_col)
            tax.tick_params(axis="y", colors=cut_col)
        aucs = []
        pct_mult = 2 if len(datas[-1]) == sum([len(d) for d in datas[:-1]]) else 1
        sd = sum([len(d) for d in datas])
        indent_amt = max(
            [
                len(r)
                for r in roc_labels_list
                + [
                    "Random",
                    "Mean Value" if plot_mean_value_line else "",
                    "Emph. Curve" if focus_on is not None else "",
                ]
            ]
        )
        for i, (fpr, tpr, thrr, data) in enumerate(zip(fprs, tprs, thresholds, datas)):
            mar = f"{roc_labels_list[i]}" if roc_labels_list[i] != "All Data" else None
            col = roc_colors[i]
            alp = 0.65 if not diff_by_opacity else max(0.05, min(1, 1 / len(fprs)))
            unif = False
            is_focus = False
            linew = 1
            if roc_labels_list[i] == "All Data":
                mar, col, alp, unif, linew = None, "black", 1, True, 2
            if focus_on is not None and roc_labels_list[i] == focus_on:
                linew, is_focus = 1, True

            roc_style_kwargs = dict(linewidth=linew, markersize=7, color=col, alpha=alp, zorder=2)
            if i == 0 and not legend_all_lines:
                roc_style_kwargs.update(dict(label=f"ROC curve of a{roc_lab_extra}"))
            elif unif:
                roc_style_kwargs.update(dict(label=f"ROC curve of all{roc_lab_extra}s"))
            elif legend_all_lines:
                roc_style_kwargs.update(dict(label=roc_labels_list[i]))

            truths, scores = data["Class"].values, data["Score"].values
            aucscore = sklearn.metrics.roc_auc_score(truths, scores)
            if fancy_legend:
                alpha = 0.05
                ci, p = ROCHelpers.auc_confidence_interval(
                    truths, scores, alpha_level := plotting_kwargs.get("alpha_level", 0.05)
                )
                is_signif = p < alpha_level

                pct_mult = 2 if len(datas[-1]) == sum([len(d) for d in datas[:-1]]) else 1
                fancy_lab = (
                    f"{roc_labels_list[i]:>{indent_amt}}"
                    f" | AUC {aucscore:1.3f} — {100-int(alpha*100)}% CI = [{ci[0]:1.3f}, {ci[1]:1.3f}]"
                    f" {ROCHelpers.get_p_stars(p)} | n = {len(data):5} = {len(data)*100/sd*pct_mult:6.2f}%"
                    " all data"
                )
                roc_style_kwargs.update(dict(label=fancy_lab))

                if not is_signif:
                    roc_style_kwargs.update(dict(alpha=0.3, linestyle="dotted", dash_capstyle="round"))

            ax.plot(fpr, tpr, **roc_style_kwargs)
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
                (focused,) = ax.plot(fpr, tpr, **roc_style_kwargs)
                focused.set_dashes([3, 1.5, 2, 1.5])  # type: ignore
                focused.set_dash_capstyle("round")
            aucs.append(aucscore)
            if plot_pointer_labels and mar is not None:
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
                            # **dict(label="Kinase|Uniprot Class") if i == 0 else {},
                        )

            if show_acc:
                accs = []
                for thr in thrr:
                    targ_labs = ["Target", 1]
                    decoy_labs = ["Decoy", 0]
                    correctly_predicted_actual_pos = len(data[(data["Class"].isin(targ_labs)) & (data["Score"] >= thr)])
                    correctly_predicted_actual_neg = len(data[(data["Class"].isin(decoy_labs)) & (data["Score"] < thr)])
                    acc = (correctly_predicted_actual_pos + correctly_predicted_actual_neg) / len(data)
                    accs.append(acc)

                taxacc = ax.twinx()
                taxacc.spines.right.set_position(("axes", 1.15))
                taxacc.set_ylabel("Accuracy")
                taxacc.yaxis.label.set_color(acc_col)
                taxacc.tick_params(axis="y", colors=acc_col)
                taxacc.set_ylim(-0.05, 1.05)

                taxacc.plot(
                    jitter(fpr),
                    accs,
                    label="accuracy",
                    marker=None,
                    markersize=6,
                    color=acc_colors[i],
                    linewidth=linew / 3,
                    markeredgewidth=0,
                    alpha=alp / 2 if not unif else 1,
                )

                if show_zones:
                    pat = ptch.Rectangle((fpr[np.argmax(accs)] - 0.01, -1), 0.02, 3, color="#f0f0f030", linewidth=0)
                    pat2 = ptch.Rectangle(
                        (-1, np.max(accs) - 0.01), 3, 0.02, color="#f0f0f030", linewidth=0, label="Best Accuracy"
                    )
                    collect = clct.PatchCollection([pat, pat2], match_original=True, zorder=-10)
                    ax.add_collection(collect)  # type: ignore
            if show_cutoff:
                tax = ax.twinx()
                tax.set_ylabel("Cutoff")
                tax.set_ylim(0, 1.1)
                tax.yaxis.label.set_color(cut_col)
                tax.tick_params(axis="y", colors=cut_col)
                tax.plot(
                    jitter(fpr),
                    thrr,
                    color=cut_colors[i],
                    label="cutoff",
                    marker=None,
                    markersize=6,
                    linewidth=linew / 3,
                    markeredgewidth=0,
                    alpha=alp / 2 if not unif else 1,
                )
                tax.set_ylim(-0.05, 1.05)

        if plot_mean_value_line:
            X, Y, aucscore = ROCHelpers.get_avg_roc(fprs, tprs, aucs)
            lab = f"{'Mean Value':>{indent_amt}} | AUC {aucscore:1.3f}"
            ax.plot(X, Y, color="grey", linewidth=2, label=lab)

        random_model_label = (
            f"{'Random':>{indent_amt}}{'' if not fancy_legend else ' | AUC 0.5    ' + ' ' * 29 + '| n =     ∞'}"
        )
        ax.plot([0, 1], [0, 1], color="black", lw=0.4, linestyle="--", label=random_model_label)

        ax.yaxis.label.set_color(roc_col)
        ax.tick_params(axis="y", colors=roc_col)

        ax.legend(loc="best", prop={"family": ["Fira Code", "monospace"], "size": 5})
        ax.set_ylim(-0.05, 1.05)
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

    def _make_plot_core(self, scores, labels, kinase_identities: list[str], kinase_group, plotting_kwargs):
        assert len(scores) == len(labels) == len(kinase_identities), "Lengths of lists must be equal"

        plot_unified_line = plotting_kwargs.get("plot_unified_line", False)

        all_datas = pd.DataFrame(
            {"Score": scores, "Class": [bool(x) for x in labels], "Kinase Class": kinase_identities}
        )

        datas = [df for _, df in all_datas.groupby("Kinase Class")]

        fprs, tprs, thresholds = [], [], []

        for data in datas:
            if set(data["Kinase Class"]) == {"ABL1|P00519"}:
                data.to_csv("sanity_check_abl1.csv", index=False)
            fpr, tpr, threshold = sklearn.metrics.roc_curve(data["Class"], data["Score"])
            fprs.append(fpr)
            tprs.append(tpr)
            thresholds.append(threshold)

        roc_labels_list = [list(set(data["Kinase Class"]))[0] for data in datas]
        if plot_unified_line:
            unified_fpr, unified_tpr, unified_threshold = sklearn.metrics.roc_curve(
                all_datas["Class"], all_datas["Score"]
            )
            fprs.append(unified_fpr)
            tprs.append(unified_tpr)
            thresholds.append(unified_threshold)
            datas.append(all_datas)
            roc_labels_list.append("All Data")

        self._roc_core_components(
            fprs,
            tprs,
            thresholds,
            datas,
            roc_labels_list=roc_labels_list,
            plotting_kwargs=plotting_kwargs,
            title_extra=f" Stratified by Kinase (Kinase Group: {kinase_group}",
            roc_lab_extra=f" group-{kinase_group} kinase",
        )

        self.cache_prepared_data(scores, labels, kinase_identities, kinase_group, plotting_kwargs)


class SplitIntoGroupsROC(ROC):
    def __init__(self, fig_kwargs={"figsize": (10, 10)}) -> None:
        super().__init__(fig_kwargs=fig_kwargs)

    def make_roc(self, scores, labels):
        return super().make_plot(scores, labels)

    def _make_plot_core(self, *args, **kwargs):
        assert grp_to_loader_true_groups is not None
        if from_loaded is None:
            grp_to_interface = {k: grp_to_interface[k] for k in grp_to_loader}
            assert grp_to_interface.keys() == grp_to_loader.keys(), (
                "The groups for the provided models are not equal to the groups for the provided loaders."
                f" Respectively, {grp_to_interface.keys()} != {grp_to_loader.keys()}"
            )
        orig_order = ["CAMK", "AGC", "CMGC", "CK1", "OTHER", "TK", "TKL", "STE", "ATYPICAL", "<UNANNOTATED>"]
        groups = grp_to_interface.keys() if from_loaded is None else from_loaded.keys()
        true_grp_to_outputs_and_labels = collections.defaultdict(lambda: collections.defaultdict(list))
        grp_to_emp_eqn = {}
        for grp in groups:
            if from_loaded is None:
                interface = grp_to_interface[grp]
                loader = grp_to_loader[grp]
                _, _, outputs, labels, _ = interface.eval(
                    loader, cutoff=0.5, metric="roc", predict_mode=False, display_pb=False
                )
                if retain_evals is not None:
                    # Group -> Tr/Vl/Te -> outputs/labels -> list
                    retain_evals.update({grp: {"test": {"outputs": outputs, "labels": labels}}})
            else:
                outputs = from_loaded[grp]["test"]["outputs"]
                labels = from_loaded[grp]["test"]["labels"]
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

            for i in range(len(grp_to_loader_true_groups[grp])):
                true_grp_to_outputs_and_labels[grp_to_loader_true_groups[grp][i]]["outputs"].append(outputs[i])
                true_grp_to_outputs_and_labels[grp_to_loader_true_groups[grp][i]]["labels"].append(labels[i])
                true_grp_to_outputs_and_labels[grp_to_loader_true_groups[grp][i]]["group_model_used"].append(grp)
                if get_emp_eqn:
                    true_grp_to_outputs_and_labels[grp_to_loader_true_groups[grp][i]]["outputs_emp_prob"].append(
                        emp_prob_lambda([outputs[i]])[0]
                    )

            assert len(outputs) == len(labels), (
                "Something is wrong in NNInterface.get_all_rocs_by_group; the length of outputs, the number of labels,"
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
                    (
                        "No repickle location found for Individual Classifiers. Not repickling; can't save empirical"
                        " equation."
                    ),
                    UserWarning,
                )
        for group in set(all_obs := list(itertools.chain(*[list(x) for x in grp_to_loader_true_groups.values()]))):
            outputs, labels = (
                true_grp_to_outputs_and_labels[group]["outputs_emp_prob" if get_emp_eqn else "outputs"],
                true_grp_to_outputs_and_labels[group]["labels"],
            )
            orig_i = orig_order.index(group)


def test():
    test_dat = pd.read_csv(
        "/Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/tools/dff_sample_test_data.csv"
    )
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
            "plot_mean_value_line": False,
        },
    )
    the_roc.display_plot()
    # the_roc.save_plot("test_roc.pdf", save_fig_kwargs={"bbox_inches": "tight", "dpi": 300})


def main():
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


if __name__ == "__main__":
    # main()
    test()
