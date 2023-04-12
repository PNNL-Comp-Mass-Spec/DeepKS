import itertools, sklearn.metrics, numpy as np, warnings, scipy
from numpy.typing import ArrayLike
from roc_comparison_modified.auc_delong import delong_roc_variance


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
