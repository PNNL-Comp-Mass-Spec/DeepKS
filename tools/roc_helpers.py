"""Helper functions for computing ROC curves and AUC scores"""

import itertools, sklearn.metrics, numpy as np, warnings, scipy, bisect
from numpy.typing import ArrayLike
from roc_comparison_modified.auc_delong import delong_roc_variance


class ROCHelpers:
    """Contains static methods for computing ROC curves and AUC scores"""

    class RocAvgValue:
        @classmethod
        def area_under_points(cls, points) -> float:
            area = 0
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                base = abs(x2 - x1)
                height = (y1 + y2) / 2
                area += base * height
            return area

        @classmethod
        def linear_interpolation(cls, x_values, y_values):
            if isinstance(x_values, np.ndarray):
                x_values = x_values.tolist()
            if isinstance(y_values, np.ndarray):
                y_values = y_values.tolist()

            def interp_func(x):
                if x < min(x_values) or x > max(x_values):
                    raise ValueError("x-value is outside the range of the input points")
                if x in x_values:
                    idx = x_values.index(x)
                    if x_values.count(x) > 1:
                        return tuple([y_values[idx + i] for i in range(x_values.count(x))])
                i = bisect.bisect_left(x_values, x)
                if i == 0:
                    return (y_values[0],)
                elif i == len(x_values):
                    return (y_values[-1],)
                else:
                    x0, x1 = x_values[i - 1], x_values[i]
                    y0, y1 = y_values[i - 1], y_values[i]
                    return (y0 + (y1 - y0) * (x - x0) / (x1 - x0),)

            return interp_func

        @classmethod
        def weighted_avg(cls, Xes, weights, lin_intp_fns):
            unique_x_vals: list[float] = sorted(list(set(list(itertools.chain(*Xes)))))
            # total_x = len(list(itertools.chain(*Xes)))
            # weights = [len(X) / total_x for X in Xes]
            all_interps = []
            for x in unique_x_vals:
                interps = [list(lif(x)) for lif in lin_intp_fns]
                ml = max(len(inte) for inte in interps)
                for l in range(len(interps)):
                    interps[l] += [interps[l][0] for _ in range(ml - len(interps[l]))]
                for i, inte in enumerate(interps):
                    inte = [x * weights[i] for x in inte]
                    interps[i] = inte
                all_interps.append(interps)
            chunky = [np.sum(np.asarray(interp), axis=0).tolist() for interp in all_interps]
            x_rep = [x for i, x in enumerate(unique_x_vals) for _ in range(len(chunky[i]))]
            flattened: list[float] = [item for sublist in chunky for item in sublist]
            assert len(x_rep) == len(flattened)
            auc = cls.area_under_points(list(zip(x_rep, flattened)))
            return x_rep, flattened, auc

        @classmethod
        def get_avg_line_pnts(
            cls, Xes, Yes, weights, num_addl_linspace_pts=0
        ) -> tuple[list[float], list[float], float]:
            lin_intp_fns = [cls.linear_interpolation(X, Y) for X, Y in zip(Xes, Yes)]
            xmin = min(min(X) for X in Xes)
            xmax = max(max(X) for X in Xes)
            linspace = np.linspace(xmin, xmax, num_addl_linspace_pts, endpoint=True).tolist()
            return cls.weighted_avg([sorted(list(set([x for x in X] + linspace))) for X in Xes], weights, lin_intp_fns)

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
            return "** "
        elif p_val < 0.05:
            return "*  "
        if p_val < 0.1:
            return ".  "
        else:
            return "   "
