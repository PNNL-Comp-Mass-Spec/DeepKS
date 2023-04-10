from matplotlib.markers import MarkerStyle
import inspect, numpy as np, warnings
from typing import NoReturn, Union, Callable, Iterable
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import LogisticRegression
from scipy.stats import gaussian_kde

rcParams["font.family"] = "P052"


def raw_score_to_probability(
    scores: list[Union[float, int]],
    truths: list[bool],
    plot_emp_eqn=False,
    true_cdf=None,
    print_emp_eqn=False,
) -> Callable[[list[Union[float, int]]], Union[NoReturn, Iterable[float], float]]:
    assert all(isinstance(x, (int, float)) for x in scores), f"`{inspect.stack()[0][3]}` requires scores to be numeric."
    assert all(isinstance(x, bool) for x in truths), f"`{inspect.stack()[0][3]}` requires truths to be boolean."
    assert sorted(scores) == scores, f"`{inspect.stack()[0][3]}` requires sorted scores as input."
    assert len(scores) == len(truths), f"`{inspect.stack()[0][3]}` requires equal length of scores and truths."

    def ident_func(x):
        return x

    def fit_curve_to_data(scores, truths):
        lr = LogisticRegression(penalty=None)
        try:
            lr.fit(np.array(scores).reshape(-1, 1), truths)
        except Exception as e:
            msg = (
                f"Failed to fit curve to data: {e}. May need to adjust logistic regression learning parameters."
                " Returning identity function instead."
            )
            warnings.warn(msg, RuntimeWarning)
            return ident_func

        m = -lr.coef_[0, 0]
        b = -lr.intercept_[0]
        underlying_lm = lambda x: m * x + b if isinstance(x, (int, float)) else [m * x_ + b for x_ in x]

        def underlying_sigmoid(x):
            logit = underlying_lm(x)
            try:
                return 1 / (1 + np.exp(logit))
            except OverflowError:
                warnings.warn(f"OverflowError encountered in `{inspect.stack()[0][3]}`. Returning 0.0 instead.")
                return 0.0

        if print_emp_eqn:
            print("Equation --- ")
            print("Let x be the raw score, then the probability of being a true hit is:")
            print(f"P(x) = 1 / (1 + exp({b} + {m} * x))")
        return underlying_sigmoid

    convert_fn = fit_curve_to_data(scores, truths)
    if convert_fn is ident_func:
        return convert_fn
    if plot_emp_eqn:
        old_ax = plt.gca()
        plt.figure()
        xy = np.vstack([scores, truths])
        z = gaussian_kde(xy)(xy)
        mark = MarkerStyle("o")
        plt.scatter(scores, truths, label="Actual Data", marker=mark, s=20, linewidth=0, alpha=1, c=z)
        X = np.linspace(min(scores) - 0.1, max(scores) + 0.1, endpoint=True, num=1000)
        plt.plot(X, convert_fn(X), label="Approximated function(raw score) => probability", linewidth=0.5)
        if true_cdf is not None:
            plt.plot(
                X,
                [true_cdf(x) / true_cdf(10) for x in X],
                label="True function(raw score) => probability",
                linewidth=0.5,
            )
        plt.legend()
        plt.xlabel("Raw score")
        plt.ylabel("Estimated non-decoy fraction given raw score")
        plt.title("Converting raw score into a true estimated probability")
        plt.sca(old_ax)
    return convert_fn
