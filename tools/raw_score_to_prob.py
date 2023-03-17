import scipy, math, inspect, numpy as np, warnings
from typing import Iterable, NoReturn, Union, Callable
from matplotlib import pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "P052"


def raw_score_to_probability(
    scores: list[Union[float, int]],
    truths: list[bool],
    incr=0.01,
    plot_emp_eqn=False,
    true_cdf=None,
    print_emp_eqn=False,
) -> Callable[[list[Union[float, int]]], Union[NoReturn, list[float]]]:
    assert all(isinstance(x, (int, float)) for x in scores), f"`{inspect.stack()[0][3]}` requires scores to be numeric."
    assert all(isinstance(x, bool) for x in truths), f"`{inspect.stack()[0][3]}` requires truths to be boolean."
    assert sorted(scores) == scores, f"`{inspect.stack()[0][3]}` requires sorted scores as input."
    assert len(scores) == len(truths), f"`{inspect.stack()[0][3]}` requires equal length of scores and truths."
    i = 0
    places = int(np.ceil(-np.log10(incr)))
    l_orig = round(np.floor(min(scores) * 10**places) / 10**places, places)
    l = l_orig
    r = l + incr
    probs = []
    midpoints = []
    while i < len(scores):
        count = 0
        total = 0
        while i < len(scores) and l <= scores[i] < r:
            if truths[i]:
                count += 1
            total += 1
            i += 1
        if total != 0:
            midpoints.append(mp := (l + r) / 2)  # type: ignore
            probs.append(prob := count / total)  # type: ignore
            # print("Cutoff {:.3f} = Prob {:.4f}".format(mp, prob))
        l += incr
        r += incr
    assert len(probs) == len(midpoints), f"`{inspect.stack()[0][3]}` failed to calculate probabilities."
    assert len(probs) != 0, f"number of probabilities is 0 for some reason."
    if len(probs) < 2:
        if len(probs) == 1:
            warnings.warn(f"Only one probability point found. Adding pseudo-points (0, 0) and (1, 1).")
            # Add pseudo-points to make the curve fit work. Assume (0, 0) and (1, 1) are true.
            midpoints = [0, midpoints[0], 1]
            probs = [0, probs[0], 1]
    try:
        base_fn = lambda x, c, d: (
            [0.5 + (1 / math.pi) * math.atan(c * (x0 - d)) for x0 in x]
            if isinstance(x, Iterable)
            else 0.5 + (1 / math.pi) * math.atan(c * (x - d))
        )
        (c, d), *_ = params, *_ = scipy.optimize.curve_fit(base_fn, midpoints, probs, maxfev=1000000, xtol=1e-6)
    except Exception as e:
        print(f"Failed to fit curve to data: {e}")
        print(f"May need to adjust `maxfev` and/or `xtol` in `{inspect.stack()[0][3]}`.")
        raise e from None
    if print_emp_eqn:
        print("Equation --- ")
        print("Let x be the raw score, then the probability of being a true hit is:")
        print(f"P(x) = 0.5 + (1/π)arctan({c:.3f}(x-{d:.3f}))")

    convert_fn = lambda x: base_fn(x, *params)
    if plot_emp_eqn:
        old_ax = plt.gca()
        plt.figure()
        plt.plot(midpoints, probs, "go-", label="Actual Data", markersize=1, linewidth=0.5)
        X = np.linspace(l_orig, max(scores), endpoint=True, num=1000)
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
