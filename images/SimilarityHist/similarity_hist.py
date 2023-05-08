import collections, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, matplotlib as mpl
import sklearn.mixture as mix
from typing import Union, Callable

KIN_AND_ORG_TO_SYMBOL = (
    pd.read_csv("../../data/raw_data/raw_data_22473.csv")
    .applymap(lambda x: x.upper() if isinstance(x, str) else x)
    .drop_duplicates(keep="first")
    .set_index(["Kinase Sequence", "organism"])
    .to_dict()["uniprot_id"]
)
KIN_AND_ORG_TO_SYMBOL_DICT = collections.defaultdict(dict[str, str])
for k, v in KIN_AND_ORG_TO_SYMBOL.items():
    KIN_AND_ORG_TO_SYMBOL_DICT[k[0]][k[1]] = v


def main(kin, org):
    kin = kin.upper()
    org = org.upper()
    try:
        token = kin.upper() + "|" + KIN_AND_ORG_TO_SYMBOL_DICT[kin][org]
    except KeyError:
        raise KeyError(f"kin, org pair {kin, org} not found.")

    m = pd.read_csv("../../data/preprocessing/mtx_822.csv", index_col=0)
    mult100: Callable[[Union[int, float]], float] = lambda x: x * 100

    def execute(token):
        row = np.asarray(m.loc[token].apply(mult100).values)
        data = row  # np.concatenate([-row, row])
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams["font.family"] = "monospace"
        bins = 50

        pct = np.percentile(data, 95)
        # outlier_pct = np.percentile(data, 98)
        # data = data[data < outlier_pct]

        # Histogram Bars
        # sns.histplot(data, stat='count', color = "#58b368", label = "Similarity Distribution", bins=bins, edgecolor='#248f69')  # type: ignore

        # Just for Bins
        hist_params = dict(
            bins=bins, alpha=1, color="#58b368", label="Similarity Distribution", edgecolor="#248f69", align="mid"
        )
        factor_hist_a = plt.hist(data, **hist_params)

        # Density Line
        density_line = sns.kdeplot(data, color="#454d66", label="Kernel Density Estimation", gridsize=5000)  # type: ignore

        scale_factor = np.max(factor_hist_a[0]) / np.max(density_line.lines[0]._y)
        density_line.lines[0]._y *= scale_factor

        # connected_y = factor_hist_a[0]

        # peaks = scipy.signal.find_peaks(connected_y, prominence = 5, distance = 5)[0]
        # cutoff3SD = add_gmm_curves(len(peaks), data, hist = factor_hist_a)#, init_means=np.array([[(connected_x[1:][i] + connected_x[:-1][i])/2] for i in peaks]))
        # ind = 0
        # for i in range(len(GMMy[0]) - 1, -1, -1):
        #     if GMMy[-2][i] >= sd3#>= 0.999: # GMMy[-2][i] and GMMy[-1][i] >= 1:
        #         ind = i
        #         break

        plt.title(
            f"Distribution of Pairwise Kinase Similarities\nBetween {token.split('|')[0]} and {len(data)} Kinases (All"
            " Species)"
        )
        plt.ylabel("Number of Kinases")
        plt.xlabel("Percent Identity (Clustal Omega)")
        plt.xticks(*[xt := list(range(0, 101, 5))] * 2)
        plt.yticks(
            *[
                yt := list(
                    range(
                        0,
                        int(np.ceil(1.1 * np.max(factor_hist_a[0]))),
                        max(10, 10 ** int(np.log10(1.1 * np.max(factor_hist_a[0]) / 25))),
                    )
                )
            ]
            * 2
        )
        plt.xlim(min(xt), max(xt))
        plt.ylim(min(yt), max(yt))
        plt.vlines(
            pct, 0, max(yt), color="teal", linestyle="dashed", label=f"95% Similarity Threshold\n(@ {pct:.3}% Identity)"
        )
        # plt.vlines(cutoff3SD, 0, max(yt), color = "magenta", linestyle = 'dashed', label = f"Alternate Gaussian Threshold\n(@ {cutoff3SD:.3}% Identity)")
        # plt.hlines(0.999, min(xt), max(xt), color = "black", linestyle = 'solid', label = f"Ref")
        plt.legend()
        plt.savefig(f"HistDist_{kin}-{org}.svg", bbox_inches="tight")
        plt.show()

    # for tok in m.index:
    #     execute(tok)
    execute(token)


def add_gmm_curves(
    n,
    data,
    hist=None,
    replot_hist=False,
    param_dict=None,
    lines_params={"color": "lime"},
    verbose=True,
    init_means=None,
):
    if len(data.shape) < 2:
        data = data.reshape(-1, 1)
    if param_dict is None:
        param_dict = {}
    h = hist
    if h is None:
        h = plt.hist(data, **param_dict)
        replot_hist = False
    if init_means is None:
        init_means = {}
    else:
        init_means = {"means_init": init_means}
    gmm = mix.GaussianMixture(n_components=(n), **init_means)  # , covariance_type = "tied")
    gmm.fit(data)
    m = gmm.means_.ravel()
    w = gmm.weights_.ravel()
    c = gmm.covariances_.ravel()
    c = np.sqrt(c)
    if verbose:
        print("Means:\n", m, "\nWeights:\n", w, "\nCovariances:\n", c)
    X = np.linspace(min(m - c * 5), max(m + c * 5), num=1000)
    Y = np.exp([-((X - m[i]) ** 2 / (2.0 * c[i] ** 2)) for i in range(n)])
    Y = (Y.T * w).T * np.max(h[0]) / np.max((Y.T * w).T)
    if replot_hist:
        print("RP")
        plt.hist(data, **param_dict)
    for i in range(n):
        if i == 0:
            lines_params.update({"label": "GMM Curves"})
        plt.plot(X, Y[i], **lines_params)
        if i == 0:
            del lines_params["label"]
    order = np.argsort(m)
    cutoff3SD = m[order][-1] - c[order][-1]  # m[order][-2] + c[order][-2]*3
    return cutoff3SD  # X, Y[order]


if __name__ == "__main__": # pragma: no cover
    main("CDK1", "HUMAN")
