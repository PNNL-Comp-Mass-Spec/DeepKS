"""Plot histogram of lengths of kinases in the dataset."""

import matplotlib.pyplot as plt

from matplotlib import rcParams
import numpy as np


rcParams["font.family"] = "P052"
special_cases = {
    1: 'unitile',
    2: 'bitile',
    3: 'tritile',
    4: 'quartile',
    5: 'quintile',
    6: 'sextile',
    7: 'septile',
    8: 'octile',
    9: 'nonatile',
    10: 'decile',
    11: 'undecile',
    12: 'duodecile'
}

# Generate random data for the histogram
def n_tile_hist(data: list[int] | list[float] | list[int | float], n_tile: int = 5, total_bins: int = 50):
    """Make a histogram, with different opacity for each n-tile.

    Parameters
    ----------
    data : 
        List of datapoints to plot.
    n_tile : optional
        The number of partitions to shade by, by default 5
    total_bins : optional
        The number of bins to plot in the histogram, by default 50
    """
    assert total_bins % n_tile == 0, "`n_tile` must be a factor of `total_bins`"
    bin_heights, bin_locs, patches = plt.hist(data, bins = total_bins, label=f"Distribution, shaded by {special_cases.get(n_tile, f'{n_tile}-tile')}", align='mid')
    bin_locs = bin_locs.tolist()
    bin_heights = bin_heights.tolist()
    col = 'blue'
    q_list = list(range(0, 100 + 1, 100//n_tile))
    q_tuple_list = [(q_list[i], q_list[i+1]) for i in range(len(q_list) - 1)]
    alphs = np.linspace(0.1, 1, n_tile)
    data_seen = 0
    for i in range(total_bins):
        which_shade_idx = 0
        pctl =  data_seen/len(data)*100
        for j, elt in enumerate(q_tuple_list):
            if elt[0] <= pctl <= elt[1]:
                which_shade_idx = j
                break
        data_seen += bin_heights[i]
        patches[i].set_facecolor(col)
        patches[i].set_alpha(alphs[which_shade_idx])
        print(pctl, which_shade_idx)
    plt.vlines(4128, 0, 200, color='red', label='Current length of padded Tensor (remove longer from dataset)')
    # Add labels and title
    plt.xlabel("Length of Kinase")
    plt.ylabel("Frequency")
    plt.title("Histogram of Kinase Lengths")
    plt.legend()
    plt.ylim(0, 100)

    # Display the plot
    plt.savefig(f"KinLenHist.pdf", bbox_inches='tight')

    plt.xlim(0, 5000)

    plt.savefig(f"KinLenHist_zoomed.pdf", bbox_inches='tight')
