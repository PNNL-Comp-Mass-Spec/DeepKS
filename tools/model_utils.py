"""Contains various useful functions for working with models (specifically CNNs) and data"""

from abc import ABC
import warnings, json, pandas as pd, numpy as np
from matplotlib.pyplot import figure, rcParams
from matplotlib import colors, pyplot as plt, cm
from torch.utils.data import Dataset
from torch import IntTensor, Tensor
from math import floor, ceil
from numpy import logspace
from mpl_toolkits.axes_grid1 import make_axes_locatable
from termcolor import colored
from typing import Literal, Union


rcParams["font.family"] = "monospace"
rcParams["font.size"] = "8"

from ..config.logging import get_logger

logger = get_logger()
"""The logger for this module."""


class cNNUtils:
    """General useful utilities for working with CNNs"""

    @staticmethod
    def calculate_cNN_params(model_self, kin_or_site: Literal["kin", "site"]) -> tuple[list, list, list, list]:
        """Calculates the parameters for the CNN(s) for either the kinase or site sequence.

        Parameters
        ----------
        kin_or_site :
            Either "kin" or "site".

        Returns
        -------
            Calculated pool sizes, in channels, whether to flatten, and whether to transpose, for each CNN layer.

        Raises
        ------
        ValueError
            If ``kin_or_site`` is not ``kin`` or ``site``.
        """
        if kin_or_site == "kin":
            param = model_self.kin_param_dict
            emb = model_self.emb_dim_kin
            first_width = model_self.kin_len
            num_conv = model_self.num_conv_kin
        elif kin_or_site == "site":
            param = model_self.site_param_dict
            emb = model_self.emb_dim_site
            first_width = model_self.site_len
            num_conv = model_self.num_conv_site
        else:
            raise ValueError("kin_or_site must be 'kin' or 'site'")

        calculated_pools = []
        calculated_in_channels = []
        calculated_do_flatten = []
        calculated_do_transpose = []

        for i in range(num_conv):
            calculated_do_transpose.append(i == 0)
            calculated_do_flatten.append(False)
            if i == 0:
                calculated_in_channel = emb
            else:
                calculated_in_channel = param["out_channels"][i - 1]
            calculated_in_channels.append(calculated_in_channel)
            if i == 0:
                input_width = first_width
            else:
                input_width = param["out_lengths"][i - 1]
            calculated_pools.append(
                cNNUtils.desired_conv_then_pool_shape(
                    length=input_width,
                    desired_length=param["out_lengths"][i],
                    kernel_size=param["kernels"][i],
                    err_message=f"{kin_or_site} CNNs",
                )
            )

        return (
            calculated_pools,
            calculated_in_channels,
            calculated_do_flatten,
            calculated_do_transpose,
        )

    @staticmethod
    def id_params(config, db_file="../architectures/HP_config_DB.tsv", index_column=0):
        db = pd.read_csv(db_file, sep="\t")
        config = str(config).replace("\n", ";; ")
        if len(db) != 0 and config in db["config_str"].values:
            res = db["config_str"][db["config_str"] == config].index[0]
        else:
            db.loc[len(db), "config_str"] = config
            db.to_csv(db_file, sep="\t", index=True)
            res = len(db) - 1

        print("Hyperparameter config id:", res)
        return res

    @staticmethod
    def output_shape(length: int, kernel_size: int, stride: int = 1, pad: int = 0, dilation: int = 1) -> int:
        """Calculate the output shape of a dimension (W, H, or D) for a convolutional layer.

        Parameters
        ----------
        length : int
            The length of the input
        kernel_size : int
            The size of the kernel
        stride : int, optional
            The stride of the convolution, by default 1
        pad : int, optional
            The padding of the convolution, by default 0
        dilation : int, optional
            The dilation of the convolution, by default 1

        Returns
        -------
        int :
            The width of the output image

        Notes
        -----
            Based on https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
        """
        try:
            return floor(((length + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
        except ZeroDivisionError as zde:
            print(zde, flush=True)
            warnings.warn("ZeroDivisionError in output_shape")
            return 0

    @staticmethod
    def desired_conv_then_pool_shape(
        length: int,
        desired_length: int,
        kernel_size: int = 10,
        dilation: int = 1,
        pad: int = 0,
        conv_stride: int = 1,
        err_message: str = "",
    ) -> int:
        """Calculate the stride for a pooling layer if it follows a convolutional layer to get a desired output shape.

        Parameters
        ----------
        length : int
            The input length (unconvolved)
        desired_length : int
            The desired output length (after convolution and pooling)
        kernel_size : int, optional
            The kernel for the convolution, by default 10. (The kernel for the pooling is assumed to be the same as the resultant stride)
        dilation : int, optional
            The convolutional dilation, by default 1
        pad : int, optional
            The convolutional padding, by default 0
        conv_stride : int, optional
            The stride for the convolution, by default 1
        err_message : str, optional
            The error message to print if no stride can be found that satisfies the input constraints, by default ""

        Returns
        -------
            The pooling stride that will result in the desired output shape

        Raises
        ------
        ValueError
            If no stride can be found that satisfies the input constraints
        """
        after_conv = cNNUtils.output_shape(length, kernel_size, conv_stride, pad, dilation)
        values_to_check = [desired_length, desired_length + 1 - 1e-6]

        stride_range = []
        for v in values_to_check:
            stride = (after_conv + 2 * pad - 1 + dilation) / (v - 1 + dilation)
            stride_range.append(stride)
        stride_res = 0
        if stride_range[0] < 1 or (
            stride_range[0] - stride_range[1] < 1
            and int(stride_range[1]) == int(stride_range[0])
            and stride_range[0] != int(stride_range[0])
        ):
            raise ValueError(
                "There is no stride for which the output shape equals the desired shape --"
                f" {desired_length} with the given kernel"
                f" {kernel_size} ({err_message}). Some close by alternative output lengths are:"
                f" {cNNUtils.close_by_out_sizes(after_conv, desired_length)}"
            )

        else:
            stride_res = int(stride_range[0])
            assert stride_res >= 1, f"Stride must be >= 1. For some reason it is not (it is {stride_res})."

        return stride_res

    @staticmethod
    def close_by_out_sizes(base_len: int, desired_out: int) -> list[int]:
        """Get a list of output sizes that are close to the desired output size.

        Parameters
        ----------
        base_len :
            The length of the input
        desired_out :
            The desired output length

        Returns
        -------
            List of output sizes that are close to the desired output size
        """
        possible = set()
        spread = 10
        upper = desired_out + spread
        lower = desired_out - spread
        while lower >= 1 and upper <= base_len and len(possible) < 3:
            print(lower, upper)
            for i in range(lower, upper + 1):
                if int(base_len / i) != int(base_len / (i + 1 - 1e-6)):
                    possible.add(i)
            upper += spread
            lower -= spread
        return sorted(list(possible))


class KSDataset(Dataset, ABC):
    def __init__(self, all_encoded_site_tensors: IntTensor, all_encoded_kin_tensors: IntTensor, all_labels: IntTensor):
        self.sites = all_encoded_site_tensors
        self.kins = all_encoded_kin_tensors
        self.labels = all_labels


class TandemKSDataset(KSDataset):
    def __init__(self, all_encoded_site_tensors: IntTensor, all_encoded_kin_tensors: IntTensor, all_labels: IntTensor):
        super().__init__(all_encoded_site_tensors, all_encoded_kin_tensors, all_labels)
        assert len(self.sites) == len(self.kins) == len(all_labels), (
            f"For a `TandemKSDataset`, the number of input sites (gave {len(all_encoded_site_tensors)}) must equal the"
            f" number of input kinases (gave {len(all_encoded_kin_tensors)}), which must equal the the number of labels"
            f" (gave {len(all_labels)})."
        )
        self.length = len(self.sites)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (self.sites[index], self.kins[index], self.labels[index])


class CartesianKSDataset(KSDataset):
    def __init__(self, all_encoded_site_tensors: IntTensor, all_encoded_kin_tensors: IntTensor, all_labels: IntTensor):
        super().__init__(all_encoded_site_tensors, all_encoded_kin_tensors, all_labels)
        if len(self.sites) == len(self.kins):
            logger.warn(
                f"The number of input sites (just gave {len(all_encoded_site_tensors)}) is equal to the number of input"
                " kinases. Did you mean to use a `TandemKSDataset`?"
            )
        self.length = len(self.sites) * len(self.kins)
        assert self.length == len(all_labels), (
            "For a `CartesianKSDataset`, the length of the dataset (gave number of sites * number of kinases ="
            f" {self.length}) must equal the number of labels (gave {len(all_labels)})"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        site_idx = index // len(self.kins)
        kin_idx = index % len(self.kins)
        return (self.sites[site_idx], self.kins[kin_idx], self.labels[index])


class DLWeightPlot:
    def __init__(self):
        self.data = []
        self.titles = []
        self.mainfig = None

    def add_weights_to_plot(self, weights, title=""):
        self.data.append(weights)
        self.titles.append(title)

    def plot_fig(self, diffs=False):
        if not diffs:
            skipsize = 1
        else:
            skipsize = 2
        if diffs:
            self.dfs = [self.data[i] - self.data[i - 1] for i in range(1, len(self.data))]
        self.mainfig, axes_list = plt.subplots(
            len(self.data) + 1,
            skipsize,
            gridspec_kw={"hspace": 0.5, "wspace": 1.5, "height_ratios": [0.2] + [1] * len(self.data)},
            figsize=(5, 3 * len(self.data)),
        )
        axes_list = np.asarray(axes_list)
        ranges_defined = False
        v_max_main, v_min_main = 1, 0
        for i in range(1, len(self.data) + 1):
            ranges_defined = True
            axes_list[i, 0].imshow(
                self.data[i - 1],
                cmap="PiYG",
                vmin=(v_min_main := min([np.amin(self.data[j]) for j in range(len(self.data))])),
                vmax=(v_max_main := max([np.amax(self.data[k]) for k in range(len(self.data))])),
            )
            if diffs and i > 1:
                axes_list[i, 1].imshow(
                    self.dfs[(i - 1) - 1], vmin=-2.5, vmax=2.5, cmap="seismic"
                )  # vmin = min([np.amin(self.dfs[i]) for i in range(len(self.dfs))]), vmax = max([np.amax(self.dfs[i]) for i in range(len(self.dfs))]))
                if np.all(np.isclose(self.dfs[(i - 1) - 1], np.zeros_like(self.dfs[(i - 1) - 1]), rtol=0, atol=1e-16)):
                    axes_list[i, 1].set_title("All zeros within 1e-16")
            axes_list[i, 0].title.set_text(self.titles[i - 1])

        # colorbar
        # axes_list[0, 1].set_aspect("equal")
        divider0 = make_axes_locatable(axes_list[0, 0])
        divider1 = make_axes_locatable(axes_list[0, 1])
        cax0 = divider0.append_axes("top", size="50%", pad=-0.5)
        cax1 = divider1.append_axes("top", size="50%", pad=-0.5)
        if ranges_defined:
            self.mainfig.colorbar(
                cm.ScalarMappable(cmap="PiGY", norm=colors.Normalize(vmax=v_max_main, vmin=v_min_main)),
                cax=cax0,
                orientation="horizontal",
            )
        self.mainfig.colorbar(
            cm.ScalarMappable(cmap="seismic", norm=colors.Normalize(vmax=2.5, vmin=-2.5)),
            cax=cax1,
            orientation="horizontal",
        )

        axes_list[0, 0].remove()
        axes_list[0, 1].remove()
        axes_list[1, 1].remove()

    def save_fig(self, file_out_name):
        if self.mainfig is None:
            print("Error: Please `plot_fig` before saving.", flush=True)
            return
        plt.figure(self.mainfig)
        if file_out_name.split(".")[-1] == "png":
            file_out_name = file_out_name.split(".")[0] + ".pdf"
        plt.savefig(file_out_name, width=25, height=100, bbox_inches="tight")
