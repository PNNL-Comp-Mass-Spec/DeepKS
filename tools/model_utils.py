import warnings, json, pandas as pd, numpy as np
from matplotlib.pyplot import figure, rcParams
from matplotlib import colors, pyplot as plt, cm
from torch.utils.data import Dataset
from math import floor, ceil
from numpy import logspace
from mpl_toolkits.axes_grid1 import make_axes_locatable
from termcolor import colored
from typing import Literal, Union


rcParams["font.family"] = "monospace"
rcParams["font.size"] = "8"


class dataUtils:
    @staticmethod
    def get_multiple_folds(dataloader):
        pass


class cNNUtils:
    @staticmethod
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
    def output_shape(width, height, kernel_size, stride=1, pad=0, dilation=1):
        """
        Based on https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5?
        """
        try:
            w_h = (width, height)
            if not isinstance(kernel_size, tuple):
                kernel_size = (kernel_size, kernel_size)
            if not isinstance(stride, tuple):
                stride = (stride, stride)
            if not isinstance(pad, tuple):
                pad = (pad, pad)
            w = floor(((w_h[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) / stride[0]) + 1)
            h = floor(((w_h[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) / stride[1]) + 1)
            return w, h
        except ZeroDivisionError as zde:
            print(zde, flush=True)
            warnings.warn("ZeroDivisionError in output_shape")
            return 0, 0

    @staticmethod
    def desired_conv_then_pool_shape(
        input_width,
        input_height,
        desired_width,
        desired_height,
        kernel_size=None,
        stride=None,
        dilation=1,
        pad=0,
        conv_stride=1,
        err_message="",
    ):
        if input_height is None:
            input_height = input_width
        if desired_height is None:
            desired_height = desired_width
        after_conv_w, after_conv_h = cNNUtils.output_shape(
            input_width, input_height, kernel_size, conv_stride, pad, dilation
        )
        return cNNUtils.desired_output_shape(
            after_conv_w,
            after_conv_h,
            desired_width,
            desired_height,
            kernel_size=kernel_size,
            stride=stride,
            err_message=err_message,
            pad=pad,
            dilation=dilation,
        )

    @staticmethod
    def close_by_out_sizes(numerator, x):
        close = numerator / x
        close_works = int(close)
        base = numerator // close_works
        return (base - 2, base - 1, base, base + 1, base + 2)

    @staticmethod
    def desired_output_shape(
        input_width,
        input_height,
        desired_width,
        desired_height,
        kernel_size=None,
        stride=None,
        dilation: Union[int, None] = 1,
        pad: Union[int, tuple[int, int]] = 0,
        err_message="",
    ):
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        if isinstance(pad, int):
            pad_tuple: tuple[int, int] = (pad, pad)
        else:
            pad_tuple = pad
        # if kernel_size not in [None, (None, None)] and stride not in [None, (None, None)]:
        #     raise RuntimeError("Cannot specify both kernel size and stride.")
        # if kernel_size in [None, (None, None)] and stride in [None, (None, None)]:
        #     raise RuntimeError("Must specify one of kernel size or stride.")
        if dilation is None:
            raise RuntimeError("Must specify dilation.")
        if pad_tuple == (None, None):
            raise RuntimeError("Must specify pad.")
        if input_height is None:
            input_height = input_width
        if desired_height is None:
            desired_height = desired_width

        assert int(desired_width) == desired_width
        desired_width = int(desired_width)
        assert int(desired_height) == desired_height
        desired_height = int(desired_height)

        values_to_check_w = [desired_width, desired_width + 1 - 1e-6]
        values_to_check_h = [desired_height, desired_height + 1 - 1e-6]

        # input_width, input_height = cNNUtils.output_shape(input_width, input_height, kernel_size, 1, pad, dilation)

        # if kernel_size in [None, (None, None)]: # specified stride, need to obtain kernel size
        #     raise RuntimeError("Not implemented yet.")
        #     # kernel_range = []
        #     # for desired_w, desired_h in zip(values_to_check_w, values_to_check_h):
        #     #     k_w = ((stride * (desired_w - 1) - input_width - 2 * pad[0] + 1)/dilation) + 1
        #     #     k_h = ((stride * (desired_h - 1) - input_height - 2 * pad[1] + 1)/dilation) + 1
        #     #     kernel_range.append((k_w, k_h))

        #     # kernel_res = []
        #     # for i in [0, 1]:
        #     #     if kernel_range[1][i] < 1 or (kernel_range[1][i] - kernel_range[0][i] < 1 and int(kernel_range[0][i]) == int(kernel_range[1][i]) and stride_range[0][i] != int(stride_range[0][i])):
        #     #         raise RuntimeError("There is no kernel for which the output shape equals the desired shape. ")
        #     #         return None, None
        #     #     else:
        #     #         kernel_res[i] = max(1, ceil(kernel_range[1][i]))
        #     #         kernel_res[i] = max(1, ceil(kernel_range[1][i]))

        #     # return tuple(kernel_res)

        if stride in [None, (None, None)]:  # specified kernel size, need to obtain stride size
            stride_range = []
            for desired_w, desired_h in zip(values_to_check_w, values_to_check_h):
                s_w = (input_width + 2 * pad_tuple[0] - 1 + dilation) / (desired_w - 1 + dilation)
                s_h = (input_height + 2 * pad_tuple[1] - 1 + dilation) / (desired_h - 1 + dilation)
                stride_range.append((s_w, s_h))

            stride_res = [0, 0]
            for i in [0, 1]:
                if stride_range[0][i] < 1 or (
                    stride_range[0][i] - stride_range[1][i] < 1
                    and int(stride_range[1][i]) == int(stride_range[0][i])
                    and stride_range[1][i] != int(stride_range[1][i])
                ):
                    raise ValueError(
                        "There is no stride for which the output shape equals the desired shape --"
                        f" {(values_to_check_w[0], values_to_check_h[0])} with the given kernel {kernel_size}"
                        f" ({err_message})."  # Some close by alternative output lengths are: {self.close_by_out_sizes(numerator=input_width, x=desired_width)}"
                    )

                else:
                    stride_res[i] = max(1, ceil(stride_range[1][i]))
                    stride_res[i] = max(1, ceil(stride_range[1][i]))
        else:
            raise RuntimeError(colored("Not Implemented", "r"))

        return stride_res


class KSDatasetSiamese(Dataset):
    def __init__(self, data, target, siamese, class_):
        self.data = data
        self.target = target
        self.siamese = siamese
        self.class_ = class_
        assert len(self.data) == len(self.target) == len(self.class_) == len(siamese)

    def __getitem__(self, index):
        return (self.data[index], self.target[index], self.siamese[index], self.class_[index])

    def __len__(self):
        return len(self.data)


class KSDataset(Dataset):
    def __init__(self, data, target, class_):
        self.data = data
        self.target = target
        self.class_ = class_
        assert (data is None and target is None and class_ is None) or (
            len(self.data) == len(self.target) == len(self.class_)
        )

    def __getitem__(self, index):
        return (self.data[index], self.target[index], self.class_[index])

    def __len__(self):
        if self.data is None:
            return 0
        return len(self.data)


class KGroupDataset(Dataset):
    def __init__(self, sequence, class_):
        self.sequence = sequence
        self.class_ = class_
        assert len(self.sequence) == len(self.class_)

    def __getitem__(self, index):
        return (self.sequence[index], self.class_[index])

    def __len__(self):
        return len(self.sequence)


def make_plot(main_fn, fn):
    figure(figsize=(15, 8))
    lRs = logspace(-4, -0.5, 10)
    results = []
    for lR in lRs:
        results.append(main_fn("data/syn_data_target_1000.csv", "data/syn_data_decoy_1000.csv", lR, 1))
    json.dump(results, open("data/lr_results.json", "w"), indent=4)

    # plot(lRs, [np[i] / ns[i] for i in range(len(lRs))], 'bo-',
    #      label="Perfect classifier fraction", alpha=0.8)
    # plot(lRs, [nc[i] / ns[i] for i in range(len(lRs))], 'ro-',
    #      label="Converged classifier fraction", alpha=0.8)
    # xscale("log")
    # legend()
    # ylim(0, 1)

    # xlabel("Learning Rate")
    # ylabel("Fraction")
    # title("Num Epochs: %d | Train Batch Size: %d | Num Seeds: %d | Embed Dim: %d | Conv Thsh %.2f | Pool Max" %
    #     (30, 2, 10, 15, 0.4))
    # savefig(fn + ".png")


class DLWeightPlot:
    def __init__(self):
        self.data = []
        self.titles = []
        self.mainfig = None

    def add_weights_to_plot(self, weights, title=""):
        self.data.append(weights)
        self.titles.append(title)

    def plot_fig(self, diffs=False):
        skipsize = 1 if not diffs else 2
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
