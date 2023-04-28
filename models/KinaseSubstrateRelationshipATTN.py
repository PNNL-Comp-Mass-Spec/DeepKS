import os, pathlib, torch, torch.nn as nn
from ..tools.formal_layers import Concatenation, Multiply, Transpose, Squeeze
from ..tools.model_utils import cNNUtils as cNNUtils
from .KSRProtocol import KSR
from typing import Literal, Tuple, Union
from pprint import pformat

torch.use_deterministic_algorithms(True)

from ..config.root_logger import get_logger

logger = get_logger()


def batch_dot(x, y):
    return torch.einsum("ij,ij->i", x, y)


class MHAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        do_transform: bool = False,
        q_k_v_transform_in_features: Union[Tuple, None] = None,
    ):
        self.do_transform = do_transform
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        if self.do_transform:
            assert q_k_v_transform_in_features is not None
            self.qif, self.kif, self.vif = q_k_v_transform_in_features
            self.trans_Q = nn.Linear(self.qif, self.embed_dim)
            self.trans_K = nn.Linear(self.kif, self.embed_dim)
            self.trans_V = nn.Linear(self.vif, self.embed_dim)

        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, site, kin):
        if self.do_transform:
            site_Q_out = self.trans_Q(site)
            kin_K_out = self.trans_K(kin)
            kin_V_out = self.trans_V(kin)
        else:
            site_Q_out = site
            kin_K_out = kin
            kin_V_out = kin

        return self.mha(site_Q_out, kin_K_out, kin_V_out)


class MultipleCNN(nn.Module):
    def __init__(
        self,
        out_channels,
        conv_kernel_size,
        pool_kernel_size,
        in_channels,
        do_flatten=False,
        do_transpose=False,
    ):
        super().__init__()
        self.do_transpose = do_transpose
        if self.do_transpose:
            self.transpose = Transpose(1, 2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=conv_kernel_size)
        self.activation = nn.ELU()  # nn.Tanh()
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)
        self.do_flatten = do_flatten
        if self.do_flatten:
            self.flat = nn.Flatten(1)

    def forward(self, x, ret_size=False):
        if self.do_transpose:
            out = self.transpose(x)
        else:
            out = x
        out = self.conv(out)
        out = self.activation(out)
        out = self.pool(out)
        if self.do_flatten:
            out = self.flat(out)
        if ret_size:
            return out.size()
        return out


# Convolutional neural network (two convolutional layers)
class KinaseSubstrateRelationshipATTN(KSR):
    def __init__(
        self,
        num_classes: int = 1,
        linear_layer_sizes: Union[list[int], None] = None,
        emb_dim_site: int = 22,
        emb_dim_kin: int = 22,
        site_param_dict={"kernels": [4], "out_lengths": [12], "out_channels": [11]},
        kin_param_dict={"kernels": [10], "out_lengths": [12], "out_channels": [11]},
        dropout_pr: float = 0.3,
        site_len: int = 15,
        kin_len: int = 4128,
        num_aa: int = 22,  # 20 + X + padding
        attn_num_heads: int = 8,
        padding_idx=21,
    ):
        super().__init__()
        site_param_vals = site_param_dict.values()
        kinase_param_vals = kin_param_dict.values()
        num_conv_site = len(site_param_dict["kernels"])
        num_conv_kin = len(kin_param_dict["kernels"])
        assert all(
            num_conv_site == len(x) for x in site_param_vals
        ), "# of site CNN params do not all equal `num_conv`."
        assert all(
            num_conv_kin == len(x) for x in kinase_param_vals
        ), "# of kinase CNN params do not all equal `num_conv`."
        self.num_conv_site = num_conv_site
        self.num_conv_kin = num_conv_kin
        self.emb_dim_site = emb_dim_site
        self.emb_dim_kin = emb_dim_kin

        self.emb_site = nn.Embedding(num_aa, self.emb_dim_site)
        self.emb_kin = nn.Embedding(num_aa, self.emb_dim_kin, padding_idx=padding_idx)
        self.site_param_dict = site_param_dict
        self.kin_param_dict = kin_param_dict
        self.site_len = site_len
        self.kin_len = kin_len
        self.squeeze_1 = Squeeze(1)
        self.squeeze_2 = Squeeze(2)

        pools_site, in_channels_site, do_flatten_site, do_transpose_site = self.calculate_cNN_params("site")
        pools_kin, in_channels_kin, do_flatten_kin, do_transpose_kin = self.calculate_cNN_params("kin")

        site_cnn_list = []
        kin_cnn_list = []
        for i in range(self.num_conv_site):
            site_cnn_list.append(
                MultipleCNN(
                    site_param_dict["out_channels"][i],
                    site_param_dict["kernels"][i],
                    pools_site[i],
                    in_channels_site[i],
                    do_flatten_site[i],
                    do_transpose_site[i],
                )
            )
        for i in range(self.num_conv_kin):
            kin_cnn_list.append(
                MultipleCNN(
                    kin_param_dict["out_channels"][i],
                    kin_param_dict["kernels"][i],
                    pools_kin[i],
                    in_channels_kin[i],
                    do_flatten_kin[i],
                    do_transpose_kin[i],
                )
            )

        self.site_cnns = nn.Sequential(*site_cnn_list)
        self.kin_cnns = nn.Sequential(*kin_cnn_list)

        self.transpose = Transpose(1, 2)

        assert (
            self.site_param_dict["out_channels"][-1] == self.kin_param_dict["out_channels"][-1]
        ), "Last CNN layer must have same number of output channels."
        assert (
            self.site_param_dict["out_channels"][-1] % attn_num_heads == 0
        ), "Last CNN layer must have number of output channels divisible by number of attention heads."
        assert (
            self.site_param_dict["out_lengths"][-1] == self.kin_param_dict["out_lengths"][-1]
        ), "Last CNN layer must have same output length."

        self.attn = MHAttention(embed_dim=self.site_param_dict["out_channels"][-1], num_heads=attn_num_heads)
        self.post_attn_avg_pool = nn.AvgPool1d(kernel_size=self.site_param_dict["out_lengths"][-1])

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout_pr)

        self.linear_layer_sizes: list[int] = linear_layer_sizes if linear_layer_sizes is not None else []

        # Create linear layers
        self.linear_layer_sizes.insert(0, self.site_param_dict["out_channels"][-1])
        self.linear_layer_sizes.append(num_classes)

        # Put linear layers into Sequential module
        lls = []

        for i in range(len(self.linear_layer_sizes) - 1):
            lls.append(nn.Linear(self.linear_layer_sizes[i], self.linear_layer_sizes[i + 1]))
            lls.append(nn.ELU())
            lls.append(self.dropout)

        self.linears = nn.Sequential(*lls)

    def get_flat_size(self):
        flat_site_size = self.site_param_dict["out_channels"][-1] * self.site_param_dict["out_lengths"][-1]
        flat_kin_size = self.kin_param_dict["out_channels"][-1] * self.kin_param_dict["out_lengths"][-1]

        return flat_site_size, flat_kin_size

    def calculate_cNN_params(self, kin_or_site: Literal["kin", "site"]) -> tuple[list, list, list, list]:
        if kin_or_site == "kin":
            param = self.kin_param_dict
            emb = self.emb_dim_kin
            first_width = self.kin_len
            num_conv = self.num_conv_kin
        elif kin_or_site == "site":
            param = self.site_param_dict
            emb = self.emb_dim_site
            first_width = self.site_len
            num_conv = self.num_conv_site
        else:
            raise ValueError("kin_or_site must be 'kin' or 'site'")

        calculated_pools = []
        calculated_in_channels = []
        calculated_do_flatten = []
        calculated_do_transpose = []

        for i in range(num_conv):
            calculated_do_transpose.append(i == 0)
            calculated_do_flatten.append(False)
            calculated_in_channels.append(emb if i == 0 else param["out_channels"][i - 1])
            input_width = first_width if i == 0 else param["out_lengths"][i - 1]
            calculated_pools.append(
                cNNUtils.desired_conv_then_pool_shape(
                    input_width,
                    None,
                    param["out_lengths"][i],
                    None,
                    kernel_size=param["kernels"][i],
                    err_message=f"{kin_or_site} CNNs",
                )[0]
            )

        return (
            calculated_pools,
            calculated_in_channels,
            calculated_do_flatten,
            calculated_do_transpose,
        )

    def forward(self, site_seq, kin_seq):
        emb_site = self.emb_site(site_seq)
        emb_kin = self.emb_kin(kin_seq)

        cnn_out_site = self.site_cnns(emb_site)  # Includes MaxPool'ing
        cnn_out_kin = self.kin_cnns(emb_kin)

        transp_out_site = self.transpose(cnn_out_site)
        transp_out_kin = self.transpose(cnn_out_kin)

        attn_out, _ = self.attn(transp_out_site, transp_out_kin)
        attn_transp_out = self.transpose(attn_out)
        avg_pool_out = self.post_attn_avg_pool(attn_transp_out)

        squeeze_out = self.squeeze_2(avg_pool_out)
        act_out = self.activation(squeeze_out)

        linears_out = self.linears(act_out)
        out = self.squeeze_1(linears_out)

        return out
