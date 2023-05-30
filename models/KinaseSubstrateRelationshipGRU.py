"""DeepKS model containing a `torch.nn.GRU` layer to help embed the kinases/sites. Otherwise analogous to \
    `models.KinaseSubstrateRelationshipATTN`."""
import torch, torch.nn as nn
from ..tools.formal_layers import Transpose, Squeeze, Unsqueeze
from ..tools.model_utils import cNNUtils as cNNUtils
from .KSRProtocol import KSR
from typing import Literal, Tuple
from pprint import pformat

torch.use_deterministic_algorithms(True)

from ..config.logging import get_logger

logger = get_logger()
"""Logger for this module."""


class MHAttention(nn.Module):
    """Wrapper (that adds an optional linear transform) for `nn.MultiheadAttention` (MHA)"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        do_transform: bool = False,
        q_k_v_transform_in_features: Tuple[int, int, int] | None = None,
    ):
        """Initialize the MHAttention module.

        Parameters
        ----------
        embed_dim : int
            The dimension of the embedding space for multi-head attention.
        num_heads : int
            The number of attention heads.
        do_transform : bool, optional
            Whether or not to do a linear transform on the input tensors before running data through `forward`, by default False
        q_k_v_transform_in_features :  optional
            The Q in-dimension, the K in-dimension, and the V in-dimension for the linear transform, by default None. Cannot be None if `do_transform` is True.
        """
        self.do_transform = do_transform
        """Whether or not to do a linear transform on the input tensors before running data through `forward`."""
        self.num_heads = num_heads
        """The number of attention heads."""
        self.embed_dim = embed_dim
        """The dimension of the embedding space for multi-head attention."""
        if self.do_transform:
            assert q_k_v_transform_in_features is not None
            self.qif: int
            """The Q in-dimension for the Q-linear layer"""
            self.kif: int
            """The K in-dimension for the K-linear layer"""
            self.vif: int
            """The V in-dimension for the V-linear layer"""
            self.qif, self.kif, self.vif = q_k_v_transform_in_features
            self.trans_Q = nn.Linear(self.qif, self.embed_dim)
            """The linear transform for the Q tensor."""
            self.trans_K = nn.Linear(self.kif, self.embed_dim)
            """The linear transform for the K tensor."""
            self.trans_V = nn.Linear(self.vif, self.embed_dim)
            """The linear transform for the V tensor."""

        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        """The multi-head attention module."""

    def forward(self, site: torch.Tensor, kin: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MHAttention module.

        Parameters
        ----------
        site :
            Tensor representing a site of shape ``(batch_size, site_length, emb_dim)``
        kin :
            Tensor representing a kinase of shape ``(batch_size, kin_length, emb_dim)``

        Returns
        -------
            Resultant Tensor of shape ``(batch_size, kin_length, emb_dim)``
        """
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
    """Module that applies a single CNNs to the input data, but allows for easy chaining of multiple `MultipleCNN` s."""

    def __init__(
        self,
        out_channels: int,
        conv_kernel_size: int,
        pool_kernel_size: int,
        in_channels: int,
        do_flatten: bool = False,
        do_transpose: bool = False,
    ):
        """Initialize the MultipleCNN module.

        Parameters
        ----------
        out_channels : int
            The number of output channels for the convolutional layer.
        conv_kernel_size : int
            The kernel size for the convolutional layer.
        pool_kernel_size : int
            The kernel size/stride for the pooling layer.
        in_channels : int
            The number of input channels for the convolutional layer.
        do_flatten : bool, optional
            Whether or not to flatten the output, by default False
        do_transpose : bool, optional
            Whether or not to transpose the 1st and 2nd dimensions of the output, by default False
        """
        super().__init__()
        self.do_transpose = do_transpose
        """Whether or not to transpose the 1st and 2nd dimensions of the output."""
        if self.do_transpose:
            self.transpose = Transpose(1, 2)
            "The transposing layer, if transposition needs to be done."
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=conv_kernel_size)
        """The underlying convolutional layer."""
        self.activation = nn.ELU()  # nn.Tanh()
        """The activation function to apply to the output of the convolutional layer."""
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)
        """The pooling layer to apply to the output of the activated convolutional layer."""
        self.do_flatten = do_flatten
        """Whether or not to flatten the output."""
        if self.do_flatten:
            self.flat = nn.Flatten(1)
            """The flattening layer, if flattening needs to be done"""

    def forward(self, x: torch.Tensor, ret_size: bool = False) -> torch.Tensor:
        """Forward pass of the MultipleCNN module.

        Parameters
        ----------
        x :
            The input tensor of shape ``(batch_size, in_channels, seq_length)``
        ret_size : bool, optional
            Should be Depricated, by default False

        Returns
        -------
            The output tensor of shape ``(batch_size, out_channels, seq_length)``
        """
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
class KinaseSubstrateRelationshipGRU(KSR):
    """DeepKS model containing a `torch.nn.MultiheadAttention` layer as the attention layer. Otherwise analogous to \
    `models.KinaseSubstrateRelationshipClassic`."""

    def __init__(
        self,
        num_classes: int = 1,
        linear_layer_sizes: list[int] = [],
        emb_dim_site: int = 22,
        emb_dim_kin: int = 22,
        site_param_dict: dict[str, list[int]] = {"kernels": [4], "out_lengths": [12], "out_channels": [11]},
        kin_param_dict: dict[str, list[int]] = {"kernels": [10], "out_lengths": [12], "out_channels": [11]},
        dropout_pr: float = 0.3,
        site_len: int = 15,
        kin_len: int = 2064,
        num_aa: int = 22,  # 20 + X + padding
        attn_num_heads: int = 8,
        padding_idx: int = 21,
        gru_hidden_dim_kin: int = 64,
        gru_hidden_dim_site: int = 64,
    ):
        """Initialize the KinaseSubstrateRelationshipATTN model.

        Parameters
        ----------
        num_classes : int, optional
            The number of classes to predict, by default 1. This would be changed if we wanted to output a probability distribution over the classes, rather than a single scalar.
        linear_layer_sizes : list[int], optional
            Input sizes of additional linear layers between the output of CNN pooling and the final prediction, by default ``[]``
        emb_dim_site : int, optional
            The embedding dimension of the site sequence, by default 22
        emb_dim_kin : int, optional
            The embedding dimension of the kinase sequence, by default 22
        site_param_dict : dict[str, list[int]], optional
            A dictionary mapping parameters to lists of values for the site CNN(s), by default ``{"kernels": [4], "out_lengths": [12], "out_channels": [11]}``. The keys should be ``"kernels"``, ``"out_lengths"``, and ``"out_channels"``. The values should be lists of integers.
        kin_param_dict : dict[str, list[int]], optional
            A dictionary mapping parameters to lists of values for the kinase CNN(s), by default ``{"kernels": [10], "out_lengths": [12], "out_channels": [11]}``. The keys should be ``"kernels"``, ``"out_lengths"``, and ``"out_channels"``. The values should be lists of integers.
        dropout_pr : float, optional
            The dropout layers' probability, by default 0.3
        site_len : int, optional
            The length of site the model should expect, by default 15
        kin_len : int, optional
            The length of kinase the model should expect, by default 2064
        num_aa : int, optional
            The number of amino acids (or by extension, n-grams) the model should expect, by default 22. (20 amino acids + "X" to pad flanking sequences whose central kinase is < 7 residues away from the ending + padding token for kinases < than 2064 residues long)
        padding_idx : int, optional
            The index of the kinase padding token within the ``num_aa`` tokens, by default 21
        """
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
        """The number of convolutional layers to apply to the site sequence."""
        self.num_conv_kin = num_conv_kin
        """The number of convolutional layers to apply to the kinase sequence."""
        self.emb_dim_site = emb_dim_site
        """The embedding dimension of the site sequence."""
        self.emb_dim_kin = emb_dim_kin
        """The embedding dimension of the kinase sequence."""
        self.gru_hidden_dim_site = gru_hidden_dim_site
        """The hidden dimension of the site sequence GRU."""
        self.gru_hidden_dim_kin = gru_hidden_dim_kin
        """The hidden dimension of the kinase sequence GRU."""

        self.emb_site = nn.Embedding(num_aa, self.emb_dim_site)
        """The site sequence embedding layer."""
        self.emb_kin = nn.Embedding(num_aa, self.emb_dim_kin, padding_idx=padding_idx)
        """The kinase sequence embedding layer."""
        self.gru_site = nn.GRU(input_size=self.emb_dim_site, hidden_size=self.gru_hidden_dim_site, batch_first=True)
        """The site sequence GRU layer."""
        self.gru_kin = nn.GRU(input_size=self.emb_dim_kin, hidden_size=self.gru_hidden_dim_kin, batch_first=True)
        """The kinase sequence GRU layer."""
        self.site_param_dict = site_param_dict
        """A dictionary mapping parameters to lists of values for the site CNN(s). The keys should be ``"kernels"``, ``"out_lengths"``, and ``"out_channels"``. The values should be lists of integers."""
        self.kin_param_dict = kin_param_dict
        """A dictionary mapping parameters to lists of values for the kinase CNN(s). The keys should be ``"kernels"``, ``"out_lengths"``, and ``"out_channels"``. The values should be lists of integers."""
        self.site_len = site_len
        """The length of site the model is expecting."""
        self.kin_len = kin_len
        """The length of kinase the model is expecting."""
        self.squeeze_0 = Squeeze(0)
        """A layer to squeeze the output of the site CNN(s) in the 0th dimension."""
        self.squeeze_1 = Squeeze(1)
        """A layer to squeeze the output of the site CNN(s) in the 1st dimension."""
        self.squeeze_2 = Squeeze(2)
        """A layer to squeeze the output of the kinase CNN(s) in the 2nd dimension."""
        self.unsqueeze_1 = Unsqueeze(1)

        # site_param_dict['out_channels'] = [1 for _ in range(num_conv_site)]
        # kin_param_dict['out_channels'] = [1 for _ in range(num_conv_kin)]

        pools_site, in_channels_site, do_flatten_site, do_transpose_site = self.calculate_cNN_params("site")
        pools_kin, in_channels_kin, do_flatten_kin, do_transpose_kin = self.calculate_cNN_params("kin")

        in_channels_site[0] = 1
        in_channels_kin[0] = 1

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

        self.linear_layer_sizes = linear_layer_sizes

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
            param = self.kin_param_dict
            emb = self.emb_dim_kin
            first_width = self.gru_hidden_dim_kin
            num_conv = self.num_conv_kin
        elif kin_or_site == "site":
            param = self.site_param_dict
            emb = self.emb_dim_site
            first_width = self.gru_hidden_dim_site
            num_conv = self.num_conv_site
        else:
            raise ValueError("kin_or_site must be 'kin' or 'site'")

        calculated_pools = []
        calculated_in_channels = []
        calculated_do_flatten = []
        calculated_do_transpose = []

        for i in range(num_conv):
            calculated_do_transpose.append(False)
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

    def forward(self, site_seq, kin_seq):
        emb_site = self.emb_site(site_seq)
        emb_kin = self.emb_kin(kin_seq)  # (batch_size, seq_len, emb_dim)

        del_1, grued_site = self.gru_site(emb_site)
        del_2, grued_kin = self.gru_kin(emb_kin)  # (num_gru_layers == 1, batch_size, hidden_dim)

        grued_site_flat = self.squeeze_0(grued_site)
        grued_kin_flat = self.squeeze_0(grued_kin)  # (batch_size, hidden_dim)

        grued_site_channeled = self.unsqueeze_1(grued_site_flat)
        grued_kin_channeled = self.unsqueeze_1(grued_kin_flat)  # (batch_size, 1, hidden_dim)

        cnn_out_site = self.site_cnns(grued_site_channeled)  # Includes MaxPool'ing
        cnn_out_kin = self.kin_cnns(grued_kin_channeled)  # ()

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
