"""Contains custom layers for operations that don't exist as layers in PyTorch. Beneficial to see flow of data in the model summary.
"""

import torch
from torch.nn import Module
from torch import concat, squeeze, unsqueeze, mul, transpose


class Concatenation(Module):
    """Concatenates multiple tensors along a given dimension."""

    def __init__(self, dim=1):
        """Initializes the Concatenation layer.
        Parameters
        ----------
        dim : int, optional
            The dimension along which to concatenate, by default 1
        """
        super().__init__()
        self.dim = dim

    def forward(self, *inputs) -> torch.Tensor:
        """Concatenates multiple tensors along a given dimension.

        Parameters
        ----------
        *inputs :
            The tensors to concatenate

        Returns
        -------
        `torch.Tensor`
            The concatenated tensor
        """
        return concat(inputs, dim=self.dim)


class Squeeze(Module):
    """Squeezes a tensor along a given dimension."""

    def __init__(self, dim=1):
        """Initializes the Squeeze layer.
        Parameters
        ----------
        dim : int, optional
            The dimension along which to squeeze, by default 1

        Returns
        -------
        `torch.Tensor`
            The squeezed tensor
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Squeeze layer.


        Parameters
        ----------
        x : `torch.Tensor`
            The tensor to squeeze

        Returns
        -------
        `torch.Tensor`
            The squeezed tensor"""
        return squeeze(x, dim=self.dim)


class Unsqueeze(Module):
    """Unsqueezes a tensor along a given dimension."""

    def __init__(self, dim=1):
        """Initializes the Unsqueeze layer.
        Parameters
        ----------
        dim : int, optional
            The dimension along which to squeeze, by default 1

        Returns
        -------
        `torch.Tensor`
            The squeezed tensor
        """

        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Unsqueeze layer.


        Parameters
        ----------
        x : `torch.Tensor`
            The tensor to unsqueeze
        """
        return unsqueeze(x, dim=self.dim)


class Multiply(Module):
    """Multiplies two tensors elementwise and supports broadcasting."""

    def __init__(self):
        """Initializes the Multiply layer."""

        super().__init__()

    def forward(self, x, y):
        """The forward pass of the Multiply layer.

        Parameters
        ----------
        x : `torch.Tensor`
            The first tensor to multiply
        y : `torch.Tensor`
            The second tensor to multiply

        Returns
        -------
        `torch.Tensor`
            The multiplied tensor
        """
        return mul(x, y)


class Transpose(Module):
    """Transposes a tensor along two given dimensions."""

    def __init__(self, dim1, dim2):
        """Initializes the Transpose layer.

        Parameters
        ----------
        dim1 : int
            The first dimension to transpose
        dim2 : int
            The second dimension to transpose

        Returns
        -------
        `torch.Tensor`
            The transposed tensor"""
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        """The forward pass of the Transpose layer.

        Parameters
        ----------
        x : `torch.Tensor`
            The tensor to transpose

        Returns
        -------
        `torch.Tensor`
            The transposed tensor
        """
        return transpose(x, self.dim1, self.dim2)
