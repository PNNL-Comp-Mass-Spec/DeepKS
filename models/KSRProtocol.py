"""Module defining how Kinase-Substrate Relationship models should be structured."""
from typing import Protocol
from abc import ABC
import torch


class KSRProtocol(Protocol):
    """Protocol for Kinase-Substrate Relationship models."""

    def __init__(self, **kwargs) -> None:
        """KSRs should have a variable number of keyword initialization arguments."""
        ...  # pragma: no cover

    def forward(self, site_seq: torch.LongTensor, kin_seq: torch.LongTensor) -> torch.FloatTensor:
        """KSRs should have a forward method that takes in a site sequence and a kinase sequence and returns a tensor of logits.

        Parameters
        ----------
        site_seq :
            The site sequence, as a torch.LongTensor of shape ``(batch_size, seq_len)``
        kin_seq :
            The kinase sequence, as a torch.LongTensor of shape ``(batch_size, seq_len)``

        Returns
        -------
            The logits, as a torch.FloatTensor of shape ``(batch_size, seq_len, 1)``

        """
        ...  # pragma: no cover

    def __call__(self, site_seq: torch.LongTensor, kin_seq: torch.LongTensor) -> torch.FloatTensor:
        """KSRs should be callable, taking in a site sequence and a kinase sequence and returning a tensor of logits. Should be alias for ``forward``.

        Parameters
        ----------
        site_seq :
            The site sequence, as a torch.LongTensor of shape ``(batch_size, seq_len)``
        kin_seq :
            The kinase sequence, as a torch.LongTensor of shape ``(batch_size, seq_len)``

        Returns
        -------
            The logits, as a torch.FloatTensor of shape ``(batch_size, seq_len, 1)``

        """
        ...  # pragma: no cover


class KSR(torch.nn.Module, KSRProtocol, ABC):
    """Abstract base class for Kinase-Substrate Relationship models, inheriting from ``torch.nn.Module`` and ``KSRProtocol``."""

    pass
