import nntplib
from typing import Protocol
from abc import ABC
import torch


class KSRProtocol(Protocol):
    def __init__(self, **kwargs) -> None:
        ...

    def forward(self, site_seq: torch.LongTensor, kin_seq: torch.LongTensor) -> torch.FloatTensor:
        ...

    def __call__(self, site_seq: torch.LongTensor, kin_seq: torch.LongTensor) -> torch.FloatTensor:
        ...


class KSR(torch.nn.Module, KSRProtocol, ABC):
    pass
