from torch import Tensor
from torch.nn import Module
from typing import Self


class SignumApprox(Module):
    _alpha: float = ...

    def __init__(self: Self, alpha: float) -> None: ...

    def forward(self: Self, x: Tensor) -> Tensor: ...
