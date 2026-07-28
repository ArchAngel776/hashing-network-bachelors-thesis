from torch import Tensor
from torch.nn import Module
from typing import Self, ClassVar


class HSDHLoss(Module):
    M_1: ClassVar[float]
    M_2: ClassVar[float]

    _beta: float

    def __init__(self: Self, beta: float) -> None: ...

    def forward(self: Self, prediction: Tensor, target: Tensor) -> Tensor: ...
