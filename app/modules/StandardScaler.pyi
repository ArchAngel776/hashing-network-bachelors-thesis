from typing import Self
from numpy.typing import ArrayLike
from torch import Tensor, no_grad
from torch.nn import Module


class StandardScaler(Module):
    _mean: Tensor
    _scale: Tensor

    _fitted: Tensor

    def __init__(self: Self, vector_size: int) -> None: ...

    @no_grad()
    def fit_params(self: Self, mean: ArrayLike, scale: ArrayLike) -> None: ...

    def forward(self: Self, x: Tensor) -> Tensor: ...
