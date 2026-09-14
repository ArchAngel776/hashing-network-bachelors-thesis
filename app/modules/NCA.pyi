from typing import Self
from numpy.typing import ArrayLike
from torch import Tensor, no_grad
from torch.nn import Module


class NCA(Module):
    _components: Tensor

    _fitted: Tensor

    def __init__(self: Self, in_features: int, n_components: int) -> None: ...

    @no_grad()
    def fit(self: Self, components: ArrayLike) -> None: ...

    def forward(self: Self, x: Tensor) -> Tensor: ...
