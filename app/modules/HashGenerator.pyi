from typing import Self, ClassVar
from torch import Tensor, no_grad
from torch.nn import Module, BatchNorm1d, Linear
from app.modules.DINO import DINO


class HashGenerator(Module):
    FEATURES_VECTOR_SIZE: ClassVar[int]

    _dino: DINO
    _batch_normalization: BatchNorm1d
    _hash_projection: Linear

    def __init__(self: Self, hash_length: int) -> None: ...

    def forward(self: Self, image: Tensor) -> Tensor: ...

    @no_grad()
    def generate(self: Self, image: Tensor) -> Tensor: ...
