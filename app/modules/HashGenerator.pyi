from typing import Self, ClassVar
from torch import Tensor, no_grad
from torch.nn import Module, BatchNorm1d, Linear
from app.modules.DINO import DINO
from app.modules.SignumApprox import SignumApprox


class HashGenerator(Module):
    FEATURES_VECTOR_SIZE: ClassVar[int]

    _dino: DINO

    _batch_normalization: BatchNorm1d

    _hash_projection: Linear
    _hash_activation: SignumApprox

    def __init__(self: Self, hash_length: int, alpha: float) -> None: ...

    def forward(self: Self, image: Tensor) -> Tensor: ...

    @no_grad()
    def generate(self: Self, image: Tensor) -> Tensor: ...
