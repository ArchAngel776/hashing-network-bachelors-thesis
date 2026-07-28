from torch import Tensor, no_grad
from torch.nn import Module, BatchNorm1d, Linear, SELU
from torchvision.models.mobilenetv2 import MobileNetV2
from typing import Self, ClassVar
from app.modules.SignumApprox import SignumApprox


class HashGenerator(Module):
    FEATURES_VECTOR_SIZE: ClassVar[int]

    _mobile_net: MobileNetV2

    _batch_normalization: BatchNorm1d
    _activation: SELU

    _hash_projection: Linear
    _hash_activation: SignumApprox

    def __init__(self: Self, hash_length: int, alpha: float) -> None: ...

    def forward(self: Self, image: Tensor) -> Tensor: ...

    @no_grad()
    def generate(self: Self, image: Tensor) -> Tensor: ...
