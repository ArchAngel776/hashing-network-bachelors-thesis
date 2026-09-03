from typing import Self
from torch import Tensor
from torch.nn import Module
from torchvision.models import VisionTransformer


class DINO(Module):
    _model: VisionTransformer

    def __init__(self: Self) -> None: ...

    def train(self: Self, mode: bool = True) -> Self: ...

    def forward(self: Self, x: Tensor) -> Tensor: ...
