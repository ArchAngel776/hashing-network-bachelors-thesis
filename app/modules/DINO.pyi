from typing import ClassVar, Self
from pathlib import Path
from torch import Tensor
from torch.nn import Module
from torchvision.models import VisionTransformer


class DINO(Module):
    CHECKPOINTS_PATH: ClassVar[Path]

    _model: VisionTransformer

    def __init__(self: Self) -> None: ...

    def load_domain_pretrained_weights(self: Self) -> None: ...

    def train(self: Self, mode: bool = True) -> Self: ...

    def forward(self: Self, x: Tensor) -> Tensor: ...
