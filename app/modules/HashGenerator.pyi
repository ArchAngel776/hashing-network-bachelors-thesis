from typing import Self, ClassVar
from torch import Tensor, no_grad, device as dev
from torch.nn import Module, Linear
from torch.utils.data import DataLoader
from app.modules.DINO import DINO
from app.modules.StandardScaler import StandardScaler


class HashGenerator(Module):
    FEATURES_VECTOR_SIZE: ClassVar[int]

    _dino: DINO
    _standard_scaler: StandardScaler
    _hash_projection: Linear

    def __init__(self: Self, hash_length: int) -> None: ...

    @no_grad()
    def fit_scaler(self: Self, dataloader: DataLoader[tuple[Tensor, Tensor]], device: dev) -> None: ...

    def load_dino_domain_pretrained_weights(self: Self) -> None: ...

    def forward(self: Self, image: Tensor) -> Tensor: ...

    @no_grad()
    def generate(self: Self, image: Tensor) -> Tensor: ...
