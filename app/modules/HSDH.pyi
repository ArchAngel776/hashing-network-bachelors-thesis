from typing import Self
from torch import Tensor, no_grad, device as dev
from torch.nn import Module, Linear
from torch.utils.data import DataLoader
from app.modules.HashGenerator import HashGenerator


class HSDH(Module):
    _hash_generator: HashGenerator
    _fully_connected_layer: Linear

    def __init__(self: Self, hash_length: int) -> None: ...

    @no_grad()
    def fit_scaler(self: Self, dataloader: DataLoader[tuple[Tensor, Tensor]], device: dev) -> None: ...

    def forward(self: Self, image_i: Tensor, image_j: Tensor) -> Tensor: ...

    @no_grad()
    def generate(self: Self, image: Tensor) -> Tensor: ...
