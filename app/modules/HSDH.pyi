from torch import Tensor, no_grad
from torch.nn import Module, Sigmoid, Sequential
from typing import Self
from app.modules.HashGenerator import HashGenerator


class HSDH(Module):
    _hash_generator: HashGenerator

    _prediction: Sequential

    def __init__(self: Self, hash_length: int, alpha: float) -> None: ...

    def forward(self: Self, image_i: Tensor, image_j: Tensor, target: Tensor) -> Tensor: ...

    @no_grad()
    def compare(self: Self, image_i: Tensor, image_j: Tensor) -> Tensor: ...
