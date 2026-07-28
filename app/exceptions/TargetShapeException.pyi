from torch import Tensor
from typing import Self


class TargetShapeException(Exception):
    _target: Tensor
    _size: int

    def __init__(self: Self, target: Tensor, size: int) -> None: ...

    def __str__(self: Self) -> str: ...
