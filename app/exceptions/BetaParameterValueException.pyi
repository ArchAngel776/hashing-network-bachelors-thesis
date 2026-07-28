from typing import Self


class BetaParameterValueException(Exception):
    _beta: float

    def __init__(self: Self, beta: float) -> None: ...

    def __str__(self: Self) -> str: ...
