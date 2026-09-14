from typing import Self


class BoolStringException(Exception):
    _target: str

    def __init__(self: Self, target: str) -> None: ...

    def __str__(self: Self) -> str: ...
