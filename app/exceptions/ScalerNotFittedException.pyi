from typing import Self


class ScalerNotFittedException(Exception):
    def __str__(self: Self) -> str: ...
