from typing import Self


class PCANotFittedException(Exception):
    def __str__(self: Self) -> str: ...
