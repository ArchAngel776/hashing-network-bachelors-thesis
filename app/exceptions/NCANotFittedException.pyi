from typing import Self


class NCANotFittedException(Exception):
    def __str__(self: Self) -> str: ...
