from typing import Self


class CUDAUnavailableException(Exception):
    def __str__(self: Self) -> str: ...
