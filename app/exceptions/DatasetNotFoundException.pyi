from typing import Self


class DatasetNotFoundException(Exception):
    _path: str

    def __init__(self: Self, path: str) -> None: ...

    def __str__(self: Self) -> str: ...
