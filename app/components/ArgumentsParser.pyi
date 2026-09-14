from typing import TypeVar, Self, Optional, Callable

TType = TypeVar("TType")


class ArgumentsParser:
    _arguments: list[str]
    _options: dict[str, str]

    def __init__(self: Self, arguments: list[str]) -> None: ...

    def parse(self: Self) -> None: ...

    def get_option(self: Self, name: str, value: Callable[[str], TType]) -> Optional[TType]: ...
