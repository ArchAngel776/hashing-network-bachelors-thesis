from _typeshed import DataclassInstance
from typing import Any


def check_dict_compatibility(source: dict[str, Any], target: type[DataclassInstance]) -> bool: ...
