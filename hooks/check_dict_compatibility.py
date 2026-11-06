from dataclasses import is_dataclass, fields


def check_dict_compatibility(source, target):
    assert is_dataclass(target)

    for field in fields(target):
        if field.name not in source:
            return False

    return True
