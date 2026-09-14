from app.exceptions.BoolStringException import BoolStringException


def str_bool(source):
    match source.lower():
        case "true":
            return True
        case "false":
            return False
        case _:
            raise BoolStringException(source)
