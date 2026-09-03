from re import match


class ArgumentsParser:
    def __init__(self, arguments):
        self._arguments = arguments
        self._options   = {}

    def parse(self):
        for argument in self._arguments:
            option = match(r"^--(?P<name>[A-Za-z0-9\-_]+)=(?P<value>.+)$", argument)

            if option is None:
                continue

            name    = option.group("name")
            value   = option.group("value")

            self._options[name] = value

    def get_option(self, name, value):
        if name not in self._options:
            return None

        try:
            return value(self._options[name])
        except ValueError:
            return None
