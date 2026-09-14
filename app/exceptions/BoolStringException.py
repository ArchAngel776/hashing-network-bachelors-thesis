class BoolStringException(Exception):
    def __init__(self, target):
        self._target = target

    def __str__(self):
        return f"Target \"{self._target}\" cannot be converted to the bool type."
