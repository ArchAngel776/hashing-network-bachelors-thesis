class BetaParameterValueException(Exception):
    def __init__(self, beta):
        super().__init__()
        self._beta = beta

    def __str__(self):
        return f"Beta parameter: {self._beta} is lower than 0."
