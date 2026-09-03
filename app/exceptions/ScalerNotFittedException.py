class ScalerNotFittedException(Exception):
    def __str__(self):
        return "Standard Scaler has not been fitted yet. You need to fit it, before usage."
