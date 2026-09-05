class PCANotFittedException(Exception):
    def __str__(self):
        return "PCA reductor has not been fitted yet. You need to fit it, before usage."
