class NCANotFittedException(Exception):
    def __str__(self):
        return "NCA reductor has not been fitted yet. You need to fit it, before usage."
