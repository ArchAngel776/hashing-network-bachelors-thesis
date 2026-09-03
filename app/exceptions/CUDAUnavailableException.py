class CUDAUnavailableException(Exception):
    def __str__(self):
        return "CUDA is not available."
