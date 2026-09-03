import torch
from torch import zeros, ones, no_grad, as_tensor
from torch.nn import Module
from app.exceptions.ScalerNotFittedException import ScalerNotFittedException


class StandardScaler(Module):
    def __init__(self, vector_size):
        super().__init__()

        self.register_buffer("_mean",    zeros(vector_size,   dtype=torch.float32))
        self.register_buffer("_scale",   ones(vector_size,    dtype=torch.float32))

        self.register_buffer("_fitted", torch.tensor(False, dtype=torch.bool))

    @no_grad()
    def fit_params(self, mean, scale):
        self._mean.copy_(as_tensor(mean,      dtype=self._mean.dtype,   device=self._mean.device))
        self._scale.copy_(as_tensor(scale,    dtype=self._scale.dtype,  device=self._scale.device))

        self._fitted.fill_(True)

    def forward(self, x):
        if not self._fitted.item():
            raise ScalerNotFittedException()

        return (x - self._mean) / self._scale
