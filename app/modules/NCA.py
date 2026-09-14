import torch
from torch import zeros, no_grad, as_tensor
from torch.nn import Module
from app.exceptions.NCANotFittedException import NCANotFittedException


class NCA(Module):
    def __init__(self, in_features, n_components):
        super().__init__()

        self.register_buffer("_components", zeros((n_components, in_features), dtype=torch.float32))

        self.register_buffer("_fitted", torch.tensor(False, dtype=torch.bool))

    @no_grad()
    def fit(self, components):
        self._components.copy_(as_tensor(components, dtype=self._components.dtype, device=self._components.device))

        self._fitted.fill_(True)

    def forward(self, x):
        if not self._fitted.item():
            raise NCANotFittedException()

        return x @ self._components.transpose(0, 1)
