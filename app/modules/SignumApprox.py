from torch.nn import Module


class SignumApprox(Module):
    def __init__(self, alpha):
        super().__init__()
        self._alpha = alpha

    def forward(self, x):
        return (self._alpha * x)/(1. + (self._alpha * x).square()).sqrt()
