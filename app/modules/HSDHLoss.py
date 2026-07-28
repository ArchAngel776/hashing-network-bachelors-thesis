from torch import clamp_min
from torch.nn import Module
from app.exceptions.BetaParameterValueException import BetaParameterValueException
from app.exceptions.TargetShapeException import TargetShapeException


class HSDHLoss(Module):
    M_1 = 0.4
    M_2 = 1.0

    def __init__(self, beta):
        super().__init__()

        if beta < 0:
            raise BetaParameterValueException(beta)

        self._beta = beta

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise TargetShapeException(target, prediction.ndim)

        return (
            (
                (1. - target) *
                clamp_min(prediction - HSDHLoss.M_1, 0.0).square()
            ) +
            (
                target *
                clamp_min(HSDHLoss.M_2 - prediction, 0.0).square()
            ) +
            (self._beta * (target - prediction).square())
        ).sum()
