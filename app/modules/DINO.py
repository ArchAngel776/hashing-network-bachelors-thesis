from torch import hub, no_grad
from torch.nn import Module


class DINO(Module):
    def __init__(self):
        super().__init__()

        self._model = hub.load(
            "facebookresearch/dino:main", "dino_vits16",
            pretrained=True,
            trust_repo=True,
            verbose=False
        )

        self._model.requires_grad_(False)
        self._model.eval()

    def train(self, mode = True):
        super().train(False)
        return self

    @no_grad()
    def forward(self, x):
        return self._model(x)
