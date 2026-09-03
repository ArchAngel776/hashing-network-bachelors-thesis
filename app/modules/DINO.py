from pathlib import Path
import torch
from torch import hub, no_grad
from torch.nn import Module


class DINO(Module):
    CHECKPOINTS_PATH = Path("data/checkpoints/dino-kather")

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

    def load_domain_pretrained_weights(self):
        if not DINO.CHECKPOINTS_PATH.joinpath("dino-domain-pretrained.pth").exists():
            print("No domain pretrained checkpoint exists. DINO will start with generally pretrained weights.")
            return

        state = torch.load(
            DINO.CHECKPOINTS_PATH.joinpath("dino-domain-pretrained.pth"),
            map_location="cpu",
            weights_only=True
        )

        self._model.load_state_dict(state)

    def train(self, mode = True):
        super().train(False)
        return self

    @no_grad()
    def forward(self, x):
        return self._model(x)
