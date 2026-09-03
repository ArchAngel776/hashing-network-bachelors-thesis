from torch import no_grad, cat, where, ones_like
from torch.nn import Module, Linear, init
from torch.nn.functional import softsign
from sklearn.preprocessing import StandardScaler as SciKitScaler
from app.modules.DINO import DINO
from app.modules.StandardScaler import StandardScaler


class HashGenerator(Module):
    FEATURES_VECTOR_SIZE = 384

    def __init__(self, hash_length):
        super().__init__()

        self._dino              = DINO()
        self._standard_scaler   = StandardScaler(HashGenerator.FEATURES_VECTOR_SIZE)
        self._hash_projection   = Linear(
            in_features=HashGenerator.FEATURES_VECTOR_SIZE,
            out_features=hash_length
        )

        init.normal_(self._hash_projection.weight, mean=0, std=0.01)
        init.zeros_(self._hash_projection.bias)

    @no_grad()
    def fit_scaler(self, dataloader, device):
        features = [self._dino(images.to(device)).float().cpu() for images, _ in dataloader]

        scikit_scaler = SciKitScaler()
        scikit_scaler.fit(cat(features, dim=0).numpy())

        self._standard_scaler.fit_params(scikit_scaler.mean_, scikit_scaler.scale_)

    def forward(self, image):
        features_vector = self._dino(image)
        scaled_features = self._standard_scaler(features_vector)

        hash_project = self._hash_projection(scaled_features)

        return softsign(hash_project)

    @no_grad()
    def generate(self, image):
        features_vector = self._dino(image)
        scaled_features = self._standard_scaler(features_vector)

        hash_project = self._hash_projection(scaled_features)

        return where(hash_project >= 0, ones_like(hash_project), -ones_like(hash_project))
