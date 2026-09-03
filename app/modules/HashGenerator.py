from torch import no_grad, where, ones_like
from torch.nn import Module, Linear, BatchNorm1d, init
from torch.nn.functional import selu
from app.modules.DINO import DINO
from app.modules.SignumApprox import SignumApprox


class HashGenerator(Module):
    FEATURES_VECTOR_SIZE = 384

    def __init__(self, hash_length, alpha):
        super().__init__()

        self._dino = DINO()

        self._batch_normalization = BatchNorm1d(num_features=HashGenerator.FEATURES_VECTOR_SIZE)

        self._hash_projection = Linear(
            in_features=HashGenerator.FEATURES_VECTOR_SIZE,
            out_features=hash_length
        )

        init.normal_(self._hash_projection.weight, mean=0, std=0.01)
        init.zeros_(self._hash_projection.bias)

        self._hash_activation = SignumApprox(alpha)

    def forward(self, image):
        features_vector = self._dino(image)
        normalized_features = self._batch_normalization(features_vector)

        hash_project = self._hash_projection(selu(normalized_features))

        return self._hash_activation(hash_project)

    @no_grad()
    def generate(self, image):
        features_vector = self._dino(image)
        normalized_features = self._batch_normalization(features_vector)

        hash_project = self._hash_projection(selu(normalized_features))

        return where(hash_project >= 0, ones_like(hash_project), -ones_like(hash_project))
