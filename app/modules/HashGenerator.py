from torch import no_grad, where, ones_like
from torch.nn import Module, Linear, BatchNorm1d, SELU, init
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from app.modules.SignumApprox import SignumApprox


class HashGenerator(Module):
    FEATURES_VECTOR_SIZE = 1000

    def __init__(self, hash_length, alpha):
        super().__init__()

        self._mobile_net = mobilenet_v2(
            weights=MobileNet_V2_Weights.IMAGENET1K_V2,
            num_classes=HashGenerator.FEATURES_VECTOR_SIZE
        )

        self._batch_normalization = BatchNorm1d(num_features=HashGenerator.FEATURES_VECTOR_SIZE)
        self._activation = SELU()

        self._hash_projection = Linear(
            in_features=HashGenerator.FEATURES_VECTOR_SIZE,
            out_features=hash_length
        )

        init.normal_(self._hash_projection.weight, mean=0, std=0.01)
        init.zeros_(self._hash_projection.bias)

        self._hash_activation = SignumApprox(alpha)

    def forward(self, image):
        features_vector = self._mobile_net(image)
        normalized_features = self._batch_normalization(features_vector)

        hash_project = self._hash_projection(self._activation(normalized_features))

        return self._hash_activation(hash_project)

    @no_grad()
    def generate(self, image):
        features_vector = self._mobile_net(image)
        normalized_features = self._batch_normalization(features_vector)

        hash_project = self._hash_projection(self._activation(normalized_features))

        return where(hash_project >= 0, ones_like(hash_project), -ones_like(hash_project))
