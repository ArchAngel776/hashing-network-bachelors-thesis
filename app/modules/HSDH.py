from torch import no_grad
from torch.nn import Module, Sigmoid, Sequential, Linear
from app.modules.HashGenerator import HashGenerator


class HSDH(Module):
    def __init__(self, hash_length, alpha):
        super().__init__()

        self._hash_generator = HashGenerator(hash_length, alpha)
        self._fully_connected_layer = Sequential(
            Linear(in_features=1, out_features=1),
            Sigmoid()
        )

    def forward(self, image_i, image_j):
        hash_i = self._hash_generator(image_i)
        hash_j = self._hash_generator(image_j)

        similarity = (hash_i * hash_j).sum(dim=1, keepdim=True)
        return self._fully_connected_layer(similarity)

    @no_grad()
    def generate(self, image):
        return self._hash_generator.generate(image)
