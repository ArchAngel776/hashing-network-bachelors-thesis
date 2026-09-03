from torch import no_grad
from torch.nn import Module, Linear, init
from torch.nn.functional import sigmoid
from app.modules.HashGenerator import HashGenerator


class HSDH(Module):
    def __init__(self, hash_length):
        super().__init__()

        self._hash_generator = HashGenerator(hash_length)
        self._fully_connected_layer = Linear(in_features=1, out_features=1)

        init.ones_(self._fully_connected_layer.weight)
        init.zeros_(self._fully_connected_layer.bias)

    def forward(self, image_i, image_j):
        hash_i = self._hash_generator(image_i)
        hash_j = self._hash_generator(image_j)

        similarity = (hash_i * hash_j).sum(dim=1, keepdim=True)
        return sigmoid(self._fully_connected_layer(similarity))

    @no_grad()
    def generate(self, image):
        return self._hash_generator.generate(image)
