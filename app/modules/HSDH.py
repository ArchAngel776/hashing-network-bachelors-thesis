from torch import sum, where, no_grad
from torch.nn import Module, Sigmoid, Sequential, Linear
from app.modules.HashGenerator import HashGenerator


class HSDH(Module):
    def __init__(self, hash_length, alpha):
        super().__init__()

        self._hash_generator = HashGenerator(hash_length, alpha)
        self._prediction = Sequential(
            Linear(
                in_features=1,
                out_features=1
            ),
            Sigmoid()
        )

    def forward(self, image_i, image_j, target):
        hash_i = self._hash_generator(image_i)
        hash_j = self._hash_generator(image_j)

        similarity = sum(hash_i * hash_j, dim=1, keepdim=True)
        prediction = self._prediction(similarity)
        #return prediction, similarity, hash_i, hash_j
        return where(target == 1., prediction, 1. - prediction)

    @no_grad()
    def compare(self, image_i, image_j):
        hash_i = self._hash_generator.generate(image_i)
        hash_j = self._hash_generator.generate(image_j)

        return sum(hash_i != hash_j, dim=1)
