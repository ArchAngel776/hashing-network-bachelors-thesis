from torch import no_grad
from torch.nn import Module, Linear, init
from torch.nn.functional import sigmoid
from app.modules.HashGenerator import HashGenerator


class HSDH(Module):
    def __init__(self, hash_length, pca_components = None, nca_components = None):
        super().__init__()

        self._hash_generator = HashGenerator(hash_length, pca_components, nca_components)
        self._fully_connected_layer = Linear(in_features=1, out_features=1)

        init.ones_(self._fully_connected_layer.weight)
        init.zeros_(self._fully_connected_layer.bias)

    @no_grad()
    def fit_scaler(self, dataloader, device):
        self._hash_generator.fit_scaler(dataloader, device)

    @no_grad()
    def fit_pca(self, dataloader, device):
        self._hash_generator.fit_pca(dataloader, device)

    @no_grad()
    def fit_nca(self, dataloader, device):
        self._hash_generator.fit_nca(dataloader, device)

    def load_dino_domain_pretrained_weights(self):
        self._hash_generator.load_dino_domain_pretrained_weights()

    def forward(self, image_i, image_j):
        hash_i = self._hash_generator(image_i)
        hash_j = self._hash_generator(image_j)

        similarity = (hash_i * hash_j).sum(dim=1, keepdim=True)
        return sigmoid(self._fully_connected_layer(similarity))

    @no_grad()
    def generate(self, image):
        return self._hash_generator.generate(image)

    @property
    def hash_generator(self):
        return self._hash_generator

    @property
    def fully_connected_layer(self):
        return self._fully_connected_layer
