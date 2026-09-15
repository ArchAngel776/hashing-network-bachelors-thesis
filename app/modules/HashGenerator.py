from torch import no_grad, cat, where, ones_like
from torch.nn import Module, Linear, init
from torch.nn.functional import softsign
from sklearn.preprocessing import StandardScaler as SciKitScaler
from sklearn.decomposition import PCA as SciKitPCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from app.modules.DINO import DINO
from app.modules.StandardScaler import StandardScaler
from app.modules.PCA import PCA
from app.modules.NCA import NCA


class HashGenerator(Module):
    FEATURES_VECTOR_SIZE = 384

    def __init__(self, hash_length, pca_components = None, nca_components = None):
        super().__init__()

        if pca_components is not None and nca_components is not None:
            raise ValueError("PCA and NCA cannot be enabled simultaneously.")

        self._pca_components    = pca_components
        self._nca_components    = nca_components

        self._dino              = DINO()
        self._standard_scaler   = StandardScaler(HashGenerator.FEATURES_VECTOR_SIZE)

        if pca_components is None:
            self._pca           = None
        else:
            self._pca           = PCA(
                in_features     = HashGenerator.FEATURES_VECTOR_SIZE,
                n_components    = pca_components
            )

        if nca_components is None:
            self._nca           = None
        else:
            self._nca           = NCA(
                in_features     = HashGenerator.FEATURES_VECTOR_SIZE,
                n_components    = nca_components
            )

        if pca_components is None and nca_components is None:
            self._hash_projection   = Linear(
                in_features     = HashGenerator.FEATURES_VECTOR_SIZE,
                out_features    = hash_length
            )
        elif pca_components is not None:
            self._hash_projection   = Linear(
                in_features     = pca_components,
                out_features    = hash_length
            )
        elif nca_components is not None:
            self._hash_projection   = Linear(
                in_features     = nca_components,
                out_features    = hash_length
            )

        init.normal_(self._hash_projection.weight, mean=0, std=0.01)
        init.zeros_(self._hash_projection.bias)

    @no_grad()
    def fit_scaler(self, dataloader, device):
        features = [
            self._dino(images.to(device)).float().cpu()
            for images, _ in dataloader
        ]

        scikit_scaler = SciKitScaler()
        scikit_scaler.fit(cat(features, dim=0).numpy())

        self._standard_scaler.fit_params(scikit_scaler.mean_, scikit_scaler.scale_)

    @no_grad()
    def fit_pca(self, dataloader, device):
        assert self._pca_components is not None
        assert self._pca is not None

        features = [
            self._standard_scaler(self._dino(images.to(device))).float().cpu()
            for images, _ in dataloader
        ]

        scikit_pca = SciKitPCA(
            n_components=self._pca_components,
            svd_solver="full"
        )
        scikit_pca.fit(cat(features, dim=0).numpy())

        self._pca.fit(scikit_pca.mean_, scikit_pca.components_)

    @no_grad()
    def fit_nca(self, dataloader, device):
        assert self._nca_components is not None
        assert self._nca is not None

        features = []
        labels = []

        for batch_images, batch_labels in dataloader:
            features.append(self._standard_scaler(self._dino(batch_images.to(device))).float().cpu())
            labels.append(batch_labels)

        scikit_nca = NeighborhoodComponentsAnalysis(
            n_components=self._nca_components,
            random_state=42
        )
        scikit_nca.fit(cat(features, dim=0).numpy(), cat(labels, dim=0).numpy())

        self._nca.fit(scikit_nca.components_)

    def load_dino_domain_pretrained_weights(self):
        self._dino.load_domain_pretrained_weights()

    def get_hash_project(self, image):
        features_vector = self._dino(image)
        features = self._standard_scaler(features_vector)

        if self._pca is not None:
            features = self._pca(features)
        elif self._nca is not None:
            features = self._nca(features)

        return self._hash_projection(features)

    def forward(self, image):
        hash_project = self.get_hash_project(image)

        return softsign(hash_project)

    @no_grad()
    def generate(self, image):
        hash_project = self.get_hash_project(image)

        return where(hash_project >= 0, ones_like(hash_project), -ones_like(hash_project))

    @property
    def hash_projection(self):
        return self._hash_projection
