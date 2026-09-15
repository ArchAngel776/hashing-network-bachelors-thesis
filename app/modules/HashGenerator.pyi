from typing import Self, ClassVar, Optional
from torch import Tensor, no_grad, device as dev
from torch.nn import Module, Linear
from torch.utils.data import DataLoader
from app.modules.DINO import DINO
from app.modules.StandardScaler import StandardScaler
from app.modules.PCA import PCA
from app.modules.NCA import NCA


class HashGenerator(Module):
    FEATURES_VECTOR_SIZE: ClassVar[int]

    _pca_components: Optional[int]
    _nca_components: Optional[int]

    _dino: DINO
    _standard_scaler: StandardScaler
    _pca: Optional[PCA]
    _nca: Optional[NCA]

    _hash_projection: Linear

    def __init__(
        self: Self,
        hash_length: int,
        pca_components: Optional[int] = None,
        nca_components: Optional[int] = None
    ) -> None: ...

    @no_grad()
    def fit_scaler(self: Self, dataloader: DataLoader[tuple[Tensor, Tensor]], device: dev) -> None: ...

    @no_grad()
    def fit_pca(self: Self, dataloader: DataLoader[tuple[Tensor, Tensor]], device: dev) -> None: ...

    @no_grad()
    def fit_nca(self: Self, dataloader: DataLoader[tuple[Tensor, Tensor]], device: dev) -> None: ...

    def load_dino_domain_pretrained_weights(self: Self) -> None: ...

    def get_hash_project(self: Self, image: Tensor) -> Tensor: ...

    def forward(self: Self, image: Tensor) -> Tensor: ...

    @no_grad()
    def generate(self: Self, image: Tensor) -> Tensor: ...

    @property
    def hash_projection(self: Self) -> Linear: ...
