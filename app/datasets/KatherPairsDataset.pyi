from PIL.Image import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform
from typing import Self, Optional
from app.datasets.KatherDataset import KatherDataset


class KatherPairsDataset(Dataset[tuple[Image, Image, bool]]):
    RANDOM_SEED: int = ...
    TRAINING_RATE: float = ...

    _transform: Optional[Transform]
    _target_transform: Optional[Transform]

    _data: list[tuple[str, str, bool]]
    _sources: list[tuple[str, int]]
    _groups: dict[int, list[str]]
    _labels: list[int]

    def __init__(
        self: Self,
        dataset: KatherDataset,
        train: bool,
        transform: Optional[Transform] = None,
        target_transform: Optional[Transform] = None
    ) -> None: ...

    def __len__(self: Self) -> int: ...

    def __getitem__(self: Self, index: int) -> tuple[Image, Image, bool]: ...

    def resample(self: Self, epoch: int) -> None: ...

    def negative_labels(self: Self, label: int) -> list[int]: ...
