from typing import Optional, Self
from PIL.Image import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform


class KatherRetrievalDataset(Dataset[tuple[Image, int]]):
    _sources: list[tuple[str, int]]

    _transform: Optional[Transform]
    _target_transform: Optional[Transform]

    def __init__(
        self,
        sources: list[tuple[str, int]],
        transform: Optional[Transform] = None,
        target_transform: Optional[Transform] = None
    ) -> None: ...

    def __len__(self: Self) -> int: ...

    def __getitem__(self: Self, index: int) -> tuple[Image, int]: ...
