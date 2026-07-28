from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform
from typing import Self, Optional


class KatherDataset(Dataset[tuple[str, int]]):
    _transform: Optional[Transform]
    _target_transform: Optional[Transform]

    _data: list[tuple[str, int]]
    _label: int | None

    def __init__(
        self: Self,
        source_dir: str,
        label: Optional[int] = None,
        transform: Optional[Transform] = None,
        target_transform: Optional[Transform] = None
    ) -> None: ...

    def __len__(self: Self) -> int: ...

    def __getitem__(self: Self, index: int) -> tuple[str, int]: ...

    def select_label(self, label: int) -> None: ...

    def deselect_label(self: Self) -> None: ...

    @property
    def data(self: Self) -> list[tuple[str, int]]: ...

    @property
    def labels(self: Self) -> list[int]: ...
