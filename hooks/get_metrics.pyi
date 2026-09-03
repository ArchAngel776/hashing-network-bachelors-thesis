from typing import TypedDict
from torch import Tensor, device as dev


class HSDHMetrics(TypedDict):
    map: float
    precision_at_m: dict[int, float]


def get_metrics(
    hash_length: int,
    database_hashes: Tensor,
    database_labels: Tensor,
    query_hashes: Tensor,
    query_labels: Tensor,
    m_values: tuple[int, ...],
    device: dev
) -> HSDHMetrics: ...
