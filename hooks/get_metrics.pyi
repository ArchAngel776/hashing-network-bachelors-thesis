from typing import TypedDict
from torch import Tensor, device as dev


class HSDHMetrics(TypedDict):
    map: float
    map_at_k: dict[int, float]
    precision_at_m: dict[int, float]


def get_metrics(
    hash_length: int,
    database_hashes: Tensor,
    database_labels: Tensor,
    query_hashes: Tensor,
    query_labels: Tensor,
    k_values: tuple[int, ...],
    m_values: tuple[int, ...],
    device: dev
) -> HSDHMetrics: ...
