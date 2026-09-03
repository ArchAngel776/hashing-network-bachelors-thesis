from torch import Tensor, device as dev


def average_tie(database_hashes: Tensor, query_hashes: Tensor, device: dev) -> float: ...
