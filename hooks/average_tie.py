import torch
from torch import no_grad


@no_grad()
def average_tie(database_hashes, query_hashes, device):
    database_hashes = database_hashes.to(device=device, dtype=torch.float32)
    query_hashes    = query_hashes.to(device=device,    dtype=torch.float32)

    similarities = query_hashes @ database_hashes.transpose(0, 1)

    return (similarities == similarities.max(dim=1, keepdim=True).values).sum(dim=1).float().mean().item()
