import torch
from torch import Generator, randperm, argsort, arange, any


def get_metrics(
    hash_length,
    database_hashes,
    database_labels,
    query_hashes,
    query_labels,
    m_values,
    device
):
    permutation = randperm(database_hashes.shape[0], generator=Generator().manual_seed(42))

    database_hashes = database_hashes[permutation]
    database_labels = database_labels[permutation]

    database_hashes = database_hashes.to(device=device, dtype=torch.float32)
    query_hashes = query_hashes.to(device=device, dtype=torch.float32)

    database_labels = database_labels.to(device)
    query_labels = query_labels.to(device)

    similarities = query_hashes @ database_hashes.transpose(0, 1)
    hamming_distances = (hash_length - similarities) / 2.0

    ranked_indices = argsort(hamming_distances, dim=1, stable=True)

    relevant = (database_labels[ranked_indices] == query_labels.unsqueeze(1)).to(dtype=torch.float32)

    database_size = database_labels.shape[0]

    ranks = arange(
        start=1,
        end=database_size + 1,
        device=device,
        dtype=torch.float32
    ).unsqueeze(0)

    cumulative_relevant = relevant.cumsum(dim=1)
    relevant_count = relevant.sum(dim=1)

    if any(relevant_count == 0):
        raise RuntimeError("At least one query class does not occur in the retrieval database.")

    average_precision = (cumulative_relevant / ranks * relevant).sum(dim=1) / relevant_count
    mean_average_precision = average_precision.mean().item()

    precision_at_m = {}

    for m in m_values:
        if m <= 0:
            raise ValueError("The value of m must be positive.")

        query_precision = relevant[:, :min(m, database_size)].mean(dim=1)
        precision_at_m[m] = query_precision.mean().item()

    return {
        "map":              mean_average_precision,
        "precision_at_m":   precision_at_m
    }
