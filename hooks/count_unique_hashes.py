def count_unique_hashes(hashes):
    return hashes.unique(dim=0, sorted=False).size(dim=0), hashes.size(dim=0)
