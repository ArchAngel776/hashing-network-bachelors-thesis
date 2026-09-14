from os import path
from sys import argv
import torch
from torch.accelerator import current_accelerator
from app.modules.HSDH import HSDH
from app.components.ArgumentsParser import ArgumentsParser
from app.exceptions.CheckpointNotFoundException import CheckpointNotFoundException
from hooks.extract_hashes import extract_hashes
from hooks.get_metrics import get_metrics
from hooks.count_unique_hashes import count_unique_hashes
from hooks.class_collisions import class_collisions
from hooks.average_tie import average_tie
from main import database_loader, query_loader, map_k_values, precision_m_values


if __name__ == "__main__":
    arguments_parser = ArgumentsParser(argv[1:])
    arguments_parser.parse()

    hash_length     = arguments_parser.get_option("hash-length",    int)
    pca_components  = arguments_parser.get_option("pca",            int)
    nca_components  = arguments_parser.get_option("nca",            int)

    try:
        assert isinstance(hash_length,      int)

        assert not (isinstance(pca_components, int) and isinstance(nca_components, int))
    except AssertionError:
        print("Incorrect arguments specified. You ned to specify: --hash-length=<int> [--pca=<int>|--nca=<int>]")
        exit(1)

    hsdh = HSDH(hash_length=hash_length, pca_components=pca_components, nca_components=nca_components)

    accelerator = current_accelerator(check_available=True)
    device = accelerator if accelerator is not None else torch.device("cpu")

    hsdh.to(device)

    model_load_path = input("Give a path for loading model: ")

    if len(model_load_path) > 0 and path.exists(model_load_path):
        params = torch.load(model_load_path, map_location=device)
        hsdh.load_state_dict(params["model"])
    else:
        raise CheckpointNotFoundException(model_load_path)

    print("")
    print(f"Display metrics report for the checkpoint: {model_load_path}")
    print("")

    database_hashes, database_labels    = extract_hashes(hsdh, database_loader, device)
    query_hashes, query_labels          = extract_hashes(hsdh, query_loader, device)

    metrics = get_metrics(
        hash_length     = hash_length,
        database_hashes = database_hashes,
        database_labels = database_labels,
        query_hashes    = query_hashes,
        query_labels    = query_labels,
        k_values        = map_k_values,
        m_values        = precision_m_values,
        device          = device
    )

    print(f"MAP: {metrics["map"]:.4f}")
    print("")

    for k, map_at_k in metrics["map_at_k"].items():
        print(f"MAP@{k}: {map_at_k:.4f}")

    print("")

    for m, precision in metrics["precision_at_m"].items():
        print(f"Precision@{m}: {precision:.4f}")

    print("")

    unique_hashes, all_hashes = count_unique_hashes(hashes=database_hashes)
    collided_codes, collided_images = class_collisions(hashes=database_hashes, labels=database_labels)
    avg_tie = average_tie(database_hashes=database_hashes, query_hashes=query_hashes, device=device)

    print(f"Unique hashes in the database: {unique_hashes}/{all_hashes}")
    print(f"(Codes/images) collision between classes: ({collided_codes}/{collided_images})")
    print(f"Average tie: {avg_tie:.4f}")
    print("")

    print("Done!")
