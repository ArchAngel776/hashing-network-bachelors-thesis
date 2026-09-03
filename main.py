from os import path
from sys import argv
import torch
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from torch.accelerator import current_accelerator
from torchvision.transforms.v2 import Compose, ToImage, Resize, InterpolationMode, CenterCrop, ToDtype,Normalize, Lambda
from app.datasets.KatherDataset import KatherDataset
from app.datasets.KatherPairsDataset import KatherPairsDataset
from app.datasets.KatherRetrievalDataset import KatherRetrievalDataset
from app.modules.HSDH import HSDH
from app.modules.HSDHLoss import HSDHLoss
from app.components.ArgumentsParser import ArgumentsParser
from hooks.extract_hashes import extract_hashes
from hooks.get_metrics import get_metrics


batch_size = 64

map_k_values        = (1, 5, 10, 20, 50, 100)
precision_m_values  = (1, 5, 10, 20, 50, 100)

start_epoch     = 0
current_epoch   = 0


transform = Compose([
    ToImage(),
    Resize((256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
    CenterCrop((224, 224)),
    ToDtype(dtype=torch.float32, scale=True),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

target_transform = Lambda(lambda label: torch.tensor(label, dtype=torch.float32))


dataset = KatherDataset(
    source_dir = "data/Kather_texture_2016_image_tiles_5000"
)


dataset_train = KatherPairsDataset(
    dataset,
    train=True,
    transform=transform,
    target_transform=target_transform
)


dataset_test = KatherPairsDataset(
    dataset,
    train=False,
    transform=transform,
    target_transform=target_transform
)


dataset_train.resample()
dataset_test.resample()


database_dataset = KatherRetrievalDataset(
    sources=dataset_train.sources,
    transform=transform
)

query_dataset = KatherRetrievalDataset(
    sources=dataset_test.sources,
    transform=transform
)


train_dataloader    = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_dataloader     = DataLoader(dataset_test,  batch_size=batch_size, shuffle=False)

database_loader     = DataLoader(database_dataset,  batch_size=batch_size, shuffle=False)
query_loader        = DataLoader(query_dataset,     batch_size=batch_size, shuffle=False)


def train_loop(model, loss_function, optimizer, device):
    loss_result = 0.0
    succeeded_predictions = 0
    proceeded_predictions = 0

    model.train()

    for index, (image_i, image_j, target) in enumerate(train_dataloader):
        image_i, image_j, target = image_i.to(device), image_j.to(device), target.to(device).unsqueeze(dim=1)

        optimizer.zero_grad(set_to_none=True)

        prediction = model(image_i, image_j)
        loss = loss_function(prediction, target)

        loss.backward()
        optimizer.step()

        loss_result += loss.detach().item()

        succeeded_predictions += torch.sum((prediction.detach() >= 0.5) == (target >= 0.5)).item()
        proceeded_predictions += target.numel()

        if index % 10 == 0:
            print(f"Current train loss: {loss.item():>7f}\t[{proceeded_predictions:>5d}/{len(dataset_train):>5d}]")

    print("")

    average_loss = loss_result / proceeded_predictions
    accuracy = succeeded_predictions / proceeded_predictions

    return average_loss, accuracy


@torch.no_grad()
def test_loop(model, loss_function, device):
    loss_result = 0.0
    succeeded_predictions = 0
    proceeded_predictions = 0

    model.eval()

    for image_i, image_j, target in test_dataloader:
        image_i, image_j, target = image_i.to(device), image_j.to(device), target.to(device).unsqueeze(dim=1)

        prediction = model(image_i, image_j)
        loss = loss_function(prediction, target)

        loss_result += loss.item()

        succeeded_predictions += torch.sum((prediction >= 0.5) == (target >= 0.5)).item()
        proceeded_predictions += target.numel()

    average_loss = loss_result / proceeded_predictions
    accuracy = succeeded_predictions / proceeded_predictions

    return average_loss, accuracy


def main_loop(epochs, model, hash_length, loss_function, optimizer, device):
    global start_epoch, current_epoch

    for epoch in range(epochs):
        print("")
        print(f"Epoch {epoch + 1}/{epochs}")
        print("------------------------------------")

        if epoch > 0:
            dataset_train.resample(epoch)

        average_loss_train, accuracy_train = train_loop(
            model           = model,
            loss_function   = loss_function,
            optimizer       = optimizer,
            device          = device
        )

        print(f"Train loss: {average_loss_train}")
        print(f"Train accuracy: {accuracy_train}")
        print("")

        average_loss_test, accuracy_test = test_loop(
            model           = model,
            loss_function   = loss_function,
            device          = device
        )

        print(f"Test loss: {average_loss_test}")
        print(f"Test accuracy: {accuracy_test}")
        print("")

        database_hashes, database_labels = extract_hashes(hsdh, database_loader, device)
        query_hashes, query_labels = extract_hashes(hsdh, query_loader, device)

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

        current_epoch = epoch + 1
        print("")


if __name__ == "__main__":
    arguments_parser = ArgumentsParser(argv[1:])
    arguments_parser.parse()

    hash_length = arguments_parser.get_option("hash-length",    int)
    epochs      = arguments_parser.get_option("epochs",         int)

    try:
        assert isinstance(hash_length,  int)
        assert isinstance(epochs,       int)
    except AssertionError:
        print("Incorrect arguments specified. You ned to specify: --hash-length=<int> and --epochs=<int>")
        exit(1)

    hsdh = HSDH(hash_length=hash_length)
    loss_function = HSDHLoss(beta=.2)

    accelerator = current_accelerator(check_available=True)
    device = accelerator if accelerator is not None else torch.device("cpu")

    hsdh.to(device)
    hsdh.load_dino_domain_pretrained_weights()

    optimizer = Adam([
        {
            "name": "hash_projection",
            "params": hsdh._hash_generator._hash_projection.parameters(),
            "lr": 1e-4
        },
        {
            "name": "fully_connected_layer_3",
            "params": hsdh._fully_connected_layer.parameters(),
            "lr": 1e-3
        }
    ])

    model_load_path = input("Give a path for loading model (leave blank, if you do not want to load it): ")

    if len(model_load_path) > 0 and path.exists(model_load_path):
        params = torch.load(model_load_path, map_location=device)

        start_epoch = params["epoch"]
        hsdh.load_state_dict(params["model"])
        optimizer.load_state_dict(params["optimizer"])
    else:
        if len(model_load_path) > 0:
            print(f"Model not found under the path: {model_load_path}. Weights will not be loaded.")

        hsdh.fit_scaler(database_loader, device)

    print("Start learning process...")
    print("")

    current_epoch = start_epoch

    print(f"Used device: {device.type}")
    print("")

    try:
        main_loop(
            epochs          = epochs,
            hash_length     = hash_length,
            model           = hsdh,
            loss_function   = loss_function,
            optimizer       = optimizer,
            device          = device
        )
    except KeyboardInterrupt:
        print("")
        print("Training loop interrupted by user.")
        print("")

    model_save_path = input("Give a path for saving model (leave blank, if you do not want to save it): ")

    if len(model_save_path) > 0:
        torch.save({
            "epoch":        current_epoch,
            "model":        hsdh.state_dict(),
            "optimizer":    optimizer.state_dict()
        }, model_save_path)

        print("Model saved successfully!")

    print("Done!")
