from os import path
import torch
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from torch.accelerator import current_accelerator
from torchvision.transforms.v2 import Compose, ToImage, Resize, ToDtype, Normalize, Lambda
from app.datasets.KatherDataset import KatherDataset
from app.datasets.KatherPairsDataset import KatherPairsDataset
from app.modules.HSDH import HSDH
from app.modules.HSDHLoss import HSDHLoss


batch_size = 64
learning_rate = 1e-3
epochs = 40


dataset = KatherDataset(
    source_dir = "data/Kather_texture_2016_image_tiles_5000"
)


dataset_train = KatherPairsDataset(
    dataset,
    train=True,
    transform=Compose([
        ToImage(),
        Resize(
            size=(224, 224),
            antialias=True
        ),
        ToDtype(
            dtype=torch.float32,
            scale=True
        ),
        Normalize(
            #mean=[0.5, 0.5, 0.5],
            #std=[0.5, 0.5, 0.5]
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    target_transform=Lambda(
        lambda label: torch.tensor(label, dtype=torch.float32)
    )
)


dataset_test = KatherPairsDataset(
    dataset,
    train=False,
    transform=Compose([
        ToImage(),
        Resize(
            size=(224, 224),
            antialias=True
        ),
        ToDtype(
            dtype=torch.float32,
            scale=True
        ),
        Normalize(
            #mean=[0.5, 0.5, 0.5],
            #std=[0.5, 0.5, 0.5]
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    target_transform=Lambda(
        lambda label: torch.tensor(label, dtype=torch.float32)
    )
)


dataset_train.resample(0)
dataset_test.resample(0)


train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


hsdh = HSDH(hash_length=128, alpha=1000)
loss_function = HSDHLoss(beta=.2)


accelerator = current_accelerator(check_available=True)
device = accelerator if accelerator is not None else torch.device("cpu")

hsdh.to(device)


if path.exists("data/Kather_texture_2016_image_tiles_5000/model.pth"):
    hsdh.load_state_dict(
        torch.load("data/Kather_texture_2016_image_tiles_5000/model.pth", map_location=device, weights_only=True)
    )


optimizer = Adam(
    hsdh.parameters(),
    lr=learning_rate
)


def train_loop():
    loss_result = 0.0
    succeeded_predictions = 0
    proceeded_predictions = 0

    hsdh.train()

    for index, (image_i, image_j, target) in enumerate(train_dataloader):
        image_i, image_j, target = image_i.to(device), image_j.to(device), target.to(device).unsqueeze(dim=1)

        optimizer.zero_grad(set_to_none=True)

        prediction = hsdh(image_i, image_j, target)
        loss = loss_function(prediction, target)

        loss.backward()
        optimizer.step()

        loss_result += loss.detach().item()

        succeeded_predictions += torch.sum((prediction.detach() >= 0.5) == (target >= 0.5)).item()
        proceeded_predictions += target.numel()

        if index % 10 == 0:
            print(f"Current train loss: {loss.item():>7f}\t[{proceeded_predictions:>5d}/{len(dataset_train):>5d}]")

    average_loss = loss_result / proceeded_predictions
    accuracy = succeeded_predictions / proceeded_predictions

    return average_loss, accuracy


@torch.no_grad()
def test_loop():
    loss_result = 0.0
    succeeded_predictions = 0
    proceeded_predictions = 0

    hsdh.eval()

    for image_i, image_j, target in test_dataloader:
        image_i, image_j, target = image_i.to(device), image_j.to(device), target.to(device).unsqueeze(dim=1)

        prediction = hsdh(image_i, image_j, target)
        loss = loss_function(prediction, target)

        loss_result += loss.item()

        succeeded_predictions += torch.sum((prediction >= 0.5) == (target >= 0.5)).item()
        proceeded_predictions += target.numel()

    average_loss = loss_result / proceeded_predictions
    accuracy = succeeded_predictions / proceeded_predictions

    return average_loss, accuracy


if __name__ == "__main__":
    print("Start learning process...")
    print("")

    print(f"Used device: {device.type}")
    print("")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("------------------------------------")

        if epoch > 0:
            dataset_train.resample(epoch)

        average_loss_train, accuracy_train = train_loop()

        print(f"Train loss: {average_loss_train}")
        print(f"Train accuracy: {accuracy_train}")
        print("")

        average_loss_test, accuracy_test = test_loop()

        print(f"Test loss: {average_loss_test}")
        print(f"Test accuracy: {accuracy_test}")
        print("")

    torch.save(hsdh.state_dict(), "data/Kather_texture_2016_image_tiles_5000/model.pth")
    print("Done!")
