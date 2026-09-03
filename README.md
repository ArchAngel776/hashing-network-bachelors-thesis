# Hashing network - Bachelor's thesis
This project contains source software for the Bachelor's thesis purpose.

## Installation

To make a project working properly, correct dependencies must be installed (recommended way is using a `pip` in the custom **venv**):

```shell
python -m pip install -r requirements.txt
```

After installation of dependencies, project is ready to use.

## Usage

Project allows developer to train own DINO-HSDH model with either domain-pretrained ViT on the Kather dataset or general-pretrained on the IMAGENET_1K dataset.

### Domain training for DINO

To make a domain training for the DINO ViT, a specially provided script shall be run:

```shell
python dino-domain-train.py --epochs=200
```

Provided parameter `epochs` can be adjusted to the developer's needs.

> [!NOTE]
> This step is optional. If no domain training has been ever proceed, model will load default IMAGENET_1K weights.

### Main training for DINO-HSDH model

A main training can be proceed by running the `main.py` module with proper parameters:

```shell
python main.py --hash-length=64 --epochs=100
```

Provided parameters `hash-length` and `epochs` can be adjusted to the developer's needs. Training program is going to ask user, whether starting from previously stored checkpoint is desirable or not (empty input means the training should be fresh).

> [!IMPORTANT]
> Loaded checkpoint must pass the provided `hash-length` parameter.
> For example: if training was previously made for the **hash-length=32** provided argument cannot be different than it.

### Reading report analysis after the training

Program can read and display a few analytics data for provided, trained checkpoint:

- MAP (general)
- MAP@k
- Precision@m
- Number of unique hashes in database
- Number of collisions between classes
- Average tie within the model

Report for the checkpoint can be displayed by typing a command:

```shell
python report.py --hash-length=64
```

Provided parameter `hash-length` must pass the parameter provided for the measured checkpoint during its training.

After launching the script, user is asked to provide a path to the desired checkpoint.
