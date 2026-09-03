from os import environ
from sys import argv
from pathlib import Path
from subprocess import run
import torch
from torch import hub, cuda
from app.components.ArgumentsParser import ArgumentsParser
from app.datasets.KatherDataset import KatherDataset
from app.datasets.KatherPairsDataset import KatherPairsDataset
from app.modules.DINO import DINO
from app.exceptions.CUDAUnavailableException import CUDAUnavailableException


if not cuda.is_available():
    print("DINO domain training requires available CUDA.")
    raise CUDAUnavailableException


arguments_parser = ArgumentsParser(argv[1:])
arguments_parser.parse()

epochs = arguments_parser.get_option("epochs", int)

try:
    assert isinstance(epochs, int)
except AssertionError:
    print("Incorrect arguments specified. You ned to specify: --epochs=<int>")
    exit(1)


dino_path =  Path(f"{hub.get_dir()}/facebookresearch_dino_main")


dataset = KatherDataset(source_dir = "data/Kather_texture_2016_image_tiles_5000")
dataset_train = KatherPairsDataset(dataset, train=True)


tmp = Path("tmp/kather-dino-train")


if not dino_path.exists():
    print("No DINO repo found. Downloading now...")

    hub.load(
        "facebookresearch/dino:main", "dino_vits16",
        pretrained=True,
        trust_repo=True,
        verbose=False
    )

    print("DINO repo successfully downloaded!")
    print("")


if not DINO.CHECKPOINTS_PATH.joinpath("checkpoint.pth").exists():
    print("No pretrained checkpoint found. Fetching now...")

    pretrained_model = hub.load(
        "facebookresearch/dino:main", "dino_vits16",
        pretrained=True,
        trust_repo=True,
        verbose=False
    )

    state = pretrained_model.state_dict()

    print("Pretrained parameters successfully fetched!")
    print("Saving parameters now...")

    pretrained_params = {
        "student": {
            f"module.backbone.{key}": value
            for key, value in state.items()
        },
        "teacher": {
            f"backbone.{key}": value
            for key, value in state.items()
        },
        "epoch": 0,
    }

    torch.save(pretrained_params, DINO.CHECKPOINTS_PATH.joinpath("checkpoint.pth"))

    print("Parameters successfully saved!")
    print("")


try:
    for image_path, label in dataset_train.sources:
        class_dir = tmp.joinpath(f"{label:02d}")
        class_dir.mkdir(parents=True, exist_ok=True)

        image_path = Path(image_path)
        class_dir.joinpath(image_path.name).symlink_to(image_path.resolve())

    run(
        args=[
            "python", "-m", "torch.distributed.launch", "--nproc-per-node=1", "--use-env",
            dino_path.joinpath("main_dino.py").resolve(),
            "--arch", "vit_small",
            "--patch_size", "16",
            "--data_path", tmp.resolve(),
            "--output_dir", DINO.CHECKPOINTS_PATH.resolve(),
            "--epochs", f"{epochs}",
            "--batch_size_per_gpu", "64",
            "--optimizer", "adamw",
            "--lr", "0.0005",
            "--min_lr", "0.000001",
            "--use_fp16", "true",
            "--num_workers", "8",
            "--saveckp_freq", "20"
        ],
        env={
            **environ,
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"
        },
        check=True
    )
except KeyboardInterrupt:
    print("Training interrupted by user.")
    print("")
    exit()
finally:
    for label_dir in tmp.iterdir():
        for link in label_dir.iterdir():
            link.unlink()

        label_dir.rmdir()

    tmp.rmdir()


checkpoint = torch.load(
    DINO.CHECKPOINTS_PATH.joinpath("checkpoint.pth"),
    map_location="cpu",
    weights_only=False
)

teacher = {
    key.removeprefix("module."): value
    for key, value in checkpoint["teacher"].items()
}

backbone = {
    key.removeprefix("backbone."): value
    for key, value in teacher.items()
    if key.startswith("backbone.")
}

torch.save(backbone, DINO.CHECKPOINTS_PATH.joinpath("dino-domain-pretrained.pth"))
