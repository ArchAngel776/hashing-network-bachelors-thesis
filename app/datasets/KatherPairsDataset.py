from random import Random
from PIL import Image
from torch import Generator
from torch.utils.data import Dataset, random_split


class KatherPairsDataset(Dataset):
    RANDOM_SEED = 42
    TRAINING_RATE = 0.7

    def __init__(
        self,
        dataset,
        train,
        transform = None,
        target_transform = None
    ):
        self._transform         = transform
        self._target_transform  = target_transform

        self._data      = []
        self._sources   = []
        self._groups    = {}
        self._labels    = dataset.labels

        generator = Generator().manual_seed(KatherPairsDataset.RANDOM_SEED)

        for label in self._labels:
            dataset.select_label(label)

            train_data_length    = int(len(dataset) * KatherPairsDataset.TRAINING_RATE)
            test_data_length     = len(dataset) - train_data_length

            train_data, test_data = random_split(
                dataset,
                lengths=[train_data_length, test_data_length],
                generator=generator
            )

            for item in (train_data if train else test_data):
                self._sources.append(item)

        dataset.deselect_label()

        for image_path, label in self._sources:
            if label not in self._groups:
                self._groups[label] = []

            self._groups[label].append(image_path)

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        image_i_path, image_j_path, positive_pair = self._data[index]

        with Image.open(image_i_path) as image_i_file:
            image_i = image_i_file.convert("RGB")

        with Image.open(image_j_path) as image_j_file:
            image_j = image_j_file.convert("RGB")

        if self._transform is not None:
            image_i = self._transform(image_i)

        if self._transform is not None:
            image_j = self._transform(image_j)

        if self._target_transform is not None:
            positive_pair = self._target_transform(positive_pair)

        return image_i, image_j, positive_pair

    def resample(self, epoch):
        self._data = []

        rng = Random(KatherPairsDataset.RANDOM_SEED + epoch)

        for image_path, label in self._sources:
            positives = self._groups[label]
            negatives = self._groups[rng.choice(self.negative_labels(label))]

            image_positive_path = rng.choice(positives)
            image_negative_path = rng.choice(negatives)

            while image_positive_path == image_path:
                image_positive_path = rng.choice(positives)

            self._data.append((image_path, image_positive_path, True))
            self._data.append((image_path, image_negative_path, False))

    def negative_labels(self, label):
        return [ negative_label for negative_label in self._labels if negative_label != label ]
