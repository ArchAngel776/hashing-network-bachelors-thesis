from os import path, listdir
from torch.utils.data import Dataset
from app.exceptions.DatasetNotFoundException import DatasetNotFoundException


class KatherDataset(Dataset):
    def __init__(self, source_dir, transform = None, target_transform = None):
        self._transform         = transform
        self._target_transform  = target_transform
        
        self._data = []
        self._label = None

        if not path.isdir(source_dir):
            raise DatasetNotFoundException(source_dir)

        labels = [ label_dir for label_dir in listdir(source_dir) if path.isdir(path.join(source_dir, label_dir)) ]
        labels.sort()

        for index, label in enumerate(labels):
            images = [ image for image in listdir(path.join(source_dir, label))
                       if path.isfile(path.join(source_dir, label, image)) ]
            images.sort()

            for image in images:
                self._data.append((path.join(source_dir, label, image), index))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]

        if self._transform is not None:
            image_path = self._transform(image_path)

        if self._target_transform is not None:
            label = self._target_transform(label)

        return image_path, label

    def select_label(self, label):
        self._label = label

    def deselect_label(self):
        self._label = None

    @property
    def data(self):
        if self._label is None:
            return self._data

        return [ (image_path, label) for image_path, label in self._data if label == self._label ]

    @property
    def labels(self):
        labels = []

        for _, label in self._data:
            if label in labels:
                continue

            labels.append(label)

        return labels
