from PIL import Image
from torch.utils.data import Dataset


class KatherRetrievalDataset(Dataset):
    def __init__(
        self,
        sources,
        transform = None,
        target_transform= None
    ):
        self._sources           = sources
        self._transform         = transform
        self._target_transform  = target_transform

    def __len__(self):
        return len(self._sources)

    def __getitem__(self, index):
        image_path, label = self._sources[index]

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        if self._transform is not None:
            image = self._transform(image)

        if self._target_transform is not None:
            label = self._target_transform(label)

        return image, label
