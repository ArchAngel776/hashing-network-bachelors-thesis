class DatasetNotFoundException(Exception):
    def __init__(self, path):
        super().__init__()
        self._path = path

    def __str__(self):
        return f"Dataset not found under the path: {self._path}"
