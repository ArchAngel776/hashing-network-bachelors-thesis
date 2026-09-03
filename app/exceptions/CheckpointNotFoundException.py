class CheckpointNotFoundException(Exception):
    def __init__(self, path):
        super().__init__()
        self._path = path

    def __str__(self):
        return f"Checkpoint not found under a path: {self._path}"
